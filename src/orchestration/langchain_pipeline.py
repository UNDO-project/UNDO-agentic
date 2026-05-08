from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.agents.surveillance_data_collector import (
    ScrapeError,
    SurveillanceDataCollector,
)
from src.agents.langchain_analyzer import SurveillanceAnalyzerAgent
from src.agents.route_finder_agent import RouteFinderAgent
from src.config.logger import logger
from src.llm.surveillance_llm import check_ollama_reachable
from src.config.pipeline_config import PipelineConfig, AnalysisScenario
from src.config.settings import DatabaseSettings, LangChainSettings, RouteSettings
from src.config.models.route_models import RouteRequest
from src.memory.store import MemoryStore


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SCRAPING = "scraping"
    ANALYZING = "analyzing"
    ROUTING = "routing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Completed with some errors
    CANCELLED = "cancelled"  # Cancelled by user


class SurveillancePipeline:
    """
    Multi-agent pipeline for end-to-end surveillance data analysis.

    Orchestrates:
    1. SurveillanceScraperAgent - Downloads data from OpenStreetMap
    2. SurveillanceAnalyzerAgent - Enriches and visualizes data

    Features:
    - Configurable analysis scenarios
    - Progress tracking and status reporting
    - Error recovery between agents
    - Shared memory for caching
    - Optional LangSmith observability
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        langchain_settings: Optional[LangChainSettings] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
        on_scrape_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_analyzer_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize the surveillance pipeline.

        :param config: Pipeline configuration (uses default if None)
        :param langchain_settings: LangChain settings for agents
        :param cancellation_check: Optional callback that returns True if pipeline should cancel
        :param on_scrape_complete: Optional callback fired once after the
            scrape stage returns and before the analyzer stage starts.
            Receives a payload dict with ``scrape_result``, ``data_path``,
            ``elements_count``, and ``will_skip_analyzer`` so callers (the
            FastAPI route) can surface the count to the polled task
            response and any future WebSocket consumer.
        :param on_analyzer_progress: Optional callback fired once per
            analyzer batch with ``(enriched_count, total)``. Wired through
            to ``AnalysisChain.on_progress`` in ``_run_analyzer``. Lets
            the API route surface live "Enriched N/total" progress on the
            polled task response.
        """
        self.config = config or PipelineConfig()
        self.settings = langchain_settings or LangChainSettings()
        self.cancellation_check = cancellation_check
        self.on_scrape_complete = on_scrape_complete
        self.on_analyzer_progress = on_analyzer_progress

        # Create shared memory for both agents
        db_settings = DatabaseSettings()
        self.memory = MemoryStore(settings=db_settings)

        # Initialize agents
        self.scraper = SurveillanceDataCollector(
            name="ScraperAgent",
            memory=self.memory,
            settings=self.settings,
        )

        self.analyzer = SurveillanceAnalyzerAgent(
            name="AnalyzerAgent",
            memory=self.memory,
            settings=self.settings,
        )

        # Initialize routing agent if routing is enabled
        self.router = None
        if self.config.routing_enabled:
            route_settings = RouteSettings()
            self.router = RouteFinderAgent(
                name="RouteFinderAgent",
                memory=self.memory,
                settings=route_settings,
            )

        # Pipeline state
        self.status = PipelineStatus.PENDING
        self.current_step = None
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.errors = []

        logger.info(
            f"Initialized SurveillancePipeline with {self.config.scenario.value} scenario"
        )

    def run(self, city: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the complete pipeline for a city.

        :param city: City name to analyze
        :param kwargs: Additional arguments (country, output_dir override)
        :return: Dictionary with complete pipeline results
        """
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()
        self.current_step = "initialization"

        logger.info(f"Starting pipeline for {city}")

        # Extract parameters
        country = kwargs.get("country", self.config.country_code)
        output_dir = kwargs.get("output_dir", self.config.output_dir)

        results = {
            "city": city,
            "country": country,
            "scenario": self.config.scenario.value,
            "start_time": self.start_time.isoformat(),
        }

        try:
            # Preflight: fail fast if Ollama is down so we don't sink
            # minutes into a scrape whose enrichment would 100% fail.
            # Skipped when the analyzer is disabled — Ollama isn't needed.
            if self.config.analyze_enabled:
                try:
                    check_ollama_reachable(self.settings)
                except RuntimeError as e:
                    logger.error(str(e))
                    results["error"] = str(e)
                    return self._finalize_results(results, PipelineStatus.FAILED)

            # Step 1: Scraping
            if self.config.scrape_enabled:
                # Check for cancellation before scraping
                if self._check_cancellation():
                    logger.info(f"Pipeline cancelled before scraping for {city}")
                    return self._finalize_results(results, PipelineStatus.CANCELLED)

                scrape_result = self._run_scraper(city, country, output_dir)
                results["scrape"] = scrape_result

                # Handle scraping result (cancellation check + error handling)
                early_exit = self._handle_stage_result(
                    "scraping", scrape_result, results, city
                )
                if early_exit:
                    return early_exit

                # Get scraped data path for analysis
                data_path = scrape_result.get("filepath") or scrape_result.get(
                    "cached_path"
                )
                if not data_path:
                    error_msg = "No data path found from scraper"
                    logger.error(error_msg)
                    results["error"] = error_msg
                    return self._finalize_results(results, PipelineStatus.FAILED)
            else:
                # User provides data path directly
                data_path = kwargs.get("data_path")
                if not data_path:
                    error_msg = "Scraping disabled but no data_path provided"
                    logger.error(error_msg)
                    results["error"] = error_msg
                    return self._finalize_results(results, PipelineStatus.FAILED)
                results["scrape"] = {"skipped": True, "reason": "scraping disabled"}

            # Step 2: Analysis
            enriched_geojson_path = None
            if self.config.analyze_enabled:
                # Check for cancellation before analysis
                if self._check_cancellation():
                    logger.info(f"Pipeline cancelled before analysis for {city}")
                    return self._finalize_results(results, PipelineStatus.CANCELLED)

                # Fire the on-scrape-complete hook now that we know the
                # element count and whether the analyzer will be skipped.
                # Errors in the callback are swallowed so a bad listener
                # never crashes the pipeline.
                self._notify_scrape_complete(data_path, results.get("scrape"))

                analyze_result = self._run_analyzer(
                    data_path, scrape_result=results.get("scrape")
                )
                results["analyze"] = analyze_result

                # Handle analysis result (cancellation check + error handling)
                early_exit = self._handle_stage_result(
                    "analysis", analyze_result, results, city, PipelineStatus.PARTIAL
                )
                if early_exit:
                    return early_exit

                # Check for visualization errors (partial success)
                if analyze_result.get("visualization_errors"):
                    self.errors.extend(analyze_result["visualization_errors"])
                    # Don't return yet - routing can still proceed

                # Get enriched geojson path for routing
                enriched_geojson_path = analyze_result.get("geojson_path")
            else:
                results["analyze"] = {"skipped": True, "reason": "analysis disabled"}

            # Step 3: Routing (if enabled)
            if self.config.routing_enabled:
                # Check for cancellation before routing
                if self._check_cancellation():
                    logger.info(f"Pipeline cancelled before routing for {city}")
                    return self._finalize_results(results, PipelineStatus.CANCELLED)

                if not enriched_geojson_path:
                    error_msg = "Routing enabled but no enriched GeoJSON path available"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    return self._finalize_results(results, PipelineStatus.PARTIAL)

                routing_result = self._run_router(
                    city, country, Path(enriched_geojson_path)
                )
                results["routing"] = routing_result

                if not routing_result.get("success"):
                    if self.config.stop_on_error:
                        return self._finalize_results(results, PipelineStatus.FAILED)
                    else:
                        self.errors.append(
                            f"Routing failed: {routing_result.get('error')}"
                        )
                        return self._finalize_results(results, PipelineStatus.PARTIAL)
            else:
                if self.config.routing_enabled:
                    results["routing"] = {"skipped": True, "reason": "routing disabled"}

            # Success!
            # Check if there were any errors accumulated
            if self.errors:
                return self._finalize_results(results, PipelineStatus.PARTIAL)
            return self._finalize_results(results, PipelineStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            results["error"] = str(e)
            return self._finalize_results(results, PipelineStatus.FAILED)

    def _run_scraper(
        self,
        city: str,
        country: Optional[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Execute the scraper agent.

        :param city: City name
        :param country: Optional country code
        :param output_dir: Output directory
        :return: Scraper results
        """
        # Check for cancellation at entry
        if self._check_cancellation():
            logger.info(f"Pipeline cancelled at scraper entry for {city}")
            return {"success": False, "error": "Pipeline cancelled", "cancelled": True}

        self.status = PipelineStatus.SCRAPING
        self.current_step = "scraping"

        logger.info(f"Scraping data for {city}")

        scrape_input: Dict[str, Any] = {
            "city": city,
            "overpass_dir": output_dir,
            "force_refresh": self.config.force_refresh,
        }
        if country:
            scrape_input["country"] = country

        try:
            result = self.scraper.scrape(scrape_input)

            if result.get("success"):
                logger.info(
                    f"Scraping completed: {result.get('elements_count', 0)} elements"
                )
            else:
                logger.error(f"Scraping failed: {result.get('error', 'Unknown error')}")

            return result

        except ScrapeError as e:
            # Permanent Overpass / build / save failure. The exception already
            # carries a precise message; surface it verbatim so the API layer
            # can show the underlying Overpass response.
            logger.error(f"Scraper {e.stage} failed for {city}: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": e.stage,
                "city": city,
                "country": country,
            }

        except Exception as e:
            logger.error(f"Unexpected scraper exception for {city}: {e}")
            return {
                "success": False,
                "error": str(e),
                "city": city,
                "country": country,
            }

    def _run_analyzer(
        self,
        data_path: str,
        scrape_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the analyzer agent, or short-circuit when scrape data is
        unchanged from the prior run.

        Skip path: when ``scrape_result["changed"] is False`` and the prior
        enriched outputs are still on disk next to ``data_path``, return a
        synthetic success result pointing at those files. The expensive LLM
        enrichment is the dominant cost here; reusing prior outputs when the
        underlying OSM data is identical is the whole point of probe-and-
        compare.

        :param data_path: Path to scraped data (may be cache-served).
        :param scrape_result: The orchestrator's ``results["scrape"]`` dict;
            consulted for ``changed`` and metadata propagation.
        :return: Analyzer results.
        """
        # Check for cancellation at entry
        if self._check_cancellation():
            logger.info(f"Pipeline cancelled at analyzer entry for {data_path}")
            return {"success": False, "error": "Pipeline cancelled", "cancelled": True}

        # Probe-and-compare skip: scrape said nothing changed and prior
        # enriched outputs are still on disk → reuse them.
        skip = self._maybe_skip_analyzer(data_path, scrape_result)
        if skip is not None:
            self.status = PipelineStatus.ANALYZING
            self.current_step = "analyzing"
            return skip

        self.status = PipelineStatus.ANALYZING
        self.current_step = "analyzing"

        # Wire the per-batch progress hook through to the chain. Set on
        # every run (rather than once at construction) so re-using a
        # pipeline instance with a different listener works, and a
        # ``None`` listener cleanly disables the hook for that run.
        self.analyzer.chain.on_progress = self.on_analyzer_progress

        analyze_input = {
            "path": data_path,
            **self.config.to_analyzer_options(),
        }

        try:
            result = self.analyzer.analyze(analyze_input)

            if result.get("success"):
                logger.info(
                    f"Analysis completed: {result.get('element_count', 0)} elements enriched"
                )
            else:
                logger.error(f"Analysis failed: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Analyzer exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "path": data_path,
            }

    @staticmethod
    def will_skip_analyzer(
        data_path: str,
        scrape_result: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Predicate: would ``_maybe_skip_analyzer`` short-circuit for this
        ``(data_path, scrape_result)`` pair?

        Used by the on-scrape-complete hook to surface the planned
        skip/run decision to the API layer without computing the full
        synthetic result dict twice.

        :return: True iff scrape reports unchanged AND the prior enriched
            geojson is on disk.
        """
        if not scrape_result or scrape_result.get("changed") is not False:
            return False
        data = Path(data_path)
        enriched_geojson = data.with_name(data.stem + "_enriched.geojson")
        return enriched_geojson.exists()

    @staticmethod
    def _maybe_skip_analyzer(
        data_path: str,
        scrape_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        If scrape reports unchanged data and prior enriched outputs exist,
        synthesize an analyzer-success dict that points at them.

        Returns ``None`` if the analyzer should run normally — either because
        ``changed`` is True/missing or the prior enriched files are gone.
        """
        if not SurveillancePipeline.will_skip_analyzer(data_path, scrape_result):
            data = Path(data_path)
            enriched_geojson = data.with_name(data.stem + "_enriched.geojson")
            # Only log the "missing prior outputs" case — the changed=True
            # case is the normal run-the-analyzer path and doesn't need a
            # log line of its own.
            if (
                scrape_result
                and scrape_result.get("changed") is False
                and not enriched_geojson.exists()
            ):
                logger.info(
                    f"Cannot skip analyzer: prior enriched geojson missing at "
                    f"{enriched_geojson}; will re-run."
                )
            return None

        data = Path(data_path)
        enriched_json = data.with_name(data.stem + "_enriched.json")
        enriched_geojson = data.with_name(data.stem + "_enriched.geojson")

        result: Dict[str, Any] = {
            "success": True,
            "path": str(data),
            "element_count": scrape_result.get("elements_count", 0),
            "cache_hit": True,
            "skipped": True,
            "skipped_reason": "scrape_unchanged",
            "geojson_path": str(enriched_geojson),
        }
        if enriched_json.exists():
            result["enriched_path"] = str(enriched_json)

        # Surface the other prior visualizations if they're still on disk so
        # the API/UI can link to them. Patterns mirror what the analyzer
        # writes today.
        city_dir = data.parent
        stem = data.stem
        for key, name in (
            ("heatmap_path", f"{stem}_heatmap.html"),
            ("hotspots_path", f"hotspots_{stem}.geojson"),
            ("hotspots_chart", f"hotspot_plot_{stem}.png"),
            ("pie_chart_path", f"stats_chart_{stem}.png"),
        ):
            candidate = city_dir / name
            if candidate.exists():
                result[key] = str(candidate)

        return result

    def _run_router(
        self,
        city: str,
        country: Optional[str],
        enriched_geojson_path: Path,
    ) -> Dict[str, Any]:
        """
        Execute the routing agent.

        :param city: City name
        :param country: Optional country code
        :param enriched_geojson_path: Path to enriched camera data
        :return: Routing results
        """
        # Check for cancellation at entry
        if self._check_cancellation():
            logger.info(f"Pipeline cancelled at router entry for {city}")
            return {"success": False, "error": "Pipeline cancelled", "cancelled": True}

        self.status = PipelineStatus.ROUTING
        self.current_step = "routing"

        logger.info(
            f"Computing low-surveillance route for {city} from "
            f"({self.config.start_lat}, {self.config.start_lon}) to "
            f"({self.config.end_lat}, {self.config.end_lon})"
        )

        route_request = RouteRequest(
            city=city,
            country=country or "DE",
            start_lat=self.config.start_lat,
            start_lon=self.config.start_lon,
            end_lat=self.config.end_lat,
            end_lon=self.config.end_lon,
            data_path=enriched_geojson_path,
            camera_filter=self.config.camera_filter,
        )

        try:
            route_result = self.router.achieve_goal(route_request)

            result = {
                "success": True,
                "route_id": route_result.route_id,
                "city": route_result.city,
                "from_cache": route_result.from_cache,
                "route_geojson_path": str(route_result.route_geojson_path),
                "route_map_path": str(route_result.route_map_path),
                "length_m": route_result.metrics.length_m,
                "exposure_score": route_result.metrics.exposure_score,
                "camera_count": route_result.metrics.camera_count_near_route,
                "camera_count_near_route": route_result.metrics.camera_count_near_route,
                "camera_count_total": route_result.metrics.camera_count_total,
                "baseline_length_m": route_result.metrics.baseline_length_m,
                "baseline_exposure_score": route_result.metrics.baseline_exposure_score,
            }

            logger.info(
                f"Routing completed: {route_result.metrics.length_m:.1f}m route with "
                f"exposure score {route_result.metrics.exposure_score:.2f} cameras/km"
            )

            return result

        except Exception as e:
            logger.error(f"Routing exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "city": city,
            }

    def _check_cancellation(self) -> bool:
        """
        Check if pipeline should be cancelled.

        :return: True if pipeline should cancel, False otherwise
        """
        if self.cancellation_check and self.cancellation_check():
            return True
        return False

    def _notify_scrape_complete(
        self,
        data_path: str,
        scrape_result: Optional[Dict[str, Any]],
    ) -> None:
        """
        Fire ``on_scrape_complete`` once with the element count and the
        planned analyzer-skip decision.

        Logs the same one-liner via loguru so the CLI surface gets the
        signal too. Exceptions raised by the listener are caught and
        logged but never propagated — a misbehaving listener must not
        crash the pipeline mid-run.
        """
        elements_count = (
            scrape_result.get("elements_count")
            if isinstance(scrape_result, dict)
            else None
        )
        will_skip = self.will_skip_analyzer(data_path, scrape_result)

        if elements_count is not None:
            if will_skip:
                logger.info(f"Reusing prior analysis of {elements_count} cameras.")
            else:
                logger.info(f"Analyzing {elements_count} cameras…")

        if self.on_scrape_complete is None:
            return

        payload = {
            "scrape_result": scrape_result,
            "data_path": data_path,
            "elements_count": elements_count,
            "will_skip_analyzer": will_skip,
        }
        try:
            self.on_scrape_complete(payload)
        except Exception as e:
            logger.error(f"on_scrape_complete listener raised: {e}")

    def _handle_stage_result(
        self,
        stage_name: str,
        stage_result: Dict[str, Any],
        results: Dict[str, Any],
        city: str,
        failed_status: PipelineStatus = PipelineStatus.FAILED,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle stage completion with cancellation check and error handling.

        :param stage_name: Name of the stage (e.g., "scraping", "analysis")
        :param stage_result: Result dictionary from the stage
        :param results: Overall pipeline results dictionary
        :param city: City name for logging
        :param failed_status: Status to use if stage fails (FAILED or PARTIAL)
        :return: Finalized results if should exit, None if should continue
        """
        # Check for cancellation after stage completes
        if self._check_cancellation():
            logger.info(f"Pipeline cancelled after {stage_name} for {city}")
            return self._finalize_results(results, PipelineStatus.CANCELLED)

        # Check if stage was successful
        if not stage_result.get("success"):
            error_msg = stage_result.get("error", "Unknown error")

            # Set top-level error for API consumption
            results["error"] = f"{stage_name.capitalize()} failed: {error_msg}"

            if self.config.stop_on_error:
                return self._finalize_results(results, PipelineStatus.FAILED)
            else:
                self.errors.append(results["error"])
                return self._finalize_results(results, failed_status)

        return None

    def _finalize_results(
        self,
        results: Dict[str, Any],
        status: PipelineStatus,
    ) -> Dict[str, Any]:
        """
        Finalize pipeline execution and add metadata.

        :param results: Current results dictionary
        :param status: Final pipeline status
        :return: Complete results with metadata
        """
        self.status = status
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        results.update(
            {
                "status": status.value,
                "end_time": self.end_time.isoformat(),
                "duration_seconds": duration,
                "success": status == PipelineStatus.COMPLETED,
                "partial_success": status == PipelineStatus.PARTIAL,
                "cancelled": status == PipelineStatus.CANCELLED,
            }
        )

        if self.errors:
            results["errors"] = self.errors

        logger.info(f"Pipeline completed with status: {status.value} ({duration:.2f}s)")
        return results

    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        :return: Status dictionary
        """
        return {
            "status": self.status.value,
            "current_step": self.current_step,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "errors": self.errors,
        }


def create_pipeline(
    scenario: AnalysisScenario = AnalysisScenario.BASIC,
    **config_kwargs,
) -> SurveillancePipeline:
    """
    Factory function for creating configured pipelines.

    :param scenario: Analysis scenario to use
    :param config_kwargs: Additional configuration overrides
    :return: Configured SurveillancePipeline instance
    """
    config = PipelineConfig.from_scenario(scenario)

    # Apply any overrides
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    pipeline = SurveillancePipeline(config=config)
    logger.info(f"Created pipeline with {scenario.value} scenario")
    return pipeline
