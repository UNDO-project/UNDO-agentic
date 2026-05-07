from pathlib import Path
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import Runnable, RunnableLambda

from src.config.logger import logger
from src.llm.surveillance_llm import SurveillanceLLM
from src.memory.store import MemoryStore
from src.utils.db import payload_hash


class AnalysisChain:
    """
    LangChain-based analysis chain for surveillance data processing.

    Implements a structured pipeline:
    1. Load → Check cache → Load or skip
    2. Enrich → Use LLM to analyze each element
    3. Save → Persist enriched data
    4. Transform → Generate GeoJSON
    5. Visualize → Create requested outputs

    Features:
    - Intelligent caching at each stage
    - Progressive error recovery
    - Intermediate result storage
    - Clear progress tracking
    """

    def __init__(
        self,
        llm: SurveillanceLLM,
        memory: MemoryStore,
        agent_name: str,
    ):
        """
        Initialize the analysis chain.

        :param llm: SurveillanceLLM instance for enrichment
        :param memory: MemoryStore for caching
        :param agent_name: Name for memory storage
        """
        self.llm = llm
        self.memory = memory
        self.agent_name = agent_name

        # Optional progress hook fired once per chunk in ``_enrich_data``.
        # Settable from the orchestrator (and from there, the API route)
        # so the polled task response can render live "Enriched N/total"
        # progress on the UI. Default ``None`` — chain is happy without
        # a listener.
        self.on_progress: Optional[Callable[[int, int], None]] = None

        # Build the core pipeline
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Runnable:
        """
        Build the main analysis pipeline as a LangChain Runnable.

        :return: Configured pipeline
        """
        # Core pipeline steps as runnables
        load_step = RunnableLambda(self._load_data)
        check_cache_step = RunnableLambda(self._check_cache)
        enrich_step = RunnableLambda(self._enrich_data)
        save_step = RunnableLambda(self._save_enriched)
        geojson_step = RunnableLambda(self._generate_geojson)

        # Build sequential pipeline
        pipeline = load_step | check_cache_step | enrich_step | save_step | geojson_step

        return pipeline

    @staticmethod
    def _load_data(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load raw surveillance data from file.

        :param input_dict: Dictionary with 'path' key
        :return: Updated dictionary with loaded elements
        """
        from src.tools.io_tools import load_overpass_elements

        path = Path(input_dict["path"])
        logger.info(f"Loading data from {path}")

        elements = load_overpass_elements(path)

        # Calculate hash for caching
        raw_hash = payload_hash({"elements": elements})

        return {
            **input_dict,
            "elements": elements,
            "raw_hash": raw_hash,
            "element_count": len(elements),
        }

    def _check_cache(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if enriched data already exists in cache.

        :param context: Current pipeline context
        :return: Updated context with cache status
        """
        path = Path(context["path"])
        enriched_path = path.with_name(f"{path.stem}_enriched.json")
        geojson_path = enriched_path.with_suffix(".geojson")

        # Check filesystem
        enriched_exists = enriched_path.exists()
        geojson_exists = geojson_path.exists()

        context["enriched_path"] = str(enriched_path)
        context["geojson_path"] = str(geojson_path)
        context["enriched_exists"] = enriched_exists
        context["geojson_exists"] = geojson_exists

        # Check memory cache
        raw_hash = context["raw_hash"]
        cache_hit = False

        for mem in self.memory.load(self.agent_name):
            if mem.step == "enriched_cache" and mem.content.startswith(raw_hash):
                _, cached_enriched, cached_geojson = mem.content.split("|")
                if Path(cached_enriched).exists() and Path(cached_geojson).exists():
                    logger.debug(f"Cache hit for {path.name}")
                    context["enriched_path"] = cached_enriched
                    context["geojson_path"] = cached_geojson
                    cache_hit = True
                    break

        context["cache_hit"] = cache_hit

        if cache_hit or enriched_exists:
            logger.debug("Using cached enriched data")

        return context

    def _enrich_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich surveillance elements using the LLM in chunks of
        ``LangChainSettings.batch_size``.

        Each chunk is one ``chain.batch(...)`` call against Ollama. If a
        chunk's batch call raises (transport error, model not loaded),
        every element in that chunk is annotated with ``{"error": ...}``
        and the loop continues with the next chunk — no per-element
        fallback path.

        :param context: Current pipeline context
        :return: Updated context with enriched elements
        """
        # Skip if cache hit
        if context.get("cache_hit") or context.get("enriched_exists"):
            import json

            enriched_path = Path(context["enriched_path"])
            enriched_data = json.loads(enriched_path.read_text())
            context["enriched"] = enriched_data["elements"]
            logger.debug(f"Loaded {len(context['enriched'])} cached enriched elements")
            return context

        elements = context["elements"]
        total = len(elements)
        batch_size = self.llm.settings.batch_size
        logger.info(f"Enriching {total} elements (batch_size={batch_size})...")
        enriched = []

        for chunk_start in range(0, total, batch_size):
            chunk = elements[chunk_start : chunk_start + batch_size]
            try:
                analyses = self.llm.analyze_surveillance_elements_batch(chunk)
            except Exception as e:
                logger.warning(
                    f"Batch enrichment failed for chunk starting at index "
                    f"{chunk_start} ({len(chunk)} elements): {e}"
                )
                analyses = [{"error": str(e)}] * len(chunk)

            for element, analysis in zip(chunk, analyses):
                enriched.append({**element, "analysis": analysis})
            logger.info(f"Enriched {len(enriched)}/{total} elements")

            # Fire the progress hook so the API route can surface live
            # counts on the polled task response. Listener exceptions are
            # swallowed so a misbehaving consumer never crashes a
            # multi-minute analyzer run.
            if self.on_progress is not None:
                try:
                    self.on_progress(len(enriched), total)
                except Exception as e:
                    logger.warning(f"on_progress listener raised: {e}")

        context["enriched"] = enriched
        logger.info(f"Successfully enriched {len(enriched)} elements")
        return context

    def _enrich_element_fallback(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback enrichment method using basic LLM generation.

        :param element: Raw OSM element
        :return: Analysis dictionary
        """
        import json
        from src.config.models.surveillance_metadata import SurveillanceMetadata

        # Use basic prompt
        tags_json = json.dumps(element.get("tags", {}), ensure_ascii=False, indent=2)
        prompt = f"Analyze these surveillance camera tags and return JSON: {tags_json}"

        try:
            raw = self.llm.generate_response(prompt)
            # Try to parse as JSON
            enriched_fields = json.loads(raw)
            # Validate with schema
            meta = SurveillanceMetadata.from_raw(element, enriched_fields)
            return meta.model_dump(exclude_none=True)
        except Exception as e:
            logger.warning(f"Fallback enrichment failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def _save_enriched(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save enriched data to disk.

        :param context: Current pipeline context
        :return: Updated context with save path
        """
        # Skip if cache hit
        if context.get("cache_hit") or context.get("enriched_exists"):
            logger.debug("Skipping save (using cache)")
            return context

        from src.tools.io_tools import save_enriched_elements

        enriched_path = save_enriched_elements(context["enriched"], context["path"])

        context["enriched_path"] = str(enriched_path)
        logger.info(f"Saved enriched data to {enriched_path}")
        return context

    def _generate_geojson(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate GeoJSON from enriched data.

        :param context: Current pipeline context
        :return: Updated context with GeoJSON path
        """
        # Skip if cache hit or exists
        if context.get("cache_hit") or context.get("geojson_exists"):
            logger.debug("Skipping GeoJSON generation (using cache)")
            # Store cache entry if not already cached
            if not context.get("cache_hit"):
                cache_value = f"{context['raw_hash']}|{context['enriched_path']}|{context['geojson_path']}"
                self.memory.store(self.agent_name, "enriched_cache", cache_value)
            return context

        from src.tools.io_tools import to_geojson

        enriched_path = Path(context["enriched_path"])
        geojson_path = enriched_path.with_suffix(".geojson")

        to_geojson(enriched_path, geojson_path)

        context["geojson_path"] = str(geojson_path)
        logger.info(f"Generated GeoJSON at {geojson_path}")

        # Cache the result
        cache_value = f"{context['raw_hash']}|{context['enriched_path']}|{geojson_path}"
        self.memory.store(self.agent_name, "enriched_cache", cache_value)

        return context

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full analysis pipeline.

        :param input_dict: Input dictionary with 'path' key
        :return: Results dictionary with all outputs
        """
        try:
            result = self.pipeline.invoke(input_dict)
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Analysis chain failed: {e}")
            return {
                **input_dict,
                "success": False,
                "error": str(e),
            }

    def generate_visualizations(
        self,
        context: Dict[str, Any],
        options: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Generate requested visualizations with error recovery.

        :param context: Pipeline context with enriched data
        :param options: Dictionary of visualization options
        :return: Updated context with visualization paths
        """
        from src.tools.mapping_tools import to_heatmap, to_hotspots
        from src.tools.stat_tools import compute_statistics
        from src.tools.chart_tools import (
            private_public_pie,
            plot_zone_sensitivity,
            plot_sensitivity_reasons,
            plot_hotspots as plot_hotspots_chart,
            plot_operator_distribution,
            plot_manufacturer_distribution,
            plot_install_timeline,
        )

        errors = []

        # Generate heatmap
        if options.get("generate_heatmap"):
            try:
                geojson_path = Path(context["geojson_path"])
                heatmap_path = geojson_path.with_suffix(".html")
                to_heatmap(geojson_path, heatmap_path)
                context["heatmap_path"] = str(heatmap_path)
                logger.info(f"Generated heatmap at {heatmap_path}")
            except Exception as e:
                error_msg = f"Heatmap generation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Generate hotspots
        if options.get("generate_hotspots"):
            try:
                geojson_path = Path(context["geojson_path"])
                hotspots_path = geojson_path.with_name(
                    f"{geojson_path.stem}_hotspots.geojson"
                )
                to_hotspots(geojson_path, hotspots_path)
                context["hotspots_path"] = str(hotspots_path)
                logger.info(f"Generated hotspots at {hotspots_path}")
            except Exception as e:
                error_msg = f"Hotspots generation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Compute statistics
        if options.get("compute_stats", True):
            try:
                stats = compute_statistics(context["enriched"])
                context["stats"] = stats
                logger.info("Computed statistics")
            except Exception as e:
                error_msg = f"Statistics computation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Generate charts (only if stats available)
        if "stats" in context:
            output_dir = Path(context["path"]).parent

            if options.get("generate_chart"):
                try:
                    chart_path = private_public_pie(context["stats"], output_dir)
                    context["pie_chart_path"] = str(chart_path)
                    logger.info(f"Generated pie chart at {chart_path}")
                except Exception as e:
                    error_msg = f"Pie chart generation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if options.get("plot_zone_sensitivity"):
                try:
                    chart_path = plot_zone_sensitivity(context["stats"], output_dir)
                    context["zone_sensitivity_chart"] = str(chart_path)
                    logger.info(f"Generated zone sensitivity chart at {chart_path}")
                except Exception as e:
                    error_msg = f"Zone sensitivity chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if options.get("plot_sensitivity_reasons"):
                try:
                    enriched_path = Path(context["enriched_path"])
                    chart_path = enriched_path.with_name(
                        f"{enriched_path.stem}_sensitivity.png"
                    )
                    plot_sensitivity_reasons(enriched_path, chart_path)
                    context["sensitivity_reasons_chart"] = str(chart_path)
                    logger.info(f"Generated sensitivity reasons chart at {chart_path}")
                except Exception as e:
                    error_msg = f"Sensitivity reasons chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if options.get("plot_hotspots") and "hotspots_path" in context:
                try:
                    hotspots_path = Path(context["hotspots_path"])
                    chart_path = hotspots_path.with_suffix(".png")
                    plot_hotspots_chart(hotspots_path, chart_path)
                    context["hotspots_chart"] = str(chart_path)
                    logger.info(f"Generated hotspots chart at {chart_path}")
                except Exception as e:
                    error_msg = f"Hotspots chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Per-city stems on these charts so a casually-downloaded PNG
            # is self-identifying without depending on the directory name.
            city_stem = Path(context["path"]).stem

            if options.get("plot_operator_distribution"):
                try:
                    chart_path = plot_operator_distribution(
                        context["stats"],
                        output_dir,
                        filename=f"operator_distribution_{city_stem}.png",
                    )
                    context["operator_chart_path"] = str(chart_path)
                    logger.info(
                        f"Generated operator distribution chart at {chart_path}"
                    )
                except Exception as e:
                    error_msg = f"Operator distribution chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if options.get("plot_manufacturer_distribution"):
                try:
                    chart_path = plot_manufacturer_distribution(
                        context["stats"],
                        output_dir,
                        filename=f"manufacturer_distribution_{city_stem}.png",
                    )
                    context["manufacturer_chart_path"] = str(chart_path)
                    logger.info(
                        f"Generated manufacturer distribution chart at {chart_path}"
                    )
                except Exception as e:
                    error_msg = f"Manufacturer distribution chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            if options.get("plot_install_timeline"):
                try:
                    chart_path = plot_install_timeline(
                        context["stats"],
                        output_dir,
                        filename=f"install_timeline_{city_stem}.png",
                    )
                    context["install_timeline_chart_path"] = str(chart_path)
                    logger.info(f"Generated install timeline chart at {chart_path}")
                except Exception as e:
                    error_msg = f"Install timeline chart failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # LLM-generated narrative report. Depends on ``stats`` having
            # been computed in the same call (we don't try to re-derive
            # them on the report-only path — keep options orthogonal).
            if options.get("generate_report"):
                try:
                    raw_path = Path(context["path"])
                    report_path = raw_path.with_name(f"{raw_path.stem}_report.md")
                    sample = [
                        el
                        for el in context.get("enriched", [])
                        if isinstance(el, dict)
                        and isinstance(el.get("analysis"), dict)
                        and el["analysis"].get("sensitive")
                    ]
                    markdown = self.llm.generate_city_report(context["stats"], sample)
                    report_path.write_text(markdown, encoding="utf-8")
                    context["report_path"] = str(report_path)
                    logger.info(f"Generated city report at {report_path}")
                except Exception as e:
                    error_msg = f"City report generation failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

        # Add errors to context if any occurred
        if errors:
            context["visualization_errors"] = errors
            logger.warning(f"Completed with {len(errors)} visualization errors")

        return context
