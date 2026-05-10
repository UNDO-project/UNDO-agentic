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

    def _cached_step(
        self,
        *,
        vis_name: str,
        error_label: str,
        artifact_path: Path,
        cache_key: str,
        fn: Callable[[], Any],
        errors: list,
        force_rerender: bool,
    ) -> Optional[Path]:
        """
        Run one visualisation step with the per-artifact cache wrapped in.

        On a cache hit (artifact + matching sidecar present, ``force_rerender``
        off) the underlying ``fn`` is not called — the existing path is
        returned. On a miss, ``fn`` runs; on success a sidecar is written
        next to the artifact recording the cache key. On failure the error
        is appended to ``errors`` (matching the legacy log format
        ``"<error_label> failed: <e>"``) and ``None`` is returned.

        ``fn`` returning ``None`` or ``False`` is interpreted as a
        deliberate opt-out (e.g. an empty-counter chart). No sidecar is
        written and ``None`` is returned, so the chain context never
        records a path to a file that doesn't exist.

        :return: The artifact path on success/cache-hit; ``None`` on
            failure or opt-out.
        """
        from src.tools.io_tools import cache_hit, write_sidecar

        if not force_rerender and cache_hit(artifact_path, cache_key):
            logger.info(f"Reusing cached {vis_name} at {artifact_path}")
            return artifact_path
        try:
            result = fn()
        except Exception as e:
            error_msg = f"{error_label} failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None
        if result is None or result is False:
            logger.info(f"{vis_name}: nothing to render — skipping artifact")
            return None
        write_sidecar(artifact_path, cache_key)
        logger.info(f"Generated {vis_name} at {artifact_path}")
        return artifact_path

    def generate_visualizations(
        self,
        context: Dict[str, Any],
        options: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Generate requested visualizations with error recovery.

        Each artifact step is wrapped in :meth:`_cached_step` so a re-run
        with identical ``raw_hash`` and option set is a no-op (per
        Architecture Proposal #5). ``options["force_rerender"]`` bypasses
        the artifact cache without touching the upstream enrichment cache.

        :param context: Pipeline context with enriched data
        :param options: Dictionary of visualization options
        :return: Updated context with visualization paths
        """
        from src.tools.density_kde import (
            compute_kde,
            write_density_geojson,
            write_kde_heatmap_html,
        )
        from src.tools.hotspot_clustering import (
            cluster_hdbscan,
            write_centroids_geojson,
            write_polygons_geojson,
        )
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
        from src.tools.io_tools import visualization_cache_key
        from src.config.settings import HotspotSettings

        hotspot_settings = HotspotSettings()

        errors: list = []
        force_rerender = bool(options.get("force_rerender", False))
        raw_hash = str(context.get("raw_hash") or "")

        # KDE-backed heatmap + density contours — two artifacts share one
        # KDE pass. The filenames derive from the city stem
        # (``<city>_heatmap.html``, ``<city>_density.geojson``), not from
        # the geojson path's suffix — the ``/api/v1/outputs/{city}/...``
        # routes serve those exact names. ``compute_kde`` is memoised so
        # the second ``_cached_step`` reuses the first call's result on
        # a cache miss.
        if options.get("generate_heatmap"):
            geojson_path = Path(context["geojson_path"])
            raw_path = Path(context["path"])
            heatmap_path = raw_path.with_name(f"{raw_path.stem}_heatmap.html")
            density_path = raw_path.with_name(f"{raw_path.stem}_density.geojson")

            kde_params = {
                "method": "kde",
                "bandwidth": str(hotspot_settings.kde_bandwidth),
                "grid_resolution_m": hotspot_settings.kde_grid_resolution_m,
            }

            kde_cache: dict = {}

            def _run_kde():
                if "result" not in kde_cache:
                    kde_cache["result"] = compute_kde(geojson_path, hotspot_settings)
                return kde_cache["result"]

            out = self._cached_step(
                vis_name="heatmap",
                error_label="Heatmap generation",
                artifact_path=heatmap_path,
                # Cache key bumps to "heatmap_kde" so any sidecar from
                # the legacy raw-points heatmap is invalidated on first
                # run after upgrade.
                cache_key=visualization_cache_key(raw_hash, "heatmap_kde", kde_params),
                fn=lambda: write_kde_heatmap_html(_run_kde(), heatmap_path),
                errors=errors,
                force_rerender=force_rerender,
            )
            if out is not None:
                context["heatmap_path"] = str(out)

            out = self._cached_step(
                vis_name="density contours",
                error_label="Density contours generation",
                artifact_path=density_path,
                cache_key=visualization_cache_key(raw_hash, "density_kde", kde_params),
                fn=lambda: write_density_geojson(_run_kde(), density_path),
                errors=errors,
                force_rerender=force_rerender,
            )
            if out is not None:
                context["density_path"] = str(out)

        # Generate HDBSCAN hotspots — two artifacts share one cluster
        # pass: ``<city>_hotspots.geojson`` (centroids) and
        # ``<city>_hotspot_polygons.geojson`` (convex hulls). We cluster
        # once and memoise via ``_run_clustering`` so the second
        # ``_cached_step`` reuses the first call's result on a cache miss.
        if options.get("generate_hotspots"):
            geojson_path = Path(context["geojson_path"])
            raw_path = Path(context["path"])
            hotspots_path = raw_path.with_name(f"{raw_path.stem}_hotspots.geojson")
            polygons_path = raw_path.with_name(
                f"{raw_path.stem}_hotspot_polygons.geojson"
            )

            # Cache key bumps to "hotspots_hdbscan" so any sidecar from
            # the legacy DBSCAN run is invalidated and the new artifact
            # is regenerated on first run after upgrade.
            cluster_params = {
                "method": "hdbscan",
                "min_cluster_size": hotspot_settings.hdbscan_min_cluster_size,
                "min_samples": hotspot_settings.hdbscan_min_samples,
            }

            cluster_cache: dict = {}

            def _run_clustering():
                if "result" not in cluster_cache:
                    cluster_cache["result"] = cluster_hdbscan(
                        geojson_path, hotspot_settings
                    )
                return cluster_cache["result"]

            out = self._cached_step(
                vis_name="hotspot centroids",
                error_label="Hotspot centroids generation",
                artifact_path=hotspots_path,
                cache_key=visualization_cache_key(
                    raw_hash, "hotspots_hdbscan", cluster_params
                ),
                fn=lambda: write_centroids_geojson(_run_clustering(), hotspots_path),
                errors=errors,
                force_rerender=force_rerender,
            )
            if out is not None:
                context["hotspots_path"] = str(out)

            out = self._cached_step(
                vis_name="hotspot polygons",
                error_label="Hotspot polygons generation",
                artifact_path=polygons_path,
                cache_key=visualization_cache_key(
                    raw_hash, "hotspot_polygons_hdbscan", cluster_params
                ),
                fn=lambda: write_polygons_geojson(_run_clustering(), polygons_path),
                errors=errors,
                force_rerender=force_rerender,
            )
            if out is not None:
                context["hotspot_polygons_path"] = str(out)

        # Compute statistics (in-memory; no artifact, no caching)
        if options.get("compute_stats", True):
            try:
                stats = compute_statistics(context["enriched"])
                context["stats"] = stats
                logger.info("Computed statistics")
            except Exception as e:
                error_msg = f"Statistics computation failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Generate charts (only if stats available). Every artifact is
        # named ``<city>_<artifact>.<ext>`` so a single naming
        # convention covers heatmap, hotspots, every chart, and the
        # report — making the ``/list`` glob and the dashboard's
        # ``findFile`` substring lookups predictable.
        if "stats" in context:
            output_dir = Path(context["path"]).parent
            city_stem = Path(context["path"]).stem

            if options.get("generate_chart"):
                pie_filename = f"{city_stem}_privacy.png"
                pie_path = output_dir / pie_filename
                out = self._cached_step(
                    vis_name="pie chart",
                    error_label="Pie chart generation",
                    artifact_path=pie_path,
                    cache_key=visualization_cache_key(raw_hash, "pie_chart", {}),
                    fn=lambda: private_public_pie(
                        context["stats"], output_dir, filename=pie_filename
                    ),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["pie_chart_path"] = str(out)

            if options.get("plot_zone_sensitivity"):
                zone_filename = f"{city_stem}_zone_sensitivity.png"
                zone_path = output_dir / zone_filename
                out = self._cached_step(
                    vis_name="zone sensitivity chart",
                    error_label="Zone sensitivity chart",
                    artifact_path=zone_path,
                    cache_key=visualization_cache_key(
                        raw_hash, "zone_sensitivity", {"top_n": 10}
                    ),
                    fn=lambda: plot_zone_sensitivity(
                        context["stats"], output_dir, filename=zone_filename
                    ),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["zone_sensitivity_chart"] = str(out)

            if options.get("plot_sensitivity_reasons"):
                enriched_path = Path(context["enriched_path"])
                reasons_path = output_dir / f"{city_stem}_sensitivity_reasons.png"
                out = self._cached_step(
                    vis_name="sensitivity reasons chart",
                    error_label="Sensitivity reasons chart",
                    artifact_path=reasons_path,
                    cache_key=visualization_cache_key(
                        raw_hash, "sensitivity_reasons", {"top_n": 5}
                    ),
                    fn=lambda: plot_sensitivity_reasons(enriched_path, reasons_path),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["sensitivity_reasons_chart"] = str(out)

            if options.get("plot_hotspots") and "hotspots_path" in context:
                hotspots_path = Path(context["hotspots_path"])
                hotspots_chart_path = output_dir / f"{city_stem}_hotspots.png"
                out = self._cached_step(
                    vis_name="hotspots chart",
                    error_label="Hotspots chart",
                    artifact_path=hotspots_chart_path,
                    cache_key=visualization_cache_key(raw_hash, "hotspots_chart", {}),
                    fn=lambda: plot_hotspots_chart(hotspots_path, hotspots_chart_path),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["hotspots_chart"] = str(out)

            if options.get("plot_operator_distribution"):
                op_filename = f"{city_stem}_operator_distribution.png"
                op_path = output_dir / op_filename
                out = self._cached_step(
                    vis_name="operator distribution chart",
                    error_label="Operator distribution chart",
                    artifact_path=op_path,
                    cache_key=visualization_cache_key(
                        raw_hash, "operator_distribution", {"top_n": 10}
                    ),
                    fn=lambda: plot_operator_distribution(
                        context["stats"], output_dir, filename=op_filename
                    ),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["operator_chart_path"] = str(out)

            if options.get("plot_manufacturer_distribution"):
                mf_filename = f"{city_stem}_manufacturer_distribution.png"
                mf_path = output_dir / mf_filename
                out = self._cached_step(
                    vis_name="manufacturer distribution chart",
                    error_label="Manufacturer distribution chart",
                    artifact_path=mf_path,
                    cache_key=visualization_cache_key(
                        raw_hash, "manufacturer_distribution", {"top_n": 10}
                    ),
                    fn=lambda: plot_manufacturer_distribution(
                        context["stats"], output_dir, filename=mf_filename
                    ),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["manufacturer_chart_path"] = str(out)

            if options.get("plot_install_timeline"):
                tl_filename = f"{city_stem}_install_timeline.png"
                tl_path = output_dir / tl_filename
                out = self._cached_step(
                    vis_name="install timeline chart",
                    error_label="Install timeline chart",
                    artifact_path=tl_path,
                    cache_key=visualization_cache_key(raw_hash, "install_timeline", {}),
                    fn=lambda: plot_install_timeline(
                        context["stats"], output_dir, filename=tl_filename
                    ),
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["install_timeline_chart_path"] = str(out)

            # LLM-generated narrative report. Depends on ``stats`` having
            # been computed in the same call (we don't try to re-derive
            # them on the report-only path — keep options orthogonal).
            if options.get("generate_report"):
                raw_path = Path(context["path"])
                report_path = raw_path.with_name(f"{raw_path.stem}_report.md")

                def _generate_report() -> Path:
                    sample = [
                        el
                        for el in context.get("enriched", [])
                        if isinstance(el, dict)
                        and isinstance(el.get("analysis"), dict)
                        and el["analysis"].get("sensitive")
                    ]
                    markdown = self.llm.generate_city_report(context["stats"], sample)
                    report_path.write_text(markdown, encoding="utf-8")
                    # Truthy return so ``_cached_step`` doesn't treat the
                    # implicit ``None`` as an opt-out.
                    return report_path

                out = self._cached_step(
                    vis_name="city report",
                    error_label="City report generation",
                    artifact_path=report_path,
                    cache_key=visualization_cache_key(raw_hash, "city_report", {}),
                    fn=_generate_report,
                    errors=errors,
                    force_rerender=force_rerender,
                )
                if out is not None:
                    context["report_path"] = str(out)

        # Add errors to context if any occurred
        if errors:
            context["visualization_errors"] = errors
            logger.warning(f"Completed with {len(errors)} visualization errors")

        return context
