"""Route Finder Agent for computing low-surveillance walking routes."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.base_agent import Agent
from src.config.logger import logger
from src.config.models.route_models import RouteMetrics, RouteRequest, RouteResult
from src.config.settings import RouteSettings
from src.memory.store import MemoryStore
from src.tools.routing_tools import (
    build_pedestrian_graph,
    build_route_geojson,
    compute_exposure_for_path,
    count_camera_features,
    generate_candidate_paths,
    load_camera_points,
    render_route_map,
    snap_to_graph,
)

Tool = Callable[..., Any]


class RouteFinderAgent(Agent):
    """
    Agent that computes low-surveillance walking routes.

    The agent orchestrates the routing workflow:
    1. Load camera data and pedestrian network
    2. Generate candidate paths between start and end points
    3. Score paths based on camera exposure
    4. Select optimal low-surveillance route
    5. Generate output artifacts (GeoJSON, HTML map)

    Results are cached in memory to avoid recomputation for identical requests.
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        settings: RouteSettings,
        tools: Optional[Dict[str, Tool]] = None,
    ) -> None:
        """
        Initialize the RouteFinderAgent.

        :param name: Agent name for identification.
        :param memory: Memory store for caching route results.
        :param settings: Route computation settings.
        :param tools: Optional tool overrides (primarily for testing).
        """
        default_tools: Dict[str, Tool] = {
            "load_cameras": load_camera_points,
            "build_graph": build_pedestrian_graph,
            "snap_start": snap_to_graph,
            "snap_end": snap_to_graph,
            "generate_paths": generate_candidate_paths,
            "compute_exposure": compute_exposure_for_path,
            "build_geojson": build_route_geojson,
            "render_map": render_route_map,
        }
        super().__init__(name=name, tools=tools or default_tools, memory=memory)
        self.settings = settings

    def perceive(self, input_data: RouteRequest) -> Dict[str, Any]:
        """
        Process route request and check for cached results.

        :param input_data: RouteRequest with city, coordinates, and optional data path.
        :return: Enriched observation with cache status and file paths.
        """
        # Create cache key from request parameters. An active camera
        # filter is appended so it cannot share a cache slot with a
        # different filter — but an absent / inert filter contributes
        # nothing to the key, keeping old cache entries valid.
        filter_signature = self._filter_signature(input_data.camera_filter)
        cache_key_input = (
            f"{input_data.city}_{input_data.country}_"
            f"{input_data.start_lat}_{input_data.start_lon}_"
            f"{input_data.end_lat}_{input_data.end_lon}_"
            f"{self.settings.max_candidates}_{self.settings.buffer_radius_m}"
        )
        if filter_signature:
            cache_key_input = f"{cache_key_input}_{filter_signature}"
        cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()[:16]

        # Always derive city_slug for output paths
        city_slug = input_data.city.lower().replace(" ", "_")

        # Determine data path
        if input_data.data_path:
            cameras_geojson_path = input_data.data_path
        else:
            # Use default pipeline output location
            cameras_geojson_path = (
                Path("overpass_data") / city_slug / f"{city_slug}_enriched.geojson"
            )

        # Setup output paths
        output_dir = Path("overpass_data") / city_slug / "routes"
        route_geojson_path = output_dir / f"route_{cache_key}.geojson"
        route_map_path = output_dir / f"route_{cache_key}.html"

        # Check memory cache
        cache_hit = False
        cached_result = None

        if self.memory:
            for mem in self.memory.load(self.name):
                if mem.step == "route_cache" and mem.content.startswith(cache_key):
                    # Parse cached content: cache_key|route_geojson|route_map|metrics_json
                    parts = mem.content.split("|", 3)
                    if len(parts) == 4:
                        _, cached_geojson, cached_map, metrics_json = parts

                        # Verify files still exist
                        if Path(cached_geojson).exists() and Path(cached_map).exists():
                            cache_hit = True
                            cached_result = {
                                "route_geojson_path": Path(cached_geojson),
                                "route_map_path": Path(cached_map),
                                "metrics": RouteMetrics(**json.loads(metrics_json)),
                            }
                            logger.debug(f"Cache hit for route request: {cache_key}")
                            break

        observation = {
            "city": input_data.city,
            "country": input_data.country,
            "start_lat": input_data.start_lat,
            "start_lon": input_data.start_lon,
            "end_lat": input_data.end_lat,
            "end_lon": input_data.end_lon,
            "camera_filter": input_data.camera_filter,
            "cameras_geojson_path": cameras_geojson_path,
            "route_geojson_path": route_geojson_path,
            "route_map_path": route_map_path,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "cached_result": cached_result,
        }

        return observation

    @staticmethod
    def _filter_signature(camera_filter) -> str:
        """Serialise a CameraFilter for cache-key inclusion.

        Returns ``""`` for ``None`` or an inert filter so that adding a
        no-op filter doesn't invalidate a previously cached unfiltered
        route. Active filters serialise their fields with sorted keys
        for determinism.
        """
        if camera_filter is None or camera_filter.is_inert():
            return ""
        return json.dumps(camera_filter.model_dump(), sort_keys=True, default=str)

    def plan(self, observation: Dict[str, Any]) -> List[str]:
        """
        Determine action sequence based on cache status.

        :param observation: Output from perceive().
        :return: List of action names to execute.
        """
        if observation["cache_hit"]:
            # Skip computation, just return cached result
            return []

        # Full routing workflow
        return [
            "load_cameras",
            "build_graph",
            "snap_start",
            "snap_end",
            "generate_paths",
            "score_paths",
            "build_geojson",
            "render_map",
        ]

    def act(self, action: str, context: Dict[str, Any]) -> Any:
        """
        Execute a routing action using the appropriate tool.

        :param action: The name of the action/tool to invoke.
        :param context: Data from perceive() and previous actions.
        :return: Result of the action.
        :raises ValueError: If action is not recognized.
        """
        if action == "load_cameras":
            cameras_path = context["cameras_geojson_path"]
            camera_filter = context.get("camera_filter")
            # ``count_camera_features`` walks the GeoJSON once to give us
            # the pre-filter denominator for the response, then the
            # filtered load happens normally. Any test override via the
            # ``load_cameras`` tool slot keeps its existing single-arg
            # contract — the filter is forwarded only when it's set.
            try:
                total = count_camera_features(cameras_path)
            except FileNotFoundError:
                total = 0
            if camera_filter is not None:
                cameras = self.tools["load_cameras"](
                    cameras_path, camera_filter=camera_filter
                )
            else:
                cameras = self.tools["load_cameras"](cameras_path)
            context["cameras"] = cameras
            context["camera_count_total"] = total
            return cameras

        elif action == "build_graph":
            graph = self.tools["build_graph"](
                context["city"],
                context["country"],
                self.settings,
            )
            context["graph"] = graph
            return graph

        elif action == "snap_start":
            start_node = self.tools["snap_start"](
                context["graph"],
                context["start_lat"],
                context["start_lon"],
                self.settings,
            )
            context["start_node"] = start_node
            return start_node

        elif action == "snap_end":
            end_node = self.tools["snap_end"](
                context["graph"],
                context["end_lat"],
                context["end_lon"],
                self.settings,
            )
            context["end_node"] = end_node
            return end_node

        elif action == "generate_paths":
            candidate_paths = self.tools["generate_paths"](
                context["graph"],
                context["start_node"],
                context["end_node"],
                self.settings.max_candidates,
            )
            context["candidate_paths"] = candidate_paths
            logger.info(f"Generated {len(candidate_paths)} candidate paths")
            return candidate_paths

        elif action == "score_paths":
            # Score all candidate paths and select the best one
            # Performance optimization: Build camera GeoDataFrame once
            import geopandas as gpd
            from shapely.geometry import Point

            cameras = context["cameras"]
            if len(cameras) > 0:
                camera_gdf = gpd.GeoDataFrame(
                    geometry=[Point(lon, lat) for lat, lon in cameras],
                    crs="EPSG:4326",
                )
                logger.debug(f"Built spatial index for {len(cameras)} cameras")
            else:
                camera_gdf = None

            scored_paths = []

            for i, path in enumerate(context["candidate_paths"]):
                metrics = self.tools["compute_exposure"](
                    context["graph"],
                    path,
                    context["cameras"],
                    self.settings,
                    camera_gdf,  # Reuse pre-built GeoDataFrame
                )

                # Compute baseline for comparison (only for first path)
                if i == 0:
                    context["baseline_metrics"] = metrics

                scored_paths.append((path, metrics))
                logger.debug(
                    f"Path {i + 1}: length={metrics.length_m:.1f}m, "
                    f"exposure={metrics.exposure_score:.2f} cameras/km"
                )

            # Select path with minimum exposure score
            best_path, best_metrics = min(
                scored_paths, key=lambda x: x[1].exposure_score
            )

            # Enrich best metrics with baseline comparison
            best_metrics.baseline_length_m = context["baseline_metrics"].length_m
            best_metrics.baseline_exposure_score = context[
                "baseline_metrics"
            ].exposure_score
            # Propagate the pre-filter total so the response carries the
            # "X of Y cameras considered" denominator. ``camera_count_near_route``
            # already came back from compute_exposure as the filtered hit count.
            best_metrics.camera_count_total = context.get("camera_count_total")

            context["best_path"] = best_path
            context["best_metrics"] = best_metrics

            logger.info(
                f"Selected route: {best_metrics.length_m:.1f}m, "
                f"{best_metrics.exposure_score:.2f} cameras/km "
                f"(baseline: {best_metrics.baseline_exposure_score:.2f} cameras/km)"
            )

            return best_metrics

        elif action == "build_geojson":
            geojson_path = self.tools["build_geojson"](
                context["graph"],
                context["best_path"],
                context["best_metrics"],
                context["cameras"],
                context["city"],
                context["route_geojson_path"],
                self.settings,
            )
            return geojson_path

        elif action == "render_map":
            map_path = self.tools["render_map"](
                context["route_geojson_path"],
                context["cameras_geojson_path"],
                context["route_map_path"],
            )
            return map_path

        else:
            raise ValueError(f"Unknown action: {action}")

    def think(self, intermediate: Any) -> Dict[str, Any]:
        """
        Update context between actions.

        :param intermediate: Result from the last action.
        :return: Updated context dict.
        """
        # The context dict is being mutated in act(), so we just return it
        # This follows the pattern where context accumulates results
        if isinstance(intermediate, dict):
            return intermediate
        return {"result": intermediate}

    def achieve_goal(self, input_data: RouteRequest) -> RouteResult:
        """
        Compute a low-surveillance route and return the result.

        :param input_data: RouteRequest with routing parameters.
        :return: RouteResult with output paths and metrics.
        """
        observation = self.perceive(input_data)

        # Check for cache hit
        if observation["cache_hit"]:
            cached = observation["cached_result"]
            return RouteResult(
                route_id=observation["cache_key"],
                city=input_data.city,
                route_geojson_path=cached["route_geojson_path"],
                route_map_path=cached["route_map_path"],
                metrics=cached["metrics"],
                from_cache=True,
            )

        # Execute routing workflow
        plan_steps = self.plan(observation)
        context = observation

        for step in plan_steps:
            self.act(step, context)
            # Note: Intermediate results are stored in context, not in memory
            # The final route result is cached separately below

        # Cache the result
        if self.memory:
            metrics_json = context["best_metrics"].model_dump_json()
            cache_content = (
                f"{context['cache_key']}|"
                f"{context['route_geojson_path']}|"
                f"{context['route_map_path']}|"
                f"{metrics_json}"
            )
            self.memory.store(self.name, "route_cache", cache_content)

        # Build final result
        return RouteResult(
            route_id=context["cache_key"],
            city=input_data.city,
            route_geojson_path=context["route_geojson_path"],
            route_map_path=context["route_map_path"],
            metrics=context["best_metrics"],
            from_cache=False,
        )
