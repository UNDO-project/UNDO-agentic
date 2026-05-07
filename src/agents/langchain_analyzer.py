from pathlib import Path
from typing import Dict, Any, Optional

from src.chains.analysis_chain import AnalysisChain
from src.config.logger import logger
from src.config.settings import LangChainSettings
from src.llm.surveillance_llm import create_surveillance_llm
from src.memory.store import MemoryStore


class SurveillanceAnalyzerAgent:
    """
    LangChain-based agent for analyzing and visualizing surveillance data.

    This agent orchestrates a complete analysis pipeline:
    - Load raw surveillance data
    - Enrich with LLM-powered analysis
    - Generate GeoJSON for mapping
    - Create visualizations and statistics
    - Handle errors gracefully with partial results

    Features:
    - Chain-based architecture using LangChain
    - Intelligent caching to avoid recomputation
    - Progressive error recovery
    - Flexible visualization options
    - Clear progress tracking
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        settings: Optional[LangChainSettings] = None,
    ):
        """
        Initialize the LangChain analyzer agent.

        :param name: Agent name for identification and caching
        :param memory: Memory store for caching results
        :param settings: Optional LangChain settings for LLM configuration
        """
        self.name = name
        self.memory = memory
        self.settings = settings or LangChainSettings()

        # Create LLM for enrichment
        self.llm = create_surveillance_llm(self.settings)

        # Create analysis chain
        self.chain = AnalysisChain(
            llm=self.llm,
            memory=memory,
            agent_name=name,
        )

        logger.info(f"Initialized {self.name} with LangChain analysis pipeline")

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze surveillance data through the complete pipeline.

        :param input_data: Dictionary with analysis parameters:
            - path: Path to raw Overpass JSON file (required)
            - generate_geojson: Generate GeoJSON (default: True)
            - generate_heatmap: Generate heatmap visualization (default: False)
            - generate_hotspots: Generate hotspot clusters (default: False)
            - compute_stats: Compute statistics (default: True)
            - generate_chart: Generate pie chart (default: False)
            - plot_zone_sensitivity: Plot zone sensitivity chart (default: False)
            - plot_sensitivity_reasons: Plot sensitivity reasons (default: False)
            - plot_hotspots: Plot hotspots visualization (default: False)
        :return: Dictionary with analysis results and output paths
        """
        # Extract path and validate
        path = Path(input_data["path"]).expanduser().resolve()
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "path": str(path),
            }

        logger.info(f"Starting analysis for {path.name}")

        # Extract visualization options
        options = {
            "generate_geojson": input_data.get("generate_geojson", True),
            "generate_heatmap": input_data.get("generate_heatmap", False),
            "generate_hotspots": input_data.get("generate_hotspots", False),
            "compute_stats": input_data.get("compute_stats", True),
            "generate_chart": input_data.get("generate_chart", False),
            "plot_zone_sensitivity": input_data.get("plot_zone_sensitivity", False),
            "plot_sensitivity_reasons": input_data.get(
                "plot_sensitivity_reasons", False
            ),
            "plot_hotspots": input_data.get("plot_hotspots", False),
            "plot_operator_distribution": input_data.get(
                "plot_operator_distribution", False
            ),
            "plot_manufacturer_distribution": input_data.get(
                "plot_manufacturer_distribution", False
            ),
            "generate_report": input_data.get("generate_report", False),
        }

        try:
            # Run core pipeline (Load → Enrich → Save → GeoJSON)
            result = self.chain.invoke({"path": str(path)})

            if not result.get("success", True):
                return result

            # Generate requested visualizations with error recovery
            result = self.chain.generate_visualizations(result, options)

            # Build response
            response = {
                "success": True,
                "path": str(path),
                "element_count": result.get("element_count", 0),
                "cache_hit": result.get("cache_hit", False),
                "enriched_path": result.get("enriched_path"),
                "geojson_path": result.get("geojson_path"),
            }

            # Add optional outputs
            if "heatmap_path" in result:
                response["heatmap_path"] = result["heatmap_path"]
            if "hotspots_path" in result:
                response["hotspots_path"] = result["hotspots_path"]
            if "stats" in result:
                response["stats"] = result["stats"]
            if "pie_chart_path" in result:
                response["pie_chart_path"] = result["pie_chart_path"]
            if "zone_sensitivity_chart" in result:
                response["zone_sensitivity_chart"] = result["zone_sensitivity_chart"]
            if "sensitivity_reasons_chart" in result:
                response["sensitivity_reasons_chart"] = result[
                    "sensitivity_reasons_chart"
                ]
            if "hotspots_chart" in result:
                response["hotspots_chart"] = result["hotspots_chart"]
            if "operator_chart_path" in result:
                response["operator_chart_path"] = result["operator_chart_path"]
            if "manufacturer_chart_path" in result:
                response["manufacturer_chart_path"] = result["manufacturer_chart_path"]
            if "report_path" in result:
                response["report_path"] = result["report_path"]

            # Add visualization errors if any
            if "visualization_errors" in result:
                response["visualization_errors"] = result["visualization_errors"]
                response["partial_success"] = True

            logger.info(f"Analysis completed for {path.name}")
            return response

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "path": str(path),
            }

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compatibility method with existing Agent interface.

        This method maintains backward compatibility with the original
        AnalyzerAgent interface for seamless migration.

        :param input_data: Dict with analysis parameters
        :return: Dict with analysis results
        """
        return self.analyze(input_data)


def create_analyzer_agent(
    name: str = "AnalyzerAgent",
    memory: Optional[MemoryStore] = None,
    settings: Optional[LangChainSettings] = None,
) -> SurveillanceAnalyzerAgent:
    """
    Factory function for creating SurveillanceAnalyzerAgent instances.

    :param name: Agent name for identification
    :param memory: Memory store for caching (creates default if None)
    :param settings: Optional LangChain settings
    :return: Configured SurveillanceAnalyzerAgent instance
    :raise: Exception if agent creation fails
    """
    try:
        if memory is None:
            from src.config.settings import DatabaseSettings

            db_settings = DatabaseSettings()
            memory = MemoryStore(db_url=db_settings.url)

        agent = SurveillanceAnalyzerAgent(name, memory, settings)
        logger.info(f"Created SurveillanceAnalyzerAgent: {name}")
        return agent
    except Exception as e:
        logger.error(f"Failed to create SurveillanceAnalyzerAgent: {e}")
        raise
