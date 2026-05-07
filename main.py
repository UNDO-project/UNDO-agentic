"""Main entry point for the counter-surveillance analysis pipeline."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.orchestration.langchain_pipeline import create_pipeline
from src.config.pipeline_config import AnalysisScenario
from src.config.logger import logger

console = Console()


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Agentic Counter-Surveillance Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py Berlin
  python main.py Athens --country GR --scenario full
  python main.py Hamburg --skip-analyze
  python main.py --data-path data.json --skip-scrape Berlin
  python main.py Berlin --enable-routing --start-lat 52.52 --start-lon 13.40 \\
                        --end-lat 52.50 --end-lon 13.42
        """,
    )

    # Positional arguments
    parser.add_argument("city", help="City name to analyze")

    # Optional arguments
    parser.add_argument(
        "--country",
        help="Country code (e.g., DE, GR, US)",
        default=None,
    )
    parser.add_argument(
        "--scenario",
        default=AnalysisScenario.BASIC.value,
        help="Analysis scenario preset: basic | full (default: basic)",
    )
    parser.add_argument(
        "--heatmap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the preset for heatmap generation (--heatmap / --no-heatmap)",
    )
    parser.add_argument(
        "--hotspots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the preset for hotspot generation. Toggles both the "
            "DBSCAN clustering output and the matplotlib plot together."
        ),
    )
    parser.add_argument(
        "--charts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the preset for chart generation. Toggles the privacy "
            "pie, zone-sensitivity bar, sensitivity-reasons bar, the "
            "operator/manufacturer distribution bars, and the install "
            "timeline together."
        ),
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the preset for the LLM-written markdown city report "
            "(--report / --no-report)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results",
        default=None,
    )
    parser.add_argument(
        "--data-path",
        help="Path to existing data file (skips scraping)",
        default=None,
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip data scraping step",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analysis step",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging", default=True
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help=(
            "Bypass the scrape cache and refetch from Overpass. "
            "Use to capture cameras newly tagged in OSM before the TTL expires."
        ),
    )

    # Routing arguments
    parser.add_argument(
        "--enable-routing",
        action="store_true",
        help="Enable low-surveillance route computation",
    )
    parser.add_argument(
        "--start-lat",
        type=float,
        help="Starting latitude for route",
        default=None,
    )
    parser.add_argument(
        "--start-lon",
        type=float,
        help="Starting longitude for route",
        default=None,
    )
    parser.add_argument(
        "--end-lat",
        type=float,
        help="Ending latitude for route",
        default=None,
    )
    parser.add_argument(
        "--end-lon",
        type=float,
        help="Ending longitude for route",
        default=None,
    )

    args = parser.parse_args()

    # Migration guard for the dropped scenarios. Raise a clear error
    # pointing at the new toggle flags rather than letting a downstream
    # ValueError surface deep in the pipeline.
    removed = {"quick", "report", "mapping"}
    if args.scenario in removed:
        parser.error(
            f"--scenario {args.scenario!r} was removed. Use "
            f"--scenario basic or --scenario full and override individual "
            f"outputs with --heatmap/--no-heatmap, --hotspots/--no-hotspots, "
            f"--charts/--no-charts."
        )
    if args.scenario not in {s.value for s in AnalysisScenario}:
        parser.error(
            f"--scenario {args.scenario!r} is not recognised. "
            f"Valid choices: basic, full."
        )

    return args


def display_results(results: dict):
    """
    Display pipeline results in a formatted table.

    :param results: Pipeline execution results
    """
    # Status panel
    status = results["status"]
    status_color = {
        "completed": "green",
        "partial": "yellow",
        "failed": "red",
    }.get(status, "white")

    console.print(
        Panel(
            f"[bold {status_color}]{status.upper()}[/bold {status_color}]",
            title="Pipeline Status",
            border_style=status_color,
        )
    )

    # Summary table
    table = Table(title="Execution Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("City", results.get("city", "N/A"))
    table.add_row("Country", results.get("country", "N/A"))
    table.add_row("Scenario", results.get("scenario", "N/A"))
    table.add_row("Duration", f"{results.get('duration_seconds', 0):.2f}s")

    # Scraping results
    if "scrape" in results:
        scrape = results["scrape"]
        if scrape.get("skipped"):
            table.add_row("Scraping", "[yellow]Skipped[/yellow]")
        elif scrape.get("success"):
            elements = scrape.get("elements_count", 0)
            cached = " (cached)" if scrape.get("from_cache") else ""
            table.add_row("Elements Scraped", f"[green]{elements}{cached}[/green]")
        else:
            table.add_row(
                "Scraping", f"[red]Failed: {scrape.get('error', 'Unknown')}[/red]"
            )

    # Analysis results
    if "analyze" in results:
        analyze = results["analyze"]
        if analyze.get("skipped"):
            table.add_row("Analysis", "[yellow]Skipped[/yellow]")
        elif analyze.get("success"):
            elements = analyze.get("element_count", 0)
            table.add_row("Elements Analyzed", f"[green]{elements}[/green]")
        else:
            table.add_row(
                "Analysis", f"[red]Failed: {analyze.get('error', 'Unknown')}[/red]"
            )

    # Routing results
    if "routing" in results:
        routing = results["routing"]
        if routing.get("skipped"):
            table.add_row("Routing", "[yellow]Skipped[/yellow]")
        elif routing.get("success"):
            length_m = routing.get("length_m", 0)
            exposure = routing.get("exposure_score", 0)
            cached = " (cached)" if routing.get("from_cache") else ""
            table.add_row(
                "Route Length",
                f"[green]{length_m:.1f}m{cached}[/green]",
            )
            table.add_row("Route Exposure", f"[green]{exposure:.2f} cameras/km[/green]")
        else:
            table.add_row(
                "Routing", f"[red]Failed: {routing.get('error', 'Unknown')}[/red]"
            )

    console.print(table)

    # Output files
    has_files = False
    files_table = Table(title="Generated Files", show_header=False)
    files_table.add_column("Type", style="cyan")
    files_table.add_column("Path", style="green")

    if "analyze" in results and results["analyze"].get("success"):
        analyze = results["analyze"]
        has_files = True

        if analyze.get("output_path"):
            files_table.add_row("Enriched Data", str(analyze["output_path"]))
        if analyze.get("geojson_path"):
            files_table.add_row("GeoJSON", str(analyze["geojson_path"]))
        if analyze.get("heatmap_path"):
            files_table.add_row("Heatmap", str(analyze["heatmap_path"]))
        if analyze.get("hotspots_path"):
            files_table.add_row("Hotspots Data", str(analyze["hotspots_path"]))
        if analyze.get("plot_hotspots"):
            files_table.add_row("Hotspots Plot", str(analyze["plot_hotspots"]))
        if analyze.get("chart_path"):
            files_table.add_row("Statistics Chart", str(analyze["chart_path"]))

    if "routing" in results and results["routing"].get("success"):
        routing = results["routing"]
        has_files = True

        if routing.get("route_geojson_path"):
            files_table.add_row("Route GeoJSON", str(routing["route_geojson_path"]))
        if routing.get("route_map_path"):
            files_table.add_row("Route Map", str(routing["route_map_path"]))

    if has_files:
        console.print(files_table)

    # Errors
    if "errors" in results and results["errors"]:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"  - {error}")


def main():
    """
    Main entry point for the surveillance pipeline.
    """
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.enable("src")
    else:
        logger.disable("src")

    # Display header
    console.print(
        Panel.fit(
            renderable="[bold cyan]Surveillance Infrastructure Analysis[/bold cyan]\n"
            "------------------------Pipeline for data analysis------------------------",
            border_style="cyan",
        )
    )

    # Create pipeline configuration
    scenario = AnalysisScenario(args.scenario)
    config_kwargs = {}

    if args.output_dir:
        config_kwargs["output_dir"] = args.output_dir
    if args.skip_scrape:
        config_kwargs["scrape_enabled"] = False
    if args.skip_analyze:
        config_kwargs["analyze_enabled"] = False
    if args.force_refresh:
        config_kwargs["force_refresh"] = True

    # Toggle overrides layered on top of the preset. Each flag covers a
    # natural visualisation group; the underlying ``PipelineConfig`` fields
    # are flipped together so the CLI surface stays small.
    if args.heatmap is not None:
        config_kwargs["generate_heatmap"] = args.heatmap
    if args.hotspots is not None:
        config_kwargs["generate_hotspots"] = args.hotspots
        config_kwargs["plot_hotspots"] = args.hotspots
    if args.charts is not None:
        config_kwargs["generate_chart"] = args.charts
        config_kwargs["plot_zone_sensitivity"] = args.charts
        config_kwargs["plot_sensitivity_reasons"] = args.charts
        config_kwargs["plot_operator_distribution"] = args.charts
        config_kwargs["plot_manufacturer_distribution"] = args.charts
        config_kwargs["plot_install_timeline"] = args.charts
    if args.report is not None:
        config_kwargs["generate_report"] = args.report

    # Routing configuration
    if args.enable_routing:
        config_kwargs["routing_enabled"] = True
        config_kwargs["start_lat"] = args.start_lat
        config_kwargs["start_lon"] = args.start_lon
        config_kwargs["end_lat"] = args.end_lat
        config_kwargs["end_lon"] = args.end_lon

    # Create and run pipeline
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Initializing pipeline...", total=None)

            pipeline = create_pipeline(scenario=scenario, **config_kwargs)
            progress.update(
                task, description=f"[cyan]Running pipeline for {args.city}..."
            )

            # Prepare kwargs
            run_kwargs = {}
            if args.country:
                run_kwargs["country"] = args.country
            if args.output_dir:
                run_kwargs["output_dir"] = args.output_dir
            if args.data_path:
                run_kwargs["data_path"] = args.data_path

            results = pipeline.run(args.city, **run_kwargs)
            progress.update(task, description="[green]Pipeline completed")

        # Display results
        console.print()  # Blank line
        display_results(results)

        # Exit with appropriate status
        if results.get("success"):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
