"""
main.py — Autonomous Logistics Agent Entry Point
=================================================
Orchestrates the full pipeline:

1. Load configuration from environment
2. Ingest a telemetry alert (simulated shipment delay)
3. Run the LangGraph workflow using the configured model
4. Capture trajectory snapshots at each agent node
5. (Optional) Run comparative benchmarking: GPT-4o vs Qwen3
6. Generate evaluation report with metrics and charts

Usage
-----
    # Single run with the primary model (set in .env)
    python main.py

    # Full comparative benchmark (both models, N scenarios)
    python main.py --benchmark

    # Run with a specific scenario file
    python main.py --scenario scenarios/port_closure.json

CLI Arguments
-------------
    --benchmark     Run full model comparison evaluation
    --scenario PATH Path to a custom telemetry alert JSON file
    --model MODEL   Override model from .env (e.g. gpt-4o, qwen/qwen3-8b)
    --verbose       Enable DEBUG-level logging
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.logger import configure_logging, get_logger
from graphs.logistics_graph import build_logistics_graph
from evaluation.model_benchmarker import ModelBenchmarker
from evaluation.report_generator import ReportGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="autonomous-logistics-agent",
        description="Autonomous shipment rerouting via multi-agent LangGraph workflow.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comparative benchmarking across both configured models.",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a custom telemetry alert JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Override the model from .env (e.g. gpt-4o).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level structured logging.",
    )
    return parser.parse_args()


def run_single(
    settings,
    logger,
    model_override: str | None = None,
    scenario_path: Path | None = None,
) -> None:
    """
    Execute a single end-to-end logistics rerouting run.

    Parameters
    ----------
    settings:
        Loaded AppSettings instance.
    logger:
        Configured structlog logger.
    model_override:
        If provided, overrides the closed-source model name from settings.
    scenario_path:
        Optional path to a JSON telemetry alert. Defaults to built-in scenario.
    """
    from graphs.logistics_graph import build_logistics_graph, DEFAULT_SCENARIO

    model_name = model_override or settings.closed_source_model
    logger.info("starting_single_run", model=model_name)

    graph = build_logistics_graph(model_name=model_name, settings=settings)

    if scenario_path:
        import json
        telemetry_alert = json.loads(scenario_path.read_text())
        logger.info("loaded_custom_scenario", path=str(scenario_path))
    else:
        telemetry_alert = DEFAULT_SCENARIO
        logger.info("using_default_scenario")

    result = graph.invoke({"telemetry_alert": telemetry_alert})

    logger.info(
        "run_complete",
        model=model_name,
        status=result.get("execution_status", "unknown"),
        route=result.get("selected_route", {}).get("carrier", "N/A"),
    )

    print("\n" + "=" * 60)
    print("  LOGISTICS REROUTING — RUN COMPLETE")
    print("=" * 60)
    print(f"  Model       : {model_name}")
    print(f"  Status      : {result.get('execution_status', 'unknown')}")
    print(f"  New Carrier : {result.get('selected_route', {}).get('carrier', 'N/A')}")
    print(f"  New Route   : {result.get('selected_route', {}).get('route_id', 'N/A')}")
    print(f"  ETA Delta   : {result.get('selected_route', {}).get('eta_delta_hours', 'N/A')}h")
    print("=" * 60 + "\n")


def run_benchmark(settings, logger) -> None:
    """
    Run the full comparative benchmarking pipeline.

    Executes N scenarios for each model (closed-source + open-source),
    captures trajectory snapshots, computes aggregate metrics, and
    writes a structured report with charts to the reports/ directory.

    Parameters
    ----------
    settings:
        Loaded AppSettings instance.
    logger:
        Configured structlog logger.
    """
    logger.info(
        "starting_benchmark",
        closed_source_model=settings.closed_source_model,
        open_source_model=settings.open_source_model,
        scenarios=settings.eval_scenarios,
    )

    benchmarker = ModelBenchmarker(settings=settings)
    results = benchmarker.run_all()

    reporter = ReportGenerator(results=results, settings=settings)
    reporter.generate()

    logger.info("benchmark_complete", output_dir=settings.eval_output_dir)
    print(f"\n✅ Benchmark complete. Reports written to: {settings.eval_output_dir}/\n")


def main() -> int:
    """
    Application entry point.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for failure.
    """
    args = parse_args()
    settings = get_settings()

    if args.verbose:
        settings.log_level = "DEBUG"

    configure_logging(level=settings.log_level, fmt=settings.log_format)
    logger = get_logger(__name__)

    logger.info(
        "application_start",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    try:
        if args.benchmark:
            run_benchmark(settings=settings, logger=logger)
        else:
            run_single(
                settings=settings,
                logger=logger,
                model_override=args.model,
                scenario_path=args.scenario,
            )
        return 0

    except KeyboardInterrupt:
        logger.warning("interrupted_by_user")
        return 130

    except Exception as exc:
        logger.exception("fatal_error", error=str(exc))
        print(f"\n❌ Fatal error: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
