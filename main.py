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

    # Full comparative benchmark (both models, 5 scenarios)
    python main.py --benchmark

    # Run with a specific scenario file
    python main.py --scenario scenarios/port_closure.json

    # Run with a specific model override
    python main.py --model qwen/qwen3-8b
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.logger import configure_logging, get_logger
from graphs import build_logistics_graph, create_initial_state, DEFAULT_SCENARIO
from evaluation import ModelBenchmarker, ReportGenerator, get_benchmark_scenarios


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
        help="Override the model from .env (e.g. gpt-4o, qwen/qwen3-8b).",
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
    """
    model_name = model_override or settings.closed_source_model
    logger.info("starting_single_run", model=model_name)

    graph = build_logistics_graph(model_name=model_name, settings=settings)

    if scenario_path:
        telemetry_alert = json.loads(scenario_path.read_text())
        logger.info("loaded_custom_scenario", path=str(scenario_path))
    else:
        telemetry_alert = DEFAULT_SCENARIO
        logger.info("using_default_scenario")

    # Correct initial state generation (prevents missing key crashes)
    initial_state = create_initial_state(telemetry_alert, model_name=model_name)
    result = graph.invoke(initial_state)

    logger.info(
        "run_complete",
        model=model_name,
        status=result.get("execution_status", "unknown"),
        route=result.get("selected_route", {}).carrier if result.get("selected_route") else "N/A",
    )

    print("\n" + "=" * 60)
    print("  LOGISTICS REROUTING — RUN COMPLETE")
    print("=" * 60)
    print(f"  Model       : {model_name}")
    print(f"  Status      : {result.get('execution_status', 'unknown')}")
    if result.get("selected_route"):
        route = result["selected_route"]
        print(f"  New Carrier : {route.carrier}")
        print(f"  New Route   : {route.route_id}")
        print(f"  Cost (USD)  : ${route.estimated_cost_usd:,.2f}")
        print(f"  ETA Delta   : {route.eta_delta_hours:+.1f}h")
    else:
        print("  New Route   : N/A (Escalated or Failed)")
    if result.get("execution_result"):
        res = result["execution_result"]
        print(f"  Confirmation: {res.confirmation_number or 'N/A'}")
        print(f"  Notified Ops: {res.notification_sent}")
    print("=" * 60 + "\n")


def run_benchmark(settings, logger) -> None:
    """
    Run the full comparative benchmarking pipeline.
    """
    closed_model = settings.closed_source_model
    open_model = settings.open_source_model
    scenarios = get_benchmark_scenarios()

    logger.info(
        "starting_benchmark",
        closed_source_model=closed_model,
        open_source_model=open_model,
        scenarios_count=len(scenarios),
    )

    benchmarker = ModelBenchmarker(output_dir="reports")

    # Run for both models
    print(f"--- Running Benchmarks for {closed_model} ---")
    closed_results = benchmarker.run_benchmark_for_model(closed_model, scenarios, dry_run=False)

    print(f"--- Running Benchmarks for {open_model} ---")
    open_results = benchmarker.run_benchmark_for_model(open_model, scenarios, dry_run=False)

    results = {
        closed_model: closed_results,
        open_model: open_results,
    }

    # Generate comparative report
    reporter = ReportGenerator(output_dir="reports")
    md_report = reporter.generate_markdown_report(results)

    logger.info("benchmark_complete", output_dir="reports")
    print("\n✅ Benchmark complete. Reports written to: reports/\n")


def main() -> int:
    """
    Application entry point.
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
