"""
evaluation/model_benchmarker.py — Benchmark Runner
===================================================

Orchestrates multi-scenario runs for both closed-source (GPT-4o) and
open-source (Qwen3) models, collecting trajectory evaluation logs.

Scenarios Evaluated
-------------------
1.  **scenario_standard_success**: Standard container delay on Shanghai-LA lane.
    Requires selecting a reliable sea carrier within normal cost/time deltas.
2.  **scenario_strict_deadline**: Perishable cargo with tight deadline.
    Forces air transport and prioritises speed over cost.
3.  **scenario_carrier_failure**: Air cargo where the top option (FedEx) fails
    with a capacity lookup error.  Tests self-healing retry logic.
4.  **scenario_hazmat_restriction**: Hazmat cargo on Shanghai-Hamburg lane.
    Forces sea/multimodal transport because air cargo is restricted.
5.  **scenario_unresolvable_escalation**: Disruption that cannot be resolved
    within the deadline.  Forces a clean escalation terminal state.

Execution
---------
For each scenario and model:
1.  Sets up scenario-specific inputs and environment variables (for failure injection).
2.  Builds the LangGraph StateGraph.
3.  Invokes the graph end-to-end.
4.  Grades the run's final state using ``TrajectoryEvaluator``.
5.  Saves the evaluation reports as structured JSON datasets.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import get_settings
from graphs.logistics_graph import build_logistics_graph, create_initial_state
from evaluation.trajectory_evaluator import TrajectoryEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Benchmark Scenario Definitions
# ---------------------------------------------------------------------------


def get_benchmark_scenarios() -> dict[str, dict[str, Any]]:
    """
    Generate the 5 test scenarios with UTC-relative deadlines.
    """
    now = datetime.now(timezone.utc)

    return {
        "scenario_1_standard_success": {
            "shipment_id": "SHP-SCEN1-001",
            "origin": "CNSHA",
            "destination": "USLAX",
            "current_carrier": "Ocean Pacific Freight",
            "current_route_id": "CNSHA-USLAX-OPF-SEA-01",
            "delay_reason": "ROUTE_DISRUPTION",
            "delay_hours": 72.0,
            "cargo_type": "STANDARD",
            "cargo_value_usd": 150_000.0,
            "weight_kg": 12_000.0,
            "volume_cbm": 65.0,
            "deadline_utc": (now + timedelta(days=25)).isoformat(),  # 600h
            "priority": "HIGH",
            "customer_id": "CUST-GLOBEX-002",
            "alert_timestamp_utc": now.isoformat(),
        },
        "scenario_2_strict_deadline": {
            "shipment_id": "SHP-SCEN2-002",
            "origin": "CNSHA",
            "destination": "USNYC",
            "current_carrier": "Pacific Orient Lines",
            "current_route_id": "CNSHA-USNYC-POL-SEA-01",
            "delay_reason": "MECHANICAL_FAILURE",
            "delay_hours": 120.0,
            "cargo_type": "PERISHABLE",
            "cargo_value_usd": 400_000.0,
            "weight_kg": 4_500.0,
            "volume_cbm": 28.0,
            "deadline_utc": (now + timedelta(hours=48)).isoformat(),  # 48h
            "priority": "CRITICAL",
            "customer_id": "CUST-ACME-001",
            "alert_timestamp_utc": now.isoformat(),
        },
        "scenario_3_carrier_failure": {
            # This scenario is run with failure simulation active
            "shipment_id": "SHP-SCEN3-003",
            "origin": "CNSHA",
            "destination": "USLAX",
            "current_carrier": "Grand Star Logistics",
            "current_route_id": "CNSHA-USLAX-GSL-AIR-01",
            "delay_reason": "WEATHER",
            "delay_hours": 48.0,
            "cargo_type": "HIGH_VALUE",
            "cargo_value_usd": 1_200_000.0,
            "weight_kg": 5_200.0,
            "volume_cbm": 34.0,
            "deadline_utc": (now + timedelta(hours=60)).isoformat(),  # 60h
            "priority": "CRITICAL",
            "customer_id": "CUST-ACME-001",
            "alert_timestamp_utc": now.isoformat(),
        },
        "scenario_4_hazmat_restriction": {
            "shipment_id": "SHP-SCEN4-004",
            "origin": "CNSHA",
            "destination": "DEHAM",
            "current_carrier": "CMA CGM Group",
            "current_route_id": "CNSHA-DEHAM-CMA-SEA-01",
            "delay_reason": "PORT_CLOSURE",
            "delay_hours": 96.0,
            "cargo_type": "HAZMAT",
            "cargo_value_usd": 500_000.0,
            "weight_kg": 18_000.0,
            "volume_cbm": 110.0,
            "deadline_utc": (now + timedelta(days=10)).isoformat(),  # 240h
            "priority": "HIGH",
            "customer_id": "CUST-INITECH-003",
            "alert_timestamp_utc": now.isoformat(),
        },
        "scenario_5_unresolvable_escalation": {
            "shipment_id": "SHP-SCEN5-005",
            "origin": "CNSHA",
            "destination": "DEHAM",
            "current_carrier": "Ocean Pacific Freight",
            "current_route_id": "CNSHA-DEHAM-OPF-SEA-01",
            "delay_reason": "CUSTOMS_HOLD",
            "delay_hours": 240.0,
            "cargo_type": "STANDARD",
            "cargo_value_usd": 80_000.0,
            "weight_kg": 4_000.0,
            "volume_cbm": 25.0,
            "deadline_utc": (now + timedelta(hours=10)).isoformat(),  # 10h (physically impossible)
            "priority": "CRITICAL",
            "customer_id": "CUST-GLOBEX-002",
            "alert_timestamp_utc": now.isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Benchmarker Engine
# ---------------------------------------------------------------------------


class ModelBenchmarker:
    """
    Orchestrates the execution and evaluation of test scenarios across
    multiple models.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.settings = get_settings()
        self.evaluator = TrajectoryEvaluator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_benchmark_for_model(
        self,
        model_name: str,
        scenarios: dict[str, dict[str, Any]],
        dry_run: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Run all scenarios for a single model and return the graded reports.

        If dry_run=True, it simulates LLM execution to allow testing the
        benchmarking pipeline itself without making expensive API calls.
        """
        results: dict[str, dict[str, Any]] = {}
        graph = build_logistics_graph(model_name=model_name, settings=self.settings)

        for name, alert_dict in scenarios.items():
            logger.info(
                "running_benchmark_scenario",
                model=model_name,
                scenario=name,
                shipment_id=alert_dict["shipment_id"],
            )

            # Failure simulation settings for scenario 3
            original_simulate = os.getenv("SIMULATE_FAILURES")
            original_carriers = os.getenv("FAILURE_CARRIER_IDS")

            if "carrier_failure" in name:
                os.environ["SIMULATE_FAILURES"] = "true"
                os.environ["FAILURE_CARRIER_IDS"] = "CARRIER-FX"
                # Reset simulation count registry
                from tools._simulation import FailureSimulator
                FailureSimulator().reset()
            else:
                os.environ["SIMULATE_FAILURES"] = "false"

            try:
                if dry_run:
                    # Simulate a completed state dict for testing the pipeline
                    final_state = self._mock_completed_state(alert_dict, model_name, name)
                else:
                    initial_state = create_initial_state(alert_dict, model_name)
                    final_state = graph.invoke(initial_state)

                # Score the run
                report = self.evaluator.evaluate_run(final_state)
                results[name] = report

            except Exception as exc:
                logger.exception(
                    "scenario_execution_failed",
                    model=model_name,
                    scenario=name,
                    error=str(exc),
                )
                results[name] = {
                    "model_name": model_name,
                    "execution_status": "failed",
                    "composite_score": 0.0,
                    "error": str(exc),
                }
            finally:
                # Restore environment variables
                if original_simulate is not None:
                    os.environ["SIMULATE_FAILURES"] = original_simulate
                else:
                    os.environ.pop("SIMULATE_FAILURES", None)

                if original_carriers is not None:
                    os.environ["FAILURE_CARRIER_IDS"] = original_carriers
                else:
                    os.environ.pop("FAILURE_CARRIER_IDS", None)

        return results

    def _mock_completed_state(
        self,
        alert_dict: dict[str, Any],
        model_name: str,
        scenario_name: str,
    ) -> dict[str, Any]:
        """
        Generate a mock completed GraphState dict for dry-run testing.
        """
        from datetime import datetime, timezone
        from graphs.state import ShipmentAlert, CarrierOption, SelectedRoute, ExecutionResult, TrajectorySnapshot, ToolCallRecord

        alert = ShipmentAlert(**alert_dict)

        # Standard successful mock
        status = "completed"
        error_history = []
        should_escalate = False

        if "unresolvable" in scenario_name:
            status = "escalated"
        elif "carrier_failure" in scenario_name:
            error_history = ["Carrier 'CARRIER-FX' API returned 503 Service Unavailable"]

        # Tool calls list
        tcs = [
            ToolCallRecord(
                tool_name="list_available_carriers",
                success=True,
                latency_ms=120.0,
                arguments={"origin": alert.origin, "destination": alert.destination},
                raw_result={"carriers": []},
            )
        ]

        snapshots = [
            TrajectorySnapshot(
                run_id="mock-run-id",
                model_name=model_name,
                agent_name="planner",
                step_index=0,
                reasoning_trace="Analyzing shipment disruption. Priority is high, deadline allows standard carrier. Excluded current carrier.",
                latency_ms=1200.0,
                prompt_tokens=450,
                completion_tokens=150,
            ),
            TrajectorySnapshot(
                run_id="mock-run-id",
                model_name=model_name,
                agent_name="research",
                step_index=1,
                reasoning_trace="Discovered alternative carrier options on Shanghai to LA lane. Identified DHL and UPS. FedEx failed due to capacity lookup error.",
                latency_ms=2500.0,
                tool_calls=tcs,
                prompt_tokens=650,
                completion_tokens=220,
                error_occurred=len(error_history) > 0,
                error_message=error_history[0] if error_history else None,
            ),
            TrajectorySnapshot(
                run_id="mock-run-id",
                model_name=model_name,
                agent_name="decision",
                step_index=2,
                reasoning_trace="Selecting DHL Express as the optimal carrier because it satisfies reliability (96%) and arrives before the deadline. Rationale: cost delta of $67k is within budget.",
                latency_ms=1800.0,
                prompt_tokens=850,
                completion_tokens=180,
            ),
            TrajectorySnapshot(
                run_id="mock-run-id",
                model_name=model_name,
                agent_name="execution",
                step_index=3,
                reasoning_trace="Confirmed reroute booking with DHL Express. Confirmation Ref: BK-DHL-789. Dispatched notifications to customer supply-chain team via email and SMS.",
                latency_ms=1400.0,
                prompt_tokens=950,
                completion_tokens=200,
            ),
        ]

        opt = CarrierOption(
            carrier_id="CARRIER-DHL",
            carrier_name="DHL Express",
            route_id="CNSHA-USLAX-DHL-AIR-01",
            origin=alert.origin,
            destination=alert.destination,
            transit_mode="AIR",
            eta_hours=35.0,
            eta_delta_hours=-20.0,
            cost_usd=82000.0,
            cost_delta_usd=67000.0,
            capacity_available=True,
            reliability_score=0.96,
            cargo_type_supported=True,
            constraints_satisfied=True,
        )

        route = SelectedRoute(
            carrier_id="CARRIER-DHL",
            carrier="DHL Express",
            route_id="CNSHA-USLAX-DHL-AIR-01",
            transit_mode="AIR",
            estimated_cost_usd=82000.0,
            cost_delta_usd=67000.0,
            eta_hours=35.0,
            eta_delta_hours=-20.0,
            reliability_score=0.96,
            decision_rationale="DHL has the highest reliability (96%) and meets constraints.",
            alternatives_considered=3,
            confidence_score=0.95,
            requires_human_approval=False,
            selected_at_utc=datetime.now(timezone.utc),
        )

        result = ExecutionResult(
            success=True if status == "completed" else False,
            confirmation_number="BK-DHL-1234" if status == "completed" else None,
            carrier="DHL Express",
            route_id="CNSHA-USLAX-DHL-AIR-01",
            final_cost_usd=82000.0,
            final_eta_hours=35.0,
            notification_sent=True if status == "completed" else False,
            notification_recipients=["ops@globex.com"],
            executed_at_utc=datetime.now(timezone.utc),
        )

        return {
            "run_id": "mock-run-id",
            "model_name": model_name,
            "telemetry_alert": alert,
            "carrier_options": [opt],
            "selected_route": route if status == "completed" else None,
            "execution_result": result if status == "completed" else None,
            "execution_status": status,
            "retry_count": 1 if "carrier_failure" in scenario_name else 0,
            "should_escalate": should_escalate,
            "step_count": len(snapshots),
            "trajectory": snapshots,
            "error_history": error_history,
            "total_tool_calls": len(tcs),
            "successful_tool_calls": len(tcs),
            "total_cost_usd": 0.0076,
        }
