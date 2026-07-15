"""
tests/test_evaluation.py — Unit Tests for Trajectory Evaluation
===============================================================

Tests the trajectory evaluation metrics, score computations, and report generation:
- Reasoning Quality scoring checks (keyword search logic).
- Tool-Calling Accuracy calculations (success rate, sequence violations, grounding).
- Error Recovery Rate detection.
- Final Output Quality metrics (completed, escalated, failed).
- Composite Score weighting logic.
- ReportGenerator Markdown formatting and file writing.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
import pytest

from evaluation.trajectory_evaluator import TrajectoryEvaluator
from evaluation.report_generator import ReportGenerator
from graphs.state import GraphState, TrajectorySnapshot, ToolCallRecord, CarrierOption, SelectedRoute, ExecutionResult, ShipmentAlert, DEFAULT_SCENARIO


def test_score_reasoning_trace():
    """Verify evaluator keywords analysis for reasoning traces."""
    evaluator = TrajectoryEvaluator()

    # Good trace with many keywords
    good_planner_trace = "Analyzing shipment delay. The delivery deadline is tight. Priority is critical. Excluded carrier OPF. Checking cargo type and cost constraints."
    score_good = evaluator._score_reasoning_trace("planner", good_planner_trace)
    assert score_good > 0.60

    # Poor trace with no keywords
    poor_trace = "Rerouting the container."
    score_poor = evaluator._score_reasoning_trace("planner", poor_trace)
    assert score_poor < 0.30


def test_evaluate_tool_calling_perfect():
    """Verify tool calling metrics when all tool calls succeed in correct order."""
    evaluator = TrajectoryEvaluator()

    opt = CarrierOption(
        carrier_id="CARRIER-DHL", carrier_name="DHL", route_id="r1", origin="A", destination="B",
        transit_mode="AIR", eta_hours=35.0, eta_delta_hours=0.0, cost_usd=82000.0,
        cost_delta_usd=0.0, capacity_available=True, reliability_score=0.96,
        cargo_type_supported=True, constraints_satisfied=True
    )

    tcs = [
        ToolCallRecord(tool_name="list_available_carriers", success=True),
        ToolCallRecord(tool_name="validate_route_constraints", success=True, arguments={
            "carrier_id": "CARRIER-DHL", "estimated_cost_usd": 82000.0, "eta_hours": 35.0
        }),
        # Notification sent with valid booking reference
        ToolCallRecord(tool_name="send_reroute_notification", success=True, arguments={
            "confirmation_number": "BK-123"
        }),
    ]

    snapshot = TrajectorySnapshot(
        run_id="run-1", model_name="gpt-4o", agent_name="research", step_index=1,
        tool_calls=tcs
    )

    state = {"trajectory": [snapshot], "carrier_options": [opt]}
    results = evaluator._evaluate_tool_calling([snapshot], state)

    assert results["success_rate"] == 1.0
    assert len(results["sequence_violations"]) == 0
    assert len(results["grounding_violations"]) == 0
    assert results["score"] == 1.0


def test_evaluate_tool_calling_violations():
    """Verify tool calling metrics when notification is sent before route booking."""
    evaluator = TrajectoryEvaluator()

    tcs = [
        # Notification sent with no confirmation reference (empty)
        ToolCallRecord(tool_name="send_reroute_notification", success=True, arguments={
            "confirmation_number": ""
        }),
    ]
    snapshot = TrajectorySnapshot(
        run_id="run-1", model_name="gpt-4o", agent_name="execution", step_index=3,
        tool_calls=tcs
    )

    state = {"trajectory": [snapshot], "carrier_options": []}
    results = evaluator._evaluate_tool_calling([snapshot], state)

    assert results["success_rate"] == 1.0
    assert len(results["sequence_violations"]) == 1
    # Deductions reduce score below perfect 1.0
    assert results["score"] < 1.0


def test_evaluate_error_recovery_no_errors():
    """Verify error recovery rate is 100% when no errors occur."""
    evaluator = TrajectoryEvaluator()
    snapshot = TrajectorySnapshot(
        run_id="run-1", model_name="gpt-4o", agent_name="research", step_index=1,
        error_occurred=False
    )
    state = {"execution_status": "completed"}
    results = evaluator._evaluate_error_recovery([snapshot], state)
    assert results["score"] == 1.0
    assert results["total_errors"] == 0


def test_evaluate_error_recovery_healed():
    """Verify error recovery rate recognizes self-healing retry cycles."""
    evaluator = TrajectoryEvaluator()

    # Step 1: Research failed
    snap1 = TrajectorySnapshot(
        run_id="run-1", model_name="gpt-4o", agent_name="research", step_index=1,
        error_occurred=True
    )
    # Step 2: Research retried and succeeded
    snap2 = TrajectorySnapshot(
        run_id="run-1", model_name="gpt-4o", agent_name="research", step_index=2,
        error_occurred=False
    )

    state = {"execution_status": "completed"}
    results = evaluator._evaluate_error_recovery([snap1, snap2], state)
    assert results["score"] == 1.0  # Successfully recovered
    assert results["total_errors"] == 1
    assert results["recovered_errors"] == 1


def test_evaluate_output_quality_completed():
    """Verify completed run outputs get full quality marks."""
    evaluator = TrajectoryEvaluator()

    res = ExecutionResult(
        success=True, confirmation_number="BK-123", carrier="DHL", route_id="r1",
        final_cost_usd=82000.0, final_eta_hours=35.0, retry_count=0,
        notification_sent=True
    )
    state = {"execution_status": "completed", "execution_result": res}
    assert evaluator._evaluate_output_quality(state) == 1.0


def test_evaluate_output_quality_escalated():
    """Verify escalated runs get correct partial quality score."""
    evaluator = TrajectoryEvaluator()
    state = {"execution_status": "escalated", "decision_summary": "No viable options."}
    assert evaluator._evaluate_output_quality(state) == 0.8


def test_report_generator_markdown():
    """Verify ReportGenerator outputs formatted markdown tables."""
    gpt_raw = {
        "scenario_1_standard_success": {
            "model_name": "gpt-4o",
            "execution_status": "completed",
            "composite_score": 0.8693,
            "tool_calling_accuracy_score": 1.0,
            "reasoning_quality_score": 0.70,
            "error_recovery_score": 1.0,
            "output_quality_score": 1.0,
            "total_latency_ms": 6900.0,
            "total_cost_usd": 0.0076,
            "total_steps": 4,
        }
    }
    generator = ReportGenerator(output_dir="reports_test_suite")
    md_report = generator.generate_markdown_report({"gpt-4o": gpt_raw})

    assert "# LLM Benchmark" in md_report
    assert "## Executive Summary" in md_report
    assert "gpt-4o" in md_report

    # Cleanup generated files
    report_path = "reports_test_suite/comparison_report.md"
    if os.path.exists(report_path):
        os.remove(report_path)
    figures_dir = "reports_test_suite/figures"
    if os.path.exists(figures_dir):
        for fig in os.listdir(figures_dir):
            os.remove(os.path.join(figures_dir, fig))
        os.rmdir(figures_dir)
    os.rmdir("reports_test_suite")
