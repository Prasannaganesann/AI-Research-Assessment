"""
tests/test_graph.py — Unit Tests for LangGraph Orchestration
============================================================

Verifies the topology and conditional routing of the StateGraph:
- Node registration in CompiledStateGraph.
- Pure routing functions (_route_after_research, _route_after_decision, _route_after_execution).
- Behavior of auxiliary nodes (retry_increment, escalation_handler).
- Initialization of GraphState variables via create_initial_state.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from graphs.logistics_graph import (
    build_logistics_graph,
    create_initial_state,
    _route_after_research,
    _route_after_decision,
    _route_after_execution,
    _retry_increment_node,
    _escalation_handler_node,
)
from graphs.state import ShipmentAlert, CarrierOption, SelectedRoute, ExecutionResult, DEFAULT_SCENARIO


def test_graph_node_registration():
    """Verify that the logistics graph registers all necessary nodes."""
    graph = build_logistics_graph(model_name="gpt-4o")
    nodes = graph.get_graph().nodes.keys()
    assert "planner" in nodes
    assert "research" in nodes
    assert "decision" in nodes
    assert "execution" in nodes
    assert "retry_increment" in nodes
    assert "escalation_handler" in nodes


def test_create_initial_state():
    """Verify create_initial_state initializes all 14+ fields correctly."""
    state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")
    assert state["model_name"] == "gpt-4o"
    assert state["execution_status"] == "pending"
    assert state["retry_count"] == 0
    assert state["step_count"] == 0
    assert state["carrier_options"] == []
    assert state["trajectory"] == []
    assert isinstance(state["telemetry_alert"], ShipmentAlert)


def test_route_after_research_viable():
    """Verify routing moves to decide if viable carriers are found."""
    opt = CarrierOption(
        carrier_id="C1", carrier_name="C1", route_id="R1", origin="A", destination="B",
        transit_mode="AIR", eta_hours=10.0, eta_delta_hours=0.0, cost_usd=1000.0,
        cost_delta_usd=0.0, capacity_available=True, reliability_score=0.95,
        cargo_type_supported=True, constraints_satisfied=True
    )
    state = {"carrier_options": [opt], "retry_count": 0, "step_count": 2}
    assert _route_after_research(state) == "decide"


def test_route_after_research_retry():
    """Verify routing triggers retry if no viable carriers found and retries remain."""
    state = {"carrier_options": [], "retry_count": 0, "step_count": 2}
    assert _route_after_research(state) == "retry"


def test_route_after_research_fail():
    """Verify routing fails if no viable carriers found and retries are exhausted."""
    state = {"carrier_options": [], "retry_count": 3, "step_count": 2}
    assert _route_after_research(state) == "fail"


def test_route_after_research_max_step_guard():
    """Verify routing aborts and fails if graph steps exceed maximum ceiling."""
    state = {"carrier_options": [], "retry_count": 0, "step_count": 25}
    assert _route_after_research(state) == "fail"


def test_route_after_decision_execute():
    """Verify routing proceeds to execution if a route is selected without escalation."""
    route = SelectedRoute(
        carrier_id="C1", carrier="C1", route_id="R1", transit_mode="AIR",
        estimated_cost_usd=1000.0, cost_delta_usd=0.0, eta_hours=10.0, eta_delta_hours=0.0,
        reliability_score=0.95, decision_rationale="Good choice", alternatives_considered=1,
        confidence_score=0.9, requires_human_approval=False
    )
    state = {"selected_route": route, "should_escalate": False, "retry_count": 0, "step_count": 3}
    assert _route_after_decision(state) == "execute"


def test_route_after_decision_escalate():
    """Verify routing escalates if should_escalate flag is set."""
    route = SelectedRoute(
        carrier_id="C1", carrier="C1", route_id="R1", transit_mode="AIR",
        estimated_cost_usd=1000.0, cost_delta_usd=0.0, eta_hours=10.0, eta_delta_hours=0.0,
        reliability_score=0.95, decision_rationale="Unsure", alternatives_considered=1,
        confidence_score=0.5, requires_human_approval=True
    )
    state = {"selected_route": route, "should_escalate": True, "retry_count": 0, "step_count": 3}
    assert _route_after_decision(state) == "escalate"


def test_route_after_execution_complete():
    """Verify routing completes on successful booking execution."""
    res = ExecutionResult(
        success=True, confirmation_number="BK-123", carrier="DHL", route_id="R1",
        final_cost_usd=1000.0, final_eta_hours=10.0, retry_count=0
    )
    state = {"execution_result": res, "retry_count": 0}
    assert _route_after_execution(state) == "complete"


def test_route_after_execution_retry():
    """Verify routing retries if booking fails and retries remain."""
    res = ExecutionResult(
        success=False, carrier="DHL", route_id="R1", final_cost_usd=1000.0,
        final_eta_hours=10.0, retry_count=0, error_message="Carrier system down"
    )
    state = {"execution_result": res, "retry_count": 0}
    assert _route_after_execution(state) == "retry"


def test_retry_increment_node():
    """Verify retry_increment node bumps retry count and resets status."""
    state = {"retry_count": 1}
    updates = _retry_increment_node(state)
    assert updates["retry_count"] == 2
    assert updates["execution_status"] == "researching"


def test_escalation_handler_node():
    """Verify escalation_handler node updates status and error message."""
    state = {
        "telemetry_alert": ShipmentAlert(**DEFAULT_SCENARIO),
        "decision_summary": "Infeasible deadline."
    }
    updates = _escalation_handler_node(state)
    assert updates["execution_status"] == "escalated"
    assert "Escalated" in updates["error_message"]
