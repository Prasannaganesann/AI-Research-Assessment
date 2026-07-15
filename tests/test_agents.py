"""
tests/test_agents.py — Unit Tests for Logistics Agents
=====================================================

This module tests each single-responsibility agent node in isolation:
- PlannerAgent
- ResearchAgent
- DecisionAgent
- ExecutionAgent

All LLM API calls are mocked using ``unittest.mock.patch`` to intercept
``BaseAgent._build_llm`` and return a mock LLM client.

Tests Cover:
- PlannerAgent: translates telemetry into a plan and constraints.
- ResearchAgent: runs the tool loop over RESEARCH_TOOLS, synthesises CarrierOptions.
- DecisionAgent: runs the validation tool call, selects a route, applies confidence scores.
- ExecutionAgent: simulates bookings, dispatches notifications/reports via tools.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.decision_agent import DecisionAgent
from agents.execution_agent import ExecutionAgent
from graphs.state import ShipmentAlert, CarrierOption, SelectedRoute, DEFAULT_SCENARIO
from graphs.logistics_graph import create_initial_state


# ---------------------------------------------------------------------------
# Test PlannerAgent
# ---------------------------------------------------------------------------


def test_planner_agent_success():
    """Verify PlannerAgent generates plan and constraints from ShipmentAlert."""
    mock_json = {
        "reasoning": "Standard delay. Plenty of time left before deadline. Will search standard modes.",
        "rerouting_plan": "Reroute via standard cargo carrier. Exclude OPF.",
        "constraints": {
            "max_cost_delta_usd": 30000.0,
            "original_cost_usd": 15000.0,
            "required_transit_modes": ["AIR", "SEA"],
            "excluded_carrier_ids": ["CARRIER-OPF"],
            "min_reliability_score": 0.85,
            "max_eta_hours": 120.0,
            "cargo_type": "STANDARD",
            "origin": "CNSHA",
            "destination": "USLAX",
            "weight_kg": 5000.0,
            "volume_cbm": 25.0,
        }
    }

    # Set up mock response
    mock_response = AIMessage(
        content=json.dumps(mock_json),
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    agent = PlannerAgent()
    initial_state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")

    with patch.object(agent, "_build_llm", return_value=mock_llm):
        updates = agent.run(initial_state)

    assert updates["execution_status"] == "researching"
    assert updates["rerouting_plan"] == "Reroute via standard cargo carrier. Exclude OPF."
    assert updates["rerouting_constraints"]["max_cost_delta_usd"] == 30000.0
    assert len(updates["trajectory"]) == 1
    assert updates["trajectory"][0].agent_name == "planner"
    assert updates["total_prompt_tokens"] == 100
    assert updates["total_completion_tokens"] == 50


# ---------------------------------------------------------------------------
# Test ResearchAgent
# ---------------------------------------------------------------------------


def test_research_agent_success():
    """Verify ResearchAgent calls tools and builds typed CarrierOptions."""
    # The research agent runs a tool loop. We mock the sequence of LLM decisions.
    # Turn 1: call list_available_carriers
    # Turn 2: call details tools for the carrier found (CARRIER-DHL)
    # Turn 3: output final JSON summary
    mock_tc1 = MagicMock()
    mock_tc1.name = "list_available_carriers"
    mock_tc1.args = {
        "origin": "CNSHA", "destination": "USLAX",
        "cargo_type": "STANDARD", "transit_mode": "AIR",
        "weight_kg": 5000.0, "volume_cbm": 25.0
    }
    mock_tc1.id = "call_1"

    mock_response_1 = AIMessage(
        content="",
        tool_calls=[{"name": "list_available_carriers", "args": mock_tc1.args, "id": "call_1"}],
        usage_metadata={"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
    )

    mock_tc2 = MagicMock()
    mock_tc2.name = "lookup_carrier_capacity"
    mock_tc2.args = {"carrier_id": "CARRIER-DHL", "origin": "CNSHA", "destination": "USLAX", "transit_mode": "AIR", "weight_kg": 5000.0, "volume_cbm": 25.0}
    mock_tc2.id = "call_2"

    mock_response_2 = AIMessage(
        content="",
        tool_calls=[
            {"name": "lookup_carrier_capacity", "args": mock_tc2.args, "id": "call_2"},
            {"name": "calculate_route_eta", "args": {"carrier_id": "CARRIER-DHL", "origin": "CNSHA", "destination": "USLAX", "transit_mode": "AIR"}, "id": "call_3"},
            {"name": "estimate_reroute_cost", "args": {"carrier_id": "CARRIER-DHL", "origin": "CNSHA", "destination": "USLAX", "transit_mode": "AIR", "weight_kg": 5000.0, "volume_cbm": 25.0, "cargo_type": "STANDARD"}, "id": "call_4"},
        ],
        usage_metadata={"input_tokens": 300, "output_tokens": 80, "total_tokens": 380},
    )

    mock_json_summary = {
        "reasoning": "Discovered DHL Express as viable AIR carrier.",
        "research_summary": "Queried available carriers. Found 1 viable option (DHL AIR).",
        "viable_carrier_ids": ["CARRIER-DHL"],
        "recommended_mode": "AIR",
        "total_carriers_queried": 1,
        "total_viable": 1,
    }

    mock_response_3 = AIMessage(
        content=json.dumps(mock_json_summary),
        usage_metadata={"input_tokens": 400, "output_tokens": 60, "total_tokens": 460},
    )

    # Set up mock LLM to return this sequence of responses
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [mock_response_1, mock_response_2, mock_response_3]

    agent = ResearchAgent()
    initial_state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")
    initial_state["rerouting_constraints"] = {
        "origin": "CNSHA", "destination": "USLAX", "cargo_type": "STANDARD",
        "weight_kg": 5000.0, "volume_cbm": 25.0, "min_reliability_score": 0.85,
        "max_eta_hours": 120.0, "original_cost_usd": 15000.0,
        "required_transit_modes": ["AIR"]
    }

    with patch.object(agent, "_build_llm", return_value=mock_llm):
        updates = agent.run(initial_state)

    assert updates["execution_status"] == "deciding"
    assert len(updates["carrier_options"]) == 3
    assert sum(1 for o in updates["carrier_options"] if o.is_viable) == 1
    assert updates["carrier_options"][0].carrier_id == "CARRIER-DHL"
    assert updates["carrier_options"][0].is_viable is True
    # Verify metadata accumulation
    assert updates["total_tool_calls"] == 4
    assert updates["successful_tool_calls"] == 4
    assert updates["total_prompt_tokens"] == 900
    assert updates["total_completion_tokens"] == 180


# ---------------------------------------------------------------------------
# Test DecisionAgent
# ---------------------------------------------------------------------------


def test_decision_agent_success():
    """Verify DecisionAgent validates top options and selects the best route."""
    # Decision agent tool loop calls validate_route_constraints once, then outputs selection
    mock_response_1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "validate_route_constraints",
            "args": {
                "carrier_id": "CARRIER-DHL",
                "route_id": "CNSHA-USLAX-DHL-AIR-01",
                "cargo_type": "STANDARD",
                "weight_kg": 5000.0,
                "volume_cbm": 25.0,
                "deadline_utc": "2030-01-01T12:00:00+00:00",
                "eta_hours": 35.0,
                "max_cost_delta_usd": 30000.0,
                "estimated_cost_usd": 82000.0,
                "original_cost_usd": 15000.0,
            },
            "id": "val_1"
        }],
        usage_metadata={"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
    )

    mock_decision = {
        "reasoning": "Verified DHL Express satisfies all physical and business constraints.",
        "selected_carrier_id": "CARRIER-DHL",
        "selected_route_id": "CNSHA-USLAX-DHL-AIR-01",
        "decision_rationale": "DHL Express AIR is selected due to 96% reliability and 35h ETA.",
        "confidence_score": 0.95,
        "requires_human_approval": False,
        "alternatives_considered": 1,
    }

    mock_response_2 = AIMessage(
        content=json.dumps(mock_decision),
        usage_metadata={"input_tokens": 250, "output_tokens": 50, "total_tokens": 300},
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [mock_response_1, mock_response_2]

    # Prepopulate state with carrier options from research
    opt = CarrierOption(
        carrier_id="CARRIER-DHL", carrier_name="DHL Express", route_id="CNSHA-USLAX-DHL-AIR-01",
        origin="CNSHA", destination="USLAX", transit_mode="AIR",
        eta_hours=35.0, eta_delta_hours=-50.0, cost_usd=82000.0, cost_delta_usd=67000.0,
        capacity_available=True, reliability_score=0.96, cargo_type_supported=True,
        constraints_satisfied=True
    )

    agent = DecisionAgent()
    initial_state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")
    initial_state["carrier_options"] = [opt]
    initial_state["rerouting_constraints"] = {"original_cost_usd": 15000.0}

    with patch.object(agent, "_build_llm", return_value=mock_llm):
        updates = agent.run(initial_state)

    assert updates["execution_status"] == "executing"
    assert updates["should_escalate"] is False
    assert updates["selected_route"] is not None
    assert updates["selected_route"].carrier_id == "CARRIER-DHL"
    assert updates["selected_route"].confidence_score == 0.95
    assert updates["total_tool_calls"] == 1


# ---------------------------------------------------------------------------
# Test ExecutionAgent
# ---------------------------------------------------------------------------


def test_execution_agent_success():
    """Verify ExecutionAgent calls notification and report tools, and finishes."""
    mock_response_1 = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "send_reroute_notification",
                "args": {
                    "shipment_id": "SHP-001", "customer_id": "CUST-001",
                    "original_carrier": "Grand Star", "new_carrier": "DHL Express",
                    "new_route_id": "r-new", "new_eta_hours": 35.0, "new_cost_usd": 82000.0,
                    "confirmation_number": "BK-DHL-123", "delay_reason": "WEATHER", "priority": "HIGH"
                },
                "id": "notif_1"
            },
            {
                "name": "generate_execution_report",
                "args": {
                    "shipment_id": "SHP-001", "run_id": "run-1", "model_name": "gpt-4o",
                    "delay_reason": "WEATHER", "original_carrier": "Grand Star", "original_route_id": "r-orig",
                    "new_carrier": "DHL Express", "new_route_id": "r-new", "new_transit_mode": "AIR",
                    "new_eta_hours": 35.0, "new_cost_usd": 82000.0, "original_cost_usd": 15000.0,
                    "confirmation_number": "BK-DHL-123", "execution_status": "completed",
                    "total_steps": 4, "total_tool_calls": 5, "successful_tool_calls": 5,
                    "total_cost_usd_llm": 0.005, "error_count": 0
                },
                "id": "rpt_1"
            }
        ],
        usage_metadata={"input_tokens": 300, "output_tokens": 90, "total_tokens": 390},
    )

    mock_summary = {
        "reasoning": "Reroute executed, notification sent, report generated.",
        "notification_sent": True,
        "report_generated": True,
        "final_status": "completed",
        "summary": "Shipment SHP-001 successfully rerouted via DHL Express."
    }

    mock_response_2 = AIMessage(
        content=json.dumps(mock_summary),
        usage_metadata={"input_tokens": 400, "output_tokens": 40, "total_tokens": 440},
    )

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = [mock_response_1, mock_response_2]

    # Prepopulate selected route
    route = SelectedRoute(
        carrier_id="CARRIER-DHL", carrier="DHL Express", route_id="r-new",
        transit_mode="AIR", estimated_cost_usd=82000.0, cost_delta_usd=67000.0,
        eta_hours=35.0, eta_delta_hours=-50.0, reliability_score=0.96,
        decision_rationale="DHL meets all constraints.", alternatives_considered=1,
        confidence_score=0.95, requires_human_approval=False
    )

    agent = ExecutionAgent()
    initial_state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")
    initial_state["selected_route"] = route

    with patch.object(agent, "_build_llm", return_value=mock_llm):
        updates = agent.run(initial_state)

    assert updates["execution_status"] == "completed"
    assert updates["execution_result"] is not None
    assert updates["execution_result"].success is True
    assert updates["execution_result"].notification_sent is True
    assert updates["total_tool_calls"] == 2
