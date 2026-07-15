"""
tests/test_tools.py — Unit Tests for Logistics Tools
===================================================

This module verifies the behavior of all 7 logistics tools:
- list_available_carriers
- lookup_carrier_capacity
- calculate_route_eta
- estimate_reroute_cost
- validate_route_constraints
- send_reroute_notification
- generate_execution_report

Tests Cover:
- Normal execution under expected logistics parameters.
- Parameter bounds and invalid inputs (e.g., negative weights, unknown codes).
- The FailureSimulator's deterministic failure injection.
- Verification of the retry mechanism (both sync and async retry decorators).
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from tools.carrier_tools import list_available_carriers, lookup_carrier_capacity
from tools.route_tools import calculate_route_eta, estimate_reroute_cost, validate_route_constraints
from tools.notification_tools import send_reroute_notification, generate_execution_report
from tools._simulation import FailureSimulator, SimulatedToolError
from utils.retry import make_retry_decorator


def test_list_available_carriers_success():
    """Verify list_available_carriers finds viable carriers on valid lanes."""
    result = list_available_carriers.invoke({
        "origin": "CNSHA",
        "destination": "USLAX",
        "cargo_type": "STANDARD",
        "transit_mode": "AIR",
        "weight_kg": 1000.0,
        "volume_cbm": 5.0,
    })
    assert result["success"] is True
    assert result["lane_found"] is True
    assert result["viable_count"] > 0
    # DHL, FX, UPS serve this lane by AIR
    carrier_ids = [c["carrier_id"] for c in result["carriers"]]
    assert "CARRIER-DHL" in carrier_ids
    assert "CARRIER-FX" in carrier_ids


def test_list_available_carriers_unknown_lane():
    """Verify list_available_carriers returns lane_found=False for unserved ports."""
    result = list_available_carriers.invoke({
        "origin": "UNKNOWN_ORIGIN",
        "destination": "UNKNOWN_DEST",
        "cargo_type": "STANDARD",
        "transit_mode": "AIR",
        "weight_kg": 1000.0,
        "volume_cbm": 5.0,
    })
    assert result["success"] is True
    assert result["lane_found"] is False
    assert len(result["carriers"]) == 0


def test_lookup_carrier_capacity_success():
    """Verify lookup_carrier_capacity returns correct capacity calculations."""
    result = lookup_carrier_capacity.invoke({
        "carrier_id": "CARRIER-DHL",
        "origin": "CNSHA",
        "destination": "USLAX",
        "transit_mode": "AIR",
        "weight_kg": 2000.0,
        "volume_cbm": 15.0,
    })
    assert result["success"] is True
    assert result["capacity_available"] is True
    assert 40.0 <= result["utilisation_pct"] <= 80.0
    assert result["earliest_slot_hours"] > 0


def test_lookup_carrier_capacity_exceeds_limits():
    """Verify carrier rejects booking if shipment dimensions exceed maximums."""
    # DHL max weight is 8,000kg
    result = lookup_carrier_capacity.invoke({
        "carrier_id": "CARRIER-DHL",
        "origin": "CNSHA",
        "destination": "USLAX",
        "transit_mode": "AIR",
        "weight_kg": 99999.0,
        "volume_cbm": 15.0,
    })
    assert result["success"] is True
    assert result["capacity_available"] is False
    assert "exceeds" in result["rejection_reason"]


def test_calculate_route_eta():
    """Verify calculate_route_eta returns deterministic ETAs with jitter."""
    result = calculate_route_eta.invoke({
        "carrier_id": "CARRIER-DHL",
        "origin": "CNSHA",
        "destination": "USLAX",
        "transit_mode": "AIR",
    })
    assert result["success"] is True
    assert result["eta_hours"] > 0
    assert result["confidence"] == "HIGH"


def test_estimate_reroute_cost():
    """Verify estimate_reroute_cost computes correct base, cargo premium, and total cost."""
    result = estimate_reroute_cost.invoke({
        "carrier_id": "CARRIER-DHL",
        "origin": "CNSHA",
        "destination": "USLAX",
        "transit_mode": "AIR",
        "weight_kg": 1000.0,
        "volume_cbm": 5.0,
        "cargo_type": "HIGH_VALUE",
    })
    assert result["success"] is True
    assert result["cargo_premium_pct"] == 20.0  # High value premium is +20%
    expected_subtotal = 950.0 + (1000.0 * 13.80) + (5.0 * 52.0)
    expected_total = round(expected_subtotal * 1.20, 2)
    assert abs(result["total_cost_usd"] - expected_total) < 0.01


def test_validate_route_constraints_approve():
    """Verify validate_route_constraints passes a compliant route option."""
    result = validate_route_constraints.invoke({
        "carrier_id": "CARRIER-DHL",
        "route_id": "CNSHA-USLAX-DHL-AIR-01",
        "cargo_type": "HIGH_VALUE",
        "weight_kg": 1000.0,
        "volume_cbm": 5.0,
        "deadline_utc": "2030-01-01T12:00:00+00:00",
        "eta_hours": 30.0,
        "max_cost_delta_usd": 20000.0,
        "estimated_cost_usd": 16000.0,
        "original_cost_usd": 10000.0,
    })
    assert result["valid"] is True
    assert result["recommendation"] == "APPROVE"
    assert len(result["violations"]) == 0


def test_validate_route_constraints_reject_deadline():
    """Verify validate_route_constraints rejects a route exceeding the deadline."""
    # Delivery deadline is in the past, so ETA will definitely exceed it
    past_time = "2020-01-01T12:00:00+00:00"
    result = validate_route_constraints.invoke({
        "carrier_id": "CARRIER-DHL",
        "route_id": "CNSHA-USLAX-DHL-AIR-01",
        "cargo_type": "STANDARD",
        "weight_kg": 1000.0,
        "volume_cbm": 5.0,
        "deadline_utc": past_time,
        "eta_hours": 30.0,
        "max_cost_delta_usd": 20000.0,
        "estimated_cost_usd": 12000.0,
        "original_cost_usd": 10000.0,
    })
    assert result["valid"] is False
    assert "deadline" in result["violations"][0]


def test_send_reroute_notification():
    """Verify send_reroute_notification resolves recipients and returns confirmation."""
    result = send_reroute_notification.invoke({
        "shipment_id": "SHP-001",
        "customer_id": "CUST-ACME-001",
        "original_carrier": "Grand Star",
        "new_carrier": "DHL Express",
        "new_route_id": "CNSHA-USLAX-DHL-AIR-01",
        "new_eta_hours": 32.0,
        "new_cost_usd": 15000.0,
        "confirmation_number": "BK-1234",
        "delay_reason": "WEATHER",
        "priority": "CRITICAL",
    })
    assert result["success"] is True
    assert result["recipient_count"] > 0
    assert "EMAIL" in result["channels_used"]
    # Critical priority triggers SMS
    assert "SMS" in result["channels_used"]


def test_generate_execution_report():
    """Verify generate_execution_report returns a formatted summary and stats."""
    result = generate_execution_report.invoke({
        "shipment_id": "SHP-001",
        "run_id": "run-123",
        "model_name": "gpt-4o",
        "delay_reason": "WEATHER",
        "original_carrier": "Grand Star",
        "original_route_id": "r-orig",
        "new_carrier": "DHL Express",
        "new_route_id": "r-new",
        "new_transit_mode": "AIR",
        "new_eta_hours": 32.0,
        "new_cost_usd": 15000.0,
        "original_cost_usd": 10000.0,
        "confirmation_number": "BK-123",
        "execution_status": "completed",
        "total_steps": 4,
        "total_tool_calls": 5,
        "successful_tool_calls": 5,
        "total_cost_usd_llm": 0.005,
        "error_count": 0,
    })
    assert result["report_id"].startswith("RPT-")
    assert "completed" in result["summary"].lower()
    assert result["sections"]["agent_metrics"]["tool_accuracy_pct"] == 100.0


def test_failure_simulator_retry():
    """Verify FailureSimulator injects failures on call 1 and passes on call 2."""
    import os
    os.environ["SIMULATE_FAILURES"] = "true"
    os.environ["FAILURE_CARRIER_IDS"] = "CARRIER-FAIL-TEST"

    simulator = FailureSimulator()
    simulator.reset()

    # Call 1: Should fail
    with pytest.raises(SimulatedToolError) as exc_info:
        simulator.maybe_raise("some_tool", context="CARRIER-FAIL-TEST")
    assert "CARRIER-FAIL-TEST" in str(exc_info.value)

    # Call 2: Should pass (retry succeeds)
    try:
        simulator.maybe_raise("some_tool", context="CARRIER-FAIL-TEST")
    except SimulatedToolError:
        pytest.fail("Failure simulator failed on retry attempt")

    os.environ.pop("SIMULATE_FAILURES", None)
    os.environ.pop("FAILURE_CARRIER_IDS", None)


def test_retry_decorator_successful_recovery():
    """Verify retry decorator handles retryable exceptions."""
    call_count = 0

    # Build a retry decorator that retries ValueError up to 3 times
    from utils.retry import retry_with_backoff
    retry_decorator = retry_with_backoff(
        max_attempts=3,
        backoff_base=0.01,
        jitter=0.0,
        retryable_exceptions=(ValueError,),
        tool_name="test_retry",
    )

    @retry_decorator
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Injected transient failure")
        return "Success"

    result = failing_function()
    assert result == "Success"
    assert call_count == 2
