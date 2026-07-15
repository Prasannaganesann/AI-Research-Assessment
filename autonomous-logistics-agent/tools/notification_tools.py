"""
tools/notification_tools.py — Stakeholder Notification & Reporting Tools
=========================================================================

Provides two LangChain tools used by the Execution Agent to close the
autonomous rerouting loop with stakeholder communication:

    send_reroute_notification
        Dispatches structured notifications to all stakeholders
        (customer, operations team, carrier contacts) confirming that
        the shipment has been rerouted, with the new carrier, route,
        ETA, and cost details.

    generate_execution_report
        Produces a structured summary of the entire rerouting event:
        original alert, selected route, execution outcome, and
        trajectory metrics.  This is the final artefact of each graph
        run — used both as the business deliverable and as the input
        to the trajectory evaluator's output quality score.

Trajectory Evaluation Contribution
------------------------------------
- ``send_reroute_notification`` success/failure is recorded in
  ``ExecutionResult.notification_sent``.  The evaluator checks that
  the Execution Agent called this tool AFTER a successful reroute,
  not before — order of tool calls is part of tool-calling accuracy.
- ``generate_execution_report`` output is scored for completeness and
  accuracy (does it mention the correct carrier, cost, ETA?).

Simulated Behaviour
-------------------
No real notifications are sent.  The tool generates deterministic
notification IDs and logs the event.  In a production system, this
tool would call an email/SMS gateway or an internal messaging bus.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from langchain_core.tools import tool

from utils.logger import get_logger
from tools._simulation import FailureSimulator, SimulatedToolError

logger = get_logger(__name__)
_simulator = FailureSimulator()

# ---------------------------------------------------------------------------
# Notification channel registry (simulated)
# ---------------------------------------------------------------------------

_NOTIFICATION_CHANNELS: dict[str, dict[str, str]] = {
    "EMAIL": {
        "provider": "SendGrid (simulated)",
        "template": "logistics_reroute_v2",
        "max_recipients": "100",
    },
    "SMS": {
        "provider": "Twilio (simulated)",
        "template": "logistics_alert_short",
        "max_recipients": "10",
    },
    "WEBHOOK": {
        "provider": "Internal event bus (simulated)",
        "template": "reroute_event_v1",
        "max_recipients": "unlimited",
    },
}

# Notification recipient mapping by customer ID
_CUSTOMER_CONTACTS: dict[str, dict[str, list[str]]] = {
    "CUST-ACME-001": {
        "email": ["logistics@acme.example.com", "ops-manager@acme.example.com"],
        "sms": ["+1-555-0100"],
        "name": "ACME Corporation",
    },
    "CUST-GLOBEX-002": {
        "email": ["supply-chain@globex.example.com"],
        "sms": ["+1-555-0200"],
        "name": "Globex Industries",
    },
    "CUST-INITECH-003": {
        "email": ["freight@initech.example.com", "cfo@initech.example.com"],
        "sms": [],
        "name": "Initech Corp",
    },
}

# Internal operations team — always notified
_OPS_TEAM_EMAILS: list[str] = [
    "ops-center@logistics-ai.example.com",
    "on-call-engineer@logistics-ai.example.com",
]


# ---------------------------------------------------------------------------
# Internal timing helper
# ---------------------------------------------------------------------------


@contextmanager
def _time_tool(tool_name: str) -> Generator[dict[str, float], None, None]:
    """Measure tool execution latency in milliseconds."""
    start = time.perf_counter()
    timing: dict[str, float] = {}
    try:
        yield timing
    finally:
        timing["latency_ms"] = (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Tool 6: send_reroute_notification
# ---------------------------------------------------------------------------


@tool
def send_reroute_notification(
    shipment_id: str,
    customer_id: str,
    original_carrier: str,
    new_carrier: str,
    new_route_id: str,
    new_eta_hours: float,
    new_cost_usd: float,
    confirmation_number: str,
    delay_reason: str,
    priority: str,
) -> dict[str, Any]:
    """
    Send reroute confirmation notifications to all relevant stakeholders.

    Call this AFTER the reroute has been successfully submitted to the
    carrier (after execute_reroute returns success=True).  Notifications
    are dispatched to: the customer's registered contacts, the internal
    operations team, and optionally via SMS for CRITICAL priority shipments.

    Args:
        shipment_id: Shipment identifier (e.g. 'SHP-20240715-0042').
        customer_id: Customer identifier for contact lookup (e.g. 'CUST-ACME-001').
        original_carrier: Name of the original (disrupted) carrier.
        new_carrier: Name of the newly selected carrier.
        new_route_id: New route identifier.
        new_eta_hours: New estimated transit time in hours.
        new_cost_usd: Final reroute cost in USD.
        confirmation_number: Carrier booking confirmation number.
        delay_reason: Reason for the original disruption (e.g. 'PORT_CLOSURE').
        priority: Shipment priority level (e.g. 'CRITICAL', 'HIGH').

    Returns:
        Dict with keys:
            success (bool): True if all notifications were dispatched.
            notification_id (str): Unique tracking ID for this notification batch.
            recipients (list[str]): All addresses notified.
            channels_used (list[str]): Communication channels used (EMAIL, SMS, etc.).
            dispatched_at_utc (str): ISO 8601 timestamp of dispatch.
            error (str|None): Error description if any dispatch failed.
    """
    with _time_tool("send_reroute_notification") as timing:
        logger.info(
            "tool_called",
            tool="send_reroute_notification",
            shipment_id=shipment_id,
            customer_id=customer_id,
            new_carrier=new_carrier,
            priority=priority,
        )

        _simulator.maybe_raise("send_reroute_notification")

        # Resolve customer contacts
        customer = _CUSTOMER_CONTACTS.get(
            customer_id,
            {
                "email": [f"unknown-customer-{customer_id}@fallback.example.com"],
                "sms": [],
                "name": customer_id,
            },
        )

        all_recipients: list[str] = []
        channels_used: list[str] = []

        # Always send email
        email_recipients = customer["email"] + _OPS_TEAM_EMAILS
        all_recipients.extend(email_recipients)
        channels_used.append("EMAIL")

        # SMS only for CRITICAL priority
        if priority.upper() == "CRITICAL" and customer.get("sms"):
            all_recipients.extend(customer["sms"])
            channels_used.append("SMS")

        notification_id = f"NOTIF-{str(uuid.uuid4())[:8].upper()}"
        dispatched_at = datetime.now(timezone.utc).isoformat()

        # Simulate notification payload construction and dispatch
        _payload = {
            "notification_id": notification_id,
            "shipment_id": shipment_id,
            "customer_name": customer["name"],
            "subject": f"[{priority}] Shipment {shipment_id} Rerouted — Action Required",
            "body_summary": (
                f"Your shipment {shipment_id} was delayed due to {delay_reason}. "
                f"We have autonomously rerouted via {new_carrier} "
                f"(Ref: {confirmation_number}). "
                f"New ETA: {new_eta_hours:.0f} hours. "
                f"Cost: ${new_cost_usd:,.2f} USD."
            ),
            "original_carrier": original_carrier,
            "new_carrier": new_carrier,
            "new_route_id": new_route_id,
            "new_eta_hours": new_eta_hours,
            "new_cost_usd": new_cost_usd,
            "confirmation_number": confirmation_number,
        }

        logger.info(
            "notification_dispatched",
            tool="send_reroute_notification",
            notification_id=notification_id,
            recipient_count=len(all_recipients),
            channels=channels_used,
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "success": True,
            "notification_id": notification_id,
            "recipients": all_recipients,
            "recipient_count": len(all_recipients),
            "channels_used": channels_used,
            "dispatched_at_utc": dispatched_at,
            "customer_name": customer["name"],
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }


# ---------------------------------------------------------------------------
# Tool 7: generate_execution_report
# ---------------------------------------------------------------------------


@tool
def generate_execution_report(
    shipment_id: str,
    run_id: str,
    model_name: str,
    delay_reason: str,
    original_carrier: str,
    original_route_id: str,
    new_carrier: str,
    new_route_id: str,
    new_transit_mode: str,
    new_eta_hours: float,
    new_cost_usd: float,
    original_cost_usd: float,
    confirmation_number: str,
    execution_status: str,
    total_steps: int,
    total_tool_calls: int,
    successful_tool_calls: int,
    total_cost_usd_llm: float,
    error_count: int,
) -> dict[str, Any]:
    """
    Generate a structured execution report summarising the entire rerouting event.

    Call this as the final step of a successful (or failed) rerouting run.
    The report is the primary business deliverable and is also used by
    the trajectory evaluator to score final output quality.

    Args:
        shipment_id: The shipment that was rerouted.
        run_id: The LangGraph run identifier (for trajectory correlation).
        model_name: The LLM model used for this run.
        delay_reason: The disruption reason from the original alert.
        original_carrier: Carrier that was disrupted.
        original_route_id: Original route that was disrupted.
        new_carrier: Carrier selected for rerouting.
        new_route_id: New route identifier.
        new_transit_mode: Transit mode of the new route.
        new_eta_hours: Final confirmed ETA in hours.
        new_cost_usd: Final confirmed cost in USD.
        original_cost_usd: Original shipment cost for delta comparison.
        confirmation_number: Carrier booking reference (or 'N/A' if failed).
        execution_status: Final workflow status (e.g. 'completed', 'failed').
        total_steps: Number of LangGraph nodes executed.
        total_tool_calls: Total tool calls made across all agents.
        successful_tool_calls: Tool calls that returned success.
        total_cost_usd_llm: Estimated LLM API cost in USD for this run.
        error_count: Number of errors encountered during the run.

    Returns:
        Dict with keys:
            report_id (str): Unique report identifier.
            summary (str): Executive summary paragraph.
            sections (dict): Full structured report content.
            generated_at_utc (str): Timestamp.
            error (str|None): Error if report generation failed.
    """
    with _time_tool("generate_execution_report") as timing:
        logger.info(
            "tool_called",
            tool="generate_execution_report",
            shipment_id=shipment_id,
            execution_status=execution_status,
        )

        _simulator.maybe_raise("generate_execution_report")

        report_id = f"RPT-{str(uuid.uuid4())[:8].upper()}"
        generated_at = datetime.now(timezone.utc).isoformat()

        cost_delta = new_cost_usd - original_cost_usd
        cost_delta_str = (
            f"+${cost_delta:,.2f}" if cost_delta >= 0 else f"-${abs(cost_delta):,.2f}"
        )
        tool_accuracy_pct = (
            round((successful_tool_calls / total_tool_calls) * 100, 1)
            if total_tool_calls > 0
            else 0.0
        )

        summary = (
            f"Shipment {shipment_id} was disrupted due to {delay_reason} on "
            f"carrier {original_carrier} (route {original_route_id}). "
            f"The autonomous rerouting system (model: {model_name}) resolved "
            f"the disruption in {total_steps} workflow steps, selecting "
            f"{new_carrier} via {new_transit_mode} ({new_route_id}). "
            f"New ETA: {new_eta_hours:.0f}h | Cost delta: {cost_delta_str} | "
            f"Status: {execution_status.upper()} | Confirmation: {confirmation_number}."
        )

        report = {
            "report_id": report_id,
            "summary": summary,
            "sections": {
                "incident": {
                    "shipment_id": shipment_id,
                    "delay_reason": delay_reason,
                    "original_carrier": original_carrier,
                    "original_route_id": original_route_id,
                },
                "resolution": {
                    "status": execution_status,
                    "new_carrier": new_carrier,
                    "new_route_id": new_route_id,
                    "new_transit_mode": new_transit_mode,
                    "new_eta_hours": new_eta_hours,
                    "confirmation_number": confirmation_number,
                },
                "financials": {
                    "original_cost_usd": original_cost_usd,
                    "new_cost_usd": new_cost_usd,
                    "cost_delta_usd": round(cost_delta, 2),
                    "cost_delta_formatted": cost_delta_str,
                },
                "agent_metrics": {
                    "model": model_name,
                    "run_id": run_id,
                    "total_steps": total_steps,
                    "total_tool_calls": total_tool_calls,
                    "successful_tool_calls": successful_tool_calls,
                    "tool_accuracy_pct": tool_accuracy_pct,
                    "error_count": error_count,
                    "llm_api_cost_usd": round(total_cost_usd_llm, 6),
                },
            },
            "generated_at_utc": generated_at,
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }

        logger.info(
            "tool_success",
            tool="generate_execution_report",
            report_id=report_id,
            status=execution_status,
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return report
