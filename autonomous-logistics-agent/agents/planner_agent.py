"""
agents/planner_agent.py — Planner Agent
========================================

Why this agent exists
---------------------
The Planner Agent is the first node in the graph.  Its job is to
translate a raw, unstructured telemetry alert into a machine-readable
rerouting plan and a structured constraints dictionary that every
downstream agent can consume without re-parsing.

Without a Planner, the Research Agent would need to infer constraints
from the raw alert — duplicating logic and making the graph fragile
to alert schema changes.  The Planner is the single point of contact
with the input data.

State fields read
-----------------
    telemetry_alert   : The inbound ShipmentAlert (all 13 fields used)
    model_name        : Which LLM to instantiate
    run_id            : For trajectory correlation
    step_count        : To compute the snapshot's step_index

State fields written
--------------------
    rerouting_plan         : str — narrative plan produced by the LLM
    rerouting_constraints  : dict — machine-readable constraint set
    execution_status       : "researching" (passes control to Research Agent)
    step_count             : incremented by 1
    trajectory             : one TrajectorySnapshot appended
    total_prompt_tokens    : updated
    total_completion_tokens: updated
    total_cost_usd         : updated

No tools
--------
The Planner Agent uses NO tools.  It reasons purely from the alert
data.  This is an intentional design choice:
- Keeps the Planner fast and deterministic
- Makes trajectory evaluation of planning quality clean (no tool calls
  to disentangle from reasoning quality)
- Follows single responsibility: plan first, then research

Prompt design rationale
-----------------------
The system prompt:
  1. Establishes the agent's identity and scope (plan ONLY, no tools)
  2. Specifies the exact JSON output schema with field-level comments
  3. Provides concrete heuristics for constraint derivation (e.g.,
     for CRITICAL priority, set max_cost_delta to 20% of cargo value)
  4. Instructs the model to reason step-by-step BEFORE outputting JSON
     (chain-of-thought → better structured output)

The human prompt injects ALL ShipmentAlert fields in a readable format
so the LLM has complete context without needing to call a lookup tool.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from config.settings import AppSettings
from graphs.state import GraphState, ShipmentAlert

# ---------------------------------------------------------------------------
# System prompt (stable across all runs)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Planner Agent in an autonomous logistics rerouting system.

YOUR ROLE: Analyse a shipment delay telemetry alert and produce:
1. A concise natural-language rerouting plan
2. A machine-readable constraints dictionary for the Research Agent

YOU DO NOT USE TOOLS. Output pure reasoning only.

CONSTRAINT DERIVATION RULES (apply these systematically):
- max_cost_delta_usd:
    CRITICAL priority → 20% of cargo_value_usd (maximum acceptable rerouting cost increase)
    HIGH priority     → 15% of cargo_value_usd
    MEDIUM priority   → 10% of cargo_value_usd
    LOW priority      →  5% of cargo_value_usd
- required_transit_modes:
    PERISHABLE cargo  → ["AIR"] only
    HIGH_VALUE cargo  → ["AIR", "MULTIMODAL"] (speed + security)
    HAZMAT cargo      → ["SEA", "MULTIMODAL"] (regulated)
    delay > 72h       → include all modes to maximise options
    delay < 24h       → prefer ["AIR"] for speed
- min_reliability_score:
    CRITICAL priority → 0.90 minimum
    HIGH priority     → 0.85 minimum
    MEDIUM/LOW        → 0.80 minimum
- excluded_carrier_ids:
    Always exclude the disrupted carrier's ID (current_carrier field)
    For WEATHER/PORT_CLOSURE: also exclude any carrier on the same lane

REASONING APPROACH:
Step 1: Identify the core constraint (deadline vs cost vs cargo type)
Step 2: Determine which transit modes are feasible
Step 3: Calculate the cost ceiling
Step 4: Set reliability floor based on priority
Step 5: Write the plan as a single actionable paragraph

OUTPUT FORMAT (valid JSON only, no markdown fences):
{
  "reasoning": "<your step-by-step reasoning, 3-5 sentences>",
  "rerouting_plan": "<1-2 sentence actionable plan for the rerouting team>",
  "constraints": {
    "max_cost_delta_usd": <float>,
    "original_cost_usd": <float — use 15000.0 as baseline if unknown>,
    "required_transit_modes": ["AIR"],
    "excluded_carrier_ids": ["carrier-id-here"],
    "min_reliability_score": <float 0.0-1.0>,
    "max_eta_hours": <float — hours from NOW that deadline allows>,
    "cargo_type": "<cargo type string>",
    "origin": "<origin code>",
    "destination": "<destination code>",
    "weight_kg": <float>,
    "volume_cbm": <float>
  }
}"""


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    """
    First node in the logistics rerouting graph.

    Converts a ``ShipmentAlert`` into a structured ``rerouting_plan`` and
    ``rerouting_constraints`` dict that the Research Agent uses to query
    carrier alternatives.

    No tools are used — the agent reasons entirely from the alert data.
    """

    agent_name: str = "planner"

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__(settings)

    def _build_human_prompt(self, alert: ShipmentAlert) -> str:
        """
        Construct the human-turn message from the ShipmentAlert fields.

        Formats every field explicitly so the LLM has full context
        and cannot hallucinate missing values.
        """
        hours_to_deadline = alert.hours_to_deadline
        deadline_str = alert.deadline_utc.strftime("%Y-%m-%d %H:%M UTC")

        return f"""TELEMETRY ALERT — SHIPMENT DISRUPTION

Shipment ID    : {alert.shipment_id}
Customer       : {alert.customer_id}
Priority       : {alert.priority}

ROUTE (DISRUPTED)
  Origin       : {alert.origin}
  Destination  : {alert.destination}
  Carrier      : {alert.current_carrier}
  Route ID     : {alert.current_route_id}

DISRUPTION
  Reason       : {alert.delay_reason}
  Delay        : {alert.delay_hours:.0f} hours

CARGO
  Type         : {alert.cargo_type}
  Value        : ${alert.cargo_value_usd:,.0f} USD
  Weight       : {alert.weight_kg:.0f} kg
  Volume       : {alert.volume_cbm:.1f} cbm

DEADLINE
  Deadline UTC : {deadline_str}
  Hours left   : {max(hours_to_deadline, 0):.1f}h (from now)

Produce the rerouting plan and constraints as specified. Output valid JSON only."""

    def run(self, state: GraphState) -> dict[str, Any]:
        """
        Execute the Planner Agent.

        Reads the telemetry alert, calls the LLM (no tools), parses
        the JSON response, and returns the plan + constraints as a
        partial state update.
        """
        agent_start = time.perf_counter()
        step_index = state.get("step_count", 0)
        run_id = state.get("run_id", "unknown")
        model_name = state.get("model_name", self.settings.closed_source_model)

        self.logger.info(
            "agent_start",
            agent=self.agent_name,
            step=step_index,
            model=model_name,
            run_id=run_id,
        )

        alert: ShipmentAlert = state["telemetry_alert"]
        error_info: dict[str, str] | None = None
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_trace = ""
        rerouting_plan = ""
        rerouting_constraints: dict[str, Any] = {}

        try:
            llm = self._build_llm(model_name)
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=self._build_human_prompt(alert)),
            ]

            # Planner uses NO tools — plain invoke
            response = llm.invoke(messages)

            usage = getattr(response, "usage_metadata", None) or {}
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)

            raw_content = response.content or ""
            parsed = self._parse_json(raw_content)

            reasoning_trace = parsed.get("reasoning", raw_content[:500])
            rerouting_plan = parsed.get(
                "rerouting_plan",
                f"Reroute shipment {alert.shipment_id} urgently due to {alert.delay_reason}.",
            )
            rerouting_constraints = parsed.get("constraints", {})

            # Enforce non-empty constraints with sensible fallbacks
            if not rerouting_constraints:
                rerouting_constraints = self._default_constraints(alert)
                self.logger.warning(
                    "constraints_parse_failed_using_defaults",
                    agent=self.agent_name,
                    shipment=alert.shipment_id,
                )

            self.logger.info(
                "agent_success",
                agent=self.agent_name,
                plan_length=len(rerouting_plan),
                constraint_keys=list(rerouting_constraints.keys()),
            )

        except Exception as exc:
            error_info = {"type": type(exc).__name__, "message": str(exc)}
            rerouting_plan = (
                f"Emergency reroute required for {alert.shipment_id} "
                f"due to {alert.delay_reason}."
            )
            rerouting_constraints = self._default_constraints(alert)
            self.logger.exception(
                "agent_error",
                agent=self.agent_name,
                error=str(exc),
            )

        # Build trajectory snapshot
        latency_ms = (time.perf_counter() - agent_start) * 1000.0
        snapshot = self._make_snapshot(
            run_id=run_id,
            model_name=model_name,
            step_index=step_index,
            input_summary={
                "shipment_id": alert.shipment_id,
                "priority": alert.priority,
                "delay_reason": alert.delay_reason,
                "delay_hours": alert.delay_hours,
                "cargo_type": alert.cargo_type,
            },
            output_summary={
                "rerouting_plan": rerouting_plan[:200],
                "constraint_keys": list(rerouting_constraints.keys()),
                "execution_status": "researching",
            },
            tool_records=[],  # Planner uses no tools
            reasoning_trace=reasoning_trace,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_info=error_info,
        )

        cost = self._estimate_cost(model_name, prompt_tokens, completion_tokens)

        self.logger.info(
            "agent_complete",
            agent=self.agent_name,
            latency_ms=round(latency_ms, 1),
            tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
        )

        return {
            "rerouting_plan": rerouting_plan,
            "rerouting_constraints": rerouting_constraints,
            "execution_status": "researching",
            "step_count": step_index + 1,
            "trajectory": [snapshot],
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cost_usd": cost,
            "error_message": error_info["message"] if error_info else None,
            "error_history": [error_info["message"]] if error_info else [],
        }

    def _default_constraints(self, alert: ShipmentAlert) -> dict[str, Any]:
        """
        Produce hard-coded fallback constraints when LLM JSON parsing fails.

        Uses the same derivation rules as the system prompt so the
        Research Agent always receives a coherent constraint set.
        """
        priority_cost_pct = {
            "CRITICAL": 0.20,
            "HIGH": 0.15,
            "MEDIUM": 0.10,
            "LOW": 0.05,
        }
        cost_pct = priority_cost_pct.get(alert.priority, 0.15)
        max_cost_delta = alert.cargo_value_usd * cost_pct

        if alert.cargo_type == "PERISHABLE":
            modes = ["AIR"]
        elif alert.cargo_type == "HIGH_VALUE":
            modes = ["AIR", "MULTIMODAL"]
        elif alert.cargo_type == "HAZMAT":
            modes = ["SEA", "MULTIMODAL"]
        else:
            modes = ["AIR", "SEA", "MULTIMODAL"]

        reliability_floor = {
            "CRITICAL": 0.90,
            "HIGH": 0.85,
            "MEDIUM": 0.80,
            "LOW": 0.80,
        }.get(alert.priority, 0.80)

        return {
            "max_cost_delta_usd": round(max_cost_delta, 2),
            "original_cost_usd": 15000.0,
            "required_transit_modes": modes,
            "excluded_carrier_ids": [alert.current_carrier],
            "min_reliability_score": reliability_floor,
            "max_eta_hours": max(alert.hours_to_deadline, 0),
            "cargo_type": alert.cargo_type,
            "origin": alert.origin,
            "destination": alert.destination,
            "weight_kg": alert.weight_kg,
            "volume_cbm": alert.volume_cbm,
        }
