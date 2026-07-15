"""
agents/decision_agent.py — Decision Agent
==========================================

Why this agent exists
---------------------
The Decision Agent is the reasoning core of the system.  It receives
the full list of ``CarrierOption`` objects from the Research Agent and
must select the single best option.  Its output — a ``SelectedRoute``
with a detailed business rationale — is what the Execution Agent acts on.

The Decision Agent demonstrates the most sophisticated LLM behaviour:
it must weigh multiple competing objectives (speed vs. cost vs. reliability)
and produce a *justified* decision, not just a ranked pick.

The quality of the ``decision_rationale`` field is the primary input to
the trajectory evaluator's **reasoning quality score**.

State fields read
-----------------
    carrier_options       : Full list from Research Agent (including rejected ones)
    rerouting_constraints : Constraint dict from Planner Agent
    telemetry_alert       : Shipment context (deadline, priority, cargo_value_usd)
    run_id, model_name, step_count

State fields written
--------------------
    selected_route         : SelectedRoute (replace semantics)
    decision_summary       : str for the ops report
    execution_status       : "executing"
    should_escalate        : True if confidence < 0.65 or no viable options
    step_count             : incremented
    trajectory             : one snapshot appended
    (evaluation metadata)

Tools used
----------
    validate_route_constraints  — checks each top candidate against all
                                  hard constraints before committing

(from tools.DECISION_TOOLS — exactly one tool)

Prompt design rationale
-----------------------
The system prompt presents the agent with a ranked option table and
instructs it to:
1.  Call ``validate_route_constraints`` for the top 3 candidates only
    (not all — this enforces efficient tool use, which is testable).
2.  Select the option with the best balance of business objectives.
3.  Express confidence as a float (this enables escalation logic).
4.  Provide a rationale that a non-technical stakeholder can understand.

The business-priority weighting is explicitly stated in the prompt:
    reliability > deadline adherence > cost minimisation
This reflects the assignment's logistics domain (cargo safety comes
first, then speed, then cost optimisation).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from config.settings import AppSettings
from graphs.state import CarrierOption, GraphState, SelectedRoute, ShipmentAlert, ToolCallRecord
from tools import DECISION_TOOLS

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Decision Agent in an autonomous logistics rerouting system.

YOUR ROLE: Select the single best rerouting option from the validated alternatives.

AVAILABLE TOOLS:
- validate_route_constraints  — checks a carrier option against all hard constraints

BUSINESS PRIORITY (apply in this order):
1. FEASIBILITY   — must satisfy all hard constraints (cargo, capacity, deadline)
2. RELIABILITY   — highest on-time delivery rate wins
3. SPEED         — faster ETA is better, especially for CRITICAL priority
4. COST          — minimise cost delta vs. original, but do not sacrifice 1-3

TOOL STRATEGY:
Step 1: Review the ranked carrier options table below.
Step 2: Call validate_route_constraints for the TOP 3 options by score.
Step 3: Among those that return valid=True, select the best option.
Step 4: If NO options return valid=True, call validate_route_constraints for
        options 4-6 as a fallback.
Step 5: Respond with the JSON decision.

CARRIER OPTIONS (ranked by composite score):
{options_table}

CONSTRAINTS:
{constraints_block}

SHIPMENT CONTEXT:
  Shipment ID    : {shipment_id}
  Priority       : {priority}
  Cargo Value    : ${cargo_value:,.0f} USD
  Deadline       : {deadline}

After all tool calls, respond ONLY with this JSON (no markdown):
{
  "reasoning": "<step-by-step evaluation of the top candidates, 4-6 sentences>",
  "selected_carrier_id": "CARRIER-XX",
  "selected_route_id": "route-id-string",
  "decision_rationale": "<business-facing rationale, 2-3 sentences suitable for ops report>",
  "confidence_score": <float 0.0-1.0>,
  "requires_human_approval": <true|false>,
  "alternatives_considered": <int>
}"""


# ---------------------------------------------------------------------------
# DecisionAgent
# ---------------------------------------------------------------------------


class DecisionAgent(BaseAgent):
    """
    Third node in the logistics rerouting graph.

    Evaluates carrier options, validates the top candidates against
    hard constraints, and selects the optimal route with a justified
    business rationale.
    """

    agent_name: str = "decision"

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__(settings)

    def _format_options_table(self, options: list[CarrierOption]) -> str:
        """Format carrier options as a readable table for the LLM prompt."""
        if not options:
            return "  (no options available)"

        lines = [
            f"  {'#':>2}  {'Carrier ID':18} {'Mode':12} {'ETA (h)':>8} "
            f"{'Cost USD':>12} {'Reliability':>12} {'Viable':>7} {'Score':>7}"
        ]
        lines.append("  " + "-" * 82)
        for i, opt in enumerate(options[:10], 1):  # Show top 10
            lines.append(
                f"  {i:>2}  {opt.carrier_id:18} {opt.transit_mode:12} "
                f"{opt.eta_hours:>8.1f} {opt.cost_usd:>12,.0f} "
                f"{opt.reliability_score:>12.2f} {str(opt.is_viable):>7} "
                f"{opt.score:>7.4f}"
            )
        return "\n".join(lines)

    def _build_prompts(
        self,
        options: list[CarrierOption],
        constraints: dict[str, Any],
        alert: ShipmentAlert,
    ) -> tuple[str, str]:
        """Build (system_prompt, human_prompt) for the decision step."""
        constraints_block = "\n".join(
            f"  {k}: {v}" for k, v in constraints.items()
        )
        system = _SYSTEM_PROMPT.format(
            options_table=self._format_options_table(options),
            constraints_block=constraints_block,
            shipment_id=alert.shipment_id,
            priority=alert.priority,
            cargo_value=alert.cargo_value_usd,
            deadline=alert.deadline_utc.strftime("%Y-%m-%d %H:%M UTC"),
        )
        viable_count = sum(1 for o in options if o.is_viable)
        human = (
            f"Evaluate the {len(options)} carrier options ({viable_count} viable). "
            f"Validate the top 3 by score using validate_route_constraints. "
            f"Select the best option and provide your JSON decision."
        )
        return system, human

    def _build_selected_route(
        self,
        parsed: dict[str, Any],
        options: list[CarrierOption],
        tool_records: list[ToolCallRecord],
    ) -> SelectedRoute | None:
        """
        Build a ``SelectedRoute`` from the LLM's JSON decision.

        Looks up the selected carrier's metrics from the options list
        (not from the LLM response) to ensure numeric values are
        grounded in actual tool data, not LLM-generated estimates.
        """
        selected_cid = parsed.get("selected_carrier_id", "")
        if not selected_cid:
            return None

        # Find the matching option
        option = next(
            (o for o in options if o.carrier_id == selected_cid),
            None,
        )
        if option is None:
            # Fall back to first viable option if LLM hallucinated a carrier ID
            option = next((o for o in options if o.is_viable), None)
            if option is None:
                return None

        return SelectedRoute(
            carrier_id=option.carrier_id,
            carrier=option.carrier_name,
            route_id=parsed.get("selected_route_id", option.route_id),
            transit_mode=option.transit_mode,
            estimated_cost_usd=option.cost_usd,
            cost_delta_usd=option.cost_delta_usd,
            eta_hours=option.eta_hours,
            eta_delta_hours=option.eta_delta_hours,
            reliability_score=option.reliability_score,
            decision_rationale=parsed.get(
                "decision_rationale",
                f"Selected {option.carrier_name} based on reliability and ETA.",
            ),
            alternatives_considered=parsed.get(
                "alternatives_considered", len(options)
            ),
            confidence_score=float(parsed.get("confidence_score", 0.75)),
            requires_human_approval=bool(
                parsed.get("requires_human_approval", False)
            ),
            selected_at_utc=datetime.now(timezone.utc),
        )

    def run(self, state: GraphState) -> dict[str, Any]:
        """
        Execute the Decision Agent.

        Validates top carrier options and selects the best route with
        a justified business rationale.
        """
        agent_start = time.perf_counter()
        step_index = state.get("step_count", 0)
        run_id = state.get("run_id", "unknown")
        model_name = state.get("model_name", self.settings.closed_source_model)
        alert: ShipmentAlert = state["telemetry_alert"]
        options: list[CarrierOption] = state.get("carrier_options", [])
        constraints: dict[str, Any] = state.get("rerouting_constraints", {})

        self.logger.info(
            "agent_start",
            agent=self.agent_name,
            step=step_index,
            model=model_name,
            run_id=run_id,
            options_count=len(options),
            viable_count=sum(1 for o in options if o.is_viable),
        )

        error_info: dict[str, str] | None = None
        tool_records: list[ToolCallRecord] = []
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_trace = ""
        selected_route: SelectedRoute | None = None
        decision_summary = ""
        should_escalate = False

        if not options:
            # No options from Research Agent — escalate immediately
            should_escalate = True
            decision_summary = (
                f"No carrier options discovered for shipment {alert.shipment_id}. "
                "Escalating to human operator."
            )
            self.logger.warning(
                "no_options_escalating",
                agent=self.agent_name,
                shipment=alert.shipment_id,
            )
        else:
            try:
                llm = self._build_llm(model_name)
                system_prompt, human_prompt = self._build_prompts(
                    options, constraints, alert
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]

                # Run tool loop — validates top candidates
                final_response, tool_records, total_tokens = self._invoke_with_tool_loop(
                    llm=llm,
                    tools=DECISION_TOOLS,
                    messages=messages,
                    max_iterations=5,
                )

                prompt_tokens = total_tokens["prompt"]
                completion_tokens = total_tokens["completion"]

                parsed = self._parse_json(final_response.content or "")
                reasoning_trace = parsed.get("reasoning", final_response.content[:500])

                selected_route = self._build_selected_route(parsed, options, tool_records)

                if selected_route:
                    should_escalate = selected_route.requires_human_approval or (
                        selected_route.confidence_score < 0.65
                    )
                    decision_summary = (
                        f"Selected {selected_route.carrier} ({selected_route.transit_mode}) "
                        f"at ${selected_route.estimated_cost_usd:,.0f} USD | "
                        f"ETA: {selected_route.eta_hours:.0f}h | "
                        f"Confidence: {selected_route.confidence_score:.0%}"
                    )
                    self.logger.info(
                        "agent_success",
                        agent=self.agent_name,
                        selected_carrier=selected_route.carrier_id,
                        confidence=selected_route.confidence_score,
                        requires_approval=selected_route.requires_human_approval,
                    )
                else:
                    should_escalate = True
                    decision_summary = (
                        "Decision Agent could not select a valid route. Escalating."
                    )
                    self.logger.warning(
                        "no_route_selected",
                        agent=self.agent_name,
                    )

            except Exception as exc:
                error_info = {"type": type(exc).__name__, "message": str(exc)}
                decision_summary = f"Decision failed: {exc}"
                should_escalate = alert.priority == "CRITICAL"
                self.logger.exception(
                    "agent_error",
                    agent=self.agent_name,
                    error=str(exc),
                )

        latency_ms = (time.perf_counter() - agent_start) * 1000.0
        successful_calls = sum(1 for r in tool_records if r.success)
        cost = self._estimate_cost(model_name, prompt_tokens, completion_tokens)

        snapshot = self._make_snapshot(
            run_id=run_id,
            model_name=model_name,
            step_index=step_index,
            input_summary={
                "options_count": len(options),
                "viable_count": sum(1 for o in options if o.is_viable),
            },
            output_summary={
                "selected_carrier": selected_route.carrier_id if selected_route else None,
                "confidence": selected_route.confidence_score if selected_route else None,
                "should_escalate": should_escalate,
                "execution_status": "executing",
            },
            tool_records=tool_records,
            reasoning_trace=reasoning_trace,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_info=error_info,
        )

        self.logger.info(
            "agent_complete",
            agent=self.agent_name,
            latency_ms=round(latency_ms, 1),
            selected=selected_route.carrier_id if selected_route else "none",
            escalate=should_escalate,
        )

        return {
            "selected_route": selected_route,
            "decision_summary": decision_summary,
            "execution_status": "executing",
            "should_escalate": should_escalate,
            "step_count": step_index + 1,
            "trajectory": [snapshot],
            "total_tool_calls": len(tool_records),
            "successful_tool_calls": successful_calls,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "total_cost_usd": cost,
            "error_message": error_info["message"] if error_info else None,
            "error_history": [error_info["message"]] if error_info else [],
        }
