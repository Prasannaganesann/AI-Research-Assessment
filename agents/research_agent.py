"""
agents/research_agent.py — Research Agent
==========================================

Why this agent exists
---------------------
The Research Agent is the information-gathering node of the graph.
Its purpose is to discover ALL viable carrier alternatives on the
disrupted lane and return them as a list of typed ``CarrierOption``
objects.  Without this agent, the Decision Agent would have no data
to reason over.

The Research Agent demonstrates the most important agentic behaviour
in this system: **tool-calling accuracy**.  The trajectory evaluator
scores whether the agent called the right tools, with correct
arguments, in the right order.

State fields read
-----------------
    telemetry_alert       : Shipment details (origin, destination, cargo, weight, volume)
    rerouting_constraints : Constraint dict from the Planner Agent
    run_id                : For trajectory correlation
    model_name            : Which LLM to instantiate
    step_count            : For snapshot step_index
    retry_count           : How many times this agent has been retried

State fields written
--------------------
    carrier_options        : list[CarrierOption] — APPENDED (operator.add)
    research_summary       : str — brief summary for the report
    execution_status       : "deciding"
    step_count             : incremented by 1
    trajectory             : one TrajectorySnapshot appended
    total_tool_calls       : updated
    successful_tool_calls  : updated
    total_prompt_tokens    : updated
    total_completion_tokens: updated
    total_cost_usd         : updated

Tools used
----------
    list_available_carriers    — discovers options on the lane
    lookup_carrier_capacity    — confirms physical capacity
    calculate_route_eta        — gets carrier-specific ETA
    estimate_reroute_cost      — gets carrier-specific cost quote

(from tools.RESEARCH_TOOLS — agent cannot call any other tool)

Prompt design rationale
-----------------------
The system prompt provides a step-by-step tool usage strategy:
1. Call ``list_available_carriers`` once to get the shortlist.
2. For each viable carrier, call the three detail tools.
3. After gathering data, produce a JSON summary.

This strategy is explicit rather than letting the LLM invent its own.
Explicit strategies are more reproducible across different models,
making the GPT-4o vs Qwen3 comparison cleaner: both receive the same
instructions, so differences in behaviour reflect model capability,
not prompt interpretation variability.

Post-processing
---------------
After the LLM's tool loop completes, Python code (not the LLM) builds
the typed ``CarrierOption`` objects from the ``ToolCallRecord`` data.
This is intentional: the LLM might hallucinate numeric values when
constructing typed objects.  Using tool results directly is safer.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from config.settings import AppSettings
from graphs.state import CarrierOption, GraphState, ShipmentAlert, ToolCallRecord
from tools import RESEARCH_TOOLS

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the Research Agent in an autonomous logistics rerouting system.

YOUR ROLE: Discover all viable carrier alternatives using your tools and summarise the findings.

AVAILABLE TOOLS:
- list_available_carriers     — get all carriers on a lane for a given cargo type and mode
- lookup_carrier_capacity     — confirm a carrier can take the shipment dimensions
- calculate_route_eta         — get the carrier-specific estimated transit time
- estimate_reroute_cost       — get the carrier-specific cost quote

TOOL STRATEGY (follow this order precisely):
Step 1: Call list_available_carriers with transit_mode="ANY" to get the full option set.
Step 2: For each carrier where available=True, call lookup_carrier_capacity.
Step 3: For each carrier where capacity_available=True, call calculate_route_eta.
Step 4: For each carrier with confirmed capacity, call estimate_reroute_cost.
Step 5: When ALL tool calls are complete, respond with a JSON summary.

CONSTRAINTS FROM PLANNER (apply these as hard filters):
{constraints_block}

SHIPMENT DETAILS:
{shipment_block}

AFTER ALL TOOL CALLS, respond ONLY with this JSON (no markdown):
{{
  "reasoning": "<brief summary of what you found and which carriers are viable>",
  "research_summary": "<1-2 sentence summary for the ops report>",
  "viable_carrier_ids": ["CARRIER-XX", ...],
  "recommended_mode": "AIR|SEA|MULTIMODAL",
  "total_carriers_queried": <int>,
  "total_viable": <int>
}}"""


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent(BaseAgent):
    """
    Second node in the logistics rerouting graph.

    Discovers alternative carriers by calling the research tool set,
    then synthesises the raw tool results into typed ``CarrierOption``
    objects using Python (not the LLM) for reliability.
    """

    agent_name: str = "research"

    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__(settings)

    def _build_prompts(
        self,
        alert: ShipmentAlert,
        constraints: dict[str, Any],
    ) -> tuple[str, str]:
        """Return (system_prompt, human_prompt) for this research run."""
        constraints_block = "\n".join(
            f"  {k}: {v}" for k, v in constraints.items()
        )
        shipment_block = (
            f"  Shipment ID  : {alert.shipment_id}\n"
            f"  Origin       : {alert.origin}\n"
            f"  Destination  : {alert.destination}\n"
            f"  Cargo Type   : {alert.cargo_type}\n"
            f"  Weight       : {alert.weight_kg:.0f} kg\n"
            f"  Volume       : {alert.volume_cbm:.1f} cbm\n"
            f"  Priority     : {alert.priority}\n"
            f"  Delay Reason : {alert.delay_reason}"
        )
        system = _SYSTEM_PROMPT.format(
            constraints_block=constraints_block,
            shipment_block=shipment_block,
        )
        human = (
            f"BEGIN RESEARCH for shipment {alert.shipment_id}. "
            f"Follow the tool strategy. Query ALL available carriers on the "
            f"{alert.origin}→{alert.destination} lane. "
            f"Produce the JSON summary after all tool calls are complete."
        )
        return system, human

    def _build_carrier_options(
        self,
        tool_records: list[ToolCallRecord],
        constraints: dict[str, Any],
        original_cost_usd: float,
    ) -> list[CarrierOption]:
        """
        Synthesise ``CarrierOption`` objects from ``ToolCallRecord`` data.

        Rather than relying on the LLM to construct Pydantic models
        (which can hallucinate numeric values), this method walks the
        tool call records and builds the objects directly from the
        structured tool outputs.

        Strategy:
        - ``list_available_carriers`` records → identify carrier IDs
        - ``lookup_carrier_capacity`` records → capacity info per carrier
        - ``calculate_route_eta`` records → ETA per carrier
        - ``estimate_reroute_cost`` records → cost per carrier
        - Join on carrier_id to build one ``CarrierOption`` per carrier
        """
        # Build per-carrier data index from tool records
        carrier_data: dict[str, dict[str, Any]] = {}

        for record in tool_records:
            if not record.success or not isinstance(record.raw_result, dict):
                continue

            result = record.raw_result
            cid = result.get("carrier_id", "")

            if record.tool_name == "list_available_carriers":
                for c in result.get("carriers", []):
                    cid = c.get("carrier_id", "")
                    if cid:
                        carrier_data.setdefault(cid, {})
                        carrier_data[cid].update(
                            {
                                "carrier_name": c.get("carrier_name", cid),
                                "transit_mode": c.get("transit_mode", "UNKNOWN"),
                                "reliability_score": c.get("reliability_score", 0.85),
                                "cargo_type_supported": c.get("cargo_type_supported", True),
                                "within_capacity": c.get("within_capacity", False),
                                "listed": True,
                            }
                        )

            elif record.tool_name == "lookup_carrier_capacity" and cid:
                carrier_data.setdefault(cid, {})
                carrier_data[cid].update(
                    {
                        "capacity_available": result.get("capacity_available", False),
                        "rejection_reason": result.get("rejection_reason"),
                    }
                )

            elif record.tool_name == "calculate_route_eta" and cid:
                carrier_data.setdefault(cid, {})
                carrier_data[cid]["eta_hours"] = result.get("eta_hours", 999.0)

            elif record.tool_name == "estimate_reroute_cost" and cid:
                carrier_data.setdefault(cid, {})
                carrier_data[cid]["cost_usd"] = result.get("total_cost_usd", 0.0)

        # Build CarrierOption objects
        options: list[CarrierOption] = []
        min_reliability = constraints.get("min_reliability_score", 0.80)
        required_modes = [
            m.upper() for m in constraints.get("required_transit_modes", [])
        ]

        for cid, data in carrier_data.items():
            if not data.get("listed", False):
                continue

            eta_hours = data.get("eta_hours", 999.0)
            cost_usd = data.get("cost_usd", 0.0)
            cost_delta = cost_usd - original_cost_usd
            max_eta = constraints.get("max_eta_hours", float("inf"))
            capacity_ok = data.get("capacity_available", data.get("within_capacity", False))
            cargo_ok = data.get("cargo_type_supported", True)
            reliability_ok = data.get("reliability_score", 0.0) >= min_reliability
            mode = data.get("transit_mode", "UNKNOWN")
            mode_ok = not required_modes or mode in required_modes
            deadline_ok = eta_hours <= max_eta if max_eta else True

            all_ok = capacity_ok and cargo_ok and reliability_ok and mode_ok and deadline_ok

            rejection_parts = []
            if not capacity_ok:
                rejection_parts.append(data.get("rejection_reason") or "no capacity")
            if not cargo_ok:
                rejection_parts.append("cargo type not certified")
            if not reliability_ok:
                rejection_parts.append(f"reliability {data.get('reliability_score', 0):.2f} < {min_reliability}")
            if not mode_ok:
                rejection_parts.append(f"mode {mode} not in required {required_modes}")
            if not deadline_ok:
                rejection_parts.append(f"ETA {eta_hours}h exceeds {max_eta}h deadline")

            origin = constraints.get("origin", "UNKNOWN")
            destination = constraints.get("destination", "UNKNOWN")

            options.append(
                CarrierOption(
                    carrier_id=cid,
                    carrier_name=data.get("carrier_name", cid),
                    route_id=f"{origin}-{destination}-{cid.replace('CARRIER-', '')}-{mode}-01",
                    origin=origin,
                    destination=destination,
                    transit_mode=mode,  # type: ignore[arg-type]
                    eta_hours=eta_hours,
                    eta_delta_hours=eta_hours - max_eta if max_eta < 9999 else 0.0,
                    cost_usd=cost_usd,
                    cost_delta_usd=cost_delta,
                    capacity_available=capacity_ok,
                    reliability_score=data.get("reliability_score", 0.85),
                    cargo_type_supported=cargo_ok,
                    constraints_satisfied=all_ok,
                    rejection_reason="; ".join(rejection_parts) if rejection_parts else None,
                )
            )

        return sorted(options, key=lambda o: o.score, reverse=True)

    def run(self, state: GraphState) -> dict[str, Any]:
        """
        Execute the Research Agent.

        Calls research tools via the LLM tool loop, then synthesises
        the tool results into typed ``CarrierOption`` objects.
        """
        agent_start = time.perf_counter()
        step_index = state.get("step_count", 0)
        run_id = state.get("run_id", "unknown")
        model_name = state.get("model_name", self.settings.closed_source_model)
        alert: ShipmentAlert = state["telemetry_alert"]
        constraints: dict[str, Any] = state.get("rerouting_constraints", {})
        original_cost = constraints.get("original_cost_usd", 15000.0)

        self.logger.info(
            "agent_start",
            agent=self.agent_name,
            step=step_index,
            model=model_name,
            run_id=run_id,
            origin=alert.origin,
            destination=alert.destination,
        )

        error_info: dict[str, str] | None = None
        tool_records: list[ToolCallRecord] = []
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_trace = ""
        carrier_options: list[CarrierOption] = []
        research_summary = ""

        try:
            llm = self._build_llm(model_name)
            system_prompt, human_prompt = self._build_prompts(alert, constraints)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]

            # Run tool loop — LLM calls research tools
            final_response, tool_records, total_tokens = self._invoke_with_tool_loop(
                llm=llm,
                tools=RESEARCH_TOOLS,
                messages=messages,
                max_iterations=8,
            )

            prompt_tokens = total_tokens["prompt"]
            completion_tokens = total_tokens["completion"]

            # Parse the JSON summary from the final response
            parsed = self._parse_json(final_response.content or "")
            reasoning_trace = parsed.get("reasoning", final_response.content[:500])
            research_summary = parsed.get(
                "research_summary",
                f"Researched alternatives for {alert.shipment_id}.",
            )

            # Build typed CarrierOption objects from tool records
            carrier_options = self._build_carrier_options(
                tool_records, constraints, original_cost
            )

            viable_count = sum(1 for o in carrier_options if o.is_viable)
            self.logger.info(
                "agent_success",
                agent=self.agent_name,
                total_options=len(carrier_options),
                viable_options=viable_count,
                tool_calls=len(tool_records),
            )

            if not carrier_options:
                research_summary += " No viable alternatives found on this lane."

        except Exception as exc:
            error_info = {"type": type(exc).__name__, "message": str(exc)}
            research_summary = f"Research failed for {alert.shipment_id}: {exc}"
            self.logger.exception(
                "agent_error",
                agent=self.agent_name,
                error=str(exc),
            )

        # Compute evaluation metadata increments
        latency_ms = (time.perf_counter() - agent_start) * 1000.0
        successful_calls = sum(1 for r in tool_records if r.success)
        cost = self._estimate_cost(model_name, prompt_tokens, completion_tokens)

        snapshot = self._make_snapshot(
            run_id=run_id,
            model_name=model_name,
            step_index=step_index,
            input_summary={
                "shipment_id": alert.shipment_id,
                "constraints_keys": list(constraints.keys()),
                "transit_modes": constraints.get("required_transit_modes"),
            },
            output_summary={
                "carrier_options_count": len(carrier_options),
                "viable_count": sum(1 for o in carrier_options if o.is_viable),
                "research_summary": research_summary[:150],
                "execution_status": "deciding",
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
            options_found=len(carrier_options),
            cost_usd=round(cost, 6),
        )

        return {
            "carrier_options": carrier_options,
            "research_summary": research_summary,
            "execution_status": "deciding",
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
