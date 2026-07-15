"""
evaluation/trajectory_evaluator.py — Trajectory-Based Evaluation Engine
========================================================================

Computes structured performance metrics from the execution trajectory
of a LangGraph run.  Rather than scoring only the final output, the
evaluator analyzes intermediate reasoning traces, tool calling sequences,
and error-recovery paths.

Evaluation Dimensions
---------------------
1.  **Reasoning Quality (0.0–1.0)**
    Scores the agent's chain-of-thought at each node.  It uses a rule-based
    checklist that looks for critical domain terms (e.g., "deadline",
    "reliability", "priority", "cost") in the reasoning traces.
    - Planner: must mention deadline, priority, cargo type
    - Research: must mention carrier discovery, transit modes
    - Decision: must justify selection by comparing reliability, cost, ETA
    - Execution: must confirm execution, report generation, notification

2.  **Tool-Calling Accuracy (0.0–1.0)**
    Evaluates:
    - Success rate: successful tool calls / total tool calls
    - Sequence correctness: does the agent call research/decision/execution
      tools in the expected order? (e.g., no notification before booking)
    - Argument validation: did it call validate_route_constraints with the
      matching carrier_id and route_id?

3.  **Error Recovery Rate (0.0–1.0)**
    Measures the system's self-healing efficiency.  Locates all snapshots
    with `error_occurred=True`.  A recovery is "successful" if:
    - The agent successfully retried the tool/action in a later snapshot.
    - The graph completed without unhandled crashes.

4.  **Final Output Quality (0.0–1.0)**
    Scores the business outcome:
    - Status "completed" + confirmation number + notification sent: 1.0
    - Status "escalated" + documented operator transfer: 0.8 (valid outcome)
    - Status "failed": 0.0

5.  **Composite Score (0.0–1.0)**
    A weighted index of the individual dimensions:
        30% Tool-Calling Accuracy
        30% Reasoning Quality
        20% Final Output Quality
        20% Error Recovery Rate
"""

from __future__ import annotations

import re
from typing import Any, cast

from graphs.state import GraphState, TrajectorySnapshot, ToolCallRecord, SelectedRoute, ExecutionResult
from utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryEvaluator:
    """
    Evaluator that parses a completed GraphState trajectory and computes
    quantitative performance scores.
    """

    def __init__(self) -> None:
        pass

    # -----------------------------------------------------------------------
    # 1. Reasoning Quality Scoring
    # -----------------------------------------------------------------------

    def _score_reasoning_trace(self, agent_name: str, trace: str) -> float:
        """
        Score a single reasoning trace using a rule-based keyword checklist.
        """
        if not trace or len(trace.strip()) < 10:
            return 0.0

        trace_lower = trace.lower()
        score = 0.0
        checks = 0

        # Planner Agent keywords
        if agent_name == "planner":
            checklist = ["deadline", "priority", "cargo", "constraint", "exclude", "cost"]
            checks = len(checklist)
            matched = sum(1 for keyword in checklist if keyword in trace_lower)
            score = matched / checks

        # Research Agent keywords
        elif agent_name == "research":
            checklist = ["carrier", "lane", "capacity", "eta", "cost", "viable", "mode"]
            checks = len(checklist)
            matched = sum(1 for keyword in checklist if keyword in trace_lower)
            score = matched / checks

        # Decision Agent keywords
        elif agent_name == "decision":
            checklist = ["select", "reliability", "cost", "eta", "validate", "justification", "balance"]
            checks = len(checklist)
            matched = sum(1 for keyword in checklist if keyword in trace_lower)
            score = matched / checks

        # Execution Agent keywords
        elif agent_name == "execution":
            checklist = ["confirm", "notification", "report", "booking", "success", "recipient"]
            checks = len(checklist)
            matched = sum(1 for keyword in checklist if keyword in trace_lower)
            score = matched / checks

        else:
            # Fallback for generic nodes
            score = 1.0 if len(trace) > 50 else 0.5

        # Small bonus for length (up to +0.1) — indicating detailed explanation
        length_bonus = min(len(trace) / 1000.0, 0.1)
        return min(score + length_bonus, 1.0)

    def _evaluate_reasoning(self, trajectory: list[TrajectorySnapshot]) -> float:
        """
        Compute the average reasoning quality score across all snapshots.
        """
        if not trajectory:
            return 0.0

        scores: list[float] = []
        for snapshot in trajectory:
            # Skip retry_increment and other aux nodes from reasoning evaluation
            if snapshot.agent_name in ["planner", "research", "decision", "execution"]:
                score = self._score_reasoning_trace(
                    snapshot.agent_name,
                    snapshot.reasoning_trace or "",
                )
                scores.append(score)

        return sum(scores) / len(scores) if scores else 1.0

    # -----------------------------------------------------------------------
    # 2. Tool-Calling Accuracy
    # -----------------------------------------------------------------------

    def _evaluate_tool_calling(
        self,
        trajectory: list[TrajectorySnapshot],
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Check:
        - Tool success rate (successful calls / total calls)
        - Sequence correctness (no notifications sent before booking is confirmed)
        - Parameter grounding (validating constraints was called with matching parameters)
        """
        total_calls = 0
        successful_calls = 0
        sequence_violations: list[str] = []
        grounding_violations: list[str] = []

        seen_booking_confirmation = False
        seen_notification = False

        for snapshot in trajectory:
            for tc in snapshot.tool_calls:
                total_calls += 1
                if tc.success:
                    successful_calls += 1

                # Sequence validation: Notification sent before route is confirmed?
                if tc.tool_name == "send_reroute_notification":
                    seen_notification = True
                    # Check if booking confirmation was set
                    confirmation = tc.arguments.get("confirmation_number")
                    if not confirmation or confirmation == "PENDING" or confirmation == "N/A":
                        sequence_violations.append(
                            f"Notification sent without valid confirmation number in step {snapshot.step_index}."
                        )

                # Grounding validation: validate_route_constraints arguments match carrier option?
                if tc.tool_name == "validate_route_constraints":
                    cid = tc.arguments.get("carrier_id")
                    cost = tc.arguments.get("estimated_cost_usd")
                    eta = tc.arguments.get("eta_hours")

                    # Look up carrier option from carrier_options list
                    options = state.get("carrier_options", [])
                    matched_opt = next((o for o in options if o.carrier_id == cid), None)

                    if matched_opt:
                        if cost is not None and abs(cost - matched_opt.cost_usd) > 0.01:
                            grounding_violations.append(
                                f"validate_route_constraints called with cost ${cost} "
                                f"but carrier_options cost is ${matched_opt.cost_usd}."
                            )
                        if eta is not None and abs(eta - matched_opt.eta_hours) > 0.01:
                            grounding_violations.append(
                                f"validate_route_constraints called with ETA {eta}h "
                                f"but carrier_options ETA is {matched_opt.eta_hours}h."
                            )
                    else:
                        grounding_violations.append(
                            f"validate_route_constraints called for unknown carrier_id '{cid}'."
                        )

        # Success rate component (0.0 to 1.0)
        success_rate = successful_calls / total_calls if total_calls > 0 else 1.0

        # Sequence score: deduct 0.5 per sequence violation, minimum 0.0
        seq_score = max(1.0 - (len(sequence_violations) * 0.5), 0.0)

        # Grounding score: deduct 0.25 per grounding error, minimum 0.0
        grounding_score = max(1.0 - (len(grounding_violations) * 0.25), 0.0)

        # Overall tool calling score is a combined index
        score = (success_rate * 0.5) + (seq_score * 0.3) + (grounding_score * 0.2)

        return {
            "score": round(score, 4),
            "success_rate": round(success_rate, 4),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "sequence_violations": sequence_violations,
            "grounding_violations": grounding_violations,
        }

    # -----------------------------------------------------------------------
    # 3. Error Recovery Rate
    # -----------------------------------------------------------------------

    def _evaluate_error_recovery(
        self,
        trajectory: list[TrajectorySnapshot],
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Scan trajectory for snapshots with `error_occurred=True`.
        For each error, check if a subsequent step completed the action successfully
        or if the run completed overall.
        """
        error_snapshots = [s for s in trajectory if s.error_occurred]
        total_errors = len(error_snapshots)

        if total_errors == 0:
            return {
                "score": 1.0,
                "total_errors": 0,
                "recovered_errors": 0,
                "recovery_rate": 1.0,
                "recovery_details": [],
            }

        recovered_count = 0
        details: list[str] = []
        final_status = state.get("execution_status", "failed")

        # If the final state succeeded, it implies the system self-healed or completed
        # But we want to look at each error node.
        for err_snap in error_snapshots:
            agent = err_snap.agent_name
            # Check if there exists a subsequent snapshot for the same agent that succeeded
            subsequent_success = False
            for post_snap in trajectory:
                if (
                    post_snap.step_index > err_snap.step_index
                    and post_snap.agent_name == agent
                    and not post_snap.error_occurred
                ):
                    subsequent_success = True
                    break

            if subsequent_success:
                recovered_count += 1
                details.append(
                    f"Agent '{agent}' recovered in subsequent step: Tool call or execution retried and succeeded."
                )
            elif final_status in ["completed", "escalated"]:
                # The specific agent didn't succeed, but the overall graph reached a terminal success/escalate state
                recovered_count += 1
                details.append(
                    f"Disruption on agent '{agent}' was bypassed or recovered: workflow completed with status '{final_status}'."
                )
            else:
                details.append(
                    f"Unrecovered failure on agent '{agent}': graph terminated in state '{final_status}'."
                )

        recovery_rate = recovered_count / total_errors

        return {
            "score": round(recovery_rate, 4),
            "total_errors": total_errors,
            "recovered_errors": recovered_count,
            "recovery_rate": round(recovery_rate, 4),
            "recovery_details": details,
        }

    # -----------------------------------------------------------------------
    # 4. Final Output Quality Scoring
    # -----------------------------------------------------------------------

    def _evaluate_output_quality(self, state: GraphState) -> float:
        """
        Evaluate the quality of the final state artifacts:
        - completed + confirmation + notification: 1.0
        - escalated + decision summary: 0.8
        - failed: 0.0
        """
        status = state.get("execution_status", "failed")
        result: ExecutionResult | None = state.get("execution_result")
        selected: SelectedRoute | None = state.get("selected_route")

        if status == "completed":
            if result and result.success and result.confirmation_number:
                score = 0.8
                if result.notification_sent:
                    score += 0.2  # Full marks
                return score
            return 0.5  # Status completed but missing booking confirmation details

        elif status == "escalated":
            # Escalation is a valid business outcome when confidence is low or constraints are hard
            if state.get("decision_summary"):
                return 0.8
            return 0.6

        elif status == "failed":
            return 0.0

        return 0.0

    # -----------------------------------------------------------------------
    # Core Entrypoint
    # -----------------------------------------------------------------------

    def evaluate_run(self, state: GraphState) -> dict[str, Any]:
        """
        Analyze a completed logistics graph run and return all scoring metrics.
        """
        trajectory = state.get("trajectory", [])

        # 1. Basic properties
        run_id = state.get("run_id", "unknown")
        model_name = state.get("model_name", "unknown")
        execution_status = state.get("execution_status", "failed")

        # 2. Timing and cost
        total_latency_ms = sum(s.latency_ms for s in trajectory)
        total_cost_usd = state.get("total_cost_usd", 0.0)

        # 3. Component Scores
        reasoning_score = self._evaluate_reasoning(trajectory)
        tool_calling_results = self._evaluate_tool_calling(trajectory, state)
        tool_score = tool_calling_results["score"]
        error_results = self._evaluate_error_recovery(trajectory, state)
        recovery_score = error_results["score"]
        output_score = self._evaluate_output_quality(state)

        # 4. Composite Score Calculation (Weighted average)
        # Weights: 30% Tool, 30% Reasoning, 20% Output, 20% Recovery
        composite_score = (
            (tool_score * 0.3)
            + (reasoning_score * 0.3)
            + (output_score * 0.2)
            + (recovery_score * 0.2)
        )

        report = {
            "run_id": run_id,
            "model_name": model_name,
            "execution_status": execution_status,
            "total_steps": len(trajectory),
            "total_latency_ms": round(total_latency_ms, 2),
            "total_cost_usd": round(total_cost_usd, 6),
            # Scores
            "reasoning_quality_score": round(reasoning_score, 4),
            "tool_calling_accuracy_score": round(tool_score, 4),
            "error_recovery_score": round(recovery_score, 4),
            "output_quality_score": round(output_score, 4),
            "composite_score": round(composite_score, 4),
            # Details
            "tool_calling_details": tool_calling_results,
            "error_recovery_details": error_results,
        }

        logger.info(
            "run_evaluated",
            run_id=run_id,
            model=model_name,
            status=execution_status,
            composite=report["composite_score"],
            cost_usd=report["total_cost_usd"],
        )

        return report
