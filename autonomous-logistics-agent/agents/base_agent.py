"""
agents/base_agent.py — Abstract Base Agent
===========================================

Provides the ``BaseAgent`` abstract class that all four logistics agents
inherit from.  Its responsibilities are:

1.  **LLM client construction** — builds ``ChatOpenAI`` or ``ChatGroq``
    based on the model name stored in ``GraphState.model_name``.
    This is the single point that makes the entire system model-agnostic.

2.  **Tool invocation loop** — ``_invoke_with_tool_loop`` runs the
    standard LangChain tool-calling cycle (invoke → process tool calls →
    invoke again) and collects ``ToolCallRecord`` objects as it goes.

3.  **TrajectorySnapshot factory** — ``_make_snapshot`` assembles a
    complete ``TrajectorySnapshot`` from timing, token counts, tool
    records, and error information.

4.  **Cost estimation** — ``_estimate_cost`` converts token counts into
    USD cost using published API pricing.

5.  **JSON parsing** — ``_parse_json`` extracts structured JSON from LLM
    responses with multiple fallback strategies (bare JSON, code fences,
    regex extraction).

Subclasses must implement:
    ``agent_name: str``  — identifies this agent in logs and snapshots
    ``run(state: GraphState) -> dict[str, Any]``  — domain logic

The ``__call__`` method delegates to ``run()``, making any ``BaseAgent``
subclass directly usable as a LangGraph node:
    ``graph.add_node("planner", PlannerAgent())``
"""

from __future__ import annotations

import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config.settings import AppSettings, get_settings
from graphs.state import GraphState, ToolCallRecord, TrajectorySnapshot
from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Token pricing table (USD per token, as of 2024-Q4)
# ---------------------------------------------------------------------------

_COST_PER_TOKEN: dict[str, dict[str, float]] = {
    # OpenAI — https://openai.com/api/pricing/
    "gpt-4o":             {"prompt": 2.50e-6,  "completion": 10.00e-6},
    "gpt-4o-mini":        {"prompt": 0.15e-6,  "completion": 0.60e-6},
    "gpt-4-turbo":        {"prompt": 10.00e-6, "completion": 30.00e-6},
    # Groq-hosted open-source — https://console.groq.com/docs/openai
    "qwen/qwen3-8b":      {"prompt": 0.10e-6,  "completion": 0.10e-6},
    "llama3-8b-8192":     {"prompt": 0.05e-6,  "completion": 0.10e-6},
    "mixtral-8x7b-32768": {"prompt": 0.27e-6,  "completion": 0.27e-6},
}

# Groq model name prefixes — used to select the right LangChain client
_GROQ_PREFIXES: tuple[str, ...] = (
    "qwen", "llama", "mistral", "gemma", "deepseek", "compound",
)


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base class for all logistics rerouting agents.

    Subclasses inherit trajectory capture, LLM construction, tool loop,
    and cost estimation for free.  They implement only domain-specific
    prompt construction and state transformation logic.

    Class Attributes
    ----------------
    agent_name : str
        Must be overridden in every subclass.  Used as the ``agent_name``
        field in ``TrajectorySnapshot`` and as the structlog module tag.
    """

    agent_name: str = "base"

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings: AppSettings = settings or get_settings()
        self.logger = get_logger(f"agents.{self.agent_name}")

    # ------------------------------------------------------------------
    # LLM construction
    # ------------------------------------------------------------------

    def _build_llm(self, model_name: str) -> Any:
        """
        Build the appropriate LangChain chat model based on model name.

        Routing logic:
        - Model names starting with any ``_GROQ_PREFIXES`` → ``ChatGroq``
        - All other names → ``ChatOpenAI``

        This is the single code path that makes all four agents
        model-agnostic: swapping the model in ``.env`` changes the
        provider transparently.

        Parameters
        ----------
        model_name:
            Model identifier from ``GraphState.model_name`` (e.g.
            ``"gpt-4o"`` or ``"qwen/qwen3-8b"``).

        Returns
        -------
        BaseChatModel
            Configured LangChain chat model instance.
        """
        is_groq = any(model_name.lower().startswith(p) for p in _GROQ_PREFIXES)

        if is_groq:
            from langchain_groq import ChatGroq

            self.logger.debug(
                "llm_client_built",
                provider="groq",
                model=model_name,
            )
            return ChatGroq(
                model=model_name,
                api_key=self.settings.groq_api_key_value,
                temperature=0.1,
                timeout=self.settings.request_timeout,
            )
        else:
            from langchain_openai import ChatOpenAI

            self.logger.debug(
                "llm_client_built",
                provider="openai",
                model=model_name,
            )
            return ChatOpenAI(
                model=model_name,
                api_key=self.settings.openai_api_key_value,
                temperature=0.1,
                timeout=self.settings.request_timeout,
            )

    # ------------------------------------------------------------------
    # Tool invocation loop
    # ------------------------------------------------------------------

    def _invoke_with_tool_loop(
        self,
        llm: Any,
        tools: list[Any],
        messages: list[BaseMessage],
        max_iterations: int = 6,
    ) -> tuple[AIMessage, list[ToolCallRecord], dict[str, int]]:
        """
        Run the standard LangChain tool-calling cycle until the LLM
        produces a response with no tool calls (or ``max_iterations`` is
        reached).

        For every tool call the LLM makes, this method:
        1.  Invokes the tool function directly.
        2.  Appends a ``ToolCallRecord`` to the accumulator.
        3.  Feeds the ``ToolMessage`` result back to the LLM.

        All token counts across every LLM invocation in the loop are
        accumulated in ``total_tokens`` so the caller can compute an
        accurate total cost for the entire agent step.

        Parameters
        ----------
        llm:
            Bare LangChain chat model (without tools bound).
        tools:
            List of LangChain tool functions to bind.
        messages:
            Initial message list (system + human prompt).
        max_iterations:
            Maximum tool-call cycles before forcing a final response.

        Returns
        -------
        (final_response, tool_records, total_tokens)
            ``final_response``  — the last ``AIMessage`` with no tool calls.
            ``tool_records``    — ordered list of all ``ToolCallRecord``s.
            ``total_tokens``    — dict with ``"prompt"`` and ``"completion"`` totals.
        """
        tool_map: dict[str, Any] = {t.name: t for t in tools}
        all_tool_records: list[ToolCallRecord] = []
        total_tokens: dict[str, int] = {"prompt": 0, "completion": 0}

        llm_with_tools = llm.bind_tools(tools)

        current_messages = list(messages)

        for iteration in range(max_iterations):
            response: AIMessage = llm_with_tools.invoke(current_messages)

            # Accumulate token usage
            usage = getattr(response, "usage_metadata", None) or {}
            total_tokens["prompt"] += usage.get("input_tokens", 0)
            total_tokens["completion"] += usage.get("output_tokens", 0)

            # No tool calls → LLM is done; return the response
            if not getattr(response, "tool_calls", None):
                return response, all_tool_records, total_tokens

            self.logger.debug(
                "tool_calls_generated",
                agent=self.agent_name,
                iteration=iteration,
                tool_count=len(response.tool_calls),
            )

            # Append AI response to message history
            current_messages.append(response)

            # Process each tool call
            tool_messages: list[ToolMessage] = []
            for tc in response.tool_calls:
                tool_name: str = tc["name"]
                tool_args: dict[str, Any] = tc.get("args", {})
                tool_id: str = tc.get("id", str(uuid.uuid4()))

                tool_fn = tool_map.get(tool_name)
                t_start = time.perf_counter()
                success = True
                raw_result: Any = None
                error_type: str | None = None
                error_msg: str | None = None

                try:
                    if tool_fn is None:
                        raise ValueError(f"Tool '{tool_name}' not in registry.")
                    raw_result = tool_fn.invoke(tool_args)
                except Exception as exc:
                    success = False
                    error_type = type(exc).__name__
                    error_msg = str(exc)
                    raw_result = {"error": str(exc), "tool": tool_name}
                    self.logger.warning(
                        "tool_call_failed",
                        agent=self.agent_name,
                        tool=tool_name,
                        error=str(exc),
                    )

                latency_ms = (time.perf_counter() - t_start) * 1000.0

                all_tool_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments=tool_args,
                        raw_result=raw_result,
                        result_summary=str(raw_result)[:300] if raw_result else "",
                        success=success,
                        latency_ms=round(latency_ms, 2),
                        error_type=error_type,
                        error_message=error_msg,
                    )
                )

                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(raw_result, default=str),
                        tool_call_id=tool_id,
                    )
                )

                self.logger.info(
                    "tool_executed",
                    agent=self.agent_name,
                    tool=tool_name,
                    success=success,
                    latency_ms=round(latency_ms, 1),
                )

            current_messages.extend(tool_messages)

        # Max iterations reached — get final response without tool binding
        self.logger.warning(
            "tool_loop_max_iterations",
            agent=self.agent_name,
            max_iterations=max_iterations,
        )
        final_response: AIMessage = llm.invoke(current_messages)
        usage = getattr(final_response, "usage_metadata", None) or {}
        total_tokens["prompt"] += usage.get("input_tokens", 0)
        total_tokens["completion"] += usage.get("output_tokens", 0)
        return final_response, all_tool_records, total_tokens

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_cost(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Estimate the USD cost for a set of token counts.

        Uses ``_COST_PER_TOKEN`` pricing table.  Falls back to a
        conservative estimate (1 µUSD/token) for unknown models rather
        than raising an error.

        Parameters
        ----------
        model_name:
            Model identifier (must match a key in ``_COST_PER_TOKEN``).
        prompt_tokens:
            Number of input tokens consumed.
        completion_tokens:
            Number of output tokens generated.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        pricing = _COST_PER_TOKEN.get(
            model_name,
            {"prompt": 1.0e-6, "completion": 2.0e-6},
        )
        return (
            prompt_tokens * pricing["prompt"]
            + completion_tokens * pricing["completion"]
        )

    # ------------------------------------------------------------------
    # TrajectorySnapshot factory
    # ------------------------------------------------------------------

    def _make_snapshot(
        self,
        run_id: str,
        model_name: str,
        step_index: int,
        input_summary: dict[str, Any],
        output_summary: dict[str, Any],
        tool_records: list[ToolCallRecord],
        reasoning_trace: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        error_info: dict[str, str] | None = None,
        retry_attempt: int = 0,
    ) -> TrajectorySnapshot:
        """
        Assemble a complete ``TrajectorySnapshot`` for one agent execution.

        Parameters
        ----------
        run_id:
            The current graph run ID from ``GraphState``.
        model_name:
            Model used in this execution step.
        step_index:
            The ``step_count`` value from the state at entry to this node.
        input_summary:
            Key state fields read by this agent (serialised to dict).
        output_summary:
            Key state fields written by this agent (serialised to dict).
        tool_records:
            All ``ToolCallRecord`` objects from ``_invoke_with_tool_loop``.
        reasoning_trace:
            The LLM's content string (chain-of-thought).
        latency_ms:
            Wall-clock duration of the entire agent execution.
        prompt_tokens:
            Total prompt tokens across all LLM calls in this step.
        completion_tokens:
            Total completion tokens across all LLM calls in this step.
        error_info:
            If an exception occurred: ``{"type": ..., "message": ...}``.
        retry_attempt:
            Which retry attempt this snapshot represents (0 = first).

        Returns
        -------
        TrajectorySnapshot
        """
        cost = self._estimate_cost(model_name, prompt_tokens, completion_tokens)
        return TrajectorySnapshot(
            run_id=run_id,
            model_name=model_name,
            agent_name=self.agent_name,
            step_index=step_index,
            input_state_summary=input_summary,
            output_state_summary=output_summary,
            tool_calls=tool_records,
            reasoning_trace=reasoning_trace,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=round(cost, 8),
            error_occurred=error_info is not None,
            error_type=error_info.get("type") if error_info else None,
            error_message=error_info.get("message") if error_info else None,
            retry_attempt=retry_attempt,
        )

    # ------------------------------------------------------------------
    # JSON extraction utility
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> dict[str, Any]:
        """
        Extract a JSON object from an LLM response string.

        Tries three strategies in order:
        1.  Direct ``json.loads`` (response IS valid JSON).
        2.  Extract content between triple-backtick JSON fences.
        3.  Find the first ``{...}`` block via regex.

        Returns an empty dict on all failures so the agent can apply
        a graceful fallback rather than raising.

        Parameters
        ----------
        text:
            Raw LLM response string.

        Returns
        -------
        dict[str, Any]
            Parsed JSON object, or empty dict if parsing fails.
        """
        # Strategy 1: bare JSON
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: JSON inside ```json ... ``` fence
        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: first {...} block (greedy, outer braces)
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        self.logger.warning(
            "json_parse_failed",
            agent=self.agent_name,
            text_preview=text[:200],
        )
        return {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, state: GraphState) -> dict[str, Any]:
        """
        Execute this agent's domain logic and return a partial state update.

        Parameters
        ----------
        state:
            The current ``GraphState`` passed by LangGraph.

        Returns
        -------
        dict[str, Any]
            Partial state update — only keys that this agent modifies.
            LangGraph merges this with the existing state.

        Notes
        -----
        - Must increment ``step_count`` by 1.
        - Must append exactly one ``TrajectorySnapshot`` to ``trajectory``.
        - Must update ``total_tool_calls``, ``successful_tool_calls``,
          ``total_prompt_tokens``, ``total_completion_tokens``,
          ``total_cost_usd`` evaluation metadata fields.
        - Must update ``execution_status`` to reflect the new state.
        """
        ...

    def __call__(self, state: GraphState) -> dict[str, Any]:
        """
        LangGraph node interface.

        Allows any ``BaseAgent`` subclass to be registered directly as
        a graph node without a wrapper function:
            ``graph.add_node("planner", PlannerAgent())``

        Delegates entirely to ``self.run(state)``.
        """
        return self.run(state)
