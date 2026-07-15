"""
tools/_simulation.py — Controlled Failure Injection for Evaluation
===================================================================

Provides a ``FailureSimulator`` singleton that injects deterministic,
configurable failures into tool calls.

Purpose
-------
Trajectory-based evaluation requires *observable* error-recovery events.
Random failures are non-reproducible — you cannot compare GPT-4o vs.
Qwen3 on "how well does it recover?" if the failures differ between
runs.

``FailureSimulator`` solves this by:
1.  Reading a failure profile from environment variables at startup.
2.  Tracking which tool has been called how many times (per process).
3.  Raising ``SimulatedToolError`` on the *Nth* call to a specific tool,
    then allowing subsequent calls to succeed.

This produces a reproducible sequence:
    call 1  → raises SimulatedToolError
    call 2  → succeeds  (retry decorator catches call 1's error)

The agent's retry loop fires, the trajectory evaluator records an
``error_occurred=True`` snapshot followed by a success snapshot — a
clean, measurable error-recovery event.

Configuration (via environment variables)
-----------------------------------------
SIMULATE_FAILURES
    Set to ``true`` to enable failure injection.  Default: ``false``.
FAILURE_CARRIER_IDS
    Comma-separated carrier IDs whose capacity lookup always fails on
    the first attempt.  Default: ``CARRIER-FX``.
    Example: ``FAILURE_CARRIER_IDS=CARRIER-FX,CARRIER-DHL``
FAILURE_TOOL_NAMES
    Comma-separated tool names that fail on their first call per run.
    Default: empty (no tool-level failures unless carrier IDs matched).
    Example: ``FAILURE_TOOL_NAMES=list_available_carriers``

Disabling
---------
Leave ``SIMULATE_FAILURES`` unset or set to ``false`` for clean runs.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any


class SimulatedToolError(RuntimeError):
    """
    Exception raised by ``FailureSimulator`` to simulate a tool failure.

    Inherits from ``RuntimeError`` so that the retry decorator's
    ``DEFAULT_RETRYABLE_EXCEPTIONS`` (which includes ``ConnectionError``
    and ``TimeoutError``) does NOT catch it by default.

    To enable retry on simulated failures, pass
    ``extra_exceptions=(SimulatedToolError,)`` to ``make_retry_decorator``.
    This is intentional: it forces the agent code to *explicitly* opt in
    to retrying simulated errors, making the retry policy visible.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"[SIMULATED] Tool '{tool_name}' failure: {reason}")


class FailureSimulator:
    """
    Singleton that tracks per-tool call counts and injects failures.

    Thread-safety note: ``_call_counts`` is a ``defaultdict(int)`` and
    integer increment is the GIL-protected in CPython.  This is
    sufficient for the single-threaded LangGraph execution model.

    Do not use in multi-threaded production code without a lock.
    """

    def __init__(self) -> None:
        self._enabled: bool = (
            os.getenv("SIMULATE_FAILURES", "false").lower() == "true"
        )
        self._failure_carriers: set[str] = {
            c.strip().upper()
            for c in os.getenv("FAILURE_CARRIER_IDS", "CARRIER-FX").split(",")
            if c.strip()
        }
        self._failure_tools: set[str] = {
            t.strip().lower()
            for t in os.getenv("FAILURE_TOOL_NAMES", "").split(",")
            if t.strip()
        }
        # Per-tool call counts: {"tool_name": count}
        self._call_counts: dict[str, int] = defaultdict(int)

    @property
    def enabled(self) -> bool:
        """True when failure simulation is active."""
        return self._enabled

    def reset(self) -> None:
        """
        Reset all call counters.

        Call this between evaluation scenarios to ensure each scenario
        starts with a clean failure-injection state.
        """
        self._call_counts.clear()

    def maybe_raise(self, tool_name: str, context: str | None = None) -> None:
        """
        Raise ``SimulatedToolError`` if failure conditions are met.

        Conditions checked (in order):
        1.  Simulation is enabled.
        2.  The tool is in ``_failure_tools`` AND this is its first call.
        3.  The ``context`` (e.g., a carrier ID) is in ``_failure_carriers``
            AND the tool's call count is exactly 1 (first call).

        After raising, the call counter is incremented so that the
        *next* call (the retry) succeeds.

        Parameters
        ----------
        tool_name:
            Name of the tool being called.
        context:
            Optional context string used for carrier-level failures.
            Pass the carrier ID when calling carrier-specific tools.
        """
        if not self._enabled:
            return

        tool_key = f"{tool_name}:{context or '_'}"
        self._call_counts[tool_key] += 1
        call_number = self._call_counts[tool_key]

        # Tool-level failure: first call to any listed tool fails
        if tool_name.lower() in self._failure_tools and call_number == 1:
            raise SimulatedToolError(
                tool_name=tool_name,
                reason=(
                    f"Simulated network timeout on first call "
                    f"(configured via FAILURE_TOOL_NAMES)."
                ),
            )

        # Carrier-level failure: first lookup of a specific carrier fails
        if (
            context
            and context.upper() in self._failure_carriers
            and call_number == 1
        ):
            raise SimulatedToolError(
                tool_name=tool_name,
                reason=(
                    f"Carrier '{context}' API returned 503 Service Unavailable "
                    f"(simulated for error-recovery evaluation)."
                ),
            )

    def get_call_counts(self) -> dict[str, int]:
        """Return a copy of the current call count registry."""
        return dict(self._call_counts)
