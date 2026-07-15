"""
graphs/state.py — Typed State Schema for the Logistics Rerouting Graph
=======================================================================

This module defines two categories of types:

1.  **Domain models** (Pydantic BaseModel subclasses):
        ShipmentAlert, CarrierOption, SelectedRoute,
        ExecutionResult, ToolCallRecord, TrajectorySnapshot

    These represent the structured data that flows *between* agents.
    Pydantic gives us runtime validation, rich field descriptions, and
    ``.model_dump()`` / ``.model_json_schema()`` for free.

2.  **Graph state** (TypedDict):
        GraphState

    LangGraph requires a TypedDict as the state container.  Fields
    annotated with ``Annotated[list[T], operator.add]`` use *append*
    merge semantics — each node contributes to the list without
    overwriting previous entries.  All other fields use *replace*
    semantics — the most recent write wins.

Merge Semantics Reference
--------------------------
+---------------------------+--------------------+------------------------------+
| Field                     | Merge strategy     | Reason                       |
+---------------------------+--------------------+------------------------------+
| trajectory                | append (add)       | Every node appends a snapshot|
| carrier_options           | append (add)       | Research accumulates options |
| messages                  | replace            | Full message history tracked |
| selected_route            | replace            | Decision agent picks one     |
| execution_result          | replace            | Execution agent sets outcome |
| execution_status          | replace            | Status machine, single value |
| retry_count               | replace            | Counter, managed by graph    |
+---------------------------+--------------------+------------------------------+

Design principle: every field that could be useful for trajectory-based
evaluation is *typed* rather than stored as a raw dict or string.
This makes scoring logic in ``evaluation/trajectory_evaluator.py``
straightforward and refactor-safe.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# Enumerations (as Literal types — no enum class needed for Pydantic v2)
# ===========================================================================

DelayReason = Literal[
    "WEATHER",
    "PORT_CLOSURE",
    "CARRIER_FAILURE",
    "CUSTOMS_HOLD",
    "MECHANICAL_FAILURE",
    "STRIKE",
    "CAPACITY_CONSTRAINT",
    "ROUTE_DISRUPTION",
    "UNKNOWN",
]

CargoType = Literal["STANDARD", "FRAGILE", "HAZMAT", "PERISHABLE", "HIGH_VALUE"]

Priority = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

TransitMode = Literal["AIR", "SEA", "RAIL", "ROAD", "MULTIMODAL"]

ExecutionStatus = Literal[
    "pending",          # Initial state — no agent has run yet
    "planning",         # Planner agent is active
    "researching",      # Research agent is querying carrier tools
    "deciding",         # Decision agent is evaluating options
    "executing",        # Execution agent is submitting the reroute
    "completed",        # Reroute successfully executed
    "failed",           # All retry attempts exhausted
    "escalated",        # Escalated to human operator
]


# ===========================================================================
# Domain Model 1: ShipmentAlert
# ===========================================================================


class ShipmentAlert(BaseModel):
    """
    Represents an inbound telemetry alert triggered by a shipment delay
    or disruption event.

    This is the *entry point* for the entire multi-agent workflow.  The
    Planner Agent receives a raw ``dict`` from the telemetry stream and
    validates it into this model as its first action.

    Fields are designed to give the Research Agent enough context to
    query carrier alternatives without any additional lookup — all
    constraints are embedded here.
    """

    shipment_id: str = Field(
        description=(
            "Globally unique identifier for the shipment.  "
            "Format: 'SHP-YYYYMMDD-NNNN'.  Used as the correlation ID "
            "across all log events and trajectory snapshots for this run."
        )
    )
    origin: str = Field(
        description="Origin port or warehouse code (e.g., 'CNSHA' for Shanghai).",
    )
    destination: str = Field(
        description="Destination port or warehouse code (e.g., 'USLAX' for Los Angeles).",
    )
    current_carrier: str = Field(
        description="Name of the carrier currently assigned to this shipment.",
    )
    current_route_id: str = Field(
        description="The route identifier currently assigned (e.g., 'CNSHA-USLAX-001').",
    )
    delay_reason: DelayReason = Field(
        description=(
            "Structured delay reason code from the telemetry system.  "
            "Drives the Research Agent's carrier filter logic — "
            "e.g., WEATHER delays exclude carriers on the same lane."
        ),
    )
    delay_hours: float = Field(
        gt=0.0,
        description=(
            "Estimated delay in hours beyond the original ETA.  "
            "Used by the Decision Agent to evaluate whether rerouting "
            "is cost-effective vs. absorbing the delay."
        ),
    )
    cargo_type: CargoType = Field(
        description=(
            "Nature of the cargo.  Constrains carrier selection: "
            "HAZMAT requires certified carriers; PERISHABLE requires "
            "air or refrigerated transit; HIGH_VALUE requires insured carriers."
        ),
    )
    cargo_value_usd: float = Field(
        gt=0.0,
        description=(
            "Declared value of the cargo in USD.  Informs the Decision "
            "Agent's cost-benefit calculation: rerouting cost must not "
            "exceed a configurable percentage of cargo value."
        ),
    )
    weight_kg: float = Field(
        gt=0.0,
        description="Total shipment weight in kilograms.  Used for carrier capacity matching.",
    )
    volume_cbm: float = Field(
        gt=0.0,
        description="Total shipment volume in cubic meters.  Used for carrier capacity matching.",
    )
    deadline_utc: datetime = Field(
        description=(
            "Hard delivery deadline in UTC.  The Decision Agent rejects "
            "any carrier option whose ETA exceeds this deadline."
        ),
    )
    priority: Priority = Field(
        default="MEDIUM",
        description=(
            "Business priority of this shipment.  CRITICAL shipments "
            "trigger immediate escalation if no valid reroute is found. "
            "LOW priority shipments may absorb delays without rerouting."
        ),
    )
    customer_id: str = Field(
        description="Customer identifier for notification routing.",
    )
    alert_timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this alert was generated by the telemetry system.",
    )

    @field_validator("shipment_id")
    @classmethod
    def validate_shipment_id_format(cls, value: str) -> str:
        """Ensure the shipment ID follows the expected format."""
        value = value.strip()
        if not value:
            raise ValueError("shipment_id must not be empty.")
        return value

    @field_validator("deadline_utc", mode="before")
    @classmethod
    def ensure_utc_aware(cls, value: Any) -> datetime:
        """
        Ensure the deadline datetime is timezone-aware (UTC).

        LangGraph serialises state to JSON between nodes; naive datetimes
        lose timezone information during round-trips.  We enforce UTC
        awareness at validation time to prevent silent bugs.
        """
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            dt = value
        else:
            raise ValueError(f"Cannot parse deadline_utc from {type(value)}")
        if dt.tzinfo is None:
            import warnings
            warnings.warn(
                f"deadline_utc '{dt}' has no timezone — assuming UTC.",
                stacklevel=2,
            )
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @property
    def hours_to_deadline(self) -> float:
        """Hours remaining until the delivery deadline from now (UTC)."""
        now = datetime.now(timezone.utc)
        delta = self.deadline_utc - now
        return delta.total_seconds() / 3600.0


# ===========================================================================
# Domain Model 2: ToolCallRecord
# ===========================================================================


class ToolCallRecord(BaseModel):
    """
    Records a single tool invocation made by an agent node.

    Every agent that calls a LangChain tool appends one of these to
    the ``TrajectorySnapshot`` for that step.  The Trajectory Evaluator
    uses these records to compute tool-calling accuracy: whether the
    right tool was called, with valid arguments, and whether it returned
    a useful result.

    This is a first-class evaluation artefact — not a debug log.
    """

    tool_name: str = Field(
        description="Name of the LangChain tool that was invoked.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arguments passed to the tool, as a key-value dict.  "
            "Stored verbatim for post-hoc correctness analysis."
        ),
    )
    raw_result: Any = Field(
        default=None,
        description="The raw value returned by the tool function.",
    )
    result_summary: str = Field(
        default="",
        description=(
            "A human-readable one-line summary of the result.  "
            "Generated by the agent after receiving the tool output."
        ),
    )
    success: bool = Field(
        description="True if the tool returned a valid result without raising an exception.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock latency of the tool call in milliseconds.",
    )
    retry_attempt: int = Field(
        default=0,
        ge=0,
        description=(
            "Which retry attempt this record represents (0 = first attempt). "
            "A value > 0 indicates the retry decorator fired."
        ),
    )
    error_type: str | None = Field(
        default=None,
        description="Exception class name if the tool raised an error; None otherwise.",
    )
    error_message: str | None = Field(
        default=None,
        description="Exception message if the tool raised an error; None otherwise.",
    )


# ===========================================================================
# Domain Model 3: TrajectorySnapshot
# ===========================================================================


class TrajectorySnapshot(BaseModel):
    """
    A complete record of one agent node execution within the graph.

    A ``TrajectorySnapshot`` is captured *at the boundary of each node*:
    once just before the node runs (input state) and once just after
    (output state).  The collection of all snapshots for a single graph
    run forms the **trajectory** — the unit of analysis in the
    trajectory-based evaluation framework.

    Why snapshots rather than just the final output?
    -------------------------------------------------
    Final-output evaluation misses intermediate failures: a model that
    calls the wrong tool three times before succeeding scores identically
    to one that calls the right tool on the first attempt.  Trajectory
    evaluation captures *how* the system reached its answer, not just
    *what* the answer was.

    Evaluation dimensions captured by this model:
    - Reasoning quality  → ``reasoning_trace``
    - Tool-calling accuracy → ``tool_calls``
    - Error recovery → ``error_occurred`` + ``retry_attempt``
    - Latency → ``latency_ms``
    - Cost → ``estimated_cost_usd``
    - Output quality → scored externally in ``TrajectoryEvaluator``
    """

    snapshot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this snapshot.  Auto-generated.",
    )
    run_id: str = Field(
        description=(
            "Identifier for the complete graph execution this snapshot "
            "belongs to.  All snapshots from one ``graph.invoke()`` call "
            "share the same ``run_id``."
        ),
    )
    model_name: str = Field(
        description="The LLM model name used by the agent in this step.",
    )
    agent_name: str = Field(
        description=(
            "Name of the agent node that produced this snapshot "
            "(e.g., 'planner', 'research', 'decision', 'execution')."
        ),
    )
    step_index: int = Field(
        ge=0,
        description=(
            "Zero-based index of this node execution within the run.  "
            "The sequence of ``step_index`` values reconstructs the "
            "exact execution path through the graph."
        ),
    )
    input_state_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Serialised summary of the relevant GraphState fields "
            "passed *into* this node.  Not the full state (too large) "
            "— just the fields this agent reads."
        ),
    )
    output_state_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Serialised summary of the GraphState fields *produced* by "
            "this node.  Together with ``input_state_summary``, this "
            "shows exactly what each agent contributed."
        ),
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Ordered list of all tool invocations made during this node execution.",
    )
    reasoning_trace: str = Field(
        default="",
        description=(
            "The LLM's chain-of-thought reasoning for this step, "
            "extracted from the model response.  Used to score "
            "reasoning quality in the evaluator."
        ),
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of this entire node execution in milliseconds.",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the prompt sent to the LLM for this step.",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens in the LLM completion for this step.",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (prompt + completion).",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Estimated API cost in USD for this step, computed from "
            "token counts and the model's published pricing."
        ),
    )
    error_occurred: bool = Field(
        default=False,
        description="True if this node raised an exception or entered a retry loop.",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception class name if an error occurred; None otherwise.",
    )
    error_message: str | None = Field(
        default=None,
        description="Exception message if an error occurred; None otherwise.",
    )
    retry_attempt: int = Field(
        default=0,
        ge=0,
        description=(
            "Which retry attempt this snapshot represents (0 = first attempt). "
            "Snapshots are recorded on every attempt, so a node that retries "
            "twice will produce three snapshots with the same ``agent_name`` "
            "and ``step_index`` but increasing ``retry_attempt`` values."
        ),
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this snapshot was captured.",
    )

    @model_validator(mode="after")
    def compute_total_tokens(self) -> "TrajectorySnapshot":
        """Auto-compute total_tokens if prompt + completion are set but total is 0."""
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(
                self, "total_tokens", self.prompt_tokens + self.completion_tokens
            )
        return self

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls made in this step."""
        return len(self.tool_calls)

    @property
    def successful_tool_calls(self) -> int:
        """Number of tool calls that returned a successful result."""
        return sum(1 for tc in self.tool_calls if tc.success)

    @property
    def tool_call_accuracy(self) -> float:
        """
        Fraction of tool calls that succeeded (0.0 – 1.0).

        Returns 1.0 if no tool calls were made (the step was purely
        reasoning with no tool use — this is not penalised).
        """
        if not self.tool_calls:
            return 1.0
        return self.successful_tool_calls / self.tool_call_count


# ===========================================================================
# Domain Model 4: CarrierOption
# ===========================================================================


class CarrierOption(BaseModel):
    """
    A single carrier + route alternative discovered by the Research Agent.

    The Research Agent populates a list of ``CarrierOption`` objects by
    calling carrier lookup and route calculation tools.  The Decision
    Agent then receives this list and selects the optimal option.

    Every option records *why* it may be rejected (``rejection_reason``)
    even when ``constraints_satisfied=False``.  This transparency lets
    the Decision Agent explain its choice to stakeholders and provides
    the evaluator with evidence that the Research Agent explored the
    full option space before settling.
    """

    carrier_id: str = Field(
        description="Internal carrier identifier (e.g., 'CARRIER-FX', 'CARRIER-UPS').",
    )
    carrier_name: str = Field(
        description="Human-readable carrier name (e.g., 'FedEx Freight').",
    )
    route_id: str = Field(
        description=(
            "Unique route identifier combining origin, destination, and "
            "carrier code (e.g., 'CNSHA-USLAX-FX-AIR-01')."
        ),
    )
    origin: str = Field(description="Route origin port/warehouse code.")
    destination: str = Field(description="Route destination port/warehouse code.")
    transit_mode: TransitMode = Field(
        description="Primary mode of transport for this route.",
    )
    eta_hours: float = Field(
        gt=0.0,
        description="Estimated transit time in hours from origin to destination.",
    )
    eta_delta_hours: float = Field(
        description=(
            "Difference between this route's ETA and the original "
            "shipment's deadline (negative = ahead of deadline, "
            "positive = behind deadline)."
        ),
    )
    cost_usd: float = Field(
        ge=0.0,
        description="Total quoted cost for this carrier option in USD.",
    )
    cost_delta_usd: float = Field(
        description=(
            "Cost difference vs. original shipment cost "
            "(positive = more expensive, negative = cheaper)."
        ),
    )
    capacity_available: bool = Field(
        description="Whether this carrier has confirmed capacity for the shipment dimensions.",
    )
    reliability_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Historical on-time delivery rate for this carrier on this lane "
            "(0.0 = never on time, 1.0 = always on time).  "
            "Sourced from simulated carrier performance data."
        ),
    )
    cargo_type_supported: bool = Field(
        description="Whether this carrier is certified to carry the shipment's cargo type.",
    )
    constraints_satisfied: bool = Field(
        description=(
            "True only if this option satisfies ALL hard constraints: "
            "capacity, cargo type certification, deadline, and any "
            "regulatory requirements."
        ),
    )
    rejection_reason: str | None = Field(
        default=None,
        description=(
            "If ``constraints_satisfied=False``, a brief explanation of "
            "which constraint failed.  None when the option is viable."
        ),
    )

    @property
    def is_viable(self) -> bool:
        """An option is viable if constraints are satisfied and capacity is available."""
        return self.constraints_satisfied and self.capacity_available

    @property
    def score(self) -> float:
        """
        Composite viability score for ranking options.

        Score = reliability_score - (eta_delta_hours * 0.05) - (cost_delta_usd * 0.0001)

        The weights encode the business priority: reliability > time > cost.
        These weights are intentionally simple — the Decision Agent
        (LLM) provides nuanced judgement on top of this numeric ranking.
        """
        if not self.is_viable:
            return -1.0
        return (
            self.reliability_score
            - (max(0.0, self.eta_delta_hours) * 0.05)
            - (max(0.0, self.cost_delta_usd) * 0.0001)
        )


# ===========================================================================
# Domain Model 5: SelectedRoute
# ===========================================================================


class SelectedRoute(BaseModel):
    """
    The carrier + route selected by the Decision Agent.

    Produced by the Decision Agent after evaluating all ``CarrierOption``
    objects returned by the Research Agent.  Contains not only the
    chosen route but also the LLM's *reasoning* for the choice.  This
    reasoning is a primary input to the trajectory evaluator's reasoning
    quality score.

    The ``confidence_score`` reflects the Decision Agent's self-assessed
    certainty.  Low confidence (< 0.6) triggers a flag in the evaluator
    and may prompt the Execution Agent to seek human confirmation.
    """

    carrier_id: str = Field(description="Carrier identifier of the selected route.")
    carrier: str = Field(description="Human-readable carrier name.")
    route_id: str = Field(description="Route identifier of the selected option.")
    transit_mode: TransitMode = Field(description="Transit mode of the selected route.")
    estimated_cost_usd: float = Field(
        ge=0.0,
        description="Total estimated cost for the reroute in USD.",
    )
    cost_delta_usd: float = Field(
        description="Cost increase (positive) or saving (negative) vs. original route.",
    )
    eta_hours: float = Field(
        gt=0.0,
        description="Estimated transit time in hours.",
    )
    eta_delta_hours: float = Field(
        description="ETA vs. original deadline (negative = before deadline).",
    )
    reliability_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Historical reliability score of the selected carrier on this lane.",
    )
    decision_rationale: str = Field(
        description=(
            "Natural-language explanation of why this option was chosen "
            "over alternatives.  Written by the Decision Agent LLM.  "
            "Evaluated for logical coherence in the trajectory evaluator."
        ),
    )
    alternatives_considered: int = Field(
        ge=0,
        description="Number of CarrierOption objects evaluated before this selection.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Decision Agent's self-assessed confidence in this selection "
            "(0.0 = uncertain, 1.0 = highly confident).  "
            "Below 0.6 triggers an escalation flag."
        ),
    )
    requires_human_approval: bool = Field(
        default=False,
        description=(
            "Set to True by the Decision Agent when confidence is low or "
            "cost delta exceeds a configurable threshold.  The Execution "
            "Agent will send an approval request instead of executing."
        ),
    )
    selected_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this selection was made.",
    )


# ===========================================================================
# Domain Model 6: ExecutionResult
# ===========================================================================


class ExecutionResult(BaseModel):
    """
    The outcome of the Execution Agent's reroute submission.

    Produced by the Execution Agent after calling the reroute execution
    tool and the notification dispatch tool.  A failed result (
    ``success=False``) triggers the graph's retry edge back to the
    Research Agent for a new set of carrier options.

    ``confirmation_number`` is the external reference returned by the
    carrier's booking system.  In the simulated environment, this is a
    generated identifier.  In a real system, this would be a carrier
    booking confirmation number used for tracking.
    """

    success: bool = Field(
        description="True if the reroute was successfully submitted to the carrier.",
    )
    confirmation_number: str | None = Field(
        default=None,
        description=(
            "Carrier booking confirmation number.  Only present when "
            "``success=True``."
        ),
    )
    carrier: str = Field(description="Name of the carrier the reroute was submitted to.")
    route_id: str = Field(description="Route ID that was booked.")
    final_cost_usd: float = Field(
        ge=0.0,
        description="Final confirmed cost of the reroute in USD.",
    )
    final_eta_hours: float = Field(
        gt=0.0,
        description="Final confirmed transit time in hours.",
    )
    notification_sent: bool = Field(
        default=False,
        description="True if stakeholder notifications were successfully dispatched.",
    )
    notification_recipients: list[str] = Field(
        default_factory=list,
        description=(
            "List of notification recipients (email addresses or user IDs) "
            "that were alerted about the reroute."
        ),
    )
    executed_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the reroute was executed.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of execution attempts made before this result.  "
            "A value > 0 means the graph traversed the retry edge at "
            "least once, which is recorded as an error recovery event "
            "in the trajectory evaluator."
        ),
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if ``success=False``; None otherwise.",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception class name if execution failed; None otherwise.",
    )


# ===========================================================================
# Graph State  (LangGraph TypedDict)
# ===========================================================================

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8 fallback (not expected)
    from typing_extensions import TypedDict  # type: ignore[assignment]


class GraphState(TypedDict, total=False):
    """
    The single source of truth for all data flowing through the
    LangGraph StateGraph.

    LangGraph merges state between nodes using the field annotations:
    - ``Annotated[list[T], operator.add]``  → append (list accumulates)
    - Plain type annotations              → replace (last write wins)

    Field groupings
    ---------------
    ① Run context   — identifies this run and the model being used
    ② Input         — the inbound telemetry alert
    ③ Planning      — the Planner Agent's structured plan and constraints
    ④ Research      — carrier options discovered by the Research Agent
    ⑤ Decision      — the route selected by the Decision Agent
    ⑥ Execution     — the outcome from the Execution Agent
    ⑦ Control flow  — retry counter, error messages, loop guards
    ⑧ Trajectory    — append-only log of TrajectorySnapshots
    ⑨ Evaluation    — aggregate metrics updated as the graph runs
    """

    # ① Run context
    # -----------------------------------------------------------------------
    run_id: str
    """
    UUID identifying this complete graph execution.  All TrajectorySnapshots
    and log events for one ``graph.invoke()`` call share this ID.
    Set by the graph's entry point, never mutated.
    """

    model_name: str
    """
    The LLM model identifier used by every agent in this run.
    Injected at graph construction time from AppSettings.
    Stored in state so that trajectory snapshots can record it without
    needing access to the settings singleton.
    """

    # ② Input
    # -----------------------------------------------------------------------
    telemetry_alert: ShipmentAlert
    """
    The inbound shipment delay alert.  Parsed and validated by the
    Planner Agent.  Read-only after the Planner step — no downstream
    agent modifies it.
    """

    # ③ Planning
    # -----------------------------------------------------------------------
    rerouting_plan: str
    """
    The Planner Agent's structured rerouting strategy, expressed as a
    natural-language + structured text plan.  Read by the Research Agent
    to understand what constraints to apply when querying carriers.
    Example: 'Priority: CRITICAL. Require AIR transit. Max cost delta: $5000.
    Exclude current carrier lane (CNSHA-USLAX). Deadline: 48h.'
    """

    rerouting_constraints: dict[str, Any]
    """
    Machine-readable constraints extracted from the rerouting plan.
    Structured as a dict so the Research Agent can programmatically
    filter carrier options without re-parsing the plan text.
    Keys: 'max_cost_delta_usd', 'required_transit_modes',
          'excluded_carriers', 'deadline_utc', 'min_reliability_score'.
    """

    # ④ Research
    # -----------------------------------------------------------------------
    carrier_options: Annotated[list[CarrierOption], operator.add]
    """
    Append-only list of carrier alternatives discovered by the Research Agent.
    Uses operator.add so that each Research Agent execution (including
    retries) appends new options without overwriting earlier ones.
    The Decision Agent receives the full accumulated list.
    """

    research_summary: str
    """
    A brief natural-language summary of what the Research Agent found:
    how many options were queried, how many were viable, and any
    noteworthy findings (e.g., 'No AIR options available; falling back
    to MULTIMODAL').  Used in the final execution report.
    """

    # ⑤ Decision
    # -----------------------------------------------------------------------
    selected_route: SelectedRoute | None
    """
    The carrier + route chosen by the Decision Agent.
    None until the Decision Agent has run.  Replace semantics — the
    Decision Agent overwrites this field if it is re-invoked after a
    failed execution attempt.
    """

    decision_summary: str
    """
    A brief summary of the Decision Agent's evaluation: number of
    alternatives assessed, the chosen option's key attributes, and
    the rationale headline.  Included verbatim in the final report.
    """

    # ⑥ Execution
    # -----------------------------------------------------------------------
    execution_result: ExecutionResult | None
    """
    The outcome of the Execution Agent's reroute submission.
    None until the Execution Agent has run.  Replace semantics.
    """

    execution_status: ExecutionStatus
    """
    Current status of the rerouting workflow.  Acts as the graph's
    state machine signal:
        pending → planning → researching → deciding → executing →
        completed | failed | escalated
    The graph's conditional edge functions read this field to determine
    the next node.
    """

    # ⑦ Control flow
    # -----------------------------------------------------------------------
    retry_count: int
    """
    Number of times the graph has re-entered the Research or Execution
    nodes due to failures.  Incremented by the graph's retry edge logic.
    When retry_count >= max_retries, the graph transitions to 'failed'.
    """

    error_message: str | None
    """
    Human-readable description of the most recent error encountered.
    Populated by any agent node that catches an exception.
    Cleared (set to None) when the error is resolved via retry.
    """

    error_history: Annotated[list[str], operator.add]
    """
    Append-only log of all error messages encountered during this run.
    Unlike error_message (which reflects only the latest error),
    error_history accumulates every failure for post-hoc analysis.
    The trajectory evaluator uses error_history to compute the
    error recovery rate.
    """

    should_escalate: bool
    """
    Set to True by the Decision Agent when confidence is low, or by the
    Execution Agent when all retries are exhausted on a CRITICAL shipment.
    The graph's conditional edge transitions to an escalation notification
    rather than a failure state.
    """

    step_count: int
    """
    Total number of node executions in this run.  Compared against
    AppSettings.max_graph_steps to detect runaway graphs.  Incremented
    at the start of every agent node.
    """

    # ⑧ Trajectory  (append-only — the evaluation backbone)
    # -----------------------------------------------------------------------
    trajectory: Annotated[list[TrajectorySnapshot], operator.add]
    """
    Ordered, append-only list of TrajectorySnapshot objects.
    One snapshot is appended per agent node execution (including retries).
    This list is the *primary input* to the trajectory-based evaluator —
    it records the full intermediate reasoning, tool calls, tokens,
    latency, and cost at every step of the graph execution.

    Uses operator.add for merge semantics so that snapshots from
    parallel branches (if any) are correctly accumulated.
    """

    # ⑨ Evaluation metadata  (updated incrementally by each agent)
    # -----------------------------------------------------------------------
    total_tool_calls: int
    """Running total of tool calls made across all agent nodes in this run."""

    successful_tool_calls: int
    """Running total of tool calls that returned a successful result."""

    total_prompt_tokens: int
    """Cumulative prompt token count across all LLM calls in this run."""

    total_completion_tokens: int
    """Cumulative completion token count across all LLM calls in this run."""

    total_cost_usd: float
    """Cumulative estimated API cost in USD across all LLM calls in this run."""

    start_time_utc: str
    """
    ISO 8601 UTC timestamp when this graph run started.
    Stored as a string for safe JSON serialisation across LangGraph
    node boundaries.  The evaluator parses it back to a datetime.
    """


# ===========================================================================
# Default telemetry scenario (used when no --scenario flag is passed)
# ===========================================================================

DEFAULT_SCENARIO: dict[str, Any] = {
    "shipment_id": "SHP-20240715-0042",
    "origin": "CNSHA",
    "destination": "USLAX",
    "current_carrier": "Ocean Pacific Freight",
    "current_route_id": "CNSHA-USLAX-OPF-SEA-01",
    "delay_reason": "PORT_CLOSURE",
    "delay_hours": 96.0,
    "cargo_type": "HIGH_VALUE",
    "cargo_value_usd": 2_500_000.0,
    "weight_kg": 4_800.0,
    "volume_cbm": 32.0,
    "deadline_utc": "2024-07-20T18:00:00+00:00",
    "priority": "CRITICAL",
    "customer_id": "CUST-ACME-001",
    "alert_timestamp_utc": "2024-07-15T08:00:00+00:00",
}
