"""
graphs/logistics_graph.py — LangGraph Orchestration
=====================================================

Assembles the four agents into a production-quality ``StateGraph`` that
orchestrates the full autonomous logistics rerouting workflow.

Graph Topology (ASCII)
-----------------------
::

    ┌────────┐
    │ START  │
    └───┬────┘
        │  (always)
    ┌───▼────────┐
    │  planner   │  Planner Agent — constraint extraction, no tools
    └───┬────────┘
        │  (always)
    ┌───▼────────────────────────────────────────────────────────┐
    │                      research                              │
    │           Research Agent — discovers carrier options       │
    └───┬──────────────────────────────────────────┬────────────┘
        │  viable options found                    │  no viable options
        │                              ┌───────────▼──────────────┐
        │                              │    retry_increment        │
        │                              │  (increments retry_count) │
        │                              └───────────┬──────────────┘
        │                           retry: ────────┘  fail: ──────────► END
    ┌───▼────────────────┐
    │     decision        │  Decision Agent — selects optimal route
    └───┬────────────────┘
        │  route selected           escalate: ──────────────────────────────┐
        │                           retry:  ── retry_increment ─► research  │
        │                           fail:   ───────────────────────────► END│
    ┌───▼────────────────┐                                                  │
    │     execution       │  Execution Agent — executes & notifies          │
    └───┬────────────────┘                                                  │
        │  success         retry: ── retry_increment ─► research            │
        │  fail: ──────────────────────────────────────────────────────► END│
    ┌───▼────────┐        ┌───────────────────────────────────────────────┐│
    │    END     │        │            escalation_handler                 ││
    │ (completed)│        │  (logs alert, sets status="escalated")        ││
    └────────────┘        └───────────────────────────────────────────────┘│
                                            │                              ││
                                            └──────────────────────────────┘▼
                                                              END (escalated)

Node Responsibilities
---------------------
``planner``
    PlannerAgent — reads ShipmentAlert, writes rerouting_plan + constraints.
    Always transitions to ``research``.

``research``
    ResearchAgent — calls RESEARCH_TOOLS, writes carrier_options list.
    Transitions to ``decision`` if viable options exist.
    Transitions to ``retry_increment`` if no viable options and retries remain.
    Transitions to END with "failed" status if max retries exceeded.

``retry_increment``
    Lightweight auxiliary node.  Increments ``retry_count`` and resets
    ``execution_status`` to "researching".  Always transitions back to
    ``research``.  This separation keeps routing functions pure (no side
    effects) and makes the retry path explicit in the graph topology.

``decision``
    DecisionAgent — validates top candidates, writes selected_route.
    Transitions to ``execution`` if a route is selected.
    Transitions to ``escalation_handler`` if should_escalate=True.
    Transitions to ``retry_increment`` if no route selected and retries remain.
    Transitions to END with "failed" if max retries exceeded.

``execution``
    ExecutionAgent — submits reroute, sends notifications, generates report.
    Transitions to END with "completed" if execution_result.success=True.
    Transitions to ``retry_increment`` if execution failed and retries remain.
    Transitions to END with "failed" if max retries exceeded.

``escalation_handler``
    Sets execution_status="escalated" and logs a human-operator alert.
    Always transitions to END.

Retry Behaviour
---------------
``retry_count`` tracks how many times the research→decision→execution cycle
has been attempted.  The ``MAX_RETRIES`` setting (default: 3) controls the
ceiling.  On each retry:
1.  ``retry_increment`` node fires — increments retry_count.
2.  Control returns to ``research`` — which re-queries carriers.
3.  If a new viable option is found, the cycle continues normally.

Escalation Behaviour
--------------------
Escalation is triggered when the Decision Agent sets ``should_escalate=True``.
This happens when:
- ``confidence_score < 0.65`` — Decision Agent is uncertain
- ``requires_human_approval=True`` — business rule requires human sign-off
- No viable carriers exist after research

On escalation:
- ``escalation_handler`` node runs (sets status, logs the alert)
- Graph terminates at END with ``execution_status="escalated"``
- No retries are attempted — escalation is a terminal state

Max-Step Guard
--------------
``route_after_research`` and ``route_after_decision`` check ``step_count``
against ``MAX_GRAPH_STEPS`` (a hard ceiling of 20 steps) to prevent any
unforeseen infinite loop.  This is a defence-in-depth measure on top of
the ``MAX_RETRIES`` ceiling.

Factory Function
----------------
``build_logistics_graph(model_name, settings, use_checkpointer)``
    Returns a compiled LangGraph graph ready for ``.invoke()`` / ``.stream()``.

``create_initial_state(alert, model_name)``
    Returns a properly initialized GraphState dict for a new run.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agents import DecisionAgent, ExecutionAgent, PlannerAgent, ResearchAgent
from config.settings import AppSettings, get_settings
from graphs.state import DEFAULT_SCENARIO, GraphState, ShipmentAlert
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants (prevents typo bugs; single source of truth)
# ---------------------------------------------------------------------------

_NODE_PLANNER    = "planner"
_NODE_RESEARCH   = "research"
_NODE_DECISION   = "decision"
_NODE_EXECUTION  = "execution"
_NODE_RETRY      = "retry_increment"
_NODE_ESCALATION = "escalation_handler"

# Hard ceiling on total graph steps (defence-in-depth, beyond MAX_RETRIES)
_MAX_GRAPH_STEPS = 20

# ---------------------------------------------------------------------------
# Auxiliary node functions
# ---------------------------------------------------------------------------


def _retry_increment_node(state: GraphState) -> dict[str, Any]:
    """
    Lightweight retry management node.

    Why this is a separate node (not inline in routing functions):
    - Routing functions must be pure (no state side effects).
    - Incrementing retry_count IS a state side effect.
    - A dedicated node makes the retry path visible in the graph topology
      and in trajectory logs.

    State read:  retry_count
    State write: retry_count (incremented), execution_status (reset)
    Always transitions to: ``research``
    """
    current = state.get("retry_count", 0)
    new_count = current + 1

    logger.info(
        "retry_increment",
        retry_count=new_count,
        shipment_id=getattr(state.get("telemetry_alert"), "shipment_id", "unknown"),
    )

    return {
        "retry_count": new_count,
        "execution_status": "researching",
    }


def _escalation_handler_node(state: GraphState) -> dict[str, Any]:
    """
    Escalation terminal node.

    Fires when the Decision Agent sets ``should_escalate=True``.
    Logs a human-operator alert and sets the final status.

    This is a distinct terminal state from "failed":
    - "escalated" = system correctly identified that human judgement is needed
    - "failed"    = system encountered an unrecoverable error

    State read:  telemetry_alert, selected_route, decision_summary
    State write: execution_status, error_message
    Always transitions to: END
    """
    alert = state.get("telemetry_alert")
    decision_summary = state.get("decision_summary", "No decision summary available.")
    selected = state.get("selected_route")

    logger.warning(
        "human_escalation_required",
        shipment_id=getattr(alert, "shipment_id", "unknown"),
        priority=getattr(alert, "priority", "unknown"),
        decision_summary=decision_summary,
        selected_carrier=(selected.carrier_id if selected else "none"),
        step_count=state.get("step_count", 0),
    )

    return {
        "execution_status": "escalated",
        "error_message": (
            f"Autonomous resolution not possible for shipment "
            f"{getattr(alert, 'shipment_id', 'unknown')}. "
            f"Escalated to human operator. Reason: {decision_summary}"
        ),
    }


# ---------------------------------------------------------------------------
# Routing functions (pure — read-only access to state)
# ---------------------------------------------------------------------------


def _route_after_research(
    state: GraphState,
) -> Literal["decide", "retry", "fail"]:
    """
    Conditional edge: Research → {Decision | RetryIncrement | END}.

    Routing logic:
    ┌──────────────────────────────────────────────────────────┐
    │ Condition                              │ Next node        │
    ├──────────────────────────────────────────────────────────┤
    │ ≥1 viable CarrierOption found         │ ``decide``       │
    │ 0 viable options, retry_count < max   │ ``retry``        │
    │ 0 viable options, max retries reached │ ``fail`` → END   │
    │ step_count ≥ MAX_GRAPH_STEPS          │ ``fail`` → END   │
    └──────────────────────────────────────────────────────────┘

    A ``CarrierOption`` is "viable" when ``constraints_satisfied=True``.
    This includes: cargo type certified, capacity available, ETA within
    deadline, reliability above floor, and transit mode in required set.
    """
    options = state.get("carrier_options", [])
    retry_count = state.get("retry_count", 0)
    step_count = state.get("step_count", 0)
    max_retries = get_settings().max_retries
    viable = [o for o in options if o.is_viable]

    # Hard step ceiling
    if step_count >= _MAX_GRAPH_STEPS:
        logger.error(
            "max_graph_steps_reached",
            step_count=step_count,
            limit=_MAX_GRAPH_STEPS,
        )
        return "fail"

    if viable:
        logger.info(
            "routing_to_decision",
            viable_options=len(viable),
            total_options=len(options),
        )
        return "decide"
    elif retry_count < max_retries:
        logger.warning(
            "routing_to_retry",
            reason="no_viable_options",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return "retry"
    else:
        logger.error(
            "routing_to_fail",
            reason="max_retries_exceeded",
            retry_count=retry_count,
        )
        return "fail"


def _route_after_decision(
    state: GraphState,
) -> Literal["execute", "escalate", "retry", "fail"]:
    """
    Conditional edge: Decision → {Execution | Escalation | RetryIncrement | END}.

    Routing logic:
    ┌────────────────────────────────────────────────────────────────┐
    │ Condition                                  │ Next node         │
    ├────────────────────────────────────────────────────────────────┤
    │ should_escalate=True (any reason)          │ ``escalate``      │
    │ selected_route is set AND no escalation    │ ``execute``       │
    │ no route + retry_count < max               │ ``retry``         │
    │ no route + max retries reached             │ ``fail`` → END    │
    │ step_count ≥ MAX_GRAPH_STEPS               │ ``fail`` → END    │
    └────────────────────────────────────────────────────────────────┘

    Note: escalation takes priority over execution even if a route is
    set — this handles the case where the Decision Agent selected a
    route but flagged it as requiring human approval (low confidence).
    """
    route = state.get("selected_route")
    escalate = state.get("should_escalate", False)
    retry_count = state.get("retry_count", 0)
    step_count = state.get("step_count", 0)
    max_retries = get_settings().max_retries

    if step_count >= _MAX_GRAPH_STEPS:
        logger.error("max_graph_steps_reached", step_count=step_count)
        return "fail"

    if escalate:
        logger.warning(
            "routing_to_escalation",
            reason="should_escalate_flag",
            route_id=getattr(route, "route_id", "none"),
            confidence=getattr(route, "confidence_score", None),
        )
        return "escalate"

    if route is not None:
        logger.info(
            "routing_to_execution",
            carrier=getattr(route, "carrier_id", "unknown"),
            confidence=getattr(route, "confidence_score", None),
        )
        return "execute"

    if retry_count < max_retries:
        logger.warning(
            "routing_to_retry",
            reason="no_route_selected",
            retry_count=retry_count,
        )
        return "retry"

    logger.error(
        "routing_to_fail",
        reason="no_route_max_retries",
        retry_count=retry_count,
    )
    return "fail"


def _route_after_execution(
    state: GraphState,
) -> Literal["complete", "retry", "fail"]:
    """
    Conditional edge: Execution → {END (completed) | RetryIncrement | END (failed)}.

    Routing logic:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Condition                                    │ Next node         │
    ├──────────────────────────────────────────────────────────────────┤
    │ execution_result.success = True              │ ``complete`` →END │
    │ execution failed + retry_count < max         │ ``retry``         │
    │ execution failed + max retries reached       │ ``fail`` → END    │
    │ execution_result is None (unexpected)        │ ``fail`` → END    │
    └──────────────────────────────────────────────────────────────────┘

    On retry after execution failure, control returns to ``research``
    (via ``retry_increment``) rather than re-trying only ``execution``.
    Rationale: if the selected carrier failed to execute, the carrier
    data may be stale — re-researching alternatives is the correct
    recovery action.
    """
    result = state.get("execution_result")
    retry_count = state.get("retry_count", 0)
    max_retries = get_settings().max_retries

    if result is not None and result.success:
        logger.info(
            "routing_to_complete",
            confirmation=getattr(result, "confirmation_number", "none"),
        )
        return "complete"

    if retry_count < max_retries:
        logger.warning(
            "routing_to_retry",
            reason="execution_failed",
            retry_count=retry_count,
        )
        return "retry"

    logger.error(
        "routing_to_fail",
        reason="execution_max_retries",
        retry_count=retry_count,
    )
    return "fail"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_logistics_graph(
    model_name: str | None = None,
    settings: AppSettings | None = None,
    use_checkpointer: bool = False,
) -> Any:
    """
    Build and compile the logistics rerouting ``StateGraph``.

    Parameters
    ----------
    model_name:
        Override the default model (from settings).  If provided, the
        initial state is pre-configured with this model.  Individual
        runs can also override via ``create_initial_state(model_name=...)``.
    settings:
        Override the global ``AppSettings`` singleton.  Used in tests.
    use_checkpointer:
        If True, compiles the graph with a ``MemorySaver`` in-memory
        checkpointer.  Enables graph interruption and resumption.
        When using a checkpointer, pass
        ``config={"configurable": {"thread_id": run_id}}`` to
        ``.invoke()`` / ``.stream()``.

    Returns
    -------
    CompiledStateGraph
        Ready for ``.invoke(initial_state)`` or ``.stream(initial_state)``.

    Example
    -------
    ::

        graph = build_logistics_graph(model_name="gpt-4o")
        state = create_initial_state(DEFAULT_SCENARIO, model_name="gpt-4o")
        result = graph.invoke(state)
    """
    cfg = settings or get_settings()
    _model = model_name or cfg.closed_source_model

    # 1. Instantiate agents (shared across retries — stateless callables)
    planner   = PlannerAgent(settings=cfg)
    research  = ResearchAgent(settings=cfg)
    decision  = DecisionAgent(settings=cfg)
    execution = ExecutionAgent(settings=cfg)

    logger.info(
        "building_graph",
        model=_model,
        max_retries=cfg.max_retries,
        use_checkpointer=use_checkpointer,
    )

    # 2. Declare the StateGraph with our typed state schema
    workflow = StateGraph(GraphState)

    # 3. Register nodes
    # -----------------
    # Agent nodes: callable objects implementing BaseAgent.__call__(state) -> dict
    workflow.add_node(_NODE_PLANNER,    planner)
    workflow.add_node(_NODE_RESEARCH,   research)
    workflow.add_node(_NODE_DECISION,   decision)
    workflow.add_node(_NODE_EXECUTION,  execution)

    # Auxiliary nodes: plain functions -> partial state dict
    workflow.add_node(_NODE_RETRY,      _retry_increment_node)
    workflow.add_node(_NODE_ESCALATION, _escalation_handler_node)

    # 4. Wire edges
    # -------------

    # START → Planner (always)
    # The first thing that always happens: translate the raw alert into
    # a structured plan and constraints.
    workflow.add_edge(START, _NODE_PLANNER)

    # Planner → Research (always)
    # Constraints are ready; discover carriers immediately.
    workflow.add_edge(_NODE_PLANNER, _NODE_RESEARCH)

    # Research → {Decision | RetryIncrement | END} (conditional)
    # The Research Agent may or may not find viable options.
    workflow.add_conditional_edges(
        _NODE_RESEARCH,
        _route_after_research,
        {
            "decide": _NODE_DECISION,
            "retry":  _NODE_RETRY,
            "fail":   END,
        },
    )

    # RetryIncrement → Research (always)
    # After bumping the counter, re-run research.
    # Note: the retry node applies to BOTH research failures (no options)
    # and execution failures (carrier booking failed). Both route back to
    # research rather than retrying only the failed node, because a
    # failed execution may mean the chosen carrier is now unavailable.
    workflow.add_edge(_NODE_RETRY, _NODE_RESEARCH)

    # Decision → {Execution | Escalation | RetryIncrement | END} (conditional)
    workflow.add_conditional_edges(
        _NODE_DECISION,
        _route_after_decision,
        {
            "execute":  _NODE_EXECUTION,
            "escalate": _NODE_ESCALATION,
            "retry":    _NODE_RETRY,
            "fail":     END,
        },
    )

    # Escalation → END (always)
    # Escalation is a clean terminal state — no further automation.
    workflow.add_edge(_NODE_ESCALATION, END)

    # Execution → {END (completed) | RetryIncrement | END (failed)} (conditional)
    workflow.add_conditional_edges(
        _NODE_EXECUTION,
        _route_after_execution,
        {
            "complete": END,
            "retry":    _NODE_RETRY,
            "fail":     END,
        },
    )

    # 5. Compile
    # ----------
    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        compiled = workflow.compile(checkpointer=checkpointer)
        logger.info("graph_compiled", checkpointer="MemorySaver")
    else:
        compiled = workflow.compile()
        logger.info("graph_compiled", checkpointer="none")

    return compiled


# ---------------------------------------------------------------------------
# Initial state factory
# ---------------------------------------------------------------------------


def create_initial_state(
    telemetry_alert: dict[str, Any] | ShipmentAlert,
    model_name: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a properly initialized ``GraphState`` dict for a new graph run.

    Every field is set explicitly — no field is allowed to be missing at
    run start.  This prevents KeyError / AttributeError inside agents that
    do ``state.get(field, default)`` on the first iteration.

    Parameters
    ----------
    telemetry_alert:
        Either a ``ShipmentAlert`` instance or a dict that matches its
        field schema.  Dict is converted to ``ShipmentAlert`` so downstream
        agents always receive a typed object.
    model_name:
        LLM model identifier (e.g. ``"gpt-4o"`` or ``"qwen/qwen3-8b"``).
        Stored in state so every agent reads the SAME model for this run.
    run_id:
        Optional run identifier.  Auto-generated UUID if not provided.

    Returns
    -------
    dict[str, Any]
        A complete ``GraphState``-compatible dict.  Pass directly to
        ``graph.invoke(state)``.

    Example
    -------
    ::

        state = create_initial_state(DEFAULT_SCENARIO, "gpt-4o")
        result = graph.invoke(state)
    """
    _run_id = run_id or str(uuid.uuid4())

    # Normalise alert to typed ShipmentAlert
    if isinstance(telemetry_alert, dict):
        alert = ShipmentAlert(**telemetry_alert)
    else:
        alert = telemetry_alert

    logger.info(
        "initial_state_created",
        run_id=_run_id,
        shipment_id=alert.shipment_id,
        model=model_name,
        priority=alert.priority,
    )

    return {
        # ① Identity
        "run_id": _run_id,
        "model_name": model_name,

        # ② Input
        "telemetry_alert": alert,

        # ③ Planning (populated by PlannerAgent)
        "rerouting_plan": "",
        "rerouting_constraints": {},

        # ④ Research (populated by ResearchAgent; list reducer via operator.add)
        "carrier_options": [],
        "research_summary": "",

        # ⑤ Decision (populated by DecisionAgent)
        "selected_route": None,
        "decision_summary": "",

        # ⑥ Execution (populated by ExecutionAgent)
        "execution_result": None,

        # ⑦ Flow control
        "execution_status": "pending",
        "retry_count": 0,
        "should_escalate": False,
        "step_count": 0,
        "error_message": None,

        # ⑧ Trajectory (list reducer via operator.add)
        "trajectory": [],

        # ⑨ Error history (list reducer via operator.add)
        "error_history": [],

        # ⑩ Evaluation metadata (int/float reducers via operator.add)
        "total_tool_calls": 0,
        "successful_tool_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,

        # ⑪ Timing
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Convenience re-exports for main.py
# ---------------------------------------------------------------------------

__all__ = [
    "build_logistics_graph",
    "create_initial_state",
    "DEFAULT_SCENARIO",
    "_route_after_research",   # exported for unit tests
    "_route_after_decision",
    "_route_after_execution",
    "_retry_increment_node",
    "_escalation_handler_node",
]
