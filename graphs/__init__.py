"""
graphs/
=======
LangGraph state schema and graph factory for the autonomous logistics
rerouting system.

Modules
-------
state.py          — ``GraphState`` TypedDict + all domain Pydantic models
logistics_graph.py — ``build_logistics_graph()`` factory + routing functions

Quick Start
-----------
::

    from graphs import build_logistics_graph, create_initial_state, DEFAULT_SCENARIO
    from config.settings import get_settings

    graph = build_logistics_graph(model_name="gpt-4o")
    state = create_initial_state(DEFAULT_SCENARIO, model_name="gpt-4o")
    result = graph.invoke(state)
    print(result["execution_status"])   # "completed" | "failed" | "escalated"
"""

# Domain models (re-exported for convenience)
from graphs.state import (
    CarrierOption,
    DEFAULT_SCENARIO,
    ExecutionResult,
    GraphState,
    SelectedRoute,
    ShipmentAlert,
    ToolCallRecord,
    TrajectorySnapshot,
)

# Graph factory
from graphs.logistics_graph import (
    build_logistics_graph,
    create_initial_state,
)

__all__ = [
    # Graph
    "build_logistics_graph",
    "create_initial_state",
    # State
    "GraphState",
    "DEFAULT_SCENARIO",
    # Domain models
    "ShipmentAlert",
    "CarrierOption",
    "SelectedRoute",
    "ExecutionResult",
    "TrajectorySnapshot",
    "ToolCallRecord",
]
