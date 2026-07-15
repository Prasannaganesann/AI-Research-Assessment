"""
tools/
======
LangChain-compatible tool definitions used by agent nodes.

All tools return structured, typed responses so that agents can
reason over results without string-parsing heuristics.

Tool Registry
-------------
``ALL_TOOLS``       — all 7 tools (use when constructing a generic agent)
``RESEARCH_TOOLS``  — tools for the Research Agent
``DECISION_TOOLS``  — tools for the Decision Agent
``EXECUTION_TOOLS`` — tools for the Execution Agent

Importing
---------
::

    from tools import RESEARCH_TOOLS, DECISION_TOOLS, EXECUTION_TOOLS
    from tools import ALL_TOOLS
"""

from tools.carrier_tools import list_available_carriers, lookup_carrier_capacity
from tools.route_tools import (
    calculate_route_eta,
    estimate_reroute_cost,
    validate_route_constraints,
)
from tools.notification_tools import (
    send_reroute_notification,
    generate_execution_report,
)
from tools._simulation import FailureSimulator, SimulatedToolError

# ------------------------------------------------------------------
# Tool registries  (one list per agent — enforces single responsibility)
# ------------------------------------------------------------------

#: Tools available to the Research Agent.
#: Discovers and characterises carrier alternatives.
RESEARCH_TOOLS = [
    list_available_carriers,
    lookup_carrier_capacity,
    calculate_route_eta,
    estimate_reroute_cost,
]

#: Tools available to the Decision Agent.
#: Validates constraints before finalising the route selection.
DECISION_TOOLS = [
    validate_route_constraints,
]

#: Tools available to the Execution Agent.
#: Submits the reroute and closes the stakeholder loop.
EXECUTION_TOOLS = [
    send_reroute_notification,
    generate_execution_report,
]

#: Full tool set — used in tests and benchmarking.
ALL_TOOLS = RESEARCH_TOOLS + DECISION_TOOLS + EXECUTION_TOOLS

__all__ = [
    # Individual tools
    "list_available_carriers",
    "lookup_carrier_capacity",
    "calculate_route_eta",
    "estimate_reroute_cost",
    "validate_route_constraints",
    "send_reroute_notification",
    "generate_execution_report",
    # Registries
    "RESEARCH_TOOLS",
    "DECISION_TOOLS",
    "EXECUTION_TOOLS",
    "ALL_TOOLS",
    # Simulation
    "FailureSimulator",
    "SimulatedToolError",
]
