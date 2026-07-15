"""
agents/
=======
Single-responsibility LangGraph agent nodes for the autonomous
logistics rerouting system.

Each agent is a pure function:
    (GraphState) -> GraphState

Agents:
    - PlannerAgent   : Parses telemetry alert, constructs rerouting plan
    - ResearchAgent  : Queries carrier/route availability via tools
    - DecisionAgent  : Evaluates alternatives, selects optimal route
    - ExecutionAgent : Executes reroute decision and verifies outcome
"""
