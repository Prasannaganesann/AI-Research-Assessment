"""
agents/
=======
Single-responsibility LangGraph agent nodes for the autonomous
logistics rerouting system.

Each agent is callable as a LangGraph node:
    (GraphState) -> dict[str, Any]  (partial state update)

Import pattern
--------------
::

    from agents import PlannerAgent, ResearchAgent, DecisionAgent, ExecutionAgent

    planner  = PlannerAgent()
    research = ResearchAgent()
    decision = DecisionAgent()
    execution = ExecutionAgent()
"""

from agents.base_agent import BaseAgent
from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.decision_agent import DecisionAgent
from agents.execution_agent import ExecutionAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ResearchAgent",
    "DecisionAgent",
    "ExecutionAgent",
]
