"""
evaluation/
===========
Trajectory-based evaluation framework for comparing closed-source
vs. open-source LLM performance on the logistics rerouting workflow.

Captures per-node TrajectorySnapshots across all graph executions
and computes aggregate metrics:
    - Reasoning quality score
    - Tool-calling accuracy
    - Error recovery rate
    - Latency (per-step and end-to-end)
    - Token-based cost estimation
    - Final output quality score

Classes:
    - TrajectoryEvaluator : Core snapshot analysis and scoring engine
    - ModelBenchmarker    : Orchestrates multiple scenario runs for models
    - ReportGenerator     : Aggregates results into Markdown reports and charts
"""

from evaluation.trajectory_evaluator import TrajectoryEvaluator
from evaluation.model_benchmarker import ModelBenchmarker, get_benchmark_scenarios
from evaluation.report_generator import ReportGenerator

__all__ = [
    "TrajectoryEvaluator",
    "ModelBenchmarker",
    "get_benchmark_scenarios",
    "ReportGenerator",
]
