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

Modules:
    - trajectory_evaluator : Core snapshot capture + metric computation
    - model_benchmarker    : Orchestrates multi-scenario runs per model
    - report_generator     : Produces markdown tables + matplotlib charts
"""
