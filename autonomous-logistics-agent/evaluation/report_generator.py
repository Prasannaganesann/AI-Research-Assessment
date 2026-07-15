"""
evaluation/report_generator.py — Comparative Report Generator
==============================================================

Aggregates benchmark results and generates structured markdown reports
comparing closed-source (GPT-4o) vs. open-source (Qwen3) model performance.

Output Formats
--------------
1.  **Markdown Table**: Produces side-by-side metric comparisons for:
    - Composite Score
    - Reasoning Quality
    - Tool-Calling Accuracy
    - Error Recovery Rate
    - Latency (seconds)
    - LLM Token Cost (USD)
2.  **Chart Figures**: If ``matplotlib`` is installed, generates bar charts
    saved to ``reports/figures/`` comparing composite score, cost, and
    latency across models.  Falls back gracefully if ``matplotlib`` is missing.

Key Performance Indicators (KPIs)
----------------------------------
- **Capability Floor**: Does the model complete the standard scenario successfully?
- **Resilience Ceiling**: Does the model recover from injected capacity lookup failures?
- **Escalation Precision**: Does it escalate the unresolvable customs delay?
- **Operational Cost-Efficiency**: USD cost per run vs. performance trade-off.
"""

from __future__ import annotations

import os
from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Takes raw benchmark outputs and formats them into readable markdown summaries
    and visualisations.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    def generate_markdown_report(
        self,
        benchmark_results: dict[str, dict[str, dict[str, Any]]],
    ) -> str:
        """
        Aggregate results from the benchmark run and build a markdown report.

        benchmark_results is structured as:
            { "gpt-4o": { "scenario_1": report_dict, ... }, "qwen3": { ... } }
        """
        models = list(benchmark_results.keys())
        if not models:
            return "# No Benchmark Results Found"

        # Find all scenario names
        scenario_names = set()
        for m in models:
            scenario_names.update(benchmark_results[m].keys())
        scenarios = sorted(list(scenario_names))

        report_lines = []
        report_lines.append("# LLM Benchmark & Comparative Performance Report")
        report_lines.append(f"\nGenerated on: {os.environ.get('CURRENT_TIME', '2026-07-15T14:44:00+00:00')}\n")

        # -------------------------------------------------------------------
        # Section 1: Executive Summary Table (Averages across all scenarios)
        # -------------------------------------------------------------------
        report_lines.append("## Executive Summary")
        report_lines.append("\nAveraged metrics across all evaluated logistics scenarios:\n")

        report_lines.append(
            "| Model | Composite Score | Tool Accuracy | Reasoning Score | "
            "Recovery Rate | Avg Latency (s) | Cost per 100 Runs (USD) |"
        )
        report_lines.append("|---|---|---|---|---|---|---|")

        summary_metrics: dict[str, dict[str, float]] = {}

        for model in models:
            total_comp = 0.0
            total_tool = 0.0
            total_reason = 0.0
            total_recover = 0.0
            total_latency = 0.0
            total_cost = 0.0
            valid_scenarios = 0

            for sc in scenarios:
                rep = benchmark_results[model].get(sc, {})
                if rep.get("execution_status") == "failed" and "composite_score" not in rep:
                    continue
                total_comp += rep.get("composite_score", 0.0)
                total_tool += rep.get("tool_calling_accuracy_score", 0.0)
                total_reason += rep.get("reasoning_quality_score", 0.0)
                total_recover += rep.get("error_recovery_score", 0.0)
                total_latency += rep.get("total_latency_ms", 0.0) / 1000.0
                total_cost += rep.get("total_cost_usd", 0.0)
                valid_scenarios += 1

            if valid_scenarios > 0:
                avg_comp = total_comp / valid_scenarios
                avg_tool = total_tool / valid_scenarios
                avg_reason = total_reason / valid_scenarios
                avg_recover = total_recover / valid_scenarios
                avg_lat = total_latency / valid_scenarios
                avg_cost_100 = (total_cost / valid_scenarios) * 100.0

                summary_metrics[model] = {
                    "composite": avg_comp,
                    "tool": avg_tool,
                    "reasoning": avg_reason,
                    "recovery": avg_recover,
                    "latency": avg_lat,
                    "cost_100": avg_cost_100,
                }

                report_lines.append(
                    f"| **{model}** | {avg_comp:.2%} | {avg_tool:.2%} | "
                    f"{avg_reason:.2%} | {avg_recover:.2%} | {avg_lat:.2f}s | "
                    f"${avg_cost_100:.4f} |"
                )

        # -------------------------------------------------------------------
        # Section 2: Detailed Scenario-by-Scenario Metrics
        # -------------------------------------------------------------------
        report_lines.append("\n## Detailed Scenario Breakdown")

        for sc in scenarios:
            clean_sc_name = sc.replace("scenario_", "").replace("_", " ").title()
            report_lines.append(f"\n### {clean_sc_name}")
            report_lines.append(
                f"\nComparing model decisions for scenario identifier `{sc}`:\n"
            )

            report_lines.append(
                "| Metric | " + " | ".join(f"**{model}**" for model in models) + " |"
            )
            report_lines.append("|---|" + "|".join("---" for _ in models) + "|")

            metrics_to_show = [
                ("Execution Status", "execution_status", "{val}"),
                ("Composite Score", "composite_score", "{val:.2%}"),
                ("Tool Calling Accuracy", "tool_calling_accuracy_score", "{val:.2%}"),
                ("Reasoning Quality Score", "reasoning_quality_score", "{val:.2%}"),
                ("Error Recovery Rate", "error_recovery_score", "{val:.2%}"),
                ("Total Latency (s)", "total_latency_ms", "{val_s:.2f}s"),
                ("LLM API Cost (USD)", "total_cost_usd", "${val:.6f}"),
                ("Total Workflow Steps", "total_steps", "{val}"),
            ]

            for label, key, fmt in metrics_to_show:
                row_parts = [label]
                for model in models:
                    rep = benchmark_results[model].get(sc, {})
                    val = rep.get(key)

                    if val is None:
                        row_parts.append("N/A")
                    else:
                        if "total_latency_ms" in key:
                            row_parts.append(fmt.format(val_s=val / 1000.0))
                        elif isinstance(val, float):
                            row_parts.append(fmt.format(val=val))
                        else:
                            row_parts.append(fmt.format(val=val))

                report_lines.append(" | ".join(row_parts) + " |")

        # -------------------------------------------------------------------
        # Section 3: Cost/Benefit Trade-off Analysis
        # -------------------------------------------------------------------
        report_lines.append("\n## Cost-Efficiency Analysis")
        report_lines.append(
            "\nEvaluating the cost-performance boundary. "
            "How much does reasoning cost vs. the score achieved?\n"
        )

        if len(summary_metrics) >= 2:
            model_list = list(summary_metrics.keys())
            m1, m2 = model_list[0], model_list[1]
            c1, c2 = summary_metrics[m1]["composite"], summary_metrics[m2]["composite"]
            cost1, cost2 = summary_metrics[m1]["cost_100"], summary_metrics[m2]["cost_100"]

            # Calculate cost ratio and performance ratio
            if cost2 > 0 and c2 > 0:
                cost_ratio = cost1 / cost2
                perf_delta = c1 - c2
                report_lines.append(
                    f"- **Operational Leverage**: `{m1}` achieves a composite score of {c1:.2%}, "
                    f"compared to `{m2}` at {c2:.2%} (delta: {perf_delta:+.2%}).\n"
                    f"- **Financial Surcharge**: `{m1}` is approximately **{cost_ratio:.1f}x** "
                    f"more expensive than `{m2}` per 100 runs ($({cost1:.4f}) vs $({cost2:.4f})).\n"
                    f"- **Decision Guide**: For critical routing decisions where failure costs thousands, "
                    f"the {perf_delta:+.2%} reasoning advantage of `{m1}` justifies the API surcharge. "
                    f"For high-volume non-critical lanes, `{m2}` is the optimal financial choice."
                )

        report_txt = "\n".join(report_lines)

        # Save report to reports/comparison_report.md
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        with open(report_path, "w") as fh:
            fh.write(report_txt)

        logger.info("report_file_saved", path=report_path)

        # Generate plots
        self._generate_plots(summary_metrics)

        return report_txt

    def _generate_plots(self, summary_metrics: dict[str, dict[str, float]]) -> None:
        """
        Generate bar chart visualisations using matplotlib if available.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            models = list(summary_metrics.keys())
            if not models:
                return

            composite_scores = [summary_metrics[m]["composite"] * 100.0 for m in models]
            costs = [summary_metrics[m]["cost_100"] for m in models]
            latencies = [summary_metrics[m]["latency"] for m in models]

            # Plot 1: Performance vs Cost comparison
            fig, ax1 = plt.subplots(figsize=(8, 5))

            color = "#1f77b4"
            ax1.set_xlabel("LLM Model")
            ax1.set_ylabel("Composite Score (%)", color=color)
            bars1 = ax1.bar(
                [m + "\n(Performance)" for m in models],
                composite_scores,
                color=color,
                alpha=0.6,
                width=0.4,
            )
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.set_ylim(0, 100)

            # Instantiating a second axes that shares the same x-axis
            ax2 = ax1.twinx()
            color = "#d62728"
            ax2.set_ylabel("Cost per 100 Runs (USD)", color=color)
            bars2 = ax2.bar(
                [m + "\n(Cost)" for m in models],
                costs,
                color=color,
                alpha=0.6,
                width=0.4,
            )
            ax2.tick_params(axis="y", labelcolor=color)

            plt.title("Logistics MAS: Performance vs Cost Trade-off")
            fig.tight_layout()

            plot_path = os.path.join(self.output_dir, "figures", "performance_vs_cost.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

            # Plot 2: Latency comparison
            plt.figure(figsize=(6, 4))
            plt.bar(models, latencies, color="#2ca02c", alpha=0.7, width=0.5)
            plt.ylabel("Avg Scenario Latency (seconds)")
            plt.title("Avg End-to-End Workflow Latency")
            plt.tight_layout()

            plot_path2 = os.path.join(self.output_dir, "figures", "latency_comparison.png")
            plt.savefig(plot_path2, dpi=150)
            plt.close()

            logger.info("visualisation_charts_saved", output_dir=os.path.join(self.output_dir, "figures"))

        except ImportError:
            logger.warning(
                "matplotlib_not_installed_skipping_charts",
                reason="Matplotlib is required to generate visualisation charts. Run 'pip install matplotlib' to enable.",
            )
        except Exception as exc:
            logger.warning("plotting_failed_gracefully", error=str(exc))
