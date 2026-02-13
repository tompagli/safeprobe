"""
safeprobe.analysis.report_gen - Report generation module.

Exports results in structured formats (JSON, CSV) and generates summary
reports with charts, facilitating integration with monitoring dashboards
and compliance documentation workflows.
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generates evaluation reports from consolidated results."""

    # Harm categories for classification
    HARM_KEYWORDS = {
        "violence": ["kill", "murder", "assault", "weapon", "bomb", "attack", "violence"],
        "illegal": ["steal", "hack", "fraud", "scam", "illegal", "crime", "drug"],
        "hate": ["racist", "hate", "discriminat", "slur", "bias", "prejudice"],
        "sexual": ["sexual", "porn", "explicit", "nsfw"],
        "privacy": ["personal", "privacy", "dox", "data", "leak", "credential"],
        "financial": ["bank", "credit", "money", "account", "fortune"],
        "misinformation": ["fake", "false", "misinformation", "conspiracy", "hoax"],
        "malware": ["virus", "malware", "exploit", "backdoor", "trojan"],
    }

    def categorize_attack(self, attack_prompt: str) -> str:
        """Categorize an attack prompt based on keywords."""
        if not isinstance(attack_prompt, str):
            return "unknown"
        lower = attack_prompt.lower()
        for category, keywords in self.HARM_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return category
        return "other"

    def generate_text_report(self, df: pd.DataFrame, metrics: Dict,
                             output_path: Optional[str] = None) -> str:
        """Generate a text summary report."""
        lines = [
            "=" * 80,
            "SAFEPROBE - SAFETY ALIGNMENT EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80, "",
            f"Total samples: {metrics.get('total_samples', len(df))}",
        ]

        # Distribution by technique
        if "attack_tool" in df.columns:
            lines.append("\nDistribution by attack technique:")
            for tool, count in df["attack_tool"].value_counts().items():
                pct = count / len(df) * 100
                lines.append(f"  {tool}: {count} ({pct:.1f}%)")

        # Overall ASR
        overall = metrics.get("overall", {})
        if overall:
            lines.extend([
                "", "=" * 80,
                "ATTACK SUCCESS RATE (ASR)",
                "=" * 80, "",
                f"Overall ASR: {overall.get('asr', 0):.2f}%",
                f"Successful attacks: {overall.get('successful', 0)} / {overall.get('total', 0)}",
            ])

        # Per-technique ASR
        by_tech = metrics.get("by_technique", {})
        if by_tech:
            lines.extend(["", "ASR by Attack Technique:", "-" * 40])
            for tool, data in sorted(by_tech.items()):
                lines.append(f"  {tool}: {data['asr']:.2f}% ({data['successful']}/{data['total']})")

        # Robustness score
        if "robustness_score" in metrics:
            lines.extend([
                "", "=" * 80,
                f"ROBUSTNESS SCORE: {metrics['robustness_score']:.2f}/100",
                "=" * 80,
            ])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Text report saved to {output_path}")

        return report

    def generate_json_report(self, df: pd.DataFrame, metrics: Dict,
                             output_path: str):
        """Generate a JSON report with full metrics and entries."""
        report = {
            "report_type": "safeprobe_evaluation",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "entries": df.to_dict(orient="records"),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON report saved to {output_path}")

    def generate_csv_report(self, df: pd.DataFrame, output_path: str):
        """Export results as CSV."""
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"CSV report saved to {output_path}")

    def generate_pdf_report(self, df: pd.DataFrame, metrics: Dict,
                            output_path: str = "safeprobe_report.pdf"):
        """Generate a PDF report with charts and summary."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.backends.backend_pdf import PdfPages
            import numpy as np
        except ImportError:
            logger.error("matplotlib/seaborn required for PDF reports")
            return

        sns.set_style("whitegrid")

        with PdfPages(output_path) as pdf:
            # Page 1: Text summary
            fig = plt.figure(figsize=(11, 14))
            report_text = self.generate_text_report(df, metrics)
            fig.text(0.05, 0.98, report_text, fontsize=8, family="monospace",
                     verticalalignment="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

            # Page 2: ASR by technique bar chart
            by_tech = metrics.get("by_technique", {})
            if by_tech:
                fig, ax = plt.subplots(figsize=(12, 6))
                tools = list(by_tech.keys())
                asrs = [by_tech[t]["asr"] for t in tools]
                totals = [by_tech[t]["total"] for t in tools]

                colors = [
                    "#e74c3c" if a > 70 else "#f39c12" if a > 40 else "#2ecc71"
                    for a in asrs
                ]
                bars = ax.bar(tools, asrs, color=colors, edgecolor="black", linewidth=1.5)
                ax.set_title("Attack Success Rate (ASR) by Technique",
                             fontsize=16, fontweight="bold")
                ax.set_ylabel("ASR (%)")
                ax.set_ylim(0, 105)
                ax.grid(axis="y", alpha=0.3, linestyle="--")

                for bar, asr, total in zip(bars, asrs, totals):
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                            f"{asr:.1f}%\n(n={total})", ha="center", va="bottom",
                            fontweight="bold")

                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

            # Metadata
            d = pdf.infodict()
            d["Title"] = "SafeProbe - Safety Alignment Evaluation Report"
            d["Author"] = "SafeProbe Toolkit"
            d["CreationDate"] = datetime.now()

        logger.info(f"PDF report saved to {output_path}")
