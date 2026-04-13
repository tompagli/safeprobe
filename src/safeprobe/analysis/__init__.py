"""SafeProbe analysis package.

Public API:

  Judges:
    JailbreakJudge   — CoT judge (DeepSeek R1 local or any API model)
    LlamaGuardJudge  — Meta Llama Guard 3 safety classifier
    HarmBenchJudge   — CAIS HarmBench binary classifier
    MultiJudge       — Run N judges + compute inter-rater agreement

  Agreement:
    compute_agreement       — Cohen's κ + Fleiss' κ for any set of judges
    format_agreement_report — Human-readable agreement table

  Pipeline:
    ResultsConsolidator — Merge JSON results, compute ASR & Robustness Score
    ReportGenerator     — TXT / JSON / PDF report generation
"""
from safeprobe.analysis.judges import BaseJudge, JailbreakJudge, LlamaGuardJudge, HarmBenchJudge
from safeprobe.analysis.multi_judge import MultiJudge
from safeprobe.analysis.agreement import compute_agreement, format_agreement_report
from safeprobe.analysis.consolidator import ResultsConsolidator
from safeprobe.analysis.report_gen import ReportGenerator

# Backward-compat alias
CoTJudge = JailbreakJudge

__all__ = [
    "BaseJudge",
    "JailbreakJudge",
    "CoTJudge",
    "LlamaGuardJudge",
    "HarmBenchJudge",
    "MultiJudge",
    "compute_agreement",
    "format_agreement_report",
    "ResultsConsolidator",
    "ReportGenerator",
]
