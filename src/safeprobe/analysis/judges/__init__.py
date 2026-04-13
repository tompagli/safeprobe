"""Judge backends for SafeProbe.

All judge classes share the BaseJudge interface:
    judge_single(attack_prompt, response_prompt) -> Dict

Available backends:

  JailbreakJudge  — Chain-of-Thought judge (DeepSeek R1 local or any API model)
  LlamaGuardJudge — Meta Llama Guard 3 safety classifier (local HF)
  HarmBenchJudge  — CAIS HarmBench classifier (local HF)

Use MultiJudge (analysis/multi_judge.py) to run multiple backends in parallel
and compute inter-rater agreement (Cohen's κ, Fleiss' κ).
"""
from .base import BaseJudge
from .deepseek import JailbreakJudge
from .llama_guard import LlamaGuardJudge
from .harmbench import HarmBenchJudge

__all__ = ["BaseJudge", "JailbreakJudge", "LlamaGuardJudge", "HarmBenchJudge"]
