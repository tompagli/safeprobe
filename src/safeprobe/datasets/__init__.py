"""
safeprobe.datasets - Manages adversarial prompt datasets and model connectivity.

Contains:
  - prompts: Curated adversarial prompts (AdvBench integration)
  - adapters: Unified interface for communicating with different LLM backends
"""

from .prompts import load_advbench, load_dataset
from .adapters import LLMAdapter, OpenAIAdapter, AnthropicAdapter, OllamaAdapter

__all__ = [
    "load_advbench",
    "load_dataset",
    "LLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
]
