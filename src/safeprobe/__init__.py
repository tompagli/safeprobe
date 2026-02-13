"""
SafeProbe: A Python Toolkit to Assess Safe Alignment of Large Language Models
"""
__version__ = "1.0.0"
__author__ = "Milton Pedro Pagliuso Neto"
__email__ = "milton.neto@edu.udesc.br"

from safeprobe.config import Config, load_config

SafeProbeConfig = Config

__all__ = ["SafeProbeConfig", "Config", "load_config"]
