"""
Configuration management for SafeProbe.

Supports both programmatic API usage and YAML-based configuration
for reproducibility and pipeline automation (e.g., CI/CD integration).
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class Config:
    """Base configuration for SafeProbe toolkit."""

    # API Keys (loaded from environment variables)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )
    xai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("XAI_API_KEY")
    )
    azure_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY")
    )

    # Target model configuration
    target_model: str = "gpt-4.1-2025-04-14"
    target_model_type: str = "openai"  # openai, anthropic, google, ollama, xai

    # Judge model (CoT-based classifier)
    judge_model: str = "gpt-4.1-2025-04-14"
    judge_model_type: str = "openai"

    # Attack configuration
    attacks: List[str] = field(default_factory=lambda: ["promptmap", "pair", "cipherchat"])

    # Dataset
    dataset: str = "advbench"  # advbench or custom path
    sample_size: Optional[int] = 50
    iterations: int = 3

    # Output settings
    results_dir: Path = field(default_factory=lambda: Path("results"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    # Runtime settings
    verbose: bool = False
    save_intermediate: bool = True

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.log_dir = Path(self.log_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, filepath: str):
        """Save configuration to a YAML file for reproducibility."""
        data = {
            "target_model": self.target_model,
            "target_model_type": self.target_model_type,
            "judge_model": self.judge_model,
            "judge_model_type": self.judge_model_type,
            "attacks": self.attacks,
            "dataset": self.dataset,
            "results_dir": str(self.results_dir),
            "log_dir": str(self.log_dir),
            "verbose": self.verbose,
            "save_intermediate": self.save_intermediate,
        }
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_api_key(self, model_type: Optional[str] = None) -> Optional[str]:
        """Get API key for a given model type."""
        mt = model_type or self.target_model_type
        keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "xai": self.xai_api_key,
            "azure": self.azure_api_key,
            "ollama": None,
        }
        return keys.get(mt)

    def validate(self) -> bool:
        """Validate that at least one API key is available."""
        any_key = any([self.openai_api_key, self.anthropic_api_key,
                       self.google_api_key, self.xai_api_key, self.azure_api_key])
        if not any_key and self.target_model_type != "ollama":
            return False
        return True


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment.

    Priority: CLI args > config file > environment variables > defaults
    """
    # Try to load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        if Path(".env").exists():
            load_dotenv()
    except ImportError:
        pass

    if config_path:
        p = Path(config_path)
        if p.suffix in (".yml", ".yaml"):
            return Config.from_yaml(config_path)
        elif p.suffix == ".json":
            return Config.from_json(config_path)

    return Config()
