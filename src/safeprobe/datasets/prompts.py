"""
safeprobe.datasets.prompts - Curated adversarial prompt datasets.

Provides access to AdvBench (walledai/AdvBench) and custom prompt collections
organized by attack category, aligned with the NIST AML taxonomy.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


def load_advbench(path: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load the AdvBench dataset.

    If a local path is provided, loads from file.
    Otherwise, attempts to load from HuggingFace datasets.

    Args:
        path: Optional local path to AdvBench CSV/JSON file.
        max_samples: Maximum number of samples to load (None = all).

    Returns:
        List of dicts with keys: 'goal', 'target', 'category' (if available).
    """
    if path and Path(path).exists():
        return _load_local(path, max_samples)

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("walledai/AdvBench", split="train")
        samples = []
        for item in ds:
            entry = {
                "goal": item.get("goal", item.get("prompt", "")),
                "target": item.get("target", "Sure, here is"),
            }
            samples.append(entry)
            if max_samples and len(samples) >= max_samples:
                break
        return samples
    except Exception:
        pass

    # Fallback: try loading from package data
    pkg_data = Path(__file__).parent / "data" / "advbench.csv"
    if pkg_data.exists():
        return _load_local(str(pkg_data), max_samples)

    raise FileNotFoundError(
        "AdvBench dataset not found. Provide a local path, or install `datasets` "
        "package: pip install datasets"
    )


def _load_local(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load dataset from a local CSV or JSON file."""
    p = Path(path)
    samples = []

    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    samples.append({
                        "goal": item.get("goal", item.get("prompt", "")),
                        "target": item.get("target", "Sure, here is"),
                    })
                elif isinstance(item, str):
                    samples.append({"goal": item, "target": "Sure, here is"})
                if max_samples and len(samples) >= max_samples:
                    break

    elif p.suffix == ".csv":
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "goal": row.get("goal", row.get("prompt", "")),
                    "target": row.get("target", "Sure, here is"),
                })
                if max_samples and len(samples) >= max_samples:
                    break

    elif p.suffix == ".txt":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append({"goal": line, "target": "Sure, here is"})
                    if max_samples and len(samples) >= max_samples:
                        break
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    return samples


def load_dataset(name: str, path: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load a named dataset.

    Args:
        name: Dataset name ('advbench' or path to file).
        path: Optional explicit file path.
        max_samples: Maximum number of samples.

    Returns:
        List of prompt dictionaries.
    """
    if name.lower() == "advbench":
        return load_advbench(path=path, max_samples=max_samples)
    elif path:
        return _load_local(path, max_samples)
    else:
        # Try treating name as a path
        return _load_local(name, max_samples)
