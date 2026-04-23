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


def load_harmbench(
    path: Optional[str] = None,
    max_samples: Optional[int] = None,
    functional_category: Optional[str] = None,
) -> List[Dict]:
    """Load the HarmBench behaviors dataset.

    Tries in order:
      1. Local file at `path` (CSV or JSON)
      2. HuggingFace: walledai/HarmBench
      3. HuggingFace: cais/harmbench_behaviors_text_test (alternate name)

    Args:
        path: Optional local file path.
        max_samples: Maximum number of samples to return.
        functional_category: Filter by functional category
            (e.g. "standard", "copyright", "contextual").

    Returns:
        List of dicts with keys: goal, target, category, functional_category.
    """
    if path and Path(path).exists():
        raw = _load_local(path, max_samples)
        # _load_local may not populate category; return as-is
        return raw

    # Try HuggingFace (walledai/HarmBench requires access approval — gated)
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("walledai/HarmBench", split="train")
        samples = []
        for item in ds:
            goal = (
                item.get("Behavior") or item.get("behavior")
                or item.get("goal") or item.get("prompt", "")
            )
            func_cat = item.get("FunctionalCategory") or item.get("functional_category", "")
            sem_cat = item.get("SemanticCategory") or item.get("category", "")
            if functional_category and func_cat and functional_category.lower() not in func_cat.lower():
                continue
            samples.append({
                "goal": goal,
                "target": "Sure, here is",
                "category": sem_cat,
                "functional_category": func_cat,
                "source": "HarmBench",
            })
            if max_samples and len(samples) >= max_samples:
                break
        if samples:
            return samples
    except Exception:
        pass

    # Fallback: download behaviors CSV directly from the HarmBench GitHub repo
    try:
        import urllib.request
        import csv
        import io
        _URL = (
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
            "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
        )
        with urllib.request.urlopen(_URL, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        samples = []
        for row in reader:
            goal = row.get("Behavior") or row.get("behavior") or ""
            func_cat = row.get("FunctionalCategory", "")
            sem_cat = row.get("SemanticCategory", "")
            if functional_category and func_cat and functional_category.lower() not in func_cat.lower():
                continue
            if not goal:
                continue
            samples.append({
                "goal": goal,
                "target": "Sure, here is",
                "category": sem_cat,
                "functional_category": func_cat,
                "source": "HarmBench",
            })
            if max_samples and len(samples) >= max_samples:
                break
        if samples:
            return samples
    except Exception:
        pass

    raise FileNotFoundError(
        "HarmBench dataset not found. Options:\n"
        "  1. Request access at https://huggingface.co/datasets/walledai/HarmBench\n"
        "     then set HF_TOKEN env var and retry.\n"
        "  2. Download the CSV from https://github.com/centerforaisafety/HarmBench\n"
        "     and pass the local path: load_harmbench(path='harmbench_behaviors_text_test.csv')"
    )


def load_jailbreakbench(
    path: Optional[str] = None,
    max_samples: Optional[int] = None,
    category: Optional[str] = None,
) -> List[Dict]:
    """Load the JailbreakBench behaviors dataset (100 behaviors, 10 categories).

    Reference: Chao et al., "JailbreakBench: An Open Robustness Benchmark
    for Jailbreaking Large Language Models", NeurIPS 2024.

    Tries in order:
      1. Local file at `path`
      2. HuggingFace: JailbreakBench/JBB-Behaviors

    Args:
        path: Optional local file path.
        max_samples: Maximum number of samples.
        category: Filter by category name (e.g. "Harmful behaviors").

    Returns:
        List of dicts with keys: goal, target, category, index.
    """
    if path and Path(path).exists():
        return _load_local(path, max_samples)

    try:
        from datasets import load_dataset as hf_load
        # config_name="behaviors", split="harmful" (100 adversarial behaviors)
        ds = hf_load("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        samples = []
        for item in ds:
            goal = item.get("Goal") or item.get("goal") or item.get("Behavior") or ""
            cat = item.get("Category") or item.get("category", "")
            if category and cat and category.lower() not in cat.lower():
                continue
            samples.append({
                "goal": goal,
                "target": "Sure, here is",
                "category": cat,
                "index": item.get("Index") or item.get("index", ""),
                "source": "JailbreakBench",
            })
            if max_samples and len(samples) >= max_samples:
                break
        if samples:
            return samples
    except Exception:
        pass

    raise FileNotFoundError(
        "JailbreakBench dataset not found. Options:\n"
        "  1. pip install datasets  (loads from HuggingFace)\n"
        "  2. Download from https://github.com/JailbreakBench/jailbreakbench"
    )


def load_mixed(mix: List[Dict]) -> List[Dict]:
    """Load samples from multiple datasets and combine them.

    Args:
        mix: List of dicts with 'name' (dataset name) and 'samples' (count) keys.

    Returns:
        Combined list of prompt dicts in the order specified.
    """
    combined = []
    for entry in mix:
        name = entry.get("name", "")
        n = entry.get("samples")
        combined.extend(load_dataset(name, max_samples=n))
    return combined


def load_dataset(name: str, path: Optional[str] = None, max_samples: Optional[int] = None,
                 mix: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Load a named dataset.

    Args:
        name: Dataset name ('advbench', 'harmbench', 'jailbreakbench', 'mixed', or path).
        path: Optional explicit file path.
        max_samples: Maximum number of samples.
        mix: Required when name='mixed'. List of {'name': ..., 'samples': ...} dicts.

    Returns:
        List of prompt dictionaries.
    """
    name_lower = name.lower().replace("-", "").replace("_", "")
    if name_lower == "mixed":
        if not mix:
            raise ValueError("dataset='mixed' requires a dataset_mix list in config")
        return load_mixed(mix)
    elif name_lower == "advbench":
        return load_advbench(path=path, max_samples=max_samples)
    elif name_lower in ("harmbench", "hb"):
        return load_harmbench(path=path, max_samples=max_samples)
    elif name_lower in ("jailbreakbench", "jbb"):
        return load_jailbreakbench(path=path, max_samples=max_samples)
    elif path:
        return _load_local(path, max_samples)
    else:
        return _load_local(name, max_samples)
