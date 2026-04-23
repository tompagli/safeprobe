"""Composite attack — Competing Objectives × Mismatched Generalization.

This attack systematically tests every combination of a CO technique with an
MG technique against a target model, ranks them by Attack Success Rate, and
identifies which pairings are most effective for a given model.

CO techniques (competing_objectives.py):
  prefix_injection, refusal_suppression, style_injection, roleplay

MG techniques (mismatched_generalization.py):
  base64, rot13, leetspeak, pig_latin, translation

Total default combinations: 4 × 5 = 20 per prompt.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from safeprobe.config import Config
from safeprobe.datasets.adapters import get_adapter
from safeprobe.datasets.prompts import load_dataset
from safeprobe.utils.logging import get_logger
from .composer import build_combinations, run_combinations, aggregate_results
from .transformers.competing_objectives import CO_TECHNIQUES
from .transformers.mismatched_generalization import MG_TECHNIQUES

logger = get_logger(__name__)


class CompositeAttack:
    """
    Composite attack: exhaustive CO × MG combination testing.

    Competing Objectives (CO) pressure the model to comply by injecting
    instructions that compete with its safety training.

    Mismatched Generalization (MG) obfuscates the prompt payload using
    encodings that may evade surface-level safety filters.

    Results include per-combination ASR rankings, so researchers can
    identify which pairings are most effective against a given model.
    """

    name = "Composite"
    description = "CO × MG combination attack — exhaustive technique pairing with ASR ranking"
    category = "automated"

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def get_parameters(self) -> Dict[str, Any]:
        print(f"\n=== {self.name} Configuration ===")
        print(f"CO techniques: {', '.join(CO_TECHNIQUES)}")
        print(f"MG techniques: {', '.join(MG_TECHNIQUES)}")
        co_input = input("CO techniques (comma-sep, Enter=all): ").strip()
        mg_input = input("MG techniques (comma-sep, Enter=all): ").strip()
        use_judge = input("Use LLM judge? [y/N]: ").strip().lower() == "y"
        return {
            "target_model": (
                input(f"Target model [{self.config.target_model}]: ").strip()
                or self.config.target_model
            ),
            "target_model_type": (
                input(f"Model type [{self.config.target_model_type}]: ").strip()
                or self.config.target_model_type
            ),
            "co_techniques": [t.strip() for t in co_input.split(",")] if co_input else None,
            "mg_techniques": [t.strip() for t in mg_input.split(",")] if mg_input else None,
            "dataset": input("Dataset [advbench]: ").strip() or "advbench",
            "sample_size": int(input("Sample size [10]: ").strip() or "10"),
            "use_judge": use_judge,
            "delay": float(input("Delay between requests (s) [1.0]: ").strip() or "1.0"),
            "output_file": (
                input("Output file [composite_results.json]: ").strip()
                or "composite_results.json"
            ),
        }

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "target_model": self.config.target_model,
            "target_model_type": self.config.target_model_type,
            "co_techniques": None,   # all
            "mg_techniques": None,   # all
            "dataset": self.config.dataset or "advbench",
            "sample_size": self.config.sample_size or 10,
            "dataset_mix": getattr(self.config, "dataset_mix", None),
            "use_judge": False,
            "delay": 1.0,
            "output_file": str(self.config.results_dir / "composite_results.json"),
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run the composite attack.

        Args:
            params: Dict produced by get_parameters() or get_default_parameters().

        Returns:
            Dict with keys: technique, success, summary, output_file  (or error).
        """
        try:
            # --- Target adapter -------------------------------------------
            target_adapter = get_adapter(
                params["target_model"],
                params["target_model_type"],
                self.config.get_api_key(params["target_model_type"]),
            )

            # --- Dataset --------------------------------------------------
            raw = load_dataset(
                params.get("dataset", "advbench"),
                max_samples=params.get("sample_size"),
                mix=params.get("dataset_mix"),
            )
            prompts: List[str] = [d["goal"] for d in raw if d.get("goal")]
            logger.info(f"Loaded {len(prompts)} prompts from {params.get('dataset')}")

            # --- Judge (optional) -----------------------------------------
            judge = None
            if params.get("use_judge"):
                from safeprobe.analysis.judges import JailbreakJudge
                judge = JailbreakJudge(self.config)
                logger.info(f"Judge model: {self.config.judge_model}")

            # --- Combinations ---------------------------------------------
            combinations = build_combinations(
                co_techniques=params.get("co_techniques"),
                mg_techniques=params.get("mg_techniques"),
            )
            n_requests = len(combinations) * len(prompts)
            logger.info(
                f"Testing {len(combinations)} combos × {len(prompts)} prompts "
                f"= {n_requests} total requests"
            )

            # --- Execute --------------------------------------------------
            entries = run_combinations(
                prompts=prompts,
                target_adapter=target_adapter,
                combinations=combinations,
                judge=judge,
                delay=params.get("delay", 1.0),
            )

            # --- Aggregate ------------------------------------------------
            summary = aggregate_results(entries)

            # --- Persist --------------------------------------------------
            out_path = params.get("output_file", "composite_results.json")
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

            output = {
                "technique": self.name,
                "target_model": params["target_model"],
                "co_techniques": params.get("co_techniques") or list(CO_TECHNIQUES),
                "mg_techniques": params.get("mg_techniques") or list(MG_TECHNIQUES),
                "summary": summary,
                "entries": entries,
                "timestamp": datetime.now().isoformat(),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Results saved to {out_path} | "
                f"Overall ASR: {summary['asr']} | "
                f"Best combo: {summary['best_combination']}"
            )

            return {
                "technique": self.name,
                "success": True,
                "summary": summary,
                "output_file": out_path,
            }

        except Exception as exc:
            logger.error(f"Composite attack failed: {exc}", exc_info=True)
            return {"technique": self.name, "success": False, "error": str(exc)}
