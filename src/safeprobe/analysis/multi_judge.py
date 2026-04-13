"""MultiJudge — run several judge backends on the same dataset and compute agreement.

Usage:

    from safeprobe.analysis.multi_judge import MultiJudge
    from safeprobe.analysis.judges import JailbreakJudge, LlamaGuardJudge, HarmBenchJudge

    judges = {
        "DeepSeek": JailbreakJudge(config),
        "LlamaGuard": LlamaGuardJudge(),
        "HarmBench": HarmBenchJudge(),
    }
    mj = MultiJudge(judges)
    df, agreement = mj.judge_dataframe(df)
    print(format_agreement_report(agreement))
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from safeprobe.utils.logging import get_logger
from .agreement import compute_agreement, format_agreement_report

logger = get_logger(__name__)


class MultiJudge:
    """Run multiple judge backends on a shared set of entries.

    Each judge column is stored as:
      judge_{name}_result           — YES | NO | AMBIGUOUS | ERROR
      judge_{name}_justification    — short reason
      judge_{name}_reasoning_chain  — full raw output

    After all judges finish, inter-rater agreement (Cohen's κ, Fleiss' κ)
    is computed and returned alongside the augmented DataFrame.
    """

    def __init__(self, judges: Dict[str, Any], delay: float = 0.5):
        """
        Args:
            judges: {judge_name: judge_instance} — any object with a
                    judge_single(attack_prompt, response_prompt) -> Dict method.
            delay:  Seconds to sleep between API calls (applies to each judge
                    sequentially, not in aggregate).
        """
        self.judges = judges
        self.delay = delay

    def judge_single(
        self,
        attack_prompt: str,
        response_prompt: str,
    ) -> Dict[str, Dict]:
        """Run all judges on one entry.

        Returns:
            {judge_name: {result, justification, reasoning_chain, error}}
        """
        verdicts: Dict[str, Dict] = {}
        for name, judge in self.judges.items():
            try:
                verdicts[name] = judge.judge_single(attack_prompt, response_prompt)
            except Exception as exc:
                logger.warning(f"Judge '{name}' raised: {exc}")
                verdicts[name] = {
                    "result": "ERROR",
                    "justification": str(exc),
                    "reasoning_chain": "",
                    "error": str(exc),
                }
        return verdicts

    def judge_dataframe(
        self,
        df: pd.DataFrame,
        attack_col: str = "attack_prompt",
        response_col: str = "response_prompt",
        skip_existing: bool = True,
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run all judges on every row of a DataFrame.

        Args:
            df: DataFrame with at least attack_col and response_col.
            attack_col: Column name for adversarial prompts.
            response_col: Column name for model responses.
            skip_existing: If True, skip rows where all judges already
                           have non-null results (checkpoint support).
            output_path: If set, save the augmented DataFrame as CSV
                         and the agreement report as JSON after each
                         checkpoint (every 10 rows).

        Returns:
            (augmented_df, agreement_dict)
        """
        df = df.copy()

        # Ensure result columns exist
        for name in self.judges:
            for suffix in ("result", "justification", "reasoning_chain"):
                col = f"judge_{name}_{suffix}"
                if col not in df.columns:
                    df[col] = pd.Series([None] * len(df), dtype="object")
                else:
                    df[col] = df[col].astype("object")

        n = len(df)
        logger.info(f"MultiJudge: {n} rows × {len(self.judges)} judges = {n * len(self.judges)} calls")

        for idx, row in df.iterrows():
            result_col = f"judge_{list(self.judges.keys())[0]}_result"
            # Skip if first judge already has a result (all judges share same rows)
            if skip_existing and pd.notna(row.get(result_col)):
                continue

            attack_prompt = str(row.get(attack_col, ""))
            response_prompt = str(row.get(response_col, ""))

            verdicts = self.judge_single(attack_prompt, response_prompt)

            for name, verdict in verdicts.items():
                df.at[idx, f"judge_{name}_result"] = verdict["result"]
                df.at[idx, f"judge_{name}_justification"] = verdict.get("justification", "")
                df.at[idx, f"judge_{name}_reasoning_chain"] = verdict.get("reasoning_chain", "")

            results_str = " | ".join(
                f"{n}={v['result']}" for n, v in verdicts.items()
            )
            logger.info(f"  [{int(idx) + 1}/{n}] {results_str}")

            # Checkpoint every 10 rows
            if output_path and (int(idx) + 1) % 10 == 0:
                df.to_csv(output_path, index=False, encoding="utf-8")
                logger.info(f"Checkpoint saved at row {int(idx) + 1}")

            time.sleep(self.delay)

        # Final save
        if output_path:
            df.to_csv(output_path, index=False, encoding="utf-8")

        # Compute agreement
        judge_results: Dict[str, List[str]] = {
            name: df[f"judge_{name}_result"].fillna("ERROR").tolist()
            for name in self.judges
        }
        agreement = compute_agreement(judge_results)

        # Optionally save agreement JSON alongside the CSV
        if output_path:
            agree_path = output_path.replace(".csv", "_agreement.json")
            with open(agree_path, "w", encoding="utf-8") as f:
                json.dump(
                    {**agreement, "timestamp": datetime.now().isoformat()},
                    f, indent=2, ensure_ascii=False,
                )
            logger.info(f"Agreement report saved to {agree_path}")
            logger.info("\n" + format_agreement_report(agreement))

        return df, agreement

    def judge_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Convenience wrapper: load a CSV, judge it, save results.

        Args:
            csv_path: Input CSV (output of `safeprobe consolidate`).
            output_path: Where to write the augmented CSV (default: overwrites input).

        Returns:
            (augmented_df, agreement_dict)
        """
        df = pd.read_csv(csv_path, encoding="utf-8")
        out = output_path or csv_path
        df, agreement = self.judge_dataframe(df, output_path=out)
        print(format_agreement_report(agreement))
        return df, agreement

    @staticmethod
    def build_from_config(config) -> "MultiJudge":
        """Factory: build a MultiJudge from a Config object.

        Uses config.judges (list of judge names) and per-judge model settings.
        Falls back to [DeepSeek] if not set.

        Supported judge names: "deepseek", "llamaguard", "harmbench"
        """
        from safeprobe.analysis.judges import JailbreakJudge, LlamaGuardJudge, HarmBenchJudge

        judge_names = getattr(config, "judges", ["deepseek"]) or ["deepseek"]
        judges: Dict[str, Any] = {}

        for jname in judge_names:
            jname_lower = jname.lower().replace("-", "").replace("_", "")
            try:
                if jname_lower in ("deepseek", "cot", "api", "default"):
                    judges["DeepSeek"] = JailbreakJudge(config)
                elif jname_lower in ("llamaguard", "llama_guard", "llamaguard3"):
                    model = getattr(config, "llamaguard_model", "meta-llama/Llama-Guard-3-8B")
                    load_4bit = getattr(config, "load_in_4bit", False)
                    hf_token = getattr(config, "hf_token", None)
                    judges["LlamaGuard"] = LlamaGuardJudge(
                        model_name=model, load_in_4bit=load_4bit, hf_token=hf_token
                    )
                elif jname_lower in ("harmbench", "hb"):
                    model = getattr(config, "harmbench_model", "cais/HarmBench-Llama-2-13b-cls")
                    # HarmBench-Llama-2-13b requires ~26 GB in fp16 — always use 4-bit.
                    # The global load_in_4bit can still force it on for other judges, but
                    # HarmBench ignores a False value here to avoid OOM.
                    load_4bit = True
                    judges["HarmBench"] = HarmBenchJudge(
                        model_name=model, load_in_4bit=load_4bit
                    )
                else:
                    logger.warning(f"Unknown judge name: {jname!r} — skipping.")
            except Exception as exc:
                logger.error(f"Failed to load judge '{jname}': {exc}")

        if not judges:
            logger.warning("No judges loaded — falling back to DeepSeek.")
            judges["DeepSeek"] = JailbreakJudge(config)

        delay = getattr(config, "judge_delay", 0.5)
        return MultiJudge(judges, delay=delay)
