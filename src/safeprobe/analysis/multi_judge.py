"""MultiJudge — run several judge backends on the same dataset and compute agreement.

Sequential mode (recommended for local GPU setups):
  Each judge is loaded, runs on every row, then unloaded before the next is loaded.
  This keeps only one ~8-16 GB model in VRAM at a time.

Simultaneous mode (API-only or high-VRAM systems):
  All judges are loaded at once.  Use only when all judges fit in VRAM together.

Usage (sequential — default for local models):

    mj = MultiJudge.build_from_config(config, sequential=True)
    df, agreement = mj.judge_csv("results/consolidated.csv")
    print(format_agreement_report(agreement))
"""
import gc
import json
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from safeprobe.utils.logging import get_logger
from .agreement import compute_agreement, format_agreement_report

logger = get_logger(__name__)

# A judge entry is either an already-instantiated judge or a zero-arg factory.
_JudgeOrFactory = Union[Any, Callable[[], Any]]


def _free_gpu() -> None:
    """Run GC and release cached VRAM back to CUDA allocator."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_factory(obj: Any) -> bool:
    return callable(obj) and not hasattr(obj, "judge_single")


class MultiJudge:
    """Run multiple judge backends on a shared dataset.

    Each judge column is stored as:
      judge_{name}_result           — YES | NO | AMBIGUOUS | ERROR
      judge_{name}_justification    — short reason
      judge_{name}_reasoning_chain  — full raw output

    After all judges finish, inter-rater agreement (Cohen's κ, Fleiss' κ)
    is computed and returned alongside the augmented DataFrame.

    Args:
        judges:     {name: judge_instance_or_factory}
                    Pass callables (zero-arg factories) when sequential=True.
        delay:      Seconds to sleep between calls (per judge, per row).
        sequential: If True, load one judge at a time — instantiate, run all
                    rows, free VRAM, then move to the next judge.
                    Required when judges do not all fit in VRAM simultaneously.
    """

    def __init__(
        self,
        judges: Dict[str, _JudgeOrFactory],
        delay: float = 0.5,
        sequential: bool = False,
    ):
        self.judges = judges
        self.delay = delay
        self.sequential = sequential

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for name in self.judges:
            for suffix in ("result", "justification", "reasoning_chain"):
                col = f"judge_{name}_{suffix}"
                if col not in df.columns:
                    df[col] = pd.Series([None] * len(df), dtype="object")
                else:
                    df[col] = df[col].astype("object")
        return df

    def _save_agreement(self, df: pd.DataFrame, output_path: str) -> Dict:
        judge_results: Dict[str, List[str]] = {
            name: df[f"judge_{name}_result"].fillna("ERROR").tolist()
            for name in self.judges
        }
        agreement = compute_agreement(judge_results)
        agree_path = output_path.replace(".csv", "_agreement.json")
        with open(agree_path, "w", encoding="utf-8") as f:
            json.dump(
                {**agreement, "timestamp": datetime.now().isoformat()},
                f, indent=2, ensure_ascii=False,
            )
        logger.info(f"Agreement report saved to {agree_path}")
        return agreement

    # ── Sequential mode (one model in VRAM at a time) ─────────────────────────

    def _judge_sequential(
        self,
        df: pd.DataFrame,
        attack_col: str,
        response_col: str,
        skip_existing: bool,
        output_path: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        """Load one judge, run it on every row, free VRAM, repeat."""
        n = len(df)

        for judge_name, judge_or_factory in self.judges.items():
            logger.info(f"=== Sequential judge: {judge_name} ({n} rows) ===")

            try:
                judge = judge_or_factory() if _is_factory(judge_or_factory) else judge_or_factory
            except Exception as exc:
                logger.error(f"Failed to load judge '{judge_name}': {exc}")
                # Fill all rows with ERROR so downstream agreement still works
                for idx in df.index:
                    df.at[idx, f"judge_{judge_name}_result"] = "ERROR"
                    df.at[idx, f"judge_{judge_name}_justification"] = str(exc)
                    df.at[idx, f"judge_{judge_name}_reasoning_chain"] = ""
                continue

            result_col = f"judge_{judge_name}_result"
            processed = 0

            for idx, row in df.iterrows():
                if skip_existing and pd.notna(df.at[idx, result_col]):
                    continue

                attack_prompt = str(row.get(attack_col, ""))
                response_prompt = str(row.get(response_col, ""))

                try:
                    verdict = judge.judge_single(attack_prompt, response_prompt)
                except Exception as exc:
                    import traceback
                    logger.warning(
                        f"Judge '{judge_name}' row {idx}: "
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                    )
                    verdict = {
                        "result": "ERROR",
                        "justification": f"{type(exc).__name__}: {exc}",
                        "reasoning_chain": "",
                        "error": str(exc),
                    }

                df.at[idx, f"judge_{judge_name}_result"] = verdict["result"]
                df.at[idx, f"judge_{judge_name}_justification"] = verdict.get("justification", "")
                df.at[idx, f"judge_{judge_name}_reasoning_chain"] = verdict.get("reasoning_chain", "")
                processed += 1

                logger.info(f"  [{int(idx) + 1}/{n}] {judge_name}={verdict['result']}")

                if output_path and processed % 10 == 0:
                    df.to_csv(output_path, index=False, encoding="utf-8")
                    logger.info(f"  Checkpoint at row {int(idx) + 1}")

                time.sleep(self.delay)

            # Save after this judge completes
            if output_path:
                df.to_csv(output_path, index=False, encoding="utf-8")

            # Free GPU memory before loading next judge.
            # del must happen here (in the scope that holds the reference) —
            # passing `judge` to a helper and del-ing the parameter there does
            # nothing to the reference count in this frame.
            del judge
            _free_gpu()
            logger.info(f"=== {judge_name} complete — VRAM freed ===\n")

        agreement = {}
        if output_path:
            agreement = self._save_agreement(df, output_path)
        else:
            judge_results = {
                name: df[f"judge_{name}_result"].fillna("ERROR").tolist()
                for name in self.judges
            }
            agreement = compute_agreement(judge_results)

        return df, agreement

    # ── Simultaneous mode (all models in VRAM at once) ────────────────────────

    def judge_single(
        self,
        attack_prompt: str,
        response_prompt: str,
    ) -> Dict[str, Dict]:
        """Run all loaded judges on one entry (simultaneous mode only)."""
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

    def _judge_simultaneous(
        self,
        df: pd.DataFrame,
        attack_col: str,
        response_col: str,
        skip_existing: bool,
        output_path: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict]:
        """All judges run on each row (original behaviour)."""
        n = len(df)
        logger.info(
            f"MultiJudge (simultaneous): {n} rows × {len(self.judges)} judges = "
            f"{n * len(self.judges)} calls"
        )
        first_result_col = f"judge_{list(self.judges.keys())[0]}_result"

        for idx, row in df.iterrows():
            if skip_existing and pd.notna(row.get(first_result_col)):
                continue

            attack_prompt = str(row.get(attack_col, ""))
            response_prompt = str(row.get(response_col, ""))
            verdicts = self.judge_single(attack_prompt, response_prompt)

            for name, verdict in verdicts.items():
                df.at[idx, f"judge_{name}_result"] = verdict["result"]
                df.at[idx, f"judge_{name}_justification"] = verdict.get("justification", "")
                df.at[idx, f"judge_{name}_reasoning_chain"] = verdict.get("reasoning_chain", "")

            results_str = " | ".join(f"{n}={v['result']}" for n, v in verdicts.items())
            logger.info(f"  [{int(idx) + 1}/{n}] {results_str}")

            if output_path and (int(idx) + 1) % 10 == 0:
                df.to_csv(output_path, index=False, encoding="utf-8")
                logger.info(f"Checkpoint saved at row {int(idx) + 1}")

            time.sleep(self.delay)

        if output_path:
            df.to_csv(output_path, index=False, encoding="utf-8")

        agreement = {}
        if output_path:
            agreement = self._save_agreement(df, output_path)
        else:
            judge_results = {
                name: df[f"judge_{name}_result"].fillna("ERROR").tolist()
                for name in self.judges
            }
            agreement = compute_agreement(judge_results)

        return df, agreement

    # ── Public API ────────────────────────────────────────────────────────────

    def judge_dataframe(
        self,
        df: pd.DataFrame,
        attack_col: str = "attack_prompt",
        response_col: str = "response_prompt",
        skip_existing: bool = True,
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Run all judges on every row of a DataFrame.

        Routes to sequential or simultaneous mode based on self.sequential.
        """
        df = df.copy()
        df = self._ensure_columns(df)

        if self.sequential:
            return self._judge_sequential(df, attack_col, response_col, skip_existing, output_path)
        return self._judge_simultaneous(df, attack_col, response_col, skip_existing, output_path)

    def judge_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Load a CSV, judge every row, write results, return (df, agreement)."""
        df = pd.read_csv(csv_path, encoding="utf-8")
        out = output_path or csv_path
        df, agreement = self.judge_dataframe(df, output_path=out)
        return df, agreement

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def build_from_config(config, sequential: bool = True) -> "MultiJudge":
        """Build a MultiJudge from a Config object.

        Args:
            config:     SafeProbe Config with .judges, .llamaguard_model, etc.
            sequential: Default True — load one judge at a time (recommended
                        for local GPU setups with limited VRAM).
                        Set False only when all judges fit in VRAM simultaneously.

        Supported judge names: "deepseek", "llamaguard", "harmbench"
        """
        from safeprobe.analysis.judges import JailbreakJudge, LlamaGuardJudge, HarmBenchJudge

        judge_names = getattr(config, "judges", ["deepseek"]) or ["deepseek"]
        delay = getattr(config, "judge_delay", 0.5)

        # Collect per-judge settings before building factories
        # (lambda default-arg capture avoids loop-variable closure bugs)
        judges: Dict[str, _JudgeOrFactory] = {}

        for jname in judge_names:
            jname_lower = jname.lower().replace("-", "").replace("_", "")

            if jname_lower in ("deepseek", "cot", "api", "default"):
                if sequential:
                    judges["DeepSeek"] = lambda _c=config: JailbreakJudge(_c)
                else:
                    try:
                        judges["DeepSeek"] = JailbreakJudge(config)
                    except Exception as exc:
                        logger.error(f"Failed to load DeepSeek judge: {exc}")

            elif jname_lower in ("llamaguard", "llama_guard", "llamaguard3"):
                model = getattr(config, "llamaguard_model", "meta-llama/Llama-Guard-3-8B")
                load_4bit = getattr(config, "load_in_4bit", False)
                hf_token = getattr(config, "hf_token", None)
                if sequential:
                    judges["LlamaGuard"] = lambda _m=model, _l=load_4bit, _h=hf_token: (
                        LlamaGuardJudge(model_name=_m, load_in_4bit=_l, hf_token=_h)
                    )
                else:
                    try:
                        judges["LlamaGuard"] = LlamaGuardJudge(
                            model_name=model, load_in_4bit=load_4bit, hf_token=hf_token
                        )
                    except Exception as exc:
                        logger.error(f"Failed to load LlamaGuard judge: {exc}")

            elif jname_lower in ("harmbench", "hb"):
                model = getattr(config, "harmbench_model", "cais/HarmBench-Llama-2-13b-cls")
                # HarmBench-Llama-2-13b requires ~26 GB fp16 — always 4-bit.
                if sequential:
                    judges["HarmBench"] = lambda _m=model: HarmBenchJudge(model_name=_m, load_in_4bit=True)
                else:
                    try:
                        judges["HarmBench"] = HarmBenchJudge(model_name=model, load_in_4bit=True)
                    except Exception as exc:
                        logger.error(f"Failed to load HarmBench judge: {exc}")

            else:
                logger.warning(f"Unknown judge name: {jname!r} — skipping.")

        if not judges:
            logger.warning("No judges configured — falling back to DeepSeek.")
            if sequential:
                judges["DeepSeek"] = lambda _c=config: JailbreakJudge(_c)
            else:
                judges["DeepSeek"] = JailbreakJudge(config)

        return MultiJudge(judges, delay=delay, sequential=sequential)
