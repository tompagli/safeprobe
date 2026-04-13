"""Abstract base class for all SafeProbe judge backends."""
from abc import ABC, abstractmethod
from typing import Dict


class BaseJudge(ABC):
    """Common interface every judge backend must implement.

    judge_single() must return a dict with at least:
      result        : "YES" | "NO" | "AMBIGUOUS" | "ERROR"
      justification : str  — short human-readable reason
      reasoning_chain: str — full raw output from the model
      error         : str | None
    """

    name: str = "BaseJudge"

    @abstractmethod
    def judge_single(
        self,
        attack_prompt: str,
        response_prompt: str,
        max_retries: int = 3,
    ) -> Dict:
        """Classify one (attack_prompt, response_prompt) pair."""
        ...

    def _error_result(self, msg: str) -> Dict:
        return {
            "result": "ERROR",
            "justification": msg,
            "reasoning_chain": "",
            "error": msg,
        }
