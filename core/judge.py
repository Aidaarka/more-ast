"""
Judge agent for pairwise multi-objective evaluation.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from more_ast.utils import load_toml


class Judge:
    """Judge agent: pairwise comparison of two prompts."""

    def __init__(self, prompts_path: Path):
        data = load_toml(prompts_path)
        self.prompt_template = data["judge"]["prompt"].strip()

    def fill(
        self,
        prompt_a: str,
        prompt_b: str,
        outputs_a: List[str],
        outputs_b: List[str],
        metric_scores_a: Dict[str, float],
        metric_scores_b: Dict[str, float],
    ) -> str:
        """Build the judge prompt."""
        out_a = "\n".join(outputs_a[:5])
        out_b = "\n".join(outputs_b[:5])
        scores_a = ", ".join(f"{k}={v:.4f}" for k, v in metric_scores_a.items())
        scores_b = ", ".join(f"{k}={v:.4f}" for k, v in metric_scores_b.items())
        return self.prompt_template.format(
            prompt_a=prompt_a[:500] + "..." if len(prompt_a) > 500 else prompt_a,
            prompt_b=prompt_b[:500] + "..." if len(prompt_b) > 500 else prompt_b,
            outputs_a=out_a[:1000],
            outputs_b=out_b[:1000],
            metric_scores_a=scores_a,
            metric_scores_b=scores_b,
        )

    def parse_winner(self, response: str) -> str:
        """Extract A, B, or Tie from response."""
        response_lower = response.lower()
        normalized = re.sub(r"[*_`]+", "", response_lower)
        patterns = [
            r"pairwise\s+winner\s*:\s*(a|b|tie)\b",
            r"\bwinner\s*:\s*(a|b|tie)\b",
            r"\bwinner\s+is\s+(a|b|tie)\b",
            r"\bchoose\s+(a|b)\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, normalized, re.I)
            if not m:
                continue
            winner = m.group(1).strip().upper()
            if winner in {"A", "B", "TIE"}:
                return "Tie" if winner == "TIE" else winner

        # Try regex on raw response as a fallback
        m = re.search(r"pairwise winner:\s*(\w+)", response, re.I)
        if m:
            winner = m.group(1).strip().upper()
            if winner in {"A", "B", "TIE"}:
                return "Tie" if winner == "TIE" else winner
        if re.search(r"\btie\b", normalized):
            return "Tie"
        return "Tie"

    def run(
        self,
        meta_llm: Any,
        prompt_a: str,
        prompt_b: str,
        outputs_a: List[str],
        outputs_b: List[str],
        metric_scores_a: Dict[str, float],
        metric_scores_b: Dict[str, float],
    ) -> Tuple[str, str]:
        """Run judge and return (winner, justification)."""
        filled = self.fill(
            prompt_a, prompt_b,
            outputs_a, outputs_b,
            metric_scores_a, metric_scores_b,
        )
        response = meta_llm.generate(filled)
        winner = self.parse_winner(response)
        return winner, response
