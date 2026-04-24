"""
Critic agent for evaluating candidate suffixes.
Returns strengths, weaknesses, trade-off profile, recommendation.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from more_ast.utils import load_toml


class Critic:
    """Critic agent: evaluates each suffix candidate."""

    def __init__(self, prompts_path: Path):
        data = load_toml(prompts_path)
        self.prompt_template = data["critic"]["prompt"].strip()

    def fill(
        self,
        base_prompt: str,
        suffix: str,
        sample_outputs: List[str],
        metric_scores: Dict[str, float],
        analyzer_analysis: str = "",
    ) -> str:
        """Build the critic prompt for one suffix."""
        outputs_str = "\n---\n".join(str(o) for o in sample_outputs[:5])
        scores_str = ", ".join(f"{k}={v:.4f}" for k, v in metric_scores.items())
        prompt = self.prompt_template.format(
            base_prompt=base_prompt[:300] + "..." if len(base_prompt) > 300 else base_prompt,
            suffix=suffix,
            sample_outputs=outputs_str[:1500] + "..." if len(outputs_str) > 1500 else outputs_str,
            metric_scores=scores_str,
        )
        if analyzer_analysis:
            prompt += (
                "\n\nShared analyzer context for this optimization step:\n"
                f"{analyzer_analysis[:1500]}"
            )
        return prompt

    def run(self, meta_llm: Any, filled_prompt: str) -> str:
        """Call meta LLM and return raw critique text."""
        return meta_llm.generate(filled_prompt)

    def extract_suggestion(self, critique: str) -> str:
        """Extract a short actionable suggestion from the critique text."""
        patterns = [
            r"recommendation\s*:\s*(.+)",
            r"suggestion\s*:\s*(.+)",
            r"next step\s*:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, critique, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        lines = [line.strip("- ").strip() for line in critique.splitlines() if line.strip()]
        return lines[-1][:300] if lines else ""
