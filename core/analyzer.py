"""
Analyzer agent for multi-objective prompt optimization.
Identifies metric conflicts, failure patterns, and improvement directions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from more_ast.utils import load_toml


class Analyzer:
    """Analyzer agent: identifies metric conflicts and improvement directions."""

    def __init__(self, prompts_path: Path):
        data = load_toml(prompts_path)
        self.prompt_template = data["analyzer"]["prompt"].strip()

    def fill(
        self,
        base_prompt: str,
        suffix: str,
        outputs: List[str],
        metric_scores: Dict[str, float],
        task_description: str,
        candidate_summaries: Optional[List[Dict[str, Any]]] = None,
        dev_summary: str = "",
    ) -> str:
        """Build the analyzer prompt."""
        outputs_str = "\n---\n".join(outputs[:10])  # Limit to 10 samples
        scores_str = ", ".join(f"{k}={v:.4f}" for k, v in metric_scores.items())

        if candidate_summaries:
            candidate_lines = []
            for item in candidate_summaries:
                candidate_lines.append(f"Candidate: {item['suffix']}")
                candidate_lines.append(f"Train scores: {item['train_scores']}")
                if item.get("dev_scores"):
                    candidate_lines.append(f"Dev scores: {item['dev_scores']}")
                if item.get("sample_outputs"):
                    candidate_lines.append("Sample outputs:")
                    candidate_lines.extend(str(out) for out in item["sample_outputs"][:3])
                candidate_lines.append("")
            outputs_str = "\n".join(candidate_lines)[:3000]
            if dev_summary:
                scores_str = f"{scores_str}\n\nDev summary:\n{dev_summary}"

        return self.prompt_template.format(
            base_prompt=base_prompt[:500] + "..." if len(base_prompt) > 500 else base_prompt,
            suffix=suffix,
            outputs=outputs_str[:2000] + "..." if len(outputs_str) > 2000 else outputs_str,
            metric_scores=scores_str,
            task_description=task_description,
        )

    def run(self, meta_llm: Any, filled_prompt: str) -> str:
        """Call meta LLM and return raw analysis text."""
        return meta_llm.generate(filled_prompt)
