"""
Receptive Suffix Optimizer.
Reads history of suffixes, critiques, suggestions; generates new suffix candidates.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from more_ast.core.suffix import SuffixCandidate
from more_ast.utils import load_toml


class ReceptiveSuffixOptimizer:
    """Optimizer agent: generates new suffixes from history."""

    def __init__(self, prompts_path: Path):
        data = load_toml(prompts_path)
        self.prompt_template = data["optimizer"]["prompt"].strip()

    def fill(
        self,
        base_prompt: str,
        current_suffix: str,
        top_k_suffixes: List[SuffixCandidate],
        analyzer_analysis: str,
        multi_metric_ranking: str,
        history: List[Dict[str, Any]],
        num_candidates: int,
    ) -> str:
        """Build the optimizer prompt."""
        lines = []
        for i, c in enumerate(top_k_suffixes, 1):
            lines.append(f"[{i}] Suffix: {c.suffix}")
            lines.append(f"    Scores: {c.scores}")
            lines.append(f"    Critique: {c.critique[:300]}..." if len(c.critique) > 300 else f"    Critique: {c.critique}")
            if c.suggestions:
                lines.append(f"    Suggestion: {c.suggestions}")
            lines.append("")
        suffixes_str = "\n".join(lines)
        history_lines = []
        for item in history[-8:]:
            history_lines.append(
                " | ".join(
                    [
                        f"step={item.get('step')}",
                        f"suffix={item.get('suffix', '')[:120]}",
                        f"train={item.get('train_scores', {})}",
                        f"dev={item.get('dev_scores', {})}",
                        f"suggestion={item.get('suggestion', '')[:200]}",
                    ]
                )
            )
        history_str = "\n".join(history_lines) if history_lines else "No prior history yet."

        prompt = self.prompt_template.format(
            base_prompt=base_prompt[:400] + "..." if len(base_prompt) > 400 else base_prompt,
            suffixes_with_critiques=suffixes_str,
            analyzer_analysis=analyzer_analysis[:1500] if len(analyzer_analysis) > 1500 else analyzer_analysis,
            multi_metric_ranking=multi_metric_ranking,
        )
        prompt += (
            "\n\nCurrent incumbent suffix:\n"
            f"{current_suffix}\n\n"
            "Recent optimization history:\n"
            f"{history_str}\n\n"
            f"Generate exactly {num_candidates} distinct candidate suffixes."
        )
        return prompt

    def parse_suffixes(
        self,
        response: str,
        base_prompt: str,
        step: int,
        limit: int,
    ) -> List[SuffixCandidate]:
        """Parse σ_new_1:, σ_new_2:, ... from LLM response."""
        candidates = []
        seen = set()
        # Match σ_new_1: or σ_new_2: etc., or numbered lines
        pattern = r"(?:σ_new_\d+:|suffix_\d+:|^\d+\.)\s*(.+?)(?=(?:σ_new_\d+:|suffix_\d+:|^\d+\.)|$)"
        matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
        for m in matches:
            s = m.strip().strip("`").strip()
            normalized = re.sub(r"\s+", " ", s)
            if s and len(s) > 5 and normalized not in seen:
                seen.add(normalized)
                candidates.append(SuffixCandidate(base_prompt=base_prompt, suffix=s, step=step))
        # Fallback: split by newlines, take non-empty lines as candidates
        if not candidates:
            for line in response.split("\n"):
                line = line.strip().strip("`").strip()
                normalized = re.sub(r"\s+", " ", line)
                if line and not line.startswith("#") and len(line) > 10 and normalized not in seen:
                    seen.add(normalized)
                    candidates.append(SuffixCandidate(base_prompt=base_prompt, suffix=line, step=step))
        return candidates[:limit]

    def run(
        self,
        meta_llm: Any,
        base_prompt: str,
        current_suffix: str,
        top_k_suffixes: List[SuffixCandidate],
        analyzer_analysis: str,
        multi_metric_ranking: str,
        history: List[Dict[str, Any]],
        num_candidates: int,
        step: int,
    ) -> List[SuffixCandidate]:
        """Generate new suffix candidates."""
        filled = self.fill(
            base_prompt,
            current_suffix,
            top_k_suffixes,
            analyzer_analysis,
            multi_metric_ranking,
            history,
            num_candidates,
        )
        response = meta_llm.generate(filled)
        return self.parse_suffixes(response, base_prompt, step, limit=num_candidates)
