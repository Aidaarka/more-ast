"""
Suffix candidate for MoRe-AST.
Holds base prompt, suffix text, scores, critique, and suggestions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SuffixCandidate:
    """A candidate suffix σ to append to base prompt P*."""

    base_prompt: str
    suffix: str
    scores: Dict[str, float] = field(default_factory=dict)
    critique: str = ""
    suggestions: str = ""
    step: int = 0

    def full_prompt(self) -> str:
        """Return P* + σ as a single string."""
        if not self.suffix.strip():
            return self.base_prompt.strip()
        return f"{self.base_prompt.strip()}\n\n{self.suffix.strip()}"

    def short_str(self, max_len: int = 60) -> str:
        """Short representation for logging."""
        s = self.suffix.strip()
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s

    def __str__(self) -> str:
        return self.full_prompt()

    def __hash__(self) -> int:
        return hash((self.base_prompt, self.suffix))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SuffixCandidate):
            return False
        return self.base_prompt == other.base_prompt and self.suffix == other.suffix
