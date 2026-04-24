"""
CNN/DailyMail dataset loader for MoRe-AST.
Uses HuggingFace `datasets` library.
"""

import random
from typing import List, Tuple

from crispo.task.example import Example


class CNNExample(Example):
    """Summarization example: article → highlights."""

    def to_xml(self) -> str:
        return (
            "<article>\n{article}\n</article>\n"
            "<highlights>\n{highlights}\n</highlights>"
        ).format(article=self.x, highlights=self.y)


def load_cnn(split: str) -> List[CNNExample]:
    """Load CNN/DailyMail examples for a given split slice.

    Uses version 2.0.0 to match the CriSPO AST CNN setup.
    """
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset("cnn_dailymail", "2.0.0", split=split)
    return [CNNExample(x=row["article"], y=row["highlights"]) for row in dataset]


def load_cnn_standard() -> Tuple[List[CNNExample], List[CNNExample], List[CNNExample]]:
    """50 train / 50 dev / 500 test (deterministic slice)."""
    train = load_cnn("train[:50]")
    dev = load_cnn("validation[:50]")
    test = load_cnn("test[:500]")
    return train, dev, test


def load_cnn_shuffled(seed: int = 42) -> Tuple[List[CNNExample], List[CNNExample], List[CNNExample]]:
    """50 train / 50 dev / 500 test (randomly sampled from full split)."""
    random.seed(seed)
    train = random.sample(load_cnn("train"), 50)
    dev = random.sample(load_cnn("validation"), 50)
    test = random.sample(load_cnn("test"), 500)
    return train, dev, test


def load_cnn_quick() -> Tuple[List[CNNExample], List[CNNExample], List[CNNExample]]:
    """60 train / 50 dev / 100 test — for quick iteration."""
    train = load_cnn("train[:60]")
    dev = load_cnn("validation[:50]")
    test = load_cnn("test[:100]")
    return train, dev, test


def load_cnn_debug() -> Tuple[List[CNNExample], List[CNNExample], List[CNNExample]]:
    """3 / 3 / 3 — for debugging without API cost."""
    train = load_cnn("train[:3]")
    dev = load_cnn("validation[:3]")
    test = load_cnn("test[:3]")
    return train, dev, test
