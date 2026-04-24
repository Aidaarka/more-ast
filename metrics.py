"""
Metric wrappers for MoRe-AST.
Uses rouge_score and optionally bert_score.
"""

from typing import Any, Union

try:
    from rouge_score.rouge_scorer import RougeScorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from experiments.summarization.metrics.align_score import AlignScorePrecisionX
    HAS_ALIGNSCORE = True
except ImportError:
    HAS_ALIGNSCORE = False


class _BaseMetric:
    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        raise NotImplementedError


class Rouge1Metric(_BaseMetric):
    def __init__(self):
        if not HAS_ROUGE:
            raise ImportError("Install rouge-score: pip install rouge-score")
        self.scorer = RougeScorer(["rouge1"], use_stemmer=True)

    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        s = self.scorer.score(gold, pred)
        return s["rouge1"].fmeasure


class Rouge2Metric(_BaseMetric):
    def __init__(self):
        if not HAS_ROUGE:
            raise ImportError("Install rouge-score: pip install rouge-score")
        self.scorer = RougeScorer(["rouge2"], use_stemmer=True)

    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        s = self.scorer.score(gold, pred)
        return s["rouge2"].fmeasure


class RougeLMetric(_BaseMetric):
    def __init__(self):
        if not HAS_ROUGE:
            raise ImportError("Install rouge-score: pip install rouge-score")
        self.scorer = RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        s = self.scorer.score(gold, pred)
        return s["rougeL"].fmeasure


class BERTScoreMetric(_BaseMetric):
    def __init__(self):
        if not HAS_BERTSCORE:
            raise ImportError("Install bert-score: pip install bert-score")

    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        _, _, f1 = bert_score_fn([pred], [gold], lang="en", verbose=False)
        return float(f1[0])


class FaithfulnessMetric(_BaseMetric):
    def __init__(self):
        if not HAS_ALIGNSCORE:
            raise ImportError(
                "Install AlignScore dependencies and ensure CriSPO is on PYTHONPATH"
            )
        self.metric = AlignScorePrecisionX()

    def score(self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None) -> float:
        return float(self.metric.score(pred, gold, x))


def build_metrics(active: list) -> dict:
    """Build metric dict from config active list."""
    registry = {
        "rouge1": Rouge1Metric,
        "rouge2": Rouge2Metric,
        "rougeL": RougeLMetric,
        "bertscore": BERTScoreMetric,
        "faithfulness": FaithfulnessMetric,
    }
    result = {}
    for name in active:
        if name in registry:
            try:
                result[name] = registry[name]()
            except ImportError as e:
                import warnings
                warnings.warn(f"Skipping metric {name}: {e}")
    if not result and HAS_ROUGE:
        result = {"rouge1": Rouge1Metric()}  # Fallback
    if not result:
        raise ImportError("No metrics available. Install rouge-score: pip install rouge-score")
    return result
