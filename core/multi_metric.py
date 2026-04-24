"""
Multi-metric rank-based aggregator for MoRe-AST.
Ranks candidates per metric, then uses average rank (lower = better).
"""

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from crispo.metrics.metric import Metric


class MultiMetricRanker:
    """
    Wraps multiple Metric instances.
    Computes per-metric scores, then aggregate rank across metrics.
    Lower average rank = better (rank 1 = best for each metric).
    """

    def __init__(self, primary: str = "rank", **metrics: Metric):
        self.primary = primary
        self.metrics = metrics
        self.metric_names = list(metrics.keys())

    def score(
        self,
        pred: Union[str, Any],
        gold: Union[str, Any],
        x: Union[str, Any] = None,
    ) -> Dict[str, float]:
        """Compute per-metric scores for a single prediction."""
        return {
            name: m.score(pred, gold, x)
            for name, m in self.metrics.items()
        }

    def aggregate(self, scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-metric scores across examples."""
        per_metric = defaultdict(list)
        for s in scores_list:
            for name, val in s.items():
                per_metric[name].append(val)
        return {
            name: statistics.mean(vals)
            for name, vals in per_metric.items()
        }

    def compute_rank_scores(
        self,
        candidate_scores: List[Dict[str, float]],
        metric_names: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Given N candidates with per-metric scores, compute average rank per candidate.
        Lower rank = better. Returns list of (negative avg_rank) for sorting (higher = better).
        """
        active_metric_names = metric_names or self.metric_names
        if not candidate_scores or not active_metric_names:
            return [0.0] * len(candidate_scores)

        ranks = []
        for name in active_metric_names:
            vals = [s.get(name, 0.0) for s in candidate_scores]
            # Higher score = better for most metrics (ROUGE, etc.)
            sorted_indices = sorted(
                range(len(vals)),
                key=lambda i: vals[i],
                reverse=True,
            )
            rank_per_idx = [0] * len(vals)
            for rank, idx in enumerate(sorted_indices, start=1):
                rank_per_idx[idx] = rank
            ranks.append(rank_per_idx)

        # Average rank per candidate (1=best)
        avg_ranks = [
            statistics.mean(r[i] for r in ranks)
            for i in range(len(candidate_scores))
        ]
        # Return negative so higher = better for sorting
        return [-ar for ar in avg_ranks]

    def key(self, score: Union[float, Dict[str, float]]) -> float:
        """Extract sort key from score. Higher = better."""
        if isinstance(score, dict):
            if self.primary == "rank" and "rank" in score:
                return -score["rank"]
            return score.get(self.primary, 0.0)
        return score
