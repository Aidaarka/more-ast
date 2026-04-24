"""
MoRe-AST Trainer: Multi-objective Receptive AST.
Main optimization loop with detailed tracing, judge-based dev ranking,
and resumable checkpointing.
"""

import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from more_ast.core.analyzer import Analyzer
from more_ast.core.critic import Critic
from more_ast.core.judge import Judge
from more_ast.core.multi_metric import MultiMetricRanker
from more_ast.core.optimizer import ReceptiveSuffixOptimizer
from more_ast.core.suffix import SuffixCandidate


def default_format_input(base_prompt: str, suffix: str, x: Any) -> str:
    """Default: full prompt + input."""
    full = f"{base_prompt.strip()}\n\n{suffix.strip()}" if suffix.strip() else base_prompt.strip()
    return f"{full}\n\n---\n\nInput:\n{x}\n\nOutput:"


def default_parse_output(generation: str, x: Any = None) -> str:
    """Default: return raw generation."""
    return generation.strip()


class MoReASTTrainer:
    """MoRe-AST optimization trainer."""

    def __init__(
        self,
        save_dir: str,
        base_prompt: str,
        prompts_path: Path,
        task_llm: Any,
        meta_llm: Any,
        ranker: MultiMetricRanker,
        format_input: Callable[[str, str, Any], str] = default_format_input,
        parse_output: Callable[[str, Any], str] = default_parse_output,
    ):
        self.save_dir = Path(save_dir)
        self.base_prompt = base_prompt
        self.prompts_path = prompts_path
        self.task_llm = task_llm
        self.meta_llm = meta_llm
        self.ranker = ranker
        self.format_input = format_input
        self.parse_output = parse_output
        self.analyzer = Analyzer(prompts_path)
        self.critic = Critic(prompts_path)
        self.optimizer = ReceptiveSuffixOptimizer(prompts_path)
        self.judge = Judge(prompts_path)
        self.logger = None
        self.events_path: Optional[Path] = None
        self.checkpoint_path = self.save_dir / "checkpoint.json"

    def _safe_log_text(self, msg: str) -> str:
        replacements = {
            "σ": "sigma",
            "Σ": "Sigma",
            "–": "-",
            "—": "-",
            "―": "-",
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "…": "...",
            "≤": "<=",
            "≥": ">=",
            "‑": "-",
            "→": "->",
        }
        safe = msg
        for old, new in replacements.items():
            safe = safe.replace(old, new)
        return safe.encode("cp1251", errors="replace").decode("cp1251")

    def has_checkpoint(self) -> bool:
        return self.checkpoint_path.exists()

    def _log(self, msg: str, level: str = "info") -> None:
        safe_msg = self._safe_log_text(msg)
        if self.logger:
            getattr(self.logger, level)(safe_msg)
        else:
            print(safe_msg)

    def _json_ready(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, SuffixCandidate):
            return self._snapshot_candidate(value)
        if isinstance(value, dict):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._json_ready(v) for v in value]
        if isinstance(value, tuple):
            return [self._json_ready(v) for v in value]
        return value

    def _llm_config_snapshot(self, llm: Any) -> Dict[str, Any]:
        if hasattr(llm, "config_snapshot"):
            return self._json_ready(llm.config_snapshot())
        keys = [
            "model",
            "max_new_tokens",
            "temperature",
            "concurrency",
            "request_timeout",
            "max_retries",
        ]
        return {
            key: self._json_ready(getattr(llm, key))
            for key in keys
            if hasattr(llm, key)
        }

    def _record_event(self, event_type: str, **payload: Any) -> None:
        if not self.events_path:
            return
        event = {
            "ts": time.time(),
            "event": event_type,
            **self._json_ready(payload),
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _save_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_ready(payload), f, indent=2, ensure_ascii=False)

    def _save_checkpoint(self, payload: Dict[str, Any]) -> None:
        tmp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._json_ready(payload), f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.checkpoint_path)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.checkpoint_path.exists():
            return None
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _snapshot_candidate(self, candidate: SuffixCandidate) -> Dict[str, Any]:
        return {
            "step": candidate.step,
            "suffix": candidate.suffix,
            "scores": dict(candidate.scores),
            "critique": candidate.critique,
            "suggestions": candidate.suggestions,
        }

    def _candidate_from_snapshot(
        self,
        payload: Optional[Dict[str, Any]],
        fallback_suffix: str = "",
    ) -> Optional[SuffixCandidate]:
        if not payload and not fallback_suffix:
            return None
        payload = payload or {}
        return SuffixCandidate(
            base_prompt=self.base_prompt,
            suffix=payload.get("suffix", fallback_suffix),
            scores=dict(payload.get("scores", {})),
            critique=payload.get("critique", ""),
            suggestions=payload.get("suggestions", ""),
            step=payload.get("step", 0),
        )

    def _clone_candidate(self, candidate: SuffixCandidate) -> SuffixCandidate:
        return SuffixCandidate(
            base_prompt=self.base_prompt,
            suffix=candidate.suffix,
            scores=dict(candidate.scores),
            critique=candidate.critique,
            suggestions=candidate.suggestions,
            step=candidate.step,
        )

    def _dev_selection_key(self, scores: Dict[str, float]) -> Tuple[float, float, float]:
        metric_values = [float(scores.get(name, 0.0)) for name in self.ranker.metric_names]
        metric_mean = sum(metric_values) / len(metric_values) if metric_values else 0.0
        judge_win_rate = float(scores.get("judge_win_rate", 0.0))
        rank_score = float(scores.get("rank_score", 0.0))
        return (judge_win_rate, metric_mean, rank_score)

    def _dedupe_candidates(self, candidates: List[SuffixCandidate]) -> List[SuffixCandidate]:
        unique: List[SuffixCandidate] = []
        seen = set()
        for candidate in candidates:
            normalized = " ".join(candidate.suffix.split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(candidate)
        return unique

    def _history_context_candidates(
        self,
        history: List[Dict[str, Any]],
        current_seed: SuffixCandidate,
        limit: int,
    ) -> List[SuffixCandidate]:
        context = [self._clone_candidate(current_seed)]
        seen = {" ".join(current_seed.suffix.split())}
        for item in reversed(history):
            normalized = " ".join(str(item.get("suffix", "")).split())
            if not normalized or normalized in seen:
                continue
            context.append(
                SuffixCandidate(
                    base_prompt=self.base_prompt,
                    suffix=item["suffix"],
                    scores=dict(item.get("train_scores", {})),
                    critique=item.get("critique", ""),
                    suggestions=item.get("suggestion", ""),
                    step=item.get("step", 0),
                )
            )
            seen.add(normalized)
            if len(context) >= limit:
                break
        return context

    def _batch_meta_generate(self, prompts: List[str], desc: str) -> List[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.meta_llm.generate(prompts[0])]
        return self.meta_llm.batch_generate(prompts, desc=desc)

    def _attach_rank_scores(
        self,
        candidates: List[SuffixCandidate],
        score_dicts: List[Dict[str, float]],
        metric_names: Optional[List[str]] = None,
    ) -> None:
        rank_scores = self.ranker.compute_rank_scores(score_dicts, metric_names=metric_names)
        for candidate, agg_scores, rank_score in zip(candidates, score_dicts, rank_scores):
            candidate.scores = dict(agg_scores)
            candidate.scores["rank_score"] = rank_score
            candidate.scores["avg_rank"] = -rank_score

    def _serialize_leaderboard(self, dev_leaderboard: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        serialized = {}
        for suffix, payload in dev_leaderboard.items():
            serialized[suffix] = {
                "candidate": self._snapshot_candidate(payload["candidate"]),
                "scores": dict(payload["scores"]),
                "step": payload["step"],
            }
        return serialized

    def _deserialize_leaderboard(self, payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for suffix, item in payload.items():
            candidate = self._candidate_from_snapshot(item.get("candidate"), fallback_suffix=suffix)
            if candidate is None:
                continue
            result[suffix] = {
                "candidate": candidate,
                "scores": dict(item.get("scores", {})),
                "step": item.get("step", 0),
            }
        return result

    def _load_llm_stats(self, state: Dict[str, Any]) -> None:
        task_stats = state.get("task_llm_stats")
        meta_stats = state.get("meta_llm_stats")
        if task_stats and hasattr(self.task_llm, "load_stats"):
            self.task_llm.load_stats(task_stats)
        if meta_stats and hasattr(self.meta_llm, "load_stats"):
            self.meta_llm.load_stats(meta_stats)

    def _build_checkpoint_payload(
        self,
        completed_step: int,
        num_search_steps: int,
        num_suffix_candidates: int,
        top_k_for_critique: int,
        dev_eval_every_n_steps: int,
        initial_suffix: str,
        current_seed: SuffixCandidate,
        best_dev: Optional[SuffixCandidate],
        best_dev_scores: Optional[Dict[str, float]],
        history: List[Dict[str, Any]],
        stepwise_scores: Dict[str, List[Dict[str, Any]]],
        dev_leaderboard: Dict[str, Dict[str, Any]],
        judge_metric_config: Dict[str, Any],
        completed: bool = False,
        final_best: Optional[SuffixCandidate] = None,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "completed_step": completed_step,
            "num_search_steps": num_search_steps,
            "num_suffix_candidates": num_suffix_candidates,
            "top_k_for_critique": top_k_for_critique,
            "dev_eval_every_n_steps": dev_eval_every_n_steps,
            "initial_suffix": initial_suffix,
            "current_seed": self._snapshot_candidate(current_seed),
            "best_dev": self._snapshot_candidate(best_dev) if best_dev else None,
            "best_dev_scores": dict(best_dev_scores or {}),
            "history": history,
            "stepwise_scores": stepwise_scores,
            "dev_leaderboard": self._serialize_leaderboard(dev_leaderboard),
            "judge_metric_config": judge_metric_config,
            "task_llm_stats": self.task_llm.stats() if hasattr(self.task_llm, "stats") else {},
            "meta_llm_stats": self.meta_llm.stats() if hasattr(self.meta_llm, "stats") else {},
            "completed": completed,
            "final_best": self._snapshot_candidate(final_best) if final_best else None,
        }

    def _judge_metric(
        self,
        candidates: List[SuffixCandidate],
        dev_examples: List[Any],
        dev_predictions_by_suffix: Dict[str, List[str]],
        dev_scores_by_suffix: Dict[str, Dict[str, float]],
        judge_metric_config: Dict[str, Any],
        step: int,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        if not judge_metric_config.get("enabled", False) or len(candidates) < 2:
            return {}, []

        compare_top_k = min(
            max(2, int(judge_metric_config.get("compare_top_k", 3))),
            len(candidates),
        )
        examples_per_comparison = max(1, int(judge_metric_config.get("examples_per_comparison", 5)))
        judge_pool = candidates[:compare_top_k]
        judge_results = {
            candidate.suffix: {
                "judge_win_rate": 0.5,
                "judge_matches": 0,
                "judge_wins": 0.0,
            }
            for candidate in candidates
        }
        judge_logs: List[Dict[str, Any]] = []

        for cand_a, cand_b in combinations(judge_pool, 2):
            outputs_a = dev_predictions_by_suffix.get(cand_a.suffix, [])[:examples_per_comparison]
            outputs_b = dev_predictions_by_suffix.get(cand_b.suffix, [])[:examples_per_comparison]
            if not outputs_a or not outputs_b:
                continue
            winner, justification = self.judge.run(
                self.meta_llm,
                cand_a.full_prompt(),
                cand_b.full_prompt(),
                outputs_a,
                outputs_b,
                dev_scores_by_suffix.get(cand_a.suffix, {}),
                dev_scores_by_suffix.get(cand_b.suffix, {}),
            )
            judge_results[cand_a.suffix]["judge_matches"] += 1
            judge_results[cand_b.suffix]["judge_matches"] += 1
            if winner == "A":
                judge_results[cand_a.suffix]["judge_wins"] += 1.0
            elif winner == "B":
                judge_results[cand_b.suffix]["judge_wins"] += 1.0
            else:
                judge_results[cand_a.suffix]["judge_wins"] += 0.5
                judge_results[cand_b.suffix]["judge_wins"] += 0.5

            log_item = {
                "step": step,
                "candidate_a": cand_a.suffix,
                "candidate_b": cand_b.suffix,
                "winner": winner,
                "justification": justification[:2000],
            }
            judge_logs.append(log_item)
            self._record_event("dev_judge_pair_finished", **log_item)

        for suffix, values in judge_results.items():
            if values["judge_matches"] > 0:
                values["judge_win_rate"] = values["judge_wins"] / values["judge_matches"]
        return judge_results, judge_logs

    def evaluate(
        self,
        candidate: SuffixCandidate,
        examples: List[Any],
        desc: str = "",
        save_path: Optional[Path] = None,
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Evaluate candidate on examples. Returns (aggregate_scores, predictions, prompts)."""
        xs = [e.x for e in examples]
        prompts = [self.format_input(self.base_prompt, candidate.suffix, x) for x in xs]
        stage_name = desc or "Evaluating"
        self._record_event(
            "evaluation_started",
            desc=stage_name,
            candidate=self._snapshot_candidate(candidate),
            num_examples=len(examples),
        )
        generations = self.task_llm.batch_generate(prompts, desc=stage_name)
        predictions = [self.parse_output(gen, x) for gen, x in zip(generations, xs)]
        raw_lengths = [len(gen or "") for gen in generations]
        prediction_lengths = [len(pred or "") for pred in predictions]
        eval_debug = {
            "raw_empty_count": sum(1 for gen in generations if not (gen or "").strip()),
            "prediction_empty_count": sum(1 for pred in predictions if not (pred or "").strip()),
            "raw_length_min": min(raw_lengths) if raw_lengths else 0,
            "raw_length_max": max(raw_lengths) if raw_lengths else 0,
            "raw_length_mean": (sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0,
            "prediction_length_min": min(prediction_lengths) if prediction_lengths else 0,
            "prediction_length_max": max(prediction_lengths) if prediction_lengths else 0,
            "prediction_length_mean": (
                sum(prediction_lengths) / len(prediction_lengths)
            )
            if prediction_lengths
            else 0.0,
        }
        self._record_event(
            "evaluation_generation_stats",
            desc=stage_name,
            candidate=self._snapshot_candidate(candidate),
            **eval_debug,
        )
        scores_list = []
        for pred, ex in tqdm(
            zip(predictions, examples),
            total=len(examples),
            desc="Scoring",
            disable=len(examples) < 20,
        ):
            s = self.ranker.score(pred, ex.y, x=ex.x)
            scores_list.append(s)
        agg = self.ranker.aggregate(scores_list)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for ex, gen, p, s in zip(examples, generations, predictions, scores_list):
                row = {
                    "x": str(ex.x),
                    "y": str(ex.y),
                    "raw_generation": (gen if gen else ""),
                    "raw_generation_length": len(gen or ""),
                    "prediction": (p if p else ""),
                    "prediction_length": len(p or ""),
                }
                row.update(s)
                rows.append(row)
            pd.DataFrame(rows).to_csv(save_path, index=False)
        self._record_event(
            "evaluation_finished",
            desc=stage_name,
            candidate=self._snapshot_candidate(candidate),
            aggregate_scores=agg,
            num_examples=len(examples),
            save_path=str(save_path) if save_path else None,
            **eval_debug,
        )
        return agg, predictions, prompts

    def fit(
        self,
        train: List[Any],
        dev: Optional[List[Any]] = None,
        num_search_steps: int = 20,
        num_suffix_candidates: int = 8,
        top_k_for_critique: int = 5,
        dev_eval_every_n_steps: int = 3,
        initial_suffix: str = "Every word of your summary must be faithful to the input text.",
        task_description: str = "Summarization / text generation",
        judge_metric_config: Optional[Dict[str, Any]] = None,
        enable_checkpointing: bool = True,
        resume: bool = False,
        logger: Any = None,
    ) -> Tuple[SuffixCandidate, Dict]:
        """Run MoRe-AST optimization loop."""
        from crispo.utilities.log_util import init_logger

        judge_metric_config = dict(judge_metric_config or {})
        self.logger = logger or init_logger(name="log.md", save_dir=str(self.save_dir), mode="a")
        os.makedirs(self.save_dir, exist_ok=True)
        self.events_path = self.save_dir / "events.jsonl"

        history: List[Dict[str, Any]] = []
        stepwise_scores: Dict[str, List[Dict[str, Any]]] = {"train": [], "dev": []}
        dev_leaderboard: Dict[str, Dict[str, Any]] = {}
        best_dev: Optional[SuffixCandidate] = None
        best_dev_scores: Optional[Dict[str, float]] = None
        current_seed = SuffixCandidate(base_prompt=self.base_prompt, suffix=initial_suffix, step=0)
        start_step = 0

        checkpoint_state = self._load_checkpoint() if resume and enable_checkpointing else None
        if checkpoint_state:
            self._load_llm_stats(checkpoint_state)
            history = list(checkpoint_state.get("history", []))
            stepwise_scores = checkpoint_state.get("stepwise_scores", stepwise_scores)
            dev_leaderboard = self._deserialize_leaderboard(checkpoint_state.get("dev_leaderboard", {}))
            best_dev = self._candidate_from_snapshot(checkpoint_state.get("best_dev"))
            best_dev_scores = checkpoint_state.get("best_dev_scores") or None
            restored_seed = self._candidate_from_snapshot(
                checkpoint_state.get("current_seed"),
                fallback_suffix=initial_suffix,
            )
            current_seed = restored_seed or current_seed
            start_step = int(checkpoint_state.get("completed_step", -1)) + 1
            if not self.events_path.exists():
                self.events_path.touch()
            self._record_event(
                "resumed_from_checkpoint",
                start_step=start_step,
                completed_step=checkpoint_state.get("completed_step", -1),
                current_seed=self._snapshot_candidate(current_seed),
            )
            self._log(f"[yellow]Resuming from checkpoint at step {start_step}.[/yellow]")
            if checkpoint_state.get("completed") and checkpoint_state.get("final_best"):
                final_best = self._candidate_from_snapshot(checkpoint_state.get("final_best"))
                return final_best or current_seed, {
                    "history": history,
                    "stepwise_scores": stepwise_scores,
                }
        else:
            if self.events_path.exists():
                self.events_path.unlink()
            self._log("[yellow]## Step 0/Initialization[/yellow]")
            self._record_event(
                "initialization",
                initial_suffix=initial_suffix,
                num_search_steps=num_search_steps,
                num_suffix_candidates=num_suffix_candidates,
                top_k_for_critique=top_k_for_critique,
                dev_eval_every_n_steps=dev_eval_every_n_steps,
                judge_metric_config=judge_metric_config,
                task_llm_config=self._llm_config_snapshot(self.task_llm),
                meta_llm_config=self._llm_config_snapshot(self.meta_llm),
            )

            init_dir = self.save_dir / "step-000"
            init_dir.mkdir(parents=True, exist_ok=True)
            init_train_scores, _, _ = self.evaluate(
                current_seed,
                train,
                desc="Initialization train eval",
                save_path=init_dir / "train-seed.csv",
            )
            current_seed.scores = dict(init_train_scores)
            stepwise_scores["train"].append(
                {
                    "step": 0,
                    "suffix": current_seed.suffix,
                    "scores": dict(init_train_scores),
                }
            )

            init_dev_scores: Dict[str, float] = {}
            if dev:
                init_dev_scores, init_dev_preds, _ = self.evaluate(
                    current_seed,
                    dev,
                    desc="Initialization dev eval",
                    save_path=init_dir / "dev-seed.csv",
                )
                init_dev_payload = {current_seed.suffix: dict(init_dev_scores)}
                init_pred_payload = {current_seed.suffix: init_dev_preds}
                judge_scores, judge_logs = self._judge_metric(
                    [current_seed],
                    dev,
                    init_pred_payload,
                    init_dev_payload,
                    judge_metric_config,
                    step=0,
                )
                if current_seed.suffix in judge_scores:
                    init_dev_scores.update(judge_scores[current_seed.suffix])
                best_dev = self._clone_candidate(current_seed)
                best_dev_scores = dict(init_dev_scores)
                dev_leaderboard[current_seed.suffix] = {
                    "candidate": self._clone_candidate(current_seed),
                    "scores": dict(init_dev_scores),
                    "step": 0,
                }
                stepwise_scores["dev"].append(
                    {
                        "step": 0,
                        "suffix": current_seed.suffix,
                        "scores": dict(init_dev_scores),
                    }
                )
                self._save_json(init_dir / "judge_pairs.json", judge_logs)

            history.append(
                {
                    "step": 0,
                    "suffix": current_seed.suffix,
                    "train_scores": dict(init_train_scores),
                    "dev_scores": dict(best_dev_scores or {}),
                    "critique": "",
                    "suggestion": "",
                    "analysis": "",
                }
            )
            if enable_checkpointing:
                self._save_checkpoint(
                    self._build_checkpoint_payload(
                        completed_step=0,
                        num_search_steps=num_search_steps,
                        num_suffix_candidates=num_suffix_candidates,
                        top_k_for_critique=top_k_for_critique,
                        dev_eval_every_n_steps=dev_eval_every_n_steps,
                        initial_suffix=initial_suffix,
                        current_seed=current_seed,
                        best_dev=best_dev,
                        best_dev_scores=best_dev_scores,
                        history=history,
                        stepwise_scores=stepwise_scores,
                        dev_leaderboard=dev_leaderboard,
                        judge_metric_config=judge_metric_config,
                    )
                )
            start_step = 1

        for step in range(start_step, num_search_steps + 1):
            self._log(f"[yellow]## Step {step}/{num_search_steps}[/yellow]")
            step_dir = self.save_dir / f"step-{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            optimizer_context = self._history_context_candidates(
                history,
                current_seed,
                limit=max(top_k_for_critique, 1),
            )
            new_candidates = self.optimizer.run(
                self.meta_llm,
                self.base_prompt,
                current_seed.suffix,
                optimizer_context,
                analyzer_analysis=history[-1].get("analysis", "") if history else "",
                multi_metric_ranking="\n".join(
                    f"{idx + 1}. {cand.short_str(120)} | scores={cand.scores}"
                    for idx, cand in enumerate(optimizer_context)
                ),
                history=history,
                num_candidates=max(1, num_suffix_candidates - 1),
                step=step,
            )
            candidates = self._dedupe_candidates([self._clone_candidate(current_seed)] + new_candidates)
            self._record_event(
                "candidate_generation_finished",
                step=step,
                incumbent=self._snapshot_candidate(current_seed),
                generated=[self._snapshot_candidate(candidate) for candidate in candidates],
            )
            self._log(
                f"[green]Candidate generation produced {len(candidates)} prompts "
                f"(including incumbent).[/green]"
            )

            train_score_dicts: List[Dict[str, float]] = []
            all_predictions: Dict[str, List[str]] = {}
            for idx, candidate in enumerate(candidates, start=1):
                self._log(f"Train eval {idx}/{len(candidates)}: {candidate.short_str(80)}")
                train_scores, preds, _ = self.evaluate(
                    candidate,
                    train,
                    desc=f"Train eval step {step} cand {idx}",
                    save_path=step_dir / f"train-cand-{idx:03d}.csv",
                )
                train_score_dicts.append(train_scores)
                all_predictions[candidate.suffix] = preds

            self._attach_rank_scores(candidates, train_score_dicts)
            candidates = sorted(candidates, key=lambda cand: cand.scores["rank_score"], reverse=True)
            train_rankings = [
                {"suffix": candidate.suffix, "scores": dict(candidate.scores)}
                for candidate in candidates
            ]
            stepwise_scores["train"].append({"step": step, "rankings": train_rankings})
            self._record_event("train_ranking_finished", step=step, rankings=train_rankings)
            self._log(
                "[light_cyan]Train rank order (best first): "
                f"{[candidate.short_str() for candidate in candidates[:5]]}[/light_cyan]"
            )

            top_k = candidates[: min(top_k_for_critique, len(candidates))]
            dev_scores_by_suffix: Dict[str, Dict[str, float]] = {}
            dev_predictions_by_suffix: Dict[str, List[str]] = {}
            judge_logs: List[Dict[str, Any]] = []
            should_eval_dev = bool(dev) and (
                dev_eval_every_n_steps <= 1 or step % dev_eval_every_n_steps == 0
            )
            if should_eval_dev:
                for idx, candidate in enumerate(top_k, start=1):
                    dev_scores, dev_preds, _ = self.evaluate(
                        candidate,
                        dev or [],
                        desc=f"Dev eval step {step} cand {idx}",
                        save_path=step_dir / f"dev-cand-{idx:03d}.csv",
                    )
                    dev_scores_by_suffix[candidate.suffix] = dict(dev_scores)
                    dev_predictions_by_suffix[candidate.suffix] = dev_preds

                judge_scores, judge_logs = self._judge_metric(
                    top_k,
                    dev or [],
                    dev_predictions_by_suffix,
                    dev_scores_by_suffix,
                    judge_metric_config,
                    step=step,
                )
                for candidate in top_k:
                    if candidate.suffix in judge_scores:
                        dev_scores_by_suffix[candidate.suffix].update(judge_scores[candidate.suffix])

                dev_metric_names = list(self.ranker.metric_names)
                if any("judge_win_rate" in scores for scores in dev_scores_by_suffix.values()):
                    dev_metric_names.append("judge_win_rate")
                dev_score_dicts = [dev_scores_by_suffix[candidate.suffix] for candidate in top_k]
                dev_rank_scores = self.ranker.compute_rank_scores(
                    dev_score_dicts,
                    metric_names=dev_metric_names,
                )
                for candidate, rank_score in zip(top_k, dev_rank_scores):
                    dev_scores_by_suffix[candidate.suffix]["rank_score"] = rank_score
                    dev_scores_by_suffix[candidate.suffix]["avg_rank"] = -rank_score
                    stepwise_scores["dev"].append(
                        {
                            "step": step,
                            "suffix": candidate.suffix,
                            "scores": dict(dev_scores_by_suffix[candidate.suffix]),
                        }
                    )
                    existing = dev_leaderboard.get(candidate.suffix)
                    current_payload = {
                        "candidate": self._clone_candidate(candidate),
                        "scores": dict(dev_scores_by_suffix[candidate.suffix]),
                        "step": step,
                    }
                    if not existing or self._dev_selection_key(current_payload["scores"]) > self._dev_selection_key(existing["scores"]):
                        dev_leaderboard[candidate.suffix] = current_payload
                    if best_dev_scores is None or self._dev_selection_key(dev_scores_by_suffix[candidate.suffix]) > self._dev_selection_key(best_dev_scores):
                        best_dev = self._clone_candidate(candidate)
                        best_dev_scores = dict(dev_scores_by_suffix[candidate.suffix])
                self._record_event(
                    "dev_ranking_finished",
                    step=step,
                    metric_names=dev_metric_names,
                    rankings=[
                        {
                            "suffix": candidate.suffix,
                            "scores": dev_scores_by_suffix[candidate.suffix],
                        }
                        for candidate in top_k
                    ],
                )
                self._save_json(step_dir / "judge_pairs.json", judge_logs)
                self._log(
                    f"[magenta]Best dev so far: {best_dev.short_str(90) if best_dev else 'n/a'} | "
                    f"{best_dev_scores}[/magenta]"
                )
            else:
                self._record_event("dev_ranking_skipped", step=step)

            dev_summary = "\n".join(
                f"- {candidate.short_str(120)} :: {dev_scores_by_suffix.get(candidate.suffix, {})}"
                for candidate in top_k
            ) or "Dev evaluation skipped on this step."

            analyzer_prompt = self.analyzer.fill(
                self.base_prompt,
                top_k[0].suffix,
                all_predictions.get(top_k[0].suffix, []),
                top_k[0].scores,
                task_description,
                candidate_summaries=[
                    {
                        "suffix": candidate.short_str(160),
                        "train_scores": dict(candidate.scores),
                        "dev_scores": dev_scores_by_suffix.get(candidate.suffix, {}),
                        "sample_outputs": all_predictions.get(candidate.suffix, [])[:3],
                    }
                    for candidate in top_k
                ],
                dev_summary=dev_summary,
            )
            analyzer_analysis = self.analyzer.run(self.meta_llm, analyzer_prompt)
            self._record_event(
                "analysis_finished",
                step=step,
                analyzer_prompt=analyzer_prompt,
                analyzer_analysis=analyzer_analysis,
            )
            self._log(f"[light_yellow]Analyzer: {analyzer_analysis[:400]}...[/light_yellow]")

            critic_prompts = [
                self.critic.fill(
                    self.base_prompt,
                    candidate.suffix,
                    all_predictions.get(candidate.suffix, [])[:5],
                    candidate.scores,
                    analyzer_analysis=analyzer_analysis,
                )
                for candidate in top_k
            ]
            critiques = self._batch_meta_generate(critic_prompts, desc=f"Critiques step {step}")
            for idx, (candidate, critique, critic_prompt) in enumerate(
                zip(top_k, critiques, critic_prompts),
                start=1,
            ):
                candidate.critique = critique
                candidate.suggestions = self.critic.extract_suggestion(critique)
                self._log(f"[white]Critique for #{idx}: {critique[:240]}...[/white]")
                history.append(
                    {
                        "step": step,
                        "suffix": candidate.suffix,
                        "train_scores": dict(candidate.scores),
                        "dev_scores": dict(dev_scores_by_suffix.get(candidate.suffix, {})),
                        "critique": critique[:2000],
                        "suggestion": candidate.suggestions,
                        "analysis": analyzer_analysis[:2000],
                    }
                )
                self._record_event(
                    "critique_finished",
                    step=step,
                    suffix=candidate.suffix,
                    critic_prompt=critic_prompt,
                    critique=critique,
                    suggestion=candidate.suggestions,
                )

            if should_eval_dev and dev_scores_by_suffix:
                incumbent_next = max(
                    top_k,
                    key=lambda candidate: self._dev_selection_key(dev_scores_by_suffix[candidate.suffix]),
                )
            else:
                incumbent_next = top_k[0]
            current_seed = self._clone_candidate(incumbent_next)
            current_seed.step = step
            self._record_event(
                "incumbent_updated",
                step=step,
                incumbent=self._snapshot_candidate(current_seed),
            )

            step_payload = {
                "step": step,
                "incumbent": self._snapshot_candidate(current_seed),
                "train_rankings": train_rankings,
                "dev_rankings": [
                    {
                        "suffix": candidate.suffix,
                        "scores": dev_scores_by_suffix.get(candidate.suffix, {}),
                    }
                    for candidate in top_k
                ],
                "judge_pairs": judge_logs,
                "analyzer_analysis": analyzer_analysis,
            }
            self._save_json(step_dir / "summary.json", step_payload)
            self._save_json(self.save_dir / "history.json", history)
            self._save_json(
                self.save_dir / "prompts.json",
                [
                    {
                        "step": item["step"],
                        "suffix": item["suffix"],
                        "train_scores": item.get("train_scores", {}),
                        "dev_scores": item.get("dev_scores", {}),
                        "suggestion": item.get("suggestion", ""),
                    }
                    for item in history
                ],
            )
            self._save_json(
                self.save_dir / "leaderboard.json",
                sorted(
                    [
                        {
                            "suffix": payload["candidate"].suffix,
                            "scores": payload["scores"],
                            "step": payload["step"],
                        }
                        for payload in dev_leaderboard.values()
                    ],
                    key=lambda item: self._dev_selection_key(item["scores"]),
                    reverse=True,
                ),
            )
            if enable_checkpointing:
                self._save_checkpoint(
                    self._build_checkpoint_payload(
                        completed_step=step,
                        num_search_steps=num_search_steps,
                        num_suffix_candidates=num_suffix_candidates,
                        top_k_for_critique=top_k_for_critique,
                        dev_eval_every_n_steps=dev_eval_every_n_steps,
                        initial_suffix=initial_suffix,
                        current_seed=current_seed,
                        best_dev=best_dev,
                        best_dev_scores=best_dev_scores,
                        history=history,
                        stepwise_scores=stepwise_scores,
                        dev_leaderboard=dev_leaderboard,
                        judge_metric_config=judge_metric_config,
                    )
                )

        top2_dev = sorted(
            [
                (payload["candidate"], payload["scores"])
                for payload in dev_leaderboard.values()
            ],
            key=lambda item: self._dev_selection_key(item[1]),
            reverse=True,
        )[:2]

        if dev and len(top2_dev) >= 2:
            c_a, scores_a = top2_dev[0]
            c_b, scores_b = top2_dev[1]
            _, preds_a, _ = self.evaluate(c_a, dev[:10], desc="Judge eval A")
            _, preds_b, _ = self.evaluate(c_b, dev[:10], desc="Judge eval B")
            winner, justification = self.judge.run(
                self.meta_llm,
                c_a.full_prompt(),
                c_b.full_prompt(),
                preds_a,
                preds_b,
                scores_a,
                scores_b,
            )
            self._record_event(
                "final_judge_finished",
                winner=winner,
                candidate_a=self._snapshot_candidate(c_a),
                candidate_b=self._snapshot_candidate(c_b),
                justification=justification,
            )
            self._log(f"[red]Judge winner: {winner}[/red]")
            self._log(f"Justification: {justification[:400]}")
            if winner == "B":
                best_dev = self._clone_candidate(c_b)
                best_dev_scores = dict(scores_b)
            elif winner == "A":
                best_dev = self._clone_candidate(c_a)
                best_dev_scores = dict(scores_a)

        final_best = best_dev or current_seed
        self._save_json(
            self.save_dir / "run_summary.json",
            {
                "best_suffix": final_best.suffix,
                "best_dev_scores": best_dev_scores or {},
                "history_size": len(history),
                "stepwise_scores": stepwise_scores,
                "task_llm_stats": self.task_llm.stats() if hasattr(self.task_llm, "stats") else {},
                "meta_llm_stats": self.meta_llm.stats() if hasattr(self.meta_llm, "stats") else {},
            },
        )
        if enable_checkpointing:
            self._save_checkpoint(
                self._build_checkpoint_payload(
                    completed_step=num_search_steps,
                    num_search_steps=num_search_steps,
                    num_suffix_candidates=num_suffix_candidates,
                    top_k_for_critique=top_k_for_critique,
                    dev_eval_every_n_steps=dev_eval_every_n_steps,
                    initial_suffix=initial_suffix,
                    current_seed=current_seed,
                    best_dev=best_dev,
                    best_dev_scores=best_dev_scores,
                    history=history,
                    stepwise_scores=stepwise_scores,
                    dev_leaderboard=dev_leaderboard,
                    judge_metric_config=judge_metric_config,
                    completed=True,
                    final_best=final_best,
                )
            )
        return final_best, {"history": history, "stepwise_scores": stepwise_scores}
