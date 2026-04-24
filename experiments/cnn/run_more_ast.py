#!/usr/bin/env python3
"""
MoRe-AST experiment on CNN/DailyMail.

Usage (from workspace root):
    python -m more_ast.experiments.cnn.run_more_ast [--mode debug|quick|standard|shuffled]

Modes:
    debug    - 3/3/3 examples (sanity check, no real API cost)
    quick    - 60/50/100 examples (fast iteration)
    standard - 100/100/500 examples (deterministic slice)
    shuffled - 100/100/500 examples (random sample, default)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Must happen before any crispo / more_ast imports.
_HERE = Path(__file__).resolve()
_MORE_AST_DIR = _HERE.parent.parent.parent          # more_ast/
_WORKSPACE_ROOT = _MORE_AST_DIR.parent              # workspace root

for p in [str(_WORKSPACE_ROOT / "CriSPO"), str(_WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ─────────────────────────────────────────────────────────────────────────────

from crispo.utilities.log_util import init_logger

from more_ast.core.multi_metric import MultiMetricRanker
from more_ast.experiments import cdroot
from more_ast.experiments.cnn.dataset import (
    load_cnn_debug,
    load_cnn_quick,
    load_cnn_shuffled,
    load_cnn_standard,
)
from more_ast.llms.openrouter import OpenRouterLLM
from more_ast.metrics import build_metrics
from more_ast.trainer import MoReASTTrainer
from more_ast.utils import load_toml

# ── Summarization helpers ─────────────────────────────────────────────────────

_SUMMARY_TAG_HINT = "\n\nEnclose your summary within <summary> tags."
_CRISPO_PLACEHOLDER = "INSERT_INPUT_HERE"


def format_cnn_input(base_prompt: str, suffix: str, article: str) -> str:
    """
    Build the final prompt for a single CNN article.
    Structure: base_prompt + suffix + article + tag hint.
    """
    if _CRISPO_PLACEHOLDER in base_prompt:
        prompt = base_prompt.replace(_CRISPO_PLACEHOLDER, f"<input>\n{article}\n</input>")
        if suffix.strip():
            return "\n\n".join([prompt.strip(), suffix.strip()])
        return prompt.strip()

    parts = [base_prompt.strip()]
    if suffix.strip():
        parts.append(suffix.strip())
    parts.append(f"\n<article>\n{article}\n</article>")
    parts.append(_SUMMARY_TAG_HINT)
    return "\n\n".join(parts)


def parse_cnn_output(generation: str, x: object = None) -> str:
    """Extract text inside <summary> tags; fall back to full generation."""
    gen = generation.strip()
    start = gen.find("<summary>")
    if start != -1:
        start += len("<summary>")
        end = gen.find("</summary>", start)
        if end != -1:
            return gen[start:end].strip()
    return gen


# ── Base prompt ───────────────────────────────────────────────────────────────

BASE_PROMPT = (
    "Below are two 100-word summary examples of the upcoming input text. "
    "Write your own 100-word summary within <summary> tags, focusing only on "
    "the three most important details, people or locations mentioned. "
    "Directly reflect the style and main topics of the examples provided "
    "without extra context:\n\n"
    "<example>\n"
    "- Organizers hope to use social media to inspire \"Occupy Wall Street\" "
    "protest on Saturday in New York's financial district\n"
    "- Adbusters co-founder wants to emulate uprisings in Egypt, Iran by "
    "drawing thousands to protest financial fraud and lack of justice\n"
    "</example>\n\n"
    "<example>\n"
    "- Hacktivist group Anonymous urged supporters to participate in planned "
    "sit-in against financial fraud in New York City\n"
    "- Protest aims to emulate uprisings in Egypt and Iran by gathering "
    "thousands to call for justice and oppose Wall Street corruption\n"
    "</example>\n\n"
    "INSERT_INPUT_HERE"
)

INITIAL_SUFFIX = "Every word of your summary must be faithful to the input text."

TASK_DESCRIPTION = (
    "Abstractive news summarization on CNN/DailyMail. "
    "Metrics: ROUGE-1 and faithfulness. "
    "Key conflict: lexical overlap with the reference summary versus strict "
    "faithfulness to the source article."
)


def load_aggregate_scores_from_csv(csv_path: Path, metric_names: list[str]) -> dict:
    """Load aggregate metric means from a saved evaluation CSV."""
    if not csv_path.exists():
        return {}
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}

    aggregates = {}
    for metric_name in metric_names:
        values = []
        for row in rows:
            raw = row.get(metric_name)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if values:
            aggregates[metric_name] = sum(values) / len(values)
    return aggregates

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoRe-AST on CNN/DailyMail")
    p.add_argument(
        "--mode",
        choices=["debug", "quick", "standard", "shuffled"],
        default="shuffled",
        help="Dataset size (default: shuffled)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: more_ast/config.toml)",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Path to prompts.toml (default: more_ast/prompts.toml)",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override num_search_steps from config",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimization from checkpoint.json in save_dir",
    )
    p.add_argument(
        "--reuse_test_evals",
        action="store_true",
        help="Reuse existing baseline_test.csv and seed_test.csv from save_dir if present",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    cdroot()

    config_path = args.config or _MORE_AST_DIR / "config.toml"
    prompts_path = args.prompts or _MORE_AST_DIR / "prompts.toml"

    cfg = load_toml(config_path)
    or_cfg = cfg["openrouter"]
    opt_cfg = cfg["optimization"]
    metrics_cfg = cfg.get("metrics", {})
    log_cfg = cfg.get("logging", {})
    judge_metric_cfg = cfg.get("judge_metric", {})
    checkpoint_cfg = cfg.get("checkpointing", {})

    # ── Dataset ───────────────────────────────────────────────────────────────
    loaders = {
        "debug": load_cnn_debug,
        "quick": load_cnn_quick,
        "standard": load_cnn_standard,
        "shuffled": load_cnn_shuffled,
    }
    train, dev, test = loaders[args.mode]()

    # ── Save directory ────────────────────────────────────────────────────────
    save_dir = args.save_dir or os.path.join(
        log_cfg.get("save_dir", "outputs/more_ast"),
        f"cnn_{args.mode}",
    )

    logger = init_logger(name="log.md", save_dir=save_dir, mode="a")
    logger.info(f"[yellow]MoRe-AST | CNN/DailyMail | mode={args.mode}[/yellow]")
    logger.info(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")
    logger.info(f"Save dir: {save_dir}")

    # ── LLMs ──────────────────────────────────────────────────────────────────
    api_key = or_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning(
            "[red]OPENROUTER_API_KEY not set. "
            "Set env var or fill api_key in config.toml.[/red]"
        )

    common_llm_kwargs = dict(
        api_key=api_key,
        base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        max_new_tokens=or_cfg.get("max_new_tokens", 100000),
        concurrency=or_cfg.get("concurrency", 8),
        max_retries=or_cfg.get("max_retries", opt_cfg.get("max_retry", 5)),
        retry_delay=or_cfg.get("retry_delay", 2.0),
        retry_jitter=or_cfg.get("retry_jitter", 0.5),
        request_timeout=or_cfg.get("request_timeout", 360.0),
        app_name=or_cfg.get("app_name", "MoRe-AST"),
        http_referer=or_cfg.get("http_referer", "https://cursor.local"),
    )
    task_llm = OpenRouterLLM(
        model=os.environ.get("MORE_AST_TASK_MODEL", or_cfg.get("task_model", "openai/gpt-4o-mini")),
        temperature=0.0,          # Deterministic task outputs
        **common_llm_kwargs,
    )
    meta_llm = OpenRouterLLM(
        model=os.environ.get("MORE_AST_META_MODEL", or_cfg.get("meta_model", "openai/gpt-4o")),
        temperature=or_cfg.get("temperature", 0.7),   # Creative meta-prompts
        **common_llm_kwargs,
    )
    logger.info(f"Task LLM : {task_llm.model}")
    logger.info(f"Meta LLM : {meta_llm.model}")
    logger.info(
        "Task LLM config: max_new_tokens=%s | temperature=%s | timeout=%s | concurrency=%s | retries=%s",
        task_llm.max_new_tokens,
        task_llm.temperature,
        task_llm.request_timeout,
        task_llm.concurrency,
        task_llm.max_retries,
    )
    logger.info(
        "Meta LLM config: max_new_tokens=%s | temperature=%s | timeout=%s | concurrency=%s | retries=%s",
        meta_llm.max_new_tokens,
        meta_llm.temperature,
        meta_llm.request_timeout,
        meta_llm.concurrency,
        meta_llm.max_retries,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    active = metrics_cfg.get("active", ["rouge1", "faithfulness"])
    metrics = build_metrics(active)
    ranker = MultiMetricRanker(primary=metrics_cfg.get("primary", "rank"), **metrics)
    logger.info(f"Active metrics: {list(metrics.keys())}")
    metric_names = list(metrics.keys())

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = MoReASTTrainer(
        save_dir=save_dir,
        base_prompt=BASE_PROMPT,
        prompts_path=prompts_path,
        task_llm=task_llm,
        meta_llm=meta_llm,
        ranker=ranker,
        format_input=format_cnn_input,
        parse_output=parse_cnn_output,
    )

    num_steps = args.steps or opt_cfg.get("num_search_steps", 20)
    initial_suffix = opt_cfg.get("initial_suffix", INITIAL_SUFFIX)
    should_resume = checkpoint_cfg.get("enabled", True) and args.resume and trainer.has_checkpoint()

    # ── Baseline: base prompt alone on test ───────────────────────────────────
    from more_ast.core.suffix import SuffixCandidate

    baseline_scores = {}
    if should_resume:
        logger.info("[yellow]Resume mode: skipping baseline test re-evaluation.[/yellow]")
    elif args.reuse_test_evals:
        baseline_scores = load_aggregate_scores_from_csv(Path(save_dir) / "baseline_test.csv", metric_names)
        if baseline_scores:
            logger.info("[yellow]Reusing existing baseline_test.csv from save_dir.[/yellow]")
            logger.info(f"[cyan]Baseline test scores: {baseline_scores}[/cyan]")
        else:
            baseline = SuffixCandidate(base_prompt=BASE_PROMPT, suffix="")
            logger.info("[yellow]Evaluating baseline (no suffix) on test...[/yellow]")
            baseline_scores, _, _ = trainer.evaluate(
                baseline,
                test,
                desc="Baseline test eval",
                save_path=Path(save_dir) / "baseline_test.csv",
            )
            logger.info(f"[cyan]Baseline test scores: {baseline_scores}[/cyan]")
    else:
        baseline = SuffixCandidate(base_prompt=BASE_PROMPT, suffix="")
        logger.info("[yellow]Evaluating baseline (no suffix) on test...[/yellow]")
        baseline_scores, _, _ = trainer.evaluate(
            baseline,
            test,
            desc="Baseline test eval",
            save_path=Path(save_dir) / "baseline_test.csv",
        )
        logger.info(f"[cyan]Baseline test scores: {baseline_scores}[/cyan]")

    # ── Seed suffix on test ───────────────────────────────────────────────────
    seed_scores = {}
    if should_resume:
        logger.info("[yellow]Resume mode: skipping seed test re-evaluation.[/yellow]")
    elif args.reuse_test_evals:
        seed_scores = load_aggregate_scores_from_csv(Path(save_dir) / "seed_test.csv", metric_names)
        if seed_scores:
            logger.info("[yellow]Reusing existing seed_test.csv from save_dir.[/yellow]")
            logger.info(f"[cyan]Seed suffix test scores: {seed_scores}[/cyan]")
        else:
            seed = SuffixCandidate(base_prompt=BASE_PROMPT, suffix=initial_suffix)
            logger.info("[yellow]Evaluating seed suffix on test...[/yellow]")
            seed_scores, _, _ = trainer.evaluate(
                seed,
                test,
                desc="Seed suffix test eval",
                save_path=Path(save_dir) / "seed_test.csv",
            )
            logger.info(f"[cyan]Seed suffix test scores: {seed_scores}[/cyan]")
    else:
        seed = SuffixCandidate(base_prompt=BASE_PROMPT, suffix=initial_suffix)
        logger.info("[yellow]Evaluating seed suffix on test...[/yellow]")
        seed_scores, _, _ = trainer.evaluate(
            seed,
            test,
            desc="Seed suffix test eval",
            save_path=Path(save_dir) / "seed_test.csv",
        )
        logger.info(f"[cyan]Seed suffix test scores: {seed_scores}[/cyan]")

    # ── Optimization ──────────────────────────────────────────────────────────
    logger.info(f"[yellow]Starting MoRe-AST optimization ({num_steps} steps)...[/yellow]")
    best, meta = trainer.fit(
        train=train,
        dev=dev,
        num_search_steps=num_steps,
        num_suffix_candidates=opt_cfg.get("num_suffix_candidates", 8),
        top_k_for_critique=opt_cfg.get("top_k_for_critique", 50),
        dev_eval_every_n_steps=opt_cfg.get("dev_eval_every_n_steps", 5),
        initial_suffix=initial_suffix,
        task_description=TASK_DESCRIPTION,
        judge_metric_config=judge_metric_cfg,
        enable_checkpointing=checkpoint_cfg.get("enabled", True),
        resume=should_resume,
        logger=logger,
    )

    # ── Final evaluation on test ──────────────────────────────────────────────
    logger.info("[yellow]Evaluating best suffix on test set...[/yellow]")
    final_scores, _, _ = trainer.evaluate(
        best,
        test,
        desc="Final test eval",
        save_path=Path(save_dir) / "final_test.csv",
    )

    logger.info("=" * 60)
    logger.info("[green]=== Final Results (CNN/DailyMail) ===[/green]")
    logger.info(f"  Mode         : {args.mode}")
    logger.info(f"  Test size    : {len(test)}")
    logger.info(f"  Best suffix  : [light_magenta]{best.suffix}[/light_magenta]")
    logger.info(f"  Baseline     : {baseline_scores}")
    logger.info(f"  Seed suffix  : {seed_scores}")
    logger.info(f"  [green]Best (MoRe-AST): {final_scores}[/green]")
    logger.info(f"  Task LLM stats: {task_llm.stats()}")
    logger.info(f"  Meta LLM stats: {meta_llm.stats()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
