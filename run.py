#!/usr/bin/env python3
"""
MoRe-AST CLI entry point.
Loads config.toml and prompts.toml, runs optimization.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure CriSPO and workspace root are on path before any imports
def _setup_path():
    more_ast_dir = Path(__file__).resolve().parent
    workspace_root = more_ast_dir.parent
    crispo_parent = workspace_root / "CriSPO"
    if crispo_parent.exists() and str(crispo_parent) not in sys.path:
        sys.path.insert(0, str(crispo_parent))
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))


_setup_path()

from crispo.task.example import Example
from crispo.utilities.log_util import init_logger

from more_ast.core.multi_metric import MultiMetricRanker
from more_ast.llms.openrouter import OpenRouterLLM
from more_ast.metrics import build_metrics
from more_ast.trainer import MoReASTTrainer
from more_ast.utils import load_toml


def load_examples(path: Path, input_key: str = "x", output_key: str = "y") -> list:
    """Load examples from JSON (list of dicts) or JSONL."""
    path = Path(path)
    if not path.exists():
        return []
    examples = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    examples.append(Example(x=d.get(input_key, d.get("input", "")), y=d.get(output_key, d.get("output", ""))))
    else:
        data = json.load(open(path))
        if isinstance(data, list):
            for d in data:
                examples.append(Example(x=d.get(input_key, d.get("input", "")), y=d.get(output_key, d.get("output", ""))))
        else:
            for x, y in zip(data.get("x", data.get("input", [])), data.get("y", data.get("output", []))):
                examples.append(Example(x=x, y=y))
    return examples


def main():
    parser = argparse.ArgumentParser(description="MoRe-AST: Multi-objective Receptive AST")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.toml")
    parser.add_argument("--prompts", type=Path, default=None, help="Path to prompts.toml")
    parser.add_argument("--base_prompt", type=str, default="Summarize the following text concisely and accurately.")
    parser.add_argument("--task", type=str, default="summarization", choices=["summarization", "qa"])
    parser.add_argument("--train", type=Path, required=True, help="Path to train JSON/JSONL")
    parser.add_argument("--dev", type=Path, default=None, help="Path to dev JSON/JSONL")
    parser.add_argument("--task_description", type=str, default="Summarization: generate concise summaries.")
    parser.add_argument("--resume", action="store_true", help="Resume optimization from checkpoint.json in save_dir")
    args = parser.parse_args()

    more_ast_dir = Path(__file__).resolve().parent
    config_path = args.config or more_ast_dir / "config.toml"
    prompts_path = args.prompts or more_ast_dir / "prompts.toml"

    config = load_toml(config_path)
    or_cfg = config["openrouter"]
    opt_cfg = config["optimization"]
    metrics_cfg = config.get("metrics", {})
    log_cfg = config.get("logging", {})
    judge_metric_cfg = config.get("judge_metric", {})
    checkpoint_cfg = config.get("checkpointing", {})

    api_key = or_cfg.get("api_key") or __import__("os").environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not set. Set env var or add api_key to config.")

    task_llm = OpenRouterLLM(
        model=or_cfg.get("task_model", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        max_new_tokens=or_cfg.get("max_new_tokens", 512),
        temperature=or_cfg.get("temperature", 0.7),
        concurrency=or_cfg.get("concurrency", 8),
        max_retries=or_cfg.get("max_retries", opt_cfg.get("max_retry", 5)),
        retry_delay=or_cfg.get("retry_delay", 2.0),
        retry_jitter=or_cfg.get("retry_jitter", 0.5),
        request_timeout=or_cfg.get("request_timeout", 180.0),
        app_name=or_cfg.get("app_name", "MoRe-AST"),
        http_referer=or_cfg.get("http_referer", "https://cursor.local"),
    )
    meta_llm = OpenRouterLLM(
        model=or_cfg.get("meta_model", "openai/gpt-4o"),
        api_key=api_key,
        base_url=or_cfg.get("base_url", "https://openrouter.ai/api/v1"),
        max_new_tokens=or_cfg.get("max_new_tokens", 512),
        temperature=or_cfg.get("temperature", 0.7),
        concurrency=or_cfg.get("concurrency", 8),
        max_retries=or_cfg.get("max_retries", opt_cfg.get("max_retry", 5)),
        retry_delay=or_cfg.get("retry_delay", 2.0),
        retry_jitter=or_cfg.get("retry_jitter", 0.5),
        request_timeout=or_cfg.get("request_timeout", 180.0),
        app_name=or_cfg.get("app_name", "MoRe-AST"),
        http_referer=or_cfg.get("http_referer", "https://cursor.local"),
    )

    active = metrics_cfg.get("active", ["rouge1", "faithfulness"])
    metrics = build_metrics(active)
    ranker = MultiMetricRanker(primary=metrics_cfg.get("primary", "rank"), **metrics)

    train = load_examples(args.train)
    dev = load_examples(args.dev) if args.dev else None
    if not train:
        print("No train examples loaded. Provide --train path to JSON/JSONL.")
        sys.exit(1)

    save_dir = str(Path(log_cfg.get("save_dir", "outputs/more_ast")).resolve())
    logger = init_logger(name="log.md", save_dir=save_dir, mode="a")

    trainer = MoReASTTrainer(
        save_dir=save_dir,
        base_prompt=args.base_prompt,
        prompts_path=prompts_path,
        task_llm=task_llm,
        meta_llm=meta_llm,
        ranker=ranker,
    )

    best, meta = trainer.fit(
        train=train,
        dev=dev,
        num_search_steps=opt_cfg.get("num_search_steps", 20),
        num_suffix_candidates=opt_cfg.get("num_suffix_candidates", 8),
        top_k_for_critique=opt_cfg.get("top_k_for_critique", 5),
        dev_eval_every_n_steps=opt_cfg.get("dev_eval_every_n_steps", 3),
        initial_suffix=opt_cfg.get("initial_suffix", "Every word of your summary must be faithful to the input text."),
        task_description=args.task_description,
        judge_metric_config=judge_metric_cfg,
        enable_checkpointing=checkpoint_cfg.get("enabled", True),
        resume=args.resume and checkpoint_cfg.get("enabled", True),
        logger=logger,
    )

    logger.info(f"[green]Best suffix: {best.suffix}[/green]")
    logger.info(f"Scores: {best.scores}")
    logger.info(f"Task LLM stats: {task_llm.stats()}")
    logger.info(f"Meta LLM stats: {meta_llm.stats()}")


if __name__ == "__main__":
    main()
