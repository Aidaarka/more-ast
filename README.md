# MoRe-AST: Multi-objective Receptive AST

MoRe-AST is a multi-objective autoprompting method with a frozen base prompt `P*` and an optimized suffix `σ`. The implementation combines:

- train-time candidate evaluation over multiple metrics;
- dev-time model selection;
- Analyzer, Critic, and Optimizer meta-agents;
- optional judge-based dev ranking;
- resumable checkpointing with full optimization history.

## What The Current Implementation Does

At a high level, each optimization step follows this pattern:

1. Start from the current incumbent suffix `σ_t`.
2. Generate new candidate suffixes from optimization history.
3. Evaluate candidates on the train set.
4. Rank them with multi-metric rank aggregation.
5. Evaluate top candidates on the dev set.
6. Optionally run `Judge` pairwise on top dev candidates and inject `judge_win_rate` into dev ranking.
7. Run `Analyzer` on the step results.
8. Run batched `Critic` calls for top candidates.
9. Update the incumbent and save a checkpoint.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_key
```

On Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

## Generic Usage

Run on your own train/dev JSON or JSONL files:

```bash
python -m more_ast.run \
  --train path/to/train.json \
  --dev path/to/dev.json \
  --base_prompt "Summarize the following."
```

Resume from the latest checkpoint in the configured `save_dir`:

```bash
python -m more_ast.run \
  --train path/to/train.json \
  --dev path/to/dev.json \
  --resume
```

Supported train/dev formats:

- list of `{"x": "input", "y": "reference"}`
- list of `{"input": "...", "output": "..."}`
- JSONL with the same field names

## CNN/DailyMail Experiment

Main experiment entry point:

```bash
python -m more_ast.experiments.cnn.run_more_ast --mode quick --config more_ast/config.openrouter.toml
```

Modes:

- `debug`: 3 / 3 / 3 examples
- `quick`: 60 / 50 / 100 examples
- `standard`: 100 / 100 / 500 examples
- `shuffled`: random 100 / 100 / 500 examples

Windows PowerShell launcher for OpenRouter:

```powershell
.\more_ast\experiments\cnn\run_cnn_openrouter.ps1 -Mode quick -Steps 4
```

Resume an interrupted run:

```powershell
.\more_ast\experiments\cnn\run_cnn_openrouter.ps1 -Mode quick -Steps 8 -Resume
```

Override models at launch time:

```powershell
.\more_ast\experiments\cnn\run_cnn_openrouter.ps1 `
  -Mode quick `
  -TaskModel openai/gpt-4o-mini `
  -MetaModel anthropic/claude-3.5-haiku
```

## OpenRouter Configuration

Use `config.openrouter.toml` for OpenRouter runs. Important fields:

- `openrouter.base_url`
- `openrouter.task_model`
- `openrouter.meta_model`
- `openrouter.concurrency`
- `openrouter.max_retries`
- `openrouter.retry_delay`
- `openrouter.retry_jitter`
- `openrouter.request_timeout`

The implementation includes:

- concurrent request execution via `batch_generate`;
- retry logic for timeout / rate-limit / transient server errors;
- exponential backoff with jitter;
- per-LLM call statistics.

## Judge-Based Dev Ranking

The dev loop can optionally add an LLM judge signal on top of standard metrics. When enabled, top dev candidates are compared pairwise and each candidate receives:

- `judge_matches`
- `judge_wins`
- `judge_win_rate`

`judge_win_rate` is then included in dev ranking together with the standard metrics.

Configuration:

```toml
[judge_metric]
enabled = true
compare_top_k = 3
examples_per_comparison = 5
```

## Checkpointing And Resume

Checkpointing is enabled through:

```toml
[checkpointing]
enabled = true
```

After every completed optimization step, the trainer saves `checkpoint.json`. It contains enough state to resume from the last finished step, including:

- current incumbent suffix;
- best dev candidate and scores;
- optimization history;
- stepwise train/dev scores;
- dev leaderboard;
- judge configuration;
- accumulated LLM statistics.

When a run is resumed with `--resume`, MoRe-AST continues from `completed_step + 1` instead of restarting from scratch.

## Output Artifacts

Each run writes logs and structured artifacts into `save_dir`.

Main files:

- `log.md`: human-readable run log
- `events.jsonl`: append-only event trace of optimization stages
- `history.json`: accumulated optimization history
- `leaderboard.json`: best dev candidates seen so far
- `prompts.json`: compact suffix history
- `run_summary.json`: final run summary
- `checkpoint.json`: resumable optimizer state

Per-step files:

- `step-000/`: initialization artifacts
- `step-XXX/train-cand-YYY.csv`: train predictions and metric scores
- `step-XXX/dev-cand-YYY.csv`: dev predictions and metric scores
- `step-XXX/judge_pairs.json`: pairwise judge results for that step
- `step-XXX/summary.json`: structured step summary

## Config Files

- `config.toml`: default project config
- `config.openrouter.toml`: OpenRouter-oriented config for CNN runs
- `prompts.toml`: meta-prompts for Analyzer, Critic, Optimizer, and Judge

## Notes

- `config.toml` may point to a non-OpenRouter backend if you change `base_url`.
- For OpenRouter experiments, prefer `config.openrouter.toml`.
- `bertscore` is optional and can be slow on larger runs.
