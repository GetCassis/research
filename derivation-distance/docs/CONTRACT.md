# Module contract

All modules live in `src/` and are run with the experiment venv from the
experiment root (`python src/<module>.py`). Paths below are relative to the
experiment root. Modules stay importable: no work at import time except cheap
constants. Every script has an `if __name__ == "__main__"` self-test path.

## `src/config.py`
- `DATASETS = ["jaffle-shop", "thelook", "olist"]`
- `MODELS = {label: model_id}` and `PRICING = {label: {"in": usd_per_mtok, "out": usd_per_mtok}}`
- `DERIVATION_LEVELS = [L1_schema_descriptions, L2_prose_rules, L3_named_metrics, L4_metric_formulas, L5_example_library, L6_answer_templates]` — the axis
- `SIDE_ARMS = [L0_bare_schema]`; `LEVELS = SIDE_ARMS + DERIVATION_LEVELS`
- `COMPARISON_LEVELS = [L5_library_heldout, L6_templates_heldout, L5_relevant_3, L5_irrelevant_3, L5_relevant_3_question, L5_relevant_3_goldsql]` — ablation arms
- `ANCHOR_LEVEL = L4_metric_formulas` (the full published ontology)
- `DATASET_FACTS` — genuinely global facts injected identically at every L1+ rung
  (snapshot date, the "last N days" window rule)
- path helpers: `dataset_dir`, `db_path`, `ontology_path`, `questions_path`, `gold_path`

## `src/db.py`
- `build(ds)` — load `datasets/<ds>/data/*.csv.gz` into `data/<ds>.duckdb` under the
  ontology's physical schema (e.g. `JAFFLE`), UPPERCASE table names. Idempotent.
- `connect(ds)` — read-only connection with `search_path` set so qualified and bare
  names both resolve. `schema(ds)`, `run_sql(ds, sql) -> (columns, rows)`.

## `data/questions/<ds>.yaml`
List of question dicts: `id`, `dataset`, `question`, `difficulty` (`single_agg` |
`multi_join` | `date_window` | `multi_step` | `hard_multistep`), `uses`,
`required_facts` (the completeness rubric — **never injected into context**),
`gap` (planted-gap questions, excluded), `gold_sql`, `example_question` +
`example_sql` (the analogous solved query that seeds the example library), `notes`.

## `src/questions.py`
- `load(ds)` — parse the YAML.
- `build_gold(ds)` — execute each `gold_sql`, write `data/gold/<id>.json`
  (`{id, columns, rows, row_count, hash}`). The hash is computed on the exact
  `(columns, rows)` the grader sees at runtime, so every gold self-grades correct
  (the CLI asserts this).

## `src/ladder.py`
- `render(ds, qrow, level) -> str` — the full context for one question at one
  level: a `# Question` block + the level's body. Bodies are question-independent
  at every level (libraries are shared galleries; leave-one-out and 3-example
  arms are per-question).
- `available_levels(qrow)`; module self-test checks monotone token counts across
  L1→L4, no rubric leakage, L3 hides metric expressions, shared bodies identical
  across questions.

## `src/contexts.py`
- Pre-rendered context cache under `data/contexts/<ds>/`. `get(ds, qrow, level)`
  is byte-for-byte equal to `ladder.render` (the CLI pre-renders everything and
  verifies this). The harness reads contexts through `get()`.

## `src/grade.py`
- `result_hash(columns, rows)`, `grade(ds, model_sql, gold_id, tolerant=True)` →
  `{exec_ok, correct, result_hash, match_kind, error}`. Strict hash first
  (multiset of rows, column-order-insensitive, floats to 4 dp); on miss,
  `tolerant_match` accepts verified presentation variants only: numeric typing
  unified, midnight timestamp == date, the model's own display precision
  (floor 2 dp), injective column mapping of gold into the model result
  (supersets/reorders). Scale changes, ID-vs-name swaps, and missing gold
  columns stay incorrect. `match_kind` records what fired (`"exact"` or a
  `+`-joined combination of `projection|rounding|typing|numeric-typing`).
- `src/regrade.py in.jsonl [out.jsonl]` — offline tolerant re-grade of an
  existing run file from its stored SQL (no API calls).

## `src/harness.py`
- `run_cell(model_label, ds, qrow, level, sample, effort) -> record` — reads the
  context via `contexts.get`, calls the model (adaptive thinking where the API
  supports it), extracts the fenced SQL, grades, computes cost. Never raises;
  failures land in `record["error"]`.
- Run record: `{id, dataset, model, level, sample, context_tokens, input_tokens,
  output_tokens, thinking_tokens, sql, exec_ok, correct, result_hash, cost_usd,
  latency_s, error}`.

## `src/run.py`
Factorial driver: `--run NAME --datasets ... --models ... --levels ...
--samples N --concurrency K [--difficulty ...] [--questions ...]`. Appends to
`results/runs/<NAME>.jsonl`; resumable (cells already present are skipped).
Builds DBs, golds and the context cache before running.

## Analysis
- `src/analyze.py --run NAME` — figures to `results/figures/` + `results/summary.md`.
- `src/article_charts.py` — the six article figures to `images/`, from the
  published run files (`main.jsonl`, `heldout.jsonl`, `relevance.jsonl`).
- `src/repeatability.py [run.jsonl]`, `src/flakiness_corr.py [run.jsonl]`,
  `src/cost.py`, `src/relevance.py` (similarity selectors for the 3-example arms).

## Conventions
- No network except the Anthropic API in `harness.py`; grading is local DuckDB.
- Invalid SQL counts as incorrect; only true infra failures (no SQL emitted) are
  dropped from accuracy.
