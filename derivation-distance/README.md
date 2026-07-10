# Derivation distance: how much does spelling things out help?

The full harness, data, and results behind the Cassis study of **derivation
distance** — how far a model must travel from the context it is shown to the
answer it gives — and how that interacts with model capability.

In one line: we render the same analytics questions against the same warehouse
under contexts that vary only in how spelled-out the implementation is (from
bare schema to a worked-example library), across three models (Haiku 4.5,
Sonnet 4.6, Opus 4.8), and grade the SQL they write by executing it.

## The context levels (independent variable)

The context is the published ontology shown at increasing depth. It is
**identical across all questions at every level** — the model must locate the
relevant table / metric / rule itself, so discoverability is part of the task.

| Level | Context contains | Role |
|-------|------------------|------|
| `bare_schema` | table + column names only — no descriptions, no facts | side arm (deliberately incomplete) |
| `descriptions` | schema + column descriptions + global operating facts (snapshot date, window rule) | axis — max distance |
| `metrics` | + domain prose (rules, guardrails), named metrics, and metric expressions — the full published ontology | axis |
| `examples` | + the **full worked-example library** (model finds the relevant analogue) | axis |
| `templates` | + the **full template gallery**: every question's gold SQL with one literal blanked | comparison arm |

Ablation variants (comparison arms): `examples_heldout` / `templates_heldout`
(leave-one-out: the question's own entry removed), `relevant_3` /
`irrelevant_3` (size-matched 3-example libraries picked by structural
similarity), and `relevant_3_question` / `relevant_3_goldsql` (3 examples
picked by question-text similarity / gold-SQL token overlap).

Every level's rendered context body is committed under
`data/contexts/<dataset>/` — `<level>.body.txt` for the shared levels, and
`<level>/<question_id>.body.txt` for the per-question ablation arms — so you
can inspect exactly what the models saw. A run prepends only a `# Question`
block to these bodies.

## The measurement

- **69 questions** across three datasets (`datasets/`): jaffle-shop, theLook,
  Olist — 12 standard + 11 hard multi-step per dataset. Questions + gold SQL
  live in `data/questions/<ds>.yaml`.
- **Execution accuracy**: the model's SQL runs against a DuckDB warehouse built
  from the real CSVs; the result set is compared to the executed gold
  (order-insensitive, column-name-insensitive, floats rounded to 4 dp).
  Grading is **tolerant of presentation variants** (column supersets/reorders,
  the model's own display precision ≥ 2 dp, midnight-timestamp == date,
  int == float); it still rejects value changes (fraction vs %, ID vs name,
  missing columns). Invalid SQL counts as incorrect. See `src/grade.py`.
- **N = 3** samples per (model × level × question) cell, for repeatability.

## Reproduce the figures (no API calls, no data download)

The published run files are in `results/runs/` (see
[`results/runs/README.md`](results/runs/README.md)), so regenerating every
figure and number in the write-up is offline:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/article_charts.py   # writes images/01,02,04,06,07,08
```

## Reproduce the benchmark itself

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python datasets/download.py              # ~70 MB of CSVs, checksum-verified
.venv/bin/python src/db.py               # smoke: builds jaffle-shop warehouse
.venv/bin/python src/questions.py        # builds + self-grades the gold answers
.venv/bin/python src/contexts.py         # pre-renders all contexts + verifies
.venv/bin/python src/ladder.py           # ladder self-test

export ANTHROPIC_API_KEY=...
.venv/bin/python src/run.py --run myrun --datasets jaffle-shop thelook olist \
    --levels bare_schema descriptions metrics examples \
    --samples 3 --concurrency 8
.venv/bin/python src/analyze.py --run myrun
```

Cost note: the main 2,484-cell grid (4 levels × 3 models × 69 questions × 3
samples) cost about **$47** in API calls at mid-2026 prices.

## Layout

```
datasets/<ds>/       ontology.json + SOURCE.md; data/*.csv.gz via download.py
data/questions/      question sets + gold SQL (the benchmark definition)
data/contexts/       the rendered context bodies, per (dataset, level) — committed
data/gold/           executed gold snapshots (generated)
data/<ds>.duckdb     built warehouse (generated)
src/                 harness (see docs/CONTRACT.md for module interfaces)
results/runs/        published run files (one JSON line per cell)
images/              the study's figures (regenerable via src/article_charts.py)
```

## License

Code and text: MIT (see the repository root). Data: per-dataset — Olist is
**CC BY-NC-SA (non-commercial)**; see `datasets/README.md` and each `SOURCE.md`.
