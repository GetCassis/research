# Published run files

One JSON line per cell: `{id, dataset, model, level, sample, context_tokens,
input_tokens, output_tokens, thinking_tokens, sql, exec_ok, correct,
result_hash, cost_usd, latency_s, error}`. Tolerant-graded records also carry
`correct_strict` and `match_kind`. Models: `haiku` = Claude Haiku 4.5,
`sonnet` = Claude Sonnet 4.6, `opus` = Claude Opus 4.8 (mid-2026).

| File | Cells | Levels | What it backs |
|---|---:|---|---|
| `main.jsonl` | 2,484 | L0, L1, L4, L5 × 69 questions × 3 models × N=3 | The article's main grid: figures 01, 02, 04, 06; accuracy, repeatability and cost numbers. Run on the convention-fixed contexts. |
| `heldout.jsonl` | 1,863 | L5_library_heldout, L6_answer_templates, L6_templates_heldout | Leave-one-out ablation (figure 07) and the answer-template note. Fixed contexts. |
| `relevance.jsonl` | 594 | L5_relevant_3, L5_irrelevant_3 | Relevance ablation (figure 08). Fixed contexts. |
| `selectors.jsonl` | 594 | L5_relevant_3_question, L5_relevant_3_goldsql | Selector ablation (the "maybe a better retriever closes the gap" check). Fixed contexts. |
| `full_ladder.jsonl` | 4,347 | all 7 ladder levels | The full factorial on the **pre-fix** contexts, tolerant re-grade. Backs the "wrong turn" section's before numbers (Opus 86% at descriptions vs 77% at metrics). |
| `context_fix_verification.jsonl` | 720 | L1–L4, the 20 affected questions | The post-fix verification run (Opus 70/47 → 93/100 on the affected questions). |

`full_ladder.jsonl` intentionally predates the context-convention fixes — it is
the "before" evidence. Its contexts differ from what `src/contexts.py` renders
from the shipped ontologies (which include the fixes); a fresh run of the full
ladder on today's contexts will not reproduce its numbers, and that is the point
of the article section it backs.
