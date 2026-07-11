# Published run files

One JSON line per cell: `{id, dataset, model, level, sample, context_tokens,
input_tokens, output_tokens, thinking_tokens, sql, exec_ok, correct,
result_hash, cost_usd, latency_s, error}`. Tolerant-graded records also carry
`correct_strict` and `match_kind`. Models: `haiku` = Claude Haiku 4.5,
`sonnet` = Claude Sonnet 4.6, `opus` = Claude Opus 4.8 (mid-2026).

| File | Cells | Levels | What it backs |
|---|---:|---|---|
| `main.jsonl` | 2,484 | bare_schema, descriptions, metrics, examples × 69 questions × 3 models × N=3 | The main grid: figures 01, 02, 04, 06; accuracy, repeatability and cost numbers. Run on the convention-fixed contexts. |
| `descriptions_plus_examples.jsonl` | 621 | descriptions_plus_examples × 69 questions × 3 models × N=3 | Rules/metrics ablation: complete descriptions + examples, without domain prose or metric sections. Updates figure 06 and backs the article's ablation table. |
| `heldout.jsonl` | 1,863 | examples_heldout, templates, templates_heldout | Leave-one-out ablation (figure 07) and the answer-template comparison. Fixed contexts. |
| `relevance.jsonl` | 594 | relevant_3, irrelevant_3 | Relevance ablation (figure 08). Fixed contexts. |
| `selectors.jsonl` | 594 | relevant_3_question, relevant_3_goldsql | Selector ablation (the "maybe a better retriever closes the gap" check). Fixed contexts. |
| `pre_fix.jsonl` | 3,105 | bare_schema, descriptions, metrics, examples, templates | The same grid on the **pre-fix** contexts (tolerant re-grade). Backs the "wrong turn" before numbers (Opus 86% at descriptions vs 77% at metrics). |
| `context_fix_verification.jsonl` | 360 | descriptions, metrics — the 20 affected questions | The post-fix verification run (Opus 70/47 → 93/100 on the affected questions). |

`pre_fix.jsonl` intentionally predates the context-convention fixes — it is the
"before" evidence. Its contexts differ from what `src/contexts.py` renders from
the shipped ontologies (which include the fixes); a fresh run will not
reproduce its numbers, and that is the point of the section it backs.

`descriptions_plus_examples.jsonl` was run under the internal level name
`D4_examples_only`; the public file normalizes that field to the repository's
descriptive level taxonomy and regrades every stored SQL query offline with the
shipped grader. It contains 621 records: 544 strict matches, 583 tolerant
matches, one invalid-SQL failure, and no API/infrastructure failures.
