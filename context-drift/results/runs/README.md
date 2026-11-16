# Published run files (behavioural half)

One JSON line per model reply, gzipped:
`<scenario>/records/<runner>/<experiment>[.<target convention>].jsonl.gz`. Read them with
`zcat`, or with `talkdemo.experiments.load_records(Path("results/runs/shop"))`.

Fields: `scenario`, `runner`, `experiment` (`e1` / `e2` / `e3`), `condition`, `target` (the
convention an E2 / E3 condition changes; null in E1), `question`, `kind` (far / near), `sample`,
`replicate`, `spec` (rebuilds the exact prompt through `talkdemo.prompt.build_prompt`; see
`ContextSpec.from_json`), `nearest_example`, `corrupted_examples`, `reply` (the model's raw text),
`sql` (the query extracted from it), `status` (`{convention: follows | violates | na | unparsed}`),
`detail` (what the checker saw, e.g. the status values filtered on), `meta` (model, stop reason,
input and output tokens, cost in USD).

| Directory | Runner | Replies | What it backs |
|---|---|---:|---|
| `shop/records/claude-sonnet-5-5/` | Claude Sonnet 5.5, no thinking before the answer | 3,582 | The talk's Claude numbers (slides 7, 9, 10, 12, 16) and the E1 / E2 / E3 tables in the README |
| `shop/records/claude-haiku-5-5/` | Claude Haiku 5.5, thinking disabled | 5,922 | The Haiku column of every table (slide 7's 67%) |
| `shop/records/claude-sonnet-5-5+thinking/` | Claude Sonnet 5.5, default adaptive thinking | 410 | The small thinking check |
| `shop_v1/records/` | both models, the first version of the scenario | 2,704 (Haiku 2,592, Sonnet 112) | The dropped first version (pilot sizes) |

`context_tokens.json` holds the context sizes of the E1 conditions, counted with the token-counting
endpoint (`python src/run.py tokens`).

Two edits were made for publication, neither touching a result: each record's `meta.request_id`
(the API request id) was removed, and the files were gzipped. `python src/talk_numbers.py
--rescore` re-scores every published `reply` with the shipped checkers and finds the stored
`status` everywhere; `python src/run.py report` rebuilds the tables quoted in the README.
