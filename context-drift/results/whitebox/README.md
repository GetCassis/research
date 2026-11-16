# Published white-box outputs (Gemma 4 12B)

The JSON files the white-box steps wrote (`src/whitebox/run.py`, one file per step), gzipped. Every
file's `meta` records the model, dtype, the command line (`argv`), and `scenario_version`: hashes
of `src/scenarios/shop.py` and of the prompt builder (`src/talkdemo/{prompt,scenario,sqlcheck}.py`)
as they were when the run was made. The shipped files hash to the same values
(`tests/test_published_data.py` checks it).

| Directory | Command (see the README, "Re-run the white-box half") | Files |
|---|---|---|
| `shop/` | `all`, every step on the standard contexts, all 19 questions; plus `gradcheck` | `grid`, `surprisal`, `perturb`, `heads`, `attention`, `gradients`, `sample`, `gradcheck` |
| `shop-aov/` | the one example that defines "order value" changed (example 4), example 2 as the control | `grid`, `heads`, `attention`, `gradients`, `sample` |
| `shop-status/` | rule against examples at the status-value token | `heads`, `attention`, `gradients` |
| `shop-ig256/` | integrated gradients with 256 steps on one cell (does the path sum close?) | `gradients` |
| `shop-gradcheck-blank/` | the gradient check with every content token moving | `gradcheck` |

The talk uses two of them: `shop/attention.json.gz` (slides 13 and 14) and `shop/perturb.json.gz`
(slide 15). The others back the white-box section of the README.

Left out: each `attention` step also wrote `attention.npz`, the full float16
[layer, head, token] attention rows of the decision position (124 MB for the three directories).
Nothing reads it; the JSON keeps the averages over the selected heads, over all heads, per layer and
per selected head. Re-running the `attention` step regenerates it. The figures that
`src/whitebox/figures.py` draws from these files are not published either (regenerate them, see
the README). For publication, `meta.argv`'s output directory was rewritten to `out/whitebox/<name>`;
nothing else was changed.
