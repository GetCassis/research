# Cassis research

Open-sourced experiments from [Cassis](https://getcassis.com). Each experiment
is self-contained in its own directory, with the code, the question sets, the
published run data, and instructions to reproduce every figure and number in
the accompanying write-up.

| Experiment | Question it answers |
|---|---|
| [`derivation-distance/`](derivation-distance/) | How much does spelling things out in the context (rules → metric formulas → worked examples) help an LLM write correct analytics SQL, and how does that interact with model capability? |

## Conventions

Every experiment directory ships:

- `README.md` — design, how to reproduce the figures offline, and how to re-run
  the benchmark itself.
- `ARTICLE.md` — the write-up, with its figures in `images/`.
- `results/runs/` — the published run files the write-up's numbers come from.
- `requirements.txt` + self-testing modules in `src/`.

## License

Code and text are MIT-licensed (see [LICENSE](LICENSE)). Benchmark **data**
carries its own per-dataset licenses — notably the Olist dataset is
CC BY-NC-SA (non-commercial); see each experiment's `datasets/` directory.
