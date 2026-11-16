#!/usr/bin/env python
"""White-box steps. One step per command; every step writes one JSON file.

From the experiment directory:

    python src/whitebox/run.py info  --scenario shop --out out/whitebox/shop   # tokenizer only, runs anywhere
    python src/whitebox/run.py grid  --scenario shop --out out/whitebox/shop   # needs the model: a GPU
    python src/whitebox/run.py perturb | surprisal | heads | attention | gradients | sample
    python src/whitebox/run.py all   --scenario shop --out out/whitebox/shop   # every model step, one model load

`--tiny` swaps the model for a random-weight Gemma 4 decoder (plumbing dry run: the numbers mean
nothing, the files are marked). Without it the model is google/gemma-4-12B-it (`wb.hf.load_model`),
which needs a GPU with about 48 GB of memory (the published runs used one NVIDIA L40S).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wb  # noqa: E402,F401  (puts src/ on sys.path for talkdemo and the scenarios)

MODEL_STEPS = ("grid", "surprisal", "perturb", "heads", "attention", "gradients", "sample")


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("info", "all", "gradcheck", *MODEL_STEPS))
    p.add_argument("--scenario", default="shop")
    p.add_argument("--out", default=None, help="output directory (default out/whitebox/<scenario>)")
    p.add_argument("--hf-model", default="google/gemma-4-12B-it")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="random-weight plumbing model, no real weights")
    p.add_argument("--steps", default=None, help="for `all`: comma-separated subset, in order")
    p.add_argument("--questions", default=None, help="comma-separated question ids, or `all`")
    p.add_argument("--conventions", default=None, help="comma-separated convention ids")
    p.add_argument("--output-prefix", default="", help="text between the prompt and the SQL (e.g. a code fence)")
    p.add_argument("--max-batch-tokens", type=int, default=16384)
    p.add_argument("--opt", action="append", default=[], help="step option key=value (JSON value), repeatable")
    return p.parse_args(argv)


def options(args: argparse.Namespace) -> dict:
    opts: dict = {"output_prefix": args.output_prefix, "max_batch_tokens": args.max_batch_tokens}
    if args.questions:
        opts["questions"] = args.questions.split(",")
    if args.conventions:
        opts["conventions"] = args.conventions.split(",")
    for kv in args.opt:
        k, _, v = kv.partition("=")
        try:
            opts[k] = json.loads(v)
        except json.JSONDecodeError:
            opts[k] = v
    return opts


def write(out_dir: Path, name: str, data: dict, args: argparse.Namespace) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data.setdefault("meta", {})
    data["meta"].update(
        step=name,
        written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        argv=" ".join(sys.argv[1:]),
        dry_run_random_weights=bool(args.tiny),
    )
    path = out_dir / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.replace(tmp, path)
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)", flush=True)
    return path


def run_step(name: str, sc, model, tok, opts: dict, out_dir: Path) -> dict:
    from wb import steps

    if name == "grid":
        return steps.run_grid(sc, model, tok, opts)
    if name == "perturb":
        return steps.run_perturb(sc, model, tok, opts)
    if name == "surprisal":
        return steps.run_surprisal(sc, model, tok, opts)
    if name == "sample":
        return steps.run_sample(sc, model, tok, opts)
    if name == "heads":
        from wb import heads

        return heads.run_heads(sc, model, tok, opts, out_dir)
    if name == "attention":
        from wb import heads

        return heads.run_attention(sc, model, tok, opts, out_dir)
    if name == "gradients":
        from wb import grads

        return grads.run_gradients(sc, model, tok, opts)
    if name == "gradcheck":
        from wb import grads

        return grads.run_gradcheck(sc, model, tok, opts)
    raise ValueError(name)


def main(argv=None) -> int:
    args = parse(argv)
    from wb.scenario import load_scenario

    sc = load_scenario(args.scenario)
    out_dir = Path(args.out or Path(__file__).resolve().parents[2] / "out" / "whitebox" / args.scenario)
    opts = options(args)
    print(f"scenario {sc.name} version {sc.version['hash']} -> {out_dir}", flush=True)

    if args.step == "info":
        from wb import steps
        from wb.model import load_tokenizer

        data = steps.run_info(sc, load_tokenizer(args.hf_model), opts)
        write(out_dir, "info", data, args)
        for c in data["contexts"]:
            print(
                f"  {c['name']:<58} prompt {c['prompt_tokens_max']:>5}  + reference SQL {c['with_reference_sql_max']:>5}"
            )
        print(f"{data['n_decisions']} decisions on {data['n_questions']} questions; {len(data['problems'])} problem(s)")
        for msg in data["problems"]:
            print("  PROBLEM:", msg)
        return 0

    from wb.model import load

    t0 = time.perf_counter()
    model, tok = load(args.hf_model, tiny=args.tiny, device=args.device, dtype=args.dtype)
    print(f"model loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    names = (args.steps.split(",") if args.steps else list(MODEL_STEPS)) if args.step == "all" else [args.step]
    failed = []
    for name in names:
        t1 = time.perf_counter()
        print(f"=== {time.strftime('%H:%M:%S', time.gmtime())} step {name}", flush=True)
        try:
            step_opts = dict(opts)  # a step never sees another step's private state
            data = run_step(name, sc, model, tok, step_opts, out_dir)
            write(out_dir, name, data, args)
        except Exception:  # one failed step must not cost the others their GPU time
            traceback.print_exc()
            failed.append(name)
        print(
            f"=== step {name} {'FAILED' if name in failed else 'done'} in {time.perf_counter() - t1:.0f}s", flush=True
        )
    if failed:
        print("FAILED steps:", ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
