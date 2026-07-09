"""Factorial run driver. See docs/CONTRACT.md.

Runs model x level x question x sample, appends one JSON line per cell to
results/runs/<run_name>.jsonl. Resumable: cells already present (by
id|model|level|sample) are skipped. Concurrent via a thread pool.

Usage:
  python src/run.py --run pilot --datasets jaffle-shop --models haiku opus \
      --samples 3 --concurrency 8
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import questions as questions_mod
import ladder
import harness

_write_lock = threading.Lock()


def _cell_key(rec):
    return (rec["id"], rec["model"], rec["level"], rec["sample"])


def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["id"], r["model"], r["level"], r["sample"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def build_cells(datasets, models, levels, samples, difficulty=None, question_ids=None):
    diffset = set(difficulty) if difficulty else None
    idset = set(question_ids) if question_ids else None
    cells = []
    for ds in datasets:
        qs = questions_mod.load(ds)
        for q in qs:
            if q.get("gap"):
                continue
            if diffset and q.get("difficulty") not in diffset:
                continue
            if idset and q["id"] not in idset:
                continue
            avail = set(ladder.available_levels(q))
            for level in levels:
                if level not in avail:
                    continue
                for model in models:
                    for s in range(samples):
                        cells.append((model, ds, q, level, s))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--datasets", nargs="+", default=["jaffle-shop"])
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--levels", nargs="+", default=config.LEVELS)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap number of cells (debug)")
    ap.add_argument("--difficulty", nargs="+", default=None,
                    help="only questions with these difficulty tags (e.g. hard_multistep)")
    ap.add_argument("--questions", nargs="+", default=None,
                    help="only these question ids")
    args = ap.parse_args()

    out_path = os.path.join(config.BASE, "results", "runs", f"{args.run}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Ensure DBs + gold + pre-rendered contexts exist
    import contexts
    for ds in args.datasets:
        import db
        db.build(ds)
        if not all(
            os.path.exists(config.gold_path(q["id"]))
            for q in questions_mod.load(ds)
            if not q.get("gap")
        ):
            questions_mod.build_gold(ds)
        contexts.prerender(ds)  # cache contexts to data/contexts/; harness reads these

    done = _load_done(out_path)
    cells = build_cells(args.datasets, args.models, args.levels, args.samples,
                        args.difficulty, args.questions)
    todo = [c for c in cells if (c[2]["id"], c[0], c[3], c[4]) not in done]
    if args.limit:
        todo = todo[: args.limit]

    total = len(cells)
    print(f"cells total={total} done={len(done)} todo={len(todo)}", flush=True)

    counts = {"ok": 0, "correct": 0, "err": 0}

    def work(cell):
        model, ds, q, level, s = cell
        return harness.run_cell(model, ds, q, level, s, effort=config.DEFAULT_EFFORT)

    with open(out_path, "a") as out, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            with _write_lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            n += 1
            if rec.get("error"):
                counts["err"] += 1
            else:
                counts["ok"] += 1
                if rec.get("correct"):
                    counts["correct"] += 1
            if n % 10 == 0 or n == len(todo):
                print(
                    f"  {n}/{len(todo)}  ok={counts['ok']} correct={counts['correct']} err={counts['err']}",
                    flush=True,
                )

    print(f"done. wrote {len(todo)} cells to {out_path}", flush=True)
    print(f"summary: {counts}", flush=True)


if __name__ == "__main__":
    main()
