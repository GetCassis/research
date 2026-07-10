"""Offline re-grade of an existing run file with the tolerant grader.

No model calls: every attempt's SQL is stored in the run record, so we re-execute
it against the (static) warehouse and apply grade.grade's tolerant pass. Records
that were already correct keep their grade (the strict hash matched). Output is a
new run file with `correct` = tolerant verdict, plus `correct_strict` (the
original) and `match_kind`.

Usage: python src/regrade.py [in.jsonl] [out.jsonl]
Also prints: flip audit (which failure-taxonomy clusters flipped, by category),
re-execution drift, and strict-vs-tolerant accuracy grids.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import yaml

import config
import grade

RUNS = os.path.join(config.RESULTS_DIR, "runs") if hasattr(config, "RESULTS_DIR") else \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "runs")
FA_DIR = os.path.join(os.path.dirname(RUNS), "failure_analysis")

LEVELS = ["bare_schema", "descriptions", "metrics", "examples", "templates"]
MODELS = ["haiku", "sonnet", "opus"]
PRESENTATION = {"output_columns_only", "output_rounding", "output_typing",
                "output_label_id_vs_name", "output_scale"}


def _err_sig(e):
    e = str(e).split("\n")[0]
    return re.sub(r'"[^"]*"', '"..."', e)[:160]


def cluster_id(r):
    key = ("EXEC_ERROR::" + _err_sig(r["error"])) if r.get("error") else \
        ("WRONG_RESULT::" + (r.get("result_hash") or "none")[:12])
    return f'{r["dataset"]}::{r["id"]}::{key}'


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUNS, "main.jsonl")
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".jsonl", "_tolerant.jsonl")
    assert out_path != in_path

    # question tier lookup
    tiers = {}
    for ds in config.DATASETS:
        for q in yaml.safe_load(open(config.questions_path(ds))):
            tiers[(ds, q["id"])] = "hard" if q.get("difficulty") == "hard_multistep" else "standard"

    # failure-taxonomy labels for the flip audit (optional)
    taxonomy = {}
    fc = os.path.join(FA_DIR, "final_classified.json")
    if os.path.exists(fc):
        taxonomy = {c["cluster_id"]: c for c in json.load(open(fc))}

    records = [json.loads(l) for l in open(in_path)]
    cache = {}  # (ds, sql) -> grade dict
    drift = 0
    flips = []

    with open(out_path, "w", encoding="utf-8") as out:
        for r in records:
            if r.get("correct"):
                r["correct_strict"] = True
                r["match_kind"] = "exact"
            elif r.get("error") and not (r.get("sql") or "").strip():
                r["correct_strict"] = r.get("correct", False)  # infra failure, unchanged
                r["match_kind"] = None
            else:
                r["correct_strict"] = False
                key = (r["dataset"], r["sql"])
                if key not in cache:
                    cache[key] = grade.grade(r["dataset"], r["sql"], r["id"])
                g = cache[key]
                if g["result_hash"] and r.get("result_hash") and g["result_hash"] != r["result_hash"]:
                    drift += 1
                r["match_kind"] = g["match_kind"]
                if g["correct"]:
                    flips.append(r)
                    r["correct"] = True
            out.write(json.dumps(r) + "\n")

    # ---- flip audit against the failure taxonomy ----
    by_cluster = defaultdict(int)
    for r in flips:
        by_cluster[cluster_id(r)] += 1
    flip_cats = Counter()
    suspicious = []
    for cid, n in by_cluster.items():
        cat = taxonomy.get(cid, {}).get("final_category", "UNKNOWN(not in taxonomy)")
        flip_cats[cat] += n
        if cat not in PRESENTATION:
            suspicious.append((cid, cat, n, taxonomy.get(cid, {}).get("diagnosis", "")[:130]))

    print(f"records: {len(records)}, flipped to correct: {len(flips)} attempts "
          f"in {len(by_cluster)} clusters; re-execution hash drift: {drift}")
    print("\nflipped attempts by taxonomy category:")
    for cat, n in flip_cats.most_common():
        mark = "" if cat in PRESENTATION else "  <-- REVIEW (not presentation-family)"
        print(f"  {cat:28s} {n:4d}{mark}")
    if suspicious:
        print("\nnon-presentation flips (potential tolerant false positives):")
        for cid, cat, n, d in sorted(suspicious, key=lambda x: -x[2]):
            print(f"  [{n}] {cid}\n      {cat}: {d}")

    print("\nmatch_kind distribution over flips:",
          dict(Counter(r["match_kind"] for r in flips)))

    # ---- strict vs tolerant accuracy grids ----
    stats = defaultdict(lambda: [0, 0, 0])  # (tier, model, level) -> [n, strict, tolerant]
    for r in records:
        if r.get("error") and not (r.get("sql") or "").strip():
            continue
        s = stats[(tiers[(r["dataset"], r["id"])], r["model"], r["level"])]
        s[0] += 1
        s[1] += bool(r["correct_strict"])
        s[2] += bool(r["correct"]) and not r.get("error")
    for tier in ("standard", "hard"):
        print(f"\n=== {tier.upper()} tier accuracy: strict -> tolerant ===")
        print(f"{'level':18s}" + "".join(f"{m:>18s}" for m in MODELS))
        for l in LEVELS:
            row = f"{l:18s}"
            for m in MODELS:
                n, s, t = stats[(tier, m, l)]
                row += f"      {100*s/n:3.0f} -> {100*t/n:3.0f}" if n else "            -"
            print(row)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
