"""Cost analysis: tokens and dollars vs result quality.

Reads results/runs/full_ladder.jsonl (L0 + L1..L6, tolerant grading).
Aggregates per (model, level, tier): accuracy, mean input/output/
context tokens, mean $ per query, and $ per CORRECT answer (mean_cost / accuracy).
Builds the accuracy-vs-cost Pareto chart.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import questions

TIER = {q["id"]: ("hard" if q["difficulty"] == "hard_multistep" else "standard")
        for ds in config.DATASETS for q in questions.load(ds)}
MODELS = ["haiku", "sonnet", "opus"]
SH = {lv: lv.split("_")[0] for lv in config.LEVELS}
SH["L5_single_example"] = "L5one"
LEVELS_PLUS = config.LEVELS + list(config.COMPARISON_LEVELS)


def load():
    # The merged realistic axis (L0, L1..L4, L5_example_library, L6_answer_templates).
    p = os.path.join(config.BASE, "results", "runs", "full_ladder.jsonl")
    return [json.loads(l) for l in open(p)]


def agg(recs, tier=None):
    g = defaultdict(list)
    for r in recs:
        if tier and TIER.get(r["id"]) != tier:
            continue
        g[(r["model"], r["level"])].append(r)
    out = {}
    for k, rs in g.items():
        acc = np.mean([1 if x["correct"] else 0 for x in rs])
        cost = np.mean([x.get("cost_usd") or 0 for x in rs])
        out[k] = {
            "n": len(rs), "acc": acc,
            "in": np.mean([x.get("input_tokens") or 0 for x in rs]),
            "out": np.mean([x.get("output_tokens") or 0 for x in rs]),
            "ctx": np.mean([x.get("context_tokens") or 0 for x in rs]),
            "cost": cost,
            "cost_per_correct": (cost / acc) if acc > 0 else float("inf"),
        }
    return out


def table(recs, tier):
    a = agg(recs, tier)
    print(f"\n=== {tier.upper()} tier: tokens & cost by model x level ===")
    print(f"  {'model':7} {'level':7} {'acc':>5} {'in_tok':>7} {'out':>5} {'$/query':>9} {'$/correct':>10}")
    for m in MODELS:
        for lv in LEVELS_PLUS:
            if (m, lv) not in a:
                continue
            d = a[(m, lv)]
            cpc = "inf" if d["cost_per_correct"] == float("inf") else f"${d['cost_per_correct']:.4f}"
            print(f"  {m:7} {SH[lv]:7} {d['acc']:>5.0%} {d['in']:>7.0f} {d['out']:>5.0f} "
                  f"${d['cost']:>8.5f} {cpc:>10}")


def pareto_fig(recs, tier, fname):
    a = agg(recs, tier)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    color = {"haiku": "#d62728", "sonnet": "#ff7f0e", "opus": "#1f77b4"}
    marker = {"haiku": "o", "sonnet": "s", "opus": "^"}
    dlevels = [lv for lv in config.DERIVATION_LEVELS]
    for m in MODELS:
        xs = [a[(m, lv)]["cost"] * 1000 for lv in dlevels if (m, lv) in a]
        ys = [a[(m, lv)]["acc"] * 100 for lv in dlevels if (m, lv) in a]
        labs = [SH[lv] for lv in dlevels if (m, lv) in a]
        ax.plot(xs, ys, "-", color=color[m], marker=marker[m], label=m, alpha=0.85, markersize=7)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=6.5, color=color[m],
                        xytext=(3, 3), textcoords="offset points")
        # single-example comparison (perfect retrieval) as a hollow marker
        if (m, "L5_single_example") in a:
            d = a[(m, "L5_single_example")]
            ax.scatter([d["cost"] * 1000], [d["acc"] * 100], facecolors="none",
                       edgecolors=color[m], marker=marker[m], s=90, linewidths=1.6)
            ax.annotate("L5one", (d["cost"] * 1000, d["acc"] * 100), fontsize=6.5,
                        color=color[m], xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("cost per query (USD x1000, i.e. $0.001 units)")
    ax.set_ylabel("execution accuracy (%)")
    ax.set_title(f"Accuracy vs cost, {tier} tier (L5 = example library; hollow = single example)\nup-and-to-the-left is better value")
    ax.legend(title="model", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(config.BASE, "results", "figures", fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    recs = load()
    for tier in ["standard", "hard"]:
        table(recs, tier)
    pareto_fig(recs, "hard", "full_hard_cost_accuracy.png")
    pareto_fig(recs, "standard", "full_standard_cost_accuracy.png")


if __name__ == "__main__":
    main()
