"""Generate article-specific charts with filtered levels and reader-friendly labels.

Reads the same data as analyze.py / repeatability.py / cost.py but shows only
the four levels kept in the article. Labels are descriptive
instead of internal codes.

Usage:
    python src/article_charts.py

Outputs directly to images/ (the article image directory).
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
import questions

ARTICLE_LEVELS = ["bare_schema", "descriptions", "metrics", "examples"]
ARTICLE_DERIV = ["descriptions", "metrics", "examples"]
ARTICLE_SIDE = ["bare_schema"]

LABEL = {
    "bare_schema": "bare schema",
    "descriptions": "descriptions",
    "metrics": "metrics",
    "examples": "examples",
}

MODELS = ["haiku", "sonnet", "opus"]
MODEL_COLOR = {"haiku": "#d62728", "sonnet": "#ff7f0e", "opus": "#1f77b4"}
MODEL_MARKER = {"haiku": "o", "sonnet": "s", "opus": "^"}
MODEL_DISPLAY = {"haiku": "Haiku 4.5", "sonnet": "Sonnet 4.6", "opus": "Opus 4.8"}

TIER = {}
for ds in config.DATASETS:
    for q in questions.load(ds):
        TIER[q["id"]] = "hard" if q["difficulty"] == "hard_multistep" else "standard"

IMG_DIR = os.path.join(config.BASE, "images")
DPI = 150
N = 3

# All figures read runs on the convention-fixed contexts (tolerant grading
# applied at run time): the article-level main axis plus the re-run ablations.
RUN_FILE = "main.jsonl"
LEAN_EXAMPLES_FILE = "descriptions_plus_examples.jsonl"
ABLATION_BASE_FILE = "main.jsonl"
LOO_FILE = "heldout.jsonl"
REL_FILE = "relevance.jsonl"


def load(fname):
    path = os.path.join(config.BASE, "results", "runs", fname)
    return [json.loads(l) for l in open(path)]


def filter_levels(recs):
    return [r for r in recs if r["level"] in ARTICLE_LEVELS]


# -- Chart 01/02: accuracy vs level (per tier) --------------------------------

def accuracy_by_model_level(recs, tier=None):
    g = defaultdict(list)
    for r in recs:
        if tier and TIER.get(r["id"]) != tier:
            continue
        g[(r["model"], r["level"])].append(1 if r.get("correct") else 0)
    return {k: np.mean(v) for k, v in g.items()}


def fig_accuracy_vs_level(recs, tier, outname):
    acc = accuracy_by_model_level(recs, tier)
    fig, ax = plt.subplots(figsize=(8.5, 5))

    xmap = {lv: i for i, lv in enumerate(ARTICLE_DERIV)}
    side_x = {"bare_schema": -1}

    tick_pos = [-1] + list(range(len(ARTICLE_DERIV)))
    tick_lab = [LABEL[s] for s in ARTICLE_SIDE] + [LABEL[lv] for lv in ARTICLE_DERIV]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)

    for model in MODELS:
        xs = [xmap[lv] for lv in ARTICLE_DERIV if (model, lv) in acc]
        ys = [acc[(model, lv)] for lv in ARTICLE_DERIV if (model, lv) in acc]
        ax.plot(xs, ys, marker=MODEL_MARKER[model], color=MODEL_COLOR[model],
                label=model, linewidth=2, markersize=7)

    arm_label_done = False
    for model in MODELS:
        if (model, "bare_schema") in acc:
            label = "bare schema (incomplete)" if not arm_label_done else None
            arm_label_done = True
            ax.scatter([-1], [acc[(model, "bare_schema")]],
                       marker="*", s=170, color=MODEL_COLOR[model],
                       edgecolors="black", linewidths=0.6, zorder=5, label=label)

    ax.axvline(-0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_ylabel("Execution accuracy (fraction correct)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, outname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


# -- Chart 04: repeatability (stacked bar) ------------------------------------

def fig_repeatability(recs, outname):
    g = defaultdict(int)
    cnt = defaultdict(int)
    for r in recs:
        k = (r["model"], r["level"], r["id"])
        g[k] += 1 if r.get("correct") else 0
        cnt[k] += 1
    cell = {k: g[k] for k in g if cnt[k] == N}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {0: "#c0392b", 1: "#e88e6a", 2: "#9cc79c", N: "#2e7d32"}
    labels = {0: f"0/{N} (always wrong)", 1: f"1/{N} flaky",
              2: f"{N-1}/{N} flaky", N: f"{N}/{N} (always right)"}
    xs = list(range(len(ARTICLE_LEVELS)))
    bottoms = [0.0] * len(ARTICLE_LEVELS)
    for c in [0, 1, 2, N]:
        heights = []
        for lv in ARTICLE_LEVELS:
            ks = [k for k in cell if k[1] == lv]
            heights.append(100 * sum(1 for k in ks if cell[k] == c) / len(ks) if ks else 0)
        ax.bar(xs, heights, bottom=bottoms, color=colors[c], label=labels[c], width=0.7,
               edgecolor="white", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[l] for l in ARTICLE_LEVELS])
    ax.set_ylabel("% of (model x question) cells")
    ax.set_title(f"Repeatability across {N} samples, by context level")
    ax.legend(loc="lower center", ncol=4, fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, outname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


# -- Chart 06: cost vs accuracy (scatter with level labels) -------------------

def fig_cost_accuracy(recs, tier, outname, comparison_recs=None):
    comparison_recs = comparison_recs or []
    g = defaultdict(list)
    for r in recs + comparison_recs:
        if tier and TIER.get(r["id"]) != tier:
            continue
        g[(r["model"], r["level"])].append(r)

    agg = {}
    for k, rs in g.items():
        acc = np.mean([1 if x["correct"] else 0 for x in rs])
        cost = np.mean([x.get("cost_usd") or 0 for x in rs])
        agg[k] = {"acc": acc, "cost": cost}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in MODELS:
        xs = [agg[(m, lv)]["cost"] * 1000 for lv in ARTICLE_DERIV if (m, lv) in agg]
        ys = [agg[(m, lv)]["acc"] * 100 for lv in ARTICLE_DERIV if (m, lv) in agg]
        labs = [LABEL[lv] for lv in ARTICLE_DERIV if (m, lv) in agg]
        ax.plot(xs, ys, "-", color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                label=m, alpha=0.85, markersize=7)
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=6.5, color=MODEL_COLOR[m],
                        xytext=(3, 3), textcoords="offset points")

        lean_key = (m, "descriptions_plus_examples")
        if lean_key in agg:
            lean = agg[lean_key]
            x, y = lean["cost"] * 1000, lean["acc"] * 100
            ax.scatter([x], [y], marker="D", s=70, facecolors="none",
                       edgecolors=MODEL_COLOR[m], linewidths=1.6, zorder=5)
            ax.annotate("descriptions + examples", (x, y), fontsize=6.5,
                        color=MODEL_COLOR[m], xytext=(4, -10),
                        textcoords="offset points")

    lean_proxy = plt.Line2D([], [], marker="D", markersize=7, markerfacecolor="none",
                            markeredgecolor="#444444", linestyle="None",
                            label="descriptions + examples (no rules/metrics)")

    ax.set_xlabel("cost per query (USD x1000)")
    ax.set_ylabel("execution accuracy (%)")
    ax.set_title(f"Accuracy vs cost, {tier} tier\nup-and-to-the-left is better value")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [lean_proxy], title="model", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, outname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


# -- Chart 07: leave-one-out example library (hard tier) ----------------------
# A library written for the question (all-inclusive) vs one written for OTHER
# questions (leave-one-out), both against the metrics baseline. Templates (the
# old template bars) are dropped: that rung is no longer carried anywhere else in the
# article, so it does not belong on this chart.

def _hard_acc_by_model(recs, level):
    g = defaultdict(list)
    for r in recs:
        if r["level"] != level or TIER.get(r["id"]) != "hard":
            continue
        g[r["model"]].append(1 if r.get("correct") else 0)
    return {m: 100 * np.mean(v) for m, v in g.items()}


def fig_leave_one_out(outname):
    fr = load(ABLATION_BASE_FILE)
    loo = load(LOO_FILE)
    metrics = _hard_acc_by_model(fr, "metrics")   # baseline, no library
    allinc = _hard_acc_by_model(fr, "examples")        # library incl. this question
    held = _hard_acc_by_model(loo, "examples_heldout")             # library of OTHER questions

    C_ALL = "#1f77b4"   # example library, all-inclusive
    C_LOO = "#ff7f0e"   # example library, leave-one-out
    C_BASE = "#555555"  # metrics baseline
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, m in enumerate(MODELS):
        x0, x1 = i - w / 2 - 0.02, i + w / 2 + 0.02
        ax.bar(x0, allinc[m], width=w, color=C_ALL,
               label="example library, all-inclusive" if i == 0 else None)
        ax.bar(x1, held[m], width=w, color=C_LOO,
               label="example library, leave-one-out" if i == 0 else None)
        ax.text(x0, allinc[m] + 1.2, f"{round(allinc[m])}", ha="center", va="bottom", fontsize=9)
        ax.text(x1, held[m] + 1.2, f"{round(held[m])}", ha="center", va="bottom", fontsize=9)
        ax.hlines(metrics[m], i - 0.48, i + 0.48, color=C_BASE, linestyle="--", linewidth=1.4)
        ax.text(i + 0.5, metrics[m], f"{round(metrics[m])}", ha="left", va="center",
                fontsize=8, color=C_BASE)

    baseline_proxy = plt.Line2D([], [], color=C_BASE, linestyle="--",
                                linewidth=1.4, label="metrics (no library)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [baseline_proxy], fontsize=9, loc="upper left")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
    ax.set_ylabel("Execution accuracy (%) — hard tier")
    ax.set_title("A library written for the question vs one written for other questions\n"
                 "(leave-one-out holds out the question's own example)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.7, 2.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, outname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


# -- Chart 08: relevance ablation (hard tier) ---------------------------------
# Size-matched libraries of three examples: the three most structurally relevant
# other examples vs the three least relevant. Reference line = the full example
# library (what the three-example libraries are trying to substitute for).

def fig_relevance(outname):
    fr = load(ABLATION_BASE_FILE)
    rel = load(REL_FILE)
    library = _hard_acc_by_model(fr, "examples")
    metrics = _hard_acc_by_model(fr, "metrics")
    irr3 = _hard_acc_by_model(rel, "irrelevant_3")
    rel3 = _hard_acc_by_model(rel, "relevant_3")

    C_IRR = "#b0b0b0"   # 3 irrelevant examples
    C_REL = "#ff7f0e"   # 3 relevant examples
    C_LIB = "#555555"   # full-library reference
    C_MET = "#1f77b4"   # metrics reference
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, m in enumerate(MODELS):
        x0, x1 = i - w / 2 - 0.02, i + w / 2 + 0.02
        ax.bar(x0, irr3[m], width=w, color=C_IRR,
               label="3 irrelevant examples" if i == 0 else None)
        ax.bar(x1, rel3[m], width=w, color=C_REL,
               label="3 relevant examples" if i == 0 else None)
        ax.text(x0, irr3[m] + 1.2, f"{round(irr3[m])}", ha="center", va="bottom", fontsize=9)
        ax.text(x1, rel3[m] + 1.2, f"{round(rel3[m])}", ha="center", va="bottom", fontsize=9)
        ax.hlines(library[m], i - 0.48, i + 0.48, color=C_LIB, linestyle="--", linewidth=1.4)
        ax.text(i + 0.5, library[m], f"{round(library[m])}", ha="left", va="center",
                fontsize=8, color=C_LIB)
        ax.hlines(metrics[m], i - 0.48, i + 0.48, color=C_MET, linestyle=":", linewidth=1.6)
        ax.text(i - 0.62, metrics[m], f"{round(metrics[m])}", ha="right", va="center",
                fontsize=8, color=C_MET)

    lib_proxy = plt.Line2D([], [], color=C_LIB, linestyle="--",
                           linewidth=1.4, label="full example library")
    met_proxy = plt.Line2D([], [], color=C_MET, linestyle=":",
                           linewidth=1.6, label="metrics (no examples)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [lib_proxy, met_proxy], fontsize=9, loc="lower right")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS])
    ax.set_ylabel("Execution accuracy (%) — hard tier")
    ax.set_title("Is the example doing the work?\n"
                 "(size-matched libraries: three relevant vs three irrelevant examples)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.7, 2.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, outname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


def main():
    recs = filter_levels(load(RUN_FILE))
    lean_examples = load(LEAN_EXAMPLES_FILE)
    fig_accuracy_vs_level(recs, "standard", "01-standard-tier.png")
    fig_accuracy_vs_level(recs, "hard", "02-hard-tier-cliff.png")
    fig_repeatability(recs, "04-repeatability.png")
    fig_cost_accuracy(recs, "hard", "06-cost-accuracy.png", lean_examples)
    fig_leave_one_out("07-leave-one-out.png")
    fig_relevance("08-relevance.png")
    print("done")


if __name__ == "__main__":
    main()
