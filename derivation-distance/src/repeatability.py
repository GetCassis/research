"""Repeatability analysis: across the N repeated samples per (model, level,
question), how decisive are results? Bimodal (always-fail / always-pass) or is
there a flaky middle, and where does the noise concentrate?

Reads results/runs/main.jsonl (the article axis, N=3). Invalid SQL counts as
incorrect (consistent with grading). Writes a stacked-bar figure and prints the
insights.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import questions

N = 3
LEVELS = config.LEVELS
SHORT = {l: l.split("_")[0] for l in LEVELS}
MODELS = ["haiku", "sonnet", "opus"]

# tier per question id
TIER = {}
for ds in config.DATASETS:
    for q in questions.load(ds):
        TIER[q["id"]] = "hard" if q["difficulty"] == "hard_multistep" else "standard"


def load(path):
    recs = [json.loads(l) for l in open(path)]
    # invalid-SQL already has correct=False; infra failures (no sql) -> treat as incorrect attempt
    return recs


def cells(recs):
    """(model, level, id) -> number correct out of N."""
    g = defaultdict(int)
    cnt = defaultdict(int)
    for r in recs:
        k = (r["model"], r["level"], r["id"])
        g[k] += 1 if r.get("correct") else 0
        cnt[k] += 1
    # only keep cells with the full N samples
    return {k: g[k] for k in g if cnt[k] == N}


def main():
    recs = load(os.path.join(config.BASE, "results", "runs", (__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "main.jsonl")))
    cell = cells(recs)

    # ---- global decisiveness ----
    from collections import Counter
    dist = Counter(cell.values())
    total = sum(dist.values())
    decisive = dist[0] + dist[N]
    flaky = total - decisive
    print(f"cells (model x level x question, full N): {total}")
    print(f"  count distribution (correct out of {N}): " +
          ", ".join(f"{k}/{N}: {dist[k]} ({100*dist[k]/total:.0f}%)" for k in range(N + 1)))
    print(f"  DECISIVE (0 or {N}): {100*decisive/total:.0f}%   FLAKY (1..{N-1}): {100*flaky/total:.0f}%")
    print()

    # ---- flaky fraction by level ----
    def frac_flaky(keys):
        ks = [k for k in keys]
        if not ks:
            return 0.0, 0
        f = sum(1 for k in ks if 0 < cell[k] < N)
        return 100 * f / len(ks), len(ks)

    print("flaky % by level:")
    for lv in LEVELS:
        p, n = frac_flaky([k for k in cell if k[1] == lv])
        print(f"  {SHORT[lv]:3}: {p:4.0f}%  (n={n})")
    print()
    print("flaky % by model:")
    for m in MODELS:
        p, n = frac_flaky([k for k in cell if k[0] == m])
        print(f"  {m:7}: {p:4.0f}%")
    print()
    print("flaky % by tier:")
    for t in ["standard", "hard"]:
        p, n = frac_flaky([k for k in cell if TIER.get(k[2]) == t])
        print(f"  {t:9}: {p:4.0f}%")
    print()

    # among flaky cells, lean toward fail (1) or pass (2)?
    f1 = sum(1 for k in cell if cell[k] == 1)
    f2 = sum(1 for k in cell if cell[k] == 2)
    print(f"flaky lean: 1/{N} (mostly fail) {f1}  vs  {N-1}/{N} (mostly pass) {f2}")
    print()

    # ---- intrinsically noisy questions: flaky across many (model,level) cells ----
    qflaky = defaultdict(int)
    qcells = defaultdict(int)
    for k in cell:
        qflaky[k[2]] += 1 if 0 < cell[k] < N else 0
        qcells[k[2]] += 1
    ranked = sorted(qflaky.items(), key=lambda kv: -kv[1])
    print("most intrinsically-noisy questions (flaky cells / total cells):")
    for qid, f in ranked[:10]:
        print(f"  {qid:42} {f:2}/{qcells[qid]}  [{TIER.get(qid)}]")

    # ---- figure: repeatability composition by level (stacked %) ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {0: "#c0392b", 1: "#e88e6a", 2: "#9cc79c", N: "#2e7d32"}
    labels = {0: f"0/{N} (always wrong)", 1: f"1/{N} flaky", 2: f"{N-1}/{N} flaky", N: f"{N}/{N} (always right)"}
    xs = list(range(len(LEVELS)))
    bottoms = [0.0] * len(LEVELS)
    for c in [0, 1, 2, N]:
        heights = []
        for lv in LEVELS:
            ks = [k for k in cell if k[1] == lv]
            heights.append(100 * sum(1 for k in ks if cell[k] == c) / len(ks) if ks else 0)
        ax.bar(xs, heights, bottom=bottoms, color=colors[c], label=labels[c], width=0.7,
               edgecolor="white", linewidth=0.5)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[l] for l in LEVELS])
    ax.set_ylabel("% of (model x question) cells")
    ax.set_xlabel("Context level")
    ax.set_title(f"Repeatability across {N} samples, by context level\n"
                 "(decisive top+bottom bands; flaky middle bands)")
    ax.legend(loc="lower center", ncol=4, fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out = os.path.join(config.BASE, "results", "figures", "full_repeatability_by_level.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
