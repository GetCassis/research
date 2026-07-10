"""What is flakiness correlated with, beyond accuracy?

Per (model, level, question) cell over N=3 samples, a cell is "flaky" if it is
neither 0/3 nor 3/3. Flakiness is mechanically tied to accuracy: for a cell with
true success prob p, P(flaky) = 1 - p^3 - (1-p)^3 = 3p(1-p), an inverted-U that
is 0 at p=0 or 1 and peaks at p=0.5. So the honest question is whether model /
level / dataset correlate with flakiness ON TOP of where they put accuracy.

Reads results/runs/main.jsonl (N=3, the article grid, 3 datasets).
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

N = 3
DS_OF = {}
TIER = {}
for ds in config.DATASETS:
    for q in questions.load(ds):
        DS_OF[q["id"]] = ds
        TIER[q["id"]] = "hard" if q["difficulty"] == "hard_multistep" else "standard"


def load_cells():
    correct = defaultdict(int)
    n = defaultdict(int)
    for line in open(os.path.join(config.BASE, "results", "runs", (__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "main.jsonl"))):
        r = json.loads(line)
        k = (r["model"], r["level"], r["id"])
        correct[k] += 1 if r.get("correct") else 0
        n[k] += 1
    # each cell: (model, level, id) -> (n_correct, flaky?, acc)
    cells = []
    for k in correct:
        if n[k] != N:
            continue
        c = correct[k]
        cells.append({
            "model": k[0], "level": k[1], "dataset": DS_OF.get(k[2]),
            "tier": TIER.get(k[2]), "id": k[2],
            "n_correct": c, "acc": c / N, "flaky": 1 if 0 < c < N else 0,
        })
    return cells


def grp(cells, key):
    out = {}
    g = defaultdict(list)
    for c in cells:
        g[c[key]].append(c)
    for k, cs in g.items():
        acc = np.mean([c["acc"] for c in cs])
        fl = np.mean([c["flaky"] for c in cs])
        exp = 3 * acc * (1 - acc)  # flaky rate if every cell sat at the group mean
        out[k] = (len(cs), acc, fl, exp)
    return out


def show(title, d, order=None):
    print(title)
    print(f"  {'group':16} {'n':>5} {'acc':>6} {'flaky':>7} {'exp@mean':>9} {'obs/exp':>8}")
    keys = order or sorted(d)
    for k in keys:
        if k not in d:
            continue
        n, acc, fl, exp = d[k]
        ratio = fl / exp if exp > 0 else float("nan")
        print(f"  {str(k):16} {n:>5} {acc:>6.2f} {fl:>6.0%} {exp:>8.0%} {ratio:>8.2f}")
    print()


def main():
    cells = load_cells()
    show("By MODEL", grp(cells, "model"), ["haiku", "sonnet", "opus"])
    show("By DATASET", grp(cells, "dataset"), config.DATASETS)
    show("By TIER", grp(cells, "tier"), ["standard", "hard"])
    show("By LEVEL", grp(cells, "level"), config.LEVELS)

    # Is flakiness explained by accuracy alone? Correlate group flaky rate with
    # distance-from-0.5 across all (model x level x dataset) groups.
    g = defaultdict(list)
    for c in cells:
        g[(c["model"], c["level"], c["dataset"])].append(c)
    accs, fls = [], []
    for cs in g.values():
        if len(cs) < 3:
            continue
        accs.append(np.mean([c["acc"] for c in cs]))
        fls.append(np.mean([c["flaky"] for c in cs]))
    accs, fls = np.array(accs), np.array(fls)
    dist = np.abs(accs - 0.5)
    r_dist = np.corrcoef(dist, fls)[0, 1]
    r_acc = np.corrcoef(accs, fls)[0, 1]
    print(f"across {len(accs)} (model x level x dataset) groups:")
    print(f"  corr(flaky, |acc-0.5|) = {r_dist:+.2f}   (strong negative => flakiness is a frontier effect)")
    print(f"  corr(flaky, acc)       = {r_acc:+.2f}   (near 0 => inverted-U, not linear)")
    print()

    # Residual ambiguity: which questions are MORE flaky than their own accuracy predicts?
    # Per question, observed flaky vs binomial-at-its-mean-accuracy across its cells.
    qg = defaultdict(list)
    for c in cells:
        qg[c["id"]].append(c)
    resid = []
    for qid, cs in qg.items():
        acc = np.mean([c["acc"] for c in cs])
        fl = np.mean([c["flaky"] for c in cs])
        exp = 3 * acc * (1 - acc)
        resid.append((qid, fl - exp, fl, exp, TIER[qid]))
    resid.sort(key=lambda t: -t[1])
    print("questions MOST flaky vs their own accuracy predicts (excess noise = ambiguity, not difficulty):")
    for qid, ex, fl, exp, t in resid[:8]:
        print(f"  {qid:42} excess {ex:+.0%}  (obs {fl:.0%} vs exp {exp:.0%})  [{t}]")
    print()

    # Scatter: group flaky vs group accuracy + binomial curve.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(accs, fls, s=24, alpha=0.6, color="#1f77b4", label="model x level x dataset group")
    xs = np.linspace(0, 1, 100)
    ax.plot(xs, 3 * xs * (1 - xs), color="#c0392b", lw=2, label="3p(1-p): flaky if all cells at that accuracy")
    ax.set_xlabel("group mean accuracy")
    ax.set_ylabel("group flaky rate")
    ax.set_title("Flakiness is a frontier effect: it tracks accuracy near 0.5, not model or dataset")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(config.BASE, "results", "figures", "full_flakiness_vs_accuracy.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
