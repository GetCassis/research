#!/usr/bin/env python
"""The talk's charts, redrawn from the published data (the slides showed the same numbers in their
own layout). The attention heatmaps of slide 14 come from `attention_heatmap.py`.

From the experiment directory:

    python src/talk_figures.py          # images/07-*.png, 11-*.png, 15-*.png, 16-*.png
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scenarios import load  # noqa: E402
from talkdemo import experiments as ex  # noqa: E402

RUNS = ROOT / "results" / "runs" / "shop"
WHITEBOX = ROOT / "results" / "whitebox" / "shop"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of a CVD-validated palette
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"


def style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    print(path)
    return path


def fig_every_convention(records: list[dict], out: Path) -> Path:
    """Slide 7: share of answers that follow every applicable convention (far questions, E1)."""
    conds = [("none", "No context"), ("rules", "Seven rules"), ("ex5", "Five examples")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.36
    for s, (runner, label) in enumerate(((SONNET, "Sonnet 5.5"), (HAIKU, "Haiku 5.5"))):
        for i, (cond, _) in enumerate(conds):
            rows = [r for r in records if r["runner"] == runner and r["experiment"] == "e1" and r["condition"] == cond]
            k = sum(all(v == "follows" for v in r["status"].values()) for r in rows)
            x = i + (s - 0.5) * (width + 0.04)
            ax.bar(x, 100 * k / len(rows), width, color=SERIES[s], label=label if i == 0 else None)
            ax.text(x, 100 * k / len(rows) + 2, f"{100 * k / len(rows):.0f}%\n{k}/{len(rows)}", ha="center",
                    va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(conds)), [c[1] for c in conds], color=INK)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, "Answers that follow every applicable convention (14 questions)", "% of answers")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, out / "07-every-convention.png")


def fig_overlap(scenario, out: Path) -> Path:
    """Slide 11: which convention each of the five standard examples writes."""
    convs = list(scenario.conventions)
    exs = list(scenario.default_examples)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, e in enumerate(exs):
        applies = set(scenario.applies(scenario.example(e)))
        for j, c in enumerate(convs):
            if c.id in applies:
                ax.add_patch(plt.Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, color=SERIES[0], linewidth=0))
    totals = [sum(c.id in scenario.applies(scenario.example(e)) for e in exs) for c in convs]
    for j, t in enumerate(totals):
        ax.text(j + 0.5, len(exs) + 0.45, str(t), ha="center", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, len(convs))
    ax.set_ylim(len(exs) + 0.9, 0)
    ax.set_xticks([j + 0.5 for j in range(len(convs))], [c.id for c in convs], rotation=30, ha="right", color=INK, fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(len(exs))], [scenario.example(e).question for e in exs], color=INK, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{len(convs)} conventions written {sum(totals)} times in the five examples (bottom row: times written)",
                 loc="left", fontsize=11, color=INK, pad=10)
    return save(fig, out / "11-overlap.png")


def fig_gemma_dose(out: Path) -> Path:
    """Slide 15: Gemma 4 12B, probability that the next token follows the changed examples."""
    per = json.loads(gzip.decompress((WHITEBOX / "perturb.json.gz").read_bytes()))
    rows = [r for r in per["rows"] if r["convention"] == "net_refunds" and r["rule"] == "0" and r["examples"]]
    n_q = len({r["question"] for r in rows})
    vals = []
    for changed in range(6):
        sel = [r["p_follow"] for r in rows if len(r["clean"]) == 5 - changed]
        vals.append(100 * (1 - sum(sel) / len(sel)))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(range(6), vals, 0.6, color=SERIES[0])
    for k, v in enumerate(vals):
        ax.text(k, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(6), [str(k) for k in range(6)], color=INK)
    ax.set_xlabel("Examples changed (refunds not subtracted), of five", color=MUTED)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, f"Gemma 4 12B: P(next token follows the changed examples), mean over {n_q} questions", "%")
    return save(fig, out / "15-gemma-dose.png")


def fig_claude_dose(records: list[dict], out: Path) -> Path:
    """Slide 16: Claude Sonnet 5.5, share of answers that follow the changed examples."""
    conds = ["ex5", "ex5:1random", "ex5:2", "ex5:3", "ex5:4", "ex5:5"]
    by = defaultdict(list)
    for r in records:
        if r["runner"] == SONNET and r["experiment"] == "e2":
            by[(r["target"], r["condition"])].append(r)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s, (target, label) in enumerate(
        (("completed_only", "Status value"), ("currency_units", "Cents to euros"), ("exclude_test", "Test orders"))
    ):
        ys = []
        for c in conds:
            rows = by[(target, c)]
            ys.append(100 * sum(r["status"][target] == "violates" for r in rows) / len(rows))
        ax.plot(range(6), ys, color=SERIES[s], linewidth=2, marker="o", markersize=6, label=label)
        ax.text(3 + 0.12, ys[3], f"{ys[3]:.0f}%", color=INK, fontsize=8, va="center")
    ax.set_xticks(range(6), [str(k) for k in range(6)], color=INK)
    ax.set_xlabel("Examples changed, of five (k = 1: a random one)", color=MUTED)
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, "Claude Sonnet 5.5: answers that follow the changed examples", "% of answers")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, out / "16-claude-dose.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(ROOT / "images"))
    out = Path(ap.parse_args(argv).images)
    records = ex.load_records(RUNS)
    scenario = load("shop")
    fig_every_convention(records, out)
    fig_overlap(scenario, out)
    fig_gemma_dose(out)
    fig_claude_dose(records, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
