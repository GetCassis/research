#!/usr/bin/env python
"""The slide-14 attention heatmaps: where Gemma 4 12B looks when it decides whether to subtract the
refunds, on "What was our net revenue in 2025?" after it has written `SELECT SUM(amount`.

From the experiment directory:

    python src/attention_heatmap.py                       # images/14-attention-{clean,changed}.png
    python src/attention_heatmap.py --html out/heatmap    # also the HTML layout the slide images used

One row per standard example: the line of its SQL that computes the revenue, each token shaded by
the attention it receives from the decision position, averaged over the 12 causally selected heads
(`token_maps.selected_heads`); the share on the right sums that attention over the whole example
(its question line and its SQL). Two contexts: the five examples as written (`examples_clean`), and
the same with example 3 no longer subtracting refunds (`one_example_broken`).

Shading: nothing below 0.15% of the attention row, then opacity sqrt(attention / 10%), capped at 1.
The slide images used the same rule and layout, rendered from the HTML with the talk's fonts.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "results" / "whitebox" / "shop" / "attention.json.gz"
TITLES = {
    1: "Net revenue per month",
    2: "Net revenue by country",
    3: "Top 10 customers",
    4: "Average order value by segment",
    5: "Month-over-month growth",
}
CARD, INK, DIM, YELLOW = "#024555", "#f5f4f3", "#8fa9b2", "#f6ed8d"
TITLE_CLEAN, TITLE_CHANGED = "#bad0f5", "#ff9766"


def alpha(a: float) -> float:
    if a < 0.0015:
        return 0.0
    return min(1.0, (a / 0.10) ** 0.5)


def load_cell(path: Path = ATTENTION) -> dict:
    data = json.loads(gzip.decompress(path.read_bytes()))
    return next(c for c in data["cells"] if c["convention"] == "net_refunds" and c["question"] == "q_year")


def revenue_line(ctx: dict, n: int) -> tuple[list[tuple[str, float]], float]:
    """Example n's revenue line as (token text, attention) pairs, and the example's attention share."""
    toks, labs, tm = ctx["tokens"], ctx["labels"], ctx["token_maps"]["selected_heads"]
    share = sum(tm[i] for i, lab in enumerate(labs) if lab.startswith(f"ex{n}."))
    lines: list[list[int]] = [[]]
    for i in (i for i, lab in enumerate(labs) if lab == f"ex{n}.sql"):
        if "\n" in toks[i]:
            lines.append([])
        else:
            lines[-1].append(i)
    line = next(ln for ln in lines if "SUM(" in "".join(toks[i] for i in ln))
    out, text, started = [], "", False
    for i in line:
        t = toks[i]
        if not started:
            if not t.strip():
                continue
            t, started = t.lstrip(), True
        text += t
        out.append((t, tm[i]))
        if text.endswith("/ 100.0"):
            break
    assert text.endswith("/ 100.0"), text
    return out, share


def rows(cell: dict, ctx_name: str, changed: int | None) -> list[dict]:
    ctx = cell["contexts"][ctx_name]
    out = []
    for n in range(1, 6):
        tokens, share = revenue_line(ctx, n)
        title = f"{n} · {TITLES[n]}" + (" · changed" if n == changed else "")
        out.append({"n": n, "title": title, "changed": n == changed, "tokens": tokens, "share": share})
    return out


def draw_png(data: list[dict], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    title_cols, code_cols, share_cols = 34, 2 + max(sum(len(t) for t, _ in r["tokens"]) for r in data) + 2, 9
    width_chars = title_cols + code_cols + share_cols
    fig_w = 13.0
    char_w_in = fig_w * 0.94 / width_chars  # the axes span 94% of the figure width
    font = char_w_in * 72 / 0.6  # DejaVu Sans Mono: one character is 0.6 em wide
    fig, ax = plt.subplots(figsize=(fig_w, 0.62 * (len(data) + 1)))
    fig.patch.set_facecolor(CARD)
    ax.set_position([0.03, 0.03, 0.94, 0.94])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, width_chars)
    ax.set_ylim(len(data) + 0.6, -0.2)
    ax.axis("off")
    head = dict(fontsize=font * 0.8, color=YELLOW, va="center", family="DejaVu Sans")
    ax.text(0, 0.2, "EXAMPLE", **head)
    ax.text(title_cols, 0.2, "WHERE THE MODEL LOOKS", **head)
    ax.text(width_chars, 0.2, "SHARE OF ATTENTION", ha="right", **{k: v for k, v in head.items()})
    for y, r in enumerate(data, start=1):
        ax.axhline(y - 0.5, color=TITLE_CLEAN, alpha=0.28, lw=0.8)
        ax.text(0, y, r["title"].upper(), fontsize=font * 0.75, va="center", family="DejaVu Sans",
                color=TITLE_CHANGED if r["changed"] else TITLE_CLEAN)
        x = title_cols
        ax.text(x, y, "… ", fontsize=font, va="center", family="DejaVu Sans Mono", color=DIM)
        x += 2
        for t, a in r["tokens"]:
            op = alpha(a)
            if op > 0:
                ax.add_patch(FancyBboxPatch((x, y - 0.32), len(t), 0.64, boxstyle="round,pad=0,rounding_size=0.15",
                                            facecolor=YELLOW, alpha=op, linewidth=0))
            ax.text(x, y, t, fontsize=font, va="center", family="DejaVu Sans Mono",
                    color="#1a1a1a" if op > 0.5 else INK)
            x += len(t)
        ax.text(x, y, " …", fontsize=font, va="center", family="DejaVu Sans Mono", color=DIM)
        ax.text(width_chars, y, f"{100 * r['share']:.0f}%", fontsize=font * 1.3, va="center", ha="right",
                family="DejaVu Sans", color=YELLOW)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=CARD, metadata={"Software": None})
    plt.close(fig)
    return path


def write_html(data: list[dict], path: Path) -> Path:
    """The layout of the slide images, as a standalone page (generic font families)."""
    body = []
    for r in data:
        spans = []
        for t, a in r["tokens"]:
            op = alpha(a)
            style = f"background:rgba(246,237,141,{op:.2f});" + ("color:#1a1a1a;" if op > 0.5 else "") if op else ""
            spans.append(f'<span style="{style}">{html.escape(t)}</span>' if style else html.escape(t))
        colour = TITLE_CHANGED if r["changed"] else TITLE_CLEAN
        body.append(
            f'<div class="r"><div class="t" style="color:{colour}">{html.escape(r["title"])}</div>'
            f'<pre><i>… </i>{"".join(spans)}<i> …</i></pre><div class="p">{100 * r["share"]:.0f}%</div></div>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><style>
html,body{{margin:0;background:#f5f4f3}}
.card{{box-sizing:border-box;width:1664px;height:560px;background:{CARD};border-radius:12px;padding:24px 44px 20px}}
.h,.r{{display:flex;align-items:center}}
.h{{height:46px;font-family:sans-serif;font-size:24px;letter-spacing:1px;text-transform:uppercase;color:{YELLOW}}}
.r{{height:94px;border-top:1px solid rgba(186,208,245,.28)}}
.a{{width:620px}}.b{{flex:1}}.c{{width:300px;text-align:right}}
.t{{width:620px;font-family:sans-serif;font-size:26px;letter-spacing:.5px;text-transform:uppercase}}
pre{{flex:1;margin:0;font-family:monospace;font-size:34px;color:{INK};white-space:pre}}
pre span{{border-radius:5px;padding:3px 0}}
pre i{{font-style:normal;color:rgba(186,208,245,.55)}}
.p{{width:130px;text-align:right;font-family:serif;font-size:50px;color:{YELLOW}}}
</style><div class="card"><div class="h"><div class="a">Example</div><div class="b">Where the model looks</div><div class="c">Share of attention</div></div>{"".join(body)}</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(ROOT / "images"), help="where the PNGs go")
    ap.add_argument("--html", default=None, help="also write clean.html and changed.html here")
    args = ap.parse_args(argv)
    cell = load_cell()
    broken = cell["broken_example"]
    for name, ctx, changed in (("clean", "examples_clean", None), ("changed", "one_example_broken", broken)):
        data = rows(cell, ctx, changed)
        print(draw_png(data, Path(args.images) / f"14-attention-{name}.png"),
              " ".join(f"ex{r['n']} {100 * r['share']:.0f}%" for r in data))
        if args.html:
            print(write_html(data, Path(args.html) / f"{name}.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
