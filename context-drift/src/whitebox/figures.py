#!/usr/bin/env python
"""Diagnostic figures from the JSON of `run.py`: static SVG images plus one HTML token heatmap.

    python src/whitebox/figures.py --out out/whitebox/shop          # writes out/whitebox/shop/figures/
    python src/whitebox/figures.py --out results/whitebox/shop --figures out/whitebox-figures/shop

Plain on purpose. No plotting library: the SVG is written directly, so this runs anywhere.
Colours follow a validated reference palette (blue / orange / aqua categorical, one-hue blue ramp for magnitude, blue <-> red around a gray midpoint for sign);
text is always ink, never the series colour. A file made from a `--tiny` dry run is stamped as such.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
from collections import defaultdict
from pathlib import Path

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
MID = "#f0efec"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

MAIN = ["none", "rule_one_line", "rules_all", "ex1", "ex3", "ex5", "rules+examples"]
SHORT = {
    "none": "none",
    "rule_one_line": "1-line",
    "rules_all": "rules",
    "ex1": "1 ex",
    "ex3": "3 ex",
    "ex5": "5 ex",
    "rules+examples": "both",
}
LONG = {
    "none": "no context",
    "rule_one_line": "the one-line rule",
    "rules_all": "all rules",
    "ex1": "1 example",
    "ex3": "3 examples",
    "ex5": "5 examples",
    "rules+examples": "rules + 5 examples",
}
CONTEXT_LABELS = {  # in display order; {b} = the broken example, {o} = the control example
    "examples_clean": "5 examples, all clean",
    "one_example_broken": "example {b} broken",
    "one_example_broken+rules": "example {b} broken, 7 rules right",
    "other_example_broken": "example {o} broken (control)",
    "examples_broken": "all examples broken",
    "rule_only": "7 rules, no examples",
    "rule_wrong_only": "7 rules, this one wrong",
    "rule_vs_examples": "rules right, examples wrong",
    "rule_wrong_vs_examples": "this rule wrong, examples right",
    "one_line_vs_examples": "one-line rule right, examples wrong",
    "one_line_wrong_vs_examples": "one-line rule wrong, examples right",
}


def contexts_of(cell: dict) -> list[tuple[str, str]]:
    """(context key, label) of the contexts a cell holds, in display order."""
    fill = {"b": cell.get("broken_example") or "?", "o": cell.get("other_example") or "?"}
    out = [(k, lab.format(**fill)) for k, lab in CONTEXT_LABELS.items() if k in cell["contexts"]]
    return [(k, lab) for k, lab in out if "error" not in cell["contexts"][k]]


# ---------------------------------------------------------------------------- tiny SVG layer


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(_hex(a), _hex(b)))


def seq(t: float) -> str:
    """Magnitude 0..1 -> one-hue ramp, light to dark (0 recedes into the surface)."""
    if not t > 0:
        return SURFACE
    x = max(0.0, min(1.0, t)) * (len(RAMP) - 1)
    i = min(int(x), len(RAMP) - 2)
    return mix(RAMP[i], RAMP[i + 1], x - i) if t > 0.04 else mix(SURFACE, RAMP[0], t / 0.04)


def div(t: float) -> str:
    """Signed -1..1 -> red (negative) / gray midpoint / blue (positive)."""
    t = max(-1.0, min(1.0, t))
    return mix(MID, "#1c5cab" if t > 0 else "#b3261e", abs(t) ** 0.8)


class Svg:
    def __init__(self, w: int, h: int, title: str, subtitle: str = "", stamp: str = ""):
        """Callers lay the plot out as if the header took two subtitle lines; a longer (wrapped)
        subtitle pushes everything drawn afterwards down, and a stamp adds a line under the title."""
        import textwrap

        lines = [seg for s in subtitle.split("\n") if s for seg in textwrap.wrap(s, max(40, int((w - 48) / 6.1)))]
        dy = 16 * max(0, len(lines) - 2) + (18 if stamp else 0)
        self.w, self.h = w, h + dy
        self.p = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + dy}" width="{w}" height="{h + dy}" font-family=\'{FONT}\'>',
            f'<rect width="{w}" height="{h + dy}" fill="{SURFACE}"/>',
        ]
        self.text(24, 30, title, 16, INK, weight=600)
        y = 50
        if stamp:
            self.text(24, y, stamp, 12, "#b3261e", weight=600)
            y += 18
        for i, line in enumerate(lines):
            self.text(24, y + 16 * i, line, 12, INK2)
        self.p.append(f'<g transform="translate(0 {dy})">')

    def text(self, x, y, s, size=11, fill=INK2, anchor="start", weight=None, rotate=None):
        extra = f' font-weight="{weight}"' if weight else ""
        if rotate is not None:
            extra += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
        self.p.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{html.escape(str(s))}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1.0):
        self.p.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
        )

    def rect(self, x, y, w, h, fill, rx=0.0, title=None):
        t = f"<title>{html.escape(title)}</title>" if title else ""
        self.p.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{fill}" rx="{rx}">{t}</rect>'
        )

    def dot(self, x, y, r, fill, opacity=1.0, ring=True, title=None):
        t = f"<title>{html.escape(title)}</title>" if title else ""
        ring_s = f' stroke="{SURFACE}" stroke-width="2"' if ring else ""
        self.p.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"{ring_s}>{t}</circle>'
        )

    def path(self, pts, stroke, width=2.0):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        self.p.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def bar(self, x, y0, y1, w, fill, title=None):
        """A column from baseline y0 to y1: rounded data end, square at the baseline."""
        top, bot = min(y0, y1), max(y0, y1)
        hgt = bot - top
        if hgt < 0.5:
            return
        r = min(4.0, w / 2, hgt)
        up = y1 < y0
        if up:
            d = f"M{x:.1f},{bot:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{bot:.1f} Z"
        else:
            d = f"M{x:.1f},{top:.1f} V{bot - r:.1f} Q{x:.1f},{bot:.1f} {x + r:.1f},{bot:.1f} H{x + w - r:.1f} Q{x + w:.1f},{bot:.1f} {x + w:.1f},{bot - r:.1f} V{top:.1f} Z"
        t = f"<title>{html.escape(title)}</title>" if title else ""
        self.p.append(f'<path d="{d}" fill="{fill}">{t}</path>')

    def legend(self, x, y, items):
        for label, colour in items:
            self.p.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{colour}"/>')
            self.text(x + 15, y, label, 11, INK2)
            x += 15 + 6.2 * len(label) + 18

    def yaxis(self, x0, x1, y_of, ticks, fmt="{:g}"):
        for v in ticks:
            self.line(x0, y_of(v), x1, y_of(v), AXIS if v == 0 else GRID)
            self.text(x0 - 5, y_of(v) + 3.5, fmt.format(v), 9, MUTED, anchor="end")

    def save(self, path: Path) -> Path:
        path.write_text("\n".join(self.p + ["</g>", "</svg>"]))
        return path


def lin(d0, d1, r0, r1):
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load(out: Path, name: str) -> dict | None:
    p = out / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    gz = out / f"{name}.json.gz"  # the published files are gzipped
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else None


def stamp_of(data: dict) -> str:
    return "DRY RUN: random weights, numbers mean nothing" if data["meta"].get("dry_run_random_weights") else ""


def credit(data: dict) -> str:
    m = data["meta"]
    return f"{m.get('model', '')} · scenario {m.get('scenario')} {m.get('scenario_version', {}).get('hash', '')} · thinking off"


def panels(n: int, cols: int, x0: int, y0: int, pw: int, ph: int, gx: int = 34, gy: int = 46):
    return [(x0 + (i % cols) * (pw + gx), y0 + (i // cols) * (ph + gy)) for i in range(n)]


# ---------------------------------------------------------------------------- behaviour


def fig_grid(grid: dict, dst: Path) -> Path:
    convs = grid["meta"]["conventions"]
    by = defaultdict(list)
    for r in grid["rows"]:
        by[(r["convention"], r["condition"])].append(r)
    conds = [c for c in MAIN if any(k[1] == c for k in by)]
    cols, pw, ph = 4, 200, 150
    rows = math.ceil(len(convs) / cols)
    svg = Svg(
        48 + cols * (pw + 34),
        96 + rows * (ph + 58),
        "Probability of following each convention, by context (decision-token read-out)",
        "P(follows) / (P(follows) + P(violates)) at the token that decides, teacher-forced on the reference SQL so far. Large dot = mean "
        "over the test questions, small dots = each question. Left out of the mean (and drawn in orange): questions where the two "
        "alternatives start with a space against a newline, so the token decides the layout too.\n"
        + credit(grid)
        + " · contexts: "
        + ", ".join(f"{SHORT[c]} = {LONG[c]}" for c in conds),
        stamp_of(grid),
    )
    one_line = set(grid["meta"].get("in_one_line_rule", []))
    layout = {(q, c) for q, per in grid["decisions"].items() for c, d in per.items() if d.get("whitespace_decision")}
    for (px, py), conv in zip(panels(len(convs), cols, 44, 96, pw, ph, gy=58), convs):
        y_of = lin(0, 1, py + ph, py + 14)
        svg.text(px, py + 4, conv + ("  (in the one-line rule)" if conv in one_line else ""), 11, INK, weight=600)
        svg.yaxis(px, px + pw, y_of, [0, 0.5, 1])
        slot = pw / len(conds)
        n_all = n_lay = 0
        for i, c in enumerate(conds):
            cx = px + slot * (i + 0.5)
            rs = by.get((conv, c), [])
            clean = [r for r in rs if (r["question"], conv) not in layout]
            n_all, n_lay = len(rs), len(rs) - len(clean)
            for j, r in enumerate(rs):
                lay = (r["question"], conv) in layout
                svg.dot(
                    cx + (j % 5 - 2) * 2.2,
                    y_of(r["p_follow"]),
                    2,
                    ORANGE if lay else BLUE,
                    0.35,
                    ring=False,
                    title=f"{r['question']}: {r['p_follow']:.2f} (the two alternatives cover {r['mass']:.2f} of the model's choice)",
                )
            m = mean([r["p_follow"] for r in (clean or rs)])
            if m is not None:
                svg.dot(
                    cx,
                    y_of(m),
                    4.5,
                    BLUE if clean else ORANGE,
                    title=f"{conv} / {LONG[c]}: mean {m:.2f} over {len(clean or rs)} questions",
                )
            svg.text(cx, py + ph + 13, SHORT[c], 9, MUTED, anchor="middle")
        svg.text(px, py + ph + 27, f"{n_all} questions" + (f", {n_lay} layout-sensitive" if n_lay else ""), 9, INK2)
    return svg.save(dst / "grid.svg")


def fig_sampled_grid(sam: dict, dst: Path) -> Path | None:
    """What the model writes (checkers on 20 samples) next to the read-out, on the sampled questions."""
    convs = sam["meta"]["conventions"]
    by = defaultdict(list)
    for p in sam["pairs"]:
        by[(p["convention"], p["condition"])].append(p)
    conds = [c for c in MAIN if any(k[1] == c for k in by)]
    if not conds:
        return None
    convs = [c for c in convs if any(k[0] == c for k in by)]  # only the conventions the sampled questions exercise
    cols, pw, ph = 4, 200, 150
    rows = math.ceil(len(convs) / cols)
    n = sam["pairs"][0]["n"] if sam["pairs"] else 0
    svg = Svg(
        48 + cols * (pw + 34),
        112 + rows * (ph + 46),
        "What the model writes, against the decision-token read-out",
        f"Share of {n} sampled answers per cell that follow the convention (the scenario's SQL checkers), and the teacher-forced "
        f"read-out on the same {len(sam['questions'])} questions; means over questions.\n"
        + credit(sam)
        + " · contexts: "
        + ", ".join(f"{SHORT[c]} = {LONG[c]}" for c in conds),
        stamp_of(sam),
    )
    svg.legend(44, 90, [("sampled answers", ORANGE), ("decision-token read-out", BLUE)])
    for (px, py), conv in zip(panels(len(convs), cols, 44, 112, pw, ph), convs):
        y_of = lin(0, 1, py + ph, py + 14)
        svg.text(px, py + 4, conv, 11, INK, weight=600)
        svg.yaxis(px, px + pw, y_of, [0, 0.5, 1])
        slot = pw / len(conds)
        for i, c in enumerate(conds):
            cx = px + slot * (i + 0.5)
            ps = by.get((conv, c), [])
            a, b = mean([p["sampled_follow"] for p in ps]), mean([p["p_follow"] for p in ps])
            if a is not None and b is not None:
                svg.line(cx - 4, y_of(a), cx + 4, y_of(b), AXIS, 1.5)
            if b is not None:
                svg.dot(cx + 4, y_of(b), 4, BLUE, title=f"{conv} / {LONG[c]}: read-out {b:.2f}")
            if a is not None:
                svg.dot(cx - 4, y_of(a), 4, ORANGE, title=f"{conv} / {LONG[c]}: sampled {a:.2f}")
            svg.text(cx, py + ph + 13, SHORT[c], 9, MUTED, anchor="middle")
    return svg.save(dst / "sampled_grid.svg")


def conflict_values(grid: dict | None, sam: dict | None) -> dict:
    """(convention, condition) -> values of P(follows): the decision-token read-out of the grid
    (layout-sensitive decisions left out when there are others), or the sampled shares."""
    if sam is not None:
        out = defaultdict(list)
        for p in sam["pairs"]:
            if p["sampled_follow"] is not None:
                out[(p["convention"], p["condition"])].append(p["sampled_follow"])
        return out
    layout = {(q, c) for q, per in grid["decisions"].items() for c, d in per.items() if d.get("whitespace_decision")}
    every, clean = defaultdict(list), defaultdict(list)
    for r in grid["rows"]:
        every[(r["convention"], r["condition"])].append(r["p_follow"])
        if (r["question"], r["convention"]) not in layout:
            clean[(r["convention"], r["condition"])].append(r["p_follow"])
    return {k: clean.get(k) or v for k, v in every.items()}


def fig_conflicts(data: dict, dst: Path, sampled: bool = False) -> Path | None:
    """Rule against examples, with the seven rules and with the one-line rule."""
    convs = data["meta"]["conventions"]
    by = conflict_values(None, data) if sampled else conflict_values(data, None)
    rows = []
    for c in convs:
        for kind, lab, ref in (("rule", "seven rules", "rules+examples"), ("one_line", "one-line rule", "ex5")):
            a, b = f"conflict:{kind}_clean+examples_broken:{c}", f"conflict:{kind}_broken+examples_clean:{c}"
            if (c, a) in by and (c, b) in by:
                rows.append((c, lab, a, b, ref))
    if not rows:
        return None
    w, x0, x1 = 900, 250, 760
    what = "Share of sampled answers that follow" if sampled else "Decision-token P(following"
    svg = Svg(
        w,
        120 + 34 * len(rows),
        "When the rule and the examples disagree, which one wins?" + (" (sampled answers)" if sampled else ""),
        f"{what} the house convention{'' if sampled else ')'}, mean over the test questions"
        + ("" if sampled else " (layout-sensitive decisions left out)")
        + ". One side states or shows the wrong computation; the tick is the context with nothing wrong.\n"
        + credit(data),
        stamp_of(data),
    )
    svg.legend(x0, 88, [("rule right, examples wrong", BLUE), ("rule wrong, examples right", ORANGE)])
    x_of = lin(0, 1, x0, x1)
    top = 104
    for v in (0, 0.25, 0.5, 0.75, 1):
        svg.line(x_of(v), top, x_of(v), top + 34 * len(rows), AXIS if v == 0 else GRID)
        svg.text(x_of(v), top + 34 * len(rows) + 14, f"{v:g}", 9, MUTED, anchor="middle")
    for i, (c, lab, ka, kb, kref) in enumerate(rows):
        y = top + 34 * i + 17
        a, b, ref = mean(by[(c, ka)]), mean(by[(c, kb)]), mean(by.get((c, kref), []))
        svg.text(x0 - 10, y + 4, f"{c}, {lab}", 11, INK, anchor="end")
        svg.line(x_of(a), y, x_of(b), y, AXIS, 2)
        if ref is not None:
            svg.line(x_of(ref), y - 7, x_of(ref), y + 7, MUTED, 2)
        svg.dot(x_of(a), y, 5, BLUE, title=f"{c}, {lab}: rule right, examples wrong: {a:.2f}")
        svg.dot(x_of(b), y, 5, ORANGE, title=f"{c}, {lab}: rule wrong, examples right: {b:.2f}")
        svg.text(x1 + 12, y + 4, f"{a:.2f} / {b:.2f}", 10, INK2)
    return svg.save(dst / ("conflicts_sampled.svg" if sampled else "conflicts.svg"))


def fig_picture_readout(grid: dict, sam: dict | None, dst: Path) -> Path | None:
    """One changed example: P(follows) for every question the convention applies to, in the picture
    contexts that the grid ran as conditions (`picture:<context>`), with the sampled share when there is one."""
    names = [c["name"] for c in grid["conditions"] if c["name"].startswith("picture:")]
    if not names:
        return None
    conv = next(c["about"][0] for c in grid["conditions"] if c["name"].startswith("picture:"))
    meta = grid["meta"]
    facts = {"broken_example": "?", "other_example": "?"}
    show = [k for k in ("examples_clean", "one_example_broken", "one_example_broken+rules", "other_example_broken") if f"picture:{k}" in names]
    by = {(r["question"], r["condition"]): r for r in grid["rows"] if r["convention"] == conv}
    sby = {(p["question"], p["condition"]): p for p in (sam or {}).get("pairs", []) if p["convention"] == conv}
    spec = {c["name"]: c["spec"] for c in grid["conditions"]}
    ex_ids = meta.get("standard_examples", [])
    for key in ("one_example_broken", "other_example_broken"):
        broken = [e["id"] for e in spec.get(f"picture:{key}", {}).get("examples", []) if e["corrupt"]]
        if broken and broken[0] in ex_ids:
            facts["broken_example" if key == "one_example_broken" else "other_example"] = ex_ids.index(broken[0]) + 1
    labels = [CONTEXT_LABELS[k].format(b=facts["broken_example"], o=facts["other_example"]) for k in show]
    qs = [q for q in grid["questions"] if conv in q["applies"]]
    x0, cw, rh, top = 300, 200, 26, 132
    svg = Svg(
        x0 + cw * len(show) + 40,
        top + rh * len(qs) + 40,
        f"One changed example: does the model still follow {conv}?",
        "P(follows) at the decision token for each test question (blue), and the share of sampled answers that follow where "
        "answers were sampled (orange). The order-value questions are the ones the changed example resembles.\n" + credit(grid),
        stamp_of(grid),
    )
    svg.legend(x0, 104, [("decision-token read-out", BLUE), ("sampled answers", ORANGE)])
    for j, lab in enumerate(labels):
        svg.text(x0 + cw * j + 6, top - 8, lab, 10, INK, weight=600)
    for i, q in enumerate(qs):
        y = top + rh * i
        svg.text(x0 - 10, y + 16, f"{q['text'][:44]}", 10, INK, anchor="end")
        for j, k in enumerate(show):
            xl, xr = x0 + cw * j + 8, x0 + cw * (j + 1) - 44
            x_of = lin(0, 1, xl, xr)
            svg.line(xl, y + 13, xr, y + 13, GRID)
            r, p = by.get((q["id"], f"picture:{k}")), sby.get((q["id"], f"picture:{k}"))
            if p and p["sampled_follow"] is not None:
                svg.dot(x_of(p["sampled_follow"]), y + 13, 4, ORANGE, title=f"{q['id']} · {labels[j]}: sampled {p['sampled_follow']:.2f} ({p['n_decided']}/{p['n']})")
            if r:
                svg.dot(x_of(r["p_follow"]), y + 13, 4, BLUE, title=f"{q['id']} · {labels[j]}: read-out {r['p_follow']:.3f}, {r['diff']:+.1f} nats")
                svg.text(xr + 6, y + 17, f"{r['p_follow']:.2f}", 9, INK2)
    for j in range(len(show)):
        xl, xr = x0 + cw * j + 8, x0 + cw * (j + 1) - 44
        for v, anchor in ((0, "start"), (1, "end")):
            svg.text(xl if v == 0 else xr, top + rh * len(qs) + 12, f"{v}", 9, MUTED, anchor=anchor)
    return svg.save(dst / f"one_example_{conv}.svg")


def fig_dose(per: dict, dst: Path) -> Path:
    s = per["summary"]
    convs = [c for c in per["meta"]["conventions"] if f"{c}|rule0" in s]
    cols, pw, ph = 4, 200, 150
    rows = math.ceil(len(convs) / cols)
    svg = Svg(
        48 + cols * (pw + 34),
        112 + rows * (ph + 46),
        "How many clean examples does it take? (k of n, the others carry the other convention)",
        "P(follows), mean over questions and over the choices of which examples are clean. All standard examples are shown; n = the "
        "examples that exercise the convention.\n" + credit(per),
        stamp_of(per),
    )
    states = [("0", "no rules", BLUE), ("+", "all rules, clean", AQUA), ("-", "this rule wrong", ORANGE)]
    svg.legend(44, 90, [(lab, col) for _, lab, col in states])
    for (px, py), conv in zip(panels(len(convs), cols, 44, 112, pw, ph), convs):
        n = len(s[f"{conv}|rule0"]["carriers"])
        y_of, x_of = lin(0, 1, py + ph, py + 14), lin(0, max(n, 1), px + 14, px + pw - 14)
        svg.text(px, py + 4, f"{conv} (n = {n})", 11, INK, weight=600)
        svg.yaxis(px, px + pw, y_of, [0, 0.5, 1])
        for k in range(n + 1):
            svg.text(x_of(k), py + ph + 13, str(k), 9, MUTED, anchor="middle")
        for rk, lab, col in states:
            d = s.get(f"{conv}|rule{rk}")
            if not d:
                continue
            pts = [
                (x_of(int(k)), y_of(v["p"]))
                for k, v in sorted(d["dose"].items(), key=lambda kv: int(kv[0]))
                if v["p"] is not None
            ]
            svg.path(pts, col)
            for (x, y), (k, v) in zip(pts, sorted(d["dose"].items(), key=lambda kv: int(kv[0]))):
                svg.dot(x, y, 4, col, title=f"{conv}, {lab}: {k} clean of {n}: {v['p']:.2f}")
        svg.text(px + pw / 2, py + ph + 27, "clean examples", 9, MUTED, anchor="middle")
    return svg.save(dst / "dose.svg")


def fig_one_place(per: dict, dst: Path) -> Path:
    s = per["summary"]
    convs = [c for c in per["meta"]["conventions"] if f"{c}|rule0" in s]
    cols, pw, ph = 4, 200, 150
    rows = math.ceil(len(convs) / cols)
    vals = [v["d_diff"] for c in convs for kind in ("break_one", "restore_one") for v in s[f"{c}|rule0"][kind].values()]
    lim = max(1.0, max((abs(v) for v in vals if v is not None), default=1.0))
    lim = math.ceil(lim)
    svg = Svg(
        48 + cols * (pw + 34),
        112 + rows * (ph + 74),
        "Changed in one place: necessity and sufficiency of each example (no rules)",
        "Change of log P(follows) - log P(violates), in nats, mean over questions. Break one: this example alone carries the other "
        "convention (against all clean). Restore one: this example alone is clean (against all broken).\n"
        + credit(per),
        stamp_of(per),
    )
    svg.legend(44, 106, [("restore one (sufficiency)", BLUE), ("break one (necessity)", ORANGE)])
    for (px, py), conv in zip(panels(len(convs), cols, 44, 128, pw, ph, gy=74), convs):
        d = s[f"{conv}|rule0"]
        car = d["carriers"]
        y_of = lin(-lim, lim, py + ph, py + 14)
        svg.text(px, py + 4, conv, 11, INK, weight=600)
        svg.yaxis(px, px + pw, y_of, [-lim, 0, lim], "{:+g}")
        slot = pw / max(len(car), 1)
        bw = min(18.0, slot / 2 - 3)
        for i, e in enumerate(car):
            cx = px + slot * (i + 0.5)
            r, b = d["restore_one"][str(e)]["d_diff"], d["break_one"][str(e)]["d_diff"]
            svg.bar(cx - bw - 1, y_of(0), y_of(r), bw, BLUE, f"{conv}: restore example {e}: {r:+.2f} nats")
            svg.bar(cx + 1, y_of(0), y_of(b), bw, ORANGE, f"{conv}: break example {e}: {b:+.2f} nats")
            svg.text(cx, py + ph + 13, f"ex {e}", 9, MUTED, anchor="middle")
        tot = d["all_clean"]["diff"] - d["all_broken"]["diff"] if d["all_clean"]["diff"] is not None else None
        if tot is not None:
            svg.text(px, py + ph + 28, f"all clean - all broken: {tot:+.2f} nats", 9, INK2)
        nr = d.get("near", {})
        for line, kind in enumerate(("break", "restore")):
            a, b = nr.get(f"{kind}_nearest", {}), nr.get(f"{kind}_other", {})
            if a.get("n") and b.get("n"):
                svg.text(
                    px,
                    py + ph + 40 + 12 * line,
                    f"near questions, {kind} nearest {a['mean']:+.1f} / another {b['mean']:+.1f}",
                    9,
                    INK2,
                )
    return svg.save(dst / "one_place.svg")


# ---------------------------------------------------------------------------- surprisal


def fig_surprisal(sur: dict, dst: Path, qid: str | None = None) -> Path:
    qid = qid or sur["questions"][0]["id"]
    toks = sur["tokens"][qid]
    rows = {r["condition"]: r for r in sur["rows"] if r["question"] == qid}
    conds = [c for c in MAIN if c in rows]
    n = len(toks) - 1  # the end-of-turn token is left out of the strip
    x0, cw, rh, top = 150, max(6.0, min(16.0, 1000 / max(n, 1))), 20, 190
    extra = "rule_one_line" in rows and "ex1" in rows
    svg = Svg(
        int(x0 + cw * n + 90),
        top + rh * len(conds) + 70 + (rh + 14 if extra else 0),
        f"Where the conventions live in the output: surprisal of the reference SQL ({qid})",
        "One column per token of the reference SQL, one row per context; darker = more bits (-log2 p). Marked columns are the decision "
        "tokens of each convention. Last row: what one example buys beyond the one-line rule, token by token (blue: the example saves "
        "more bits; red: the rule does).\n" + credit(sur),
        stamp_of(sur),
    )
    cap = 8.0
    for j, t in enumerate(toks[:n]):
        svg.text(x0 + cw * (j + 0.5) + 3, top - 6, t.replace("\n", "\\n").strip() or "·", 8, MUTED, rotate=-60)
    for i, c in enumerate(conds):
        y = top + rh * i
        svg.text(x0 - 8, y + 14, LONG[c], 11, INK, anchor="end")
        for j in range(n):
            b = rows[c]["bits"][j]
            svg.rect(
                x0 + cw * j,
                y + 1,
                cw - 1,
                rh - 2,
                seq(min(b, cap) / cap),
                title=f"{LONG[c]} · {toks[j]!r}: {b:.2f} bits",
            )
        svg.text(x0 + cw * n + 8, y + 14, f"{rows[c]['total_bits']:.1f} bits", 10, INK2)
    yb = top + rh * len(conds)
    if extra:
        yb += 14
        svg.text(x0 - 8, yb + 14, "1 example vs one-line rule", 11, INK, anchor="end")
        gain = [a - b for a, b in zip(rows["rule_one_line"]["bits"][:n], rows["ex1"]["bits"][:n])]
        for j, g in enumerate(gain):
            svg.rect(x0 + cw * j, yb + 1, cw - 1, rh - 2, div(g / cap), title=f"{toks[j]!r}: the example saves {g:+.2f} bits more")
        svg.text(x0 + cw * n + 8, yb + 14, f"{sum(gain):+.1f} bits", 10, INK2)
        yb += rh
    for k, (conv, j) in enumerate(sorted(sur["decision_token"][qid].items(), key=lambda kv: kv[1])):
        x = x0 + cw * (j + 0.5)
        svg.line(x, yb + 2, x, yb + 10 + 11 * (k % 3), INK2)
        svg.text(x + 3, yb + 14 + 11 * (k % 3), conv, 9, INK2)
    lx = x0
    for step in range(9):
        svg.rect(lx + 22 * step, yb + 48, 21, 8, seq(step / 8))
    svg.text(lx, yb + 44, "0 bits", 9, MUTED)
    svg.text(lx + 22 * 9, yb + 44, f"{cap:g}+ bits", 9, MUTED, anchor="end")
    return svg.save(dst / f"surprisal__{qid}.svg")


def fig_bits(sur: dict, dst: Path) -> Path:
    s = sur["summary"]
    conds = [c for c in MAIN if c in s and c != "none"]
    x0, w = 170, 1010
    vals = [s[c]["saved_per_100_context_tokens"] or 0 for c in conds]
    hi = max(max(vals, default=1.0), 1e-6)
    lo = min(min(vals, default=0.0), 0.0)
    x_of = lin(lo, hi, x0, w - 430)
    svg = Svg(
        w,
        110 + 34 * len(conds),
        "What a context buys: bits saved on the reference SQL per 100 context tokens",
        "Bits saved against no context (mean over the test questions), divided by the prompt tokens the context adds.\n"
        + credit(sur),
        stamp_of(sur),
    )
    svg.line(x_of(0), 86, x_of(0), 96 + 34 * len(conds), AXIS)
    for i, c in enumerate(conds):
        y = 96 + 34 * i
        v = s[c]["saved_per_100_context_tokens"] or 0
        svg.text(x0 - 10, y + 14, LONG[c], 11, INK, anchor="end")
        a, b = sorted((x_of(0), x_of(v)))
        svg.rect(a, y + 2, b - a, 18, BLUE, rx=3, title=f"{LONG[c]}: {v:.2f} bits per 100 context tokens")
        svg.text(
            max(b, x_of(0)) + 8,
            y + 15,
            f"{v:.1f}   ({s[c]['saved_bits']:.1f} bits for {s[c]['context_tokens']:.0f} tokens, {s[c]['saved_at_decisions']:.1f} at the decision tokens)",
            10,
            INK2,
        )
    return svg.save(dst / "bits_per_context_token.svg")


# ---------------------------------------------------------------------------- heads


def pair_of(rows: list[dict]) -> str:
    """The two prompts of the causal test, in words, from the rows of heads.json."""
    names = {
        "examples_clean": "the examples as written",
        "examples_broken": "every example that shows the convention broken",
        "one_example_broken": "one example broken",
        "rule_only": "the seven rules, right",
        "rule_wrong_only": "the seven rules with this one wrong",
    }
    if not rows:
        return "clean against corrupt"
    r = rows[0]
    clean, corrupt = r.get("clean_context", "examples_clean"), r.get("corrupt_context", "examples_broken")
    b = f" (example {r['broken_example']})" if corrupt == "one_example_broken" and r.get("broken_example") else ""
    return f"clean = {names.get(clean, clean)}; corrupt = {names.get(corrupt, corrupt)}{b}"


def head_share(heads: dict, conv: str, mode: str = "last", direction: str = "denoise"):
    prs = [p for p in heads["pairs"] if p["convention"] == conv and direction in p["modes"][mode]]
    if not prs:
        return None, []
    total = sum(p["total"] for p in prs)
    L, H = len(prs[0]["modes"][mode][direction]["effect"]), len(prs[0]["modes"][mode][direction]["effect"][0])
    m = [
        [
            sum(p["modes"][mode][direction]["effect"][l][h] for p in prs) / total if abs(total) > 1e-9 else 0.0
            for h in range(H)
        ]
        for l in range(L)
    ]
    return m, prs


def fig_heads(heads: dict, dst: Path) -> Path:
    convs = [c for c in heads["meta"]["conventions"] if any(p["convention"] == c for p in heads["pairs"])]
    L, H = heads["meta"]["n_layers"], heads["meta"]["n_heads"]
    cell = 7 if L <= 48 else 4
    mw, ph = L * cell, H * cell  # the map; a panel is at least wide enough for its two title lines
    pw = max(mw, 330)
    cols = 2
    rows = math.ceil(len(convs) / cols)
    svg = Svg(
        70 + cols * (pw + 60),
        136 + rows * (ph + 86),
        "Which heads carry the convention to the decision token (causal test)",
        f"Two prompts: {pair_of(heads['pairs'])}. Each cell is one head (column = layer, row = head): its output at the decision "
        "position is taken from the clean prompt and put into the corrupt one;\nthe colour is the share of the clean - corrupt gap it "
        "restores alone (summed over the "
        "tested questions). Ticks under the map: full-attention layers.  " + credit(heads),
        stamp_of(heads),
    )
    full = set(heads["meta"].get("full_attention_layers", []))
    for (px, py), conv in zip(panels(len(convs), cols, 50, 136, pw, ph, gx=60, gy=86), convs):
        m, prs = head_share(heads, conv)
        lim = max(max(abs(v) for row in m for v in row), 1e-9)
        tot = mean([p["total"] for p in prs])
        allh = mean([p["modes"]["last"]["denoise"]["all_heads"] / p["total"] for p in prs if abs(p["total"]) > 1e-9])
        svg.text(px, py - 22, conv, 11, INK, weight=600)
        svg.text(
            px,
            py - 8,
            f"gap {tot:+.2f} nats, {len(prs)} questions; all heads at this position restore {100 * (allh or 0):.0f}%",
            10,
            INK2,
        )
        pw_panel, pw = pw, mw  # the map itself is `mw` wide
        for l in range(L):
            for h in range(H):
                svg.rect(
                    px + l * cell,
                    py + h * cell,
                    cell - 0.6,
                    cell - 0.6,
                    div(m[l][h] / lim),
                    title=f"{conv} L{l} H{h}: {100 * m[l][h]:+.1f}% of the gap",
                )
            if l in full:
                svg.line(px + l * cell + cell / 2, py + ph + 2, px + l * cell + cell / 2, py + ph + 6, INK2)
            if l % 6 == 0:
                svg.text(px + l * cell + cell / 2, py + ph + 17, str(l), 9, MUTED, anchor="middle")
        svg.text(px + pw / 2, py + ph + 30, "layer", 9, MUTED, anchor="middle")
        svg.text(px - 6, py + ph / 2, "head", 9, MUTED, anchor="end")
        pw = pw_panel
        for k in range(9):
            svg.rect(px + 44 + k * 14, py + ph + 40, 13, 6, div((k - 4) / 4))
        svg.text(px + 40, py + ph + 46, f"-{100 * lim:.0f}%", 9, MUTED, anchor="end")
        svg.text(px + 44 + 9 * 14 + 4, py + ph + 46, f"+{100 * lim:.0f}% of the gap", 9, MUTED)
    return svg.save(dst / "heads.svg")


def fig_cumulative(heads: dict, dst: Path) -> Path:
    convs = [c for c in heads["meta"]["conventions"] if any(p["convention"] == c for p in heads["pairs"])]
    cols, pw, ph = 4, 200, 150
    rows = math.ceil(len(convs) / cols)
    svg = Svg(
        48 + cols * (pw + 34),
        112 + rows * (ph + 50),
        "How much of the effect the top heads carry (patched jointly)",
        f"Two prompts: {pair_of(heads['pairs'])}. Share of the clean - corrupt gap restored when the k best heads are patched together "
        "(mean over the tested questions with a gap of at least 1 nat; heads "
        "ranked on the same question).\n" + credit(heads),
        stamp_of(heads),
    )
    series = [("last", "decision position only", BLUE), ("tail", "every position after the context", ORANGE)]
    svg.legend(44, 90, [(lab, col) for _, lab, col in series])
    kmax = heads["meta"]["n_layers"] * heads["meta"]["n_heads"]
    for (px, py), conv in zip(panels(len(convs), cols, 44, 112, pw, ph, gy=50), convs):
        prs = [p for p in heads["pairs"] if p["convention"] == conv and abs(p["total"]) >= 1.0]
        if not prs:
            svg.text(px, py + 4, f"{conv}: no question with a gap of 1 nat", 11, INK, weight=600)
            continue
        y_of, x_of = lin(0, 1.2, py + ph, py + 14), lin(0, math.log2(kmax), px + 8, px + pw - 8)
        svg.text(px, py + 4, conv, 11, INK, weight=600)
        svg.yaxis(px, px + pw, y_of, [0, 0.5, 1], "{:.0%}")
        for k in (1, 4, 16, 64, 256):
            if k < kmax:
                svg.text(x_of(math.log2(k)), py + ph + 13, str(k), 9, MUTED, anchor="middle")
        for mode, lab, col in series:
            ks = [c["k"] for c in prs[0]["modes"][mode]["denoise"]["cumulative"]]
            pts = []
            for i, k in enumerate(ks):
                v = mean([p["modes"][mode]["denoise"]["cumulative"][i]["share"] for p in prs])
                pts.append((x_of(math.log2(k)), y_of(max(-0.1, min(1.25, v)))))
            allv = mean([p["modes"][mode]["denoise"]["all_heads"] / p["total"] for p in prs])
            pts.append((x_of(math.log2(kmax)), y_of(max(-0.1, min(1.25, allv)))))
            svg.path(pts, col)
            svg.dot(*pts[-1], 4, col, title=f"{conv}: all heads, {lab}: {allv:.0%}")
        svg.text(px + pw / 2, py + ph + 27, "heads patched together (k, last point = all)", 9, MUTED, anchor="middle")
    return svg.save(dst / "heads_cumulative.svg")


# ---------------------------------------------------------------------------- attention and gradients, per unit


def unit_groups(ctx: dict, conv: str = "") -> tuple[list[str], dict[str, str]]:
    """Display groups, in prompt order, and unit id -> group: both halves of an example together,
    the convention's own rule sentence (or the one-line rule) apart from the other rules."""
    of = {"other": "other", "special": "template", "output": "SQL so far"}
    order: list[str] = []
    for u in ctx["units"]:
        if u["kind"] == "rule":
            g = "the rule" if u["id"] in (f"rule.{conv}", "rule.one_line") else "other rules"
        elif u["group"].startswith("ex"):
            g = u["group"].replace("ex", "ex ")
        else:
            g = u["id"]
        of[u["id"]] = g
        if g not in order:
            order.append(g)
    return order + ["SQL so far", "other", "template"], of


def grouped(ctx: dict, values: list[float], conv: str = "") -> dict[str, float]:
    order, of = unit_groups(ctx, conv)
    out = dict.fromkeys(order, 0.0)
    for u, v in zip(ctx["unit_order"], values):
        out[of[u]] += v
    return out


ALL_GROUPS = [
    "instructions",
    "schema",
    "the rule",
    "other rules",
    "ex 1",
    "ex 2",
    "ex 3",
    "ex 4",
    "ex 5",
    "question",
    "SQL so far",
    "other",
    "template",
]


def state_notes(ctx: dict) -> dict[str, str]:
    """group -> "broken" / "wrong" for the examples and the rule a context changed."""
    out = {}
    for u in ctx["units"]:
        if "(broken" in u["label"]:
            out[u["group"].replace("ex", "ex ")] = "broken"
        if "(wrong" in u["label"]:
            out["the rule"] = "wrong"
    return out


def fig_attention(cell: dict, meta: dict, dst: Path) -> Path:
    conv, qid = cell["convention"], cell["question"]
    ctxs = contexts_of(cell)
    pw, ph, x0 = 660, 96, 280
    svg = Svg(
        x0 + pw + 40,
        132 + len(ctxs) * (ph + 40),
        f"Attention from the decision token, by context unit: {conv} ({qid})",
        f"Mass of the attention row of the token that decides the convention. Selected heads: the {len(cell['selected_heads'])} heads "
        f"that restore most of the gap in the causal test ({'; '.join(cell.get('selection_pair') or ['clean against corrupt'])}; "
        f"{cell['selection']}).\nPlain average: all heads of all layers, the hub detector. "
        + f"{meta.get('model', '')} · scenario {meta.get('scenario_version', {}).get('hash', '')}",
        "DRY RUN: random weights, numbers mean nothing" if meta.get("dry_run_random_weights") else "",
    )
    series = [
        ("selected_heads", "causally selected heads", BLUE),
        ("all_heads_all_layers", "all heads, all layers", ORANGE),
    ]
    svg.legend(x0, 104, [(lab, col) for _, lab, col in series])
    groups = [
        g
        for g in ALL_GROUPS
        if any(g in grouped(cell["contexts"][k], cell["contexts"][k]["unit_mass"]["all_heads_all_layers"], conv) for k, _ in ctxs)
    ]
    slot = pw / len(groups)
    bw = min(16.0, slot / 2 - 3)
    for i, (key, lab) in enumerate(ctxs):
        ctx = cell["contexts"][key]
        py = 132 + i * (ph + 40)
        y_of = lin(0, 1, py + ph, py + 6)
        svg.text(24, py + 22, lab, 12, INK, weight=600)
        svg.text(24, py + 38, f"decision-token log-odds {ctx['first_token_diff']:+.2f} nats", 10, INK2)
        notes = state_notes(ctx)
        svg.yaxis(x0, x0 + pw, y_of, [0, 0.5, 1])
        for j, g in enumerate(groups):
            cx = x0 + slot * (j + 0.5)
            for s, (name, slab, col) in enumerate(series):
                if name not in ctx["unit_mass"]:
                    continue
                v = grouped(ctx, ctx["unit_mass"][name], conv).get(g, 0.0)
                svg.bar(cx - bw - 1 + s * (bw + 2), y_of(0), y_of(v), bw, col, f"{lab} · {g} · {slab}: {v:.3f}")
            svg.text(cx, py + ph + 13, g, 9, MUTED, anchor="middle")
            if g in notes:
                svg.text(cx, py + ph + 24, notes[g], 9, INK, anchor="middle", weight=600)
    return svg.save(dst / f"attention_{conv}__{qid}.svg")


def ig_closes(ctx: dict) -> bool:
    """Is this cell's integrated-gradients map worth showing? The path sum must account for
    M(prompt) - M(baseline) to within a quarter of it (or 1 nat). On the 12B in bf16 most cells
    fail this by tens of nats (see the gradcheck step): they are kept in the JSON, not drawn."""
    return abs(ctx["completeness_gap"]) <= max(1.0, 0.25 * abs(ctx["m"] - ctx["m_baseline"]))


def fig_gradients(cell: dict, meta: dict, dst: Path) -> Path | None:
    conv, qid = cell["convention"], cell["question"]
    ctxs = [(k, lab) for k, lab in contexts_of(cell) if ig_closes(cell["contexts"][k])]
    if not ctxs:
        return None
    pw, ph, x0 = 660, 96, 280
    vals = [
        v for k, _ in ctxs for v in grouped(cell["contexts"][k], cell["contexts"][k]["unit_sum"]["ig"], conv).values()
    ]
    lim = max(1.0, math.ceil(max((abs(v) for v in vals), default=1.0)))
    svg = Svg(
        x0 + pw + 40,
        132 + len(ctxs) * (ph + 40),
        f"Integrated gradients of the decision, by context unit: {conv} ({qid})",
        "Contrastive integrated gradients: attribution of log P(follows) - log P(violates) to the prompt embeddings, in nats, summed per "
        f"unit. Baseline: {meta.get('baseline', 'a filler token')}.\nA second opinion next to the attention picture. "
        f"{meta.get('model', '')} · scenario {meta.get('scenario_version', {}).get('hash', '')}",
        "DRY RUN: random weights, numbers mean nothing" if meta.get("dry_run_random_weights") else "",
    )
    svg.legend(x0, 104, [("pushes towards the house convention", "#1c5cab"), ("pushes away from it", "#b3261e")])
    groups = [
        g
        for g in ALL_GROUPS
        if any(g in grouped(cell["contexts"][k], cell["contexts"][k]["unit_sum"]["ig"], conv) for k, _ in ctxs)
    ]
    slot = pw / max(len(groups), 1)
    bw = min(20.0, slot - 8)
    for i, (key, lab) in enumerate(ctxs):
        ctx = cell["contexts"][key]
        py = 132 + i * (ph + 40)
        y_of = lin(-lim, lim, py + ph, py + 6)
        svg.text(24, py + 22, lab, 12, INK, weight=600)
        svg.text(24, py + 38, f"M = {ctx['m']:+.2f}, baseline {ctx['m_baseline']:+.2f} nats", 10, INK2)
        svg.text(24, py + 52, f"unexplained by the path sum: {ctx['completeness_gap']:+.2f}", 10, INK2)
        svg.yaxis(x0, x0 + pw, y_of, [-lim, 0, lim], "{:+g}")
        g_of = grouped(ctx, ctx["unit_sum"]["ig"], conv)
        for j, g in enumerate(groups):
            cx = x0 + slot * (j + 0.5)
            v = g_of.get(g, 0.0)
            svg.bar(
                cx - bw / 2,
                y_of(0),
                y_of(max(-lim, min(lim, v))),
                bw,
                "#1c5cab" if v >= 0 else "#b3261e",
                f"{lab} · {g}: {v:+.3f} nats",
            )
            svg.text(cx, py + ph + 13, g, 9, MUTED, anchor="middle")
            if g in state_notes(ctx):
                svg.text(cx, py + ph + 24, state_notes(ctx)[g], 9, INK, anchor="middle", weight=600)
    return svg.save(dst / f"gradients_{conv}__{qid}.svg")


TOKEN_MAP_CONTEXTS = (
    "examples_clean",
    "one_example_broken",
    "one_example_broken+rules",
    "other_example_broken",
    "rule_vs_examples",
    "rule_wrong_vs_examples",
    "one_line_vs_examples",
    "one_line_wrong_vs_examples",
)
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def fig_tokens(cell: dict, key: str, label: str, meta: dict, dst: Path, which: str = "selected_heads") -> Path | None:
    """The prompt as text, from the rules or the first example on, each token shaded by the attention
    it gets from the decision token (mean of the causally selected heads; the plain average when no
    head was selected). The static twin of tokens.html, for one context."""
    ctx = cell["contexts"][key]
    which = which if which in ctx["token_maps"] else "all_heads_all_layers"
    vals, toks, labs = ctx["token_maps"][which], ctx["tokens"], ctx["labels"]
    start = next((i for i, lab in enumerate(labs) if lab.startswith(("rule.", "ex"))), None)
    if start is None:
        start = next((i for i, lab in enumerate(labs) if lab == "question"), 0)
    while start > 0 and labs[start - 1] == "other":  # keep the heading ("Rules:", "Examples:", "Question:")
        start -= 1
    shown = sum(vals[start:])
    top = max(max(vals[start:], default=0.0), 1e-12)
    cw, lh, x0, y0, width = 6.63, 18, 24, 104, 150
    cells, x, y = [], 0, 0
    for i in range(start, len(toks)):
        parts = toks[i].split("\n")
        for k, part in enumerate(parts):
            if k:
                x, y = 0, y + 1
            if x + len(part) > width:
                x, y = 0, y + 1
            if part:
                cells.append((x, y, part, vals[i], i))
                x += len(part)
    conv, qid = cell["convention"], cell["question"]
    n_sel = len(cell.get("selected_heads") or [])
    who = f"mean of the {n_sel} causally selected heads" if which == "selected_heads" else "plain average of all heads and layers"
    svg = Svg(
        int(x0 * 2 + cw * width),
        y0 + lh * (y + 1) + 30,
        f"What the decision token reads: {conv} ({qid}), {label}",
        f"Attention from the token that decides the convention (outlined, last token), {who}. Darker = more of the row; the text shown "
        f"holds {100 * shown:.0f}% of it (instructions and schema above are not shown). Decision-token log-odds "
        f"{ctx['first_token_diff']:+.1f} nats.\n{meta.get('model', '')} · scenario {meta.get('scenario_version', {}).get('hash', '')}",
        "DRY RUN: random weights, numbers mean nothing" if meta.get("dry_run_random_weights") else "",
    )
    for cx, cy, part, v, i in cells:
        t = (v / top) ** 0.6
        px, py = x0 + cw * cx, y0 + lh * cy
        if labs[i] != "special" and v > 0:
            svg.rect(px, py - 12, cw * len(part), 16, seq(t), title=f"{toks[i]!r} · {labs[i]} · {100 * v:.2f}% of the row")
        if i == len(toks) - 1:
            svg.p.append(f'<rect x="{px - 1:.1f}" y="{py - 13:.1f}" width="{cw * len(part) + 2:.1f}" height="18" fill="none" stroke="{INK}" stroke-width="1.5"/>')
        ink = "#ffffff" if t > 0.55 and labs[i] != "special" else (MUTED if labs[i] in ("special", "other") else INK)
        svg.p.append(
            f'<text x="{px:.1f}" y="{py:.1f}" font-size="11" font-family=\'{MONO}\' fill="{ink}" xml:space="preserve">{html.escape(part)}</text>'
        )
    out = dst / "tokens"
    out.mkdir(exist_ok=True)
    return svg.save(out / f"{conv}__{qid}__{key.replace('+', '_')}.svg")


# ---------------------------------------------------------------------------- sampled check


def fig_sampled(sam: dict, dst: Path) -> Path:
    pts = [p for p in sam["pairs"] if p["sampled_follow"] is not None]
    x0, y0, side = 70, 96, 380
    s = sam["summary"]
    note = f"{len(pts)} cells (context x question x convention), {sam['pairs'][0]['n'] if sam['pairs'] else 0} generations each"
    if "pearson" in s:
        note += f"; r = {s['pearson']:.2f}, same side of 0.5 in {100 * s['same_side_of_half']:.0f}%, mean gap {s['mean_abs_gap']:.2f}"
    svg = Svg(
        x0 + side + 60,
        y0 + side + 70,
        "Does the decision-token probability predict what the model writes?",
        note + f"; {100 * s.get('fenced_share', 0):.0f}% of answers open a code fence\n" + credit(sam),
        stamp_of(sam),
    )
    x_of, y_of = lin(0, 1, x0, x0 + side), lin(0, 1, y0 + side, y0)
    for v in (0, 0.5, 1):
        svg.line(x_of(v), y0, x_of(v), y0 + side, AXIS if v == 0 else GRID)
        svg.line(x0, y_of(v), x0 + side, y_of(v), AXIS if v == 0 else GRID)
        svg.text(x_of(v), y0 + side + 14, f"{v:g}", 9, MUTED, anchor="middle")
        svg.text(x0 - 6, y_of(v) + 3, f"{v:g}", 9, MUTED, anchor="end")
    svg.line(x_of(0), y_of(0), x_of(1), y_of(1), AXIS)
    for p in pts:
        svg.dot(
            x_of(p["p_follow"]),
            y_of(p["sampled_follow"]),
            4,
            BLUE,
            0.55,
            title=f"{p['condition']} / {p['question']} / {p['convention']}: read-out {p['p_follow']:.2f}, sampled {p['sampled_follow']:.2f} ({p['n_decided']} classified)",
        )
    svg.text(x0 + side / 2, y0 + side + 32, "decision-token P(follows), teacher-forced", 10, INK2, anchor="middle")
    svg.text(x0 - 40, y0 + side / 2, "share of sampled answers that follow", 10, INK2, anchor="middle", rotate=-90)
    return svg.save(dst / "sampled_check.svg")


# ---------------------------------------------------------------------------- the HTML token heatmap

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Decision-token maps</title>
<style>
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --mid:#f0efec; --pos:#1c5cab; --neg:#b3261e; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --mid:#383835; --pos:#3987e5; --neg:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --mid:#383835; --pos:#3987e5; --neg:#e66767; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
h1 { font-size:18px; margin:0 0 4px; } p { color:var(--ink2); margin:4px 0 12px; }
.bar { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; margin:12px 0; }
label { color:var(--ink2); font-size:12px; } select { font:inherit; padding:3px 6px; }
#stamp { color:var(--neg); font-weight:600; }
#prompt { background:var(--surface); border:1px solid var(--grid); border-radius:6px; padding:14px; white-space:pre-wrap; word-break:break-word;
  font:12.5px/1.9 ui-monospace,SFMono-Regular,Menlo,monospace; }
#prompt span { border-radius:2px; padding:1px 0; } #prompt span.decision { outline:2px solid var(--ink); }
#prompt span.unit-start { border-left:2px solid var(--muted); padding-left:2px; }
#facts { color:var(--ink2); font-size:12px; margin:8px 0; }
#tip { position:fixed; pointer-events:none; background:var(--ink); color:var(--surface); font-size:12px; padding:4px 8px; border-radius:4px; display:none; max-width:360px; z-index:2; }
table { border-collapse:collapse; font-size:12px; margin-top:10px; font-variant-numeric:tabular-nums; } td,th { padding:2px 10px 2px 0; text-align:right; } td:first-child,th:first-child { text-align:left; }
</style></head><body><main>
<h1>What the decision token reads</h1>
<p>Each token of the prompt is shaded by the selected map, read at the token that decides the convention (outlined, at the end).
Attention maps: share of the attention row, one-hue scale. Integrated gradients: signed, blue pushes towards the house convention, red away.
A bar on the left of a token marks the start of a unit. <span id="stamp"></span></p>
<div class="bar">
 <label>decision <select id="cell"></select></label>
 <label>context <select id="ctx"></select></label>
 <label>map <select id="map"></select></label>
</div>
<div id="facts"></div>
<div id="prompt"></div>
<table id="units"></table>
</main><div id="tip"></div>
<script>
const DATA = __DATA__;
const NAMES = {selected_heads:"attention, causally selected heads", selected_heads_weighted:"attention, selected heads weighted by effect",
  all_heads_all_layers:"attention, all heads and layers (plain average)", full_attention_layers:"attention, the full-attention layers",
  ig:"integrated gradients (signed)", grad_x_input:"gradient x input (signed)"};
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function hex(c){ c=c.replace('#',''); return [0,2,4].map(i=>parseInt(c.slice(i,i+2),16)); }
function mix(a,b,t){ const x=hex(a), y=hex(b); return 'rgb('+x.map((v,i)=>Math.round(v+(y[i]-v)*t)).join(',')+')'; }
function fill(sel, items){ sel.innerHTML=''; items.forEach(([v,l])=>{ const o=document.createElement('option'); o.value=v; o.textContent=l; sel.appendChild(o); }); }
function draw(){
  const cell = DATA.cells[+$('cell').value], ctx = cell.contexts[$('ctx').value], name = $('map').value;
  const vals = ctx.maps[name]; if(!vals){ $('prompt').textContent='this map was not computed for this context'; $('units').innerHTML=''; return; }
  const signed = name==='ig' || name==='grad_x_input';
  const content = vals.map((v,i)=> ctx.labels[i]==='special' ? 0 : Math.abs(v));
  const top = Math.max(...content, 1e-12);
  const mid = css('--surface'), pos = css('--pos'), neg = css('--neg');
  const box = $('prompt'); box.innerHTML='';
  const sums = {};
  ctx.tokens.forEach((t,i)=>{
    const s = document.createElement('span'); const v = vals[i]; const lab = ctx.labels[i];
    sums[lab] = (sums[lab]||0) + v;
    const a = Math.min(1, Math.abs(v)/top);
    s.style.background = lab==='special' ? 'transparent' : mix(mid, (signed && v<0) ? neg : pos, Math.pow(a, 0.6)*0.85);
    if(a > 0.6 && lab!=='special') s.style.color = css('--surface');
    s.textContent = t; s.dataset.tip = JSON.stringify(t)+' · '+lab+' · '+(signed ? v.toFixed(4)+' nats' : (100*v).toFixed(2)+'% of the row');
    if(i>0 && lab!==ctx.labels[i-1] && !['other','special'].includes(lab)) s.className='unit-start';
    if(i===ctx.tokens.length-1) s.classList.add('decision');
    box.appendChild(s);
  });
  $('facts').textContent = cell.convention+' / '+cell.question+' · '+ctx.tokens.length+' tokens · decision-token log-odds '+
    (ctx.first_token_diff===undefined ? 'n/a' : ctx.first_token_diff.toFixed(2)+' nats')+
    (cell.selected_heads && cell.selected_heads.length ? ' · selected heads (layer.head): '+cell.selected_heads.map(h=>h.join('.')).join(' ')+' ('+cell.selection+')' : '');
  const rows = Object.entries(sums).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  $('units').innerHTML = '<tr><th>unit</th><th>'+(signed?'nats':'share of the row')+'</th></tr>'+rows.map(([k,v])=>'<tr><td>'+k+'</td><td>'+(signed?v.toFixed(3):(100*v).toFixed(1)+'%')+'</td></tr>').join('');
}
function refill(){
  const cell = DATA.cells[+$('cell').value];
  const keepC = $('ctx').value, keepM = $('map').value;
  fill($('ctx'), Object.keys(cell.contexts).map(k=>[k, cell.contexts[k].label||k]));
  if(cell.contexts[keepC]) $('ctx').value = keepC;
  const maps = new Set(); Object.values(cell.contexts).forEach(c=>Object.keys(c.maps).forEach(m=>maps.add(m)));
  fill($('map'), Object.keys(NAMES).filter(m=>maps.has(m)).map(m=>[m, NAMES[m]]));
  if(maps.has(keepM)) $('map').value = keepM;
  draw();
}
fill($('cell'), DATA.cells.map((c,i)=>[i, c.convention+' / '+c.question]));
$('stamp').textContent = DATA.stamp;
['ctx','map'].forEach(id=>$(id).addEventListener('change', draw)); $('cell').addEventListener('change', refill);
document.addEventListener('mousemove', e=>{ const t=$('tip'), d=e.target.dataset && e.target.dataset.tip;
  if(!d){ t.style.display='none'; return; } t.textContent=d; t.style.display='block';
  t.style.left=Math.min(e.clientX+12, innerWidth-380)+'px'; t.style.top=(e.clientY+16)+'px'; });
refill();
</script></body></html>
"""


def html_tokens(att: dict, grads: dict | None, dst: Path) -> Path:
    g_of = {(c["convention"], c["question"]): c for c in (grads or {}).get("cells", [])}
    cells = []
    for cell in att["cells"]:
        out = {k: cell[k] for k in ("convention", "question", "selected_heads", "selection")}
        out["contexts"] = {}
        for name, label in contexts_of(cell):
            ctx = cell["contexts"][name]
            maps = {k: [round(v, 4) for v in vals] for k, vals in ctx["token_maps"].items()}
            g = g_of.get((cell["convention"], cell["question"]), {}).get("contexts", {}).get(name)
            if g and "token_maps" in g and len(g["token_maps"]["ig"]) == len(ctx["tokens"]) and ig_closes(g):
                maps["ig"] = [round(v, 4) for v in g["token_maps"]["ig"]]
                maps["grad_x_input"] = [round(v, 4) for v in g["token_maps"]["grad_x_input"]]
            out["contexts"][name] = {
                "label": label,
                "tokens": ctx["tokens"],
                "labels": ctx["labels"],
                "maps": maps,
                "first_token_diff": ctx["first_token_diff"],
            }
        cells.append(out)
    data = {"cells": cells, "stamp": stamp_of(att)}
    path = dst / "tokens.html"
    path.write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    return path


# ---------------------------------------------------------------------------- entry point


def build(out: Path, dst: Path | None = None) -> list[Path]:
    out = Path(out)
    dst = Path(dst) if dst else out / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    grid, per, sur, heads, att, grads, sam = (
        load(out, n) for n in ("grid", "perturb", "surprisal", "heads", "attention", "gradients", "sample")
    )
    if grid:
        made += [fig_grid(grid, dst), fig_conflicts(grid, dst), fig_picture_readout(grid, sam, dst)]
    if per and per["summary"]:
        made += [fig_dose(per, dst), fig_one_place(per, dst)]
    if sur:
        ids = [q["id"] for q in sur["questions"]]
        near = [q["id"] for q in sur["questions"] if q["kind"] == "near"]
        picks = dict.fromkeys(ids[:1] + [q for q in ("q_hour", "q_aov_month") if q in ids] + near[:1])
        made += [fig_surprisal(sur, dst, q) for q in picks] + [fig_bits(sur, dst)]
    if heads and heads["pairs"]:
        made += [fig_heads(heads, dst), fig_cumulative(heads, dst)]
    if att:  # one figure per (convention, question) the step ran
        made += [fig_attention(cell, att["meta"], dst) for cell in att["cells"]]
        for cell in att["cells"]:
            for key, label in contexts_of(cell):
                if key in TOKEN_MAP_CONTEXTS:
                    made.append(fig_tokens(cell, key, label, att["meta"], dst))
        made.append(html_tokens(att, grads, dst))
    if grads:
        for cell in grads["cells"]:
            if any("error" not in c for c in cell["contexts"].values()):
                made.append(fig_gradients(cell, grads["meta"], dst))
    if sam and sam["pairs"]:
        made += [fig_sampled(sam, dst), fig_sampled_grid(sam, dst), fig_conflicts(sam, dst, sampled=True)]
    return [p for p in made if p is not None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory holding the JSON of run.py")
    ap.add_argument("--figures", default=None, help="where to write (default <out>/figures)")
    args = ap.parse_args(argv)
    for p in build(Path(args.out), args.figures):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
