#!/usr/bin/env python
"""Print the headline numbers of a run from its JSON (no model, no plotting).

    python src/whitebox/summary.py --out results/whitebox/shop     # the published run
    python src/whitebox/summary.py --out out/whitebox/shop         # your own
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

MAIN = ["none", "rule_one_line", "rules_all", "ex1", "ex3", "ex5", "rules+examples"]


def load(out: Path, name: str):
    p = out / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    gz = out / f"{name}.json.gz"  # the published files are gzipped
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def show_grid(g: dict) -> None:
    m = g["meta"]
    print(f"== grid: {m['model']} {m['dtype']}, scenario {m['scenario']} {m['scenario_version']['hash']}, {m['elapsed_s']}s, "
          f"longest prompt {m['max_common_tokens']} tokens (over window: {m['over_window']})")
    layout = {(q, c) for q, per in g["decisions"].items() for c, d in per.items() if d.get("whitespace_decision")}
    by = defaultdict(list)
    for r in g["rows"]:
        by[(r["convention"], r["condition"])].append(r)
    print("P(follows), mean over questions (decisions whose token also decides the layout are left out when others exist)")
    print(f"{'convention':<16}" + "".join(f"{c[:9]:>10}" for c in MAIN) + "   n  layout  conflictA conflictB  mass(none/ex5)")
    for c in m["conventions"]:
        cells = []
        for cond in MAIN:
            rs = by.get((c, cond), [])
            clean = [r for r in rs if (r["question"], c) not in layout] or rs
            cells.append(mean([r["p_follow"] for r in clean]))
        rs = by.get((c, "none"), [])
        a = mean([r["p_follow"] for r in by.get((c, f"conflict:rule_clean+examples_broken:{c}"), [])])
        b = mean([r["p_follow"] for r in by.get((c, f"conflict:rule_broken+examples_clean:{c}"), [])])
        mass = f"{mean([r['mass'] for r in by.get((c, 'none'), [])]):.2f}/{mean([r['mass'] for r in by.get((c, 'ex5'), [])]):.2f}"
        n_lay = sum((r["question"], c) in layout for r in rs)
        print(f"{c:<16}" + "".join(f"{v:>10.2f}" for v in cells) + f"  {len(rs):>2}  {n_lay:>6}  {a:>9.2f} {b:>9.2f}  {mass}")
    print("conflictA = rules right, examples wrong; conflictB = this rule wrong, examples right (P of the house convention)")


def show_perturb(p: dict) -> None:
    print(f"\n== perturb ({p['meta']['elapsed_s']}s, {len(p['questions'])} questions): diff = log P(follows) - log P(violates), nats")
    for key, s in p["summary"].items():
        conv, rule = key.split("|rule")
        dose = " ".join(f"{k}:{v['p']:.2f}" for k, v in sorted(s["dose"].items(), key=lambda kv: int(kv[0])) if v["p"] is not None)
        line = f"{conv:<16} rule {rule}  no examples {s['no_examples']['p']:.2f}  k clean -> P: {dose}"
        print(line)
        if rule == "0":
            br = " ".join(f"ex{i}:{v['d_diff']:+.2f}" for i, v in s["break_one"].items())
            rs = " ".join(f"ex{i}:{v['d_diff']:+.2f}" for i, v in s["restore_one"].items())
            gap = s["all_clean"]["diff"] - s["all_broken"]["diff"]
            print(f"{'':<16}   all clean - all broken {gap:+.2f} | break one {br} | restore one {rs}")
            if s["near"]:
                print(f"{'':<16}   near questions: " + ", ".join(f"{k} {v['mean']:+.2f} (n={v['n']})" for k, v in s["near"].items()))


def show_surprisal(s: dict) -> None:
    print(f"\n== surprisal ({s['meta']['elapsed_s']}s): mean over {len(s['questions'])} questions")
    print(f"{'context':<16}{'bits':>8}{'saved':>8}{'at decisions':>14}{'ctx tokens':>12}{'saved/100 tok':>15}")
    for c in MAIN:
        v = s["summary"].get(c)
        if v:
            per = v["saved_per_100_context_tokens"]
            print(f"{c:<16}{v['total_bits']:>8.1f}{v['saved_bits']:>8.1f}{v['saved_at_decisions']:>14.1f}{v['context_tokens']:>12.0f}{(per if per is not None else float('nan')):>15.2f}")


def show_heads(h: dict) -> None:
    print(f"\n== heads ({h['meta']['elapsed_s']}s): clean - corrupt gap and what patching restores (denoise)")
    for r in h["pairs"]:
        last, tail = r["modes"]["last"]["denoise"], r["modes"]["tail"]["denoise"]
        cum = {c["k"]: c["share"] for c in last["cumulative"]}
        top = ", ".join(f"L{a}H{b} {100 * e / r['total'] if r['total'] else 0:+.0f}%" for a, b, e in last["top"][:5])
        noise = r["modes"]["last"].get("noise", {})
        print(f"{r['convention']:<16}{r['question']:<18} gap {r['total']:+6.2f} (M clean {r['m_clean'][2]:+.2f}, corrupt {r['m_corrupt'][2]:+.2f}) "
              f"all heads last {100 * last['all_heads'] / r['total'] if r['total'] else 0:>4.0f}% tail {100 * tail['all_heads'] / r['total'] if r['total'] else 0:>4.0f}% | "
              f"top1 {100 * (cum.get(1) or 0):.0f}% top8 {100 * (cum.get(8) or 0):.0f}% top32 {100 * (cum.get(32) or 0):.0f}% | "
              f"noise: all heads {100 * noise.get('all_heads', 0) / r['total'] if r['total'] else 0:.0f}% | floor {last['noise_floor']:.3f} | {top}")
    for c, rk in h["ranking"].items():
        cor = rk["pairwise_correlation"]
        print(f"  ranking {c}: {rk['n_pairs']} questions, effect maps correlate {cor if cor is None else round(cor, 2)}; top: " + ", ".join(f"L{a}H{b} {100 * e:.0f}%" for a, b, e in rk["top"][:8]))


def show_attention(a: dict) -> None:
    print(f"\n== attention ({a['meta']['elapsed_s']}s): mass by unit, selected heads vs plain average (first question of each convention)")
    seen = set()
    for cell in a["cells"]:
        if cell["convention"] in seen:
            continue
        seen.add(cell["convention"])
        print(f"{cell['convention']} / {cell['question']}: heads {' '.join(f'L{l}H{h}' for l, h in cell['selected_heads'])} ({cell['selection']})")
        for name, ctx in cell["contexts"].items():
            def top(key):
                if key not in ctx["unit_mass"]:
                    return "n/a"
                pairs = sorted(zip(ctx["unit_order"], ctx["unit_mass"][key]), key=lambda kv: -kv[1])[:5]
                return " ".join(f"{u}:{v:.2f}" for u, v in pairs)
            print(f"   {name:<20} log-odds {ctx['first_token_diff']:+6.2f} | selected: {top('selected_heads')} | all: {top('all_heads_all_layers')}")


def show_gradients(g: dict) -> None:
    print(f"\n== gradients ({g['meta']['elapsed_s']}s, {g['meta']['ig_steps']} steps): integrated gradients by unit, nats")
    for cell in g["cells"]:
        for name, ctx in cell["contexts"].items():
            if "error" in ctx:
                print(f"{cell['convention']:<16}{name:<20} {ctx['error']}")
                continue
            pairs = sorted(zip(ctx["unit_order"], ctx["unit_sum"]["ig"]), key=lambda kv: -abs(kv[1]))[:5]
            print(f"{cell['convention']:<16}{cell['question']:<12}{name:<20} M {ctx['m']:+6.2f} base {ctx['m_baseline']:+6.2f} gap {ctx['completeness_gap']:+6.2f} | " + " ".join(f"{u}:{v:+.2f}" for u, v in pairs))


def show_sample(s: dict) -> None:
    print(f"\n== sampled check ({s['meta']['elapsed_s']}s): {json.dumps(s['summary'])}")
    by = defaultdict(list)
    for p in s["pairs"]:
        by[(p["convention"], p["condition"])].append(p)
    print(f"{'convention':<16}" + "".join(f"{c[:9]:>12}" for c in MAIN) + "    (sampled share / read-out, mean over questions)")
    for c in s["meta"]["conventions"]:
        cells = []
        for cond in MAIN:
            ps = by.get((c, cond), [])
            cells.append(f"{mean([p['sampled_follow'] for p in ps]):.2f}/{mean([p['p_follow'] for p in ps]):.2f}" if ps else "-")
        print(f"{c:<16}" + "".join(f"{v:>12}" for v in cells))
    per = defaultdict(list)
    for p in s["pairs"]:
        if p["sampled_follow"] is not None:
            per[p["convention"]].append(abs(p["p_follow"] - p["sampled_follow"]))
    print("mean |read-out - sampled| per convention: " + ", ".join(f"{c} {mean(v):.2f}" for c, v in per.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args(argv).out)
    for name, fn in (("grid", show_grid), ("perturb", show_perturb), ("surprisal", show_surprisal), ("heads", show_heads),
                     ("attention", show_attention), ("gradients", show_gradients), ("sample", show_sample)):
        data = load(out, name)
        if data:
            fn(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
