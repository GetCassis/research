#!/usr/bin/env python
"""The four things the behavioural half found on Claude, read on the open model.

    python src/whitebox/claims.py --out results/whitebox   # reads shop, shop-aov, shop-status under it

1. one changed example defines a figure (order value) even with the rules right;
2. redundancy: break-one against restore-one, and the dose-response;
3. rule against examples at the status-value token;
4. what one example carries that the one-line rule does not.

Prints numbers only: decision-token read-outs (teacher-forced), sampled answers where they exist,
and where the selected heads look. Nothing is tuned here; a claim that does not hold shows as such.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    """`<name>.json`, or `<name>.json.gz` next to it (the published files are gzipped)."""
    if path.exists():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def pct(x) -> str:
    return "  n/a" if x is None or x != x else f"{100 * x:4.0f}%"


def group_mass(ctx: dict, conv: str, key: str) -> dict[str, float]:
    """Attention mass by display group: each example, the convention's rule, the other rules, the rest."""
    out: dict[str, float] = defaultdict(float)
    kinds = {u["id"]: u for u in ctx["units"]}
    for uid, v in zip(ctx["unit_order"], ctx["unit_mass"].get(key, [])):
        u = kinds.get(uid)
        if u is None:
            g = {"output": "SQL so far", "special": "template"}.get(uid, "other")
        elif u["kind"] == "rule":
            g = "the rule" if uid in (f"rule.{conv}", "rule.one_line") else "other rules"
        elif u["group"].startswith("ex"):
            g = u["group"]
        else:
            g = uid
        out[g] += v
    return out


def shapley(per: dict, conv: str) -> dict[str, dict[int, float]]:
    """Per question, each example's Shapley value for the log-odds, from the exhaustive subsets
    of clean examples the perturb step ran without rules: the average, over the orders in which
    the examples could be restored, of what restoring this one adds. Unlike break-one it credits
    redundant examples, and the values of a question sum to all clean - all broken exactly."""
    from itertools import combinations
    from math import factorial

    by: dict = defaultdict(dict)
    for r in per["rows"]:
        if r["convention"] == conv and r["rule"] == "0" and r["examples"]:
            by[r["question"]][frozenset(r["clean"])] = r["diff"]
    out = {}
    for q, m in by.items():
        players = sorted(max(m, key=len))
        n = len(players)
        if len(m) != 2**n:
            continue
        phi = {}
        for i in players:
            rest = [j for j in players if j != i]
            phi[i] = sum(
                factorial(k) * factorial(n - k - 1) / factorial(n) * (m[frozenset(sub) | {i}] - m[frozenset(sub)])
                for k in range(n)
                for sub in combinations(rest, k)
            )
        out[q] = phi
    return out


def claim1(shop: Path, aov: Path) -> None:
    print("== 1. One changed example (example 4, order value by segment: SUM(o.amount - o.refunded) -> SUM(o.amount))")
    grid, sam, heads, att = (load(aov / f"{n}.json") for n in ("grid", "sample", "heads", "attention"))
    if grid:
        ctxs = ["examples_clean", "one_example_broken", "one_example_broken+rules", "other_example_broken"]
        by = {(r["question"], r["condition"]): r for r in grid["rows"] if r["convention"] == "net_refunds"}
        sby = {(p["question"], p["condition"]): p for p in (sam or {}).get("pairs", []) if p["convention"] == "net_refunds"}
        print("   P(net of refunds) at the token after SUM(amount: read-out [sampled share, n classified]")
        print(f"   {'question':<46}" + "".join(f"{c.replace('one_example', 'ex4').replace('other_example', 'ex2'):>24}" for c in ctxs))
        for q in grid["questions"]:
            if "net_refunds" not in q["applies"]:
                continue
            cells = []
            for c in ctxs:
                r, p = by.get((q["id"], f"picture:{c}")), sby.get((q["id"], f"picture:{c}"))
                s = f" [{pct(p['sampled_follow']).strip()}, {p['n_decided']}]" if p else ""
                cells.append(f"{pct(r['p_follow']) if r else '  n/a'}{s}")
            print(f"   {q['text'][:44]:<46}" + "".join(f"{c:>24}" for c in cells))
    per = load(shop / "perturb.json")
    if per:
        print("   break-one per example, change of log-odds in nats (no rules | seven rules right), by question")
        for rk, name in (("0", "no rules"), ("+", "rules right")):
            s = per["summary"].get(f"net_refunds|rule{rk}")
            if not s:
                continue
            qs = sorted(next(iter(s["break_one"].values()))["per_question"])
            print(f"   [{name}] {'question':<22}" + "".join(f"{'ex' + str(i):>9}" for i in s["break_one"]))
            for q in qs:
                print(f"   {'':<{len(name) + 3}}{q:<22}" + "".join(f"{v['per_question'][q]:>+9.1f}" for v in s["break_one"].values()))
    if per:
        sh = shapley(per, "net_refunds")
        if sh:
            print("   Shapley value of each example for the log-odds, nats (no rules; a row sums to all clean - all broken)")
            print(f"   {'question':<22}" + "".join(f"{'ex' + str(i):>9}" for i in next(iter(sh.values()))))
            for q in sorted(sh):
                print(f"   {q:<22}" + "".join(f"{v:>+9.1f}" for v in sh[q].values()))
    if heads:
        print("   causal test, clean against example 4 broken (denoise at the decision position):")
        for r in heads["pairs"]:
            last = r["modes"]["last"]["denoise"]
            cum = {c["k"]: c["share"] for c in last["cumulative"]}
            top = ", ".join(f"L{a}H{b}" for a, b, _ in last["top"][:6])
            share = lambda v: pct(v / r["total"]) if abs(r["total"]) > 0.5 else "  n/a"  # noqa: E731
            print(f"   {r['question']:<18} gap {r['total']:+7.2f} nats | all heads at the decision token {share(last['all_heads'])} | "
                  f"top 1 / 8 / 32 heads jointly {pct(cum.get(1)) if abs(r['total']) > 0.5 else 'n/a'} / {pct(cum.get(8)) if abs(r['total']) > 0.5 else 'n/a'} / "
                  f"{pct(cum.get(32)) if abs(r['total']) > 0.5 else 'n/a'} | {top}")
    if att:
        print("   where the decision token looks (share of the attention row): selected heads | all heads and layers")
        for cell in att["cells"]:
            print(f"   {cell['question']} (heads ranked on {cell['selected_on']}):")
            for name, ctx in cell["contexts"].items():
                sel, allh = group_mass(ctx, "net_refunds", "selected_heads"), group_mass(ctx, "net_refunds", "all_heads_all_layers")
                ex = [f"ex{i}" for i in range(1, 6)]
                row = lambda m: " ".join(f"{g}:{m.get(g, 0):.2f}" for g in ex if g in m) + f" rule:{m.get('the rule', 0):.2f} rules:{m.get('other rules', 0):.2f} sql:{m.get('SQL so far', 0):.2f}"  # noqa: E731
                print(f"     {name:<26} log-odds {ctx['first_token_diff']:+6.1f} | {row(sel)} | {row(allh)}")


def claim2(shop: Path) -> None:
    per = load(shop / "perturb.json")
    if not per:
        return
    print("\n== 2. Redundancy: P(follows) by number of WRONG examples among the n that show the convention (no rules)")
    for key, s in per["summary"].items():
        conv, rk = key.split("|rule")
        if rk != "0":
            continue
        n = len(s["carriers"])
        dose = {n - int(k): v["p"] for k, v in s["dose"].items()}
        print(f"   {conv:<16} n={n} " + " ".join(f"{w} wrong:{pct(dose[w]).strip()}" for w in sorted(dose)))
        br = " ".join(f"ex{i}:{v['d_diff']:+.1f}" for i, v in s["break_one"].items())
        rs = " ".join(f"ex{i}:{v['d_diff']:+.1f}" for i, v in s["restore_one"].items())
        gap = s["all_clean"]["diff"] - s["all_broken"]["diff"]
        print(f"   {'':<16} all clean - all broken {gap:+.1f} nats | break one: {br} | restore one: {rs}")
        sh = shapley(per, conv)
        if sh:
            ids = list(next(iter(sh.values())))
            mean_phi = {i: mean([v[i] for v in sh.values()]) for i in ids}
            top_share = mean([max(v.values()) / sum(v.values()) for v in sh.values() if abs(sum(v.values())) > 1])
            print(f"   {'':<16} Shapley value per example (mean over questions): " + " ".join(f"ex{i}:{v:+.1f}" for i, v in mean_phi.items())
                  + f" | the largest example holds {100 * top_share:.0f}% of a question's gap on average")
        if s["near"].get("break_nearest", {}).get("n") and s["near"].get("break_other", {}).get("n"):
            nr = s["near"]
            print(f"   {'':<16} near questions: break nearest {nr['break_nearest']['mean']:+.1f} / another {nr['break_other']['mean']:+.1f}; "
                  f"restore nearest {nr['restore_nearest']['mean']:+.1f} / another {nr['restore_other']['mean']:+.1f}")
    print("   with the seven rules right (+) or this rule wrong (-): P(follows) by number of wrong examples")
    for key, s in per["summary"].items():
        conv, rk = key.split("|rule")
        if rk == "0":
            continue
        n = len(s["carriers"])
        dose = {n - int(k): v["p"] for k, v in s["dose"].items() if v["p"] is not None}
        print(f"   {conv:<16} rule {rk} " + " ".join(f"{w}:{pct(dose[w]).strip()}" for w in sorted(dose)) + f" | no examples {pct(s['no_examples']['p']).strip()}")


def claim3(shop: Path, status: Path) -> None:
    grid, sam = load(shop / "grid.json"), load(shop / "sample.json")
    if not grid:
        return
    print("\n== 3. Rule against examples: P(house convention), read-out (layout-sensitive decisions left out) [sampled]")
    layout = {(q, c) for q, per in grid["decisions"].items() for c, d in per.items() if d.get("whitespace_decision")}
    every, clean = defaultdict(list), defaultdict(list)
    for r in grid["rows"]:
        every[(r["convention"], r["condition"])].append(r["p_follow"])
        if (r["question"], r["convention"]) not in layout:
            clean[(r["convention"], r["condition"])].append(r["p_follow"])
    by = {k: clean.get(k) or v for k, v in every.items()}
    sby = defaultdict(list)
    for p in (sam or {}).get("pairs", []):
        if p["sampled_follow"] is not None:
            sby[(p["convention"], p["condition"])].append(p["sampled_follow"])
    print(f"   {'convention':<16}{'rule':<10}{'rule right, examples wrong':>30}{'rule wrong, examples right':>30}")
    for c in grid["meta"]["conventions"]:
        for kind, lab in (("rule", "7 rules"), ("one_line", "one-line")):
            a, b = f"conflict:{kind}_clean+examples_broken:{c}", f"conflict:{kind}_broken+examples_clean:{c}"
            if (c, a) not in by:
                continue
            cell = lambda k: f"{pct(mean(by[(c, k)]))} [{pct(mean(sby[(c, k)])).strip() if (c, k) in sby else 'n/a'}]"  # noqa: E731
            print(f"   {c:<16}{lab:<10}{cell(a):>30}{cell(b):>30}")
    for src, label in ((shop, "heads that carry the examples (clean against all examples broken)"), (status, "heads that carry the rule (right rule against wrong rule, no examples)")):
        att = load(src / "attention.json")
        if not att:
            continue
        print(f"   completed_only, where the status-value token looks, {label}:")
        for cell in att["cells"]:
            if cell["convention"] != "completed_only":
                continue
            print(f"   {cell['question']}:")
            for name in ("rule_only", "examples_clean", "rule_vs_examples", "rule_wrong_vs_examples", "one_line_vs_examples", "one_line_wrong_vs_examples"):
                ctx = cell["contexts"].get(name)
                if not ctx:
                    continue
                sel, allh = group_mass(ctx, "completed_only", "selected_heads"), group_mass(ctx, "completed_only", "all_heads_all_layers")
                ex = lambda m: sum(v for g, v in m.items() if g.startswith("ex"))  # noqa: E731
                print(f"     {name:<28} log-odds {ctx['first_token_diff']:+6.1f} | selected: the rule {sel.get('the rule', 0):.2f}, other rules {sel.get('other rules', 0):.2f}, "
                      f"examples {ex(sel):.2f} | all heads: the rule {allh.get('the rule', 0):.2f}, examples {ex(allh):.2f}")


def claim4(shop: Path) -> None:
    grid, sur, sam = load(shop / "grid.json"), load(shop / "surprisal.json"), load(shop / "sample.json")
    if not grid:
        return
    print("\n== 4. What one example carries: P(follows), read-out [sampled], under the one-line rule and under one example")
    layout = {(q, c) for q, per in grid["decisions"].items() for c, d in per.items() if d.get("whitespace_decision")}
    every, clean = defaultdict(list), defaultdict(list)
    for r in grid["rows"]:
        every[(r["convention"], r["condition"])].append(r["p_follow"])
        if (r["question"], r["convention"]) not in layout:
            clean[(r["convention"], r["condition"])].append(r["p_follow"])
    by = {k: clean.get(k) or v for k, v in every.items()}
    sby = defaultdict(list)
    for p in (sam or {}).get("pairs", []):
        if p["sampled_follow"] is not None:
            sby[(p["convention"], p["condition"])].append(p["sampled_follow"])
    conds = ["none", "rule_one_line", "rules_all", "ex1", "ex3", "ex5", "rules+examples"]
    print(f"   {'convention':<16}" + "".join(f"{c[:13]:>15}" for c in conds))
    for c in grid["meta"]["conventions"]:
        cells = [f"{pct(mean(by.get((c, k), [])))} [{pct(mean(sby[(c, k)])).strip() if (c, k) in sby else '-'}]" for k in conds]
        tag = " *" if c in grid["meta"]["in_one_line_rule"] else ""
        print(f"   {c + tag:<16}" + "".join(f"{v:>15}" for v in cells))
    print("   * stated by the one-line rule. Example 1 shows every convention but the two percentage ones.")
    if sur:
        s = sur["summary"]
        print("   bits of the reference SQL (mean over questions): " + ", ".join(f"{k} {s[k]['total_bits']:.0f}" for k in conds if k in s))
        print("   bits saved / context tokens / at the decision tokens: " + "; ".join(
            f"{k} {s[k]['saved_bits']:.0f} / {s[k]['context_tokens']:.0f} / {s[k]['saved_at_decisions']:.0f}" for k in conds[1:] if k in s))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the directory holding shop/, shop-aov/ and shop-status/")
    out = Path(ap.parse_args(argv).out)
    shop, aov, status = out / "shop", out / "shop-aov", out / "shop-status"
    for name in ("grid", "perturb", "sample"):
        d = load(shop / f"{name}.json")
        if d:
            m = d["meta"]
            v = m["scenario_version"]
            print(f"{m['model']} ({m['dtype']}), scenario {m['scenario']} (content {v.get('content', '?')}, files {v['hash']}), thinking off\n")
            break
    claim1(shop, aov)
    claim2(shop)
    claim3(shop, status)
    claim4(shop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
