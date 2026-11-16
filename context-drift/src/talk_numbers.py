#!/usr/bin/env python
"""Recompute every number shown on the slides of the talk from the published data, offline.

From the experiment directory:

    python src/talk_numbers.py              # one line per claim: as shown, as recomputed, source
    python src/talk_numbers.py --rescore    # also re-score every published reply with the checkers
    python src/talk_numbers.py --markdown   # the same table as Markdown

Reads results/runs/shop/ (behavioural records) and results/whitebox/shop/ (white-box JSON). No API
call, no model. Exits 1 if a recomputed number differs from the slide. Percentages are whole
numbers formatted as in the result tables (`.0f`: an exact half goes to the even number, so 1/40 =
2.5% shows as 2%, the one such case on the slides; a `caveat` line gives the exact fraction).
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from scenarios import load  # noqa: E402
from talkdemo import experiments as ex  # noqa: E402

RUNS = ROOT / "results" / "runs" / "shop"
WHITEBOX = ROOT / "results" / "whitebox" / "shop"
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"
METRIC = ("net_refunds", "completed_only", "exclude_test", "currency_units")


@dataclass
class Claim:
    slide: str
    claim: str
    shown: str
    got: str
    source: str

    @property
    def ok(self) -> bool:
        return self.shown == self.got

    @property
    def caveat(self) -> bool:
        """Context for a slide (what it implies, or the exact value behind a rounded one), not a
        number printed on it: reported, never counted as a mismatch."""
        return "caveat" in self.slide

    @property
    def flag(self) -> str:
        return "note" if self.caveat else ("ok" if self.ok else "DIFF")


def pct(k: float, n: float = 1.0) -> str:
    """Whole percent, formatted as the result tables format it (`.0f`)."""
    return f"{100 * k / n:.0f}%"


def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def load_json(path: Path) -> dict:
    gz = path.with_name(path.name + ".gz")
    return json.loads(path.read_text()) if path.exists() else json.loads(gzip.decompress(gz.read_bytes()))


# ---------------------------------------------------------------------------------------------
# behavioural (Claude)


def behavioural(records: list[dict], scenario) -> list[Claim]:
    out: list[Claim] = []
    by = defaultdict(list)
    for r in records:
        by[(r["runner"], r["experiment"], r["target"], r["condition"])].append(r)

    def perfect(rows):
        return sum(all(s == "follows" for s in r["status"].values()) for r in rows), len(rows)

    def wrong(runner, target, condition):
        rows = by[(runner, "e2", target, condition)]
        return sum(r["status"][target] == "violates" for r in rows), len(rows)

    src_e1 = "results/runs/shop/records/claude-sonnet-5-5/e1.jsonl.gz"
    e1 = {c: by[(SONNET, "e1", None, c)] for c in ("none", "rules", "ex5", "rule1", "ex1")}

    # slide 7 "Both approaches work"
    questions = sorted({r["question"] for r in e1["ex5"]})
    runs = Counter((r["question"], r["condition"]) for c in ("none", "rules", "ex5") for r in e1[c])
    out.append(Claim("7", "questions asked", "14", str(len(questions)), src_e1))
    out.append(Claim("7", "runs per question and context", "5", ",".join(map(str, sorted(set(runs.values())))), src_e1))
    for cond, shown in (("none", "0%"), ("rules", "100%"), ("ex5", "90%")):
        k, n = perfect(e1[cond])
        out.append(Claim("7", f"Sonnet, answers following every applicable convention, `{cond}`", shown, pct(k, n), src_e1))
    k, n = perfect(e1["ex5"])
    out.append(Claim("7 (notes)", "Sonnet `ex5`: correct answers of answers", "63/70", frac(k, n), src_e1))
    broken = Counter(
        tuple(c for c, s in r["status"].items() if s == "violates")
        for r in e1["ex5"]
        if any(s == "violates" for s in r["status"].values())
    )
    out.append(Claim("7", "Sonnet `ex5`: the only convention the failing answers break", "pct_round", ",".join("+".join(c) for c in broken), src_e1))
    rounding = defaultdict(list)
    for r in e1["ex5"]:
        if r["status"].get("pct_round") in ("follows", "violates"):
            rounding[r["question"]].append(r["status"]["pct_round"] == "follows")
    out.append(Claim("7", "questions that need the rounding", "4", str(len(rounding)), src_e1))
    flaky = [q for q, v in rounding.items() if 0 < sum(v) < len(v)]
    out.append(Claim("7", "of those, questions not answered consistently over the 5 runs", "3", str(len(flaky)), src_e1))
    order = [q.id for q in scenario.far_questions]
    per_q = ", ".join(f"Q{order.index(q) + 1} {sum(v)}/{len(v)}" for q, v in sorted(rounding.items(), key=lambda kv: order.index(kv[0])))
    out.append(Claim("7 (notes)", "rounded answers per question (questions 7, 8, 10, 13)", "Q7 2/5, Q8 2/5, Q10 4/5, Q13 5/5", per_q, src_e1))
    k, n = perfect(by[(HAIKU, "e1", None, "ex5")])
    out.append(Claim("7 (notes)", "Haiku, every applicable convention, `ex5`", "67%", pct(k, n), "results/runs/shop/records/claude-haiku-5-5/e1.jsonl.gz"))

    # slide 9 "Examples require more edits": the change made everywhere
    src_e2 = "results/runs/shop/records/claude-sonnet-5-5/e2.<convention>.jsonl.gz"
    for cond, label in (("rules:wrong", "the rule changed"), ("ex5:5", "all five examples changed")):
        pairs = [wrong(SONNET, t, cond) for t in METRIC]
        k, n = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        out.append(Claim("9", f"answers following the change, {label} (`{cond}`, 4 conventions)", "100%", pct(k, n), src_e2))
        out.append(Claim("9 (notes)", f"answers, {label}", "164", str(n), src_e2))
        rows = by[(SONNET, "e2", "completed_only", cond)]
        values = Counter(tuple(r["detail"].get("completed_only", {}).get("values") or ()) for r in rows)
        out.append(Claim("9", f"status answers that filter on exactly 'delivered', {label}", f"{len(rows)}/{len(rows)}", frac(values[("delivered",)], len(rows)), src_e2))

    # slide 10 "An example conveys implicit rules": q_best_days, one rule against one example
    for cond, shown in (("rule1", "2"), ("ex1", "5")):
        rows = [r for r in e1[cond] if r["question"] == "q_best_days"]
        applied = {sum(s == "follows" for s in r["status"].values()) for r in rows}
        out.append(Claim("10", f"'Which 5 days had the highest net revenue?', `{cond}`: conventions applied in every run", shown, ",".join(map(str, sorted(applied))), src_e1))
        layouts = {re.sub(r"\s+", " ", r["sql"]).strip() for r in rows}
        out.append(Claim("10", f"same answer in all 5 runs, `{cond}` (distinct SQL once whitespace is collapsed)", "1", str(len(layouts)), src_e1))
    out.append(Claim("10 (caveat)", "the one-line rule as given to the model (the slide shortens it)", "Net revenue = amount minus refunds, completed orders.", scenario.one_line_rule, "src/scenarios/shop.py"))

    # slide 11 "In practice, examples always overlap"
    per_conv = [sum(c.id in scenario.applies(scenario.example(e)) for e in scenario.default_examples) for c in scenario.conventions]
    out.append(Claim("11", "times each convention is written in the five examples", "5,5,5,5,2,1,1", ",".join(map(str, per_conv)), "src/scenarios/shop.py"))
    out.append(Claim("11", "7 conventions written N times", "24", str(sum(per_conv)), "src/scenarios/shop.py"))

    # slide 12 "One drifting copy is hard to detect": one random example of five changed
    rows = by[(SONNET, "e2", "completed_only", "ex5:1random")]
    k, n = wrong(SONNET, "completed_only", "ex5:1random")
    out.append(Claim("12", "status: answers that follow the drifted copy (`ex5:1random`)", "0 of 42", f"{k} of {n}", src_e2))
    values = Counter(tuple(r["detail"].get("completed_only", {}).get("values") or ()) for r in rows)
    out.append(Claim("12", "of those, answers still filtering on status = 'completed'", "42", str(values[("completed",)]), src_e2))
    which = Counter(r["corrupted_examples"][0] for r in rows)
    out.append(Claim("12 (caveat)", "which example was the drifted one (slide draws example 2)", "ex_country", ", ".join(f"{e} {n}" for e, n in sorted(which.items())), src_e2))
    pairs = [wrong(SONNET, t, "ex5:1random") for t in METRIC]
    out.append(Claim("12 (notes)", "four conventions pooled", "2 of 164", f"{sum(p[0] for p in pairs)} of {sum(p[1] for p in pairs)}", src_e2))

    # slide 16 "What happens if the examples disagree?"
    conds = ["ex5", "ex5:1random", "ex5:2", "ex5:3", "ex5:4", "ex5:5"]
    for target, label, shown in (
        ("completed_only", "status value", "0,0,5,95,100,100"),
        ("currency_units", "cents to euros", "0,0,2,28,88,100"),
        ("exclude_test", "test orders", "0,0,0,2,29,100"),
    ):
        got = ",".join(pct(*wrong(SONNET, target, c))[:-1] for c in conds)
        out.append(Claim("16", f"{label}: answers following the changed examples, 0 to 5 changed", shown, got, src_e2.replace("<convention>", target)))
    k, n = wrong(SONNET, "currency_units", "ex5:2")
    out.append(Claim("16 (caveat)", "cents to euros, 2 changed: the exact value behind the 2%", "2%", f"{frac(k, n)} = {100 * k / n:.1f}%", src_e2.replace("<convention>", "currency_units")))
    nearest = ",".join(pct(*wrong(SONNET, t, "ex5:1nearest"))[:-1] for t in ("completed_only", "currency_units", "exclude_test"))
    out.append(Claim("16 (caveat)", "k = 1 with the nearest example changed instead of a random one (status, cents, test)", "0,0,0", nearest, src_e2))
    return out


# ---------------------------------------------------------------------------------------------
# white-box (Gemma 4 12B)


def whitebox() -> list[Claim]:
    out: list[Claim] = []
    att = load_json(WHITEBOX / "attention.json")
    meta = att["meta"]
    cell = next(c for c in att["cells"] if c["convention"] == "net_refunds" and c["question"] == "q_year")
    src_att = "results/whitebox/shop/attention.json.gz"
    out.append(Claim("13", "model", "google/gemma-4-12B-it", meta["model"], src_att))
    out.append(Claim("13", "attention heads selected causally", "12", str(len(cell["selected_heads"])), src_att))
    out.append(Claim("13 (notes)", "attention heads in the model", "768", str(meta["n_layers"] * meta["n_heads"]), src_att))
    clean = cell["contexts"]["examples_clean"]
    out.append(Claim("13", "the SQL so far at the decision token", "SELECT SUM(amount", "".join(t for t, l in zip(clean["tokens"], clean["labels"]) if l == "output").strip(), src_att))

    def shares(ctx: dict) -> list[float]:
        tm, labs = ctx["token_maps"]["selected_heads"], ctx["labels"]
        return [sum(v for v, l in zip(tm, labs) if l.startswith(f"ex{n}.")) for n in range(1, 6)]

    out.append(Claim("14", "attention share per example, five correct examples", "32,14,4,8,8", ",".join(pct(s)[:-1] for s in shares(clean)), src_att))
    changed = cell["contexts"]["one_example_broken"]
    out.append(Claim("14", "the changed example", "3", str(cell["broken_example"]), src_att))
    out.append(Claim("14", "attention share per example, example 3 changed", "30,14,5,10,9", ",".join(pct(s)[:-1] for s in shares(changed)), src_att))
    toks, labs, tm = clean["tokens"], clean["labels"], clean["token_maps"]["selected_heads"]
    minus = [i for i in range(len(toks)) if labs[i].endswith(".sql") and toks[i].strip() == "-" and toks[i - 1].endswith("amount")]
    out.append(Claim("14", "minus signs right after SUM(amount in the examples", "5", str(len(minus)), src_att))
    out.append(Claim("14", "attention share of those five minus signs", "27%", pct(sum(tm[i] for i in minus)), src_att))
    out.append(Claim("14 (notes)", "tokens in the prompt at the decision token", "622", str(clean["n_tokens"]), src_att))
    out.append(Claim("14 (notes)", "attention share of the five examples together", "67%", pct(sum(shares(clean))), src_att))
    out.append(Claim("14", "the model still subtracts the refunds with example 3 changed (log-odds > 0)", "yes", "yes" if changed["first_token_diff"] > 0 else "no", src_att))

    per = load_json(WHITEBOX / "perturb.json")
    rows = [r for r in per["rows"] if r["convention"] == "net_refunds" and r["rule"] == "0" and r["examples"]]
    src_per = "results/whitebox/shop/perturb.json.gz"
    dose = []
    for changed_n in range(6):
        sel = [r["p_follow"] for r in rows if len(r["clean"]) == 5 - changed_n]
        dose.append(pct(1 - sum(sel) / len(sel))[:-1])
    n_q = len({r["question"] for r in rows})
    out.append(Claim("15", f"P(next token follows the changed examples), 0 to 5 changed (mean over {n_q} questions and every subset)", "0,5,12,26,72,100", ",".join(dose), src_per))
    one = [r for r in rows if r["question"] == "q_year"]
    single = []
    for changed_n in range(6):
        sel = [r["p_follow"] for r in one if len(r["clean"]) == 5 - changed_n]
        single.append(pct(1 - sum(sel) / len(sel))[:-1])
    out.append(Claim("15 (caveat)", "the same on 'What was our net revenue in 2025?' alone (prefix SELECT SUM(amount)", "0,5,12,26,72,100", ",".join(single), src_per))
    return out


def rescore(records: list[dict], scenario) -> tuple[int, int]:
    """Re-score every stored reply with the shipped checkers; count disagreements with the stored status."""
    bad = 0
    for r in records:
        checks = scenario.score(scenario.question(r["question"]), r["reply"])
        if {c: k.status for c, k in checks.items()} != r["status"]:
            bad += 1
    return bad, len(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rescore", action="store_true", help="re-score every published reply (about a minute)")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = ap.parse_args(argv)
    scenario = load("shop")
    records = ex.load_records(RUNS)
    claims = behavioural(records, scenario) + whitebox()
    if args.markdown:
        print("| slide | claim | on the slide | recomputed | source |\n|---|---|---|---|---|")
        for c in claims:
            print(f"| {c.slide} | {c.claim} | {c.shown} | {c.got}{' **differs**' if c.flag == 'DIFF' else ''} | `{c.source}` |")
    else:
        for c in claims:
            print(f"{c.flag:<4} slide {c.slide:<11} {c.claim}\n     slide: {c.shown}\n     data:  {c.got}   ({c.source})")
    caveats = [c for c in claims if c.caveat]
    diffs = [c for c in claims if c.flag == "DIFF"]
    print(f"\n{len(claims)} checks: {len(claims) - len(caveats) - len(diffs)} match, {len(diffs)} differ; {len(caveats)} notes (context for a slide, not a number printed on it)")
    if args.rescore:
        bad, n = rescore(records, scenario)
        print(f"re-scored {n} published replies with the shipped checkers: {bad} disagree with the stored status")
        if bad:
            return 1
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
