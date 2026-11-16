"""Behavioural experiments (E1, E2, E3) on Claude: build prompts, run, score, report.

From the experiment directory:

    python src/run.py prompt ex5 q_hour                       # print one prompt (offline)
    python src/run.py report                                  # tables from results/runs/shop (offline)
    python src/run.py show --runner claude-sonnet-5-5 --exp e2 --condition ex5:4 --limit 2
    python src/run.py final --model claude-sonnet-5-5         # the reported runs (API key, writes out/runs)
    python src/run.py --results out/runs report               # tables from your own runs

Published records are read from results/runs/<scenario>/; new runs are written to
out/runs/<scenario>/ unless --results says otherwise, so they never overwrite the published files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent  # the experiment directory
sys.path.insert(0, str(SRC))

from scenarios import NAMES, load  # noqa: E402
from talkdemo import experiments as ex  # noqa: E402
from talkdemo.prompt import ContextSpec, build_prompt  # noqa: E402
from talkdemo.runners import CACHE_DIR, PRICES, ClaudeRunner, Ledger  # noqa: E402

PUBLISHED = ROOT / "results" / "runs"  # the published records (read by report / show)
FRESH = ROOT / "out" / "runs"  # where run / final / tokens / prompts write by default (git-ignored)
REPORT = ROOT / "out" / "report"  # where report writes its tables (git-ignored)
WRITERS = ("run", "final", "tokens", "prompts")
E1_ORDER = ["none", "rule1", "rules", "ex1", "ex3", "ex5", "rule1+ex5", "rules+ex5"]
E2_ORDER = [
    ("rules", "all rules, correct"),
    ("rules:wrong", "all rules, this one stated wrongly"),
    ("rule1", "one-line rule, correct"),
    ("rule1:wrong", "one-line rule, stated wrongly"),
    ("ex5", "5 examples, all correct"),
    ("ex5:1random", "1 of 5 wrong (random one)"),
    ("ex5:1nearest", "1 of 5 wrong (the nearest one)"),
    ("ex5:2", "2 of 5 wrong"),
    ("ex5:3", "3 of 5 wrong"),
    ("ex5:4", "4 of 5 wrong (one correct left)"),
    ("ex5:5", "5 of 5 wrong"),
    ("rules+ex5", "rules + 5 examples, all correct"),
    ("rules+ex5:1nearest", "rules correct, nearest example wrong"),
    ("rules+ex5:5", "rules correct, all 5 examples wrong"),
    ("rules:wrong+ex5", "rule wrong, all 5 examples correct"),
    ("rule1+ex5", "one-line rule + 5 examples, all correct"),
    ("rule1+ex5:5", "one-line rule correct, all 5 examples wrong"),
    ("rule1:wrong+ex5", "one-line rule wrong, all 5 examples correct"),
]
E3_ORDER = [
    ("ex5", "5 examples, all correct"),
    ("ex5:1nearest", "the nearest example wrong"),
    ("ex5:1far", "another example wrong"),
]


def _results(args) -> Path:
    """The records of a scenario: --results/<scenario>/ if given, else the published files for the
    commands that read and out/runs/<scenario>/ for the commands that call the API."""
    if args.results:
        return Path(args.results) / args.scenario
    return (FRESH if args.command in WRITERS else PUBLISHED) / args.scenario


def _report_dir(args) -> Path:
    return Path(args.report_dir or REPORT) / args.scenario


def _conventions(scenario, arg: str | None) -> list[str]:
    """Targets of E2 / E3: the ones asked for, else every convention all five examples show."""
    return [c.strip() for c in arg.split(",")] if arg else ex.corruptible(scenario)


def _split(arg: str | None) -> list[str] | None:
    return [x.strip() for x in arg.split(",")] if arg else None


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------


def cmd_export(args) -> None:
    scenario = load(args.scenario)
    out = SRC / "scenarios" / f"{scenario.id}.json"
    out.write_text(json.dumps(scenario.to_json(), indent=2, ensure_ascii=False) + "\n")
    dps = [
        {
            "question_id": dp.question_id,
            "question": scenario.question(dp.question_id).question,
            "convention_id": dp.convention_id,
            "prefix": dp.prefix,
            "follows": dp.follows,
            "violates": dp.violates,
            "follows_head": dp.follows_head,
            "violates_head": dp.violates_head,
        }
        for dp in scenario.decision_points()
    ]
    dp_path = SRC / "scenarios" / f"{scenario.id}.decision_points.jsonl"
    _dump(dp_path, dps)
    print(f"{out}\n{dp_path} ({len(dps)} decision points)")


def _find_condition(scenario, experiment: str, name: str, conv: str):
    conds = {
        "e1": ex.e1_conditions,
        "e2": lambda: ex.e2_conditions(scenario, conv),
        "e3": lambda: ex.e3_conditions(conv),
    }[experiment]()
    if name not in conds:
        raise SystemExit(f"unknown condition {name!r} for {experiment}; choose from {list(conds)}")
    return conds[name]


def cmd_prompt(args) -> None:
    scenario = load(args.scenario)
    experiment = args.exp or ("e1" if args.condition in ex.e1_conditions() else "e2")
    make = _find_condition(scenario, experiment, args.condition, args.conv)
    question = scenario.question(args.question)
    spec = make(scenario, question, args.sample)
    if args.canonical:
        by_id = {u.id: u for u in spec.examples}
        spec = ContextSpec(
            spec.rules, spec.wrong_rules, tuple(by_id[e] for e in scenario.default_examples if e in by_id)
        )
    prompt = build_prompt(scenario, spec, question)
    print(f"--- system ---\n{prompt.system}\n--- user ---\n{prompt.user}\n--- reference ---\n{question.sql}")


def cmd_prompts(args) -> None:
    """The exact prompts of an experiment plan, with the decision points of each question."""
    scenario = load(args.scenario)
    jobs = ex.plan(scenario, args.experiment, args.samples, _conventions(scenario, args.conv), _split(args.conditions))
    rows = []
    for job in jobs:
        conventions = [job.target] if job.target else list(scenario.applies(job.question))
        rows.append(
            {
                "experiment": job.experiment,
                "condition": job.condition,
                "target": job.target,
                "question_id": job.question.id,
                "sample": job.sample,
                "spec": job.spec.to_json(),
                "system": job.prompt.system,
                "user": job.prompt.user,
                "decision_points": {
                    cid: {
                        k: getattr(scenario.decision_point(job.question, cid), k)
                        for k in ("prefix", "follows", "violates", "follows_head", "violates_head")
                    }
                    for cid in conventions
                },
            }
        )
    path = _results(args) / "prompts" / f"{args.experiment}.jsonl"
    _dump(path, rows)
    print(f"{path} ({len(rows)} prompts)")


def _runner(args) -> ClaudeRunner:
    return ClaudeRunner(args.model, thinking=args.thinking, budget_usd=args.budget)


def cmd_tokens(args) -> None:
    """Context tokens of each E1 condition: prompt tokens minus the no-context prompt, mean over
    the far questions (sample 0)."""
    scenario = load(args.scenario)
    runner = _runner(args)
    conds = ex.e1_conditions()
    out, full = {}, {}
    for name, make in conds.items():
        deltas, totals = [], []
        for q in scenario.far_questions:
            base = runner.count_tokens(build_prompt(scenario, ContextSpec(), q))
            total = runner.count_tokens(build_prompt(scenario, make(scenario, q, 0), q))
            deltas.append(total - base)
            totals.append(total)
        out[name] = round(sum(deltas) / len(deltas), 1)
        full[name] = round(sum(totals) / len(totals), 1)
    path = _results(args) / "context_tokens.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[runner.name] = {"context_tokens": out, "prompt_tokens": full}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    for name in conds:
        print(f"{name:12s} context {out[name]:7.1f}   whole prompt {full[name]:7.1f}")


def cmd_run(args) -> None:
    scenario = load(args.scenario)
    jobs = ex.plan(
        scenario,
        args.experiment,
        args.samples,
        _conventions(scenario, args.conv),
        _split(args.conditions),
        _split(args.questions),
    )
    prompts = {(j.prompt.system, j.prompt.user) for j in jobs}
    print(f"{args.experiment}: {len(jobs)} jobs, {len(prompts)} distinct prompts, model {args.model}")
    if args.dry_run:
        return
    runner = _runner(args)
    before = runner.ledger.spent
    try:
        records = ex.run(scenario, runner, jobs, workers=args.workers)
    finally:
        print(f"spent this run ${runner.ledger.spent - before:.3f}, total ${runner.ledger.spent:.3f}")
    for path in ex.save_records(_results(args), records):
        print(path)
    _print_quick(scenario, records)


# Samples per (condition, question) of the reported runs. E2 samples are set per target so every
# cell holds about 40 outputs on Sonnet and 60 to 70 on Haiku, whatever the number of questions
# the convention applies to (net revenue 10, status and test orders 14, cents 8).
PROTOCOL = {
    "claude-haiku-5-5": {
        "e1": 10,
        "e2": {"net_refunds": 6, "completed_only": 5, "exclude_test": 5, "currency_units": 8},
        "e3": 12,
    },
    "claude-sonnet-5-5": {
        "e1": 5,
        "e2": {"net_refunds": 4, "completed_only": 3, "exclude_test": 3, "currency_units": 5},
        "e3": 8,
    },
}


def cmd_final(args) -> None:
    """The reported runs of one model: E1, then E2 and E3 for the four metric conventions."""
    scenario = load(args.scenario)
    plan = PROTOCOL[args.model]
    runner = _runner(args)
    steps: list[tuple[str, int, list[str]]] = [("e1", plan["e1"], [])]
    steps += [("e2", samples, [conv]) for conv, samples in plan["e2"].items()]
    steps += [("e3", plan["e3"], list(plan["e2"]))]
    for experiment, samples, convs in steps:
        jobs = ex.plan(scenario, experiment, samples, convs)
        print(f"{experiment} {','.join(convs)}: {len(jobs)} jobs, {samples} samples per (condition, question)")
        before = runner.ledger.spent
        try:
            records = ex.run(scenario, runner, jobs, workers=args.workers)
        finally:
            print(f"  spent ${runner.ledger.spent - before:.3f}, total ${runner.ledger.spent:.3f}")
        ex.save_records(_results(args), records)
        bad = sum(r["meta"].get("stop_reason") not in (None, "end_turn") for r in records)
        unparsed = sum("unparsed" in r["status"].values() for r in records)
        print(f"  replies {len(records)}/{len(jobs)}, unparsed {unparsed}, stop_reason other than end_turn {bad}")


def _print_quick(scenario, records: list[dict]) -> None:
    summary = ex.summarise(scenario, records)
    for row in summary["e1"]:
        if row["convention"] in (*scenario.convention_ids, "every_applicable_convention"):
            print(f"  {row['condition']:10s} {row['convention']:28s} {_pct(row)}")
    for row in summary["e2"] + summary["e3"]:
        print(f"  {row['target']:16s} {row['condition']:20s} wrong {_pct(row)}")
    bad = sum(r["meta"].get("stop_reason") not in (None, "end_turn") for r in records)
    unparsed = sum("unparsed" in r["status"].values() for r in records)
    print(f"  replies: {len(records)}, unparsed: {unparsed}, stop_reason other than end_turn: {bad}")


def _pct(row: dict) -> str:
    if row["rate"] is None:
        return "   -  (n=0)"
    extra = f", {row['not_scored']} not scored" if row["not_scored"] else ""
    return f"{100 * row['rate']:5.1f}% (n={row['n']}{extra})"


def cmd_report(args) -> None:
    scenario = load(args.scenario)
    records = ex.load_records(_results(args))
    tokens_path = _results(args) / "context_tokens.json"
    tokens = json.loads(tokens_path.read_text()) if tokens_path.exists() else {}
    summary = ex.summarise(scenario, records, {r: v["context_tokens"] for r, v in tokens.items()})
    summary["prompt_tokens"] = {r: v["prompt_tokens"] for r, v in tokens.items()}
    summary["replies"] = _reply_counts(records)
    out = _report_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("e1", "e1_by_family", "e1_efficiency", "e2", "e3", "by_question", "one_wrong_example"):
        rows = summary.get(name) or []
        if not rows:
            continue
        cols = list(dict.fromkeys(k for row in rows for k in row))
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    md = _markdown(scenario, summary)
    (out / "tables.md").write_text(md)
    print(md)
    print(f"wrote {out}/summary.json, the CSVs and tables.md")


def _reply_counts(records: list[dict]) -> list[dict]:
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by[(r["runner"], r["experiment"])].append(r)
    return [
        {
            "runner": runner,
            "experiment": experiment,
            "replies": len(rows),
            "unparsed": sum("unparsed" in r["status"].values() for r in rows),
            "not_end_turn": sum(r["meta"].get("stop_reason") not in (None, "end_turn") for r in rows),
        }
        for (runner, experiment), rows in sorted(by.items())
    ]


def _markdown(scenario, summary: dict) -> str:
    out: list[str] = []
    runners = sorted({row["runner"] for key in ("e1", "e2", "e3") for row in summary[key]})

    def follow(row):
        return "-" if row is None or row["rate"] is None else f"{100 * row['rate']:.0f}% ({row['follows']}/{row['n']})"

    def wrong(row):
        return "-" if row is None or row["rate"] is None else f"{100 * row['rate']:.0f}% ({row['violates']}/{row['n']})"

    for runner in runners:
        e1 = {(r["condition"], r["convention"]): r for r in summary["e1"] if r["runner"] == runner}
        if e1:
            conds = [c for c in E1_ORDER if any(k[0] == c for k in e1)]
            out += [f"### E1, {runner}: share of outputs that follow each convention", ""]
            out += ["| convention | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
            labels = [
                *[
                    (c.id, c.id + (" (in the one-line rule)" if c.in_one_line_rule else ""))
                    for c in scenario.conventions
                ],
                ("stated_by_one_line_rule", "**stated by the one-line rule (pooled)**"),
                ("not_stated_by_one_line_rule", "**not stated by the one-line rule (pooled)**"),
                ("every_applicable_convention", "**outputs following every applicable convention**"),
            ]
            for key, label in labels:
                out.append(f"| {label} | " + " | ".join(follow(e1.get((c, key))) for c in conds) + " |")
            eff = {r["condition"]: r for r in summary.get("e1_efficiency", []) if r["runner"] == runner}
            if eff:
                n_conv = len(scenario.conventions)
                out.append(
                    f"| conventions followed (sum of the rates, of {n_conv}) | "
                    + " | ".join(f"{eff[c]['conventions_followed']:.2f}" for c in conds)
                    + " |"
                )
                if any(eff[c].get("context_tokens") for c in conds):
                    out.append(
                        "| context tokens (beyond schema and question) | "
                        + " | ".join(f"{eff[c].get('context_tokens') or 0:.0f}" for c in conds)
                        + " |"
                    )
                    out.append(
                        "| conventions gained per 100 context tokens | "
                        + " | ".join(
                            f"{eff[c]['gain_per_100_tokens']:.2f}" if "gain_per_100_tokens" in eff[c] else "-"
                            for c in conds
                        )
                        + " |"
                    )
            out.append("")
            fam = {
                (r["condition"], r["family"], r["convention"]): r
                for r in summary["e1_by_family"]
                if r["runner"] == runner
            }
            shared = [cid for cid in scenario.convention_ids if any(k[1] == "other" and k[2] == cid for k in fam)]
            out += [f"### E1, {runner}: the same, split by what the question asks for", ""]
            out += ["| convention | questions | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
            for cid in shared:
                for family, label in (("metric", "net revenue figures"), ("other", "other figures")):
                    cells = " | ".join(follow(fam.get((c, family, cid))) for c in conds)
                    out.append(f"| {cid} | {label} | {cells} |")
            out.append("")
        for key, order, title in (
            ("e2", E2_ORDER, "E2, {runner}: share of outputs that break the targeted convention (far questions)"),
            ("e3", E3_ORDER, "E3, {runner}: share of outputs that break the targeted convention (near questions)"),
        ):
            rows = {(r["condition"], r["target"]): r for r in summary[key] if r["runner"] == runner}
            targets = [cid for cid in scenario.convention_ids if any(k[1] == cid for k in rows)]
            if not targets:
                continue
            out += [f"### {title.format(runner=runner)}", ""]
            out += ["| context | condition | " + " | ".join(targets) + " |", "|---|---|" + "---|" * len(targets)]
            for cond, label in order:
                if any((cond, t) in rows for t in targets):
                    out.append(
                        f"| {label} | `{cond}` | " + " | ".join(wrong(rows.get((cond, t))) for t in targets) + " |"
                    )
            out.append("")
        one = [r for r in summary["one_wrong_example"] if r["runner"] == runner and r["split"] != "which+position"]
        if one:
            targets = [cid for cid in scenario.convention_ids if any(r["target"] == cid for r in one)]
            cell = {(r["questions"], r["split"], r["corrupted_example"], r["target"]): r for r in one}
            out += [f"### One wrong example of five, {runner}: which one, and where in the list", ""]
            out += [
                "| questions | the wrong example is | " + " | ".join(targets) + " |",
                "|---|---|" + "---|" * len(targets),
            ]
            for kind in ("far", "near"):
                for split, value in (
                    ("which", "nearest"),
                    ("which", "other"),
                    ("position", "first"),
                    ("position", "middle"),
                    ("position", "last"),
                ):
                    if any((kind, split, value, t) in cell for t in targets):
                        label = (
                            ("the nearest one" if value == "nearest" else "another one")
                            if split == "which"
                            else f"shown {value}"
                        )
                        cells = " | ".join(wrong(cell.get((kind, split, value, t))) for t in targets)
                        out.append(f"| {kind} | {label} | {cells} |")
            out.append("")
        # the nearest example wrong, by which example that is (far and near questions pooled)
        for cond, label in (("ex5:1nearest", "examples only"), ("rules+ex5:1nearest", "with all rules, correct")):
            rows = [r for r in summary["by_question"] if r["runner"] == runner and r["condition"] == cond]
            if not rows:
                continue
            targets = [cid for cid in scenario.convention_ids if any(r["target"] == cid for r in rows)]
            pooled: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
            for r in rows:
                pooled[(r["nearest_example"], r["target"])][0] += r["violates"]
                pooled[(r["nearest_example"], r["target"])][1] += r["n"]
            out += [f"### The nearest example wrong, {runner}, {label}: by the example that was changed", ""]
            out += ["| changed example | " + " | ".join(targets) + " |", "|---|" + "---|" * len(targets)]
            for eid in scenario.default_examples:
                cells = []
                for t in targets:
                    bad, n = pooled.get((eid, t), (0, 0))
                    cells.append(f"{100 * bad / n:.0f}% ({bad}/{n})" if n else "-")
                out.append(f"| {eid}: {scenario.example(eid).question} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out) + "\n"


def cmd_show(args) -> None:
    """Print stored replies, filtered."""
    want = dict(pair.split("=") for pair in args.status or [])
    shown = 0
    for r in ex.load_records(_results(args)):
        if args.runner and r["runner"] != args.runner:
            continue
        if args.exp and r["experiment"] != args.exp:
            continue
        if args.condition and r["condition"] != args.condition:
            continue
        if args.question and r["question"] != args.question:
            continue
        if args.target and r["target"] != args.target:
            continue
        if any(r["status"].get(cid) != status for cid, status in want.items()):
            continue
        shown += 1
        if shown > args.limit:
            continue
        head = f"{r['runner']} {r['experiment']} {r['condition']} target={r['target']} {r['question']} sample={r['sample']}"
        print(f"=== {head}\n    status {r['status']}\n    detail {r['detail']}")
        if r["spec"]["examples"]:
            print("    examples " + ", ".join(e["id"] + ("*" if e["corrupt"] else "") for e in r["spec"]["examples"]))
        print(r["reply"] + "\n")
    print(f"{shown} matching record(s)")


def cmd_spend(args) -> None:
    ledger = Ledger(CACHE_DIR / "spend.jsonl", budget_usd=args.budget)
    by: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0, 0.0])
    for row in ledger.rows():
        acc = by[row["model"]]
        acc[0] += 1
        acc[1] += row["in"]
        acc[2] += row["out"]
        acc[3] += row["usd"]
    for model, (calls, tin, tout, usd) in sorted(by.items()):
        print(f"{model:20s} {calls:6d} calls  {tin:9d} in  {tout:8d} out  ${usd:.3f}")
    print(f"total ${ledger.spent:.3f} of ${args.budget:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="shop", choices=list(NAMES))
    p.add_argument("--results", help="records directory (default: results/runs to read, out/runs to write)")
    p.add_argument("--report-dir", help="where report writes (default out/report)")
    p.add_argument("--budget", type=float, default=14.0, help="hard stop on cumulative API spend, USD")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("export", help="write src/scenarios/<name>.json and the decision points").set_defaults(fn=cmd_export)

    s = sub.add_parser("prompt", help="print one prompt")
    s.add_argument("condition")
    s.add_argument("question")
    s.add_argument("--exp", choices=["e1", "e2", "e3"])
    s.add_argument("--conv", default="net_refunds")
    s.add_argument("--sample", type=int, default=0)
    s.add_argument("--canonical", action="store_true", help="examples in canonical order, not the sample's shuffle")
    s.set_defaults(fn=cmd_prompt)

    for name, fn, text in (
        ("run", cmd_run, "run an experiment"),
        ("prompts", cmd_prompts, "dump an experiment's prompts"),
    ):
        s = sub.add_parser(name, help=text)
        s.add_argument("experiment", choices=["e1", "e2", "e3"])
        s.add_argument("--samples", type=int, default=5, help="samples per (condition, question)")
        s.add_argument("--conv", help="target conventions for e2 / e3, comma separated")
        s.add_argument("--conditions", help="only these conditions, comma separated")
        if name == "run":
            s.add_argument("--questions", help="only these questions, comma separated")
            s.add_argument("--model", required=True, choices=sorted(PRICES))
            s.add_argument("--thinking", default="off", choices=["off", "default"])
            s.add_argument("--workers", type=int, default=8)
            s.add_argument("--dry-run", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("final", help="the reported runs of one model (sample counts in PROTOCOL)")
    s.add_argument("--model", required=True, choices=sorted(PROTOCOL))
    s.add_argument("--thinking", default="off", choices=["off", "default"])
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_final)

    s = sub.add_parser("tokens", help="context tokens per E1 condition (free count_tokens endpoint)")
    s.add_argument("--model", required=True, choices=sorted(PRICES))
    s.add_argument("--thinking", default="off", choices=["off", "default"])
    s.set_defaults(fn=cmd_tokens)

    s = sub.add_parser("show", help="print stored replies")
    s.add_argument("--runner")
    s.add_argument("--exp")
    s.add_argument("--condition")
    s.add_argument("--question")
    s.add_argument("--target")
    s.add_argument("--status", action="append", help="convention=status, e.g. net_refunds=violates (repeatable)")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(fn=cmd_show)

    sub.add_parser("report", help="aggregate the records into tables (out/report/<scenario>/)").set_defaults(fn=cmd_report)
    sub.add_parser("spend", help="API spend so far").set_defaults(fn=cmd_spend)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
