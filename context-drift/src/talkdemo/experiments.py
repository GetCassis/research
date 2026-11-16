"""The three experiments: conditions, jobs, running, scoring, aggregation.

E1  what transfers     no context / one-line rule / all rules / 1, 3, 5 examples / rules + examples,
                       far questions, every convention scored.
E2  change one place   for one target convention: the rule stated wrongly; k of the 5 examples
                       corrupted (one random, the nearest, 2, 3, 4 = "restore one", 5); rule
                       against examples in both directions. Far questions the convention applies to.
E3  nearest example    near questions: corrupt the example they resemble, or another one.

A condition is a function (scenario, question, sample) -> ContextSpec. Example order and the
choice of corrupted examples are seeded by (question, sample) only, so conditions are paired.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import sqlcheck
from .prompt import ContextSpec, Prompt, build_prompt, examples_spec, rules_spec
from .runners import BudgetExceeded, Completion, Runner, TransientError
from .scenario import FOLLOWS, VIOLATES, Item, Scenario

Make = Callable[[Scenario, Item, int], ContextSpec]


def _fixed(spec: ContextSpec) -> Make:
    return lambda scenario, question, sample: spec


def _examples(**kw) -> Make:
    return lambda scenario, question, sample: examples_spec(scenario, question, sample, **kw)


def e1_conditions() -> dict[str, Make]:
    return {
        "none": _fixed(ContextSpec()),
        "rule1": _fixed(rules_spec("one_line")),
        "rules": _fixed(rules_spec("all")),
        "ex1": _examples(n=1),
        "ex3": _examples(n=3),
        "ex5": _examples(n=5),
        "rule1+ex5": _examples(n=5, rules="one_line"),
        "rules+ex5": _examples(n=5, rules="all"),
    }


def e2_conditions(scenario: Scenario, conv: str) -> dict[str, Make]:
    out: dict[str, Make] = {
        # baselines (same prompts as E1, so they come from the cache)
        "rules": _fixed(rules_spec("all")),
        "ex5": _examples(n=5),
        "rules+ex5": _examples(n=5, rules="all"),
        # (a) rules only, the rule states the wrong computation
        "rules:wrong": _fixed(rules_spec("all", [conv])),
        # (b) examples only, k of 5 corrupted
        "ex5:1random": _examples(n=5, convention=conv, k=1),
        "ex5:1nearest": _examples(n=5, convention=conv, k=1, pick="nearest"),
        "ex5:2": _examples(n=5, convention=conv, k=2),
        "ex5:3": _examples(n=5, convention=conv, k=3),
        "ex5:4": _examples(n=5, convention=conv, k=4),
        "ex5:5": _examples(n=5, convention=conv, k=5),
        # (c) rule against examples
        "rules+ex5:5": _examples(n=5, rules="all", convention=conv, k=5),  # rule right, examples wrong
        "rules:wrong+ex5": _examples(n=5, rules="all", wrong_rules=[conv]),  # rule wrong, examples right
        "rules+ex5:1nearest": _examples(n=5, rules="all", convention=conv, k=1, pick="nearest"),
    }
    if conv in scenario.one_line_wrong:
        out["rule1"] = _fixed(rules_spec("one_line"))
        out["rule1:wrong"] = _fixed(rules_spec("one_line", [conv]))
        # the one-line rule is prose ("for completed orders"), the full list names values:
        # the same conflicts with the vaguer rule
        out["rule1+ex5"] = _examples(n=5, rules="one_line")
        out["rule1+ex5:5"] = _examples(n=5, rules="one_line", convention=conv, k=5)
        out["rule1:wrong+ex5"] = _examples(n=5, rules="one_line", wrong_rules=[conv])
    return out


def e3_conditions(conv: str) -> dict[str, Make]:
    return {
        "ex5": _examples(n=5),
        "ex5:1nearest": _examples(n=5, convention=conv, k=1, pick="nearest"),
        "ex5:1far": _examples(n=5, convention=conv, k=1, pick="far"),
    }


# conventions every one of the standard examples shows: the only ones "k of 5" can corrupt
def corruptible(scenario: Scenario) -> list[str]:
    shown = [set(scenario.applies(scenario.example(e))) for e in scenario.default_examples]
    return [cid for cid in scenario.convention_ids if all(cid in s for s in shown)]


@dataclass(frozen=True)
class Job:
    experiment: str
    condition: str
    target: str | None
    question: Item
    sample: int
    spec: ContextSpec
    prompt: Prompt


def plan(
    scenario: Scenario,
    experiment: str,
    samples: int,
    conventions: Iterable[str] = (),
    conditions: Iterable[str] | None = None,
    questions: Iterable[str] | None = None,
) -> list[Job]:
    """Every (condition, question, sample) of one experiment."""
    keep_c = set(conditions) if conditions else None
    keep_q = set(questions) if questions else None
    cells: list[tuple[str | None, dict[str, Make], tuple[Item, ...]]] = []
    if experiment == "e1":
        cells.append((None, e1_conditions(), scenario.far_questions))
    elif experiment in ("e2", "e3"):
        for conv in conventions:
            if conv not in corruptible(scenario):
                raise ValueError(f"{conv} is not shown by all standard examples; choose from {corruptible(scenario)}")
            pool = scenario.far_questions if experiment == "e2" else scenario.near_questions
            qs = tuple(q for q in pool if conv in scenario.applies(q))
            cells.append((conv, e2_conditions(scenario, conv) if experiment == "e2" else e3_conditions(conv), qs))
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    jobs = []
    for target, conds, qs in cells:
        for name, make in conds.items():
            if keep_c is not None and name not in keep_c:
                continue
            for q in qs:
                if keep_q is not None and q.id not in keep_q:
                    continue
                for sample in range(samples):
                    spec = make(scenario, q, sample)
                    jobs.append(Job(experiment, name, target, q, sample, spec, build_prompt(scenario, spec, q)))
    return jobs


def _record(scenario: Scenario, runner: str, job: Job, completion: Completion, replicate: int) -> dict:
    checks = scenario.score(job.question, completion.text)
    return {
        "scenario": scenario.id,
        "runner": runner,
        "experiment": job.experiment,
        "condition": job.condition,
        "target": job.target,
        "question": job.question.id,
        "kind": "near" if job.question.near else "far",
        "sample": job.sample,
        "replicate": replicate,
        "spec": job.spec.to_json(),
        "nearest_example": scenario.nearest_example(job.question),
        "corrupted_examples": [u.id for u in job.spec.examples if u.corrupt],
        "reply": completion.text,
        "sql": sqlcheck.extract_sql(completion.text),
        "status": {cid: chk.status for cid, chk in checks.items()},
        "detail": {cid: chk.detail for cid, chk in checks.items() if chk.detail},
        "meta": {k: v for k, v in completion.meta.items() if k != "cached"},
    }


def run(scenario: Scenario, runner: Runner, jobs: list[Job], workers: int = 8, log=print) -> list[dict]:
    """Run the jobs and score them. Jobs with the same prompt share one call for N completions;
    the i-th such job of a (experiment, condition, target, question) cell takes completion i, so
    identical cells of two experiments read the same completions."""
    groups: dict[tuple[str, str], list[tuple[Job, int]]] = defaultdict(list)
    seen: dict[tuple, int] = defaultdict(int)
    for job in jobs:
        key = (job.prompt.system, job.prompt.user)
        cell = (job.experiment, job.condition, job.target, job.question.id, key)
        groups[key].append((job, seen[cell]))
        seen[cell] += 1

    records: list[dict] = []
    failed = 0
    stop: BudgetExceeded | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(runner.complete, members[0][0].prompt, max(r for _, r in members) + 1): members
            for members in groups.values()
        }
        for done, fut in enumerate(as_completed(futures), 1):
            members = futures[fut]
            try:
                completions = fut.result()
            except TransientError as e:
                failed += len(members)
                log(f"  skipped {len(members)} job(s): {e}")
                continue
            except BudgetExceeded as e:
                stop = e
                continue
            records += [_record(scenario, runner.name, job, completions[r], r) for job, r in members]
            if done % 100 == 0:
                log(f"  {done}/{len(futures)} prompts")
    if stop is not None:
        raise stop
    if failed:
        log(f"  {failed} job(s) not run (transient errors); run again to fill them from the API")
    by_job = {(r["experiment"], r["condition"], r["target"], r["question"], r["sample"]): r for r in records}
    return [
        by_job[k]
        for job in jobs  # back in plan order
        if (k := (job.experiment, job.condition, job.target, job.question.id, job.sample)) in by_job
    ]


# ---------------------------------------------------------------------------------------------
# storage


def records_path(results: Path, runner: str, experiment: str, target: str | None) -> Path:
    name = experiment if target is None else f"{experiment}.{target}"
    return results / "records" / runner / f"{name}.jsonl"


def save_records(results: Path, records: list[dict]) -> list[Path]:
    """One file per (runner, experiment, target); a run replaces the conditions it re-ran and
    keeps the others."""
    buckets: dict[Path, list[dict]] = defaultdict(list)
    for r in records:
        buckets[records_path(results, r["runner"], r["experiment"], r["target"])].append(r)
    for path, rows in buckets.items():
        fresh = {(r["condition"], r["question"], r["sample"]) for r in rows}
        old = [r for r in _read_jsonl(path) if (r["condition"], r["question"], r["sample"]) not in fresh]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in [*old, *rows]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sorted(buckets)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_records(results: Path) -> list[dict]:
    """Every record under ``results/records/<runner>/``, plain ``.jsonl`` (fresh runs) or
    ``.jsonl.gz`` (the published files)."""
    paths = sorted([*(results / "records").glob("*/*.jsonl"), *(results / "records").glob("*/*.jsonl.gz")])
    return [r for path in paths for r in _read_jsonl(path)]


# ---------------------------------------------------------------------------------------------
# aggregation


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval of k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denom = 1 + z * z / n
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def _rate(follows: int, violates: int, other: int, want: str) -> dict:
    n = follows + violates
    k = follows if want == FOLLOWS else violates
    lo, hi = wilson(k, n)
    return {
        "follows": follows,
        "violates": violates,
        "not_scored": other,  # unparsed replies and replies the convention could not be read on
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def _count(records: Iterable[dict], conventions: Iterable[str]) -> tuple[int, int, int]:
    f = v = o = 0
    wanted = set(conventions)
    for r in records:
        for cid, status in r["status"].items():
            if cid in wanted:
                f += status == FOLLOWS
                v += status == VIOLATES
                o += status not in (FOLLOWS, VIOLATES)
    return f, v, o


def summarise(scenario: Scenario, records: list[dict], context_tokens: dict | None = None) -> dict:
    """Result tables as plain data: e1 (follow rate), e2 and e3 (violation rate of the target)."""
    stated = [c.id for c in scenario.conventions if c.in_one_line_rule]
    unstated = [c.id for c in scenario.conventions if not c.in_one_line_rule]
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by[(r["runner"], r["experiment"], r["target"], r["condition"])].append(r)

    e1, e2, e3, by_question = [], [], [], []
    for (runner, experiment, target, condition), rows in sorted(by.items(), key=lambda kv: str(kv[0])):
        base = {"runner": runner, "condition": condition}
        if experiment == "e1":
            for label, convs in [
                *[(cid, [cid]) for cid in scenario.convention_ids],
                ("stated_by_one_line_rule", stated),
                ("not_stated_by_one_line_rule", unstated),
                ("all", list(scenario.convention_ids)),
            ]:
                e1.append(base | {"convention": label} | _rate(*_count(rows, convs), want=FOLLOWS))
            scored = [r for r in rows if all(s in (FOLLOWS, VIOLATES) for s in r["status"].values())]
            perfect = sum(all(s == FOLLOWS for s in r["status"].values()) for r in scored)
            lo, hi = wilson(perfect, len(scored))
            e1.append(
                base
                | {
                    "convention": "every_applicable_convention",
                    "follows": perfect,
                    "violates": len(scored) - perfect,
                    "not_scored": len(rows) - len(scored),
                    "n": len(scored),
                    "rate": round(perfect / len(scored), 4) if scored else None,
                    "ci95": [round(lo, 4), round(hi, 4)],
                }
            )
        else:
            row = base | {"target": target} | _rate(*_count(rows, [target]), want=VIOLATES)
            (e2 if experiment == "e2" else e3).append(row)
            per_q: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                per_q[r["question"]].append(r)
            for qid, qrows in per_q.items():
                by_question.append(
                    {"experiment": experiment}
                    | base
                    | {"target": target, "question": qid, "nearest_example": qrows[0]["nearest_example"]}
                    | _rate(*_count(qrows, [target]), want=VIOLATES)
                )

    # E1 split by question family: does the question ask for the metric, or for something else
    # (an order count, a refund rate)? The examples only ever show the metric.
    metric = scenario.conventions[0].id
    family = {q.id: "metric" if metric in scenario.applies(q) else "other" for q in scenario.questions}
    e1_family = []
    for (runner, experiment, _target, condition), rows in sorted(by.items(), key=lambda kv: str(kv[0])):
        if experiment != "e1":
            continue
        for fam in ("metric", "other"):
            fam_rows = [r for r in rows if family[r["question"]] == fam]
            for cid in scenario.convention_ids:
                rate = _rate(*_count(fam_rows, [cid]), want=FOLLOWS)
                if rate["n"]:
                    e1_family.append(
                        {"runner": runner, "condition": condition, "family": fam, "convention": cid} | rate
                    )

    out = {
        "scenario": scenario.id,
        "e1": e1,
        "e1_by_family": e1_family,
        "e2": e2,
        "e3": e3,
        "by_question": by_question,  # E2 and E3, per question
        "one_wrong_example": _one_wrong(records),
    }
    if context_tokens:
        out["e1_efficiency"] = _efficiency(scenario, e1, context_tokens)
    return out


def _one_wrong(records: list[dict]) -> list[dict]:
    """Every examples-only context with exactly one corrupted example (E2 and E3 pooled, far and
    near questions apart): is the corrupted one the nearest, and where does it sit in the list?"""
    by: dict[tuple, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for r in records:
        spec = r["spec"]
        if r["target"] is None or spec["rules"] != "none" or len(r["corrupted_examples"]) != 1:
            continue
        # two conditions can pick the same example: the same completion must count once
        same = (r["runner"], r["target"], r["question"], json.dumps(spec, sort_keys=True), r["replicate"])
        if same in seen:
            continue
        seen.add(same)
        ids = [e["id"] for e in spec["examples"]]
        (bad,) = r["corrupted_examples"]
        which = "nearest" if bad == r["nearest_example"] else "other"
        position = "last" if ids[-1] == bad else "first" if ids[0] == bad else "middle"
        for key in (("which", which), ("position", position), ("which+position", f"{which}, {position}")):
            by[(r["runner"], r["target"], r["kind"], *key)].append(r)
        # far and near pooled: the changed example against the example the question is nearest to
        by[(r["runner"], r["target"], "all", "changed/nearest", f"{bad} / {r['nearest_example']}")].append(r)
    return [
        {"runner": runner, "target": target, "questions": kind, "split": split, "corrupted_example": value}
        | _rate(*_count(rows, [target]), want=VIOLATES)
        for (runner, target, kind, split, value), rows in sorted(by.items())
    ]


def _efficiency(scenario: Scenario, e1: list[dict], context_tokens: dict) -> list[dict]:
    """Conventions followed (sum of the per-convention follow rates, max = number of conventions)
    per 100 context tokens, absolute and as a gain over no context."""
    total: dict[tuple[str, str], float] = defaultdict(float)
    for row in e1:
        if row["convention"] in scenario.convention_ids and row["rate"] is not None:
            total[(row["runner"], row["condition"])] += row["rate"]
    out = []
    for (runner, condition), followed in sorted(total.items()):
        tokens = (context_tokens.get(runner) or {}).get(condition)
        base = total.get((runner, "none"), 0.0)
        row = {
            "runner": runner,
            "condition": condition,
            "context_tokens": tokens,
            "conventions_followed": round(followed, 3),
            "gain_vs_none": round(followed - base, 3),
        }
        if tokens:
            row["gain_per_100_tokens"] = round(100 * (followed - base) / tokens, 3)
            row["followed_per_100_tokens"] = round(100 * followed / tokens, 3)
        out.append(row)
    return out
