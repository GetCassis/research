"""Each scenario is self-consistent: examples and references follow the conventions, every
corrupted variant breaks exactly the convention it targets, decision points line up."""

import json
import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios import NAMES, load  # noqa: E402
from talkdemo import sqlcheck  # noqa: E402
from talkdemo.sqlcheck import FOLLOWS, VIOLATES  # noqa: E402

SCENARIOS = [load(name) for name in NAMES]
S = load("shop")
ITEMS = [pytest.param(s, item, id=f"{s.id}:{item.id}") for s in SCENARIOS for item in (*s.examples, *s.questions)]
QUESTIONS = [pytest.param(s, q, id=f"{s.id}:{q.id}") for s in SCENARIOS for q in s.questions]
METRIC = {"net_refunds", "completed_only", "exclude_test", "currency_units"}


def statuses(scenario, sql: str) -> dict[str, str]:
    return {cid: chk.status for cid, chk in scenario.check_sql(sql).items()}


def test_every_scenario_loads():
    assert [s.id for s in SCENARIOS] == list(NAMES)


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_exported_files_are_in_step_with_the_module(s):
    """scenarios/<name>.json and the decision points are what other tools read: a scenario edit
    without `python run.py export` must fail here, not feed them stale SQL."""
    folder = Path(__file__).resolve().parents[1] / "src" / "scenarios"
    assert json.loads((folder / f"{s.id}.json").read_text()) == json.loads(json.dumps(s.to_json()))
    rows = [json.loads(line) for line in (folder / f"{s.id}.decision_points.jsonl").read_text().splitlines()]
    assert [(r["question_id"], r["convention_id"], r["prefix"], r["follows"], r["violates"]) for r in rows] == [
        (dp.question_id, dp.convention_id, dp.prefix, dp.follows, dp.violates) for dp in s.decision_points()
    ]


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_sizes_meet_the_brief(s):
    assert len(s.examples) >= 8
    assert len(s.far_questions) >= 10
    assert len(s.near_questions) >= 3
    assert len(s.default_examples) == 5
    assert {q.near for q in s.near_questions} <= set(s.default_examples)
    ids = [i.id for i in (*s.examples, *s.questions)]
    assert len(ids) == len(set(ids))
    assert not {e.question for e in s.examples} & {q.question for q in s.questions}


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_slide_constraints(s):
    for e in s.examples:
        assert len(e.sql.splitlines()) <= 8, e.id
    lines = sum(len(s.example(eid).sql.splitlines()) + 2 for eid in s.default_examples)
    assert lines <= 60


@pytest.mark.parametrize("s, item", ITEMS)
def test_clean_sql_parses_and_follows_the_shared_conventions(s, item):
    assert sqlcheck.parse(item.sql) is not None
    assert {"completed_only", "exclude_test"} <= set(s.applies(item))  # every query reads orders


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_the_standard_five_all_rely_on_net_revenue(s):
    for eid in s.default_examples:
        assert METRIC <= set(s.applies(s.example(eid))), eid


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s.id)
def test_every_convention_is_shown_and_tested(s):
    shown = {cid for eid in s.default_examples for cid in s.applies(s.example(eid))}
    assert shown == set(s.convention_ids)
    for cid in s.convention_ids:
        assert sum(cid in s.applies(q) for q in s.far_questions) >= 4, cid


def test_the_one_line_rule_states_two_of_the_seven_conventions():
    assert [c.id for c in S.conventions if c.in_one_line_rule] == ["net_refunds", "completed_only"]
    assert set(S.one_line_wrong) == {"net_refunds", "completed_only"}
    assert len(S.conventions) == 7


@pytest.mark.parametrize("s, item", ITEMS)
def test_each_corruption_breaks_exactly_its_convention(s, item):
    clean = statuses(s, item.sql)
    for cid in s.applies(item):
        broken = s.variant(item, [cid])
        assert broken != item.sql
        assert sqlcheck.parse(broken) is not None, (item.id, cid)
        got = statuses(s, broken)
        assert got[cid] == VIOLATES, (item.id, cid, broken)
        for other in s.applies(item):
            if other != cid:
                assert got[other] == clean[other] == FOLLOWS, (item.id, cid, other, broken)


@pytest.mark.parametrize("s, item", ITEMS)
def test_corruptions_combine(s, item):
    applies = s.applies(item)
    for r in (2, len(applies)):
        for combo in combinations(applies, r):
            got = statuses(s, s.variant(item, combo))
            for cid in applies:
                assert got[cid] == (VIOLATES if cid in combo else FOLLOWS), (item.id, combo, cid)


def test_corrupting_a_convention_that_does_not_apply_fails():
    with pytest.raises(ValueError):
        S.variant(S.example("ex_country"), ["local_time"])


@pytest.mark.parametrize("s, question", QUESTIONS)
def test_decision_points(s, question):
    for cid in s.applies(question):
        dp = s.decision_point(question, cid)
        assert dp.prefix + dp.follows == question.sql
        assert dp.prefix + dp.violates == s.variant(question, [cid])
        assert dp.follows_head != dp.violates_head
        assert dp.prefix and not dp.prefix[-1].isspace()
        # the prefix never ends inside a word that both continuations go on with
        nxt = {dp.follows[:1], dp.violates[:1]}
        assert not (dp.prefix[-1].isalnum() and all(c.isalnum() or c == "_" for c in nxt))


def test_decision_point_example():
    dp = S.decision_point(S.question("q_hour"), "net_refunds")
    assert dp.prefix.endswith("SUM(amount")
    assert dp.follows_head == " -" and dp.violates_head == ")"
    dp = S.decision_point(S.question("q_hour"), "local_time")
    assert dp.prefix.endswith("EXTRACT(hour FROM ordered_at")
    assert dp.follows_head == " AT" and dp.violates_head == ")"
    dp = S.decision_point(S.question("q_hour"), "exclude_test")
    assert dp.prefix.endswith("WHERE status = 'completed'")
    assert dp.follows_head == " AND" and dp.violates_head == "\nGROUP"
    dp = S.decision_point(S.question("q_hour"), "completed_only")  # a swapped value, not a dropped filter
    assert dp.prefix.endswith("WHERE status = '")
    assert dp.follows_head == "completed'" and dp.violates_head == "delivered'"
    dp = S.decision_point(S.question("q_refund_rate"), "pct_round")
    assert dp.follows_head == "\n       ROUND(100.0" and dp.violates_head == "\n       100.0"


def test_nearest_example():
    for q in S.near_questions:
        assert S.nearest_example(q) == q.near
    # far questions: highest word overlap, first listed on a tie
    assert S.nearest_example(S.question("q_aov_month")) == "ex_aov_segment"
    assert S.nearest_example(S.question("q_segment")) == "ex_month"  # ties with ex_country
    assert S.nearest_example(S.question("q_segment"), among=["ex_aov_segment", "ex_country"]) == "ex_country"


def test_json_export_is_complete():
    data = S.to_json()
    assert [c["id"] for c in data["conventions"]] == list(S.convention_ids)
    q = next(q for q in data["questions"] if q["id"] == "q_december_share")
    assert set(q["applies"]) == {
        "net_refunds",
        "completed_only",
        "exclude_test",
        "local_time",
        "pct_scale",
        "pct_round",
    }
    assert set(q["decision_points"]) == set(q["applies"]) == set(q["corrupted"])
    assert q["kind"] == "far" and q["near"] is None
