"""Context builder, conditions and the experiment plumbing (no API calls: a fake runner)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios import load  # noqa: E402
from talkdemo import experiments as ex  # noqa: E402
from talkdemo.prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    ContextSpec,
    ExampleUse,
    build_context,
    build_prompt,
    corruption_order,
    display_order,
    examples_spec,
    rule_lines,
    rules_spec,
)
from talkdemo.runners import FunctionRunner  # noqa: E402
from talkdemo.sqlcheck import FOLLOWS, VIOLATES  # noqa: E402

S = load("shop")
Q = S.question("q_hour")
FIVE = list(S.default_examples)


def shown_sql(spec: ContextSpec) -> dict[str, str]:
    return {u.id: S.variant(S.example(u.id), u.corrupt) for u in spec.examples}


# -- rules ------------------------------------------------------------------------------------


def test_no_context_is_schema_and_question_only():
    p = build_prompt(S, ContextSpec(), Q)
    assert p.system == SYSTEM_PROMPT
    assert p.user.startswith("Database schema (PostgreSQL):\n\nCREATE TABLE customers")
    assert "Rules:" not in p.user and "Examples:" not in p.user
    assert p.user.endswith(f"Question: {Q.question}\nSQL:")
    assert p.user.count("Question:") == 1


def test_one_line_rule_only():
    user = build_prompt(S, rules_spec("one_line"), Q).user
    assert f"Rules:\n- {S.one_line_rule}\n\nQuestion:" in user


def test_all_rules_in_convention_order():
    lines = rule_lines(S, rules_spec("all"))
    assert lines == [c.rule for c in S.conventions]
    assert "Rules:\n" + "\n".join(f"- {r}" for r in lines) in build_context(S, rules_spec("all"))


def test_a_rule_can_be_stated_wrongly():
    lines = rule_lines(S, rules_spec("all", ["net_refunds"]))
    assert lines[0] == S.convention("net_refunds").wrong_rule
    assert lines[1:] == [c.rule for c in S.conventions[1:]]
    assert rule_lines(S, rules_spec("one_line", ["net_refunds"])) == [S.one_line_wrong["net_refunds"]]


def test_wrong_rule_errors():
    with pytest.raises(ValueError):  # the one-line rule does not state this convention
        rule_lines(S, rules_spec("one_line", ["exclude_test"]))
    with pytest.raises(ValueError):
        rule_lines(S, rules_spec("all", ["no_such_convention"]))
    with pytest.raises(ValueError):
        ContextSpec(rules="none", wrong_rules=frozenset(["net_refunds"]))
    with pytest.raises(ValueError):
        ContextSpec(rules="some")


# -- examples ---------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 5])
def test_n_examples_are_a_prefix_of_the_standard_five(n):
    spec = examples_spec(S, Q, 0, n=n)
    assert sorted(u.id for u in spec.examples) == sorted(FIVE[:n])
    user = build_prompt(S, spec, Q).user
    assert user.count("Question:") == n + 1
    for eid in FIVE[:n]:
        assert f"Question: {S.example(eid).question}\nSQL:\n{S.example(eid).sql}" in user
    assert all(not u.corrupt for u in spec.examples)


def test_canonical_order_when_not_shuffled():
    spec = examples_spec(S, Q, 3, shuffle=False)
    assert [u.id for u in spec.examples] == FIVE


def test_corruption_flags_touch_only_the_flagged_example():
    spec = ContextSpec(examples=(ExampleUse("ex_month", frozenset(["net_refunds"])), ExampleUse("ex_country")))
    sql = shown_sql(spec)
    assert sql["ex_country"] == S.example("ex_country").sql
    assert "SUM(amount) / 100.0" in sql["ex_month"] and "refunded" not in sql["ex_month"]
    assert S.check_sql(sql["ex_month"])["net_refunds"].status == VIOLATES
    assert S.check_sql(sql["ex_month"])["local_time"].status == FOLLOWS
    user = build_prompt(S, spec, Q).user
    assert sql["ex_month"] in user and S.example("ex_month").sql not in user


@pytest.mark.parametrize("conv", ["net_refunds", "completed_only", "exclude_test", "currency_units"])
def test_k_of_5_is_nested_and_breaks_only_the_target(conv):
    previous: set[str] = set()
    for k in range(6):
        spec = examples_spec(S, Q, 2, convention=conv, k=k)
        corrupted = {u.id for u in spec.examples if u.corrupt}
        assert len(corrupted) == k and previous <= corrupted
        previous = corrupted
        for u in spec.examples:
            checks = S.check_sql(S.variant(S.example(u.id), u.corrupt))
            assert checks[conv].status == (VIOLATES if u.corrupt else FOLLOWS)
            assert all(
                c.status == FOLLOWS for cid, c in checks.items() if cid != conv and cid in S.applies(S.example(u.id))
            )
    assert previous == set(FIVE)
    assert [u.id for u in examples_spec(S, Q, 2, convention=conv, k=1).examples if u.corrupt] == corruption_order(
        S, Q, 2
    )[:1]


def test_nearest_and_far_picks():
    for q in S.near_questions:
        for sample in range(4):
            near = examples_spec(S, q, sample, convention="net_refunds", k=1, pick="nearest")
            far = examples_spec(S, q, sample, convention="net_refunds", k=1, pick="far")
            assert [u.id for u in near.examples if u.corrupt] == [q.near]
            (other,) = [u.id for u in far.examples if u.corrupt]
            assert other != q.near and other in FIVE


def test_conditions_are_paired_on_order_and_differ_across_samples():
    clean = examples_spec(S, Q, 1)
    broken = examples_spec(S, Q, 1, convention="net_refunds", k=1, pick="nearest")
    with_rules = examples_spec(S, Q, 1, rules="all")
    assert [u.id for u in clean.examples] == [u.id for u in broken.examples] == [u.id for u in with_rules.examples]
    assert [u.id for u in clean.examples] == display_order(S, Q, 1, FIVE)
    orders = {tuple(display_order(S, Q, s, FIVE)) for s in range(10)}
    assert len(orders) > 3
    # fewer examples: the same relative order
    assert display_order(S, Q, 1, FIVE[:3]) == [e for e in display_order(S, Q, 1, FIVE) if e in FIVE[:3]]


def test_cannot_corrupt_a_convention_an_example_does_not_show():
    with pytest.raises(ValueError):
        build_prompt(S, examples_spec(S, Q, 0, convention="local_time", k=5), Q)
    with pytest.raises(ValueError):
        examples_spec(S, Q, 0, k=1)  # no convention given
    with pytest.raises(ValueError):
        examples_spec(S, Q, 0, convention="net_refunds", k=2, pick="nearest")


# -- rules + examples, conflicts --------------------------------------------------------------


def test_rules_plus_examples_and_both_conflicts():
    both = build_prompt(S, examples_spec(S, Q, 0, rules="all"), Q).user
    assert both.index("Rules:") < both.index("Examples:") < both.rindex("Question:")

    rule_right = examples_spec(S, Q, 0, rules="all", convention="net_refunds", k=5)
    assert rule_lines(S, rule_right) == [c.rule for c in S.conventions]
    assert all(S.check_sql(sql)["net_refunds"].status == VIOLATES for sql in shown_sql(rule_right).values())

    rule_wrong = examples_spec(S, Q, 0, rules="all", wrong_rules=["net_refunds"])
    assert rule_lines(S, rule_wrong)[0] == S.convention("net_refunds").wrong_rule
    assert all(S.check_sql(sql)["net_refunds"].status == FOLLOWS for sql in shown_sql(rule_wrong).values())


def test_spec_round_trips_through_json():
    spec = examples_spec(S, Q, 4, rules="all", wrong_rules=["net_refunds"], convention="exclude_test", k=2)
    again = ContextSpec.from_json(spec.to_json())
    assert again == spec
    assert build_prompt(S, again, Q).user == build_prompt(S, spec, Q).user


def test_prompt_accepts_question_text_or_item():
    assert build_prompt(S, ContextSpec(), Q).user == build_prompt(S, ContextSpec(), Q.question).user


# -- experiments ------------------------------------------------------------------------------


def test_corruptible_conventions_are_the_metric_ones():
    assert ex.corruptible(S) == ["net_refunds", "completed_only", "exclude_test", "currency_units"]
    with pytest.raises(ValueError):
        ex.plan(S, "e2", 1, ["local_time"])


def test_plan_sizes():
    e1 = ex.plan(S, "e1", 2)
    assert len(e1) == 8 * len(S.far_questions) * 2
    assert {j.condition for j in e1} == {"none", "rule1", "rules", "ex1", "ex3", "ex5", "rule1+ex5", "rules+ex5"}
    e2 = ex.plan(S, "e2", 1, ["net_refunds"])
    qs = [q for q in S.far_questions if "net_refunds" in S.applies(q)]
    assert len(e2) == 18 * len(qs) and {j.target for j in e2} == {"net_refunds"}
    assert "rule1:wrong" not in {j.condition for j in ex.plan(S, "e2", 1, ["exclude_test"])}
    e3 = ex.plan(S, "e3", 2, ["net_refunds"])
    assert len(e3) == 3 * len(S.near_questions) * 2 and all(j.question.near for j in e3)


def test_baselines_of_e2_reuse_the_prompts_of_e1():
    e1 = {(j.condition, j.question.id, j.sample): j.prompt.user for j in ex.plan(S, "e1", 2)}
    for j in ex.plan(S, "e2", 2, ["net_refunds"], conditions=["rules", "ex5", "rules+ex5", "rule1"]):
        assert e1[(j.condition, j.question.id, j.sample)] == j.prompt.user


def _echo_runner(calls: list) -> FunctionRunner:
    """Follows whatever the prompt shows: copies net revenue from the first example, if any."""

    def fn(prompt, n):
        calls.append((prompt.user, n))
        gross = "SUM(amount) / 100.0" in prompt.user.split("Examples:")[-1].split("Question:")[1]
        sql = "SELECT SUM(amount{}) / 100.0 FROM orders WHERE status = 'completed' AND channel <> 'test';"
        return [sql.format("" if gross else " - refunded")] * n

    return FunctionRunner("fake", fn)


def test_run_scores_and_shares_identical_prompts():
    calls: list = []
    jobs = ex.plan(S, "e1", 3, conditions=["none", "ex5"], questions=["q_year", "q_orders_week"])
    records = ex.run(S, _echo_runner(calls), jobs, workers=2, log=lambda *_: None)
    assert len(records) == len(jobs) == 12
    # the no-context prompt is the same for the 3 samples: one call for 3 completions
    assert sorted(n for user, n in calls if "Examples:" not in user) == [3, 3]
    first = records[0]
    assert (first["condition"], first["question"], first["sample"], first["runner"]) == ("none", "q_year", 0, "fake")
    assert first["status"]["net_refunds"] == FOLLOWS and first["status"]["local_time"] == "na"
    count = next(r for r in records if r["question"] == "q_orders_week")
    assert set(count["status"]) == {"completed_only", "exclude_test", "local_time"}


def test_summarise_counts_violations_of_the_target():
    jobs = ex.plan(S, "e2", 2, ["net_refunds"], conditions=["ex5", "ex5:5"], questions=["q_year", "q_hour"])
    records = ex.run(S, _echo_runner([]), jobs, workers=1, log=lambda *_: None)
    rows = {r["condition"]: r for r in ex.summarise(S, records)["e2"]}
    assert rows["ex5"]["rate"] == 0.0 and rows["ex5"]["n"] == 4
    assert rows["ex5:5"]["rate"] == 1.0 and rows["ex5:5"]["violates"] == 4
    assert rows["ex5:5"]["ci95"][0] > 0.4


def test_summarise_e1_rows_and_efficiency():
    jobs = ex.plan(S, "e1", 1, conditions=["none", "ex5"])
    records = ex.run(S, _echo_runner([]), jobs, workers=1, log=lambda *_: None)
    summary = ex.summarise(S, records, {"fake": {"none": 0, "ex5": 600}})
    rows = {(r["condition"], r["convention"]): r for r in summary["e1"]}
    assert rows[("ex5", "net_refunds")]["rate"] == 1.0
    assert rows[("ex5", "exclude_test")]["n"] == len(S.far_questions)
    assert rows[("ex5", "every_applicable_convention")]["rate"] < 1.0  # the fake ignores dates and ratios
    eff = {r["condition"]: r for r in summary["e1_efficiency"]}
    assert eff["ex5"]["context_tokens"] == 600 and eff["ex5"]["gain_vs_none"] == 0.0


def test_wilson_interval():
    lo, hi = ex.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = ex.wilson(10, 20)
    assert 0.29 < lo < 0.31 and 0.69 < hi < 0.71
    assert ex.wilson(0, 0) == (0.0, 1.0)
