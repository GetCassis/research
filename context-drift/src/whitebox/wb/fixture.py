"""A small stand-in scenario for the plumbing tests, built with the behavioural package's own classes
(`talkdemo.scenario`) so it goes through the same builder and checkers as the real one.

Invented data. It is NOT the talk's scenario: it only keeps the tests independent of edits to
`src/scenarios/`. Three conventions: revenue is net of refunds, test orders are excluded,
amounts are in euros.
"""

from __future__ import annotations

from functools import partial

from talkdemo import sqlcheck
from talkdemo.scenario import Convention, Item, Scenario

SCHEMA = """\
CREATE TABLE sales (
  sale_id      bigint PRIMARY KEY,
  shop         text,
  sold_at      timestamptz,
  price_cents  bigint,
  refund_cents bigint,
  is_test      boolean
);"""

CONVENTIONS = (
    Convention(
        id="net",
        layer="metric",
        description="Revenue subtracts refunds: price_cents - refund_cents.",
        rule="Revenue is the price minus refunds.",
        wrong_rule="Revenue is the price; refunds are not subtracted.",
        in_one_line_rule=True,
        check=partial(sqlcheck.check_subtracted, base="price_cents", minus="refund_cents"),
        corrupt=((r"(price_cents) - refund_cents", r"\1"),),
    ),
    Convention(
        id="no_test",
        layer="metric",
        description="Test sales are excluded: NOT is_test.",
        rule="Test sales are excluded.",
        wrong_rule="Test sales are included.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_flag_excluded, table="sales", column="is_test"),
        corrupt=((r"\nWHERE NOT is_test", ""),),
    ),
    Convention(
        id="euros",
        layer="output",
        description="Amounts are in euros, not cents: / 100.0.",
        rule="Amounts are reported in euros, not cents.",
        wrong_rule="Amounts are reported in cents.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_scaled, columns=("price_cents", "refund_cents"), divisor=100),
        corrupt=((r" / 100\.0", ""),),
    ),
)


def _sql(select: str, tail: str) -> str:
    return f"SELECT {select}\nFROM sales\nWHERE NOT is_test{tail};"


_REV = "SUM(price_cents - refund_cents) / 100.0 AS revenue"

EXAMPLES = (
    Item("ex_shop", "Revenue by shop.", _sql(f"shop,\n       {_REV}", "\nGROUP BY shop")),
    Item("ex_total", "Total revenue.", _sql(_REV, "")),
    Item("ex_day", "Revenue per day.", _sql(f"sold_at::date AS day,\n       {_REV}", "\nGROUP BY 1\nORDER BY 1")),
    Item("ex_count", "Number of sales by shop.", _sql("shop,\n       COUNT(*) AS sales", "\nGROUP BY shop")),
    Item(
        "ex_top",
        "Top 3 shops by revenue.",
        _sql(f"shop,\n       {_REV}", "\nGROUP BY shop\nORDER BY revenue DESC\nLIMIT 3"),
    ),
)

QUESTIONS = (
    Item(
        "q_month",
        "Revenue per month.",
        _sql(f"date_trunc('month', sold_at) AS month,\n       {_REV}", "\nGROUP BY 1\nORDER BY 1"),
    ),
    Item(
        "q_best",
        "Which shop has the highest revenue?",
        _sql(f"shop,\n       {_REV}", "\nGROUP BY shop\nORDER BY revenue DESC\nLIMIT 1"),
    ),
    Item(
        "q_count_day",
        "Number of sales per day.",
        _sql("sold_at::date AS day,\n       COUNT(*) AS sales", "\nGROUP BY 1\nORDER BY 1"),
    ),
    Item(
        "n_top5",
        "Top 5 shops by revenue.",
        _sql(f"shop,\n       {_REV}", "\nGROUP BY shop\nORDER BY revenue DESC\nLIMIT 5"),
        near="ex_top",
    ),
)

SCENARIO = Scenario(
    id="fixture",
    title="Plumbing fixture",
    dialect="postgres",
    schema=SCHEMA,
    conventions=CONVENTIONS,
    one_line_rule="Revenue = price minus refunds.",
    one_line_wrong={"net": "Revenue = price (refunds are not subtracted)."},
    examples=EXAMPLES,
    default_examples=tuple(e.id for e in EXAMPLES),
    questions=QUESTIONS,
)
