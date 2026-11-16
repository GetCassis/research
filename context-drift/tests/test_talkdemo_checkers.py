"""Convention checkers: hand-written SQL in the shapes models actually produce."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scenarios import load  # noqa: E402
from talkdemo import sqlcheck  # noqa: E402
from talkdemo.sqlcheck import FOLLOWS, NA, UNPARSED, VIOLATES  # noqa: E402

S = load("shop")
V1 = load("shop_v1")


def status(cid: str, sql: str) -> str:
    return S.check_sql(sql)[cid].status


# -- extraction -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("SELECT 1;", "SELECT 1;"),
        ("```sql\nSELECT 1;\n```", "SELECT 1;"),
        ("Here is the query:\n\n```\nSELECT 1\n```\nIt returns one.", "SELECT 1"),
        ("Here you go:\nSELECT 1;\nThis returns one.", "SELECT 1;"),
        ("SQL:\nWITH a AS (SELECT 1) SELECT * FROM a; SELECT 2;", "WITH a AS (SELECT 1) SELECT * FROM a;"),
    ],
)
def test_extract_sql(reply, expected):
    assert sqlcheck.extract_sql(reply) == expected


def test_unparseable_reply_is_unparsed():
    assert sqlcheck.parse("I cannot answer that.") is None
    assert {c.status for c in S.check_sql("SELECT FROM WHERE (").values()} == {UNPARSED}


# -- net_refunds ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT SUM(amount - refunded) FROM orders", FOLLOWS),
        ("SELECT SUM(o.amount - COALESCE(o.refunded, 0)) FROM orders o", FOLLOWS),
        ("SELECT SUM(amount) - SUM(refunded) FROM orders", FOLLOWS),
        ("WITH n AS (SELECT amount - refunded AS net FROM orders) SELECT SUM(net) FROM n", FOLLOWS),
        ("SELECT SUM(amount) FROM orders", VIOLATES),
        ("SELECT SUM(amount), SUM(refunded) FROM orders", VIOLATES),
        ("SELECT SUM(refunded) / SUM(amount) FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders", NA),
    ],
)
def test_net_refunds(sql, expected):
    assert status("net_refunds", sql) == expected


# -- completed_only ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM orders WHERE status = 'completed'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders o WHERE o.status IN ('completed') AND o.channel <> 'test'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE 'completed' = status", FOLLOWS),
        ("SELECT COUNT(*) FILTER (WHERE status = 'completed') FROM orders", FOLLOWS),
        # the filter sits in the query that reads the CTE
        (
            "WITH o AS (SELECT * FROM orders WHERE channel <> 'test') SELECT COUNT(*) FROM o WHERE status = 'completed'",
            FOLLOWS,
        ),
        ("SELECT COUNT(*) FROM orders WHERE (status = 'completed' OR status IS NULL)", FOLLOWS),
        ("SELECT COUNT(*) FROM (SELECT * FROM orders) o WHERE o.status = 'completed'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE status <> 'cancelled'", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE status IN ('completed', 'pending')", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE status = 'completed' OR status = 'pending'", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE NOT (status = 'completed')", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE status = 'paid'", VIOLATES),
        # one of the two subqueries forgets the filter
        (
            "SELECT (SELECT COUNT(*) FROM orders WHERE status = 'completed') * 1.0 / (SELECT COUNT(*) FROM orders)",
            VIOLATES,
        ),
        ("SELECT COUNT(*) FROM customers", NA),
    ],
)
def test_completed_only(sql, expected):
    assert status("completed_only", sql) == expected


def test_completed_only_reports_the_values_filtered_on():
    def detail(sql):
        return S.check_sql(sql)["completed_only"].detail["values"]

    assert detail("SELECT COUNT(*) FROM orders WHERE status = 'delivered' AND channel <> 'test'") == ["delivered"]
    assert detail("SELECT COUNT(*) FROM orders WHERE status IN ('completed', 'delivered')") == [
        "completed",
        "delivered",
    ]
    assert detail("SELECT COUNT(*) FROM orders WHERE channel <> 'test'") == []


# -- exclude_test -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM orders WHERE NOT is_test", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE is_test = false", FOLLOWS),
        ("SELECT COUNT(*) FROM orders o WHERE o.is_test IS FALSE", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE is_test IS NOT TRUE", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE is_test <> true", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE NOT COALESCE(is_test, false)", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE COALESCE(is_test, FALSE) = FALSE", FOLLOWS),
        ("SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id AND NOT o.is_test", FOLLOWS),
        ("SELECT COUNT(*) FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE is_test", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE is_test = true", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE NOT (is_test = false)", VIOLATES),
        ("SELECT is_test, COUNT(*) FROM orders GROUP BY is_test", VIOLATES),
    ],
)
def test_exclude_test_flag(sql, expected):
    """shop_v1 marks test orders with a boolean flag."""
    assert V1.check_sql(sql)["exclude_test"].status == expected


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM orders WHERE channel <> 'test'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders o WHERE o.status = 'completed' AND o.channel != 'test'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE channel NOT IN ('test')", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE channel NOT IN ('test', 'internal')", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE NOT (channel = 'test')", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE channel IS DISTINCT FROM 'test'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE (channel <> 'test' OR channel IS NULL)", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE COALESCE(channel, '') <> 'test'", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE channel = 'web'", FOLLOWS),  # a positive filter excludes them too
        ("SELECT COUNT(*) FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE channel = 'test'", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE channel IN ('web', 'test')", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE channel <> 'test' OR status = 'completed'", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE NOT (channel <> 'test')", VIOLATES),
        ("SELECT channel, COUNT(*) FROM orders GROUP BY channel", VIOLATES),
        ("SELECT COUNT(*) FROM customers", NA),
    ],
)
def test_exclude_test(sql, expected):
    assert status("exclude_test", sql) == expected


# -- currency_units ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT SUM(amount - refunded) / 100.0 FROM orders", FOLLOWS),
        ("SELECT SUM(amount)::numeric / 100 FROM orders", FOLLOWS),
        ("SELECT ROUND(SUM((amount - refunded) / 100.0), 2) FROM orders", FOLLOWS),
        ("SELECT SUM(amount) * 0.01 FROM orders", FOLLOWS),
        ("SELECT SUM(amount - refunded) FROM orders", VIOLATES),
        ("SELECT 100.0 * SUM(refunded) / SUM(amount) FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders", NA),
    ],
)
def test_currency_units(sql, expected):
    assert status("currency_units", sql) == expected


# -- local_time -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') FROM orders", FOLLOWS),
        ("SELECT date_trunc('month', ordered_at, 'Europe/Paris') FROM orders", FOLLOWS),
        ("SELECT date_trunc('day', timezone('Europe/Paris', ordered_at)) FROM orders", FOLLOWS),
        ("SELECT (o.ordered_at AT TIME ZONE 'UTC' AT TIME ZONE 'Europe/Paris')::date FROM orders o", FOLLOWS),
        ("SELECT COUNT(*) FROM orders WHERE ordered_at >= TIMESTAMPTZ '2025-01-01 00:00 Europe/Paris'", FOLLOWS),
        (
            "SELECT COUNT(*) FROM orders WHERE ordered_at >= '2025-01-01'::timestamp AT TIME ZONE 'Europe/Paris'",
            FOLLOWS,
        ),
        (
            "WITH o AS (SELECT ordered_at AT TIME ZONE 'Europe/Paris' AS ordered_at FROM orders) "
            "SELECT date_trunc('month', ordered_at) FROM o",
            FOLLOWS,
        ),
        ("SELECT date_trunc('month', ordered_at) FROM orders", VIOLATES),
        ("SELECT date_trunc('month', ordered_at AT TIME ZONE 'UTC') FROM orders", VIOLATES),
        ("SELECT COUNT(*) FROM orders WHERE ordered_at >= '2025-01-01' AND ordered_at < '2026-01-01'", VIOLATES),
        # strict: the bucket is converted, the filter is not
        (
            "SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') FROM orders WHERE ordered_at >= '2025-01-01'",
            VIOLATES,
        ),
        ("SELECT COUNT(*) FROM orders", NA),
    ],
)
def test_local_time(sql, expected):
    assert status("local_time", sql) == expected


def test_local_time_reports_partial_use():
    chk = S.check_sql(
        "SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') FROM orders WHERE ordered_at >= '2025-01-01'"
    )["local_time"]
    assert chk.detail == {"uses": 2, "converted": 1}


# -- pct_scale / pct_round --------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, scale, rounding",
    [
        ("SELECT ROUND(100.0 * SUM(refunded) / SUM(amount), 1) FROM orders", FOLLOWS, FOLLOWS),
        ("SELECT ROUND((SUM(refunded) * 100.0 / SUM(amount))::numeric, 1) FROM orders", FOLLOWS, FOLLOWS),
        (
            "SELECT ROUND(100 * AVG((c.segment = 'business')::int), 1) FROM orders o JOIN customers c USING (customer_id)",
            FOLLOWS,
            FOLLOWS,
        ),
        (
            "WITH r AS (SELECT 100.0 * SUM(refunded) / SUM(amount) AS pct FROM orders) SELECT ROUND(pct, 1) FROM r",
            FOLLOWS,
            FOLLOWS,
        ),
        ("SELECT ROUND(100.0 * SUM(refunded) / SUM(amount), 2) FROM orders", FOLLOWS, VIOLATES),
        ("SELECT 100.0 * SUM(refunded) / NULLIF(SUM(amount), 0) FROM orders", FOLLOWS, VIOLATES),
        ("SELECT SUM(refunded)::numeric / SUM(amount) FROM orders", VIOLATES, VIOLATES),
        ("SELECT ROUND(SUM(refunded)::numeric / SUM(amount), 1) FROM orders", VIOLATES, FOLLOWS),
        ("SELECT ROUND(SUM(refunded)::numeric / SUM(amount)) FROM orders", VIOLATES, VIOLATES),
        # dividing by 100 (cents to euros) is not a percentage
        ("SELECT SUM(amount) / 100.0 FROM orders", VIOLATES, VIOLATES),
    ],
)
def test_pct_scale_and_round(sql, scale, rounding):
    assert status("pct_scale", sql) == scale
    assert status("pct_round", sql) == rounding


def test_pct_round_reports_the_decimals_found():
    chk = S.check_sql("SELECT ROUND(100.0 * SUM(refunded) / SUM(amount), 2) FROM orders")["pct_round"]
    assert chk.detail == {"round_decimals": [2]}


# -- scoring a reply --------------------------------------------------------------------------


def test_score_keeps_only_applicable_conventions():
    q = S.question("q_orders_week")  # a count: no money, no ratio
    reply = "```sql\nSELECT date_trunc('week', ordered_at) AS week, COUNT(*) FROM orders WHERE status = 'completed' GROUP BY 1;\n```"
    got = {cid: chk.status for cid, chk in S.score(q, reply).items()}
    assert got == {"completed_only": FOLLOWS, "exclude_test": VIOLATES, "local_time": VIOLATES}


def test_score_marks_prose_as_unparsed():
    q = S.question("q_year")
    assert {chk.status for chk in S.score(q, "I need more information.").values()} == {UNPARSED}
