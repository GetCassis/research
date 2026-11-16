"""Scenario "shop": a tiny invented shop warehouse and its house metric, net revenue.

Everything here is made up for the talk (no customer data). Three layers of conventions:

* metric   how net revenue is computed: refunds subtracted, completed orders only, test orders
           excluded, cents converted to euros;
* date     timestamps are stored in UTC and bucketed in Paris time;
* output   ratios are percentages (0 to 100), rounded to one decimal: two conventions, because
           models pick up the scale and the rounding separately.

The one-line rule people would write ("Net revenue = amount minus refunds, for completed
orders.") states the first two only. Every example in the pool follows every convention that
applies to it; ``DEFAULT_EXAMPLES`` are five figures that all rely on net revenue.

The schema is deliberately plain: no comments, ``amount`` rather than ``amount_cents``, test
orders marked by ``channel = 'test'`` rather than an ``is_test`` flag. With self-explanatory
names (``shop_v1``) the models follow most of the metric conventions without any context.
"""

from __future__ import annotations

from functools import partial

from talkdemo import sqlcheck
from talkdemo.scenario import Convention, Item, Scenario

SCHEMA = """\
CREATE TABLE customers (
  customer_id bigint PRIMARY KEY,
  segment     text,
  signup_at   timestamptz
);
CREATE TABLE orders (
  order_id    bigint PRIMARY KEY,
  customer_id bigint REFERENCES customers,
  ordered_at  timestamptz,
  status      text,
  channel     text,
  amount      bigint,
  refunded    bigint,
  country     text
);"""

ZONE = "Europe/Paris"
_Q = r"(?:\w+\.)?"  # optional table alias

CONVENTIONS = (
    Convention(
        id="net_refunds",
        layer="metric",
        description="Net revenue subtracts refunds: amount - refunded.",
        rule="Net revenue is the order amount minus refunds.",
        wrong_rule="Net revenue is the order amount; refunds are not subtracted.",
        in_one_line_rule=True,
        check=partial(sqlcheck.check_subtracted, base="amount", minus="refunded"),
        corrupt=((rf"({_Q}amount) - {_Q}refunded", r"\1"),),
    ),
    Convention(
        id="completed_only",
        layer="metric",
        description="Only completed orders count: status = 'completed'.",
        # The one corruption that swaps a value instead of dropping an operation: the wrong
        # variant counts delivered orders, so rule and examples can disagree symmetrically.
        # The rule names the value: "completed orders" alone is read as whatever the examples
        # filter on (pilot: rule "completed" + examples 'delivered' gave 'delivered' 41/42).
        rule="Only completed orders count (status 'completed').",
        wrong_rule="Only delivered orders count (status 'delivered').",
        in_one_line_rule=True,
        check=partial(sqlcheck.check_equals_filter, table="orders", column="status", value="completed"),
        corrupt=((rf"({_Q}status) = 'completed'", r"\1 = 'delivered'"),),
    ),
    Convention(
        id="exclude_test",
        layer="metric",
        description="Test orders are excluded: channel <> 'test'.",
        rule="Test orders (channel 'test') are excluded.",
        wrong_rule="Test orders (channel 'test') are included.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_value_excluded, table="orders", column="channel", value="test"),
        corrupt=(
            (rf" AND {_Q}channel <> 'test'", ""),
            (rf"WHERE {_Q}channel <> 'test'\n\s*AND ", "WHERE "),
            (rf"[ \t]*WHERE {_Q}channel <> 'test'\n", ""),
            (rf"\n[ \t]*WHERE {_Q}channel <> 'test'(?=;|\)|$)", ""),
        ),
    ),
    Convention(
        id="currency_units",
        layer="metric",
        description="Amounts are stored in cents and reported in euros: / 100.0.",
        rule="Amounts are stored in cents and reported in euros.",
        wrong_rule="Amounts are stored in cents and reported in cents.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_scaled, columns=("amount", "refunded"), divisor=100),
        corrupt=((r" / 100\.0", ""),),
    ),
    Convention(
        id="local_time",
        layer="date",
        description="Timestamps are stored in UTC; days, months and hours are in Paris time: "
        "ordered_at AT TIME ZONE 'Europe/Paris'.",
        rule="Dates and times are in Paris time (timestamps are stored in UTC).",
        wrong_rule="Dates and times are in UTC.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_timezone, table="orders", column="ordered_at", zone=ZONE),
        corrupt=(
            (rf"\(({_Q}ordered_at) AT TIME ZONE '{ZONE}'\)", r"\1"),
            (rf" AT TIME ZONE '{ZONE}'", ""),
        ),
    ),
    Convention(
        id="pct_scale",
        layer="output",
        description="Ratios are reported as percentages (0 to 100), not fractions: 100.0 * ratio.",
        rule="Ratios are reported as percentages (0 to 100).",
        wrong_rule="Ratios are reported as fractions between 0 and 1.",
        in_one_line_rule=False,
        check=sqlcheck.check_percent_scale,
        corrupt=((r"100\.0 \* ", "1.0 * "), (r" AS (\w+?)_pct", r" AS \1")),
    ),
    Convention(
        id="pct_round",
        layer="output",
        description="Percentages are rounded to one decimal: ROUND(..., 1).",
        rule="Percentages are rounded to one decimal.",
        wrong_rule="Percentages are not rounded.",
        in_one_line_rule=False,
        check=partial(sqlcheck.check_rounded, decimals=1),
        corrupt=((r"(?s)ROUND\((.*), 1\) AS", r"\1 AS"),),
    ),
)

ONE_LINE_RULE = "Net revenue = amount minus refunds, for completed orders."
ONE_LINE_WRONG = {
    "net_refunds": "Net revenue = amount (refunds are not subtracted), for completed orders.",
    "completed_only": "Net revenue = amount minus refunds, for delivered orders.",
}

# ---------------------------------------------------------------------------------------------
# example pool: every query follows every convention that applies to it

EXAMPLES = (
    Item(
        "ex_month",
        "Net revenue per month.",
        """\
SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') AS month,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "ex_country",
        "Net revenue by country.",
        """\
SELECT country,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY country
ORDER BY net_revenue DESC;""",
    ),
    Item(
        "ex_top_customers",
        "Top 10 customers by net revenue.",
        """\
SELECT customer_id,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY customer_id
ORDER BY net_revenue DESC
LIMIT 10;""",
    ),
    Item(
        "ex_aov_segment",
        "Average order value by customer segment.",
        """\
SELECT c.segment,
       SUM(o.amount - o.refunded) / 100.0 / COUNT(*) AS avg_order_value
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.channel <> 'test'
GROUP BY c.segment;""",
    ),
    Item(
        "ex_growth",
        "Month-over-month growth of net revenue.",
        """\
SELECT month,
       ROUND(100.0 * (net / LAG(net) OVER (ORDER BY month) - 1), 1) AS growth_pct
FROM (SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') AS month,
             SUM(amount - refunded) / 100.0 AS net
      FROM orders
      WHERE status = 'completed' AND channel <> 'test'
      GROUP BY 1) AS monthly
ORDER BY month;""",
    ),
    # --- rest of the pool (not in the standard five) ---
    Item(
        "ex_orders_day",
        "Number of orders per day.",
        """\
SELECT (ordered_at AT TIME ZONE 'Europe/Paris')::date AS day,
       COUNT(*) AS orders
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "ex_refund_rate_month",
        "Refund rate per month.",
        """\
SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') AS month,
       ROUND(100.0 * SUM(refunded) / SUM(amount), 1) AS refund_rate_pct
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "ex_weekday",
        "Net revenue by day of the week.",
        """\
SELECT EXTRACT(isodow FROM ordered_at AT TIME ZONE 'Europe/Paris') AS weekday,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "ex_fr_since_2024",
        "Net revenue in France since 2024.",
        """\
SELECT SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
  AND country = 'FR'
  AND ordered_at AT TIME ZONE 'Europe/Paris' >= '2024-01-01';""",
    ),
)

# five figures that all rely on net revenue; n = 1 and n = 3 contexts are prefixes of this list
DEFAULT_EXAMPLES = ("ex_month", "ex_country", "ex_top_customers", "ex_aov_segment", "ex_growth")

# ---------------------------------------------------------------------------------------------
# test questions with reference SQL (house style, all conventions followed)

QUESTIONS = (
    # --- far: not a near-copy of any of the standard five ---
    Item(
        "q_year",
        "What was our net revenue in 2025?",
        """\
SELECT SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
  AND EXTRACT(year FROM ordered_at AT TIME ZONE 'Europe/Paris') = 2025;""",
    ),
    Item(
        "q_hour",
        "Net revenue per hour of the day.",
        """\
SELECT EXTRACT(hour FROM ordered_at AT TIME ZONE 'Europe/Paris') AS hour,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "q_segment",
        "Net revenue by customer segment.",
        """\
SELECT c.segment,
       SUM(o.amount - o.refunded) / 100.0 AS net_revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.channel <> 'test'
GROUP BY c.segment;""",
    ),
    Item(
        "q_best_days",
        "Which 5 days had the highest net revenue?",
        """\
SELECT (ordered_at AT TIME ZONE 'Europe/Paris')::date AS day,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY net_revenue DESC
LIMIT 5;""",
    ),
    Item(
        "q_per_customer",
        "Average net revenue per customer.",
        """\
SELECT SUM(amount - refunded) / 100.0 / COUNT(DISTINCT customer_id) AS net_revenue_per_customer
FROM orders
WHERE status = 'completed' AND channel <> 'test';""",
    ),
    Item(
        "q_fr_business",
        "Net revenue from business customers in France.",
        """\
SELECT SUM(o.amount - o.refunded) / 100.0 AS net_revenue
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.channel <> 'test'
  AND c.segment = 'business' AND o.country = 'FR';""",
    ),
    Item(
        "q_business_share",
        "What share of net revenue comes from business customers?",
        """\
SELECT ROUND(100.0 * SUM(o.amount - o.refunded) FILTER (WHERE c.segment = 'business')
             / SUM(o.amount - o.refunded), 1) AS business_share_pct
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.channel <> 'test';""",
    ),
    Item(
        "q_refund_rate",
        "Refund rate by country.",
        """\
SELECT country,
       ROUND(100.0 * SUM(refunded) / SUM(amount), 1) AS refund_rate_pct
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY country;""",
    ),
    Item(
        "q_orders_week",
        "Number of orders per week.",
        """\
SELECT date_trunc('week', ordered_at AT TIME ZONE 'Europe/Paris') AS week,
       COUNT(*) AS orders
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "q_business_orders",
        "What proportion of orders come from business customers?",
        """\
SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE c.segment = 'business') / COUNT(*), 1) AS business_orders_pct
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.channel <> 'test';""",
    ),
    Item(
        "q_aov_month",
        "Average order value per month.",
        """\
SELECT date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') AS month,
       SUM(amount - refunded) / 100.0 / COUNT(*) AS avg_order_value
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
    ),
    Item(
        "q_median",
        "Median order value.",
        """\
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY (amount - refunded) / 100.0) AS median_order_value
FROM orders
WHERE status = 'completed' AND channel <> 'test';""",
    ),
    Item(
        "q_december_share",
        "What share of 2025 net revenue was made in December?",
        """\
SELECT ROUND(100.0 * SUM(amount - refunded)
               FILTER (WHERE EXTRACT(month FROM ordered_at AT TIME ZONE 'Europe/Paris') = 12)
             / SUM(amount - refunded), 1) AS december_share_pct
FROM orders
WHERE status = 'completed' AND channel <> 'test'
  AND EXTRACT(year FROM ordered_at AT TIME ZONE 'Europe/Paris') = 2025;""",
    ),
    Item(
        "q_customers_march",
        "How many customers placed an order in March 2025?",
        """\
SELECT COUNT(DISTINCT customer_id) AS customers
FROM orders
WHERE status = 'completed' AND channel <> 'test'
  AND date_trunc('month', ordered_at AT TIME ZONE 'Europe/Paris') = '2025-03-01';""",
    ),
    # --- near: each written to resemble one of the standard five ---
    Item(
        "n_week",
        "Net revenue per week.",
        """\
SELECT date_trunc('week', ordered_at AT TIME ZONE 'Europe/Paris') AS week,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY 1
ORDER BY 1;""",
        near="ex_month",
    ),
    Item(
        "n_country_top3",
        "Net revenue by country, top 3 only.",
        """\
SELECT country,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY country
ORDER BY net_revenue DESC
LIMIT 3;""",
        near="ex_country",
    ),
    Item(
        "n_top5_customers",
        "Top 5 customers by net revenue.",
        """\
SELECT customer_id,
       SUM(amount - refunded) / 100.0 AS net_revenue
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY customer_id
ORDER BY net_revenue DESC
LIMIT 5;""",
        near="ex_top_customers",
    ),
    Item(
        "n_aov_country",
        "Average order value by country.",
        """\
SELECT country,
       SUM(amount - refunded) / 100.0 / COUNT(*) AS avg_order_value
FROM orders
WHERE status = 'completed' AND channel <> 'test'
GROUP BY country;""",
        near="ex_aov_segment",
    ),
    Item(
        "n_growth_quarter",
        "Quarter-over-quarter growth of net revenue.",
        """\
SELECT quarter,
       ROUND(100.0 * (net / LAG(net) OVER (ORDER BY quarter) - 1), 1) AS growth_pct
FROM (SELECT date_trunc('quarter', ordered_at AT TIME ZONE 'Europe/Paris') AS quarter,
             SUM(amount - refunded) / 100.0 AS net
      FROM orders
      WHERE status = 'completed' AND channel <> 'test'
      GROUP BY 1) AS quarterly
ORDER BY quarter;""",
        near="ex_growth",
    ),
)

SCENARIO = Scenario(
    id="shop",
    title="Shop warehouse: net revenue",
    dialect="postgres",
    schema=SCHEMA,
    conventions=CONVENTIONS,
    one_line_rule=ONE_LINE_RULE,
    one_line_wrong=ONE_LINE_WRONG,
    examples=EXAMPLES,
    default_examples=DEFAULT_EXAMPLES,
    questions=QUESTIONS,
)
