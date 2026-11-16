"""SQL extraction and convention checks on the sqlglot AST (PostgreSQL).

Every check takes a parsed statement and returns a ``Check``: ``follows``, ``violates`` or ``na``
(the SQL has nothing the convention could apply to), plus a small ``detail`` dict. The checks are
generic (column names, literals and zones are arguments); a scenario binds them to its schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import sqlglot
from sqlglot import exp

FOLLOWS, VIOLATES, NA, UNPARSED = "follows", "violates", "na", "unparsed"
DIALECT = "postgres"


@dataclass(frozen=True)
class Check:
    status: str
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# extraction / parsing

_FENCE = re.compile(r"```[a-zA-Z]*[ \t]*\n(.*?)```", re.S)
_START = re.compile(r"^[ \t]*(WITH|SELECT)\b", re.I | re.M)


def extract_sql(reply: str) -> str:
    """The SQL of a model reply: the first fenced block if there is one, else the text from the
    first line that starts with SELECT / WITH; first statement only."""
    m = _FENCE.search(reply)
    text = m.group(1) if m else reply
    text = re.sub(r"^\s*SQL:\s*", "", text, flags=re.I)
    m = _START.search(text)
    if m:
        text = text[m.start() :]
    head, sep, _ = text.partition(";")
    return (head + sep).strip()


def parse(sql: str) -> exp.Expression | None:
    """Parse one statement; None when it is not a parseable query."""
    try:
        trees = [t for t in sqlglot.parse(sql, read=DIALECT) if t is not None]
    except Exception:  # sqlglot raises ParseError / TokenError, both mean "not SQL we can read"
        return None
    if not trees or trees[0].find(exp.Select) is None:
        return None
    return trees[0]


# ---------------------------------------------------------------------------------------------
# small AST helpers


def _columns(node: exp.Expression, name: str) -> list[exp.Column]:
    return [c for c in node.find_all(exp.Column) if c.name.lower() == name]


def _has_col(node: exp.Expression | None, name: str) -> bool:
    return node is not None and bool(_columns(node, name))


def _unwrap(node: exp.Expression | None) -> exp.Expression | None:
    while isinstance(node, (exp.Paren, exp.Cast, exp.TryCast)):
        node = node.this
    return node


def _num(node: exp.Expression | None) -> float | None:
    node = _unwrap(node)
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return float(node.this)
        except ValueError:
            return None
    return None


def _str(node: exp.Expression | None) -> str | None:
    node = _unwrap(node)
    return node.this if isinstance(node, exp.Literal) and node.is_string else None


def _parent(node: exp.Expression) -> exp.Expression | None:
    """Parent, skipping parentheses."""
    p = node.parent
    while isinstance(p, exp.Paren):
        p = p.parent
    return p


def _direct_tables(sel: exp.Select) -> set[str]:
    frm = sel.args.get("from_") or sel.args.get("from")
    nodes = [frm.this] if frm is not None else []
    nodes += [j.this for j in sel.args.get("joins") or []]
    return {n.name.lower() for n in nodes if isinstance(n, exp.Table)}


def _scope_nodes(sel: exp.Select, table: str) -> Iterator[exp.Expression]:
    """Nodes of ``sel`` that are not inside a nested SELECT reading ``table`` itself."""

    def prune(n: exp.Expression) -> bool:
        return isinstance(n, exp.Select) and n is not sel and table in _direct_tables(n)

    for n in sel.walk(prune=prune):
        if not prune(n):
            yield n


def _enclosing_alias(sel: exp.Select) -> str | None:
    """Name under which the rest of the query reads ``sel`` (CTE or derived-table alias)."""
    p = sel.parent
    while p is not None and not isinstance(p, (exp.CTE, exp.Subquery, exp.Select)):
        p = p.parent
    if isinstance(p, (exp.CTE, exp.Subquery)) and p.alias:
        return p.alias.lower()
    return None


def _consumers(sel: exp.Select, tree: exp.Expression) -> list[exp.Select]:
    """SELECTs that read the rows of ``sel``: the enclosing ones and, transitively, those that
    read the CTE / derived table it sits in. A filter there filters the rows of ``sel`` too."""
    out: list[exp.Select] = []
    seen = {id(sel)}
    todo = [sel]
    while todo:
        cur = todo.pop()
        p = cur.parent
        while p is not None:
            if isinstance(p, exp.Select) and id(p) not in seen:
                seen.add(id(p))
                out.append(p)
                todo.append(p)
            p = p.parent
        alias = _enclosing_alias(cur)
        if alias:
            for other in tree.find_all(exp.Select):
                if id(other) not in seen and alias in _direct_tables(other):
                    seen.add(id(other))
                    out.append(other)
                    todo.append(other)
    return out


def _each_scope(tree: exp.Expression, table: str, holds: Callable[[exp.Expression], bool]) -> Check:
    """``follows`` when every SELECT reading ``table`` (or a consumer of it) has a node on which
    ``holds`` is true; ``na`` when nothing reads ``table``."""
    scopes = [s for s in tree.find_all(exp.Select) if table in _direct_tables(s)]
    if not scopes:
        return Check(NA)
    ok = 0
    for sel in scopes:
        if any(any(holds(n) for n in _scope_nodes(s, table)) for s in [sel, *_consumers(sel, tree)]):
            ok += 1
    return Check(FOLLOWS if ok == len(scopes) else VIOLATES, {"scopes": len(scopes), "filtered": ok})


# ---------------------------------------------------------------------------------------------
# checks


def check_subtracted(tree: exp.Expression, base: str, minus: str) -> Check:
    """``minus`` is subtracted wherever the query uses ``base`` (net = base - minus)."""
    if not _columns(tree, base):
        return Check(NA)
    subtracted = any(_has_col(sub.expression, minus) for sub in tree.find_all(exp.Sub))
    return Check(FOLLOWS if subtracted else VIOLATES)


def _compared_value(n: exp.Expression, column: str) -> str | None:
    """The string literal a binary predicate compares ``column`` with, if that is its shape."""
    for a, b in ((n.this, n.expression), (n.expression, n.this)):
        if _has_col(a, column) and _str(b) is not None:
            return _str(b)
    return None


def _alternative(n: exp.Expression, column: str) -> bool:
    """The predicate sits in an OR that lets other rows through (``OR column IS NULL`` does not)."""
    node: exp.Expression = n
    p = _parent(node)
    while isinstance(p, (exp.Or, exp.And, exp.Not)):
        if isinstance(p, exp.Or):
            other = _unwrap(p.expression if _unwrap(p.this) is node else p.this)
            null_test = (
                isinstance(other, exp.Is) and _has_col(other.this, column) and isinstance(other.expression, exp.Null)
            )
            if not null_test:
                return True
        node, p = p, _parent(p)
    return False


def check_equals_filter(tree: exp.Expression, table: str, column: str, value: str) -> Check:
    """Rows of ``table`` are restricted to ``column = value`` (exactly that value)."""

    def holds(n: exp.Expression) -> bool:
        if isinstance(_parent(n), exp.Not) or _alternative(n, column):
            return False
        if isinstance(n, (exp.EQ, exp.Like, exp.ILike)):
            return _compared_value(n, column) == value
        if isinstance(n, exp.In) and _has_col(n.this, column):
            return [_str(e) for e in n.expressions] == [value]
        return False

    check = _each_scope(tree, table, holds)
    # which values the query does filter on, to tell "another value" from "no filter"
    values = {_compared_value(n, column) for n in tree.find_all(exp.EQ, exp.NEQ, exp.Like, exp.ILike)}
    for n in tree.find_all(exp.In):
        if _has_col(n.this, column):
            values |= {_str(e) for e in n.expressions}
    return Check(check.status, check.detail | {"values": sorted(v for v in values if v is not None)})


def check_value_excluded(tree: exp.Expression, table: str, column: str, value: str) -> Check:
    """Rows of ``table`` with ``column = value`` are filtered out: ``column <> value``,
    ``NOT IN (.., value, ..)``, ``IS DISTINCT FROM``, or a positive filter on other values."""

    def holds(n: exp.Expression) -> bool:
        if _alternative(n, column):
            return False
        negated = isinstance(_parent(n), exp.Not)
        if isinstance(n, (exp.NEQ, exp.NullSafeNEQ)):
            return _compared_value(n, column) == value and not negated
        if isinstance(n, (exp.EQ, exp.Like, exp.ILike)):
            other = _compared_value(n, column)
            return other is not None and (other == value) == negated
        if isinstance(n, exp.In) and _has_col(n.this, column):
            return (value in [_str(e) for e in n.expressions]) == negated
        return False

    return _each_scope(tree, table, holds)


_FALSE = {"false", "f", "no", "n", "0"}
_TRUE = {"true", "t", "yes", "y", "1"}


def _bool(node: exp.Expression | None) -> bool | None:
    node = _unwrap(node)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal):
        text = str(node.this).lower()
        return False if text in _FALSE else True if text in _TRUE else None
    return None


def check_flag_excluded(tree: exp.Expression, table: str, column: str) -> Check:
    """Rows of ``table`` whose boolean ``column`` is true are filtered out."""

    def holds(n: exp.Expression) -> bool:
        if not (isinstance(n, exp.Column) and n.name.lower() == column):
            return False
        node: exp.Expression = n
        while isinstance(node.parent, (exp.Paren, exp.Coalesce)):
            node = node.parent
        p = node.parent
        if isinstance(p, exp.Not):
            return True
        if isinstance(p, (exp.EQ, exp.Is, exp.NEQ, exp.NullSafeNEQ)):
            other = p.expression if p.this is node else p.this
            value = _bool(other)
            if value is None:
                return False
            excluded = value is False
            if isinstance(p, (exp.NEQ, exp.NullSafeNEQ)):
                excluded = not excluded
            if isinstance(_parent(p), exp.Not):
                excluded = not excluded
            return excluded
        return False

    return _each_scope(tree, table, holds)


def check_scaled(tree: exp.Expression, columns: Iterable[str], divisor: float) -> Check:
    """Amounts built from ``columns`` are divided by ``divisor`` (cents to currency units)."""
    if not any(_columns(tree, c) for c in columns):
        return Check(NA)
    for div in tree.find_all(exp.Div):
        v = _num(div.expression)
        if v is not None and abs(v - divisor) < 1e-9:
            return Check(FOLLOWS)
    for mul in tree.find_all(exp.Mul):
        for side in (mul.this, mul.expression):
            v = _num(side)
            if v is not None and abs(v * divisor - 1) < 1e-9:
                return Check(FOLLOWS)
    return Check(VIOLATES)


def _zone_in(node: exp.Expression | None, zone: str) -> bool:
    """``node`` mentions the zone: AT TIME ZONE 'zone', timezone('zone', ..) or a literal ending
    with it ('2025-01-01 Europe/Paris')."""
    if node is None:
        return False
    z = zone.lower()
    for n in node.walk():
        if isinstance(n, exp.AtTimeZone) and (_str(n.args.get("zone")) or "").lower() == z:
            return True
        if isinstance(n, exp.Literal) and n.is_string and z in str(n.this).lower():
            return True
    return False


_COMPARISONS = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between)


def _converted(col: exp.Column, zone: str) -> bool:
    z = zone.lower()
    node: exp.Expression = col
    p = node.parent
    while p is not None and not isinstance(p, (exp.Select, exp.Subquery)):
        if isinstance(p, exp.AtTimeZone) and (_str(p.args.get("zone")) or "").lower() == z:
            return True
        if isinstance(p, (exp.TimestampTrunc, exp.DateTrunc)) and (_str(p.args.get("zone")) or "").lower() == z:
            return True
        if isinstance(p, exp.Anonymous) and str(p.this).lower() == "timezone":
            if any((_str(a) or "").lower() == z for a in p.expressions):
                return True
        if isinstance(p, _COMPARISONS):
            # an instant compared with a boundary expressed in the zone
            others = [a for a in p.args.values() if isinstance(a, exp.Expression) and a is not node]
            if any(_zone_in(o, zone) for o in others):
                return True
        node, p = p, p.parent
    return False


def check_timezone(tree: exp.Expression, table: str, column: str, zone: str) -> Check:
    """Every use of the timestamp ``column`` is converted to ``zone`` (strict: all uses)."""
    cols = _columns(tree, column)
    redefined = any(a.alias.lower() == column and _zone_in(a.this, zone) for a in tree.find_all(exp.Alias))
    if redefined:  # a CTE re-exposes the converted value under the same name: count raw uses only
        cols = [c for c in cols if (s := c.find_ancestor(exp.Select)) is not None and table in _direct_tables(s)]
    if not cols:
        return Check(NA)
    done = sum(_converted(c, zone) for c in cols)
    return Check(FOLLOWS if done == len(cols) else VIOLATES, {"uses": len(cols), "converted": done})


def check_percent_scale(tree: exp.Expression) -> Check:
    """A ratio is reported on a 0-100 scale: something is multiplied by 100."""
    scaled = any(isinstance(n, exp.Mul) and any(_num(s) == 100 for s in (n.this, n.expression)) for n in tree.walk())
    return Check(FOLLOWS if scaled else VIOLATES)


def check_rounded(tree: exp.Expression, decimals: int) -> Check:
    """A value is rounded to exactly ``decimals`` decimals: ROUND(x, decimals)."""
    found = sorted({int(_num(r.args.get("decimals")) or 0) for r in tree.find_all(exp.Round)})
    return Check(FOLLOWS if decimals in found else VIOLATES, {"round_decimals": found})
