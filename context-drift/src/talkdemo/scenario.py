"""Scenario model: schema, conventions (rule sentence + checker + corruption recipe), example
pool, test questions with reference SQL, and the derived decision points.

A *corruption* is a minimal text edit that breaks one convention and leaves the others intact.
It is written once per convention as regex steps (``Convention.corrupt``) and applies to every
example and reference SQL; an item overrides it where its SQL needs a specific edit
(``Item.corrupt``). Several conventions can be corrupted at once: the edits are applied in the
scenario's convention order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from sqlglot import exp

from . import sqlcheck
from .sqlcheck import FOLLOWS, NA, UNPARSED, VIOLATES, Check

# one corruption = regex substitutions, all applied in order; at least one must match
Steps = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Convention:
    id: str
    layer: str  # "metric" | "date" | "output"
    description: str  # plain description
    rule: str  # the sentence a rule list would use to state it
    wrong_rule: str  # the same sentence stating the wrong computation
    in_one_line_rule: bool  # stated by the scenario's one-line rule
    check: Callable[[exp.Expression], Check] = field(repr=False, compare=False)
    corrupt: Steps = ()


@dataclass(frozen=True)
class Item:
    """A question with its SQL: a context example or a test question (reference SQL)."""

    id: str
    question: str
    sql: str
    corrupt: dict[str, Steps] = field(default_factory=dict)  # per-convention override
    near: str | None = None  # test questions: id of the example it was written to resemble


@dataclass(frozen=True)
class DecisionPoint:
    """Where the reference SQL commits to a convention: after ``prefix`` the next token differs
    between the SQL that follows the convention and the minimally edited SQL that violates it."""

    question_id: str
    convention_id: str
    prefix: str
    follows: str  # rest of the reference SQL
    violates: str  # rest of the corrupted reference SQL
    follows_head: str  # first whitespace-delimited chunk of each continuation,
    violates_head: str  # with its leading whitespace


def _apply(sql: str, steps: Steps, what: str) -> str:
    out, hits = sql, 0
    for pattern, repl in steps:
        out, n = re.subn(pattern, repl, out)
        hits += n
    if not hits:
        raise ValueError(f"corruption matched nothing: {what}")
    return out


_WORD = re.compile(r"\w")
_APPLIES: dict[tuple[str, str], tuple[str, ...]] = {}  # (scenario id, sql) -> conventions followed


def _split_point(a: str, b: str) -> int:
    """Length of the common prefix of a and b, moved back to a token boundary (never inside a
    word or number, never after trailing whitespace)."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    while i > 0 and _WORD.match(a[i - 1]) and any(_WORD.match(s[i]) for s in (a, b) if i < len(s)):
        i -= 1
    while i > 0 and a[i - 1].isspace():
        i -= 1
    return i


def _head(rest: str) -> str:
    m = re.match(r"\s*\S+", rest)
    return m.group(0) if m else rest


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    dialect: str
    schema: str
    conventions: tuple[Convention, ...]
    one_line_rule: str
    one_line_wrong: dict[str, str]  # convention id -> the one-line rule restated wrongly
    examples: tuple[Item, ...]  # the pool
    default_examples: tuple[str, ...]  # the standard context, canonical order (n=1, 3: prefixes)
    questions: tuple[Item, ...]
    # seeds the per-sample example order and corruption picks; two scenarios with the same key
    # shuffle alike, so their results are paired question by question (default: the id)
    shuffle_key: str = ""

    def __post_init__(self) -> None:
        if not self.shuffle_key:
            object.__setattr__(self, "shuffle_key", self.id)

    # -- lookups -------------------------------------------------------------------------
    def convention(self, cid: str) -> Convention:
        return next(c for c in self.conventions if c.id == cid)

    def example(self, eid: str) -> Item:
        return next(e for e in self.examples if e.id == eid)

    def question(self, qid: str) -> Item:
        return next(q for q in self.questions if q.id == qid)

    @property
    def convention_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.conventions)

    @property
    def far_questions(self) -> tuple[Item, ...]:
        return tuple(q for q in self.questions if q.near is None)

    @property
    def near_questions(self) -> tuple[Item, ...]:
        return tuple(q for q in self.questions if q.near is not None)

    # -- checking ------------------------------------------------------------------------
    def check_sql(self, sql: str) -> dict[str, Check]:
        """Status of every convention on one SQL string."""
        tree = sqlcheck.parse(sql)
        if tree is None:
            return {c.id: Check(UNPARSED) for c in self.conventions}
        return {c.id: c.check(tree) for c in self.conventions}

    def applies(self, item: Item) -> tuple[str, ...]:
        """Conventions an item is about: those its own (clean) SQL follows."""
        key = (self.id, item.sql)
        if key not in _APPLIES:
            checks = self.check_sql(item.sql)
            _APPLIES[key] = tuple(cid for cid, chk in checks.items() if chk.status == FOLLOWS)
        return _APPLIES[key]

    def score(self, question: Item, reply: str) -> dict[str, Check]:
        """Status of the conventions that apply to ``question`` on a model reply."""
        checks = self.check_sql(sqlcheck.extract_sql(reply))
        return {cid: checks[cid] for cid in self.applies(question)}

    # -- corruption ----------------------------------------------------------------------
    def variant(self, item: Item, corrupt: Iterable[str] = ()) -> str:
        """SQL of ``item`` with the given conventions broken (minimal edits)."""
        wanted = set(corrupt)
        unknown = wanted - set(self.applies(item))
        if unknown:
            raise ValueError(f"{item.id}: cannot corrupt {sorted(unknown)} (does not apply)")
        sql = item.sql
        for conv in self.conventions:
            if conv.id in wanted:
                sql = _apply(sql, item.corrupt.get(conv.id, conv.corrupt), f"{item.id}/{conv.id}")
        return sql

    # -- white-box hand-off ---------------------------------------------------------------
    def decision_point(self, question: Item, cid: str) -> DecisionPoint:
        clean, broken = question.sql, self.variant(question, [cid])
        i = _split_point(clean, broken)
        return DecisionPoint(
            question_id=question.id,
            convention_id=cid,
            prefix=clean[:i],
            follows=clean[i:],
            violates=broken[i:],
            follows_head=_head(clean[i:]),
            violates_head=_head(broken[i:]),
        )

    def decision_points(self) -> list[DecisionPoint]:
        return [self.decision_point(q, cid) for q in self.questions for cid in self.applies(q)]

    # -- nearest example -------------------------------------------------------------------
    def nearest_example(self, question: Item, among: Iterable[str] | None = None) -> str:
        """Example closest to the question: the one it was written to resemble (``near``), else
        the highest word overlap (Jaccard) between the two question texts; ties: first listed."""
        ids = list(among if among is not None else self.default_examples)
        if question.near in ids:
            return question.near
        return max(ids, key=lambda eid: (similarity(question.question, self.example(eid).question), -ids.index(eid)))

    # -- export ----------------------------------------------------------------------------
    def to_json(self) -> dict:
        def item(it: Item) -> dict:
            applies = self.applies(it)
            out = {
                "id": it.id,
                "question": it.question,
                "sql": it.sql,
                "applies": list(applies),
                "corrupted": {cid: self.variant(it, [cid]) for cid in applies},
            }
            return out

        questions = []
        for q in self.questions:
            d = item(q)
            d["kind"] = "near" if q.near else "far"
            d["near"] = q.near
            d["nearest_example"] = self.nearest_example(q)
            d["similarity"] = {
                eid: round(similarity(q.question, self.example(eid).question), 3) for eid in self.default_examples
            }
            d["decision_points"] = {
                cid: {
                    k: getattr(self.decision_point(q, cid), k)
                    for k in ("prefix", "follows", "violates", "follows_head", "violates_head")
                }
                for cid in self.applies(q)
            }
            questions.append(d)
        return {
            "id": self.id,
            "title": self.title,
            "dialect": self.dialect,
            "schema": self.schema,
            "conventions": [
                {
                    "id": c.id,
                    "layer": c.layer,
                    "description": c.description,
                    "rule": c.rule,
                    "wrong_rule": c.wrong_rule,
                    "in_one_line_rule": c.in_one_line_rule,
                    "corrupt_regex": [list(s) for s in c.corrupt],
                }
                for c in self.conventions
            ],
            "one_line_rule": self.one_line_rule,
            "one_line_wrong": self.one_line_wrong,
            "default_examples": list(self.default_examples),
            "examples": [item(e) for e in self.examples],
            "questions": questions,
        }


_STOP = frozenset("a an the of in on for to from by per our was is are were what which how many had did do and".split())


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP)


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of the content words of two questions."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


__all__ = [
    "Check",
    "Convention",
    "DecisionPoint",
    "Item",
    "Scenario",
    "Steps",
    "similarity",
    "FOLLOWS",
    "VIOLATES",
    "NA",
    "UNPARSED",
]
