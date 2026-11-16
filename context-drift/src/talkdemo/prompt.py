"""Prompt assembly: the ONLY place that turns (scenario, context spec, question) into the system
and user text a model receives. Every runner, and the white-box side, must go through
``build_prompt`` so all models read byte-identical text.

A context is described by a ``ContextSpec``:

* ``rules``        "none" | "one_line" (the one-line rule only) | "all" (one sentence per convention)
* ``wrong_rules``  conventions whose rule sentence is replaced by the wrong one
* ``examples``     the examples shown, in display order, each with the conventions corrupted in it

The helpers at the bottom build the specs of the standard conditions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from .scenario import Item, Scenario

SYSTEM_PROMPT = (
    "You write PostgreSQL queries for the database described by the user. "
    "Answer the last question with one SQL query and nothing else: "
    "no explanation, no comments, no code fence."
)

RULE_MODES = ("none", "one_line", "all")


@dataclass(frozen=True)
class ExampleUse:
    id: str
    corrupt: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ContextSpec:
    rules: str = "none"
    wrong_rules: frozenset[str] = frozenset()
    examples: tuple[ExampleUse, ...] = ()

    def __post_init__(self) -> None:
        if self.rules not in RULE_MODES:
            raise ValueError(f"rules must be one of {RULE_MODES}, got {self.rules!r}")
        if self.wrong_rules and self.rules == "none":
            raise ValueError("wrong_rules needs rules")

    def to_json(self) -> dict:
        return {
            "rules": self.rules,
            "wrong_rules": sorted(self.wrong_rules),
            "examples": [{"id": e.id, "corrupt": sorted(e.corrupt)} for e in self.examples],
        }

    @classmethod
    def from_json(cls, data: dict) -> "ContextSpec":
        """Inverse of ``to_json``: rebuilds the spec stored with every result record."""
        return cls(
            rules=data.get("rules", "none"),
            wrong_rules=frozenset(data.get("wrong_rules", ())),
            examples=tuple(ExampleUse(e["id"], frozenset(e.get("corrupt", ()))) for e in data.get("examples", ())),
        )


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    spec: ContextSpec = field(default_factory=ContextSpec, compare=False)

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user}]


# ---------------------------------------------------------------------------------------------
# rendering


def rule_lines(scenario: Scenario, spec: ContextSpec) -> list[str]:
    """The rule sentences a spec states, in convention order."""
    if spec.rules == "none":
        return []
    if spec.rules == "one_line":
        wrong = [cid for cid in scenario.convention_ids if cid in spec.wrong_rules]
        if not wrong:
            return [scenario.one_line_rule]
        if len(wrong) > 1 or wrong[0] not in scenario.one_line_wrong:
            raise ValueError(f"the one-line rule has no wrong variant for {wrong}")
        return [scenario.one_line_wrong[wrong[0]]]
    unknown = spec.wrong_rules - set(scenario.convention_ids)
    if unknown:
        raise ValueError(f"unknown conventions in wrong_rules: {sorted(unknown)}")
    return [c.wrong_rule if c.id in spec.wrong_rules else c.rule for c in scenario.conventions]


def example_pairs(scenario: Scenario, spec: ContextSpec) -> list[tuple[str, str]]:
    """(question, SQL) of every example a spec shows, corruptions applied."""
    return [
        (scenario.example(u.id).question, scenario.variant(scenario.example(u.id), u.corrupt)) for u in spec.examples
    ]


def build_context(scenario: Scenario, spec: ContextSpec) -> str:
    """Schema, then the rules and the examples the spec asks for. No question."""
    parts = [f"Database schema (PostgreSQL):\n\n{scenario.schema}"]
    rules = rule_lines(scenario, spec)
    if rules:
        parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
    pairs = example_pairs(scenario, spec)
    if pairs:
        parts.append("Examples:")
        parts += [f"Question: {q}\nSQL:\n{sql}" for q, sql in pairs]
    return "\n\n".join(parts)


def build_prompt(scenario: Scenario, spec: ContextSpec, question: str | Item) -> Prompt:
    """System and user text for one question under one context."""
    text = question.question if isinstance(question, Item) else question
    user = f"{build_context(scenario, spec)}\n\nQuestion: {text}\nSQL:"
    return Prompt(system=SYSTEM_PROMPT, user=user, spec=spec)


# ---------------------------------------------------------------------------------------------
# standard conditions


def display_order(scenario: Scenario, question: Item, sample: int, ids: Iterable[str]) -> list[str]:
    """Display order of the examples for one (question, sample): a seeded shuffle of the standard
    five, restricted to ``ids``. It does not depend on the condition, so two conditions differ
    only by what they change (paired design)."""
    order = list(scenario.default_examples)
    random.Random(f"order|{scenario.shuffle_key}|{question.id}|{sample}").shuffle(order)
    wanted = list(ids)
    return [e for e in order if e in wanted] + [e for e in wanted if e not in order]


def corruption_order(scenario: Scenario, question: Item, sample: int) -> list[str]:
    """Which examples get corrupted first for one (question, sample): k corrupted examples are
    the first k of this list, so the k-of-5 conditions are nested."""
    order = list(scenario.default_examples)
    random.Random(f"corrupt|{scenario.shuffle_key}|{question.id}|{sample}").shuffle(order)
    return order


def examples_spec(
    scenario: Scenario,
    question: Item,
    sample: int = 0,
    *,
    n: int = 5,
    rules: str = "none",
    wrong_rules: Iterable[str] = (),
    convention: str | None = None,
    k: int = 0,
    pick: str = "random",
    shuffle: bool = True,
) -> ContextSpec:
    """The first ``n`` standard examples, ``k`` of them with ``convention`` corrupted.

    ``pick`` chooses the corrupted ones: "random" (the first k of ``corruption_order``),
    "nearest" (the example closest to the question, k = 1) or "far" (a random one that is not
    the closest, k = 1). ``shuffle=False`` keeps the canonical order (slides, white-box runs).
    """
    ids = list(scenario.default_examples[:n])
    shown = display_order(scenario, question, sample, ids) if shuffle else ids
    corrupted: list[str] = []
    if k:
        if convention is None:
            raise ValueError("k > 0 needs a convention")
        nearest = scenario.nearest_example(question, ids)
        ranked = [e for e in corruption_order(scenario, question, sample) if e in ids]
        if pick == "random":
            corrupted = ranked[:k]
        elif pick == "nearest" and k == 1:
            corrupted = [nearest]
        elif pick == "far" and k == 1:
            corrupted = [e for e in ranked if e != nearest][:1]
        else:
            raise ValueError(f"unsupported pick={pick!r} with k={k}")
        if len(corrupted) != k:
            raise ValueError(f"cannot corrupt {k} of {len(ids)} examples")
    uses = tuple(ExampleUse(e, frozenset([convention]) if e in corrupted else frozenset()) for e in shown)
    return ContextSpec(rules=rules, wrong_rules=frozenset(wrong_rules), examples=uses)


def rules_spec(rules: str, wrong_rules: Iterable[str] = ()) -> ContextSpec:
    return ContextSpec(rules=rules, wrong_rules=frozenset(wrong_rules))
