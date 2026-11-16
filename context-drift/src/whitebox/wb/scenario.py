"""Adapter over the behavioural package (`src/talkdemo`, `src/scenarios`).

Prompts are ALWAYS assembled by `talkdemo.prompt.build_prompt`, the one place that turns
(scenario, context spec, question) into the text a model reads: nothing is rebuilt here. This
module only (1) names the contexts each step needs as `talkdemo.prompt.ContextSpec` objects,
(2) says where the units of a built prompt are (schema, each rule sentence, each example's question
and SQL, the question), by searching their text in what the builder returned, and (3) turns the
scenario's decision points into the two short continuations that are scored.

Vocabulary: a convention is CLEAN in a rule or an example when it is the house one, BROKEN when
the rule states the wrong computation (`wrong_rules`) or the example's SQL was minimally edited to
violate it (`ExampleUse.corrupt`). The carriers of a convention are the standard examples whose SQL
exercises it (`Scenario.applies`): only those can be broken.
"""

from __future__ import annotations

import hashlib
import json
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

from wb import DEMO_DIR

import sys

if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from talkdemo import prompt as tdprompt  # noqa: E402  (the behavioural side's prompt builder)
from talkdemo.prompt import ContextSpec, ExampleUse  # noqa: E402
from talkdemo.scenario import Item  # noqa: E402
from talkdemo.scenario import Scenario as TDScenario  # noqa: E402
from talkdemo.sqlcheck import FOLLOWS, VIOLATES  # noqa: E402


@dataclass
class UnitText:
    """A context unit, located by its text: it appears verbatim in the built prompt, the units of
    one prompt in prompt order and without overlap."""

    id: str  # "schema", "rule.net_refunds", "ex2.q", "ex2.sql", "question"
    kind: str  # instructions | schema | rule | example_question | example_sql | question
    group: str  # "ex2" for both halves of example 2, else the id
    text: str
    label: str = ""  # for figures: "example 2 (ex_country)"


@dataclass
class Built:
    messages: list[dict]  # chat messages exactly as the builder's Prompt gives them
    units: list[UnitText]
    spec: ContextSpec


@dataclass
class Decision:
    convention: str
    prefix: str  # prefix of the reference SQL; the next token decides the convention
    follows: str  # short continuation that follows the house convention
    violates: str  # short continuation of the minimally corrupted SQL
    rest_follows: str = field(default="", repr=False)  # the scenario's full continuations
    rest_violates: str = field(default="", repr=False)


@dataclass
class Condition:
    """One column of the behavioural grid."""

    name: str
    spec: ContextSpec
    family: str  # none | rule_one_line | rules_all | examples | rules_examples | conflict
    about: list[str]  # the conventions the condition is about (the ones it can inform or contradict)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Demo:
    """One scenario, as the white-box steps read it."""

    def __init__(self, td: TDScenario, name: str, sources: list[Path]):
        self.td = td
        self.name = name
        self.conventions: list[str] = list(td.convention_ids)
        self.questions: list[Item] = list(td.questions)
        self.examples: list[str] = list(td.default_examples)  # canonical order, never shuffled here
        files = {str(p.relative_to(DEMO_DIR)): _sha(p) for p in sources}
        # Every output records which scenario text and builder produced its prompts. `files` and
        # `hash` are over the bytes of the scenario module and of the builder (a reformat changes
        # them); `content` is over what the model reads, the scenario's export (the same data as
        # scenarios/<name>.json) and the system prompt, so it survives a reformat of the code.
        export = json.dumps(td.to_json(), sort_keys=True, ensure_ascii=False) + tdprompt.SYSTEM_PROMPT
        self.version = {
            "files": {k: v[:16] for k, v in files.items()},
            "hash": hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()[:16],
            "content": hashlib.sha256(export.encode()).hexdigest()[:16],
        }

    # -- lookups
    def question(self, qid: str) -> Item:
        return self.td.question(qid)

    def applies(self, item: Item) -> list[str]:
        """Conventions a question (or example) is about, in scenario order."""
        return list(self.td.applies(item))

    def carriers(self, convention: str) -> list[str]:
        return [e for e in self.examples if convention in self.td.applies(self.td.example(e))]

    def in_one_line_rule(self) -> list[str]:
        return [c.id for c in self.td.conventions if c.in_one_line_rule]

    def near(self, question: Item) -> str | None:
        return question.near if question.near in self.examples else None

    def example_number(self, example_id: str) -> int:
        return self.examples.index(example_id) + 1

    # -- context specs
    def spec(self, *, rules: str = "none", wrong=(), examples=(), broken: dict[str, list[str]] | None = None):
        """`examples`: ids in display order; `broken`: example id -> conventions corrupted in it."""
        broken = broken or {}
        uses = tuple(ExampleUse(e, frozenset(broken.get(e, ()))) for e in examples)
        return ContextSpec(rules=rules, wrong_rules=frozenset(wrong), examples=uses)

    def grid_conditions(self) -> list[Condition]:
        """no context / the one-line rule / all rules / 1, 3, 5 examples / rules + examples / conflicts."""
        cv, ex = self.conventions, self.examples
        out = [
            Condition("none", self.spec(), "none", []),
            Condition("rule_one_line", self.spec(rules="one_line"), "rule_one_line", self.in_one_line_rule()),
            Condition("rules_all", self.spec(rules="all"), "rules_all", cv),
        ]
        for k in (1, 3, 5):
            if k <= len(ex):
                about = [c for c in cv if any(c in self.td.applies(self.td.example(e)) for e in ex[:k])]
                out.append(Condition(f"ex{k}", self.spec(examples=ex[:k]), "examples", about))
        out.append(Condition("rules+examples", self.spec(rules="all", examples=ex), "rules_examples", cv))
        for c in cv:
            car = self.carriers(c)
            if not car:
                continue
            out.append(  # the rules state the house convention, every example that shows it carries the other one
                Condition(
                    f"conflict:rule_clean+examples_broken:{c}",
                    self.spec(rules="all", examples=ex, broken={e: [c] for e in car}),
                    "conflict",
                    [c],
                )
            )
            out.append(  # the rule states the wrong computation, the examples carry the house one
                Condition(
                    f"conflict:rule_broken+examples_clean:{c}",
                    self.spec(rules="all", wrong=[c], examples=ex),
                    "conflict",
                    [c],
                )
            )
        for c in cv:  # the same two conflicts with the one-line rule, for the conventions it states
            car = self.carriers(c)
            if c not in self.td.one_line_wrong or not car:
                continue
            out.append(
                Condition(
                    f"conflict:one_line_clean+examples_broken:{c}",
                    self.spec(rules="one_line", examples=ex, broken={e: [c] for e in car}),
                    "conflict_one_line",
                    [c],
                )
            )
            out.append(
                Condition(
                    f"conflict:one_line_broken+examples_clean:{c}",
                    self.spec(rules="one_line", wrong=[c], examples=ex),
                    "conflict_one_line",
                    [c],
                )
            )
        return out

    def perturbations(self, convention: str) -> list[tuple[str, ContextSpec, dict]]:
        """ "Changed in one place" for one convention: (name, spec, facts).

        Rule state `0` absent (no rules), `+` clean (all rules), `-` broken (all rules, this one
        wrong). All standard examples are shown; a subset of the carriers is clean, the others have
        the convention broken. Without rules every subset is run (break-one, restore-one and k of n
        with every choice of examples); with rules, to keep the cost down, the sizes 0, 1, n - 1, n
        with every choice, and one nested chain in between. Plus each rule state with no example.
        `facts`: rule, clean (carrier numbers that are clean), n (number of carriers)."""
        car, ex = self.carriers(convention), self.examples
        num = {e: self.example_number(e) for e in car}
        out = []
        for rk in ("0", "+", "-"):
            rules = "none" if rk == "0" else "all"
            wrong = [convention] if rk == "-" else []
            facts = {"rule": rk, "examples": False, "clean": [], "n": 0}
            out.append((f"r{rk}|none", self.spec(rules=rules, wrong=wrong), facts))
            for k in range(len(car) + 1):
                subsets = list(itertools.combinations(car, k))
                if rk != "0" and k not in (0, 1, len(car) - 1, len(car)):
                    subsets = [tuple(car[:k])]  # nested chain: the first k carriers clean
                for clean in subsets:
                    bits = "".join("1" if e in clean else "0" for e in car)
                    spec = self.spec(
                        rules=rules, wrong=wrong, examples=ex, broken={e: [convention] for e in car if e not in clean}
                    )
                    facts = {"rule": rk, "examples": True, "clean": sorted(num[e] for e in clean), "n": len(car)}
                    out.append((f"r{rk}|e{bits}", spec, facts))
        return out

    def example_id(self, which: int | str) -> str:
        """An example by its number in the standard list (1-based) or by its id."""
        return self.examples[int(which) - 1] if str(which).isdigit() else str(which)

    def picture_contexts(self, convention: str, break_example=None, other_example=None) -> list[tuple[str, ContextSpec]]:
        """The contexts of the attention pictures for one convention.

        `one_example_broken` breaks the convention in ONE example: `break_example` (number or id),
        by default a middle carrier. `other_example_broken` is the control: the same edit in another
        example (`other_example`, by default the first other carrier). `rule_*` use the seven rules,
        `one_line_*` the one-line rule (only for the conventions it states). `examples_broken` is
        the default corrupted prompt of the causal test."""
        car, ex = self.carriers(convention), self.examples
        one = self.example_id(break_example) if break_example else (car[len(car) // 2] if car else None)
        if one is not None and one not in car:
            raise ValueError(f"{one} does not show {convention}: it cannot be broken there")
        others = [e for e in car if e != one]
        other = self.example_id(other_example) if other_example else (others[0] if others else None)
        every = {e: [convention] for e in car}
        out = [("examples_clean", self.spec(examples=ex))]
        if one:
            out.append(("one_example_broken", self.spec(examples=ex, broken={one: [convention]})))
            out.append(("one_example_broken+rules", self.spec(rules="all", examples=ex, broken={one: [convention]})))
        if other:
            out.append(("other_example_broken", self.spec(examples=ex, broken={other: [convention]})))
        out.append(("rule_only", self.spec(rules="all")))
        out.append(("rule_wrong_only", self.spec(rules="all", wrong=[convention])))
        out.append(("rule_vs_examples", self.spec(rules="all", examples=ex, broken=every)))
        out.append(("rule_wrong_vs_examples", self.spec(rules="all", wrong=[convention], examples=ex)))
        if convention in self.td.one_line_wrong:
            out.append(("one_line_vs_examples", self.spec(rules="one_line", examples=ex, broken=every)))
            out.append(("one_line_wrong_vs_examples", self.spec(rules="one_line", wrong=[convention], examples=ex)))
        out.append(("examples_broken", self.spec(examples=ex, broken=every)))
        return out

    def picture_facts(self, convention: str, break_example=None, other_example=None) -> dict:
        """Which examples the picture contexts break (numbers in the standard list)."""
        car = self.carriers(convention)
        one = self.example_id(break_example) if break_example else (car[len(car) // 2] if car else None)
        others = [e for e in car if e != one]
        other = self.example_id(other_example) if other_example else (others[0] if others else None)
        return {
            "broken_example": self.example_number(one) if one else None,
            "broken_example_id": one,
            "other_example": self.example_number(other) if other else None,
            "other_example_id": other,
        }

    # -- prompts
    def build(self, question: Item, spec: ContextSpec) -> Built:
        p = tdprompt.build_prompt(self.td, spec, question)
        messages = [{"role": "system", "content": p.system}, *p.messages()]
        units = [
            UnitText("instructions", "instructions", "instructions", p.system, "instructions"),
            UnitText("schema", "schema", "schema", self.td.schema, "schema"),
        ]
        stated = self.conventions if spec.rules == "all" else ["one_line"]
        for cid, line in zip(stated, tdprompt.rule_lines(self.td, spec)):
            state = " (wrong)" if cid in spec.wrong_rules or (cid == "one_line" and spec.wrong_rules) else ""
            units.append(UnitText(f"rule.{cid}", "rule", f"rule.{cid}", line, f"rule: {cid}{state}"))
        for use, (q, sql) in zip(spec.examples, tdprompt.example_pairs(self.td, spec)):
            n = self.example_number(use.id) if use.id in self.examples else use.id
            state = f" (broken: {', '.join(sorted(use.corrupt))})" if use.corrupt else ""
            units.append(UnitText(f"ex{n}.q", "example_question", f"ex{n}", q, f"example {n} question"))
            units.append(UnitText(f"ex{n}.sql", "example_sql", f"ex{n}", sql, f"example {n} SQL{state}"))
        units.append(UnitText("question", "question", "question", question.question, "question"))
        return Built(messages, units, spec)

    # -- decisions
    def decision(self, question: Item, convention: str, extend: int = 0) -> Decision:
        """The scenario's decision point, scored on its `follows_head` / `violates_head`: the first
        whitespace-delimited chunk of each continuation. `extend` adds that many more chunks."""
        dp = self.td.decision_point(question, convention)

        def chunks(rest: str) -> str:
            m = re.match(r"(?:\s*\S+){%d}" % (1 + extend), rest)
            return m.group(0) if m else rest

        return Decision(convention, dp.prefix, chunks(dp.follows), chunks(dp.violates), dp.follows, dp.violates)

    # -- checking generations
    def check(self, question: Item, reply: str) -> dict[str, bool | None]:
        """Per applicable convention: True follows, False violates, None neither (not applicable
        to what was written, or not parseable)."""
        out = {}
        for cid, chk in self.td.score(question, reply).items():
            out[cid] = True if chk.status == FOLLOWS else False if chk.status == VIOLATES else None
        return out

    def describe(self) -> dict:
        return {
            "scenario": self.name,
            "scenario_version": self.version,
            "conventions": self.conventions,
            "standard_examples": self.examples,
            "carriers": {c: [self.example_number(e) for e in self.carriers(c)] for c in self.conventions},
            "in_one_line_rule": self.in_one_line_rule(),
        }


def load_scenario(name: str) -> Demo:
    """`shop` (or any module under src/scenarios); `fixture` = the small stand-in in wb/fixture.py."""
    builder = [DEMO_DIR / "talkdemo" / f for f in ("prompt.py", "scenario.py", "sqlcheck.py")]
    if name == "fixture":
        from wb import fixture

        return Demo(fixture.SCENARIO, "fixture", [Path(fixture.__file__), *builder])
    from importlib import import_module

    module = import_module(f"scenarios.{name}")
    return Demo(module.SCENARIO, name, [Path(module.__file__), *builder])
