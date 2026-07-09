"""Derivation-distance ladder context renderer. See docs/CONTRACT.md + README.

The DERIVATION axis holds COMPLETENESS constant: every rung (L1+) shows the
published ontology at increasing depth, plus the global dataset facts. The context
is IDENTICAL across questions at every level (discoverability is part of the task);
rungs differ only in how spelled-out the *implementation* is, so accuracy
differences along the axis are derivation distance, not missing information:

  L1 schema_descriptions  schema + column descriptions + global operating facts — max distance
  L2 prose_rules          + domain context_md prose (rules, guardrails)
  L3 named_metrics        + named metric (entity + description, NO expression)
  L4 metric_formulas      + metric expression + filters + table   (the full published ontology)
  L5 example_library      + the FULL worked-example library (model finds the relevant analogue)
  L6 answer_templates     + the FULL template gallery: every question's gold SQL, one literal
                            blanked to <???> (model finds its own and fills the blank)

  L0 bare_schema    SIDE ARM (not on the axis): column NAMES only, NO facts.
                    Deliberately incomplete for most questions — measures whether a
                    model can recover the conventions from names alone.

Only facts present in THIS dataset's ontology (plus the authored required facts)
are emitted; nothing is invented.
"""
import json
import os
import re

import config

# ---- level helpers ---------------------------------------------------------

# Inclusion flags are keyed off the level's position within DERIVATION_LEVELS.
_DRANK = {name: i for i, name in enumerate(config.DERIVATION_LEVELS)}
# also accept short codes ("L3")
for _name, _i in list(_DRANK.items()):
    _DRANK[_name.split("_", 1)[0]] = _i

_SIDE = set(config.SIDE_ARMS) | {s.split("_", 1)[0] for s in config.SIDE_ARMS}


def _is_side(level):
    return level in _SIDE


def _drank(level):
    if level not in _DRANK:
        raise ValueError(f"unknown derivation level {level!r}; expected {config.DERIVATION_LEVELS}")
    return _DRANK[level]


def _load_ontology(ds):
    with open(config.ontology_path(ds), "r", encoding="utf-8") as f:
        return json.load(f)


# ---- section builders ------------------------------------------------------


def _facts_section(ds, qrow):
    """GLOBAL dataset facts only (e.g. the snapshot date) — IDENTICAL across all
    questions. We deliberately do NOT inject the question's `required_facts`: those
    are the completeness rubric, not context. Discoverability of the relevant
    conventions is part of the task. Injected at every derivation rung (L1+)."""
    facts = list(config.DATASET_FACTS.get(ds, []))
    if not facts:
        return None
    lines = ["# Operating context", ""]
    lines += [f"- {f}" for f in facts]
    lines.append("")
    return lines


def _schema_section(ds, ontology, include_descriptions):
    """Tables + columns. names only or names + descriptions."""
    try:
        import db

        schema = db.schema(ds)
    except Exception:
        schema = None

    lines = ["# Schema", ""]
    for t in ontology.get("tables", []):
        tname = t["table_name"]
        if include_descriptions and t.get("description"):
            lines.append(f"## {tname} — {t['description']}")
        else:
            lines.append(f"## {tname}")

        onto_cols = {c["name"]: c for c in t.get("columns", [])}
        db_cols = schema.get(tname) if schema else None
        if db_cols:
            col_iter = [(c["name"], c.get("type")) for c in db_cols]
        else:
            col_iter = [(c["name"], c.get("data_type")) for c in t.get("columns", [])]

        for cname, ctype in col_iter:
            oc = onto_cols.get(cname, {})
            typ = ctype or oc.get("data_type") or ""
            base = f"- {cname}" + (f" ({typ})" if typ else "")
            if include_descriptions and oc.get("description"):
                lines.append(f"{base}: {oc['description']}")
            else:
                lines.append(base)
        lines.append("")

    joins = ontology.get("joins", [])
    if joins:
        lines.append("## Joins")
        for j in joins:
            cond = j.get("condition_sql")
            if cond:
                line = f"- {cond}"
                if include_descriptions and j.get("description"):
                    line += f"  ({j['description']})"
                lines.append(line)
        lines.append("")
    return lines


def _prose_section(ontology):
    lines = ["# Domain context", ""]
    for d in ontology.get("domains", []):
        md = d.get("context_md")
        if not md:
            continue
        title = d.get("display_name") or d.get("path") or "(root)"
        lines += [f"## {title}", "", md.strip(), ""]
    return lines


def _metric_named_section(ontology):
    lines = ["# Metrics", ""]
    for m in ontology.get("metrics", []):
        lines.append(f"## {m.get('display_name') or m['name']} (`{m['name']}`)")
        if m.get("description"):
            lines.append(f"- Description: {m['description']}")
        if m.get("unit"):
            lines.append(f"- Unit: {m['unit']}")
        if m.get("synonyms"):
            lines.append(f"- Also called: {', '.join(m['synonyms'])}")
        lines.append("")
    return lines


def _metric_impl_section(ontology):
    lines = ["# Metric implementations", ""]
    for m in ontology.get("metrics", []):
        lines.append(f"## `{m['name']}`")
        lines.append(f"- Expression: {m['expression']}")
        tname, tschema = m.get("table_name"), m.get("table_schema")
        if tname:
            lines.append(f"- Table: {tschema + '.' + tname if tschema else tname}")
        lines.append(f"- Filters: {m.get('filters') or 'none'}")
        lines.append("")
    return lines


def _worked_example_section(qrow):
    es = qrow.get("example_sql")
    if not es:
        return None
    lines = ["# Worked example (analogous question)", ""]
    if qrow.get("example_question"):
        lines += [f"Question: {qrow['example_question']}", ""]
    lines += ["```sql", es.strip(), "```", ""]
    return lines


def _examples_from(chosen):
    """Render a worked-example library from an explicit list of question dicts."""
    lines = ["# Worked examples (library)", ""]
    n = 0
    for q in chosen:
        if not q.get("example_sql"):
            continue
        n += 1
        if q.get("example_question"):
            lines += [f"## {q['example_question']}", ""]
        lines += ["```sql", q["example_sql"].strip(), "```", ""]
    return lines if n else None


def _example_library_section(ds, exclude_id=None):
    """Worked examples for the dataset. If exclude_id is given (leave-one-out), the
    matching question's own analogue is omitted, so the library was not written for
    the question being asked. Sorted by id for a stable ordering."""
    import questions  # local import: questions imports config/db/grade, not ladder

    qs = sorted(questions.load(ds), key=lambda q: q["id"])
    lines = ["# Worked examples (library)", ""]
    n = 0
    for q in qs:
        if q["id"] == exclude_id:
            continue
        es = q.get("example_sql")
        if not es:
            continue
        n += 1
        if q.get("example_question"):
            lines.append(f"## {q['example_question']}")
            lines.append("")
        lines += ["```sql", es.strip(), "```", ""]
    return lines if n else None


_NUM_LITERAL = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_STR_LITERAL = re.compile(r"'[^']*'")


def _blank_one_literal(sql):
    m = _STR_LITERAL.search(sql) or _NUM_LITERAL.search(sql)
    if m:
        return sql[: m.start()] + "<???>" + sql[m.end() :]
    return sql


def _template_library_section(ds, exclude_id=None):
    """Answer templates for the dataset: each question paired with its gold query,
    one literal blanked (a certified-query gallery). If exclude_id is given
    (leave-one-out), the matching question's own template is omitted, so the gallery
    does not contain the answer to the question being asked. Sorted by id."""
    import questions  # local import: questions imports config/db/grade, not ladder

    qs = sorted(questions.load(ds), key=lambda q: q["id"])
    lines = ["# Answer templates (library — find the one matching the question, fill the `<???>`)", ""]
    n = 0
    for q in qs:
        if q["id"] == exclude_id:
            continue
        gold = q.get("gold_sql")
        if not gold:
            continue
        n += 1
        lines.append(f"## {q.get('question', q['id'])}")
        lines.append("")
        lines += ["```sql", _blank_one_literal(gold.strip()), "```", ""]
    return lines if n else None


def _template_section(qrow):
    gold = qrow.get("gold_sql")
    if not gold:
        return None
    return [
        "# Answer template (gold query, one literal blanked)",
        "",
        "Fill in the `<???>` value:",
        "",
        "```sql",
        _blank_one_literal(gold.strip()),
        "```",
        "",
    ]


# ---- public API ------------------------------------------------------------


# Every level's BODY (everything except the question line) is now identical across
# all questions in a dataset, so each is pre-rendered once per (dataset, level).
SHARED_BODY_LEVELS = ["L0_bare_schema", "L1_schema_descriptions", "L2_prose_rules",
                      "L3_named_metrics", "L4_metric_formulas", "L5_example_library", "L6_answer_templates"]


def _body_sections(ds, qrow, level):
    """The context sections WITHOUT the question block. Returns a list of line-lists.
    Every level's body is question-independent (shared across the dataset)."""
    ontology = _load_ontology(ds)
    parts = []

    if _is_side(level):
        # L0 bare schema: names only, NO facts. Deliberately incomplete.
        parts.append(_schema_section(ds, ontology, include_descriptions=False))
        return parts

    if level in ("L5_example_library", "L6_answer_templates", "L5_library_heldout", "L6_templates_heldout",
                 "L5_relevant_3", "L5_irrelevant_3", "L5_relevant_3_question", "L5_relevant_3_goldsql"):
        # Full ontology (= L4) + a library the model must search. *_loo = leave-one-out
        # (this question's own entry held out). L5_relevant_3 / L5_irrelevant_3 = a 3-example library
        # of the most / least relevant OTHER examples (relevance ablation);
        # L5_relevant_3_question / L5_relevant_3_goldsql = same but selected by question-text similarity
        # (realistic retriever) / gold-SQL token overlap (retrieval upper bound).
        parts.append(_facts_section(ds, qrow))
        parts.append(_schema_section(ds, ontology, include_descriptions=True))
        parts.append(_prose_section(ontology))
        parts.append(_metric_named_section(ontology))
        parts.append(_metric_impl_section(ontology))
        if level in ("L5_relevant_3", "L5_irrelevant_3", "L5_relevant_3_question", "L5_relevant_3_goldsql"):
            import relevance
            which = {"L5_relevant_3": "rel", "L5_irrelevant_3": "irr",
                     "L5_relevant_3_question": "relq", "L5_relevant_3_goldsql": "relg"}[level]
            chosen = relevance.select_examples(ds, qrow["id"], which, k=3)
            parts.append(_examples_from(chosen))
        elif level in ("L5_example_library", "L5_library_heldout"):
            parts.append(_example_library_section(ds, exclude_id=qrow["id"] if level == "L5_library_heldout" else None))
        else:
            parts.append(_template_library_section(ds, exclude_id=qrow["id"] if level == "L6_templates_heldout" else None))
        return [p for p in parts if p]

    r = _drank(level)  # L1..L4 only (L5/L6 libraries are handled above)
    # Global operating facts (e.g. snapshot date) — identical across questions.
    parts.append(_facts_section(ds, qrow))
    # Schema with descriptions at every derivation rung (L1+); L0 had names only.
    parts.append(_schema_section(ds, ontology, include_descriptions=True))
    if r >= 1:
        parts.append(_prose_section(ontology))
    if r >= 2:
        parts.append(_metric_named_section(ontology))
    if r >= 3:
        parts.append(_metric_impl_section(ontology))
    return [p for p in parts if p]


def render_body(ds, qrow, level):
    """The full context WITHOUT the leading question block (the part that is shared
    across questions for SHARED_BODY_LEVELS)."""
    blocks = ["\n".join(s).rstrip() for s in _body_sections(ds, qrow, level) if s]
    return "\n\n".join(blocks) + "\n"


def render(ds, qrow, level):
    """Build the full context string shown to the model for `qrow` at `level`."""
    qblock = "\n".join(["# Question", "", qrow.get("question", ""), ""]).rstrip()
    return qblock + "\n\n" + render_body(ds, qrow, level)


def available_levels(qrow):
    """Levels renderable for this question. The library levels are dataset-wide and
    always available; everything on the axis renders for any question with a gold."""
    out = list(config.SIDE_ARMS) + list(config.DERIVATION_LEVELS)
    out += list(getattr(config, "COMPARISON_LEVELS", []))
    return out


# ---- self-verify -----------------------------------------------------------

if __name__ == "__main__":
    def _proxy(text):
        return len(text.split())

    try:
        import harness  # type: ignore

        _model = config.MODELS["haiku"]

        def count_tokens(text):
            try:
                return harness.count_tokens(_model, text)
            except Exception:
                return _proxy(text)
    except Exception:
        count_tokens = _proxy

    # Prefer a real question if the set exists; else a synthetic one.
    try:
        import questions

        qrow = [q for q in questions.load("jaffle-shop") if q["id"] == "jaffle__revenue_by_store_top5"][0]
    except Exception:
        qrow = {
            "id": "jaffle__revenue_by_store",
            "dataset": "jaffle-shop",
            "question": "Total revenue by store (top 5)",
            "required_facts": ["Revenue uses ORDERS.SUBTOTAL (pre-tax), not ORDER_TOTAL."],
            "gold_sql": "SELECT s.STORE_NAME, SUM(o.SUBTOTAL) FROM ORDERS o JOIN STORES s ON o.STORE_ID=s.STORE_ID GROUP BY 1 ORDER BY 2 DESC LIMIT 5",
            "example_question": "Total revenue by month",
            "example_sql": "SELECT DATE_TRUNC('month', ORDERED_AT) m, SUM(SUBTOTAL) FROM ORDERS GROUP BY 1 ORDER BY 1",
        }
    ds = "jaffle-shop"

    out_dir = os.path.join(config.BASE, "data", "contexts")
    os.makedirs(out_dir, exist_ok=True)

    levels = available_levels(qrow)
    print(f"available levels: {levels}\n")
    counts = {}
    for level in levels:
        text = render(ds, qrow, level)
        counts[level] = count_tokens(text)
        with open(os.path.join(out_dir, f"{qrow['id']}__{level}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{level:20s} tokens={counts[level]:5d}")
    print()

    # Monotone across the ontology-depth prefix L1..L4 (L5_example_library is a large
    # library and L6/comparison are smaller, so the full axis is not monotone).
    depth = ["L1_schema_descriptions", "L2_prose_rules", "L3_named_metrics", "L4_metric_formulas"]
    nums = [counts[l] for l in depth]
    for i in range(1, len(nums)):
        assert nums[i] >= nums[i - 1], f"token count dropped at {depth[i]}"
    assert counts["L5_example_library"] > counts["L4_metric_formulas"], "L5_example_library should add the example library"
    assert counts["L6_answer_templates"] > counts["L4_metric_formulas"], "L6_answer_templates should add the template library"
    print("OK: token counts non-decreasing L1->L4; L5/L6 libraries add their galleries")

    onto = _load_ontology(ds)
    # Context is the GLOBAL ontology, identical across questions through L4. L1
    # carries column descriptions + the global operating facts (snapshot date).
    d0 = render(ds, qrow, "L1_schema_descriptions")
    assert "# Operating context" in d0, "L1 must carry the global operating facts"
    assert "2026-06-17" in d0, "L1 must carry the global snapshot date"
    # Context must NOT contain the per-question required_facts (those are the rubric).
    for f in qrow.get("required_facts", []):
        assert f not in d0, f"L1 leaked a per-question required fact: {f!r}"
    # L3 names the metric but must NOT leak the expression; L4 includes it.
    d2 = render(ds, qrow, "L3_named_metrics")
    for m in onto.get("metrics", []):
        assert m["expression"] not in d2, f"L3 leaked expression {m['expression']!r}"
    assert "Metric implementations" not in d2
    d3 = render(ds, qrow, "L4_metric_formulas")
    rev = next(m["expression"] for m in onto["metrics"] if m["name"] == "revenue")
    assert rev in d3, "L4 must include the revenue expression"
    print("OK: L1 carries global facts, NOT per-question facts; L3 hides expr; L4 reveals it")

    # Two different questions must share identical context at L4 (discoverability task).
    qs2 = [q for q in questions.load("jaffle-shop") if q["id"] != qrow["id"]]
    if qs2:
        other = render(ds, qs2[0], "L4_metric_formulas")
        a = "\n".join(l for l in d3.splitlines() if not l.startswith("# Question") and l.strip() != qrow["question"])
        b = "\n".join(l for l in other.splitlines() if not l.startswith("# Question") and l.strip() != qs2[0]["question"])
        assert a == b, "L4 context must be identical across questions (only the question line differs)"
        print("OK: L4 context identical across questions (only question text differs)")

    # L0 side arm: names only, NO facts/descriptions.
    s0 = render(ds, qrow, "L0_bare_schema")
    assert "# Operating context" not in s0, "L0 must NOT carry facts"
    print("OK: L0 bare schema carries names only")

    # L6_answer_templates: a shared gallery of blanked gold templates; the current question's
    # own template (and a <???> blank to fill) must be present.
    d5 = render(ds, qrow, "L6_answer_templates")
    assert "<???>" in d5, "L6_answer_templates templates must carry blanks to fill"
    assert qrow["question"] in render_body(ds, qrow, "L6_answer_templates"), "L6_answer_templates must include this question's template"
    other5 = render_body(ds, qs2[0], "L6_answer_templates") if qs2 else d5
    if qs2:
        assert render_body(ds, qrow, "L6_answer_templates") == other5, "L6_answer_templates body must be shared across questions"
    print("OK: L6_answer_templates is a shared template gallery carrying every question's blanked gold")
