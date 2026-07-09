"""Score how relevant one query is to another, by shared tables + SQL constructs.
Used to build relevant-vs-irrelevant example sub-libraries for the ablation that
tests whether the other examples carry transferable information or are just noise.
"""
import re
from functools import lru_cache

import db
import questions

_CONSTRUCTS = [
    ("window", r"OVER\s*\("),
    ("lag_lead", r"\b(LAG|LEAD)\s*\("),
    ("rank", r"\b(ROW_NUMBER|RANK|DENSE_RANK)\s*\("),
    ("date", r"DATE_TRUNC|INTERVAL|DATEADD|DATE_DIFF|DATEDIFF"),
    ("distinct", r"\bDISTINCT\b"),
    ("case", r"\bCASE\s+WHEN"),
    ("cte", r"\bWITH\b"),
    ("having", r"\bHAVING\b"),
    ("ratio", r"/\s"),
    ("groupby", r"GROUP\s+BY"),
    ("limit", r"\bLIMIT\b"),
]


@lru_cache(maxsize=None)
def _schema_tables(ds):
    return frozenset(db.schema(ds).keys())


def tables_used(ds, sql):
    toks = set(re.findall(r"[A-Z_][A-Z0-9_]+", sql.upper()))
    return {t for t in _schema_tables(ds) if t in toks}


def constructs(sql):
    u = sql.upper()
    return {n for n, p in _CONSTRUCTS if re.search(p, u)}


def _jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def sim(ds, gold_sql, example_sql):
    """0..1 similarity: half table overlap, half SQL-construct overlap."""
    return 0.5 * _jac(tables_used(ds, gold_sql), tables_used(ds, example_sql)) \
        + 0.5 * _jac(constructs(gold_sql), constructs(example_sql))


_STOP = {"the", "a", "an", "of", "for", "in", "by", "is", "are", "what", "which",
         "how", "many", "much", "each", "per", "with", "and", "or", "to", "our",
         "we", "their", "its", "that", "this", "at", "on", "was", "were"}


def _qtoks(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def sim_question(question, example_question):
    """Question-text similarity — what a production retriever would see (no gold)."""
    return _jac(_qtoks(question), _qtoks(example_question))


def sim_goldtok(gold_sql, example_sql):
    """Token-level SQL similarity to the gold — a gold-aware retrieval upper bound."""
    t = lambda s: set(re.findall(r"[A-Z_][A-Z0-9_]*", s.upper()))
    return _jac(t(gold_sql), t(example_sql))


def rank_other_examples(ds, qid, mode="construct"):
    """Other questions' examples ranked by similarity to this question (desc).
    mode: 'construct' = shared tables + SQL constructs of the gold (original arm);
          'question'  = question-text overlap only (realistic retriever);
          'goldtok'   = SQL token overlap with the gold (gold-aware upper bound)."""
    qs = questions.load(ds)
    me = next(q for q in qs if q["id"] == qid)
    others = [q for q in qs if q["id"] != qid and q.get("example_sql")]
    if mode == "construct":
        scored = [(q, sim(ds, me["gold_sql"], q["example_sql"])) for q in others]
    elif mode == "question":
        scored = [(q, sim_question(me["question"], q["example_question"])) for q in others]
    elif mode == "goldtok":
        scored = [(q, sim_goldtok(me["gold_sql"], q["example_sql"])) for q in others]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    scored.sort(key=lambda t: -t[1])
    return scored


def select_examples(ds, qid, which, k=3):
    """k examples for an ablation arm. which: 'rel'/'irr' = top/bottom-k by the
    original construct ranking; 'relq' = top-k by question-text (realistic
    retriever); 'relg' = top-k by gold-SQL token overlap (upper bound)."""
    mode = {"rel": "construct", "irr": "construct",
            "relq": "question", "relg": "goldtok"}[which]
    ranked = rank_other_examples(ds, qid, mode)
    chosen = ranked[-k:] if which == "irr" else ranked[:k]
    return [q for q, _ in chosen]
