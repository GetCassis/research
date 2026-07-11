"""Shared configuration for the derivation-distance study. See docs/CONTRACT.md."""
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASETS = ["jaffle-shop", "thelook", "olist"]

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

# USD per million tokens (input, output). Source: claude-api skill, 2026-06.
PRICING = {
    "haiku": {"in": 1.0, "out": 5.0},
    "sonnet": {"in": 3.0, "out": 15.0},
    "opus": {"in": 5.0, "out": 25.0},
}

# --- Derivation-distance axis (REALISTIC: discoverability is part of the task) ---
# The context is the published ontology shown at increasing depth, and it is
# IDENTICAL across all questions at every level (the model must locate the
# relevant tables / metrics / rules itself — we do NOT pre-select per-question
# facts). Completeness is therefore not hand-fed; it is a property of the
# ontology, MEASURED per question against each question's `required_facts`
# rubric (see docs) rather than injected.
DERIVATION_LEVELS = [
    "descriptions",  # schema + column descriptions + global dataset facts. Max derivation distance.
    "metrics",       # + domain prose, named metrics, and metric expressions (= the full published ontology)
    "examples",      # + the FULL worked-example library (model must find the relevant analogue)
]
# Side arm — NOT on the derivation axis. Bare schema names only (no descriptions,
# no facts). Measures "can the model recover everything from column names alone?"
SIDE_ARMS = ["bare_schema"]

# Comparison arms, off the main axis:
#   descriptions_plus_examples = the complete descriptions foundation + the
#     full example library, with domain prose and metric definitions omitted.
#   templates = the FULL template gallery (every question's gold SQL, one literal
#     blanked, shared; model finds its own and fills the blank). Off-axis because
#     it presumes the answers were written down ahead of time.
#   *_heldout = leave-one-out: for each question the library is rebuilt from ALL
#     OTHER questions, so the gallery was not written for the question being asked.
#   relevant_3 / irrelevant_3: ontology + a 3-example library of the most / least
#     relevant OTHER examples (by shared tables + SQL constructs). Same size, so
#     comparing them isolates specific transfer from a generic "examples present"
#     effect. relevant_3_question / relevant_3_goldsql select by question-text
#     similarity (realistic retriever) / gold-SQL token overlap (upper bound).
COMPARISON_LEVELS = ["descriptions_plus_examples", "templates",
                     "examples_heldout", "templates_heldout",
                     "relevant_3", "irrelevant_3",
                     "relevant_3_question", "relevant_3_goldsql"]

LEVELS = SIDE_ARMS + DERIVATION_LEVELS
# The canonical published ontology corresponds to this level:
ANCHOR_LEVEL = "metrics"

# Genuinely GLOBAL facts the ontology does not carry (e.g. the snapshot date,
# which previously lived only in the dataset README). Injected at every derivation
# rung (descriptions and above), IDENTICAL across all questions — not per-question tailoring.
# The window-convention fact pins the house reading of "last N days" (lower-bounded
# only) — an audit showed models split between open and closed upper bounds, and the
# golds use the open reading.
_WINDOW_RULE = (
    "Relative time windows ('last N days') are lower-bounded only: include everything "
    "from today minus N days onward, and do not add an upper bound at today."
)
DATASET_FACTS = {
    "jaffle-shop": [
        "The data is a static snapshot; treat the current date ('today') as 2026-06-17 for any relative time window.",
        _WINDOW_RULE,
    ],
    "thelook": [
        "The data is a static snapshot; treat the current date ('today') as 2026-04-25 for any relative time window.",
        _WINDOW_RULE,
    ],
    "olist": [
        "The data is a static snapshot; treat the current date ('today') as 2026-05-30 for any relative time window.",
        _WINDOW_RULE,
    ],
}

# Sampling: fixed temperature for repetition variance. Thinking on (adaptive) for
# the emergent derivation-distance measure. Opus/Sonnet/Haiku 4.x: adaptive only.
DEFAULT_EFFORT = "medium"


def dataset_dir(ds):
    return os.path.join(BASE, "datasets", ds)


def db_path(ds):
    return os.path.join(BASE, "data", f"{ds}.duckdb")


def ontology_path(ds):
    return os.path.join(BASE, "datasets", ds, "ontology.json")


def questions_path(ds):
    return os.path.join(BASE, "data", "questions", f"{ds}.yaml")


def gold_path(qid):
    return os.path.join(BASE, "data", "gold", f"{qid}.json")
