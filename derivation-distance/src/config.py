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
# IDENTICAL across all questions through L4 (the model must locate the relevant
# tables / metrics / rules itself — we do NOT pre-select per-question facts).
# Only L5/L6 are tailored, because a worked example / answer template is
# inherently for one specific question. Completeness is therefore not hand-fed;
# it is a property of the ontology, MEASURED per question against each question's
# `required_facts` rubric (see docs) rather than injected.
DERIVATION_LEVELS = [
    "L1_schema_descriptions",    # schema names + descriptions + global dataset facts. Max derivation distance.
    "L2_prose_rules",          # + domain context_md prose (rules, guardrails)
    "L3_named_metrics",   # + named metric (entity + description, NO expression)
    "L4_metric_formulas",    # + metric expression (= the canonical published ontology)
    "L5_example_library",        # + the FULL worked-example library (model must find the relevant analogue)
    "L6_answer_templates",        # + the FULL template library: a blanked gold query per question, shared, model finds its own
]
# Side arm — NOT on the derivation axis. Bare schema names only (no descriptions,
# no facts). Measures "can the model recover everything from column names alone?"
SIDE_ARMS = ["L0_bare_schema"]

# Held-out (leave-one-out) library variants: for each question the library is built
# from ALL OTHER questions, excluding its own bespoke entry — so the gallery was not
# written for the question being asked (a more realistic "library covers a different
# question set" condition). Run for comparison against the all-inclusive libraries.
#   L5_library_heldout = worked-example library minus this question's own analogue
#   L6_templates_heldout = template gallery minus this question's own (blanked) gold
COMPARISON_LEVELS = ["L5_library_heldout", "L6_templates_heldout", "L5_relevant_3", "L5_irrelevant_3", "L5_relevant_3_question", "L5_relevant_3_goldsql"]
# L5_relevant_3 / L5_irrelevant_3: ontology + a 3-example library of the most / least relevant
# OTHER examples (by shared tables + SQL constructs). Same size, so comparing them
# isolates specific table/pattern transfer from a generic "examples are present"
# effect — does the held-out library inform the answer, or is its content noise?

LEVELS = SIDE_ARMS + DERIVATION_LEVELS
# The canonical published ontology corresponds to this level:
ANCHOR_LEVEL = "L4_metric_formulas"

# Genuinely GLOBAL facts the ontology does not carry (e.g. the snapshot date,
# which previously lived only in the dataset README). Injected at every derivation
# rung (L1+), IDENTICAL across all questions — not per-question tailoring.
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
