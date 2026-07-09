"""Pre-rendered context cache.

The body of a context (everything except the leading question line) is identical
across all questions in a dataset for `ladder.SHARED_BODY_LEVELS` (L0, L1..L4,
L5_example_library). We store that body once per (dataset, level). Per-question levels
(L6_single_template, L5_single_example) are stored per (dataset, level, id).

`get(ds, qrow, level)` composes the question block with the cached body and is
byte-for-byte equal to `ladder.render(ds, qrow, level)`. The benchmark uses
`get()` so every run reads the audited, on-disk contexts.

CLI:  python src/contexts.py            # pre-render all + verify == ladder.render
"""
import os

import config
import ladder
import questions

ROOT = os.path.join(config.BASE, "data", "contexts")
ALL_LEVELS = list(config.LEVELS) + list(getattr(config, "COMPARISON_LEVELS", []))


def _shared_path(ds, level):
    return os.path.join(ROOT, ds, f"{level}.body.txt")


def _perq_path(ds, level, qid):
    return os.path.join(ROOT, ds, level, f"{qid}.body.txt")


def body_path(ds, level, qid):
    if level in ladder.SHARED_BODY_LEVELS:
        return _shared_path(ds, level)
    return _perq_path(ds, level, qid)


def _question_block(qrow):
    return "\n".join(["# Question", "", qrow.get("question", ""), ""]).rstrip()


def prerender(ds):
    """Write all body files for a dataset. Returns count written."""
    qs = questions.load(ds)
    n = 0
    for level in ALL_LEVELS:
        if level in ladder.SHARED_BODY_LEVELS:
            # qrow-independent body: render once (use the first question).
            body = ladder.render_body(ds, qs[0], level)
            p = _shared_path(ds, level)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(body)
            n += 1
        else:
            for q in qs:
                if level not in ladder.available_levels(q):
                    continue
                body = ladder.render_body(ds, q, level)
                p = _perq_path(ds, level, q["id"])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(body)
                n += 1
    return n


def prerender_all():
    return {ds: prerender(ds) for ds in config.DATASETS}


def get(ds, qrow, level):
    """Composed context string (question + cached body). Falls back to a live
    render if the cache file is missing, so the benchmark never breaks on a miss."""
    p = body_path(ds, level, qrow["id"])
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            body = f.read()
    else:
        body = ladder.render_body(ds, qrow, level)
    return _question_block(qrow) + "\n\n" + body


if __name__ == "__main__":
    import sys

    print("pre-rendering contexts:", prerender_all())
    bad = checked = 0
    for ds in config.DATASETS:
        for q in questions.load(ds):
            for level in ladder.available_levels(q):
                checked += 1
                if get(ds, q, level) != ladder.render(ds, q, level):
                    bad += 1
                    print("MISMATCH:", ds, q["id"], level)
    print(f"verified {checked} cached contexts == ladder.render(); {bad} mismatches")
    sys.exit(1 if bad else 0)
