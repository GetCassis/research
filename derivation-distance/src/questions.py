"""Question loading + gold-answer building for the derivation-distance study.

See docs/CONTRACT.md. Loads data/questions/<ds>.yaml, executes each non-gap
question's gold_sql against the dataset's DuckDB, and writes the gold result set
(with a grader-matching hash) to data/gold/<id>.json.

CRITICAL: the stored hash is computed via grade.result_hash on the EXACT
(columns, rows) that db.run_sql returns — the same call grade() makes at runtime —
so the gold grades itself as correct. Rows are JSON-serialized only AFTER hashing.
"""
import datetime
import json
import os
from decimal import Decimal

import yaml

import config
import db
import grade


def load(ds):
    """Parse data/questions/<ds>.yaml into a list of question dicts."""
    with open(config.questions_path(ds), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{config.questions_path(ds)} must be a YAML list")
    return data


def _jsonable(v):
    """Make a single DB cell JSON-serializable (for storage only, not hashing)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    return str(v)


def build_gold(ds):
    """Execute each non-gap gold_sql, write data/gold/<id>.json. Returns list of golds."""
    qs = load(ds)
    out_dir = os.path.dirname(config.gold_path("x"))
    os.makedirs(out_dir, exist_ok=True)

    golds = []
    for q in qs:
        if q.get("gap"):
            continue
        qid = q["id"]
        sql = q["gold_sql"]
        cols, rows = db.run_sql(ds, sql)  # raises on error — surfaces bad SQL
        # Hash the RAW rows exactly as the grader will at runtime.
        h = grade.result_hash(cols, rows)
        gold = {
            "id": qid,
            "columns": list(cols),
            "rows": [[_jsonable(c) for c in r] for r in rows],
            "row_count": len(rows),
            "hash": h,
        }
        with open(config.gold_path(qid), "w", encoding="utf-8") as f:
            json.dump(gold, f, ensure_ascii=False, indent=2)
        golds.append(gold)
    return golds


if __name__ == "__main__":
    ds = "jaffle-shop"
    qs = [q for q in load(ds) if not q.get("gap")]
    golds = build_gold(ds)
    by_id = {q["id"]: q for q in qs}

    print(f"built {len(golds)} gold answers for {ds}\n")
    header = f"{'id':<32} {'difficulty':<12} {'rows':>5}  first_row"
    print(header)
    print("-" * len(header))

    all_ok = True
    for g in golds:
        q = by_id[g["id"]]
        # Self-grade: the gold_sql must grade itself as correct.
        res = grade.grade(ds, q["gold_sql"], g["id"])
        ok = res["exec_ok"] and res["correct"]
        all_ok = all_ok and ok
        first = g["rows"][0] if g["rows"] else None
        flag = "" if ok else "  <-- FAIL"
        print(f"{g['id']:<32} {q['difficulty']:<12} {g['row_count']:>5}  {first}{flag}")
        assert ok, f"gold did not self-grade as correct: {g['id']} -> {res}"

    print()
    print("ALL SELF-GRADE CORRECT" if all_ok else "SOME FAILED")
    assert all_ok
