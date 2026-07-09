"""Result-set canonicalization + grading for the derivation-distance study.

See docs/CONTRACT.md. Canonicalization is order-insensitive and float-tolerant:
the result set is treated as a *multiset of rows*, floats are rounded to 4 dp,
and column names are ignored (model SQL may alias columns differently) — only
values, by position within each row, matter.
"""
import hashlib
import json
import re
from decimal import Decimal

import config
import db

FLOAT_DP = 4

# Matches a leading ```sql ... ``` fence (or a bare ``` ... ``` fence).
_FENCE_RE = re.compile(
    r"```[ \t]*(?:sql)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def strip_fences(text):
    """Return the SQL inside the first ```sql fenced block, else the stripped text."""
    if text is None:
        return ""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # No fence: drop a stray leading "sql" label if present, then strip.
    return text.strip()


def _canon_cell(v):
    """Deterministic, float-tolerant stringification of a single cell value."""
    if v is None:
        return "\x00NULL"
    if isinstance(v, bool):
        # bool is a subtype of int; keep it distinct and stable.
        return "true" if v else "false"
    if isinstance(v, (float, Decimal)):
        # Treat DuckDB DECIMAL and FLOAT alike: round to 4 dp, fixed format.
        # (A model may type a numeric literal as int/float while gold is DECIMAL.)
        r = round(float(v), FLOAT_DP)
        # Normalize -0.0 to 0.0 for a stable form.
        if r == 0:
            r = 0.0
        return f"{r:.{FLOAT_DP}f}"
    if isinstance(v, int):
        return str(v)
    # Dates, datetimes, Decimals, strings, bytes, etc.: stable repr-ish string.
    return str(v)


def _canon_row(row):
    return tuple(_canon_cell(c) for c in row)


def canonical(columns, rows):
    """Order-insensitive, float-rounded canonical string of a result set.

    Rows are treated as a multiset (sorted after per-cell canonicalization).
    Column names are intentionally ignored — comparison is by value/position.
    """
    canon_rows = sorted(_canon_row(r) for r in rows)
    return json.dumps(canon_rows, ensure_ascii=False, separators=(",", ":"))


def result_hash(columns, rows):
    """sha256 hex of canonical(columns, rows)."""
    return hashlib.sha256(canonical(columns, rows).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tolerant matching (false-negative reduction)
#
# The strict hash comparison fails answers whose *values* are right but whose
# *form* differs from gold: an explicit ROUND(x, 2) on the final metric, an
# extra projected column (e.g. the LAG intermediate), a reordered projection,
# a ::DATE cast where gold keeps a midnight TIMESTAMP, or an integer-typed
# literal where gold is DECIMAL. A failure audit of the full-ladder run
# found 97% of such "presentation" failures carry verifiably correct values.
# The tolerant pass accepts exactly those forms and nothing else:
#   - numeric typing unified (int == float == Decimal at equal value)
#   - midnight timestamps == dates (same instant, different type)
#   - the model's own display precision, never coarser than 2 dp (a model
#     that rounds to integers still fails; one that rounds to 2 dp passes
#     iff gold rounded to 2 dp is identical)
#   - any injective column mapping of gold into the model result (supersets
#     and reorders), row multisets compared after the normalizations above
# Deliberately NOT accepted: fraction where gold is percent (x100 scale),
# entity IDs where gold projects display names, and any value difference
# beyond the model's evident rounding.
# ---------------------------------------------------------------------------
import itertools
import math

TOLERANT_MIN_DP = 2      # never compare coarser than 2 decimal places
_MAX_PROJECTIONS = 20000  # give up (stay strict) beyond this many column mappings

_TS_SEP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}(?:\.\d+)?)$")
_MIDNIGHT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00(?:\.0+)?$")


def _norm_cell(v, dp, collapse_midnight=True):
    """Like _canon_cell but: all numerics unified to fixed dp, and (optionally)
    midnight timestamps collapsed to their date. Handles both live DB values and
    gold rows deserialized from JSON (datetimes as ISO strings)."""
    if v is None:
        return "\x00NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float, Decimal)):
        r = round(float(v), dp)
        if r == 0:
            r = 0.0
        return f"{r:.{dp}f}"
    s = str(v)
    s = _TS_SEP_RE.sub(r"\1 \2", s)  # ISO 'T' separator == space separator
    if collapse_midnight:
        m = _MIDNIGHT_RE.match(s)
        if m:
            return m.group(1)
    return s


def _decimals(v):
    """Display precision of a numeric value (decimals after stripping zeros)."""
    s = f"{float(v):.8f}".rstrip("0")
    return min(len(s.split(".")[1]) if "." in s else 0, FLOAT_DP)


def _column_dps(rows):
    """Per-column comparison precision from the MODEL's own output: the max
    decimals it printed, floored at TOLERANT_MIN_DP, capped at FLOAT_DP.
    A model that rounded to 2 dp is compared at 2 dp; full-precision output
    keeps the strict 4 dp."""
    if not rows:
        return []
    dps = [0] * len(rows[0])
    for row in rows:
        for i, v in enumerate(row):
            if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool):
                dps[i] = max(dps[i], _decimals(v))
    return [max(TOLERANT_MIN_DP, min(d, FLOAT_DP)) for d in dps]


def _norm_multiset(rows, dps, collapse_midnight=True):
    return sorted(
        tuple(_norm_cell(v, dps[i], collapse_midnight) for i, v in enumerate(row))
        for row in rows
    )


def tolerant_match(gold_cols, gold_rows, model_cols, model_rows):
    """Return a match-kind string if the model result is a presentation-variant
    of gold (see block comment above), else None."""
    k, n = len(gold_cols), len(model_cols)
    if n < k or len(model_rows) != len(gold_rows):
        return None
    if math.perm(n, k) > _MAX_PROJECTIONS:
        return None
    for perm in itertools.permutations(range(n), k):
        proj = [[row[j] for j in perm] for row in model_rows]
        dps = _column_dps(proj)
        if _norm_multiset(proj, dps) != _norm_multiset(gold_rows, dps):
            continue
        kinds = []
        if n > k or perm != tuple(range(k)):
            kinds.append("projection")
        if any(d < FLOAT_DP for d in dps):
            # only claim rounding if strict-precision comparison actually fails
            strict_dps = [FLOAT_DP] * k
            if _norm_multiset(proj, strict_dps) != _norm_multiset(gold_rows, strict_dps):
                kinds.append("rounding")
        if _norm_multiset(proj, dps, collapse_midnight=False) != _norm_multiset(
            gold_rows, dps, collapse_midnight=False
        ):
            kinds.append("typing")
        return "+".join(kinds) if kinds else "numeric-typing"
    return None


def grade(ds, model_sql, gold_id, tolerant=True):
    """Execute model SQL and compare its result to the stored gold.

    Strict hash comparison first; if that fails and `tolerant` is set, accept
    verified presentation-variants of the gold result (see tolerant_match).
    The returned dict carries `match_kind`: "exact", a tolerant kind such as
    "projection+rounding", or None when incorrect.

    Returns {"exec_ok", "correct", "result_hash", "match_kind", "error"}.
    Never raises on a SQL execution error — that becomes exec_ok=False with
    the message in error.
    """
    sql = strip_fences(model_sql)
    try:
        cols, rows = db.run_sql(ds, sql)
    except Exception as e:  # noqa: BLE001 - any execution failure is a graded miss
        return {"exec_ok": False, "correct": False, "result_hash": None,
                "match_kind": None, "error": str(e)}

    rh = result_hash(cols, rows)

    try:
        with open(config.gold_path(gold_id), "r", encoding="utf-8") as f:
            gold = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {
            "exec_ok": True,
            "correct": False,
            "result_hash": rh,
            "match_kind": None,
            "error": f"gold load failed: {e}",
        }

    correct = rh == gold.get("hash")
    match_kind = "exact" if correct else None
    if not correct and tolerant:
        match_kind = tolerant_match(gold["columns"], gold["rows"], cols, rows)
        correct = match_kind is not None
    return {
        "exec_ok": True,
        "correct": correct,
        "result_hash": rh,
        "match_kind": match_kind,
        "error": None,
    }


if __name__ == "__main__":
    import datetime
    import os
    import tempfile

    # --- Order-insensitivity: row order must not affect the hash ---
    a = result_hash(["x"], [(1,), (2,)])
    b = result_hash(["x"], [(2,), (1,)])
    assert a == b, "row order should not affect hash"

    # --- Column-name-insensitivity: same values, different column labels ---
    c1 = result_hash(["revenue"], [(10.0,)])
    c2 = result_hash(["total_rev"], [(10.0,)])
    assert c1 == c2, "column names should not affect hash"

    # --- Float tolerance: rounding to 4 dp makes near-equal floats equal ---
    f1 = result_hash(["a", "b"], [(1, 2.00001)])
    f2 = result_hash(["a", "b"], [(1, 2.0)])
    assert f1 == f2, "floats should be equal after 4dp rounding"

    # --- But genuinely different values differ ---
    assert result_hash(["x"], [(1,)]) != result_hash(["x"], [(2,)])

    # --- Multiset semantics: duplicates matter ---
    assert result_hash(["x"], [(1,), (1,)]) != result_hash(["x"], [(1,)])

    # --- NULL handling is stable and distinct from values ---
    assert result_hash(["x"], [(None,)]) != result_hash(["x"], [(0,)])
    assert result_hash(["x"], [(None,)]) == result_hash(["x"], [(None,)])

    # --- grade() against a tiny gold file in a temp location ---
    # Build a real gold for a trivial DuckDB query and confirm grading works.
    gold_id = "selftest__grade"
    gold_dir = os.path.dirname(config.gold_path(gold_id))
    os.makedirs(gold_dir, exist_ok=True)
    gold_file = config.gold_path(gold_id)
    created = not os.path.exists(gold_file)
    try:
        ds = "jaffle-shop"
        cols, rows = db.run_sql(ds, "SELECT 1 AS a, 2.0 AS b")
        gold = {
            "id": gold_id,
            "columns": cols,
            # serialize like questions.py._jsonable: Decimal -> float
            "rows": [[float(v) if isinstance(v, Decimal) else v for v in r] for r in rows],
            "row_count": len(rows),
            "hash": result_hash(cols, rows),
        }
        with open(gold_file, "w", encoding="utf-8") as f:
            json.dump(gold, f, default=str)

        # Correct answer (different column aliases, equivalent values).
        g = grade(ds, "```sql\nSELECT 1 AS foo, 2.00001 AS bar\n```", gold_id)
        assert g["exec_ok"] and g["correct"], f"expected correct, got {g}"
        assert g["error"] is None

        # Wrong answer.
        g = grade(ds, "SELECT 1 AS a, 3 AS b", gold_id)
        assert g["exec_ok"] and not g["correct"], f"expected incorrect, got {g}"

        # --- Tolerant matching ---
        gm = tolerant_match
        gold_cols = ["month", "rev", "growth"]
        gold_rows = [["2025-06-01T00:00:00", 5676.0, None],
                     ["2025-07-01T00:00:00", 17807.0, 213.7244538407329]]
        # Superset (extra LAG column) + ::DATE cast + ROUND(...,2): accept.
        model = [[datetime.date(2025, 6, 1), 5676.0, None, None],
                 [datetime.date(2025, 7, 1), 17807.0, 5676.0, 213.72]]
        kind = gm(gold_cols, gold_rows, ["m", "r", "prev", "g"], model)
        assert kind and "projection" in kind and "rounding" in kind and "typing" in kind, kind
        # Reordered columns only: accept as projection.
        model = [[5676.0, "2025-06-01 00:00:00", None],
                 [17807.0, "2025-07-01 00:00:00", 213.7244538407329]]
        assert gm(gold_cols, gold_rows, ["r", "m", "g"], model) == "projection"
        # int where gold is float: accept (numeric typing).
        assert gm(["n"], [[10.0]], ["n"], [[10]]) == "numeric-typing"
        # Rounded to 2 dp: accept; to integer: reject (floor at 2 dp).
        assert gm(["g"], [[213.7244538407329]], ["g"], [[213.72]]) == "rounding"
        assert gm(["g"], [[213.7244538407329]], ["g"], [[214.0]]) is None
        # Fraction vs percent: reject. Different value: reject.
        assert gm(["s"], [[34.56]], ["s"], [[0.3456]]) is None
        assert gm(["x"], [[1.0]], ["x"], [[2.0]]) is None
        # Missing column / row-count mismatch: reject.
        assert gm(["a", "b"], [[1, 2]], ["a"], [[1]]) is None
        assert gm(["x"], [[1.0], [2.0]], ["x"], [[1.0]]) is None
        # Hedging shape: 1-col gold found inside a wider result is accepted —
        # measured on the real run to occur only in verified-benign clusters.
        assert gm(["x"], [[10.0]], ["a", "x"], [[99.0, 10.0]]) == "projection"

        # grade() end-to-end tolerant: extra column + rounding still correct.
        g = grade(ds, "SELECT 1 AS foo, 2.0 AS bar, 99 AS extra", gold_id)
        assert g["correct"] and g["match_kind"] == "projection", g
        g = grade(ds, "SELECT 1 AS foo, 2.0 AS bar, 99 AS extra", gold_id, tolerant=False)
        assert not g["correct"], "strict mode must stay strict"

        # Broken SQL -> exec_ok False, error populated, never raises.
        g = grade(ds, "SELECT FROM NOPE", gold_id)
        assert g["exec_ok"] is False and g["correct"] is False and g["error"]
    finally:
        if created and os.path.exists(gold_file):
            os.remove(gold_file)

    print("grade.py self-test passed")
