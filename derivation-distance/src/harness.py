"""Model-calling harness for the derivation-distance study. See docs/CONTRACT.md.

Renders one (model, level, question, sample) cell's context via ladder.render,
calls the Claude model (adaptive thinking), extracts the SQL, grades it against
gold, and returns a run record. run_cell never raises: API/render failures are
captured into the record's "error" field.
"""
import time

from anthropic import Anthropic

import config
import grade
import ladder

# One client per process; cheap to construct, reads ANTHROPIC_API_KEY from env.
_client = Anthropic()

MAX_TOKENS = 4096

SYSTEM = (
    "You are a careful analytics engineer writing DuckDB SQL.\n"
    "Answer the question using ONLY the facts in the provided context "
    "(schema, descriptions, metrics, examples). Do not invent tables, "
    "columns, or business rules that are not in the context.\n"
    "Output exactly one DuckDB SQL query, wrapped in a single ```sql fenced "
    "code block. Use UPPERCASE for table and column names. Output no prose, "
    "no explanation, and no text outside the fenced block."
)


def count_tokens(model_label, text):
    """Token count of `text` as a user message, for the given model label."""
    model = config.MODELS[model_label]
    resp = _client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def _thinking_tokens(blocks):
    """Best-effort thinking-token estimate.

    Thinking is billed as output tokens but the SDK does not break it out in
    `usage`. We estimate from the emitted thinking block text using a cheap
    len//4 chars-per-token heuristic (avoids a second count_tokens round-trip
    per cell across a large factorial run).
    """
    chars = 0
    for b in blocks:
        if getattr(b, "type", None) == "thinking":
            chars += len(getattr(b, "thinking", "") or "")
    return chars // 4


def _extract_sql(blocks):
    """Concatenate text blocks and pull the SQL out of the ```sql fence."""
    text_parts = []
    for b in blocks:
        if getattr(b, "type", None) == "text":
            text_parts.append(b.text)
    raw = "\n".join(text_parts)
    return grade.strip_fences(raw), raw


# Cache of models known not to accept adaptive thinking (e.g. Haiku 4.5),
# so we skip the doomed adaptive attempt after the first 400.
_NO_ADAPTIVE = set()


def _call_model(model, user_msg, effort):
    """Call the model with adaptive thinking + effort, falling back to a plain
    call for models that reject adaptive thinking (verified: Haiku 4.5 does).

    Thinking is the *emergent* derivation-distance signal we want; we only forgo
    it when the API rejects it for that model.
    """
    messages = [{"role": "user", "content": user_msg}]
    base = dict(model=model, max_tokens=MAX_TOKENS, system=SYSTEM, messages=messages)

    if model not in _NO_ADAPTIVE:
        try:
            return _client.messages.create(
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                **base,
            )
        except Exception as e:  # noqa: BLE001
            if "adaptive thinking is not supported" in str(e):
                _NO_ADAPTIVE.add(model)
            else:
                raise
    # Plain call (no thinking) for models that don't support adaptive thinking.
    return _client.messages.create(**base)


def run_cell(model_label, ds, qrow, level, sample, effort=None):
    """Run one experiment cell. Returns a run record dict; never raises."""
    effort = effort or config.DEFAULT_EFFORT
    qid = qrow.get("id")
    question = qrow.get("question", "")

    record = {
        "id": qid,
        "dataset": ds,
        "model": model_label,
        "level": level,
        "sample": sample,
        "context_tokens": None,
        "input_tokens": None,
        "output_tokens": None,
        "thinking_tokens": None,
        "sql": None,
        "exec_ok": False,
        "correct": False,
        "result_hash": None,
        "cost_usd": None,
        "latency_s": None,
        "error": None,
    }

    # Load the pre-rendered context (falls back to a live render on cache miss).
    try:
        import contexts
        context = contexts.get(ds, qrow, level)
    except Exception as e:  # noqa: BLE001
        record["error"] = f"render failed: {e}"
        return record

    try:
        record["context_tokens"] = count_tokens(model_label, context)
    except Exception:  # noqa: BLE001 - non-fatal; leave as None
        pass

    user_msg = f"{context}\n\nQuestion: {question}"
    model = config.MODELS[model_label]

    t0 = time.time()
    try:
        resp = _call_model(model, user_msg, effort)
    except Exception as e:  # noqa: BLE001 - API failure must not crash the batch
        record["latency_s"] = round(time.time() - t0, 3)
        record["error"] = f"api call failed: {e}"
        return record
    record["latency_s"] = round(time.time() - t0, 3)

    blocks = resp.content or []
    sql, _raw = _extract_sql(blocks)
    record["sql"] = sql
    record["thinking_tokens"] = _thinking_tokens(blocks)

    usage = resp.usage
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    record["input_tokens"] = in_tok
    record["output_tokens"] = out_tok

    price = config.PRICING[model_label]
    record["cost_usd"] = round(in_tok * price["in"] / 1e6 + out_tok * price["out"] / 1e6, 6)

    # Grade (grade.grade itself never raises on SQL errors).
    try:
        g = grade.grade(ds, sql, qid)
        record["exec_ok"] = g["exec_ok"]
        record["correct"] = g["correct"]
        record["result_hash"] = g["result_hash"]
        if g["error"]:
            record["error"] = g["error"]
    except Exception as e:  # noqa: BLE001 - defensive; should not happen
        record["error"] = f"grade failed: {e}"

    return record


if __name__ == "__main__":
    # Always exercise the token counter (no ladder dependency).
    n = count_tokens("haiku", "SELECT 1")
    assert isinstance(n, int) and n > 0, n
    print(f"count_tokens ok: 'SELECT 1' -> {n} tokens (haiku)")

    # Live cell only if ladder is importable AND a question/gold exists.
    try:
        import ladder as _ladder  # noqa: F401
        _have_ladder = True
    except Exception as e:  # noqa: BLE001
        _have_ladder = False
        print(f"ladder not importable yet ({e}); skipping live run_cell self-test")

    if _have_ladder:
        import os
        ds = "jaffle-shop"
        qpath = config.questions_path(ds)
        if not os.path.exists(qpath):
            print(f"no questions yet at {qpath}; skipping live run_cell self-test")
        else:
            import yaml
            with open(qpath, encoding="utf-8") as f:
                qs = yaml.safe_load(f) or []
            qrow = next((q for q in qs if not q.get("gap")), None)
            if qrow is None:
                print("no non-gap question available; skipping live run_cell")
            else:
                rec = run_cell("haiku", ds, qrow, "metrics", 0)
                keys = list(rec.keys())
                print("run_cell returned keys:", keys)
                print("sample record:", {k: rec[k] for k in
                      ("id", "model", "level", "exec_ok", "correct",
                       "context_tokens", "input_tokens", "output_tokens",
                       "thinking_tokens", "cost_usd", "latency_s", "error")})
