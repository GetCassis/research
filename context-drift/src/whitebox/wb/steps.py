"""The behavioural steps: decision-token probabilities, the grid, "changed in one place",
surprisal, and the sampled check. Each returns a JSON-ready dict (`run.py` writes it)."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
import torch
from wb.hf import model_device, text_config

from wb.model import Head, model_facts, score_labels, two_way
from wb.scenario import Demo
from wb.tokens import ALTS, Aligned, PrefixError, align_decision, encode, render_prompt, unit_spans

LN2 = math.log(2.0)
THINKING = (
    "disabled: the HF chat template is rendered with enable_thinking=False, so the prompt ends with an "
    "empty thought channel and the SQL is what follows the prompt"
)


def base_meta(sc: Demo, model, opts: dict) -> dict:
    return {
        **sc.describe(),
        "thinking": THINKING,
        "output_prefix": opts.get("output_prefix", ""),
        "continuation": "the scenario's follows_head / violates_head (first whitespace-delimited chunk)",
        "units": "log-probabilities in nats; diff = log P(follows) - log P(violates); surprisal in bits",
        **(model_facts(model) if model is not None else {}),
    }


def questions_json(sc: Demo, questions) -> list[dict]:
    return [
        {
            "id": q.id,
            "text": q.question,
            "kind": "near" if q.near else "far",
            "near": q.near,
            "near_example": sc.example_number(q.near) if sc.near(q) else None,
            "applies": sc.applies(q),
            "reference_sql": q.sql,
        }
        for q in questions
    ]


def pick_questions(sc: Demo, opts: dict, n_far: int | None = None, n_near: int | None = None):
    """`--questions a,b` or `all`; else the first `n_far` far and `n_near` near ones (None = all)."""
    ids = opts.get("questions")
    if ids and ids != ["all"]:
        return [sc.question(i) for i in ids]
    if ids == ["all"]:
        return list(sc.questions)
    far = [q for q in sc.questions if not q.near]
    near = [q for q in sc.questions if q.near]
    return far[:n_far] + near[:n_near]


def aligned(sc: Demo, tok, built, question, convention: str, opts: dict) -> Aligned:
    """The decision of (question, convention) on one built prompt, in tokens. When one head is a
    token prefix of the other, both are extended by one more chunk until they differ."""
    cache = opts.setdefault("_extend", {})
    key = (question.id, convention)
    for extend in range(cache.get(key, 0), 6):
        try:
            al = align_decision(tok, built, sc.decision(question, convention, extend), opts.get("output_prefix", ""))
        except PrefixError:
            continue
        cache[key] = extend
        return al
    raise ValueError(f"{key}: the two continuations never differ")


# ---------------------------------------------------------------------------- decision scoring


def decision_row(al: Aligned, res: dict[str, dict], tok) -> dict:
    """The measurement at one decision: whole-continuation log-probabilities and their difference."""
    lp = {a: float(res[a]["logp"].sum()) for a in ALTS}
    first = {a: float(res[a]["logp"][0]) for a in ALTS}
    row = {
        "lp_follows": lp["follows"],
        "lp_violates": lp["violates"],
        "diff": lp["follows"] - lp["violates"],
        "p_follow": two_way(lp["follows"], lp["violates"]),
        "mass": math.exp(lp["follows"]) + math.exp(lp["violates"]),  # how much of the model's choice the two cover
        "first_diff": first["follows"] - first["violates"],  # the decision token alone
        "n_prompt_tokens": al.prompt_len,
        "n_common_tokens": len(al.common),
    }
    if "topk" in res["follows"]:
        row["top"] = [[tok.decode([i]), round(math.exp(v), 4)] for i, v in res["follows"]["topk"]]
    return row


def score_decisions(
    model, tok, jobs: list[tuple[dict, Aligned]], *, opts: dict, label: str, topk: int = 5
) -> list[dict]:
    """`jobs`: (row keys, alignment). Two teacher-forced sequences per job, batched and deduplicated."""
    head = Head(model)
    items = []
    for _, al in jobs:
        for alt in ALTS:
            items.append((al.full(alt), al.label_positions(alt)))
    res = score_labels(
        model, items, head=head, max_batch_tokens=opts.get("max_batch_tokens", 16384), topk=topk, progress=label
    )
    rows = []
    for k, (keys, al) in enumerate(jobs):
        rows.append({**keys, **decision_row(al, {"follows": res[2 * k], "violates": res[2 * k + 1]}, tok)})
    head.release()
    return rows


def describe_decisions(sc: Demo, tok, questions, opts: dict) -> dict:
    """Per question and convention: how the decision point fell on tokens (no-context prompt).

    Two cautions are computed here because the read-out is teacher-forced on the reference SQL:
    `conditioned_on` lists the conventions whose own decision lies earlier in the prefix (the SQL
    so far already follows them, which can carry the next one along), and `whitespace_decision` is
    true when one alternative starts with a whitespace-only token (a space against a newline: the
    token then decides the layout at least as much as the convention)."""
    out: dict = {}
    for q in questions:
        built = sc.build(q, sc.spec())
        out[q.id] = {}
        split = {c: len(sc.decision(q, c).prefix) for c in sc.applies(q)}
        for c in sc.applies(q):
            al = aligned(sc, tok, built, q, c, opts)
            d = sc.decision(q, c, opts["_extend"][(q.id, c)])
            desc = al.describe(tok)
            desc.pop("n_prompt_tokens"), desc.pop("n_common_tokens")
            out[q.id][c] = {
                "prefix": d.prefix,
                "follows_text": d.follows,
                "violates_text": d.violates,
                **desc,
                "conditioned_on": [c2 for c2 in sc.applies(q) if split[c2] < split[c]],
                "whitespace_decision": any(not desc[a][0].strip() for a in ALTS),
            }
    return out


# ---------------------------------------------------------------------------- 1-2: the grid


def conditions_for(sc: Demo, opts: dict) -> list:
    """The grid's conditions. `picture=<convention>` adds that convention's picture contexts as
    conditions named `picture:<context>` (`picture_break` / `picture_other` choose the examples);
    `conditions=[...]` keeps the named ones, a name ending with ":" being a prefix."""
    from wb.scenario import Condition

    out = list(sc.grid_conditions())
    conv = opts.get("picture")
    if conv:
        for name, spec in sc.picture_contexts(conv, opts.get("picture_break"), opts.get("picture_other")):
            out.append(Condition(f"picture:{name}", spec, "picture", [conv]))
    keep = opts.get("conditions")
    if keep:
        out = [c for c in out if any(c.name == k or (k.endswith(":") and c.name.startswith(k)) for k in keep)]
    return out


def run_grid(sc: Demo, model, tok, opts: dict) -> dict:
    """Probability of following each convention, under every grid condition, on every test question."""
    t0 = time.perf_counter()
    questions = pick_questions(sc, opts)
    conditions = conditions_for(sc, opts)
    jobs = []
    for cond in conditions:
        for q in questions:
            built = sc.build(q, cond.spec)
            for c in sc.applies(q):
                jobs.append(
                    ({"condition": cond.name, "question": q.id, "convention": c}, aligned(sc, tok, built, q, c, opts))
                )
    rows = score_decisions(model, tok, jobs, opts=opts, label="grid")
    window = getattr(text_config(model), "sliding_window", None)
    longest = max(r["n_common_tokens"] for r in rows)
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "max_common_tokens": longest,
            "over_window": bool(window) and longest > window,
        },
        "conditions": [
            {"name": c.name, "family": c.family, "about": c.about, "spec": c.spec.to_json()} for c in conditions
        ],
        "questions": questions_json(sc, questions),
        "decisions": describe_decisions(sc, tok, questions, opts),
        "rows": rows,
    }


# ---------------------------------------------------------------------------- 3: changed in one place


def run_perturb(sc: Demo, model, tok, opts: dict) -> dict:
    """Break-one / restore-one / k of n / the same with the rule, per convention and question."""
    t0 = time.perf_counter()
    questions = pick_questions(sc, opts, n_far=opts.get("n_far", 5))
    conventions = opts.get("conventions") or sc.conventions
    jobs = []
    for c in conventions:
        if not sc.carriers(c):
            continue
        for name, spec, facts in sc.perturbations(c):
            for q in questions:
                if c in sc.applies(q):
                    keys = {"convention": c, "context": name, "question": q.id, **facts}
                    jobs.append((keys, aligned(sc, tok, sc.build(q, spec), q, c, opts)))
    rows = score_decisions(model, tok, jobs, opts=opts, label="perturb", topk=0)
    near = {q.id: sc.example_number(q.near) for q in questions if sc.near(q)}
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "rule_states": {"0": "no rules", "+": "all rules, clean", "-": "all rules, this convention's rule wrong"},
            "context_names": "r<rule state>|e<one bit per carrier example, 1 = clean, 0 = convention broken>, or |none (no examples)",
        },
        "questions": questions_json(sc, questions),
        "rows": rows,
        "summary": summarize_perturb(rows, near),
    }


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def summarize_perturb(rows: list[dict], near: dict[str, int]) -> dict:
    """Per convention and rule state, means over questions.

    - dose[k]: k clean carrier examples of n (mean over the subsets run at that size);
    - break_one[i]: all clean but example i: change of diff against all clean (necessity of i);
    - restore_one[i]: all broken but example i: change of diff against all broken (sufficiency of i);
    - near: for the "near" questions, the same two effects for the nearest example and for the others."""
    by: dict = defaultdict(lambda: defaultdict(dict))  # (conv, rule) -> question -> clean tuple -> row
    rule_only: dict = defaultdict(dict)
    for r in rows:
        if not r["examples"]:
            rule_only[(r["convention"], r["rule"])][r["question"]] = r
        else:
            by[(r["convention"], r["rule"])][r["question"]][tuple(r["clean"])] = r
    out: dict = {}
    for (conv, rk), per_q in sorted(by.items()):
        any_q = next(iter(per_q.values()))
        full = max(any_q, key=len)
        n = len(full)
        dose = {}
        for k in range(n + 1):
            sel = [r for qr in per_q.values() for cl, r in qr.items() if len(cl) == k]
            dose[k] = {
                "p": _mean([r["p_follow"] for r in sel]),
                "diff": _mean([r["diff"] for r in sel]),
                "n_subsets": len({tuple(r["clean"]) for r in sel}),
            }
        break_one, restore_one, nr = {}, {}, defaultdict(list)
        for i in full:
            others = tuple(j for j in full if j != i)
            break_one[i] = {
                "d_diff": _mean([qr[others]["diff"] - qr[full]["diff"] for qr in per_q.values()]),
                "p": _mean([qr[others]["p_follow"] for qr in per_q.values()]),
                "per_question": {q: qr[others]["diff"] - qr[full]["diff"] for q, qr in per_q.items()},
            }
            restore_one[i] = {
                "d_diff": _mean([qr[(i,)]["diff"] - qr[()]["diff"] for qr in per_q.values()]),
                "p": _mean([qr[(i,)]["p_follow"] for qr in per_q.values()]),
                "per_question": {q: qr[(i,)]["diff"] - qr[()]["diff"] for q, qr in per_q.items()},
            }
        for qid, qr in per_q.items():
            if near.get(qid) not in full:
                continue
            for i in full:
                others = tuple(j for j in full if j != i)
                tag = "nearest" if i == near[qid] else "other"
                nr[f"break_{tag}"].append(qr[others]["diff"] - qr[full]["diff"])
                nr[f"restore_{tag}"].append(qr[(i,)]["diff"] - qr[()]["diff"])
        ro = rule_only.get((conv, rk), {})
        out[f"{conv}|rule{rk}"] = {
            "carriers": list(full),
            "n_questions": len(per_q),
            "no_examples": {
                "p": _mean([r["p_follow"] for r in ro.values()]),
                "diff": _mean([r["diff"] for r in ro.values()]),
            },
            "all_clean": {k: dose[n][k] for k in ("p", "diff")},
            "all_broken": {k: dose[0][k] for k in ("p", "diff")},
            "dose": dose,
            "break_one": break_one,
            "restore_one": restore_one,
            "near": {k: {"mean": _mean(v), "n": len(v)} for k, v in nr.items()},
        }
    return out


# ---------------------------------------------------------------------------- 4: surprisal


def end_of_turn(tok) -> str:
    """The text the chat template writes after an assistant message (`<turn|>` for Gemma 4)."""
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ZZZ"}]
    tail = tok.apply_chat_template(msgs, tokenize=False).rsplit("ZZZ", 1)[1]
    return tail.strip() or (tok.eos_token or "")


def run_surprisal(sc: Demo, model, tok, opts: dict) -> dict:
    """Per-token surprisal of the reference SQL under each grid condition, and the bits saved
    against no context: where in the output the conventions live, and what a context buys per token."""
    t0 = time.perf_counter()
    questions = pick_questions(sc, opts)
    conditions = sc.grid_conditions()
    prefix, eot = opts.get("output_prefix", ""), end_of_turn(tok)
    items, keys, tokens, decision_tokens = [], [], {}, {}
    for q in questions:
        for cond in conditions:
            built = sc.build(q, cond.spec)
            enc = encode(tok, built.messages, q.sql + eot, prefix)
            sql_start = next(i for i, (a, _) in enumerate(enc.offsets) if a >= enc.output_start)
            pieces = enc.pieces(sql_start)
            if q.id in tokens and tokens[q.id] != pieces:
                raise ValueError(f"{q.id}: the reference SQL tokenises differently under {cond.name}")
            tokens[q.id] = pieces
            if q.id not in decision_tokens:
                decision_tokens[q.id] = {}
                for c in sc.applies(q):
                    al = aligned(sc, tok, built, q, c, opts)
                    n = len(al.common)
                    if enc.ids[: n + 1] != al.full("follows")[: n + 1]:
                        raise ValueError(f"{q.id}/{c}: the reference SQL does not continue with `follows`")
                    decision_tokens[q.id][c] = n - sql_start
            items.append((enc.ids, list(range(sql_start, len(enc.ids)))))
            keys.append({"condition": cond.name, "question": q.id, "n_prompt_tokens": enc.prompt_len})
    res = score_labels(model, items, max_batch_tokens=opts.get("max_batch_tokens", 16384), progress="surprisal")
    rows = []
    for k, r in zip(keys, res):
        bits = (-r["logp"] / LN2).tolist()
        rows.append(
            {**k, "bits": [round(b, 4) for b in bits], "total_bits": float(sum(bits[:-1])), "eot_bits": bits[-1]}
        )
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "note": "bits[i] = -log2 p(token i | prompt, tokens before); the last token is the end of turn "
            "and is left out of total_bits",
        },
        "conditions": [{"name": c.name, "family": c.family, "about": c.about} for c in conditions],
        "questions": questions_json(sc, questions),
        "tokens": tokens,
        "decision_token": decision_tokens,
        "rows": rows,
        "summary": summarize_surprisal(rows, decision_tokens),
    }


def summarize_surprisal(rows: list[dict], decision_tokens: dict) -> dict:
    """Per condition, mean over questions: total bits, bits saved against `none`, the context's cost
    in prompt tokens, bits saved per 100 context tokens, and the bits saved AT the decision tokens."""
    none = {r["question"]: r for r in rows if r["condition"] == "none"}
    per: dict = defaultdict(list)
    for r in rows:
        base = none.get(r["question"])
        if base is None:
            continue
        saved = [a - b for a, b in zip(base["bits"][:-1], r["bits"][:-1])]
        dec = sorted(set(decision_tokens[r["question"]].values()))
        per[r["condition"]].append(
            {
                "total_bits": r["total_bits"],
                "saved_bits": sum(saved),
                "saved_at_decisions": sum(saved[i] for i in dec if i < len(saved)),
                "context_tokens": r["n_prompt_tokens"] - base["n_prompt_tokens"],
                "eot_bits": r["eot_bits"],
            }
        )
    out = {}
    for cond, xs in per.items():
        m = {k: float(np.mean([x[k] for x in xs])) for k in xs[0]}
        m["saved_per_100_context_tokens"] = (
            100.0 * m["saved_bits"] / m["context_tokens"] if m["context_tokens"] else None
        )
        m["n_questions"] = len(xs)
        out[cond] = m
    return out


# ---------------------------------------------------------------------------- 7: sampled check


@torch.no_grad()
def generate(model, tok, messages: list[dict], n: int, *, opts: dict, seed: int) -> list[str]:
    """`n` sampled answers to one prompt, at the model's own sampling defaults (temperature 1,
    top-k 64, top-p 0.95 in Gemma 4's generation_config), thinking off."""
    dev = model_device(model)
    ids = tok(render_prompt(tok, messages) + opts.get("output_prefix", ""), add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device=dev).repeat(n, 1)
    eot_id = tok(end_of_turn(tok), add_special_tokens=False)["input_ids"][0]
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=x,
        attention_mask=torch.ones_like(x),
        do_sample=True,
        temperature=opts.get("temperature", 1.0),
        top_k=opts.get("top_k", 64),
        top_p=opts.get("top_p", 0.95),
        max_new_tokens=opts.get("max_new_tokens", 200),
        eos_token_id=sorted({tok.eos_token_id, eot_id}),
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
    )
    return [tok.decode(row[len(ids) :], skip_special_tokens=True).strip() for row in out]


def run_sample(sc: Demo, model, tok, opts: dict) -> dict:
    """Does the decision-token probability predict what the model writes? `n` generations per cell
    of a subset of the grid, scored with the scenario's checkers, next to the teacher-forced numbers."""
    t0 = time.perf_counter()
    n = int(opts.get("n", 20))
    questions = pick_questions(sc, opts, n_far=opts.get("sample_far", 3), n_near=opts.get("sample_near", 3))
    conditions = conditions_for(sc, opts)
    jobs, cells = [], []
    for ci, cond in enumerate(conditions):
        for qi, q in enumerate(questions):
            built = sc.build(q, cond.spec)
            texts = generate(model, tok, built.messages, n, opts=opts, seed=opts.get("seed", 0) + 1000 * ci + qi)
            verdicts = [sc.check(q, t) for t in texts]
            cells.append(
                {
                    "condition": cond.name,
                    "question": q.id,
                    "n": n,
                    "fenced": sum(t.lstrip().startswith("```") for t in texts),
                    "sampled": {
                        c: {
                            "follows": sum(v.get(c) is True for v in verdicts),
                            "violates": sum(v.get(c) is False for v in verdicts),
                            "other": sum(v.get(c) is None for v in verdicts),
                        }
                        for c in sc.applies(q)
                    },
                    "texts": texts,
                }
            )
            for c in sc.applies(q):
                jobs.append(
                    ({"condition": cond.name, "question": q.id, "convention": c}, aligned(sc, tok, built, q, c, opts))
                )
            print(f"  sample: {cond.name} / {q.id} ({len(cells)}/{len(conditions) * len(questions)})", flush=True)
    scored = {
        (r["condition"], r["question"], r["convention"]): r
        for r in score_decisions(model, tok, jobs, opts=opts, label="sample", topk=0)
    }
    pairs = []
    for cell in cells:
        for c, s in cell["sampled"].items():
            r = scored[(cell["condition"], cell["question"], c)]
            decided = s["follows"] + s["violates"]
            pairs.append(
                {
                    "condition": cell["condition"],
                    "question": cell["question"],
                    "convention": c,
                    "p_follow": r["p_follow"],
                    "mass": r["mass"],
                    "sampled_follow": s["follows"] / decided if decided else None,
                    "n_decided": decided,
                    "n": cell["n"],
                }
            )
    ok = [p for p in pairs if p["sampled_follow"] is not None]
    summary = {
        "n_pairs": len(pairs),
        "n_pairs_decided": len(ok),
        "fenced_share": sum(c["fenced"] for c in cells) / max(1, n * len(cells)),
    }
    if len(ok) >= 3:
        x, y = np.array([p["p_follow"] for p in ok]), np.array([p["sampled_follow"] for p in ok])
        summary["mean_abs_gap"] = float(np.abs(x - y).mean())
        summary["same_side_of_half"] = float(((x >= 0.5) == (y >= 0.5)).mean())
        if x.std() > 0 and y.std() > 0:
            summary["pearson"] = float(np.corrcoef(x, y)[0, 1])
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "sampling": {
                k: opts.get(k, d)
                for k, d in (("temperature", 1.0), ("top_k", 64), ("top_p", 0.95), ("max_new_tokens", 200))
            },
            "note": "p_follow is teacher-forced on the reference SQL prefix (two-way, on the short continuations); "
            "sampled_follow is the share of the generations the checker could classify (n_decided) that follow",
        },
        "questions": questions_json(sc, questions),
        "pairs": pairs,
        "summary": summary,
        "cells": cells,
    }


# ---------------------------------------------------------------------------- info (tokenizer only)


def run_info(sc: Demo, tok, opts: dict, window: int = 1024) -> dict:
    """No model: token counts per context, unit spans, decision alignment, checker sanity."""
    prefix = opts.get("output_prefix", "")
    problems: list[str] = []
    conds = []
    specs = [(c.name, c.spec) for c in sc.grid_conditions()]
    for conv in sc.conventions:
        if sc.carriers(conv):
            specs += [(f"picture:{conv}:{n}", s) for n, s in sc.picture_contexts(conv)]
    for name, spec in specs:
        lens = []
        for q in sc.questions:
            built = sc.build(q, spec)
            enc = encode(tok, built.messages, q.sql, prefix)
            try:
                unit_spans(enc, built)
            except ValueError as e:
                problems.append(f"{name}/{q.id}: {e}")
            lens.append((enc.prompt_len, len(enc.ids)))
        worst = max(n for _, n in lens)
        conds.append({"name": name, "prompt_tokens_max": max(p for p, _ in lens), "with_reference_sql_max": worst})
        if worst > window:
            problems.append(f"{name}: {worst} tokens with the reference SQL, over the {window}-token sliding window")
    decisions = describe_decisions(sc, tok, sc.questions, opts)
    for q in sc.questions:
        ref = sc.check(q, q.sql)
        for c in sc.applies(q):
            d = sc.decision(q, c)
            info = decisions[q.id][c]
            if not q.sql.startswith(d.prefix + d.follows):
                problems.append(f"{q.id}/{c}: the reference SQL does not start with prefix + follows")
            if ref.get(c) is not True:
                problems.append(f"{q.id}/{c}: the checker does not accept the reference SQL")
            if sc.check(q, d.prefix + d.rest_violates).get(c) is not False:
                problems.append(f"{q.id}/{c}: the checker does not reject the corrupted reference SQL")
            if info["shift_chars"]:
                info["note"] = (
                    f"the decision token starts {info['shift_chars']:+d} chars from the scenario's split (token boundaries)"
                )
    return {
        "meta": {**base_meta(sc, None, opts), "window": window, "end_of_turn": end_of_turn(tok)},
        "n_questions": len(sc.questions),
        "n_decisions": sum(len(v) for v in decisions.values()),
        "contexts": conds,
        "decisions": decisions,
        "problems": problems,
    }
