"""A gradient picture as a second opinion: contrastive integrated gradients.

The quantity explained is the same contrast as everywhere else, M = log P(follows) - log P(violates)
(whole continuations) at one decision. Integrated gradients (Sundararajan et al. 2017) attribute
M(x) - M(baseline) to the input embeddings: for token i, (x_i - b_i) . mean over the path of dM/dx_i.
The baseline is the same prompt with every content token (context, question, SQL so far) replaced by
a filler token, a space, and the template tokens kept, so those get exactly zero. Not the pad
token: its embedding row is zero, and Gemma normalises the residual stream (RMSNorm) before every
block, so the model is not continuous at a zero input and the path integral never reaches M(pad)
(measured on the random fixture: the gap does not shrink with more steps). Per token the embedding
dimensions are summed; per unit the tokens are summed. `completeness_gap` is what the Riemann sum
and the bf16 arithmetic lose: sum of attributions vs M(x) - M(baseline).

Gradient x input at the real prompt (one backward pass, no path) is kept alongside as the cheap
variant. Both are signed: positive pushes towards the house convention.
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch
from wb.hf import base_model, model_device

from wb.heads import demo_pairs, picture_list
from wb.model import Head
from wb.scenario import Demo
from wb.steps import aligned, base_meta, questions_json
from wb.tokens import ALTS, Aligned, special_token_ids, token_labels, unit_spans


def _terms(model, head: Head, x: torch.Tensor, al: Aligned, embed):
    """M as differentiable functions of the embeddings `x` [n, D] of the common sequence: one
    (sign, value) per forward pass. M is their signed sum; yielding them one at a time lets the
    caller backpropagate each pass and free its graph before the next (two graphs of a 12B model at
    800 tokens do not fit next to the weights on a 48 GB card)."""
    dev = x.device
    n = x.shape[0]
    if all(len(al.cont[a]) == 1 for a in ALTS):  # one pass, two logits: the normaliser cancels
        h = base_model(model)(inputs_embeds=x[None], use_cache=False).last_hidden_state[0, -1]
        z = head.rows(h[None], torch.tensor([al.cont["follows"][0], al.cont["violates"][0]], device=dev))[0]
        yield 1.0, z[0] - z[1]
        return
    for sign, alt in ((1.0, "follows"), (-1.0, "violates")):
        cont = al.cont[alt]
        seq = x
        if len(cont) > 1:
            seq = torch.cat([x, embed(torch.tensor(cont[:-1], device=dev)).to(x.dtype)], dim=0)
        h = base_model(model)(inputs_embeds=seq[None], use_cache=False).last_hidden_state[0, n - 1 :]
        lp = head.logprobs(h)
        yield sign, lp.gather(-1, torch.tensor(cont, device=dev)[:, None]).sum()


def _metric(model, head: Head, x: torch.Tensor, al: Aligned, embed) -> float:
    with torch.no_grad():
        return sum(sign * float(v) for sign, v in _terms(model, head, x, al, embed))


def _grad(model, head: Head, x: torch.Tensor, al: Aligned, embed) -> torch.Tensor:
    """dM/dx [n, D] in float32, one backward pass per forward pass."""
    x = x.detach().requires_grad_(True)
    total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    with torch.enable_grad():
        for sign, v in _terms(model, head, x, al, embed):
            (g,) = torch.autograd.grad(v, x)
            total += sign * g.float()
            del v, g
    return total


def integrated_gradients(
    model, al: Aligned, head: Head, *, keep: list[bool], steps: int = 16, baseline_id: int = 0
) -> dict:
    """Per-token attributions of the common sequence. `keep[i]` = token i is not replaced in the
    baseline (template tokens)."""
    dev = model_device(model)
    embed = base_model(model).embed_tokens  # scaled word embeddings, what `inputs_embeds` expects
    frozen = [p for p in model.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            ids = torch.tensor(al.common, device=dev)
            x = embed(ids)
            b = x.clone()
            mask = torch.tensor(keep, device=dev)
            b[~mask] = embed(torch.tensor([baseline_id], device=dev))[0]
            delta = (x - b).float()
        m_x, m_b = _metric(model, head, x, al, embed), _metric(model, head, b, al, embed)
        total = torch.zeros_like(delta)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            total += _grad(model, head, (b.float() + alpha * delta).to(x.dtype), al, embed)
        g1 = _grad(model, head, x, al, embed)
        ig = (delta * total / steps).sum(-1)
        gxi = (x.float() * g1).sum(-1)
        return {
            "ig": ig.cpu().numpy(),
            "grad_x_input": gxi.cpu().numpy(),
            "grad_norm": g1.norm(dim=-1).cpu().numpy(),
            "m": m_x,
            "m_baseline": m_b,
            "completeness_gap": float(ig.sum()) - (m_x - m_b),
        }
    finally:
        for p in frozen:
            p.requires_grad_(True)


def convention_tokens(sc: Demo, built, enc, spans: dict, conv: str) -> list[int]:
    """Token positions that carry the convention in a built prompt: the convention's rule sentence
    (or the one-line rule), and in each example's SQL the tokens that differ from the same SQL with
    the convention flipped. Where the flipped SQL only ADDS text (a broken example lacks the
    filter), the evidence is an absence: the tokens on both sides of that point are taken."""
    import difflib

    from wb.tokens import token_span

    out: set[int] = set()
    prompt = enc.text[: enc.prompt_chars]
    uses = iter(built.spec.examples)
    cursor = 0
    for u in built.units:
        cs = prompt.find(u.text, cursor)
        cursor = cs + len(u.text)
        if u.kind == "rule" and u.id in (f"rule.{conv}", "rule.one_line"):
            out |= set(range(*spans[u.id]))
        if u.kind != "example_sql":
            continue
        use = next(uses)
        item = sc.td.example(use.id)
        if conv not in sc.td.applies(item):
            continue
        flipped = sc.td.variant(item, set(use.corrupt) ^ {conv})
        words = [(m.start(), m.end()) for m in re.finditer(r"\w+|\s+|[^\w\s]", u.text)]  # diff on words, not characters
        mine, theirs = [u.text[a:b] for a, b in words], re.findall(r"\w+|\s+|[^\w\s]", flipped)
        for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, mine, theirs, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            if i2 > i1:
                lo, hi = words[i1][0], words[i2 - 1][1]
            else:  # an insertion point: the non-blank words on both sides of it
                left = next((k for k in range(i1 - 1, -1, -1) if mine[k].strip()), 0)
                right = next((k for k in range(i1, len(mine)) if mine[k].strip()), len(mine) - 1)
                lo, hi = words[left][0], words[right][1]
            a, b = token_span(enc.offsets, cs + lo, cs + hi)
            out |= {i for i in range(a, b) if spans[u.id][0] <= i < spans[u.id][1]}
    return sorted(out)


def run_gradients(sc: Demo, model, tok, opts: dict) -> dict:
    """Integrated gradients at the decision token in the four picture contexts, for the picture
    question of each convention.

    `ig_baseline`: `convention` (default) moves only the tokens that carry the convention (see
    `convention_tokens`) from a filler to their real embedding, everything else staying as written:
    the path is short, and the attributions split M(prompt) - M(those tokens blanked) among the
    examples and the rule. `blank` moves every content token of the prompt: on the 12B in bf16 that
    path sum does not close (hundreds of nats off), see the `gradcheck` step."""
    t0 = time.perf_counter()
    head = Head(model)
    steps = int(opts.get("ig_steps", 32))
    mode = opts.get("ig_baseline", "convention")
    specials = special_token_ids(tok)
    filler = opts.get("ig_baseline_text", " ")
    pad = tok(filler, add_special_tokens=False)["input_ids"][0]
    pairs = demo_pairs(sc, opts, opts.get("questions_per_convention", 1))
    if not opts.get("contexts"):  # the four contexts of the picture unless the run names others
        opts = {**opts, "contexts": ["examples_clean", "one_example_broken", "rule_only", "rule_vs_examples"]}
    cells = []
    for c, q in pairs:
        facts = sc.picture_facts(c, opts.get("picture_break"), opts.get("picture_other"))
        cell = {"convention": c, "question": q.id, **facts, "contexts": {}}
        for name, spec in picture_list(sc, c, opts):
            t1 = time.perf_counter()
            built = sc.build(q, spec)
            al = aligned(sc, tok, built, q, c, opts)
            enc = al.enc["follows"]
            n = len(al.common)
            labels = token_labels(enc, unit_spans(enc, built), specials)[:n]
            order = [u.id for u in built.units] + ["other", "special", "output"]
            if mode == "convention":
                moved = set(convention_tokens(sc, built, enc, unit_spans(enc, built), c))
                keep = [i not in moved for i in range(n)]
            else:
                keep = [lab == "special" for lab in labels]
            try:
                res = integrated_gradients(model, al, head, keep=keep, steps=steps, baseline_id=pad)
            except torch.OutOfMemoryError as e:  # one prompt too long for the card must not stop the others
                torch.cuda.empty_cache()
                cell["contexts"][name] = {"error": f"out of memory: {str(e)[:120]}"}
                print(f"  gradients: {c} / {q.id} / {name}: OUT OF MEMORY", flush=True)
                continue
            lab = np.array(labels)
            cell["contexts"][name] = {
                "n_tokens": n,
                "tokens": enc.pieces(0, n),
                "labels": labels,
                "unit_order": order,
                "units": [{"id": u.id, "kind": u.kind, "group": u.group, "label": u.label} for u in built.units],
                "m": res["m"],
                "m_baseline": res["m_baseline"],
                "completeness_gap": res["completeness_gap"],
                "n_moved_tokens": int(n - sum(keep)),
                "token_maps": {k: np.round(res[k], 5).tolist() for k in ("ig", "grad_x_input", "grad_norm")},
                "unit_sum": {
                    k: [round(float(res[k][lab == u].sum()), 5) for u in order] for k in ("ig", "grad_x_input")
                },
                "elapsed_s": round(time.perf_counter() - t1, 1),
            }
            print(
                f"  gradients: {c} / {q.id} / {name}: M {res['m']:+.3f}, baseline {res['m_baseline']:+.3f}, "
                f"gap {res['completeness_gap']:+.3f}, {cell['contexts'][name]['elapsed_s']}s",
                flush=True,
            )
        cells.append(cell)
    head.release()
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "ig_steps": steps,
            "ig_baseline": mode,
            "baseline": f"the embedding of {tok.decode([pad])!r} (token {pad}) at "
            + (
                "the tokens that carry the convention (its rule sentence; in each example the tokens that differ from the flipped SQL); every other token as written"
                if mode == "convention"
                else "every content token (context, question, SQL so far); template tokens unchanged"
            ),
            "maps": "ig / grad_x_input: signed, in nats of M = log P(follows) - log P(violates); sum of ig = M - M(baseline) + completeness_gap",
        },
        "questions": questions_json(sc, list({q.id: q for _, q in pairs}.values())),
        "cells": cells,
    }


def run_gradcheck(sc: Demo, model, tok, opts: dict) -> dict:
    """Why does the path sum of integrated gradients close or not? For each picture cell: M along
    the straight path from the baseline (alpha = 0) to the prompt (alpha = 1), the slope the
    gradients claim at the midpoints (dM/dalpha = gradient . (x - baseline)), and the slope the
    forward passes show (finite differences). A wild M(alpha) means the path needs more steps; a
    smooth M(alpha) with slopes that disagree means the bf16 gradients are not accurate enough."""
    t0 = time.perf_counter()
    head = Head(model)
    n = int(opts.get("path_points", 32))
    specials = special_token_ids(tok)
    filler = tok(opts.get("ig_baseline_text", " "), add_special_tokens=False)["input_ids"][0]
    dev = model_device(model)
    embed = base_model(model).embed_tokens
    if not opts.get("contexts"):
        opts = {**opts, "contexts": ["examples_clean", "rule_only"]}
    cells = []
    for c, q in demo_pairs(sc, opts, opts.get("questions_per_convention", 1)):
        for name, spec in picture_list(sc, c, opts):
            built = sc.build(q, spec)
            al = aligned(sc, tok, built, q, c, opts)
            enc = al.enc["follows"]
            labels = token_labels(enc, unit_spans(enc, built), specials)[: len(al.common)]
            with torch.no_grad():
                x = embed(torch.tensor(al.common, device=dev))
                b = x.clone()
                if opts.get("ig_baseline", "convention") == "convention":
                    moved = set(convention_tokens(sc, built, enc, unit_spans(enc, built), c))
                    keep = torch.tensor([i not in moved for i in range(len(al.common))], device=dev)
                else:
                    keep = torch.tensor([lab == "special" for lab in labels], device=dev)
                b[~keep] = embed(torch.tensor([filler], device=dev))[0]
                delta = (x - b).float()
            at = lambda a: (b.float() + a * delta).to(x.dtype)  # noqa: E731
            m = [_metric(model, head, at(k / n), al, embed) for k in range(n + 1)]
            slopes = []
            for k in range(n):
                g = _grad(model, head, at((k + 0.5) / n), al, embed)
                slopes.append(float((g.double() * delta.double()).sum()))
            fd = [(m[k + 1] - m[k]) * n for k in range(n)]
            cells.append(
                {
                    "convention": c,
                    "question": q.id,
                    "context": name,
                    "ig_baseline": opts.get("ig_baseline", "convention"),
                    "n_tokens": len(al.common),
                    "n_moved_tokens": int((~keep).sum()),
                    "m_path": [round(v, 3) for v in m],
                    "slope_from_gradients": [round(v, 2) for v in slopes],
                    "slope_from_forward_passes": [round(v, 2) for v in fd],
                    "total_variation": float(sum(abs(m[k + 1] - m[k]) for k in range(n))),
                    "path_sum": float(sum(slopes) / n),
                    "m_end_minus_start": m[-1] - m[0],
                }
            )
            print(
                f"  gradcheck: {c} / {q.id} / {name}: M end - start {m[-1] - m[0]:+.2f}, path sum {sum(slopes) / n:+.2f}, "
                f"total variation of M {cells[-1]['total_variation']:.1f}",
                flush=True,
            )
    head.release()
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "path_points": n,
            "note": "m_path[k] = M at alpha = k / n; the two slope lists are dM/dalpha on the n intervals",
        },
        "cells": cells,
    }
