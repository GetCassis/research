"""Attention at the decision token, from heads selected by a causal test.

The causal test (`run_heads`) is activation patching between two prompts that differ only in what
the examples carry: CLEAN (every example follows the house convention) and CORRUPT (every example
that shows the convention carries the other one). A head's output is the slice of the input of the
layer's `o_proj` that belongs to it, `[.., h * head_dim : (h + 1) * head_dim]`: that is the one place
where Gemma 4's heads are separate. Patching head (l, h) means overwriting that slice, at some
positions of the RECIPIENT prompt, with the slice the DONOR prompt produced at the same tokens, and
reading the change of the metric (log P(follows) - log P(violates), whole continuations).

- denoise (sufficiency): recipient CORRUPT, donor CLEAN. effect = metric(patched) - metric(corrupt).
- noise (necessity): recipient CLEAN, donor CORRUPT. effect = metric(clean) - metric(patched).
- positions `last`: the decision position only (the token whose output picks the convention), which
  is the attention row the picture shows. `tail`: every position after the last token that differs
  between the two prompts (rest of the last example, question, generation prompt, SQL so far).
  Patching ALL heads over the tail reproduces the donor's metric exactly (the tail then reads the
  donor's context only), which is the test of this plumbing; all heads at `last` tells how much of
  the effect enters at the decision token itself.

What Gemma 4's layout allows (12B: 48 layers x 16 heads):

- every layer has 16 query heads, so 16 attention patterns per layer, in all 48 layers;
- 40 sliding-window layers (window 1024, 8 KV heads: heads 2g and 2g + 1 share keys and values) and
  8 full-attention layers (5, 11, ..., 47: ONE KV head and k = v, so their 16 heads read the same
  value vectors and differ by where they look and by their slice of o_proj). Per-head patching is
  well defined in both kinds. A sliding layer cannot see further than 1024 tokens back: the toy
  prompts are shorter (checked, `over_window`), so every layer sees the whole prompt;
- after o_proj comes an RMSNorm over the SUM of the heads, before the residual: head outputs are not
  additive in the residual stream, so a per-head "direct logit attribution" would be unsound. Only
  interventions are reported, and the share of the effect carried by the top heads is measured by
  patching them JOINTLY, never by adding single-head effects;
- no soft cap and scaling 1 in the attention itself: the eager attention rows are the real ones.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from wb.hf import attn_implementation, base_model, model_device, prefill, shared_prefix_cache, text_config
from transformers.cache_utils import DynamicCache

from wb.model import Head
from wb.scenario import Demo
from wb.steps import aligned, base_meta, questions_json
from wb.tokens import ALTS, Aligned, encode, special_token_ids, token_labels, unit_spans

CLEAN_CTX, CORRUPT_CTX = "examples_clean", "examples_broken"


# ---------------------------------------------------------------------------- a prompt cut in two


class Tail:
    """One decision on one prompt, split into a cached prefix and the rows that are re-run.

    rows: one per distinct teacher-forced continuation (`common[cut:] + cont[alt][:-1]`), right
    padded; `labels[alt]` = (row, [(index in the row, token id), ...]) of that alternative."""

    def __init__(self, model, al: Aligned, n_tail: int):
        self.model, self.al, self.n_tail = model, al, n_tail
        dev = model_device(model)
        cut = len(al.common) - n_tail
        if cut < 1:
            raise ValueError("the tail must leave at least one token in the prefix")
        self.prefix = prefill(model, torch.tensor([al.common[:cut]], device=dev), cache=DynamicCache())
        seqs: list[list[int]] = []
        self.labels: dict[str, tuple[int, list[tuple[int, int]]]] = {}
        for alt in ALTS:
            extra = al.cont[alt][:-1]
            if extra not in seqs:
                seqs.append(extra)
            self.labels[alt] = (seqs.index(extra), [(n_tail - 1 + i, t) for i, t in enumerate(al.cont[alt])])
        self.width = n_tail + max(len(s) for s in seqs)
        rows = np.zeros((len(seqs), self.width), dtype=np.int64)
        for r, extra in enumerate(seqs):
            rows[r, : n_tail + len(extra)] = al.common[cut:] + extra
        self.rows = torch.tensor(rows, device=dev)
        self.n_rows = len(seqs)

    def run(self, n_cfg: int = 1) -> torch.Tensor:
        """Hidden states [n_cfg * n_rows, width, D] of the rows on the cached prefix (hooks apply)."""
        x = self.rows.repeat(n_cfg, 1)
        return base_model(self.model)(
            input_ids=x, past_key_values=shared_prefix_cache(self.prefix), use_cache=True
        ).last_hidden_state

    def metric(self, hidden: torch.Tensor, head: Head) -> np.ndarray:
        """[n_cfg, 3]: log P(follows), log P(violates), their difference, per patch configuration."""
        n_cfg = hidden.shape[0] // self.n_rows
        h = hidden.view(n_cfg, self.n_rows, *hidden.shape[1:])
        out = np.zeros((n_cfg, 3))
        for k, alt in enumerate(ALTS):
            row, labels = self.labels[alt]
            idx = torch.tensor([i for i, _ in labels], device=hidden.device)
            tok = torch.tensor([t for _, t in labels], device=hidden.device)
            lp = head.logprobs(h[:, row][:, idx])  # [n_cfg, n_labels, V]
            out[:, k] = lp.gather(-1, tok[None, :, None].expand(n_cfg, -1, 1))[..., 0].sum(-1).double().cpu().numpy()
        out[:, 2] = out[:, 0] - out[:, 1]
        return out


def attn_modules(model) -> list:
    return [layer.self_attn for layer in base_model(model).layers]


class Hooks:
    """Forward pre-hooks on every layer's o_proj, removed on exit whatever happens."""

    def __init__(self, model, fn):
        self.mods, self.fn, self.handles = attn_modules(model), fn, []

    def __enter__(self):
        for li, m in enumerate(self.mods):
            self.handles.append(m.o_proj.register_forward_pre_hook(lambda mod, args, li=li: self.fn(li, args[0])))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def capture(tail: Tail, head: Head) -> tuple[list[torch.Tensor], np.ndarray]:
    """Run the rows unpatched: per layer the o_proj input [n_rows, width, H * d], and the metric."""
    acts: list[torch.Tensor | None] = [None] * len(attn_modules(tail.model))

    def fn(li, x):
        acts[li] = x.detach().clone()
        return None

    with Hooks(tail.model, fn):
        hidden = tail.run()
    return acts, tail.metric(hidden, head)[0]  # type: ignore[return-value]


@torch.no_grad()
def patched(
    tail: Tail,
    donor: list[torch.Tensor],
    configs: list[tuple[list[tuple[int, int]] | str, list[int] | str]],
    head: Head,
    *,
    max_rows: int = 96,
    max_batch_tokens: int = 8192,
) -> np.ndarray:
    """Metric [len(configs), 3] of the recipient `tail` under each patch configuration.

    A configuration is (heads, positions): heads = [(layer, head), ...] or "all"; positions =
    indices in the row or "all". The donor slices come from `capture` on the donor's tail, whose
    rows hold the same tokens."""
    dev = model_device(tail.model)
    cfg = text_config(tail.model)
    L, H, R, S = cfg.num_hidden_layers, cfg.num_attention_heads, tail.n_rows, tail.width
    per = max(1, min(max_rows // R, max_batch_tokens // (R * S)))
    out = np.zeros((len(configs), 3))
    for c0 in range(0, len(configs), per):
        chunk = configs[c0 : c0 + per]
        n = len(chunk)
        hmask = torch.zeros(L, n, H, dtype=torch.bool, device=dev)
        pmask = torch.zeros(n, S, dtype=torch.bool, device=dev)
        for k, (heads, positions) in enumerate(chunk):
            if isinstance(heads, str):
                hmask[:, k] = True
            else:
                for li, h in heads:
                    hmask[li, k, h] = True
            if isinstance(positions, str):
                pmask[k] = True
            else:
                pmask[k, positions] = True
        hmask_b = hmask.repeat_interleave(R, dim=1)  # [L, n * R, H]
        pmask_b = pmask.repeat_interleave(R, dim=0)  # [n * R, S]
        active = hmask.flatten(1).any(dim=1).tolist()

        def fn(li, x):
            if not active[li]:
                return None
            B = x.shape[0]
            x4 = x.view(B, S, H, -1)
            d4 = donor[li].view(R, S, H, -1).repeat(n, 1, 1, 1)
            sel = hmask_b[li][:, None, :, None] & pmask_b[:, :, None, None]
            return (torch.where(sel, d4, x4).view_as(x),)

        with Hooks(tail.model, fn):
            hidden = tail.run(n)
        out[c0 : c0 + n] = tail.metric(hidden, head)
        del hidden
    return out


def common_suffix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


# ---------------------------------------------------------------------------- the causal test


def patch_pair(model, tok, al_clean: Aligned, al_corrupt: Aligned, head: Head, opts: dict) -> dict:
    """All single heads, all heads of a layer, top-k jointly, and each tail position, between a
    clean and a corrupt prompt for one decision."""
    cfg = text_config(model)
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    kw = dict(max_rows=opts.get("max_rows", 128), max_batch_tokens=opts.get("patch_batch_tokens", 16384))
    n_tail = common_suffix(al_clean.common, al_corrupt.common)
    n_tail = min(n_tail, len(al_clean.common) - 1, len(al_corrupt.common) - 1)
    singles = [([(li, h)], "all") for li in range(L) for h in range(H)]
    ks = [k for k in (1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if k < L * H]
    out: dict = {"tail_len": n_tail, "modes": {}}
    directions = {"denoise": (al_corrupt, al_clean), "noise": (al_clean, al_corrupt)}
    for mode, width in (("last", 1), ("tail", n_tail)):
        out["modes"][mode] = {}
        tails = {"clean": Tail(model, al_clean, width), "corrupt": Tail(model, al_corrupt, width)}
        acts, base = {}, {}
        for name, t in tails.items():
            acts[name], base[name] = capture(t, head)
        if mode == "last":
            out.update(
                m_clean=base["clean"].tolist(),
                m_corrupt=base["corrupt"].tolist(),
                total=float(base["clean"][2] - base["corrupt"][2]),
            )
        # `last` patches the decision position only; `tail` every position of the rows
        where = [width - 1] if mode == "last" else "all"
        for direction, (rec_al, don_al) in directions.items():
            if mode == "tail" and direction == "noise" and not opts.get("tail_noise", False):
                continue
            rec, don = ("corrupt", "clean") if direction == "denoise" else ("clean", "corrupt")
            sign = 1.0 if direction == "denoise" else -1.0
            total = float(base[don][2] - base[rec][2]) * sign  # clean - corrupt in both directions
            t, d = tails[rec], acts[don]
            controls = patched(t, d, [([], "all"), ("all", where)], head, **kw)
            single = patched(t, d, [(hs, where) for hs, _ in singles], head, **kw)
            eff = (single[:, 2] - base[rec][2]) * sign
            order = np.argsort(-eff)
            layers = patched(t, d, [([(li, h) for h in range(H)], where) for li in range(L)], head, **kw)
            joint = patched(t, d, [([divmod(int(i), H) for i in order[:k]], where) for k in ks], head, **kw)
            res = {
                "total": total,
                "noise_floor": float(abs(controls[0, 2] - base[rec][2])),  # an unpatched row in a batch vs alone
                "all_heads": float((controls[1, 2] - base[rec][2]) * sign),
                "effect": np.round(eff.reshape(L, H), 5).tolist(),
                "effect_lp_follows": np.round((single[:, 0] - base[rec][0]).reshape(L, H) * sign, 5).tolist(),
                "per_layer": np.round((layers[:, 2] - base[rec][2]) * sign, 5).tolist(),
                "top": [[*divmod(int(i), H), round(float(eff[i]), 5)] for i in order[:32]],
                "cumulative": [
                    {
                        "k": k,
                        "effect": float((m - base[rec][2]) * sign),
                        "share": float((m - base[rec][2]) * sign / total) if total else None,
                    }
                    for k, m in zip(ks, joint[:, 2])
                ],
            }
            if mode == "tail":  # where in the tail the information enters: all heads at one position
                rows_pos = list(range(width))
                per_pos = patched(t, d, [("all", [p]) for p in rows_pos], head, **kw)
                pieces = [tok.decode([i]) for i in rec_al.common[-width:]]
                res["per_position"] = [
                    {
                        "rel": p - (width - 1),
                        "token": pieces[p],
                        "effect": round(float((per_pos[p, 2] - base[rec][2]) * sign), 5),
                    }
                    for p in rows_pos
                ]
            out["modes"][mode][direction] = res
        del tails, acts
    return out


def demo_pairs(sc: Demo, opts: dict, per_convention: int) -> list[tuple[str, object]]:
    """(convention, question): for each convention with carrier examples, far and near questions
    it applies to, interleaved; the first one is the picture question.

    Questions whose decision comes EARLIEST in the SQL go first: the read-out is conditional on the
    reference SQL so far, and a short prefix has committed to fewer of the other conventions."""
    wanted = opts.get("questions")
    out = []
    for c in opts.get("conventions") or sc.conventions:
        if not sc.carriers(c):
            continue
        qs = [q for q in sc.questions if c in sc.applies(q)]
        qs.sort(key=lambda q: len(sc.decision(q, c).prefix))  # stable: scenario order among equals
        if wanted and wanted != ["all"]:
            qs = [q for q in qs if q.id in wanted]
        elif wanted != ["all"]:
            far, near = [q for q in qs if not q.near], [q for q in qs if q.near]
            mixed = [q for pair in zip(far, near) for q in pair] + far[len(near) :] + near[len(far) :]
            qs = (far[:1] + [q for q in mixed if q is not far[0]] if far else mixed)[:per_convention]
        out += [(c, q) for q in qs]
    return out


def picture_list(sc: Demo, convention: str, opts: dict) -> list:
    """The picture contexts of a convention under the run's options (`picture_break`,
    `picture_other`: which example is the broken one and which the control; `contexts`: a subset)."""
    ctxs = sc.picture_contexts(convention, opts.get("picture_break"), opts.get("picture_other"))
    keep = opts.get("contexts")
    return [(n, s) for n, s in ctxs if not keep or n in keep]


def run_heads(sc: Demo, model, tok, opts: dict, out_dir: Path) -> dict:
    """The causal test on several (convention, question) pairs: which heads carry the examples'
    convention to the decision token. The two prompts are `clean_context` (default: the examples
    as written) and `corrupt_context` (default: every carrier example broken; `one_example_broken`
    asks which heads carry the effect of ONE changed example)."""
    t0 = time.perf_counter()
    head = Head(model)
    pairs = demo_pairs(sc, opts, opts.get("questions_per_convention", 3))
    clean_name, corrupt_name = opts.get("clean_context", CLEAN_CTX), opts.get("corrupt_context", CORRUPT_CTX)
    rows = []
    for i, (c, q) in enumerate(pairs):
        t1 = time.perf_counter()
        ctx = dict(sc.picture_contexts(c, opts.get("picture_break"), opts.get("picture_other")))
        al = {name: aligned(sc, tok, sc.build(q, ctx[name]), q, c, opts) for name in (clean_name, corrupt_name)}
        res = patch_pair(model, tok, al[clean_name], al[corrupt_name], head, opts)
        rows.append(
            {
                "convention": c,
                "question": q.id,
                "clean_context": clean_name,
                "corrupt_context": corrupt_name,
                **sc.picture_facts(c, opts.get("picture_break"), opts.get("picture_other")),
                "n_tokens_clean": len(al[clean_name].common),
                "n_tokens_corrupt": len(al[corrupt_name].common),
                "elapsed_s": round(time.perf_counter() - t1, 1),
                **res,
            }
        )
        den = res["modes"]["last"]["denoise"]
        print(
            f"  heads: {c} / {q.id} ({i + 1}/{len(pairs)}): clean - corrupt = {res['total']:+.3f} nats, "
            f"all heads at last {den['all_heads']:+.3f}, top head {den['top'][0]}, {rows[-1]['elapsed_s']}s",
            flush=True,
        )
    head.release()
    window = getattr(text_config(model), "sliding_window", None)
    longest = max([r["n_tokens_clean"] for r in rows] + [r["n_tokens_corrupt"] for r in rows], default=0)
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "metric": "log P(follows) - log P(violates), whole continuations, nats; m_clean / m_corrupt = [lp_follows, lp_violates, diff]",
            "effect": "denoise: metric(corrupt prompt, head <- clean) - metric(corrupt); noise: metric(clean) - metric(clean prompt, head <- corrupt); "
            "both are positive when the head carries the clean examples' convention",
            "positions": "last = the decision position only; tail = every position after the last token that differs between the two prompts",
            "over_window": bool(window) and longest > window,
        },
        "questions": questions_json(sc, list({q.id: q for _, q in pairs}.values())),
        "pairs": rows,
        "ranking": rank_heads(rows),
    }


def rank_heads(rows: list[dict], mode: str = "last", direction: str = "denoise", exclude: str | None = None) -> dict:
    """Per convention: heads ranked by their share of the effect, summed over the pairs (sum of the
    head's effects / sum of clean - corrupt; optionally leaving one question out), and how alike the
    pairs' effect maps are (mean pairwise correlation)."""
    out: dict = {}
    for c in sorted({r["convention"] for r in rows}):
        sel = [r for r in rows if r["convention"] == c and r["question"] != exclude]
        if not sel:
            continue
        eff = np.stack([np.array(r["modes"][mode][direction]["effect"]) for r in sel])  # [pairs, L, H]
        total = float(sum(r["total"] for r in sel))
        mean = eff.sum(0) / total if abs(total) > 1e-9 else eff.mean(0)
        H = mean.shape[1]
        order = np.argsort(-mean.ravel())
        flat = eff.reshape(len(sel), -1)
        ok = [i for i in range(len(sel)) if flat[i].std() > 0]
        cors = [np.corrcoef(flat[i], flat[j])[0, 1] for i in ok for j in ok if j < i]
        out[c] = {
            "n_pairs": len(sel),
            "questions": [r["question"] for r in sel],
            "sum_total": total,
            "top": [[*divmod(int(i), H), round(float(mean.ravel()[i]), 5)] for i in order[:32]],
            "pairwise_correlation": float(np.mean(cors)) if cors else None,
        }
    return out


# ---------------------------------------------------------------------------- the attention rows


@torch.no_grad()
def attention_rows(model, ids: list[int], pos: int) -> np.ndarray:
    """[L, H, len(ids)] attention of query position `pos` in every layer and head (eager, no cache:
    column j is absolute position j; a sliding layer puts zero beyond its window)."""
    dev = model_device(model)
    rows: list[torch.Tensor | None] = [None] * len(attn_modules(model))

    def hook(module, args, output):
        w = output[1]
        if w is None:
            raise RuntimeError("attention module returned no weights: eager attention not active")
        rows[module.layer_idx] = w[0, :, pos, :].float().cpu()

    handles = [m.register_forward_hook(hook) for m in attn_modules(model)]
    try:
        with attn_implementation(model, "eager"):
            base_model(model)(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(rows).numpy()  # type: ignore[arg-type]


def unit_mass(att: np.ndarray, labels: list[str], order: list[str]) -> np.ndarray:
    """[..., len(order)] attention mass per token label (a partition: the masses sum to the row sum)."""
    lab = np.array(labels[: att.shape[-1]])
    return np.stack([att[..., lab == u].sum(-1) for u in order], axis=-1)


def run_attention(sc: Demo, model, tok, opts: dict, out_dir: Path) -> dict:
    """Token-level attention at the decision token in the picture contexts: from the causally
    selected heads (read from heads.json in the same directory), and the plain averages over heads
    and layers as the contrast. Full [layer, head, token] rows go to attention.npz."""
    import json

    t0 = time.perf_counter()
    head = Head(model)
    k_top = int(opts.get("top_heads", 12))
    heads_path = Path(opts.get("heads_from") or Path(out_dir) / "heads.json")
    heads_rows = json.loads(heads_path.read_text())["pairs"] if heads_path.exists() else []
    cfg = text_config(model)
    types = list(getattr(cfg, "layer_types", None) or [])
    full_layers = [i for i, t in enumerate(types) if t == "full_attention"]
    specials = special_token_ids(tok)
    pairs = demo_pairs(sc, opts, opts.get("questions_per_convention", 2))
    cells, npz = [], {}
    for c, q in pairs:
        loo = len({r["question"] for r in heads_rows if r["convention"] == c}) > 1
        ranking = rank_heads(heads_rows, exclude=q.id if loo else None).get(c)
        selected = [(li, h) for li, h, _ in ranking["top"][:k_top]] if ranking else []
        weights = np.array([max(w, 0.0) for _, _, w in ranking["top"][:k_top]]) if ranking else np.zeros(0)
        cell = {
            "convention": c,
            "question": q.id,
            "selected_heads": [list(s) for s in selected],
            "selected_on": (ranking or {}).get("questions", []),
            "selection": (
                "leave this question out" if loo else ("same question" if ranking else "none: run the heads step first")
            ),
            "selection_pair": sorted(
                {f"{r['clean_context']} vs {r['corrupt_context']}" for r in heads_rows if r["convention"] == c}
            ),
            **sc.picture_facts(c, opts.get("picture_break"), opts.get("picture_other")),
            "contexts": {},
        }
        for name, spec in picture_list(sc, c, opts):
            built = sc.build(q, spec)
            al = aligned(sc, tok, built, q, c, opts)
            enc = al.enc["follows"]
            n = len(al.common)
            spans = unit_spans(enc, built)
            labels = token_labels(enc, spans, specials)[:n]
            att = attention_rows(model, al.common, al.decision_pos)  # [L, H, n]
            order = [u.id for u in built.units] + ["other", "special", "output"]
            mass = unit_mass(att, labels, order)  # [L, H, U]
            sel = np.stack([att[li, h] for li, h in selected]) if selected else np.zeros((0, n))
            maps = {
                "all_heads_all_layers": att.mean((0, 1)),
                "full_attention_layers": att[full_layers].mean((0, 1)) if full_layers else att.mean((0, 1)),
            }
            if len(sel):
                maps["selected_heads"] = sel.mean(0)
                maps["selected_heads_weighted"] = (
                    (sel * weights[:, None]).sum(0) / weights.sum() if weights.sum() > 0 else sel.mean(0)
                )
            with torch.no_grad():
                ids = torch.tensor([al.common], device=model_device(model))
                res = head.logprobs(base_model(model)(input_ids=ids, use_cache=False).last_hidden_state[0, -1:])
            first = {alt: float(res[0, al.cont[alt][0]]) for alt in ALTS}
            cell["contexts"][name] = {
                "spec": spec.to_json(),
                "n_tokens": n,
                "decision_pos": al.decision_pos,
                "tokens": enc.pieces(0, n),
                "labels": labels,
                "units": [{"id": u.id, "kind": u.kind, "group": u.group, "label": u.label} for u in built.units],
                "unit_order": order,
                "first_token_diff": first["follows"] - first["violates"],
                "token_maps": {k: np.round(v, 5).tolist() for k, v in maps.items()},
                "unit_mass": {k: np.round(unit_mass(v, labels, order), 5).tolist() for k, v in maps.items()},
                "unit_mass_per_selected_head": (
                    np.round(np.stack([mass[li, h] for li, h in selected]), 5).tolist() if selected else []
                ),
                "unit_mass_per_layer": np.round(mass.mean(1), 5).tolist(),
            }
            npz[f"{c}|{q.id}|{name}"] = att.astype(np.float16)
        cells.append(cell)
        print(f"  attention: {c} / {q.id}: {len(selected)} selected heads ({cell['selection']})", flush=True)
    head.release()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out_dir) / "attention.npz", **npz)
    return {
        "meta": {
            **base_meta(sc, model, opts),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "top_heads": k_top,
            "full_attention_layers": full_layers,
            "npz": "attention.npz: '<convention>|<question>|<context>' -> float16 [layer, head, token] attention of the decision position",
            "maps": "token_maps[name][i] = attention from the decision position to token i, averaged over the named set of heads; "
            "unit_mass sums it per unit (unit_order; other = prompt text outside units, special = template tokens, output = the SQL so far)",
        },
        "questions": questions_json(sc, list({q.id: q for _, q in pairs}.values())),
        "cells": cells,
    }
