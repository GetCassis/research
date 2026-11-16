"""Model loading and teacher-forced scoring.

The real model is loaded by `wb.hf.load_model` (Gemma 4: text decoder only, bf16, SDPA) and only
ever on a GPU. `tiny=True` builds a random-weight Gemma 4 decoder with the real tokenizer's
vocabulary: the plumbing fixture of the tests and of local dry runs, whose numbers mean nothing.

Read-out precision: the hidden states are the model's own (bf16 on the GPU), but the final
projection onto the vocabulary and the soft cap are computed in float32 here. In bf16 a logit near
20 is rounded to steps of 0.125, which is the size of the effects the head patching measures.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from wb.hf import base_model, model_device, text_config

DEFAULT_MODEL = "google/gemma-4-12B-it"

# The 12B's layer quirks at toy size: sliding-window layers with GQA, full-attention layers with
# bigger heads, one KV head and k = v, final logit soft-capping. The window is larger than any toy
# prompt, as on the real model (1024); tests that need eviction pass a small one.
TINY_TEXT = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    global_head_dim=32,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    sliding_window=4096,
    final_logit_softcapping=30.0,
    max_position_embeddings=8192,
)


def load_tokenizer(name: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tiny_model(tok, seed: int = 0, dtype: str = "float32", **overrides):
    """Random-weight Gemma 4 text decoder over `tok`'s vocabulary (CPU, eval, SDPA)."""
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(**{**TINY_TEXT, "vocab_size": len(tok), **overrides})
    cfg._attn_implementation = "sdpa"
    model = Gemma4UnifiedForCausalLM(cfg).to(getattr(torch, dtype))
    model.set_attn_implementation("sdpa")
    model.config._name_or_path = "tiny-random-gemma4"
    return model.eval()


def load(hf_model: str = DEFAULT_MODEL, *, tiny: bool = False, device: str | None = None, dtype: str = "bfloat16"):
    """(model, tokenizer). `tiny` never touches model weights."""
    tok = load_tokenizer(hf_model)
    if tiny:
        return tiny_model(tok), tok
    from wb.hf import load_model

    model = load_model(hf_model, device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_facts(model) -> dict:
    """What the JSON says about the model and its attention layout."""
    cfg = text_config(model)
    types = list(getattr(cfg, "layer_types", None) or [])
    return {
        "model": getattr(model.config, "_name_or_path", "") or type(model).__name__,
        "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "device": str(model_device(model)),
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "full_attention_layers": [i for i, t in enumerate(types) if t == "full_attention"],
        "sliding_window": getattr(cfg, "sliding_window", None),
        "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
    }


class Head:
    """The model's output head in float32: logits = softcap(h @ W^T), computed from the hidden states."""

    def __init__(self, model):
        self.model = model
        self.cap = getattr(text_config(model), "final_logit_softcapping", None)
        self._w32: torch.Tensor | None = None

    def w32(self) -> torch.Tensor:
        if self._w32 is None:
            self._w32 = self.model.lm_head.weight.detach().float()
        return self._w32

    def release(self) -> None:
        self._w32 = None

    def _softcap(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z / self.cap) * self.cap if self.cap else z

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._softcap(hidden.float() @ self.w32().T)

    def logprobs(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits(hidden), dim=-1)

    def rows(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Soft-capped logits of a few tokens only (no vocabulary-sized product)."""
        return self._softcap(hidden.float() @ self.model.lm_head.weight[token_ids].float().T)


def batches(lengths: list[int], max_batch_tokens: int) -> list[list[int]]:
    """Indices grouped longest first so that batch x longest length stays under the cap."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, i = [], 0
    while i < len(order):
        size = max(1, min(len(order) - i, max_batch_tokens // max(lengths[order[i]], 1)))
        out.append(order[i : i + size])
        i += size
    return out


@torch.no_grad()
def score_labels(
    model,
    items: list[tuple[list[int], list[int]]],
    *,
    head: Head | None = None,
    max_batch_tokens: int = 16384,
    topk: int = 0,
    progress: str | None = None,
) -> list[dict]:
    """Teacher-forced log-probabilities, without sampling.

    `items`: (full token ids, label positions). For each label position p the result has the
    natural-log probability of token `full[p]` given `full[:p]`. With `topk`, also the k most likely
    tokens at the FIRST label position (the decision token) as (id, log-prob) pairs.
    Sequences are right-padded in a batch: attention is causal, so padding is never read. An input
    that is a prefix of another one is not run: its rows are read from the longer sequence (the two
    alternatives of a decision share everything up to the decision token, and the reference SQL
    contains every `follows` continuation)."""
    head = head or Head(model)
    dev = model_device(model)
    base = base_model(model)
    inputs = [tuple(full[: max(pos)]) for full, pos in items]
    uniq = sorted(set(inputs))
    runs: list[tuple[int, ...]] = []
    served: dict[tuple[int, ...], int] = {}  # input -> index in `runs` of a sequence it is a prefix of
    for k in range(len(uniq) - 1, -1, -1):  # the lexicographic successor extends a prefix, if anything does
        nxt = uniq[k + 1] if k + 1 < len(uniq) else None
        if nxt is not None and nxt[: len(uniq[k])] == uniq[k]:
            served[uniq[k]] = served[nxt]
        else:
            served[uniq[k]] = len(runs)
            runs.append(uniq[k])
    users: dict[int, list[int]] = {}
    for j, x in enumerate(inputs):
        users.setdefault(served[x], []).append(j)
    out: list[dict | None] = [None] * len(items)
    groups = batches([len(x) for x in runs], max_batch_tokens)
    for gi, idx in enumerate(groups):
        S = len(runs[idx[0]])
        x = np.zeros((len(idx), S), dtype=np.int64)
        for r, k in enumerate(idx):
            x[r, : len(runs[k])] = runs[k]
        hidden = base(input_ids=torch.tensor(x, device=dev), use_cache=False).last_hidden_state
        for r, k in enumerate(idx):
            for j in users[k]:
                full, pos = items[j]
                rows = torch.tensor([p - 1 for p in pos], device=dev)
                lp = head.logprobs(hidden[r, rows])
                labels = torch.tensor([full[p] for p in pos], device=dev)
                res = {"logp": lp.gather(-1, labels[:, None])[:, 0].double().cpu().numpy()}
                if topk:
                    top = lp[0].topk(topk)
                    res["topk"] = [(int(i), float(v)) for i, v in zip(top.indices.tolist(), top.values.tolist())]
                out[j] = res
        del hidden
        if progress and (gi % 10 == 0 or gi == len(groups) - 1):
            print(
                f"  {progress}: batch {gi + 1}/{len(groups)} (B={len(idx)}, S={S}, {len(runs)} runs for {len(items)} items)",
                flush=True,
            )
    return out  # type: ignore[return-value]


def two_way(lp_follows: float, lp_violates: float) -> float:
    """P(follows) renormalised over the two alternatives."""
    return 1.0 / (1.0 + math.exp(min(700.0, lp_violates - lp_follows)))
