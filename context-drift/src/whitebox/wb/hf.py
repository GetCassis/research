"""Hugging Face plumbing shared by the white-box steps: load the model, reach its decoder stack,
switch the attention kernel, prefill a KV cache and reuse one prefix across a batch.

Verified against transformers 5.22 on Gemma 4 (`google/gemma-4-12B-it`, `gemma4_unified`):

- The checkpoint is the multimodal `Gemma4UnifiedForConditionalGeneration` (keys
  `model.language_model.*` plus a small encoder-free vision / audio embedder). `load_model` loads
  only the text decoder as `Gemma4UnifiedForCausalLM` (key_mapping `model.language_model.` ->
  `model.`, lm_head tied to the embeddings), so every module here sees `model.model.layers`,
  `model.lm_head` and `model.config.num_hidden_layers`.
- 48 layers: 40 `sliding_attention` (window 1024; head_dim 256, 8 KV heads) and 8
  `full_attention` (5, 11, ..., 47; head_dim 512, ONE KV head, `attention_k_eq_v`: no v_proj, the
  values are `v_norm(k_proj(x))`). Attention scaling is 1.0 (q_norm / k_norm instead), no attention
  logit soft-capping in the attention call; the final logits ARE soft-capped
  (`final_logit_softcapping` 30: `tanh(z / 30) * 30`).
- The attention kernel is picked per call from `config._attn_implementation`, and
  `model.set_attn_implementation("eager" | "sdpa")` flips it at runtime, so a prefill can run with
  SDPA and a later chunk, reusing the same KV cache, with eager.
- The decoder derives positions from `past_key_values.get_seq_length()`, so a cache holding a
  prefix of length P makes the next chunk start at position P.
- `DynamicCache(config=...)` gives sliding layers a window cache that keeps only the last 1023
  positions; a cache built without a config (`DynamicCache()`) keeps every position for every
  layer, and the sliding mask then comes from `config.sliding_window`.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


# multimodal checkpoints whose text decoder loads alone: model_type -> (text CausalLM class, key_mapping)
_TEXT_ONLY = {
    "gemma4_unified": ("Gemma4UnifiedForCausalLM", {r"^model\.language_model\.": "model."}),
}


def load_model(name: str, device: str | None = None, dtype: str | torch.dtype = "bfloat16"):
    """The causal LM on `device` (cuda > mps > cpu), SDPA attention, eval mode.

    Loaded straight onto the device (no CPU copy). A multimodal checkpoint in `_TEXT_ONLY`
    (Gemma 4) loads its text decoder only (the vision / audio embedder weights are skipped as
    unexpected keys)."""
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    device = pick_device(device)
    torch_dtype = _DTYPES[dtype] if isinstance(dtype, str) else dtype
    kw = dict(dtype=torch_dtype, attn_implementation="sdpa", device_map=device)
    cfg = AutoConfig.from_pretrained(name)
    if cfg.model_type in _TEXT_ONLY:
        cls_name, key_mapping = _TEXT_ONLY[cfg.model_type]
        cls = getattr(transformers, cls_name)
        model = cls.from_pretrained(name, config=cfg.get_text_config(), key_mapping=key_mapping, **kw)
        model.config._name_or_path = name
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    return model.eval()


def text_config(model):
    """The decoder's config (the model's own for a CausalLM, `text_config` for a multimodal wrapper)."""
    return model.config.get_text_config()


def base_model(model):
    """The decoder stack (`model.model`, or its `language_model` for a multimodal wrapper)."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


@contextlib.contextmanager
def attn_implementation(model, impl: str) -> Iterator[None]:
    """Temporarily switch every attention module of `model` to `impl` ("eager", "sdpa", ...)."""
    prev = model.config._attn_implementation
    if prev == impl:
        yield
        return
    model.set_attn_implementation(impl)
    try:
        yield
    finally:
        model.set_attn_implementation(prev)


@torch.no_grad()
def prefill(model, ids: torch.Tensor, cache: Cache | None = None, chunk: int = 2048) -> Cache:
    """Run `ids` ([1, n]) through the base model in chunks, filling and returning a KV cache.

    Chunking bounds the attention working set to chunk x n. The default cache follows the config
    (sliding-window layers keep only their window); pass `DynamicCache()` to keep every position of
    every layer."""
    cache = cache if cache is not None else DynamicCache(config=text_config(model))
    base = base_model(model)
    for s in range(0, ids.shape[1], chunk):
        base(input_ids=ids[:, s : s + chunk], past_key_values=cache, use_cache=True)
    return cache


class _SharedPrefixLayer(DynamicLayer):
    """A cache layer over a shared batch-1 prefix, for B sequences that differ only after it.

    `update` returns prefix (broadcast to B) ++ suffix-so-far ++ new keys. With `grow=False` it
    stores nothing, so a one-shot forward keeps only the current layer's B x (P+S) concatenation
    alive instead of all layers'. With `grow=True` it appends the new keys to a per-batch suffix,
    for a suffix run in several chunks."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, grow: bool):
        super().__init__()
        self.prefix_k, self.prefix_v = keys, values
        self.keys = self.values = None
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True
        self.grow = grow

    def update(self, key_states, value_states, *args, **kwargs):
        b = key_states.shape[0]
        parts_k = [self.prefix_k.expand(b, -1, -1, -1)]
        parts_v = [self.prefix_v.expand(b, -1, -1, -1)]
        if self.keys is not None:
            parts_k.append(self.keys)
            parts_v.append(self.values)
        if self.grow:
            self.keys = key_states if self.keys is None else torch.cat([self.keys, key_states], dim=-2)
            self.values = value_states if self.values is None else torch.cat([self.values, value_states], dim=-2)
        return torch.cat([*parts_k, key_states], dim=-2), torch.cat([*parts_v, value_states], dim=-2)

    def get_seq_length(self) -> int:
        return self.prefix_k.shape[-2] + (0 if self.keys is None else self.keys.shape[-2])


def shared_prefix_cache(prefix: Cache, grow: bool = False) -> Cache:
    """A cache whose every layer starts from `prefix` (batch 1) and serves any batch size."""
    layers = [_SharedPrefixLayer(layer.keys, layer.values, grow) for layer in prefix.layers]
    return Cache(layers=layers)
