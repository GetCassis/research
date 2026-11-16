"""Runners: a runner turns one built prompt into N completions.

The interface is model-agnostic on purpose: ``complete(prompt, n) -> list[Completion]``. The
experiments only use that method, so another model plugs in with ``FunctionRunner`` (wrap any
``fn(prompt, n) -> list[str]``) or its own class. Prompts come from ``talkdemo.prompt`` only.

``ClaudeRunner`` calls the Messages API through the official SDK, one request per completion
(the current models reject non-default sampling parameters, so every sample is drawn at the
API default), with a disk cache and a spend ledger that enforces a hard budget.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .prompt import Prompt

ROOT = Path(__file__).resolve().parents[2]  # the experiment directory
CACHE_DIR = ROOT / "cache"  # git-ignored

# USD per million tokens (input, output), Anthropic first-party API, prompts under 100K tokens
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


@dataclass(frozen=True)
class Completion:
    text: str
    meta: dict = field(default_factory=dict)


class Runner(Protocol):
    name: str

    def complete(self, prompt: Prompt, n: int) -> list[Completion]:
        """N independent completions of the prompt."""
        ...


class FunctionRunner:
    """Adapter for any model: ``fn(prompt, n)`` returns n reply strings."""

    def __init__(self, name: str, fn: Callable[[Prompt, int], list[str]]):
        self.name = name
        self._fn = fn

    def complete(self, prompt: Prompt, n: int) -> list[Completion]:
        texts = self._fn(prompt, n)
        if len(texts) != n:
            raise ValueError(f"{self.name}: asked for {n} completions, got {len(texts)}")
        return [Completion(t) for t in texts]


class BudgetExceeded(RuntimeError):
    pass


class TransientError(RuntimeError):
    """The call failed for a reason a later run can recover from (nothing was cached)."""


class DiskCache:
    """One JSON file per key under ``root`` (two-level fan-out)."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, key: str, value: dict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Ledger:
    """Append-only record of what every uncached API call cost; shared by all runners."""

    def __init__(self, path: Path, budget_usd: float):
        self.path = path
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self._spent = sum(row["usd"] for row in self.rows())

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    @property
    def spent(self) -> float:
        return self._spent

    def check(self) -> None:
        if self._spent >= self.budget_usd:
            raise BudgetExceeded(f"spent ${self._spent:.2f}, budget ${self.budget_usd:.2f}")

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = PRICES[model]
        usd = (input_tokens * price_in + output_tokens * price_out) / 1e6
        row = {"ts": round(time.time(), 1), "model": model, "in": input_tokens, "out": output_tokens, "usd": usd}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self._spent += usd
        return usd


def thinking_param(model: str, thinking: str) -> dict | None:
    """``off``: the model does not think before answering (the usual setting of a one-shot SQL
    writer). Claude Sonnet 5.5 rejects ``disabled``; its off switch is
    ``between_tools``. ``default``: send nothing (adaptive thinking)."""
    if thinking == "default":
        return None
    if thinking == "off":
        return {"type": "between_tools"} if model == "claude-sonnet-5-5" else {"type": "disabled"}
    raise ValueError(f"thinking must be 'off' or 'default', got {thinking!r}")


class ClaudeRunner:
    def __init__(
        self,
        model: str,
        *,
        thinking: str = "off",
        max_tokens: int = 1024,
        cache_dir: Path = CACHE_DIR,
        budget_usd: float = 14.0,
        max_retries: int = 6,
    ):
        if model not in PRICES:
            raise ValueError(f"no price for {model}; add it to PRICES")
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.name = model if thinking == "off" else f"{model}+thinking"
        self.cache = DiskCache(cache_dir / "completions")
        self.ledger = Ledger(cache_dir / "spend.jsonl", budget_usd)
        self._max_retries = max_retries
        self._client = None
        self._client_lock = threading.Lock()

    def _api(self):
        with self._client_lock:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic(max_retries=self._max_retries)
            return self._client

    def _request(self, prompt: Prompt) -> dict:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": prompt.system,
            "messages": prompt.messages(),
        }
        thinking = thinking_param(self.model, self.thinking)
        if thinking is not None:
            request["thinking"] = thinking
        return request

    def complete(self, prompt: Prompt, n: int) -> list[Completion]:
        return [self._one(prompt, i) for i in range(n)]

    def _one(self, prompt: Prompt, index: int) -> Completion:
        request = self._request(prompt)
        key = _key("completion", request, index)
        hit = self.cache.get(key)
        if hit is not None:
            return Completion(hit["text"], hit["meta"] | {"cached": True})
        self.ledger.check()
        import anthropic

        try:
            response = self._api().messages.create(**request)
        except anthropic.RateLimitError as e:  # still limited after the SDK's own retries
            raise TransientError(f"rate limited: {e.message}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientError(f"server error {e.status_code}") from e
            raise  # 4xx: a bug in the request, do not hide it
        except anthropic.APIConnectionError as e:
            raise TransientError("connection error") from e
        # A refusal or a truncated reply is kept as it came: the checkers then report it unparsed.
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        usd = self.ledger.add(self.model, usage.input_tokens, usage.output_tokens)
        meta = {
            "model": response.model,
            "stop_reason": response.stop_reason,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "usd": usd,
            "request_id": response._request_id,
        }
        self.cache.put(key, {"text": text, "meta": meta})
        return Completion(text, meta | {"cached": False})

    def count_tokens(self, prompt: Prompt) -> int:
        """Input tokens of a prompt for this model (free endpoint, cached)."""
        request = {"model": self.model, "system": prompt.system, "messages": prompt.messages()}
        key = _key("count_tokens", request)
        hit = self.cache.get(key)
        if hit is None:
            hit = {"input_tokens": self._api().messages.count_tokens(**request).input_tokens}
            self.cache.put(key, hit)
        return hit["input_tokens"]
