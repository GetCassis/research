"""Smoke test: confirm the three model IDs respond and we can read token + thinking usage."""
import os, json
from anthropic import Anthropic

client = Anthropic()

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
}

PROMPT = "Reply with only the SQL to count rows in a table named ORDERS. No prose."

for label, model in MODELS.items():
    try:
        # First try WITHOUT thinking (fast sanity)
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": PROMPT}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        print(f"[{label}] {model} OK  in={msg.usage.input_tokens} out={msg.usage.output_tokens}  -> {text.strip()[:80]!r}")
    except Exception as e:
        print(f"[{label}] {model} ERROR: {type(e).__name__}: {e}")

# Now test extended thinking on the smallest model to confirm we can capture thinking tokens
print("--- thinking test (haiku) ---")
try:
    msg = client.messages.create(
        model=MODELS["haiku"],
        max_tokens=2000,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "What is 17*23? Think step by step, then answer."}],
    )
    thinking = "".join(getattr(b, "thinking", "") for b in msg.content if b.type == "thinking")
    text = "".join(b.text for b in msg.content if b.type == "text")
    print(f"thinking_chars={len(thinking)} out_tokens={msg.usage.output_tokens} answer={text.strip()[:60]!r}")
except Exception as e:
    print(f"thinking ERROR: {type(e).__name__}: {e}")
