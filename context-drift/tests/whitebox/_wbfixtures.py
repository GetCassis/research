"""Fixtures of the white-box plumbing tests: a random-weight Gemma 4 decoder over the real
tokenizer's vocabulary (no model weights are downloaded or loaded) and the small fixture scenario.
They check the machinery, not the model: every number they produce is meaningless.

They need torch and transformers (requirements-gpu.txt; no GPU needed) and the Gemma 4 tokenizer
files in the Hugging Face cache (the model is gated: accept its licence on huggingface.co, then
`huggingface-cli download google/gemma-4-12B-it tokenizer.json tokenizer_config.json config.json
chat_template.jinja`). Without them every test here is skipped. The test modules import the
fixtures they use from here by name.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

WB = Path(__file__).resolve().parents[2] / "src" / "whitebox"
if str(WB) not in sys.path:
    sys.path.insert(0, str(WB))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import wb  # noqa: E402,F401  (puts src/ on sys.path for talkdemo and the scenarios)


def load_script(name: str):
    """`src/whitebox/<name>.py` as module `wb_<name>`: `run` and `figures` are too common as module
    names to import bare in a test session that also collects the behavioural tests."""
    key = f"wb_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, WB / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


@pytest.fixture(scope="session")
def tok():
    try:
        from wb.model import load_tokenizer

        return load_tokenizer()
    except Exception as e:  # pragma: no cover - tokenizer files not in the HF cache
        pytest.skip(f"Gemma 4 tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def model(tok):
    import torch
    from wb.model import tiny_model

    torch.set_num_threads(int(os.environ.get("WB_TEST_THREADS", "4")))
    return tiny_model(tok)


@pytest.fixture(scope="session")
def sc():
    from wb.scenario import load_scenario

    return load_scenario("fixture")
