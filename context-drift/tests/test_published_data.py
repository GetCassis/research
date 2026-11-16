"""The published data matches the shipped code and the numbers shown in the talk (offline)."""

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "whitebox"))

import talk_numbers  # noqa: E402
from wb.scenario import load_scenario  # noqa: E402

WHITEBOX = sorted((ROOT / "results" / "whitebox").glob("*/*.json.gz"))


@pytest.mark.parametrize("path", WHITEBOX, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_whitebox_outputs_were_made_with_the_shipped_scenario_and_builder(path):
    meta = json.loads(gzip.decompress(path.read_bytes()))["meta"]
    assert meta["scenario"] == "shop" and not meta["dry_run_random_weights"]
    assert meta["scenario_version"] == load_scenario("shop").version


def test_every_slide_number_is_recomputed_from_the_published_data(capsys):
    assert talk_numbers.main([]) == 0
    assert " 0 differ" in capsys.readouterr().out
