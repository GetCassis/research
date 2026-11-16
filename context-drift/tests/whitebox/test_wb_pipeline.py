"""The whole command line on the random-weight model: every step writes its JSON, and the figures
build from them. A dry run of a GPU session, nothing more."""

import json

import pytest
from _wbfixtures import load_script, tok  # noqa: F401  (tok: fixture)

run = load_script("run")


def test_info_runs_without_a_model(tok, tmp_path):  # noqa: F811
    assert run.main(["info", "--scenario", "fixture", "--out", str(tmp_path)]) == 0
    info = json.loads((tmp_path / "info.json").read_text())
    assert info["problems"] == [] and info["n_decisions"] == 10
    assert max(c["with_reference_sql_max"] for c in info["contexts"]) < 1024


@pytest.fixture(scope="module")
def dry_run(tok, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("dry")
    argv = ["all", "--scenario", "fixture", "--tiny", "--out", str(out)]
    argv += ["--opt", "n=2", "--opt", "max_new_tokens=5", "--opt", "ig_steps=2", "--opt", "questions_per_convention=2"]
    assert run.main(argv) == 0
    return out


def test_all_steps_dry_run(dry_run):
    for step in run.MODEL_STEPS:
        meta = json.loads((dry_run / f"{step}.json").read_text())["meta"]
        assert meta["step"] == step and meta["dry_run_random_weights"] is True
        assert meta["scenario"] == "fixture" and len(meta["scenario_version"]["content"]) == 16
        assert meta["model"] == "tiny-random-gemma4" and "enable_thinking=False" in meta["thinking"]
    assert (dry_run / "attention.npz").exists()


def test_figures_and_summary_build_from_the_json(dry_run, capsys):
    made = load_script("figures").build(dry_run)
    assert {p.suffix for p in made} == {".svg", ".html"} and len(made) >= 8
    for p in made:
        assert p.stat().st_size > 500
    assert load_script("summary").main(["--out", str(dry_run)]) == 0
    assert "== grid" in capsys.readouterr().out


def test_claims_read_the_three_directories(dry_run, tmp_path, capsys):
    (tmp_path / "shop").symlink_to(dry_run, target_is_directory=True)
    assert load_script("claims").main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "== 2. Redundancy" in out and "== 4. What one example carries" in out
