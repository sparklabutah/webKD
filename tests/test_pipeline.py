"""Synthetic data-to-training regression tests, without models or network calls."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from webkd import weights
from webkd.build_anchor import EPS_UNANCHORED
from webkd.training import token_weights

TARGET = (
    "<think>\nThe search box is visible. Searching will reveal the requested page.\n</think>\n\n"
    "<plan>\n1. Search for the page.\n2. Read the answer.\n</plan>\n\n"
    "<step>1</step>\n\n<memory>\nThe search box is empty.\n</memory>\n\n"
    "<action>\nfill('14', 'page')\n</action>"
)


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def run(*args, success=True):
    result = subprocess.run(
        [sys.executable, "-m", "webkd", *map(str, args)], capture_output=True, text=True
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "webkd.json"
    path.write_text(
        json.dumps(
            [
                {
                    "system": "Act on the visible page.",
                    "conversations": [
                        {"from": "human", "value": "The page has a search box."},
                        {"from": "gpt", "value": TARGET},
                    ],
                }
            ]
        )
    )
    return path


def test_scoring_weights_anchor_and_training_dry_run(corpus, tmp_path):
    scoring = tmp_path / "scoring.json"
    run("prepare-scoring", "--dataset", corpus, "--out", scoring)
    records = json.loads(scoring.read_text())
    assert len(records) == 1
    row = records[0]
    for i, segment in enumerate(row["segments"]):
        segment["contribution"] = float(i)
    row.update(base_nll=2.0, action_tokens=5)
    scores = tmp_path / "scores.jsonl"
    scores.write_text(json.dumps(row) + "\n")
    sidecar = tmp_path / "weights.json"
    run("build-weights", "--dataset", corpus, "--scores", scores, "--out", sidecar)
    run("build-anchor", "--corpus", corpus, "--out-dir", tmp_path)
    anchor = tmp_path / "webkd.klgamma_rationale.json"
    entries = json.loads(sidecar.read_text())["entries"]
    assert len(entries) == 1
    spans = entries[weights.content_key(TARGET)]["weight_spans"]
    assert spans and min(s[2] for s in spans) < max(s[2] for s in spans)

    output = tmp_path / "training"
    config = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    run(
        "train",
        "--model",
        "synthetic-student",
        "--dataset",
        corpus,
        "--output",
        output,
        "--template",
        "qwen3_5",
        "--config",
        config,
        "--weights",
        sidecar,
        "--anchor",
        anchor,
        "--dry-run",
    )
    options = json.loads((output / "method.json").read_text())
    assert options["WEBKD_KL_REF"] == "synthetic-student"
    assert options["WEBKD_TOKEN_WEIGHTS"] == str(sidecar)
    assert options["WEBKD_KL_GAMMA_WEIGHTS"] == str(anchor)
    assert not any("API_KEY" in key for key in options)


def test_incomplete_scoring_is_rejected(corpus, tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text("")
    output = tmp_path / "weights.json"
    result = run(
        "build-weights", "--dataset", corpus, "--scores", scores, "--out", output, success=False
    )
    assert "incomplete" in result.stderr.lower()
    assert not output.exists()


def test_anchor_preserves_action_supervision(corpus, tmp_path, monkeypatch):
    run("build-anchor", "--corpus", corpus, "--out-dir", tmp_path)
    anchor = tmp_path / "webkd.klgamma_rationale.json"
    monkeypatch.setenv("WEBKD_KL_GAMMA_WEIGHTS", str(anchor))
    monkeypatch.delenv("WEBKD_TOKEN_WEIGHT_FLOOR", raising=False)
    ids = [ord(c) for c in TARGET]
    labels = ids.copy()
    gamma = token_weights.target_kl_gamma(CharacterTokenizer(), TARGET, ids, labels)
    action_start, action_end = weights.tag_spans(TARGET)["action"]["inner"]
    think_start, think_end = weights.tag_spans(TARGET)["think"]["inner"]
    assert all(g == EPS_UNANCHORED for g in gamma[action_start:action_end])
    assert all(g == 1 for g in gamma[think_start:think_end])
    assert labels == ids


def test_stale_sidecar_is_rejected():
    from webkd.train import validate_sidecar

    # The token reader separately guards stale content lengths.
    entry = {"len": len(TARGET) + 1}
    with pytest.raises(ValueError, match="stale"):
        token_weights.target_token_weights(
            CharacterTokenizer(),
            TARGET,
            [ord(c) for c in TARGET],
            [ord(c) for c in TARGET],
            tables=({weights.content_key(TARGET): entry}, {}),
        )
    assert callable(validate_sidecar)


def test_span_boundaries_and_hard_mask():
    result = token_weights.token_weight_vector(
        [(0, 2), (2, 4), (4, 6)], [[0, 3, 0.2]], 1.0, 0.1, [[4, 6]]
    )
    assert result == pytest.approx([0.2, 0.6, 0.0])
