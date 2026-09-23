import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fw_jev.config import load, training_batches
from fw_jev.runner import fingerprint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/raw-base-v1/config.json"


def test_dataset_is_frozen_and_bounded():
    subprocess.run([sys.executable, "experiments/scale-v1/build_dataset.py", "--check"], cwd=ROOT, check=True)
    c, train, validation, root = load(CONFIG)
    assert root == ROOT and (len(train), len(validation)) == (96, 24)
    assert len(training_batches(c, train)) == 24
    assert len({r["family"] for r in train}) == 7
    assert len({r["family"] for r in validation}) == 7
    manifest = json.loads((ROOT / "experiments/scale-v1/manifest.json").read_text())
    assert manifest["smoke_ids"] == [r["id"] for r in train[:6]]
    assert set(manifest) == {
        "version", "train_count", "validation_count", "smoke_ids", "files",
        "builder_sha256", "authored_sha256", "source_cache_sha256", "families", "rules",
    }


def test_dataset_check_rejects_an_edited_committed_file(tmp_path):
    folder = tmp_path / "scale-v1"
    folder.mkdir()
    for name in ("build_dataset.py", "authored.py", "sources.json", "manifest.json", "train.jsonl", "validation.jsonl"):
        (folder / name).write_bytes((ROOT / "experiments/scale-v1" / name).read_bytes())
    command = [sys.executable, str(folder / "build_dataset.py"), "--check"]
    subprocess.run(command, check=True, capture_output=True)
    rows = (folder / "train.jsonl").read_text().splitlines()
    (folder / "train.jsonl").write_text("\n".join([rows[1], rows[0], *rows[2:]]) + "\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "manifest hash" in result.stderr


def test_dataset_contract_and_code_are_snapshotted():
    hashes = fingerprint(ROOT)
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "manifest.json",
        "sources.json",
        "authored.py",
        "build_dataset.py",
        "evaluate.py",
    ):
        assert "experiments/scale-v1/" + name in hashes
    for name in ("config.json", "smoke.json", "admission-v2.json", "contract.json"):
        assert "experiments/raw-base-v1/" + name in hashes


def evaluation_module():
    spec = importlib.util.spec_from_file_location(
        "scale_evaluation", ROOT / "experiments/scale-v1/evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_is_deterministic_and_clusters_prompt_changes():
    module = evaluation_module()
    assert module.bootstrap([0.1] * 24, 1000, 1) == pytest.approx([0.1, 0.1])
    assert module.bootstrap([-0.1] * 24, 1000, 1) == pytest.approx([-0.1, -0.1])
    assert module.bootstrap([0.1, -0.1] * 12, 1000, 1) == module.bootstrap([0.1, -0.1] * 12, 1000, 1)


def test_incomplete_or_unblinded_review_cannot_pass(tmp_path):
    path = tmp_path / "blind-ratings.json"
    path.write_text(json.dumps({"reviewer": "", "blinded_to_key_and_rewards": False}))
    assert evaluation_module().review_result(tmp_path, {}) == {"complete": False, "passed": False}


def test_mock_cannot_claim_meaningful_gain(tmp_path):
    (tmp_path / "source/experiments/raw-base-v1").mkdir(parents=True)
    (tmp_path / "source/experiments/raw-base-v1/contract.json").write_text(
        (ROOT / "experiments/raw-base-v1/contract.json").read_text()
    )
    for name, value in (
        ("status.json", {"state": "complete", "mock": True}),
        ("records.json", []),
        ("summary.json", {}),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed live run"):
        evaluation_module().evaluate(tmp_path)
