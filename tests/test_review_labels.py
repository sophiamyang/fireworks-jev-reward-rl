import json

import pytest
from test_raw_base import CONFIG, SMOKE, review_fixture

from fw_jev.config import load
from fw_jev.raw_base import check_mixed_review
from fw_jev.runner import run
from fw_jev.storage import save


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    folder = tmp_path_factory.mktemp("labels") / "smoke"
    run(SMOKE, folder, mock=True)
    _, train, _, _ = load(CONFIG)
    return folder, train


def check(folder, train, better, worse, axes):
    review = review_fixture(folder, train)
    for item in review["samples"]:
        item.update(source_faithful=True, key_content_present=True)
        if item["sample"] == 3:
            item.update(better)
        elif item["sample"] == 0:
            item.update(worse)
    for pair in review["accepted_rankings"]:
        pair["axes"] = axes
    save(folder / "smoke-review.json", review)
    rows = json.loads((folder / "records.json").read_text())
    policy = json.loads((CONFIG.parent / "admission-v2.json").read_text())
    return check_mixed_review(folder, rows, policy, {c["id"] for c in train[:6]})


def test_audit_repro_unfaithful_incomplete_better_is_rejected(smoke):
    bad = {"source_faithful": False, "key_content_present": False}
    with pytest.raises(ValueError, match="defect"):
        check(*smoke, bad, {"source_faithful": True}, ["source_support"])


def test_legitimate_source_support_pair_passes(smoke):
    assert check(*smoke, {}, {"source_faithful": False}, ["source_support"]) == 3


def test_source_support_needs_an_actual_source_difference(smoke):
    with pytest.raises(ValueError, match="Source-support"):
        check(*smoke, {}, {}, ["source_support"])


def test_style_pair_cannot_prefer_missing_key_content(smoke):
    with pytest.raises(ValueError, match="defect"):
        check(*smoke, {"key_content_present": False}, {}, ["style"])


def test_style_pair_with_equal_labels_passes(smoke):
    assert check(*smoke, {}, {}, ["style"]) == 3
    both = {"key_content_present": False}
    assert check(*smoke, both, both, ["style", "quality"]) == 3


def test_smoke_check_failures_say_what_to_do():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/review_raw_smoke.py"
    spec = importlib.util.spec_from_file_location("review_raw_smoke", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.explain("Record observed scorer limitations").startswith("Review form")
    assert helper.explain("Reviewed preference lacks the frozen reward margin").startswith("Jev disagrees")
    assert helper.explain("Insufficient mixed-objective numeric signal").startswith("Signal")
    assert helper.explain("Smoke source is stale").startswith("Setup")
