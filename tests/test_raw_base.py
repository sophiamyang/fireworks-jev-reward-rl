import copy
import json
from pathlib import Path

import nbformat
import pytest

from fw_jev.config import load, plan
from fw_jev.raw_base import check_admission, check_mixed_review, verify_model
from fw_jev.runner import fingerprint, run
from fw_jev.storage import digest, save

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/raw-base-v1/config.json"
SMOKE = ROOT / "experiments/raw-base-v1/smoke.json"


def test_training_and_smoke_recipes_share_every_setting_but_the_schedule():
    current, train, val, root = load(CONFIG)
    assert root == ROOT and (len(train), len(val)) == (96, 24)
    assert plan(current, train, val)["limits"] == {
        "generations": 864,
        "jev": 896,
        "references": 768,
        "optimizer": 24,
    }
    assert plan(current, train, val)["starting_policy"].startswith("fresh raw base")
    smoke, tasks, _, _ = load(SMOKE)
    assert smoke == dict(current, steps=0, train_limit=6, group_size=4)
    assert tasks == train[:6]


@pytest.mark.parametrize(
    "change",
    [
        {"restore_state": "accounts/example/state-6"},
        {"experiment": "other-v1"},
        {"steps": 25},
        {"steps": 12, "train_limit": 48},
        {"train_limit": 100},
        {"epochs": 2},
        {"weights": {"style": 0.8, "quality": 0.2}},
        {"reward_variant": "other-v1"},
        {"group_size": 4},
        {"validation_file": "experiments/scale-v1/train.jsonl"},
    ],
)
def test_raw_profile_cannot_silently_restore_or_expand(tmp_path, change):
    c = json.loads(CONFIG.read_text())
    c.update(change)
    save(tmp_path / "bad.json", c)
    with pytest.raises(ValueError):
        load(tmp_path / "bad.json")


def test_model_identity_is_exact():
    model = "accounts/fireworks/models/qwen3p8-27b"
    verify_model(model, model)
    verify_model(model, "/llm-downloader-destination/base/fireworks/qwen3p8-27b/hf")
    for wrong in ("qwen3p8-27b", model + "-other", "accounts/fireworks/models/other"):
        with pytest.raises(ValueError):
            verify_model(model, wrong)


def test_raw_mock_is_fresh_and_not_a_live_admission(raw_mock_run, tmp_path):
    out, result = raw_mock_run
    assert result["status"]["optimizer_updates"] == 24
    assert result["status"]["initial_sampler"] == result["status"]["reference_sampler"] == "mock/base"
    assert result["before"]["reward_mean"] == result["after"]["reward_mean"]
    run(SMOKE, tmp_path / "smoke", mock=True)
    c, train, _, _ = load(CONFIG)
    with pytest.raises(ValueError, match="LIVE"):
        check_admission(tmp_path / "smoke", ROOT, c, train)


def strip_mock_markers(folder):
    # Deliberately relabel synthetic artifacts only inside an isolated unit fixture.
    status = json.loads((folder / "status.json").read_text())
    save(folder / "status.json", dict(status, mock=False))
    for path in (folder / "records.json", folder / "scorer-check/records.json"):
        rows = json.loads(path.read_text())
        for row in rows:
            row["judge"]["telemetry"].pop("mock")
            if "case" in row:
                row["judge"]["grounding_applicable"] = row["case"]["grounding_applicable"]
        save(path, rows)


def review_fixture(folder, train):
    rows = json.loads((folder / "records.json").read_text())
    return {
        "policy": "raw-base-mixed-objective-smoke-v2",
        "records_sha256": digest(folder / "records.json"),
        "approved": True,
        "reviewer": "synthetic unit test only, not live approval",
        "samples": [
            dict(
                case_id=r["case"]["id"],
                sample=r["sample"],
                answers_task=True,
                source_faithful=r["sample"] == 3,
                key_content_present=True,
                ambiguous_request=False,
                note="Synthetic label",
            )
            for r in rows
        ],
        "accepted_rankings": [
            dict(
                case_id=c["id"],
                better=3,
                worse=0,
                axes=["source_support"],
                reason="Synthetic source difference; not a style-only comparison",
            )
            for c in train[:3]
        ],
        "scorer_limitations": ["Synthetic fixture, not a real quality test"],
    }


def test_admission_replays_exact_rubric_inputs_and_identity(tmp_path):
    folder = tmp_path / "smoke"
    run(SMOKE, folder, mock=True)
    c, train, _, _ = load(CONFIG)
    fixed = json.loads((folder / "scorer-check/summary.json").read_text())
    assert fixed["protocol_sha256"] == {
        f"src/fw_jev/{name}": digest(ROOT / "src/fw_jev" / name) for name in ("calibration.py", "reward.py")
    }
    assert len(fixed["checks"]) == 21 and fixed["passed"]
    strip_mock_markers(folder)
    save(folder / "smoke-review.json", review_fixture(folder, train))
    save(folder / "verify-base-model-result.json", {"model_data": {"model_name": c["model"]}})
    admission = check_admission(folder, ROOT, c, train)
    assert admission["reviewed_groups"] == 3 and admission["reward_version"] == "jev-style-quality-support-v1"
    config = json.loads((folder / "config.json").read_text())
    save(folder / "config.json", dict(config, temperature=0.9))
    with pytest.raises(ValueError, match="settings differ"):
        check_admission(folder, ROOT, c, train)
    save(folder / "config.json", config)
    rows = json.loads((folder / "records.json").read_text())
    changed = copy.deepcopy(rows)
    changed[0]["score"]["reward"] += 0.01
    save(folder / "records.json", changed)
    with pytest.raises(ValueError, match="reward replay"):
        check_admission(folder, ROOT, c, train)
    save(folder / "records.json", rows)
    save(folder / "step-01/optimizer-attempt.json", {})
    with pytest.raises(ValueError, match="optimizer attempt"):
        check_admission(folder, ROOT, c, train)


@pytest.mark.parametrize("marker", ["records", "scorer-check", "grounding"])
def test_admission_rejects_mock_telemetry_and_grounding_mismatch(tmp_path, marker):
    folder = tmp_path / "smoke"
    run(SMOKE, folder, mock=True)
    c, train, _, _ = load(CONFIG)
    strip_mock_markers(folder)
    save(folder / "verify-base-model-result.json", {"model_data": {"model_name": c["model"]}})
    path = folder / ("scorer-check/records.json" if marker == "scorer-check" else "records.json")
    rows = json.loads(path.read_text())
    if marker == "grounding":
        rows[0]["judge"]["grounding_applicable"] = not rows[0]["case"]["grounding_applicable"]
    else:
        rows[0]["judge"]["telemetry"]["mock"] = True
    save(path, rows)
    save(folder / "smoke-review.json", review_fixture(folder, train))
    with pytest.raises(ValueError, match="grounding differs" if marker == "grounding" else "mock judge"):
        check_admission(folder, ROOT, c, train)


@pytest.mark.parametrize("defect", ["wrong-policy", "missing", "off-task", "tie", "no-limitations", "bad-axis"])
def test_v2_review_fails_closed(tmp_path, defect):
    folder = tmp_path / "smoke"
    run(SMOKE, folder, mock=True)
    _, train, _, _ = load(CONFIG)
    review = review_fixture(folder, train)
    if defect == "wrong-policy":
        review["policy"] = "raw-base-unknown-policy"
    elif defect == "missing":
        review["samples"].pop()
    elif defect == "off-task":
        review["samples"][0]["answers_task"] = False
    elif defect == "tie":
        review["accepted_rankings"][0]["worse"] = 3
    elif defect == "no-limitations":
        review["scorer_limitations"] = []
    elif defect == "bad-axis":
        review["accepted_rankings"][0]["axes"] = ["worth sharing"]
    save(folder / "smoke-review.json", review)
    rows = json.loads((folder / "records.json").read_text())
    policy = json.loads((CONFIG.parent / "admission-v2.json").read_text())
    with pytest.raises(ValueError):
        check_mixed_review(folder, rows, policy, {c["id"] for c in train[:6]})


def test_success_contract_is_pinned_and_requires_review():
    contract = json.loads((CONFIG.parent / "contract.json").read_text())
    assert digest(CONFIG.parent / "contract.json") == "1b38b46283b67aedffdb325dc82428b57640ab8b7c4c9dc2e4ec6b22157f1f9f"
    assert contract["reward_version"] == "jev-style-quality-support-v1"
    assert contract["maximum_quality_drop"] == 0.02
    assert contract["limits"]["optimizer_updates"] == 24
    assert contract["limits"]["automatic_extra_training_passes"] == 0
    assert contract["blind_review"]["required"]
    hashes = fingerprint(ROOT)
    assert "src/fw_jev/raw_base.py" in hashes
    assert "experiments/raw-base-v1/contract.json" in hashes
    # The notebook only shells out to the CLI; autosaves must not stale a smoke.
    assert "notebooks/raw_base_rl.ipynb" not in hashes


def test_notebook_is_clean_opt_in_wrapper_not_second_trainer():
    notebook = nbformat.read(ROOT / "notebooks/raw_base_rl.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = []
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, "notebook", "exec")
            code.append(cell.source)
    joined = "\n".join(code)
    assert "RUN_LIVE = False" in joined and "fw_jev.cli" in joined
    assert "--smoke-from" in joined
    assert "forward_backward" not in joined and "optim_step" not in joined


def test_review_is_optional_unless_required(tmp_path):
    folder = tmp_path / "smoke"
    run(SMOKE, folder, mock=True)
    c, train, _, _ = load(CONFIG)
    strip_mock_markers(folder)
    save(folder / "verify-base-model-result.json", {"model_data": {"model_name": c["model"]}})
    admission = check_admission(folder, ROOT, c, train)
    assert admission["review"] == "not provided" and admission["reviewed_groups"] is None
    with pytest.raises(ValueError, match="review required"):
        check_admission(folder, ROOT, c, train, require_review=True)
    review = review_fixture(folder, train)
    save(folder / "smoke-review.json", dict(review, approved=False, accepted_rankings=[]))
    assert check_admission(folder, ROOT, c, train)["review"] == "not approved"
    with pytest.raises(ValueError):
        check_admission(folder, ROOT, c, train, require_review=True)
    save(folder / "smoke-review.json", review)
    assert check_admission(folder, ROOT, c, train)["review"] == "approved"
