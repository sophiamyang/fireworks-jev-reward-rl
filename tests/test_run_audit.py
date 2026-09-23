import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_audit", Path(__file__).parents[1] / "scripts/audit_run.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_complete_sdk_journal_and_replayed_math(journaled_source):
    result = audit_module.audit(journaled_source)
    assert result["passed"] and result["mock"]
    assert result["checked"] == dict(rewards=864, references=768, groups=96, tensors=768, updates=24)


def test_missing_reference_is_not_a_clean_math_audit(journaled_run):
    next((journaled_run / "references").glob("*.json")).unlink()
    result = audit_module.audit(journaled_run)
    assert not result["passed"]
    assert result["checked"]["references"] == 767
    assert any("missing reference" in issue for issue in result["issues"])


def test_uncertain_optimizer_is_not_a_success(journaled_run):
    (journaled_run / "step-03" / "optimizer-result.json").unlink()
    result = audit_module.audit(journaled_run)
    assert not result["passed"]
    assert result["checked"]["updates"] == 23
    assert "Unresolved optimizer attempt" in result["issues"]


def test_plain_mock_reports_missing_sdk_receipts_without_crashing(raw_mock_run):
    result = audit_module.audit(raw_mock_run[0])
    assert result["mock"] and not result["passed"]
    assert result["checked"]["rewards"] == 864
    assert any("plain --mock does not provide SDK receipts" in issue for issue in result["issues"])


def test_missing_source_snapshot_is_not_a_clean_audit(journaled_run):
    # Move only this temporary test fixture; never delete real run evidence.
    (journaled_run / "source").rename(journaled_run / "source-moved")
    result = audit_module.audit(journaled_run)
    assert not result["passed"]
    assert "Missing frozen source snapshot" in result["issues"]
    assert result["source_snapshot_required"] and not result["source_snapshot_available"]
    relaxed = audit_module.audit(journaled_run, require_source_snapshot=False)
    assert relaxed["passed"] and relaxed["warnings"]
    assert not relaxed["source_snapshot_required"] and not relaxed["source_snapshot_available"]


def test_duplicate_checkpoint_is_reported(journaled_run):
    path = journaled_run / "checkpoints.json"
    checkpoints = json.loads(path.read_text())
    checkpoints.append(checkpoints[-1])
    path.write_text(json.dumps(checkpoints))
    result = audit_module.audit(journaled_run)
    assert not result["passed"]
    assert "Duplicate checkpoint step" in result["issues"]


def test_alignment_gate_failure_is_reported_not_raised(journaled_run):
    path = journaled_run / "step-02" / "forward-backward-result.json"
    body = json.loads(path.read_text())
    for output in body["loss_fn_outputs"]:
        values = output["logprobs"]["data"] if isinstance(output["logprobs"], dict) else output["logprobs"]
        values[:] = [x - 5 for x in values]
    path.write_text(json.dumps(body))
    result = audit_module.audit(journaled_run)
    assert not result["passed"]
    assert any("step-02: Behavior/training-policy alignment failed" in issue for issue in result["issues"])


def test_inputs_must_match_the_source_dataset_snapshot(journaled_run):
    inputs = json.loads((journaled_run / "inputs.json").read_text())
    changed = inputs["validation"][0]
    changed["content_checks"] = [*changed["content_checks"], "Edited after launch."]
    (journaled_run / "inputs.json").write_text(json.dumps(inputs))
    rows = json.loads((journaled_run / "records.json").read_text())
    for row in rows:
        if row["case"]["id"] == changed["id"]:
            row["case"] = changed
    (journaled_run / "records.json").write_text(json.dumps(rows))
    result = audit_module.audit(journaled_run)
    assert result["issues"] == ["Inputs differ from source dataset: validation"]
