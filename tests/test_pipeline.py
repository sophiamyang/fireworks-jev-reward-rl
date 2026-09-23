import json
from pathlib import Path

import pytest

from fw_jev.config import load, plan, prompt
from fw_jev.economics import summarize
from fw_jev.mock import response
from fw_jev.raw_base import check_admission
from fw_jev.rl_math import advantages, alignment, datum
from fw_jev.runner import run
from fw_jev.storage import Budget, rpc

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/raw-base-v1/config.json"
SMOKE = ROOT / "experiments/raw-base-v1/smoke.json"


def test_dataset_and_budget():
    c, train, val, _ = load(CONFIG)
    assert len(train) == 96 and len(val) == 24
    assert set(r["scenario_id"] for r in train).isdisjoint(r["scenario_id"] for r in val)
    assert set(r["source_id"] for r in train).isdisjoint(r["source_id"] for r in val)
    assert sum(r["family"] == "social" for r in val) == 12
    assert {r["family"] for r in train[:6]} == {"social", "summary", "rewrite", "reply", "notes", "direct"}
    assert plan(c, train, val)["limits"] == {
        "generations": 864,
        "jev": 896,
        "references": 768,
        "optimizer": 24,
    }
    c, train, val, _ = load(SMOKE)
    assert plan(c, train, val)["limits"] == {"generations": 24, "jev": 56, "references": 24, "optimizer": 0}
    row = dict(train[0], content_checks=["UNIQUE_PRIVATE_CHECKLIST"])
    assert "UNIQUE_PRIVATE_CHECKLIST" not in prompt(row)


def test_advantages_and_masks():
    g = advantages([0.1, 0.2, 0.8, 0.9])
    assert g["eligible"] and sum(g["advantages"]) == pytest.approx(0)
    assert not advantages([0.8, 0.81])["eligible"]
    a, kl = datum([1, 2, 3], [4, 5], [-1.0, -2.0], [-1.0, -2.0], 1.0)
    assert a["input_tokens"] == [1, 2, 3, 4]
    assert a["target_tokens"] == [0, 0, 4, 5]
    assert a["logprobs"] == [0, 0, -1, -2]
    assert a["advantages"] == [0, 0, 128, 128]
    assert kl["sampled_reference_kl"] == 0
    assert alignment([{"logprobs": a["logprobs"]}], [a], [2]) == 0
    with pytest.raises(ValueError):
        alignment([{"logprobs": [0, 0, -10, -20]}], [a], [2])


def test_no_rpc_replay_even_on_timeout(tmp_path):
    calls = []

    def fail():
        calls.append(1)
        raise TimeoutError("uncertain result")

    with pytest.raises(TimeoutError):
        rpc(tmp_path, "optimizer", fail)
    with pytest.raises(FileExistsError):
        rpc(tmp_path, "optimizer", fail)
    assert calls == [1]
    assert not (tmp_path / "optimizer-result.json").exists()


def test_budget_enforced_before_call(tmp_path):
    budget = Budget(tmp_path, {"optimizer": 0})
    with pytest.raises(ValueError):
        budget.take("optimizer")
    assert budget.attempted["optimizer"] == 0


def test_cost_uses_usage_not_word_estimate():
    r = response(0.8)
    r["usage"]["input_tokens"] = 1000
    assert summarize([r])["estimated_usd"] == pytest.approx(0.000042)
    del r["usage"]
    assert summarize([r])["estimated_usd"] is None


def test_mock_full_workflow_no_invented_lift(raw_mock_run, tmp_path):
    out, summary = raw_mock_run
    assert summary["status"]["optimizer_updates"] == 24
    assert summary["status"]["generated_drafts"] == 864
    assert summary["before"]["samples"] == summary["after"]["samples"] == 48
    assert summary["before"]["reward_mean"] == summary["after"]["reward_mean"]
    assert set(summary["before"]["components"]) == {"style", "quality"}
    assert len(summary["prompt_paired"]) == 24
    records = json.loads((out / "records.json").read_text())
    assert len(records) == 864
    metrics = json.loads((out / "step-metrics.json").read_text())
    assert len(metrics) == 24
    for step in metrics:
        assert all(
            step[k] >= 0
            for k in ("reference_seconds", "forward_backward_seconds", "optimizer_and_checkpoint_seconds")
        )
    assert all(r["score"]["grounding"]["applicable"] == r["case"]["grounding_applicable"] for r in records)
    assert "MOCK" in (out / "comparison.html").read_text()
    assert "After · 24 updates" in (out / "paired.html").read_text()
    with pytest.raises(FileExistsError):
        run(CONFIG, out, mock=True)


def test_mock_cannot_authorize_live_training(tmp_path, capsys):
    out = tmp_path / "smoke"
    result = run(SMOKE, out, mock=True)
    assert result["status"]["optimizer_updates"] == 0
    assert not list(out.glob("step-*/optimizer-attempt.json"))
    logs = capsys.readouterr().out
    assert "Smoke batch 1/1: confirmed_updates=0" in logs
    assert "Training batch" not in logs
    metrics = json.loads((out / "step-metrics.json").read_text())
    assert metrics[0]["reference_seconds"] >= 0
    assert "optimizer_and_checkpoint_seconds" not in metrics[0]
    c, train, _, _ = load(CONFIG)
    with pytest.raises(ValueError, match="LIVE"):
        check_admission(out, ROOT, c, train)
    with pytest.raises(ValueError, match="smoke first"):
        run(CONFIG, tmp_path / "live", mock=False)
