import copy
import importlib.util
import json
import random
import shutil
from pathlib import Path

import pytest

from fw_jev.report import stats
from fw_jev.reward import calculate, make_judge
from fw_jev.storage import save

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluation", ROOT / "experiments/scale-v1/evaluate.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def synthetic(ids):
    # Synthetic test fixtures; published experiments are not unit-test inputs.
    judge = make_judge(mock=True)
    rows = []
    for cid in ids:
        for phase in ("before", "after"):
            for sample in range(2):
                draft = f"MOCK {sample} — unit fixture, not a model output."
                raw = judge.score("Synthetic test prompt", draft)
                rows.append({"case": {"id": cid, "content_checks": []},
                             "request": "Synthetic test prompt", "draft": draft,
                             "phase": phase, "sample": sample,
                             "score": calculate(raw, draft), "tokens": [], "truncated": False})
    summary = {p: stats([r for r in rows if r["phase"] == p]) for p in ("before", "after")}
    summary["prompt_paired"] = []
    for cid in sorted({r["case"]["id"] for r in rows}):
        pair = {
            p: stats([r for r in rows if r["phase"] == p and r["case"]["id"] == cid])
            for p in ("before", "after")
        }
        summary["prompt_paired"].append(
            {
                "case_id": cid,
                **pair,
                "reward_change": pair["after"]["reward_mean"] - pair["before"]["reward_mean"],
                "words_change": pair["after"]["words_mean"] - pair["before"]["words_mean"],
            }
        )
    return rows, summary


@pytest.fixture
def rows_and_summary():
    return synthetic(("synthetic-a", "synthetic-b"))


@pytest.mark.parametrize("change", ["reward", "length", "quality", "paired", "coverage"])
def test_stale_summary_cannot_select_a_checkpoint(rows_and_summary, change):
    rows, summary = rows_and_summary
    evaluation.verify_summary(rows, summary)
    bad = copy.deepcopy(summary)
    if change == "paired":
        bad["prompt_paired"][0]["reward_change"] += 1
    elif change == "coverage":
        bad["prompt_paired"].pop()
    elif change == "quality":
        bad["after"]["components"]["quality"] = 1.0
    else:
        bad["after"]["reward_mean" if change == "reward" else "words_mean"] += 1
    with pytest.raises(ValueError, match="summary|Summary"):
        evaluation.verify_summary(rows, bad)


@pytest.mark.parametrize("artifact", ["blind-packet.json", "blind-key.json"])
def test_edited_review_inputs_are_rejected_even_with_unchanged_hash(tmp_path, rows_and_summary, artifact):
    rows, _ = rows_and_summary
    save(tmp_path / "records.json", rows)
    contract = {"blind_review": {"sample_index_per_prompt": 0}}
    evaluation.blind_packet(tmp_path, rows, contract, create=True)
    path = tmp_path / artifact
    body = json.loads(path.read_text())
    if artifact == "blind-packet.json":
        body["pairs"][0]["A"] = "Not the saved output."
    else:
        first = next(iter(body["key"]))
        body["key"][first]["A"] = body["key"][first]["B"]
    save(path, body)
    with pytest.raises(ValueError, match="packet/key changed"):
        evaluation.blind_packet(tmp_path, rows, contract)


@pytest.mark.parametrize("reviewer,blinded", [(" ", True), (123, True), ("test", "false"), ("test", 1)])
def test_review_confirmation_requires_actual_boolean_and_name(tmp_path, reviewer, blinded):
    save(tmp_path / "blind-ratings.json", {"reviewer": reviewer, "blinded_to_key_and_rewards": blinded})
    assert evaluation.review_result(tmp_path, {}) == {"complete": False, "passed": False}


def test_blind_key_is_secret_random_and_never_regenerated(tmp_path, monkeypatch):
    rows, _ = synthetic([f"synthetic-{i:02}" for i in range(12)])
    contract = {"blind_review": {"sample_index_per_prompt": 0}}
    seeds = iter([1, 2, 3])
    used = []

    def system_random():
        used.append(True)
        return random.Random(next(seeds))

    monkeypatch.setattr(evaluation.secrets, "SystemRandom", system_random)
    keys = []
    for name in ("one", "two"):
        save(tmp_path / name / "records.json", rows)
        keys.append(evaluation.blind_packet(tmp_path / name, rows, contract, create=True))
    assert len(used) == 2 and keys[0] != keys[1]  # Same public IDs, different secret keys.
    assert evaluation.blind_packet(tmp_path / "one", rows, contract, create=True) == keys[0]
    assert evaluation.blind_packet(tmp_path / "one", rows, contract) == keys[0] and len(used) == 2
    save(tmp_path / "three/records.json", rows)
    with pytest.raises(ValueError, match="prepare-review"):
        evaluation.blind_packet(tmp_path / "three", rows, contract)


def test_prepare_review_writes_no_scores(tmp_path):
    rows, summary = synthetic([f"synthetic-{i:02}" for i in range(24)])
    contract = "experiments/raw-base-v1/contract.json"
    (tmp_path / "source" / contract).parent.mkdir(parents=True)
    shutil.copy(ROOT / contract, tmp_path / "source" / contract)
    for name, body in (
        ("status.json", {"state": "complete", "mock": False}),
        ("records.json", rows),
        ("summary.json", summary),
        ("config.json", {"experiment": "raw-simple-v1"}),
        ("manifest.json", {"reward_version": json.loads((ROOT / contract).read_text())["reward_version"]}),
    ):
        save(tmp_path / name, body)
    result = evaluation.evaluate(tmp_path, prepare_review=True)
    assert set(result) == {"packet", "ratings", "next"}
    written = {p.name for p in tmp_path.iterdir()} - {"source", "status.json", "records.json", "summary.json",
                                                         "config.json", "manifest.json"}
    assert written == {"blind-packet.json", "blind-ratings.json", "blind-key.json"}
    packet = (tmp_path / "blind-packet.json").read_text()
    assert "reward" not in packet and "score" not in packet
    assert len(json.loads(packet)["pairs"]) == 24
