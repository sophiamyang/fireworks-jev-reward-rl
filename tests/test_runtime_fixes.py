import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker
from tinker.types import SampledSequence
from tinker.types.optim_step_response import OptimStepResponse
from tinker.types.save_weights_response import SaveWeightsResponse

from fw_jev import mock, report, runner
from fw_jev.backend import Fireworks, end_tokens, parse
from fw_jev.config import load
from fw_jev.economics import summarize
from fw_jev.mock import response
from fw_jev.runner import fingerprint, run

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/raw-base-v1/config.json"
SMOKE = ROOT / "experiments/raw-base-v1/smoke.json"
IM_END, EOT = 99, 98
VOCAB = {1: "Hello", 2: " world", 3: " <|endoftext|> again", IM_END: "<|im_end|>", EOT: "<|endoftext|>"}
ENDS = {IM_END, EOT}


def decode(tokens):
    return "".join(VOCAB[t] for t in tokens)


def sequence(tokens, stop_reason):
    # Same construction as FiretitanSamplingClient._sample_async_impl.
    return SampledSequence(stop_reason=stop_reason, _tokens_list=tokens, _logprobs_list=[-0.5] * len(tokens))


def test_im_end_terminated_stop():
    row = parse(sequence([1, 2, IM_END], "stop"), decode, ENDS, 128)
    assert row["draft"] == "Hello world" and row["raw"] == "Hello world<|im_end|>"
    assert not row["truncated"] and not row["malformed"]
    assert row["stop_reason"] == row["finish_reason"] == "stop"


def test_stop_ending_with_endoftext_is_complete_and_stripped():
    row = parse(sequence([1, 2, EOT], "stop"), decode, ENDS, 128)
    assert row["draft"] == "Hello world"
    assert not row["truncated"] and not row["malformed"]


def test_length_stop_is_truncated():
    row = parse(sequence([1, 2], "length"), decode, ENDS, 128)
    assert row["truncated"] and row["finish_reason"] == "length"
    assert parse(sequence([1, 2], "stop"), decode, ENDS, 2)["truncated"]


def test_stop_reason_without_end_token_is_not_proof_of_completion():
    # The SDK reports "stop" for every non-length finish, including aborts.
    assert parse(sequence([1, 2], "stop"), decode, ENDS, 100)["truncated"]


def test_stray_endoftext_inside_text_is_malformed():
    row = parse(sequence([1, 3, IM_END], "stop"), decode, ENDS, 128)
    assert row["malformed"] and not row["truncated"]


def test_end_tokens_include_distinct_eos():
    tok = SimpleNamespace(eos_token_id=EOT, unk_token_id=None, convert_tokens_to_ids=lambda t: None)
    assert end_tokens(tok, IM_END) == {IM_END, EOT}


class FakeTokenizer:
    eos_token_id, unk_token_id = IM_END, None

    def convert_tokens_to_ids(self, token):
        return EOT

    def decode(self, tokens, skip_special_tokens=False):
        return decode(tokens)


def fake_backend(folder, sequences):
    backend = object.__new__(Fireworks)
    result = SimpleNamespace(sequences=sequences)
    backend.tinker, backend.params = tinker, lambda **kw: kw
    backend.c, backend.folder, backend.path = {"max_tokens": 128, "temperature": 1}, folder, "fake/sampler"
    backend.tok, backend.stop, backend.prompts = FakeTokenizer(), [IM_END], {"case": [5, 6]}
    backend.current = SimpleNamespace(sample=lambda **kw: SimpleNamespace(result=lambda timeout: result))
    return backend


def test_provider_result_saved_before_validation(tmp_path):
    bad = SampledSequence(stop_reason="stop", _tokens_list=[1, IM_END], _logprobs_list=[-0.1, float("nan")])
    backend = fake_backend(tmp_path, [sequence([1, 2, IM_END], "stop"), bad])
    with pytest.raises(ValueError, match="log probabilities"):
        backend.sample({"id": "case"}, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        backend.sample({"id": "case"}, 3)
    first, second = sorted((tmp_path / "provider-samples").glob("*.json"))
    saved = json.loads(first.read_text())
    assert saved["case_id"] == "case" and saved["count"] == 2
    assert saved["sequences"][1] == {"tokens": [1, IM_END], "logprobs": [-0.1, "nan"], "stop_reason": "stop"}
    assert json.loads(second.read_text())["count"] == 3


def test_backend_rows_record_real_stop_reason(tmp_path):
    backend = fake_backend(tmp_path, [sequence([1, 2, EOT], "stop"), sequence([1, 2], "length")])
    rows = backend.sample({"id": "case"}, 2)
    assert [r["finish_reason"] for r in rows] == ["stop", "length"]
    assert [r["truncated"] for r in rows] == [False, True]
    assert rows[0]["draft"] == "Hello world" and rows[0]["prompt_tokens"] == [5, 6]


def fake_repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src/fw_jev").mkdir(parents=True)
    (tmp_path / "experiments").symlink_to(ROOT / "experiments")
    return tmp_path


def write_config(path, **change):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**json.loads(SMOKE.read_text()), **change}))
    return path


def test_config_root_found_from_nested_folder(tmp_path):
    root = fake_repo(tmp_path)
    _, train, _, found = load(write_config(root / "a/b/c/smoke.json"))
    assert found == root and train


@pytest.mark.parametrize("value", [1, 1.0])
def test_temperature_one_accepted(tmp_path, value):
    root = fake_repo(tmp_path)
    assert load(write_config(root / "recipes/t.json", temperature=value))[0]["temperature"] == 1


@pytest.mark.parametrize("change", [{"temperature": 0.7}, {"temperature": True}, {"restore_state": {}}])
def test_invalid_temperature_or_unknown_field_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        load(write_config(fake_repo(tmp_path) / "recipes/bad.json", **change))


def test_null_usage_is_unknown_cost_not_zero():
    r = response(0.8)
    r["usage"] = None
    result = summarize([r, response(0.8)])
    assert result["estimated_usd"] is None and not result["usage_complete"]
    assert result["requests_without_usage"] == 1


def test_setup_failure_before_requests_is_recorded(tmp_path, monkeypatch):
    def copy(source, destination):
        Path(destination).write_text("changed")

    monkeypatch.setattr(runner.shutil, "copy2", copy)
    out = tmp_path / "halted"
    with pytest.raises(ValueError, match="Source changed"):
        run(SMOKE, out, mock=True)
    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "halted" and status["report_state"] == "skipped"
    assert json.loads((out / "failure.json").read_text())["message"] == "Source changed during snapshot"


def test_checkpoint_recorded_when_sampler_swap_fails(tmp_path, monkeypatch):
    class Trainer:
        def optim_step(self, params):
            return OptimStepResponse(metrics={"step_id:last": 1.0})

        def save_state(self, name):
            return SaveWeightsResponse(path="state/" + name)

        def save_weights_for_sampler(self, name):
            return SaveWeightsResponse(path="sampler/" + name)

    class Current:
        def close(self):
            raise RuntimeError("injected close failure")

    class Backend(mock.Backend):
        def __init__(self, c, *args):
            super().__init__(c, *args)
            self.c, self.tinker, self.trainer, self.current = c, tinker, Trainer(), Current()

        def optimize(self, folder, step, on_optimizer=None, on_checkpoint=None):
            return Fireworks.optimize(
                self, folder, step, on_optimizer=on_optimizer, on_checkpoint=on_checkpoint
            )

        def close(self):
            pass

    monkeypatch.setattr(mock, "Backend", Backend)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected close"):
        run(CONFIG, out, mock=True)
    checkpoints = json.loads((out / "checkpoints.json").read_text())
    assert [c["state_path"] for c in checkpoints] == ["state/demo-state-1"]
    assert json.loads((out / "status.json").read_text())["checkpoint_bundles"] == 1


def test_unrecorded_and_unsaved_jev_responses_are_separate(tmp_path):
    out = tmp_path / "smoke"
    run(SMOKE, out, mock=True)
    eco = json.loads((out / "summary.json").read_text())["jev_economics"]
    assert eco["requests_without_saved_response"] == eco["saved_responses_without_record"] == 0
    # One response saved after scoring but before recording; one request with no saved response.
    shutil.copy(next((out / "judge").glob("*.json")), out / "judge" / "smoke-extra-0.json")
    budget = json.loads((out / "budget.json").read_text())
    budget["attempted"]["jev"] += 2
    (out / "budget.json").write_text(json.dumps(budget))
    eco = report.build(out)["jev_economics"]
    assert eco["requests_without_saved_response"] == eco["saved_responses_without_record"] == 1
    assert not eco["includes_all_attempted_usage"]


def test_notebooks_do_not_stale_the_smoke():
    assert not any(name.startswith("notebooks/") for name in fingerprint(ROOT))


def test_non_finite_sdk_values_keep_the_receipt(tmp_path):
    from fw_jev.storage import rpc

    rpc(tmp_path, "optimizer", lambda: {"metrics": {"grad_norm": float("nan"), "loss": 0.5}})
    saved = json.loads((tmp_path / "optimizer-result.json").read_text())
    assert saved == {"metrics": {"grad_norm": "nan", "loss": 0.5}}
