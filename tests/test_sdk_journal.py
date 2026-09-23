"""Exercise installed SDK types, not just the primitive-only mock backend."""

import json

import pytest
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.optim_step_response import OptimStepResponse
from tinker.types.save_weights_for_sampler_response import SaveWeightsForSamplerResponse
from tinker.types.save_weights_response import SaveWeightsResponse
from tinker.types.tensor_data import TensorData

from fw_jev.storage import rpc


def test_real_backward_response_preserves_tensor_values_and_identity(tmp_path):
    tensor = TensorData(data=[-1.0, -2.0], dtype="float32", shape=[1, 2])
    value = ForwardBackwardOutput("ArrayRecord", [{"logprobs": tensor}], {"loss": 0.5})
    assert rpc(tmp_path, "forward-backward", lambda: value) is value
    saved = json.loads((tmp_path / "forward-backward-result.json").read_text())
    assert saved["metrics"] == {"loss": 0.5}
    payload = saved["loss_fn_outputs"][0]["logprobs"]
    assert payload == dict(
        data=[-1.0, -2.0], dtype="float32", shape=[1, 2], sparse_crow_indices=None, sparse_col_indices=None
    )
    assert TensorData(**payload).tolist() == tensor.tolist()


@pytest.mark.parametrize(
    "value",
    [
        OptimStepResponse(metrics={"learning_rate": 0.00005}),
        SaveWeightsResponse(path="test/state"),
        SaveWeightsForSamplerResponse(path="test/sampler"),
        TensorData(
            data=[4, 5], dtype="int64", shape=[2, 3], sparse_crow_indices=[0, 1, 2], sparse_col_indices=[1, 2]
        ),
    ],
)
def test_other_sdk_response_types(tmp_path, value):
    assert rpc(tmp_path, "operation", lambda: value) is value
    saved = json.loads((tmp_path / "operation-result.json").read_text())
    if isinstance(value, TensorData):
        restored = TensorData(**saved)
        for key in ("data", "dtype", "shape", "sparse_crow_indices", "sparse_col_indices"):
            assert getattr(restored, key) == getattr(value, key)
    else:
        assert saved == value.model_dump(mode="json")


@pytest.mark.parametrize("value", [object(), {1: "bad key"}])
def test_serialization_failure_never_allows_replay(tmp_path, value):
    calls = []

    def operation():
        calls.append(1)
        return value

    with pytest.raises((TypeError, ValueError)):
        rpc(tmp_path, "operation", operation)
    assert (tmp_path / "operation-error.json").exists()
    assert not (tmp_path / "operation-result.json").exists()
    with pytest.raises(FileExistsError):
        rpc(tmp_path, "operation", operation)
    assert calls == [1]
