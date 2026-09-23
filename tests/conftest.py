"""Shared offline fixtures. A full 24-update mock run takes about a minute, so each
kind is produced once per session and copied for any test that edits it."""

import shutil
from pathlib import Path

import pytest
import tinker
from tinker.types.forward_backward_output import ForwardBackwardOutput
from tinker.types.save_weights_response import SaveWeightsResponse

from fw_jev import mock
from fw_jev.backend import Fireworks
from fw_jev.runner import run
from fw_jev.storage import rpc

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/raw-base-v1/config.json"
SMOKE = ROOT / "experiments/raw-base-v1/smoke.json"


class SDKTrainer:
    def forward_backward(self, datums, loss):
        assert loss == "importance_sampling"
        return ForwardBackwardOutput("ArrayRecord", [{"logprobs": d.loss_fn_inputs["logprobs"]} for d in datums])


class SDKMockBackend(mock.Backend):
    """Mock sampling, but real SDK response types and journal receipts for updates."""

    def backward(self, folder, arrays, lengths):
        self.tinker, self.trainer = tinker, SDKTrainer()
        return Fireworks.backward(self, folder, arrays, lengths)

    def optimize(self, folder, step, on_optimizer=None, on_checkpoint=None):
        checkpoint = super().optimize(folder, step, on_optimizer=on_optimizer, on_checkpoint=on_checkpoint)
        for name in ("checkpoint", "sampler"):
            rpc(folder, name, lambda: SaveWeightsResponse(path=self.path))
        return checkpoint


@pytest.fixture(scope="session")
def raw_mock_run(tmp_path_factory):
    """Plain --mock run of the published 24-update recipe: (folder, summary). Read-only."""
    out = tmp_path_factory.mktemp("raw-mock") / "run"
    return out, run(CONFIG, out, mock=True)


@pytest.fixture(scope="session")
def journaled_source(tmp_path_factory):
    out = tmp_path_factory.mktemp("journaled") / "run"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mock, "Backend", SDKMockBackend)
        run(CONFIG, out, mock=True)
    return out


@pytest.fixture
def journaled_run(journaled_source, tmp_path):
    """A private copy of the journaled mock run that a test may edit."""
    return Path(shutil.copytree(journaled_source, tmp_path / "journaled"))
