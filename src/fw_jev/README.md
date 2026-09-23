# Code map

The single training entry point is `fw-jev run`. The
[notebook](../../notebooks/raw_base_rl.ipynb) only calls the same CLI. Read [the tutorial](../../docs/TUTORIAL.md) before making paid calls.

| Area | Files |
|---|---|
| CLI and validated configuration | `cli.py`, `config.py`, `writing.py` |
| Generate, score, update, checkpoint | `runner.py`, `backend.py`, `raw_base.py` |
| Judge questions, reward and Jev client | `reward.py` |
| Advantages, masks and KL calculations | `rl_math.py` |
| Fixed-draft checks and smoke admission | `calibration.py`, `smoke.py`, `raw_base.py` |
| Reports, telemetry and durable journals | `report.py`, `economics.py`, `storage.py` |
| Offline test backend and judge | `mock.py` |

`reward.py` holds the one reward: the style, quality and conditional
source-support questions, the probability arithmetic and guards, the Jev
client, and the fixed drafts used by the preflight in `calibration.py`.
[test_reward_protocol.py](../../tests/test_reward_protocol.py) checks that it
reproduces the published questions, request payloads and scores exactly.

`config.py` accepts only the raw-base recipe in `experiments/raw-base-v1/`: the
24-update training schedule and its zero-update smoke. `raw_base.py` verifies the
fresh base model and admits a reviewed live smoke before training.

The cookbook never re-dispatches an optimizer call, but the Fireworks/Tinker SDK
can retransmit `forward_backward`, `optim_step` and save requests after transient
errors (same `seq_id`); `max_retries=0` disables only HTTP-client retries, and
server-side de-duplication is not verified here.
