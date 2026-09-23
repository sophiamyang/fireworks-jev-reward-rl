# Helper scripts

The trainer is `uv run fw-jev`. These scripts support it, not a second trainer.

| Script | Purpose | Model API calls |
|---|---|---|
| [review_raw_smoke.py](review_raw_smoke.py) | Create a blank review / check a completed review | None |
| [audit_run.py](audit_run.py) | Verify saved rewards, tensors and journals | None |
| [publish_comparison.py](publish_comparison.py) | Render a complete saved-baseline comparison without raw operational records | None |

Use [the tutorial](../docs/TUTORIAL.md) for the command sequence.
The 96/24 data checker is
`uv run python experiments/scale-v1/build_dataset.py --check`.
Do not rebuild frozen inputs just to make a check pass.

The publication helper takes explicit `--baseline`, `--trained`, `--output`
and private `--receipt` paths. The receipt must be a new file under `runs/`,
outside both run folders. It validates complete synthetic evaluation coverage,
checks that writer protocol and judge question text match when both runs record
them, and replays saved scores before writing with `--write`. The published
baseline's manifest does not record question text, so its receipt says
`rubric_text_verified: false`; the published claim is limited to matching
prompts, model, sampling settings and reward version, plus exact score replay.
It never samples, rescores or changes training records.
