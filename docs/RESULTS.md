# No AI Slop: results after 24 RL updates

The task is No AI Slop: train Qwen3.8-27B to produce natural, specific writing instead of generic filler, canned hype and repetitive templates. Tasks include social posts, summaries, rewrites, replies and short creative pieces. The reward also checks usefulness and support for claims when source context is supplied. Jev is the judge, not the model being trained.

Compared against a saved untrained baseline on the same 24 prompts with the same reward.

Qwen3.8-27B, 96 training prompts and 24 evaluation prompts. Two responses per evaluation prompt per model; all 96 responses are included. No SFT.

| Metric | Untrained | After 24 updates |
|---|---:|---:|
| Combined Jev reward | 0.583 | 0.759 |
| Style | 0.681 | 0.767 |
| Quality | 0.947 | 0.932 |
| Source support | 0.695 | 0.868 |
| Mean words | 243.438 | 146.562 |

Mean reward gain: **+0.176**. 20/24 prompt means increased, 1 unchanged at three decimals, 3 decreased. Responses were **39.8% shorter**.

[Read every prompt and response](examples/comparison.md) · [Run the tutorial](TUTORIAL.md)

## Method and limits

The untrained responses and scores come from a saved untrained baseline evaluated on the same 24 prompts with the same reward; the trained responses and scores come from the completed 24-update run. They were not generated or rescored together. The prompts, model, sampling settings and reward version match, and every saved score replays with the reward calculation. This is a descriptive fixed-baseline comparison, not the run's own before-training evaluation or a checkpoint-selection test.

Jev supplied both training rewards and evaluation scores. The evaluation prompts were excluded from training but had already been inspected. These are stochastic responses on an opened suite, not independent evidence of human preference or authorship. The evaluation mix also differs from training: 12 of 24 evaluation prompts are social posts, versus 18 of 96 training prompts.

Inspect quality and omissions alongside length: shortening may explain part of a reward gain. Hard-guard failures were 2/48 versus 1/48. A trained bio invents a nonsensical occupation. One roof edit contains 'I lentish a ladder'. All responses remain in the comparison.

This page demonstrates the integration, not a best-model claim. Full operational records are retained privately; no saved baseline, score or training journal was overwritten.

## All tasks

| Task | Untrained reward | Trained reward |
|---|---:|---:|
| adapter-review | 0.702 | 0.847 |
| audio-beta | 0.544 | 0.513 |
| ceramic-exhibit | 0.233 | 0.789 |
| changelog-edit | 0.708 | 0.945 |
| festival-notes | 0.479 | 0.682 |
| funding-pilot | 0.429 | 0.673 |
| garden-night | 0.530 | 0.530 |
| membership-reply | 0.661 | 0.828 |
| mentor-post | 0.771 | 0.964 |
| model-abstention | 0.401 | 0.761 |
| museum-returns | 0.617 | 0.877 |
| museum-sign | 0.907 | 0.954 |
| onboarding-job | 0.850 | 0.952 |
| orchard-walk | 0.284 | 0.842 |
| parser-reply | 0.872 | 0.925 |
| policy-rewrite | 0.841 | 0.882 |
| queue-benchmark | 0.622 | 0.877 |
| rainwater-trial | 0.433 | 0.524 |
| release-rollback | 0.484 | 0.695 |
| repair-pilot | 0.473 | 0.825 |
| retrieval-release | 0.000 | 0.505 |
| roof-cut | 0.901 | 0.633 |
| school-map | 0.657 | 0.803 |
| speaker-bio | 0.589 | 0.392 |
