# Frozen 96/24 dataset and evaluation helper

The [tutorial](../../docs/TUTORIAL.md) uses these prompt-only inputs
with [the raw-base recipe](../raw-base-v1/config.json).
No drafts, scores or preferred completions are training targets.

```bash
uv run python experiments/scale-v1/build_dataset.py --check
```

Every prompt and source text is fictional and was written with AI assistance
for this cookbook; like the rest of the repository, it is MIT-licensed.
`manifest.json` records file hashes and
`sources.json` holds 60 of the training inputs. Inputs cover source-to-social
posts, summaries, edits, replies, personal notes and direct requests.
Train/evaluation splits remain frozen.

## Task mix

| Writing task | Training prompts | Validation prompts |
|---|---:|---:|
| Source-to-social posts and threads | 18 | 12 |
| Summaries and explainers | 15 | 3 |
| Rewrites and cuts | 24 | 4 |
| Contextual replies | 15 | 2 |
| Notes-to-post writing | 11 | 2 |
| Direct writing requests, including creative pieces | 13 | 1 |
| Total | 96 | 24 |

The validation comparison contains two responses per prompt per model: 48 per
model, 96 responses total. They are evaluation outputs, not training targets.

## Shared helpers

- `build_dataset.py`, `authored.py`, `sources.json`: input assembly and the
  `--check` that verifies `train.jsonl`, `validation.jsonl` and `manifest.json`.
- `evaluate.py`: local integrity checks, score-hidden review and the declared
  success targets in [contract.json](../raw-base-v1/contract.json) for your own run.
- [REWARD.md](REWARD.md): the reward questions, formula and known limitations.

The publication helper is [scripts/publish_comparison.py](../../scripts/publish_comparison.py).
It does not publish raw trajectories or operational records.
