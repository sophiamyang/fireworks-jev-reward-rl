# The recipe: 24 updates from raw base

Use [the tutorial](../../docs/TUTORIAL.md) for the complete command sequence.

- Fresh Qwen3.8-27B and rank-8 LoRA; no checkpoint or optimizer state restored.
- Frozen 96 training / 24 evaluation prompts.
- One pass; eight drafts per prompt, four groups per update; at most 24 updates.
- Same `((style + quality) / 2) × source_support` reward throughout, with technical guards.
- Six-task zero-update smoke and explicit full-draft review before training.
- Both evaluation draws for every task; no best-sample filtering.

[config.json](config.json) is the main run and the CLI default;
[smoke.json](smoke.json) is the matching zero-update preflight.
[admission-v2.json](admission-v2.json) freezes the smoke-review admission
checks. [contract.json](contract.json) records the success criteria that
`experiments/scale-v1/evaluate.py` applies to your own completed run.

Operational records stay private. The [displayed comparison](../../docs/RESULTS.md)
uses a saved untrained baseline; every new execution also records its own
before-training evaluation.

No automatic extra epochs, checkpoint promotion or publication.
Use fresh folders and never replay an uncertain optimizer call. SDK-level
retransmission is a separate caveat; see the
[tutorial](../../docs/TUTORIAL.md#good-to-know).
