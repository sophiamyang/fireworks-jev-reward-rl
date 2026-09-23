# No AI Slop — RL-24 adapter

**The trained adapter is not published.** There is no public download, Hugging
Face release or hosted demo, and none is planned. This is a toy demonstration
of Fireworks RL with Jev as the reward scorer, not a production writing model
or an AI-authorship detector.

The final step-24 sampler checkpoint is kept privately in the author's
Fireworks account. To see what it writes, read the
[saved comparison](../docs/examples/comparison.md). To train your own adapter,
run the [tutorial](../docs/TUTORIAL.md); its
[last step](../docs/TUTORIAL.md#6-keep-and-download-your-model) explains how to
keep and download the result.

## Model card

| Item | Value |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Adapter | LoRA, rank 8; text-only RL exercise |
| Training | 24 RL optimizer updates from the untrained base; no SFT |
| Data | 96 synthetic training prompts; 24 separate evaluation prompts |
| Reward | Mean of Jev style and quality, multiplied by conditional source support, with guards |
| Inference format used during training | One user message, no system prompt; thinking disabled |
| Sampling used for the published comparison | Temperature 1; at most 3,072 output tokens |
| Base-model license | [Apache 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) |

The pinned tokenizer revision was
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. This is a tokenizer provenance
record, not independent verification of the revision of Fireworks' base weights.
The adapter alone will not run without the compatible base model.

## Results and limitations

The displayed Jev reward is **0.583 → 0.759**, compared against a saved untrained
baseline evaluated on the same 24 prompts with the same reward and sampling
settings. Those responses were not generated or rescored together. The
comparison includes two responses per evaluation prompt
per model, 48 per model.

- Jev supplied both training rewards and evaluation scores.
- The evaluation prompts were excluded from training updates but had already
  been inspected. This is not an untouched holdout or independent human rating.
- Responses became about 40% shorter; quality fell slightly, from 0.947 to
  0.932. Shortening may account for part of the reward gain.
- Hallucinations, omissions, awkward writing and typos remain. Vision, tools,
  reasoning-enabled generation and general capabilities were not evaluated.
- This is the final checkpoint of the published toy run, not a claim that it
  is the best writing checkpoint.

[All responses and scores](../docs/examples/comparison.md) ·
[Methods](../docs/RESULTS.md) · [Training recipe](../docs/TUTORIAL.md)
