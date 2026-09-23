# No AI Slop: RL on Fireworks with Jev as a scorer

A runnable Python cookbook that trains a writer to produce less generic, more
specific text. Fireworks samples drafts and updates a LoRA adapter;
[Jev, TypeSafe's scoring model](https://docs.typesafe.ai/introduction), scores
each draft and those scores become the RL reward. No local GPU, SFT data or
preference dataset required.

**[Run the tutorial](docs/TUTORIAL.md) · [See every response](docs/examples/comparison.md) ·
[Results and limitations](docs/RESULTS.md)**

This is a toy task: a working example of the integration, not a
production-ready writing model.

## How it works

- **Writer:** Qwen3.8-27B with a rank-8 LoRA adapter, trained from the untrained base.
- **Prompts:** 96 training and 24 evaluation prompts: social posts, summaries,
  rewrites, replies, notes and short creative pieces. Just the user's request and
  any source text; no hidden system prompt and no ideal answers.
  [Dataset](experiments/scale-v1/README.md).
- **Loop:** eight drafts per prompt, scored by Jev; drafts are rewarded relative
  to the others in their group. Four prompts per update, 24 updates.
- **Reward:** Jev answers two writing questions, plus a source check when the
  prompt supplies facts:

  ```python
  reward = ((style + quality) / 2) * source_support
  ```

  *Style*: does it read naturally rather than like generic AI slop?
  *Quality*: is it useful writing for the request? *Source support*: does it
  avoid unsupported claims about the supplied material? Empty, truncated,
  malformed or looping drafts, and drafts Jev is at least 90% sure invent a
  claim, score zero. [Implementation](src/fw_jev/reward.py).

## Results

One run, 24 updates. Mean Jev reward on the 24 evaluation prompts, two
responses per prompt:

| | Untrained | After 24 updates |
|---|---:|---:|
| Jev reward | **0.583** | **0.759** |
| Style | 0.681 | 0.767 |
| Quality | 0.947 | 0.932 |
| Source support | 0.695 | 0.868 |
| Mean words | 243 | 147 |

The untrained numbers are a saved untrained baseline evaluated on the same 24
prompts with the same reward and sampling settings. 20 prompts improved, 1 was
unchanged and 3 got worse.

Read these as a demo, not proof of better writing: Jev scored both training and
evaluation, answers got about 40% shorter, and errors remain.
[Method and limitations](docs/RESULTS.md).

### One example: cleaning up a release note

A selected evaluation prompt: rewrite a fictional release note and keep its
limitations. The quotations are unchanged.

Untrained, opening excerpt (Jev reward 0.544; full response 157 words):

> Here is the cleaned-up release note for **Draftkite 1.9**. It maintains a professional tone, clearly explains the fix, highlights the specific limitation as requested, and directly answers whether users need to take action.

After 24 updates, full response (Jev reward 0.952; 56 words):

> **Draftkite 1.9**
>
> *   **Fixed:** Exporting numbered footnotes from Markdown to HTML no longer duplicates the final note. This issue only affected documents ending with a footnote reference. Existing exported HTML files are not repaired automatically; please re-export to see the fix.
> *   **Note:** Nested footnotes are still unsupported. A separate PDF-export problem is still under investigation.

[All 24 prompts with both models' responses](docs/examples/comparison.md),
including the regressions.

### Cost and speed

The run made **896 Jev requests** for an estimated **$0.061**, with a
**186 ms** median round trip, about 3 minutes of scoring in a 72-minute run.
Fireworks sampling and training add an estimated $3–5 at list prices.
[Details](docs/cost-and-speed.md).

## Try it offline

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which
also fetches Python 3.12, then:

```bash
git clone https://github.com/sophiamyang/fireworks-jev-reward-rl.git
cd fireworks-jev-reward-rl
uv sync --locked
uv run pytest -q
uv run fw-jev run --output runs/quickstart-mock --mock
```

The mock makes no API calls and simulates no improvement; open
`runs/quickstart-mock/paired.html` to see the report format. To train for real,
follow [the tutorial](docs/TUTORIAL.md): add keys, run a small smoke test,
look at it, then train.

## What's in the repo

| Path | What it is |
|---|---|
| [src/fw_jev](src/fw_jev/README.md) | Trainer, reward and Jev client (`fw-jev` CLI) |
| [experiments/raw-base-v1](experiments/raw-base-v1/README.md) | The recipe: training and smoke configs, review policy |
| [experiments/scale-v1](experiments/scale-v1/README.md) | Frozen prompts and the evaluation helper |
| [scripts](scripts/README.md) | Smoke review, run audit and comparison publishing |
| [notebooks](notebooks/raw_base_rl.ipynb) | The tutorial as a notebook; it calls the same CLI |
| [docs](docs/README.md) | Tutorial, results, lessons, cost details |
| [model](model/README.md) | Model card; the trained adapter is not published |

Keys go in a local `.env` (see [.env.example](.env.example)). Run records,
probabilities and checkpoint receipts stay in the Git-ignored `runs/` folder.

## License

The code, documentation and prompt data are [MIT-licensed](LICENSE). The Qwen
base model keeps its own license.
