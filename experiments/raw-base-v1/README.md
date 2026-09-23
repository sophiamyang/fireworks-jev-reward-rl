# The recipe: 24 updates from raw base

Use [the tutorial](../../docs/TUTORIAL.md) for the complete command sequence.

- Fresh Qwen3.8-27B and rank-8 LoRA; no checkpoint or optimizer state restored.
- Frozen 96 training / 24 evaluation prompts.
- One pass; eight drafts per prompt, four groups per update; at most 24 updates.
- Same `((style + quality) / 2) × source_support` reward throughout, with technical guards.
- Six-task zero-update smoke, checked automatically before training; a full-draft review is optional (`--require-review`).
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

## Optional smoke review

`fw-jev run --smoke-from … --require-review` also requires an approved
full-draft review. Create the form, fill it in, then check it:

```bash
uv run python scripts/review_raw_smoke.py init runs/tutorial-smoke
uv run python scripts/review_raw_smoke.py check runs/tutorial-smoke
```

In `smoke-review.json`:

- **For every draft:** replace the four `null` labels with `true`/`false` and
  add a short note.
  - `answers_task`: it attempts what the prompt asked for.
  - `source_faithful`: it doesn't contradict or invent facts beyond the supplied
    source (`true` when there's no source).
  - `key_content_present`: it keeps the details the prompt says matter.
  - `ambiguous_request`: the prompt itself is unclear enough that reasonable
    drafts could differ.
- **Add at least three entries to `accepted_rankings`**, from three different
  prompts, each naming a draft that's clearly better than another, with a
  `reason` and one or more axes (`style`, `quality`, `source_support`). Both
  drafts must attempt the task, and Jev must agree: the better draft needs a
  reward at least 0.05 higher, in a prompt whose rewards spread by 0.10 or more.
- **Keep rankings consistent with your labels:** the better draft can't have a
  problem the worse one doesn't. Both drafts may have errors; then rank them on
  `style` or `quality`. A `source_support` pair needs the better draft faithful
  and the worse one not.
- **List at least one Jev limitation** in `scorer_limitations`, set `reviewer`
  honestly (say if an assistant helped), and set `approved: true`.

```json
{"case_id": "the-smoke-case-id", "better": 0, "worse": 1,
 "axes": ["quality"], "reason": "Draft 1 invents a document list."}
```

`check` says whether a failure is the form (fix and re-check), a Jev
disagreement (pick another pair; if there aren't three, stop), the signal
(stop) or setup (run a new smoke). Don't resample until a review passes.
The published run passed this review.

