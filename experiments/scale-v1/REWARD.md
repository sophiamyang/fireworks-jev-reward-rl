# Style + quality reward

Reward version `jev-style-quality-support-v1` is the rubric described in the
[main README](../../README.md#how-it-works). The published 24-update run
([raw-base-v1](../raw-base-v1/config.json)) uses it from start to finish.

## What Jev answers

| Question | Role in reward |
|---|---|
| **Does this writing read naturally rather than like generic, formulaic AI slop?** | Style: `P(yes) + 0.5 × P(uncertain)` |
| **Is this high-quality writing for the user's requested purpose?** | Quality: expected readiness level, divided by 3 |
| **Does the draft introduce a material unsupported factual claim about this case?** | Source support: `P(no) + 0.5 × P(uncertain)`; only asked on source-bound tasks |

Quality levels are: 0 unusable; 1 substantial editing needed; 2 useful with minor
editing; 3 ready to use and covers the supplied key points. The full questions,
criteria and safety instructions are in [reward.py](../../src/fw_jev/reward.py).

```text
writing = (style + quality) / 2
reward  = writing × source_support
```

For example, style `0.80`, quality `0.60`, support `0.90` gives
`(0.80 + 0.60) / 2 × 0.90 = 0.63`.

Source support is 1 when the task does not require it, such as fiction. Empty,
truncated, malformed or looping outputs still receive zero. A source-bound draft
also receives zero if `P(material unsupported claim) ≥ 0.90`. API errors halt the
run; they never become training zeros. Every question asked affects the reward.

## Why these questions?

The style question asks one thing: does the prose read naturally rather than
like generic, formulaic AI slop? It judges the writing and does not try to infer
the true author. Quality stays separate so a natural-sounding but incomplete
answer is not treated as ready to use. Source checking stays separate so
natural-sounding invented facts cannot earn full reward.

Style and quality are weighted equally. The two writing scores can still
correlate; fewer questions do not establish that the judge is accurate or
unbiased. No length, virality or punctuation penalty is added.

## Scorer checks

Before any Fireworks session, the runner scores 16 fixed synthetic drafts twice
(32 Jev calls) and runs 21 checks: rankings, omissions, awkward phrasing,
source errors, within-group advantages and repeat drift. This is a sanity check,
not proof of real-world calibration or model improvement. No train/validation
drafts are used to tune the rubric. A failed check stops the run before any
generation or training; never reuse a halted folder.

The success targets and blinded-review requirement for your own run are in
[contract.json](../raw-base-v1/contract.json). The published comparison uses a
saved untrained baseline; its responses and the 24-update run's final drafts were
not generated or rescored together. [Results and limitations](../../docs/RESULTS.md).

W&B logs `style`, `quality`, source support, final reward, lengths, guard failures,
KL, and Jev usage/latency. It does not label the reward a probability of being human.

## Known reward limitations

The reward is frozen so the published scores replay exactly
([tests/test_reward_protocol.py](../../tests/test_reward_protocol.py)); these
limitations are therefore documented rather than patched.

- **Repetition guard.** The whole-draft repeated-trigram guard (zero at ≥0.5)
  misfires both ways: a paragraph pasted twice can score 0.48 and pass, while a
  legitimate seven-line weekly schedule with repeated times can score about
  0.56 and be zeroed.
- **Length bias.** The per-sequence `256 / length` loss scaling has the known
  GRPO length bias (see Dr. GRPO): it can favor shorter answers and plausibly
  contributes to the roughly 40% shortening.
- **Judge encoding.** The judge `state` is sent as a JSON string with ASCII
  escapes (for example `\u2014` for an em dash). This is standard JSON and is
  fine if Jev decodes the field; verify that against Jev before relying on
  punctuation-sensitive judgments.
- **Weak fixed checks.** Two of the 21 fixed checks
  (`grounding_fiction_repeat_*`) are structural and cannot fail. The single
  injection canary is also an obvious non-answer, so it does not isolate
  injection resistance.
- **Fail-closed consistency.** If Jev's reported quality score differs from
  the level implied by its option probabilities by more than 0.06, scoring
  raises instead of guessing; this can halt a run.
