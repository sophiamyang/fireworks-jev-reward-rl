# Lessons from connecting Jev to an RL loop

## Use fewer, distinct signals

Several questions about how “AI-like” a text sounds tend to measure one
overlapping impression and count it more than once. This recipe asks one style
question and one task-quality question, plus source support where it applies.
Simpler arithmetic is easier to debug; it is not proof of a better judge.
[Questions and formula](../README.md#how-it-works).

## Probabilities are useful, but not truth

Use all option probabilities rather than the winning label.
Uncertainty receives half credit by design. These are rewards for writing
behavior, not calibrated probabilities of human authorship.

Test fixed faithful, corrupted, incomplete and repetitive drafts before training.
Within-group ranking stability matters because relative advantages drive updates.

## Grounding needs a scope

A source-grounded social post should not invent a date or product promise.
A fictional story is allowed to invent. Applicability comes from explicit task
metadata, not a broad format such as “social post.”

Source support multiplies the writing score; a high-confidence unsupported claim
also triggers a zero. This helps distinguish unsupported claims from merely
formulaic prose. Jev can still miss errors, and saying less can evade the check.

## Watch length and omissions together

In the displayed comparison, mean length fell from 243 to 147 words while
quality fell slightly. Removing filler is useful; removing the requested detail
is not. Penalizing short answers across the board could instead reward padding.

Read the full outputs and report quality, source support and omissions alongside
reward. Do not assume higher reward means more useful writing. The loss scaling
also has a known length bias; see the
[known reward limitations](../experiments/scale-v1/REWARD.md#known-reward-limitations).

## Separate bad writing from broken infrastructure

Empty, malformed, truncated or looping outputs can receive zero.
API failures must stop the run rather than becoming fake negative examples.
Record an optimizer attempt before sending it, and never replay it if the
outcome is uncertain ([SDK retry caveat](TUTORIAL.md#good-to-know)).

Check how drafts end before trusting those zeros. A draft is complete only if
it ends with an end token; reading the wrong SDK field once made normal
stops look truncated, which would silently zero good drafts. Test termination
handling against the SDK's real sequence objects, not only a mock.

## Keep a stable comparison

A model can generate different answers to the same prompt. A judge can also vary.
The displayed comparison uses a saved untrained baseline evaluated on the same
prompts with the same reward; it does not pretend those answers were generated
during the training run.

Keep original drafts and scores privately, even when publishing only a concise
tutorial and comparison. Never overwrite saved records to make a cleaner story.

## Define success narrowly

This demo shows that structured Jev scores can drive Fireworks RL.
It does not establish a universal anti-slop model, independent human preference,
or general factual accuracy. Keep failures in the comparison and avoid treating
an opened evaluation suite as an untouched holdout.
