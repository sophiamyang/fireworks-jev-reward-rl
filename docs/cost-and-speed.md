# Cost and speed

The **main 24-update run**, including its fixed checks and before/after scoring:

| Measurement | Observed |
|---|---:|
| Jev requests | 896 |
| Questions answered | 2,518 |
| Reported input tokens | 1,462,630 |
| Median / p95 client round trip | 186 ms / 360 ms |
| Sum of request durations | 186.5 seconds |
| Estimated Jev cost | $0.06143 |
| Total run wall time | 71.9 minutes |

Jev request durations accounted for 4.3% of run wall time. These are measured
sequential client round trips, not server latency, concurrent throughput or a
comparison against another scorer.

The saved pricing basis for `jev-1.13.0` was $0.042 per million input tokens;
output tokens were free. [TypeSafe pricing](https://docs.typesafe.ai/models).
This is a recorded list-price estimate, not a current quote or invoice:

```python
estimated_usd = reported_input_tokens * 0.042 / 1_000_000
```

The six-cent figure excludes Fireworks (estimated below), the separate 56-call
smoke (estimated $0.00302), other diagnostics and creation of the saved untrained
baseline shown in the public comparison. The main-run call count includes the
run's own before-training evaluation.

Missing usage is unknown cost, not zero. API failure can also leave billable
usage unreported. Raw telemetry and receipts are retained privately.

For your own measurements, the tutorial's smoke and training commands
automatically record request counts, token usage, latency and estimated scorer
cost. W&B is optional. Do not run another scorer check merely to read this demo.

## Estimated Fireworks cost

Fireworks' bill for the published run was not recorded. At the
[Serverless Training API list prices](https://fireworks.ai/pricing) for
Qwen3.8-27B (checked September 23, 2026), the run comes to an estimated
**$3–5**, and the smoke test to about **$0.10**. Serverless training has no
provisioning or idle fee; you pay per token.

| Operation | List price per 1M tokens | Used for |
|---|---:|---|
| Prefill | $1.86 | Prompts when sampling; the reference pass over each training draft |
| Sample | $5.595 | Generated draft tokens |
| Train | $4.103 | The training pass over each draft's prompt and response |

| Main-run part | Tokens (approx.) | Estimated cost |
|---|---:|---:|
| Sampling 864 drafts (768 training, 96 evaluation) | 190k–300k generated | $1.05–1.70 |
| Prompt prefill for sampling | 27k–175k | $0.05–0.35 |
| Reference pass on 768 training drafts | 320k–430k | $0.60–0.80 |
| Training pass on at most 768 drafts | up to 320k–430k | up to $1.30–1.75 |
| **Total** | | **about $3–5** |

The token counts come from the pinned tokenizer (training prompts average 209
tokens, evaluation prompts 146) and the published responses (about 1.44
tokens per word, 147–243 words on average). The ranges reflect two unknowns:
how long training drafts were between the untrained and final lengths, and
whether a prompt is billed once per group of eight drafts or once per draft.
Groups with no reward spread skip the training pass, so that line is an upper
bound. This is a list-price estimate, not an invoice; your Fireworks billing
page has the real figure.
