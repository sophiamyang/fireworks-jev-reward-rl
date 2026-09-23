# Tutorial: RL on Fireworks with Jev

Train Qwen3.8-27B from its untrained base for 24 RL updates, with Jev scoring
every draft. No GPU needed. You need:

- A **Fireworks API key** ([create one](https://app.fireworks.ai/settings/users/api-keys)),
  on an account with [serverless training](https://docs.fireworks.ai/fine-tuning/training-api/serverless)
  enabled for `accounts/fireworks/models/qwen3p8-27b`.
- A **TypeSafe API key** for Jev ([TypeSafe docs](https://docs.typesafe.ai/introduction)).

| Step | What happens | Cost |
|---|---|---|
| 1. Install | Offline tests and a mock run | Free |
| 2. Keys | Local `.env` | Free |
| 3. Smoke test | 24 drafts, 56 Jev calls, no training | About $0.10 (Fireworks estimate) + under $0.01 Jev |
| 4. Train | 864 drafts, 896 Jev calls, 24 updates, about 72 minutes | About $3–5 (Fireworks estimate) + $0.06 Jev |
| 5. Inspect | Audit, blind review, report | Free |
| 6. Keep your model | Promote and download the adapter | Optional |

Paid commands only run with `--execute`. Results are stochastic; your scores
will differ from the published run. There's also a
[notebook version](../notebooks/raw_base_rl.ipynb) that calls the same commands.

## 1. Install and test offline

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (it also
downloads Python 3.12 for you), then:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or see the uv install page
git clone https://github.com/sophiamyang/fireworks_rl_jev_scorer.git
cd fireworks_rl_jev_scorer
uv sync --locked
uv run pytest -q
uv run fw-jev plan
uv run fw-jev run --output runs/tutorial-mock --mock
```

`plan` prints the request budget. The mock run makes no API calls and
simulates no improvement; open `runs/tutorial-mock/paired.html` to see the
report format. All commands run from the repository root, and every `fw-jev`
command uses [experiments/raw-base-v1/config.json](../experiments/raw-base-v1/config.json)
unless you pass `--config`.

## 2. Add your keys

Copy [.env.example](../.env.example) to `.env` and replace the
`replace_locally` placeholders:

```dotenv
FIREWORKS_API_KEY=replace_locally
TYPESAFE_API_KEY=replace_locally
```

W&B is optional. To log metrics, set `WANDB_API_KEY` (and optionally
`WANDB_ENTITY`) and add `--wandb` to a paid command.

## 3. Run and review a smoke test

The smoke test samples four drafts for each of six training prompts and checks
Jev on 16 fixed drafts. It makes no model updates.

```bash
uv run fw-jev run --config experiments/raw-base-v1/smoke.json \
  --output runs/tutorial-smoke --execute
uv run python scripts/review_raw_smoke.py init runs/tutorial-smoke
```

Open `runs/tutorial-smoke/comparison.html`, read every draft against its
prompt, and fill in `runs/tutorial-smoke/smoke-review.json`:

- **For every draft:** replace the four `null` labels with `true`/`false` and
  add a short note.
  - `answers_task`: it attempts what the prompt asked for.
  - `source_faithful`: it doesn't contradict or invent facts beyond the supplied
    source (`true` when there's no source).
  - `key_content_present`: it keeps the details the prompt says matter.
  - `ambiguous_request`: the prompt itself is unclear enough that reasonable
    drafts could differ.
- **Add at least three entries to `accepted_rankings`**, from three different
  prompts, each naming a draft that's clearly better than another. Each pair needs:
  - a reason, and one or more axes: `style`, `quality` or `source_support`;
  - two drafts that both attempt the task;
  - Jev must agree with you: the draft you rank better needs a Jev reward at
    least 0.05 higher, and the prompt's four rewards must spread by 0.10 or more.

  Rankings must agree with your labels: the better draft can't have a problem
  the worse one doesn't (if the worse draft is faithful or has the key content,
  the better one must too). Both drafts may have errors; then rank them on
  `style` or `quality`, for example one small misstatement against several
  invented details. A `source_support` pair needs the better draft faithful
  and the worse one not.
- **List at least one Jev limitation or disagreement** you noticed in
  `scorer_limitations` (required), set `reviewer` honestly (say if an assistant
  helped), and set `approved: true` when you're done.

A ranking entry looks like this:

```json
{"case_id": "the-smoke-case-id", "better": 0, "worse": 1,
 "axes": ["source_support"], "reason": "Draft 1 invents a launch date."}
```

Then check it:

```bash
uv run python scripts/review_raw_smoke.py check runs/tutorial-smoke
```

This check is the only pass/fail gate for training. When it fails it says why:

- **Review form:** a missing label, field or rule in your file. Fix it and check again.
- **Jev disagrees:** a ranking where Jev's reward gap is under 0.05. Pick another
  pair you and Jev agree on; if there aren't three, stop.
- **Signal:** not enough spread, broken drafts or failed fixed checks. Stop.
  Don't resample until it passes or loosen the rules.
- **Setup:** the smoke doesn't match the current code or config. Run a new smoke.

## 4. Train

```bash
uv run fw-jev run --smoke-from runs/tutorial-smoke \
  --output runs/tutorial-live --execute
```

The trainer re-checks the smoke, scores the untrained model on the 24
evaluation prompts, runs 24 updates (eight drafts per prompt, four prompts per
update), then scores the final model. It stops on its own if something looks
wrong, such as the policy drifting too far from the base model.

The console shows training progress but hides evaluation scores, so you can
still do a blind review. (W&B, if you turned it on, does show them.) If you
change code after the smoke, run a new smoke first; the trainer rejects a smoke
recorded with different code.

If Jev returns a transient error (429, 5xx, timeout), the client retries up to
three times with backoff before stopping. Optimizer calls are never retried.

## 5. Inspect the result

Do the blind review before you look at any scores:

```bash
uv run python scripts/audit_run.py runs/tutorial-live \
  --output runs/tutorial-live/local-audit.json
uv run python experiments/scale-v1/evaluate.py runs/tutorial-live --prepare-review
```

Read each pair in `blind-packet.json` and fill `blind-ratings.json`: your
`reviewer` name, and for every pair a `winner` (`A`, `B` or `tie`), a `reason`,
and `source_error`, `key_omission` and `answers_task` for each side. Set
`blinded_to_key_and_rewards: true` only if you haven't seen any scores. Don't
open `blind-key.json`, the HTML pages or the report until you're done. If you
opened W&B or the report, you've seen scores: set it to `false`, and the
evaluation will report the blind review as incomplete.

Then reveal the scores:

```bash
uv run python experiments/scale-v1/evaluate.py runs/tutorial-live
uv run fw-jev report runs/tutorial-live
```

Open `runs/tutorial-live/paired.html` to read every response. Check for
invented facts, dropped content and off-task answers, not just mean reward.

## 6. Keep and download your model

Checkpoints are only reachable while the Fireworks training session exists, so
promote the final one to a model soon after training. These steps follow
Fireworks'
[serverless training guide](https://docs.fireworks.ai/fine-tuning/training-api/serverless#promote-a-sampler-checkpoint-to-a-model).

1. **Promote the final checkpoint.** The session name is in
   `runs/tutorial-live/fireworks.json`; the final checkpoint is `demo-step-24`.

   Run it with `uv run python` from the repository root:

   ```python
   import os
   from dotenv import load_dotenv
   from fireworks.training.sdk import FireworksClient

   load_dotenv()  # reads FIREWORKS_API_KEY from .env

   fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
   rows = fw.list_training_session_checkpoints("<session_name from fireworks.json>")
   # Find the row for demo-step-24, then:
   fw.promote_session_checkpoint(
       name="<that checkpoint's name>",
       output_model_id="my-no-ai-slop-lora",
       base_model="accounts/fireworks/models/qwen3p8-27b",
   )
   ```

2. **Download it** with firectl ([install it](https://docs.fireworks.ai/tools-sdks/firectl/firectl),
   then run `firectl signin`). Your account may need model-download permission:

   ```bash
   firectl model download accounts/<account-id>/models/my-no-ai-slop-lora ./my-adapter/
   ```

   The [download API](https://docs.fireworks.ai/api-reference/get-model-download-endpoint)
   returns signed URLs instead; treat them like passwords.

3. **Use it** with the exact base model it was trained on. To serve it on
   Fireworks, create an on-demand deployment
   ([deploying trained models](https://docs.fireworks.ai/fine-tuning/deploying-loras)):

   ```bash
   firectl deployment create accounts/<account-id>/models/my-no-ai-slop-lora --deployment-shape default
   ```

   **An on-demand deployment bills for its GPUs by the hour while it exists,**
   whether or not you send requests
   ([pricing](https://fireworks.ai/pricing)). Serverless per-token serving of
   your own LoRA isn't available. Delete it when you're done, using the
   deployment ID that `create` printed:

   ```bash
   firectl deployment delete <deployment-id>
   ```

## Good to know

- **Expected noise in live logs.** The Fireworks SDK may print a warning about
  sampling logprobs, and Tinker may print telemetry tracebacks. Both are
  harmless; the run's `status.json` is what tells you whether it succeeded.

- **Keep `runs/` private and backed up.** It holds every draft, score,
  checkpoint path and optimizer receipt. W&B is not a full backup.
- **Use a new output folder for every run.** The CLI refuses to overwrite one.
- **Never re-run an optimizer step by hand** after an error or timeout; keep
  the folder and investigate. The cookbook doesn't retry optimizer calls
  itself, but the Fireworks SDK can resend a request after a network error
  (with the same sequence ID). This repo can't confirm the server
  de-duplicates it.
- **Temperature is fixed at 1.** The loss math assumes the sampler's
  probabilities are untempered.
- **The loss** is group-relative, unclipped importance sampling with a KL
  penalty to the untrained base; it is not PPO-clipped GRPO.
  [Exact math](../src/fw_jev/rl_math.py).
- **Your report compares against your own untrained evaluation.** The published
  numbers use a [saved untrained baseline](RESULTS.md).
- **Costs:** Fireworks figures are list-price estimates from token counts, not
  a recorded bill; check your Fireworks billing page for actual charges.
  [Cost details](cost-and-speed.md).
