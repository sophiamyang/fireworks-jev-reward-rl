"""Thin text-only Fireworks adapter. No imports from another project."""

import math
import os
import time

from .config import prompt
from .rl_math import alignment, logprobs
from .storage import rpc, save
from .writing import messages

MALFORMED = ("<think>", "</think>", "<tool_call>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


def end_tokens(tok, stop):
    """<|im_end|> plus the tokenizer's EOS/<|endoftext|> IDs when they exist."""
    ends = {stop}
    for value in (tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")):
        if type(value) is int and value != tok.unk_token_id:
            ends.add(value)
    return ends


def raw_sequence(sequence):
    """Provider output as returned, before validation; non-finite values kept as text."""
    probs = sequence.logprobs
    return {
        "tokens": list(sequence.tokens),
        "logprobs": None
        if probs is None
        else [x if type(x) in (int, float) and math.isfinite(x) else repr(x) for x in probs],
        "stop_reason": getattr(sequence, "stop_reason", None) or getattr(sequence, "finish_reason", None),
    }


def parse(sequence, decode, ends, max_tokens):
    """Validate one sampled sequence and derive its draft and termination flags."""
    tokens, probs = list(sequence.tokens), list(sequence.logprobs or [])
    logprobs(probs)
    if not tokens or len(tokens) != len(probs):
        raise ValueError("Missing sampled trajectory")
    # Tinker/Firetitan sequences expose stop_reason ("stop" | "length"); finish_reason is a fallback.
    # The SDK maps every non-"length" finish (including aborts) to "stop", so only a final end
    # token proves a natural stop.
    reason = str(getattr(sequence, "stop_reason", None) or getattr(sequence, "finish_reason", None) or "")
    reason = reason.lower()
    ended = tokens[-1] in ends
    draft = decode(tokens[:-1] if ended else tokens).strip()
    return {
        "draft": draft,
        "raw": decode(tokens),
        "tokens": tokens,
        "logprobs": probs,
        "truncated": reason == "length" or not ended or len(tokens) >= max_tokens,
        "malformed": any(marker in draft for marker in MALFORMED),
        "stop_reason": reason,
        "finish_reason": reason,
    }


class Fireworks:
    def __init__(self, config, folder, cases):
        self.service = self.trainer = self.reference = self.current = None
        try:
            self._initialize(config, folder, cases)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the initialization failure, not a secondary cleanup exception.
            raise

    def _initialize(self, config, folder, cases):
        import tinker
        from fireworks.training.sdk import FiretitanSamplingParams, FiretitanServiceClient
        from transformers import AutoTokenizer, PreTrainedConfig

        self.tinker, self.params = tinker, FiretitanSamplingParams
        self.c, self.folder = config, folder
        self.service = self.trainer = self.reference = self.current = None
        self.tok = AutoTokenizer.from_pretrained(
            config["tokenizer"],
            revision=config["tokenizer_revision"],
            trust_remote_code=False,
            config=PreTrainedConfig(),
        )
        self.stop = self.tok.encode("<|im_end|>", add_special_tokens=False)
        if len(self.stop) != 1:
            raise ValueError("Unexpected Qwen stop token")
        self.prompts = {}
        for case in cases:
            tokens = self.tok.apply_chat_template(
                messages(prompt(case)),
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
                preserve_thinking=False,
            )
            if not isinstance(tokens, list) or any(type(x) is not int for x in tokens):
                raise ValueError("Tokenizer must return a flat list of token IDs")
            if len(tokens) + config["max_tokens"] > config["max_seq_len"]:
                raise ValueError(f"Prompt exceeds sequence limit: {case['id']}")
            self.prompts[case["id"]] = list(tokens)
        save(folder / "prompt-tokens.json", self.prompts)
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key or key == "replace_locally":
            raise ValueError("Set FIREWORKS_API_KEY locally")
        # max_retries=0 disables only HTTP-client retries; tinker's execute_with_retries can still
        # retransmit training calls (optim_step, forward_backward, saves) with the same seq_id.
        self.service = FiretitanServiceClient(
            api_key=key, base_url="https://api.fireworks.ai/training/v1/serverless", max_retries=0
        )
        self.trainer = self.service.create_lora_training_client(
            base_model=config["model"], rank=config["rank"], seed=config["seed"]
        )
        if not self.trainer.run_id:
            raise ValueError("Fireworks returned no training run ID")
        self.url = "https://app.fireworks.ai/dashboard/fine-tuning/serverless-run/" + self.trainer.run_id
        save(
            folder / "fireworks.json",
            {
                "run_id": self.trainer.run_id,
                "url": self.url,
                "session_id": self.service.training_session_id,
                "session_name": self.service.training_session_name,
            },
        )
        self.initial = rpc(
            folder, "initial-sampler", lambda: self.trainer.save_weights_for_sampler("demo-base")
        ).path
        if not self.initial:
            raise ValueError("Missing initial sampler")
        self.path = self.initial
        self.reference = self.service.create_sampling_client(model_path=self.initial, tokenizer=self.tok)
        self.reference_id = self.initial
        self.current = self.service.create_sampling_client(model_path=self.path, tokenizer=self.tok)

    def sample(self, case, count):
        p = self.prompts[case["id"]]
        started = time.perf_counter()
        result = self.current.sample(
            prompt=self.tinker.ModelInput.from_ints(p),
            num_samples=count,
            sampling_params=self.params(
                max_tokens=self.c["max_tokens"], temperature=self.c["temperature"], stop=self.stop
            ),
        ).result(timeout=600)
        elapsed = time.perf_counter() - started
        # Keep the paid provider result even if validation below rejects it.
        folder = getattr(self, "folder", None)
        if folder is not None:
            self.sample_calls = getattr(self, "sample_calls", 0) + 1
            save(
                folder / "provider-samples" / f"{self.sample_calls:04}-{case['id']}.json",
                {
                    "case_id": case["id"],
                    "count": count,
                    "elapsed_seconds": elapsed,
                    "sequences": [raw_sequence(s) for s in result.sequences],
                },
            )
        if len(result.sequences) != count:
            raise ValueError("Incomplete generation group")
        ends = end_tokens(self.tok, self.stop[0])
        rows = []
        for sequence in result.sequences:
            row = parse(
                sequence,
                lambda t: self.tok.decode(t, skip_special_tokens=False),
                ends,
                self.c["max_tokens"],
            )
            rows.append(
                {
                    **row,
                    "prompt_tokens": p,
                    "sampler": self.path,
                    "generation_group_seconds": elapsed,
                }
            )
        return rows

    def reference_probs(self, row):
        return self._probs(self.reference, row)

    def _probs(self, client, row):
        values = client.compute_logprobs(
            self.tinker.ModelInput.from_ints(row["prompt_tokens"] + row["tokens"])
        ).result(timeout=600)
        if len(values) != len(row["prompt_tokens"]) + len(row["tokens"]):
            raise ValueError("Reference logprob length mismatch")
        return logprobs(list(values[len(row["prompt_tokens"]) :]))

    def backward(self, folder, arrays, lengths):
        datums = [
            self.tinker.Datum(
                model_input=self.tinker.ModelInput.from_ints(a["input_tokens"]),
                loss_fn_inputs={k: a[k] for k in ("target_tokens", "logprobs", "advantages")},
            )
            for a in arrays
        ]
        result = rpc(
            folder, "forward-backward", lambda: self.trainer.forward_backward(datums, "importance_sampling")
        )
        measured = alignment(result.loss_fn_outputs, arrays, lengths)
        save(folder / "alignment.json", {"sampled_kl": measured, "limit": 0.05})
        return measured

    def optimize(self, folder, step, on_optimizer=None, on_checkpoint=None):
        rpc(
            folder,
            "optimizer",
            lambda: self.trainer.optim_step(
                self.tinker.AdamParams(
                    learning_rate=self.c["learning_rate"], beta1=0.9, beta2=0.95, eps=1e-12, weight_decay=0.0
                )
            ),
        )
        if on_optimizer is not None:
            on_optimizer()
        state = rpc(folder, "checkpoint", lambda: self.trainer.save_state(f"demo-state-{step}")).path
        path = rpc(folder, "sampler", lambda: self.trainer.save_weights_for_sampler(f"demo-step-{step}")).path
        if not state or not path:
            raise ValueError("Checkpoint missing after optimizer; do not replay")
        checkpoint = {"step": step, "state_path": state, "sampler_path": path, "fireworks_url": self.url}
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)  # Record before the sampler swap, which can still fail.
        self.current.close()
        self.path = path
        self.current = self.service.create_sampling_client(model_path=path, tokenizer=self.tok)
        return checkpoint

    def close(self):
        errors = []
        for client in (self.current, self.reference, self.service):
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]
