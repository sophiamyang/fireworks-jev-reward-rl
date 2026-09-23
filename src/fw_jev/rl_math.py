"""Group-relative, UNCLIPPED importance sampling. Not PPO-clipped GRPO."""

import math
import statistics


def logprobs(values):
    if not values or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x > 1e-5
        for x in values
    ):
        raise ValueError("Invalid token log probabilities")
    return list(values)


def advantages(rewards, minimum_spread=0.1, std_floor=0.1):
    if len(rewards) < 2 or any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError("Invalid reward group")
    mean = statistics.mean(rewards)
    sd = statistics.pstdev(rewards)
    spread = max(rewards) - min(rewards)
    eligible = spread >= minimum_spread and sd > 1e-12
    values = [max(-3.0, min(3.0, (r - mean) / max(sd, std_floor))) for r in rewards] if eligible else None
    return {"mean": mean, "std": sd, "spread": spread, "eligible": eligible, "advantages": values}


def datum(prompt, tokens, behavior, reference, advantage, beta=0.05):
    if not prompt or not tokens or len(tokens) != len(behavior) or len(tokens) != len(reference):
        raise ValueError("Misaligned trajectory")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in prompt + tokens):
        raise ValueError("Invalid token IDs")
    logprobs(behavior)
    logprobs(reference)
    if not math.isfinite(advantage):
        raise ValueError("Invalid advantage")
    delta = [b - r for b, r in zip(behavior, reference, strict=True)]
    prefix = len(prompt) - 1
    # Per-sequence 256/len normalization has the known GRPO length bias (Dr. GRPO): short positive-advantage
    # drafts get larger per-token updates, long negative-advantage drafts smaller per-token penalties.
    scale = 256 / len(tokens)
    a = {
        "input_tokens": prompt + tokens[:-1],
        "target_tokens": [0] * prefix + tokens,
        "logprobs": [0.0] * prefix + behavior,
        "advantages": [0.0] * prefix + [(advantage - beta * max(-5.0, min(5.0, d))) * scale for d in delta],
    }
    kl = statistics.mean(math.exp(max(-50.0, min(50.0, -d))) - 1 + d for d in delta)
    return a, {"sampled_reference_kl": kl, "response_tokens": len(tokens), "loss_scale": scale}


def alignment(outputs, arrays, lengths):
    if len(outputs) != len(arrays) or len(arrays) != len(lengths):
        raise ValueError("Missing forward outputs")
    gaps = []
    for output, a, n in zip(outputs, arrays, lengths, strict=True):
        p = output.get("logprobs")
        p = p.data if hasattr(p, "data") else p
        if p is None or len(p) != len(a["logprobs"]):
            raise ValueError("Policy output shape mismatch")
        logprobs(list(p[-n:]))
        gaps.extend(float(x) - y for x, y in zip(p[-n:], a["logprobs"][-n:], strict=True))
    measured = statistics.mean(math.exp(min(50.0, g)) - 1 - g for g in gaps)
    if measured > 0.05:
        raise ValueError("Behavior/training-policy alignment failed; no optimizer call")
    return measured
