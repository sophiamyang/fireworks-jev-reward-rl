"""Synthetic plumbing tests ONLY. Nothing here simulates learning or model quality."""

from .reward import CANARIES, GROUNDING_FIXTURES, MODEL
from .storage import rpc


def response(value):
    return {
        "model": MODEL,
        "answers": {
            "style": {
                "type": "choice",
                "probabilities": {"yes": value, "no": 1 - value, "uncertain": 0.0},
            },
            "quality": {
                "type": "score",
                "score": 3 * value,
                "probabilities": {"0": 1 - value, "1": 0.0, "2": 0.0, "3": value},
            },
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "telemetry": {"latency_seconds": 0.0, "question_count": 2, "mock": True},
    }


def add_support(result, blocked):
    result["grounding_applicable"] = True
    result["answers"]["unsupported_claim"] = {
        "type": "choice",
        "probabilities": {"yes": 1.0 if blocked else 0.0, "no": 0.0 if blocked else 1.0, "uncertain": 0.0},
    }
    result["telemetry"]["question_count"] = 3
    return result


class Judge:
    def score(self, request, draft, *, grounded=False):
        for _, _, text, applies, blocked in GROUNDING_FIXTURES:
            if draft == text:
                result = response(0.8)
                return add_support(result, blocked) if applies else result
        levels = {
            "plain": 0.9,
            "hype": 0.2,
            "empty": 0.1,
            "loop": 0.1,
            "injection": 0.1,
            "nonsense": 0.1,
            "complete": 0.9,
            "omission": 0.5,
            "dialogue": 0.8,
            "awkward": 0.2,
        }
        for name, _, text in CANARIES:
            if draft == text:
                return response(levels[name])
        result = response(0.2 + 0.1 * (int(draft.split()[1]) % 8))
        return add_support(result, False) if grounded else result

    def close(self):
        pass


class Backend:
    def __init__(self, config, folder, cases):
        self.path = self.initial = "mock/base"
        self.reference_id = self.initial
        self.url = None

    def sample(self, case, count):
        return [
            {
                "draft": f"MOCK {i} — plumbing fixture, not a model output.",
                "raw": f"MOCK {i}",
                "tokens": [10, 11, 12],
                "logprobs": [-1.0, -1.0, -1.0],
                "prompt_tokens": [1, 2],
                "truncated": False,
                "malformed": False,
                "sampler": self.path,
                "stop_reason": "stop",
                "finish_reason": "stop",
                "generation_group_seconds": 0.0,
            }
            for i in range(count)
        ]

    def reference_probs(self, row):
        return row["logprobs"]

    def backward(self, folder, arrays, lengths):
        rpc(folder, "forward-backward", lambda: {"mock": True})
        return 0.0

    def optimize(self, folder, step, on_optimizer=None, on_checkpoint=None):
        rpc(folder, "optimizer", lambda: {"mock": True})
        if on_optimizer is not None:
            on_optimizer()
        path = f"mock/step-{step}"
        checkpoint = {"step": step, "state_path": path, "sampler_path": path, "mock": True}
        if on_checkpoint is not None:
            on_checkpoint(checkpoint)
        self.path = path
        return checkpoint

    def close(self):
        pass
