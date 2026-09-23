import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .reward import DEFAULT_WEIGHTS, PREFLIGHT_REQUESTS, VARIANT, VERSION, validate_weights
from .writing import PROTOCOL

EXPERIMENT = "raw-simple-v1"
DATASET = {
    "train_file": "experiments/scale-v1/train.jsonl",
    "validation_file": "experiments/scale-v1/validation.jsonl",
    "dataset_manifest": "experiments/scale-v1/manifest.json",
}
SCHEDULE_KEYS = ("steps", "train_limit", "validation_limit", "group_size", "groups_per_step", "eval_samples")
# The only accepted schedules: 24-update training and its zero-update smoke.
SCHEDULES = {"train": (24, 96, 24, 8, 4, 2), "smoke": (0, 6, 24, 4, 4, 2)}

FAMILIES = {"social", "summary", "rewrite", "reply", "cut", "notes", "direct"}


def prompt(case):
    # Content checklists are ONLY in the report, never generator or judge input.
    return case["request"].strip() + ("\n\n" + case["context"].strip() if case["context"].strip() else "")


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        if set(row) != {
            "id",
            "scenario_id",
            "source_id",
            "family",
            "request",
            "context",
            "content_checks",
            "grounding_applicable",
            "input_shape",
        }:
            raise ValueError("Unexpected dataset schema")
        if row["family"] not in FAMILIES or not re.fullmatch(r"[a-z0-9_-]+", row["id"]):
            raise ValueError("Invalid case ID or family")
        if any(
            not isinstance(row[k], str) or not row[k].strip()
            for k in ("scenario_id", "source_id", "request", "input_shape")
        ):
            raise ValueError("Empty or invalid request")
        if not isinstance(row["context"], str) or type(row["grounding_applicable"]) is not bool:
            raise ValueError("Context must be text and grounding applicability must be explicit boolean")
        if not isinstance(row["content_checks"], list) or not row["content_checks"]:
            raise ValueError("Missing report-only content checklist")
        if any(not isinstance(x, str) or not x for x in row["content_checks"]):
            raise ValueError("Invalid content checklist")
    return rows


def load(path):
    path = Path(path).resolve()
    root = next(
        (d for d in path.parents if (d / "pyproject.toml").is_file() and (d / "src/fw_jev").is_dir()), None
    )
    if root is None:
        root = path.parent.parent
        if path.parent.name in {"scale-v1", "raw-base-v1"} and root.name == "experiments":
            root = root.parent
    c = json.loads(path.read_text())
    if c.get("experiment") != EXPERIMENT:
        raise ValueError("Unknown experiment profile; expected " + EXPERIMENT)
    if c.get("reward_variant") != VARIANT or c.get("weights") != DEFAULT_WEIGHTS:
        raise ValueError("The recipe requires the frozen style + quality reward and weights")
    validate_weights(c["weights"])
    integer_bounds = {
        "train_limit": (1, 96),
        "validation_limit": (1, 24),
        "steps": (0, 24),
        "groups_per_step": (1, 4),
        "group_size": (2, 8),
        "eval_samples": (1, 2),
        "max_tokens": (128, 8192),
        "max_seq_len": (512, 32768),
        "rank": (1, 64),
        "seed": (0, 2**31 - 1),
    }
    float_bounds = {
        "learning_rate": (1e-7, 1e-4),
        "kl_beta": (0.001, 1),
        "max_reference_kl": (0.001, 0.2),
        "min_group_spread": (0, 1),
        "advantage_std_floor": (0.001, 1),
    }
    required = (
        set(integer_bounds)
        | set(float_bounds)
        | {"model", "tokenizer", "tokenizer_revision", "weights", "temperature"}
        | {"experiment", "reward_variant", *DATASET}
    )
    if set(c) - {"epochs"} != required:
        raise ValueError("Unknown or missing config fields")
    # Sampler logprobs are tempered; reference/trainer logprobs are T=1, so KL and ratios need T=1.
    if type(c["temperature"]) not in (int, float) or c["temperature"] != 1:
        raise ValueError("Invalid config: temperature must be 1")
    for key, (lo, hi) in {**integer_bounds, **float_bounds}.items():
        x = c[key]
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or not lo <= x <= hi
        ):
            raise ValueError(f"Invalid config: {key}")
        if key in integer_bounds and not isinstance(x, int):
            raise ValueError(f"Expected integer: {key}")
    if c["max_tokens"] >= c["max_seq_len"] or not re.fullmatch(r"[a-f0-9]{40}", c["tokenizer_revision"]):
        raise ValueError("Invalid token budget or unpinned tokenizer")
    epochs = c.get("epochs", 1)
    if type(epochs) is not int or epochs != 1:
        raise ValueError("Epoch count differs from the bounded experiment profile")
    if c["steps"] and c["steps"] * c["groups_per_step"] != c["train_limit"] * epochs:
        raise ValueError("Step count must match the bounded training schedule")
    if tuple(c[k] for k in SCHEDULE_KEYS) not in SCHEDULES.values():
        raise ValueError("Schedule differs from the bounded training/smoke profile")
    for key, expected in DATASET.items():
        if c[key] != expected:
            raise ValueError("Recipe must use its frozen dataset: " + key)
    train, val = (load_rows(root / c[k]) for k in ("train_file", "validation_file"))
    manifest = json.loads((root / c["dataset_manifest"]).read_text())
    for key in ("train_file", "validation_file"):
        path = root / c[key]
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
            raise ValueError("Dataset differs from frozen manifest")
    if manifest["smoke_ids"] != [r["id"] for r in train[:6]]:
        raise ValueError("Smoke selection differs from frozen manifest")
    for key in ("id", "scenario_id", "source_id"):
        values = [r[key] for r in train + val]
        if len(set(values)) != len(values):
            raise ValueError(f"Dataset overlap/duplicate: {key}")
    texts = [" ".join(prompt(r).lower().split()) for r in train + val]
    if len(set(texts)) != len(texts):
        raise ValueError("Duplicate prompt across dataset")
    if len(train) < c["train_limit"] or len(val) < c["validation_limit"]:
        raise ValueError("Dataset smaller than configured limit")
    return c, train[: c["train_limit"]], val[: c["validation_limit"]], root


def plan(c, train, val):
    validation = 2 * len(val) * c["eval_samples"] if c["steps"] else 0
    training = len(train) * c["group_size"] * c.get("epochs", 1)
    generations = training + validation
    return {
        "experiment": c["experiment"],
        "writer_protocol": PROTOCOL,
        "reward_variant": c["reward_variant"],
        "reward_version": VERSION,
        "system_prompt": None,
        "starting_policy": "fresh raw base + new LoRA; no restored checkpoint",
        "mode": "train" if c["steps"] else "zero-update smoke",
        "training_prompts": len(train),
        "validation_prompts_used": len(val) if c["steps"] else 0,
        "families": dict(Counter(r["family"] for r in train)),
        "weights": c["weights"],
        "limits": {
            "generations": generations,
            "jev": generations + PREFLIGHT_REQUESTS,
            "references": training,
            "optimizer": c["steps"],
        },
        "note": f"{PREFLIGHT_REQUESTS} fixed-draft checks before service creation. No automatic retries or resumes.",
    }


def training_batches(c, train):
    if not c["steps"]:
        return [train]
    # Preserve the original case order in each epoch. Each visit generates fresh samples.
    return [
        train[i : i + c["groups_per_step"]]
        for _ in range(c.get("epochs", 1))
        for i in range(0, len(train), c["groups_per_step"])
    ]
