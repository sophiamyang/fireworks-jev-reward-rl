"""Fresh-model identity check and zero-update smoke admission."""

import json
from pathlib import Path

from .backend import Fireworks
from .config import prompt
from .reward import VERSION, calculate
from .rl_math import advantages, datum
from .smoke import TECHNICAL
from .storage import digest, rpc


def read(path):
    return json.loads(Path(path).read_text())


def verify_model(requested, reported):
    parts = requested.split("/")
    if len(parts) != 4 or parts[0] != "accounts" or parts[2] != "models":
        raise ValueError("Invalid requested model identity")
    if reported not in {requested, f"/llm-downloader-destination/base/{parts[1]}/{parts[3]}/hf"}:
        raise ValueError("Serverless base model does not match requested model")


class VerifiedRawBase(Fireworks):
    def _initialize(self, config, folder, cases):
        super()._initialize(config, folder, cases)
        info = rpc(folder, "verify-base-model", self.trainer.get_info)
        verify_model(config["model"], info.model_data.model_name)


def check_admission(folder, root, config, train):
    from .runner import fingerprint

    folder = Path(folder)
    policy_path = root / "experiments/raw-base-v1/admission-v2.json"
    policy = read(policy_path)
    status = read(folder / "status.json")
    smoke = read(folder / "config.json")
    manifest = read(folder / "manifest.json")
    if status["state"] != "complete" or status["mock"] or smoke["steps"] != 0:
        raise ValueError("A completed LIVE zero-update smoke is required")
    if manifest["sha256"] != fingerprint(root):
        raise ValueError("Smoke source is stale")
    for name, sha in manifest["sha256"].items():
        if digest(folder / "source" / name) != sha:
            raise ValueError("Smoke source snapshot differs")
    expected = dict(config, steps=0, train_limit=6, group_size=4)
    if smoke != expected:
        raise ValueError("Smoke settings differ from this raw-base training recipe")
    if status.get("optimizer_updates") != 0 or any(folder.glob("step-*/optimizer-attempt.json")):
        raise ValueError("Smoke must perform zero updates and contain no optimizer attempt")
    info = read(folder / "verify-base-model-result.json")
    verify_model(config["model"], info["model_data"]["model_name"])
    fixed = read(folder / "scorer-check/summary.json")
    if fixed.get("passed") is not True:
        raise ValueError("Smoke's fixed-draft checks failed")
    rows = read(folder / "records.json")
    expected_keys = {(case["id"], i) for case in train[:6] for i in range(4)}
    if len(rows) != 24 or {(r["case"]["id"], r["sample"]) for r in rows} != expected_keys:
        raise ValueError("Smoke must contain exactly the six declared training groups")
    cases = {case["id"]: case for case in train[:6]}
    if manifest["reward_version"] != VERSION:
        raise ValueError("Smoke reward version differs")
    fixed_path = folder / "scorer-check/records.json"
    fixed_rows = read(fixed_path) if fixed_path.exists() else []
    if any(r["judge"].get("telemetry", {}).get("mock") for r in rows + fixed_rows):
        raise ValueError("Smoke contains mock judge telemetry")
    for row in rows:
        if (
            row["phase"] != "smoke"
            or row["case"] != cases[row["case"]["id"]]
            or row["request"] != prompt(row["case"])
        ):
            raise ValueError("Smoke contains changed inputs or evaluation data")
        if row["judge"].get("grounding_applicable") is not row["case"]["grounding_applicable"]:
            raise ValueError("Smoke judge grounding differs from its case")
        if (
            calculate(
                row["judge"],
                row["draft"],
                truncated=row["truncated"],
                malformed=row["malformed"],
                weights=config["weights"],
            )
            != row["score"]
        ):
            raise ValueError("Smoke reward replay differs")
        if not all(row["score"]["guards"][k] for k in TECHNICAL):
            raise ValueError("Smoke has technical failures")
        ref = read(folder / "references" / f"smoke-{row['case']['id']}-{row['sample']}.json")
        _, measured = datum(
            row["prompt_tokens"],
            row["tokens"],
            row["logprobs"],
            ref["logprobs"],
            0,
            config["kl_beta"],
        )
        if ref["fixed_sampler"] != status["reference_sampler"] or any(
            ref[k] != value for k, value in measured.items()
        ):
            raise ValueError("Smoke reference trajectory differs")
    eligible = {
        cid
        for cid in cases
        if advantages(
            [r["score"]["reward"] for r in rows if r["case"]["id"] == cid],
            config["min_group_spread"],
            config["advantage_std_floor"],
        )["eligible"]
    }
    if len(eligible) < policy["minimum_eligible_groups"]:
        raise ValueError("Insufficient mixed-objective numeric signal")
    reviewed = check_mixed_review(folder, rows, policy, eligible)
    return {
        "policy": policy["version"],
        "policy_sha256": digest(policy_path),
        "reviewed_groups": reviewed,
        "eligible_groups": len(eligible),
        "reward_version": VERSION,
        "smoke_records_sha256": digest(folder / "records.json"),
        "review_sha256": digest(folder / "smoke-review.json"),
        "fixed_checks_passed": True,
    }


def check_mixed_review(folder, rows, policy, eligible):
    """A combined-objective sanity check, not a pure-style preference benchmark."""
    review = read(folder / "smoke-review.json")
    if (
        review.get("policy") != policy["version"]
        or review.get("records_sha256") != digest(folder / "records.json")
        or review.get("approved") is not True
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
    ):
        raise ValueError("Smoke needs an explicit v2 full-draft review")
    raw = {(r["case"]["id"], r["sample"]): r for r in rows}
    samples = review.get("samples", [])
    labels = {(r["case_id"], r["sample"]): r for r in samples}
    if len(samples) != len(raw) or set(labels) != set(raw):
        raise ValueError("Every smoke sample must be reviewed")
    for item in samples:
        if any(
            type(item.get(k)) is not bool
            for k in (
                "answers_task",
                "source_faithful",
                "key_content_present",
                "ambiguous_request",
            )
        ) or not item.get("note"):
            raise ValueError("Smoke needs explicit task/source/omission/ambiguity labels")
    if not isinstance(review.get("scorer_limitations"), list) or not review["scorer_limitations"]:
        raise ValueError("Record observed scorer limitations")
    pairs = review.get("accepted_rankings", [])
    if len({p["case_id"] for p in pairs}) < policy["minimum_reviewed_groups"]:
        raise ValueError("Insufficient reviewed mixed-objective rankings")
    for pair in pairs:
        cid = pair["case_id"]
        better, worse = (cid, pair["better"]), (cid, pair["worse"])
        if (
            cid not in eligible
            or better == worse
            or better not in raw
            or worse not in raw
            or not pair.get("reason")
            or not pair.get("axes")
            or not set(pair["axes"]).issubset(policy["allowed_preference_axes"])
        ):
            raise ValueError("Invalid mixed-objective ranking evidence")
        good, bad = labels[better], labels[worse]
        if not (good["answers_task"] and bad["answers_task"]):
            raise ValueError("Rankings cannot rely only on nonanswers")
        # The preferred draft may not carry a reviewed defect the other lacks.
        if any(bad[k] and not good[k] for k in ("source_faithful", "key_content_present")):
            raise ValueError("Better draft has a reviewed source or omission defect")
        if "source_support" in pair["axes"] and not (good["source_faithful"] and not bad["source_faithful"]):
            raise ValueError("Source-support ranking needs faithful better and unfaithful worse")
        if (
            raw[better]["score"]["reward"] - raw[worse]["score"]["reward"]
            < policy["minimum_pair_reward_margin"]
        ):
            raise ValueError("Reviewed preference lacks the frozen reward margin")
    return len({p["case_id"] for p in pairs})
