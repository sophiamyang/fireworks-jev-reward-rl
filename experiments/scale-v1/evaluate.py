"""Offline, predeclared all-task evaluation. Never calls a judge or trains.

--prepare-review creates a score-blinded packet of sample 0 for every validation
task, with a secret random A/B key. Fill its ratings before looking at the key;
a later run without the flag writes evaluation.json. No completion is an SFT target.
"""

import argparse
import importlib.util
import json
import random
import secrets
import statistics
from pathlib import Path

from fw_jev.report import stats
from fw_jev.storage import digest, save

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def bootstrap(differences, repetitions, seed):
    rng = random.Random(seed)
    n = len(differences)
    values = sorted(statistics.mean(rng.choices(differences, k=n)) for _ in range(repetitions))
    return [values[int(0.025 * repetitions)], values[int(0.975 * repetitions)]]


def verify_summary(rows, summary):
    """A stale/edited report must not select a checkpoint over the actual drafts."""
    phases = {p: [r for r in rows if r["phase"] == p] for p in ("before", "after")}
    for phase, selected in phases.items():
        if summary.get(phase) != stats(selected):
            raise ValueError("Summary differs from saved records: " + phase)
    ids = sorted({r["case"]["id"] for r in phases["before"]})
    if set(ids) != {r["case"]["id"] for r in phases["after"]}:
        raise ValueError("Before/after task coverage differs")
    paired = []
    for cid in ids:
        pair = {p: stats([r for r in selected if r["case"]["id"] == cid]) for p, selected in phases.items()}
        paired.append(
            {
                "case_id": cid,
                **pair,
                "reward_change": pair["after"]["reward_mean"] - pair["before"]["reward_mean"],
                "words_change": pair["after"]["words_mean"] - pair["before"]["words_mean"],
            }
        )
    if summary.get("prompt_paired") != paired:
        raise ValueError("Prompt-paired summary differs from saved records")


def blind_packet(folder, rows, contract, create=False):
    ids = sorted({r["case"]["id"] for r in rows if r["phase"] == "before"})
    sample = contract["blind_review"]["sample_index_per_prompt"]
    source_hash = digest(folder / "records.json")
    key_path = folder / "blind-key.json"
    if not key_path.exists():
        if not create:
            raise ValueError("Missing blind key; run --prepare-review and complete the review first")
        # Secret, never regenerated: the public validation IDs must not determine the key.
        rng = secrets.SystemRandom()
        key = {
            f"pair-{index:02}": dict(zip(("case_id", "A", "B"), (cid, *rng.sample(["before", "after"], 2))))
            for index, cid in enumerate(rng.sample(ids, len(ids)), 1)
        }
        with key_path.open("x") as handle:
            json.dump({"records_sha256": source_hash, "key": key}, handle, indent=2)
            handle.write("\n")
    stored = read(key_path)
    key = stored["key"]
    if stored["records_sha256"] != source_hash:
        raise ValueError("Blind review belongs to different saved outputs")
    if sorted(k["case_id"] for k in key.values()) != ids or any(
        {k["A"], k["B"]} != {"before", "after"} for k in key.values()
    ):
        raise ValueError("Blind packet/key changed; preserve the original review inputs")
    packet, ratings = [], []
    for pid, assignment in sorted(key.items()):
        selected = {
            r["phase"]: r
            for r in rows
            if r["case"]["id"] == assignment["case_id"]
            and r["sample"] == sample
            and r["phase"] in {"before", "after"}
        }
        packet.append(
            {
                "id": pid,
                "request": selected["before"]["request"],
                "content_checks": selected["before"]["case"]["content_checks"],
                "A": selected[assignment["A"]]["draft"],
                "B": selected[assignment["B"]]["draft"],
            }
        )
        ratings.append(
            {
                "id": pid,
                "winner": None,
                "reason": "",
                "A": {"source_error": None, "key_omission": None, "answers_task": None},
                "B": {"source_error": None, "key_omission": None, "answers_task": None},
            }
        )
    for name, body in (
        ("blind-packet.json", {"records_sha256": source_hash, "pairs": packet}),
        (
            "blind-ratings.json",
            {
                "records_sha256": source_hash,
                "reviewer": "",
                "blinded_to_key_and_rewards": False,
                "pairs": ratings,
            },
        ),
    ):
        path = folder / name
        if not path.exists():
            save(path, body)
        elif read(path)["records_sha256"] != source_hash:
            raise ValueError("Blind review belongs to different saved outputs")
        elif name != "blind-ratings.json" and read(path) != body:
            raise ValueError("Blind packet/key changed; preserve the original review inputs")
    return key


def review_result(folder, key):
    ratings = read(folder / "blind-ratings.json")
    if (
        not isinstance(ratings.get("reviewer"), str)
        or not ratings["reviewer"].strip()
        or ratings.get("blinded_to_key_and_rewards") is not True
    ):
        return {"complete": False, "passed": False}
    pairs = ratings["pairs"]
    if len(pairs) != len(key) or {p["id"] for p in pairs} != set(key):
        raise ValueError("Review must cover each fixed pair exactly once")
    counts = {
        phase: {"wins": 0, "source_error": 0, "key_omission": 0, "nonanswer": 0}
        for phase in ("before", "after")
    }
    for p in pairs:
        if p["winner"] not in {"A", "B", "tie"} or not p["reason"].strip():
            return {"complete": False, "passed": False}
        for side in ("A", "B"):
            if any(
                type(p[side][field]) is not bool for field in ("source_error", "key_omission", "answers_task")
            ):
                return {"complete": False, "passed": False}
            target = counts[key[p["id"]][side]]
            target["source_error"] += p[side]["source_error"]
            target["key_omission"] += p[side]["key_omission"]
            target["nonanswer"] += not p[side]["answers_task"]
        if p["winner"] != "tie":
            counts[key[p["id"]][p["winner"]]]["wins"] += 1
    passed = counts["after"]["wins"] > counts["before"]["wins"] and all(
        counts["after"][field] <= counts["before"][field]
        for field in ("source_error", "key_omission", "nonanswer")
    )
    return {
        "complete": True,
        "passed": passed,
        "counts": counts,
        "reviewer": ratings["reviewer"],
        "caveat": "Assistant-assisted blinded review, not human labels or independently replicated evaluation.",
    }


def evaluate(folder, prepare_review=False):
    folder = Path(folder)
    status, rows, summary = (read(folder / name) for name in ("status.json", "records.json", "summary.json"))
    if status["state"] != "complete" or status["mock"]:
        raise ValueError("Need a completed live run; mock/partial outputs cannot demonstrate improvement")
    verify_summary(rows, summary)
    config = read(folder / "config.json")
    if config.get("experiment") != "raw-simple-v1":
        raise ValueError("Unknown experiment contract")
    path = "experiments/raw-base-v1/contract.json"
    contract = read(folder / "source" / path)
    if contract != read(ROOT / path):
        raise ValueError("Success contract changed after launch")
    if (
        "reward_version" in contract
        and contract["reward_version"] != read(folder / "manifest.json")["reward_version"]
    ):
        raise ValueError("Reward version differs from success contract")
    if len(summary["prompt_paired"]) != 24 or any(summary[p]["samples"] != 48 for p in ("before", "after")):
        raise ValueError("Missing predeclared validation tasks or draws")
    key = blind_packet(folder, rows, contract, create=prepare_review)
    if prepare_review:  # No scores or evaluation.json before the review is filled.
        return {
            "packet": str(folder / "blind-packet.json"),
            "ratings": str(folder / "blind-ratings.json"),
            "next": "Read all pairs and fill ratings before opening the key or evaluation scores.",
        }
    spec = importlib.util.spec_from_file_location("audit", ROOT / "scripts/audit_run.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    integrity = audit.audit(folder)
    changes = [p["reward_change"] for p in summary["prompt_paired"]]
    interval = bootstrap(changes, contract["bootstrap_repetitions"], contract["bootstrap_seed"])
    before, after = summary["before"], summary["after"]
    writing_change = (
        after["writing_reward_before_grounding_mean"] - before["writing_reward_before_grounding_mean"]
    )
    length_ratio = after["words_mean"] / before["words_mean"]
    checks = {
        "completed_all_updates": status["optimizer_updates"] == contract["limits"]["optimizer_updates"],
        "local_integrity": integrity["passed"],
        "combined_reward_gain": statistics.mean(changes) >= contract["minimum_reward_gain"],
        "writing_gain": writing_change >= contract["minimum_writing_gain"],
        "positive_prompt_bootstrap": interval[0] > contract["minimum_prompt_bootstrap_lower_bound"],
        "prompt_win_fraction": sum(x > 0 for x in changes) / len(changes)
        >= contract["minimum_prompt_win_fraction"],
        "guard_failures": (after["guard_failures"] - before["guard_failures"]) / 48
        <= contract["maximum_increase_guard_failure_fraction"],
        "fresh_raw_base": status.get("parent_updates", 0) == 0
        and (folder / "verify-base-model-result.json").exists(),
        "style_gain": after["components"]["style"] - before["components"]["style"]
        >= contract["minimum_style_gain"],
        "quality_not_materially_lower": after["components"]["quality"] - before["components"]["quality"]
        >= -contract["maximum_quality_drop"],
    }
    review = review_result(folder, key)
    result = {
        "contract": contract["version"],
        "records_sha256": digest(folder / "records.json"),
        "before": before,
        "after": after,
        "reward_gain": statistics.mean(changes),
        "writing_gain": writing_change,
        "prompt_cluster_bootstrap_95": interval,
        "length_ratio": length_ratio,
        "substantial_shortening_requires_content_review": length_ratio
        < contract["shortening_review_trigger_ratio"],
        "checks": checks,
        "blind_review": review,
        "rubric_targets_passed": all(checks.values()),
        "meaningful_gain_demonstrated": all(checks.values()) and review["passed"],
        "limitations": contract["honesty"],
    }
    save(folder / "evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--prepare-review",
        action="store_true",
        help="Write only the blind packet, ratings form and key; no scores",
    )
    args = parser.parse_args()
    result = evaluate(args.folder, prepare_review=args.prepare_review)
    print(json.dumps(result, indent=2))
