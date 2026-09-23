"""Fixed-draft preflight. Never uses training or validation generations."""

import statistics
from pathlib import Path

from . import economics
from .reward import CANARIES, GROUNDING_FIXTURES, VERSION, calculate, canaries_pass, questions
from .rl_math import advantages
from .storage import digest, save


def run(judge, folder, weights, budget=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save(
        folder / "frozen-check.json",
        {
            "version": VERSION,
            "weights": weights,
            "questions": questions(grounded=True),
            "fixtures": CANARIES,
            "grounding_fixtures": GROUNDING_FIXTURES,
            "repeats": 2,
            "purpose": "Sanity check, not a validated quality benchmark",
            "source_sha256": digest(__file__),
        },
    )
    records = []
    for repeat in range(2):
        for name, request, draft in CANARIES:
            if budget:
                budget.take("jev")
            raw = judge.score(request, draft)
            result = calculate(raw, draft, weights=weights)
            records.append(
                {"id": name, "repeat": repeat, "request": request, "draft": draft, "judge": raw, **result}
            )
            save(folder / "records.json", records)
            print(f"fixed-draft {repeat + 1}/2 {name}: reward={result['reward']:.3f}", flush=True)
        for name, request, draft, grounded, expected_block in GROUNDING_FIXTURES:
            if budget:
                budget.take("jev")
            raw = judge.score(request, draft, grounded=grounded)
            result = calculate(raw, draft, weights=weights)
            records.append(
                {
                    "id": "gate_" + name,
                    "repeat": repeat,
                    "request": request,
                    "draft": draft,
                    "judge": raw,
                    "expected_block": expected_block,
                    **result,
                }
            )
            save(folder / "records.json", records)
            print(f"grounding {repeat + 1}/2 {name}: blocked={result['grounding']['blocked']}", flush=True)
    by_id = {name: [r for r in records if r["id"] == name] for name in sorted({r["id"] for r in records})}
    checks = {}
    for repeat in range(2):
        rows = {r["id"]: r for r in records if r["repeat"] == repeat}
        checks[f"scalar_sanity_repeat_{repeat}"] = canaries_pass({k: v["reward"] for k, v in rows.items()})
        checks[f"quality_detects_omission_repeat_{repeat}"] = (
            rows["complete"]["components"]["quality"] >= rows["omission"]["components"]["quality"] + 0.10
        )
        checks[f"style_detects_awkwardness_repeat_{repeat}"] = (
            rows["plain"]["components"]["style"] >= rows["awkward"]["components"]["style"] + 0.10
        )
        for name, _, _, _, expected_block in GROUNDING_FIXTURES:
            checks[f"grounding_{name}_repeat_{repeat}"] = (
                rows["gate_" + name]["grounding"]["blocked"] == expected_block
            )
        ordered = [rows["gate_" + k] for k in ("supported", "fabricated", "contradiction", "paraphrase")]
        adv = advantages([r["reward"] for r in ordered])["advantages"]
        checks[f"grounded_within_group_advantages_repeat_{repeat}"] = (
            adv is not None and adv[0] > 0 and adv[1] < 0 and adv[2] < 0 and adv[3] > 0
        )
    drift = {key: abs(rows[0]["reward"] - rows[1]["reward"]) for key, rows in by_id.items()}
    checks["fixed_draft_repeat_drift_at_most_0_10"] = max(drift.values()) <= 0.10
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "reward_repeat_absolute_difference": drift,
        "mean_rewards": {key: statistics.mean(r["reward"] for r in rows) for key, rows in by_id.items()},
        "economics": economics.summarize([r["judge"] for r in records]),
        "component_correlations": correlations(records),
        "next": "Run zero-update model smoke before training"
        if all(checks.values())
        else "Stop before optimizer. Inspect disagreements; do not silently tune and rerun.",
        "authorship_detection_validated": False,
        "protocol_sha256": protocol_hashes(),
    }
    save(folder / "summary.json", summary)
    return summary


def protocol_hashes():
    # Questions, fixtures and thresholds live in reward.py; the checks live here.
    here = Path(__file__).parent
    return {f"src/fw_jev/{name}": digest(here / name) for name in ("calibration.py", "reward.py")}


def correlations(records):
    # Average repeats first; don't pretend repeats are independent drafts.
    keys = list(records[0]["components"])
    ids = sorted({r["id"] for r in records})
    values = {
        k: [statistics.mean(r["components"][k] for r in records if r["id"] == i) for i in ids] for k in keys
    }
    return {
        f"{a}__{b}": statistics.correlation(values[a], values[b])
        if statistics.pstdev(values[a]) > 0 and statistics.pstdev(values[b]) > 0
        else None
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    }
