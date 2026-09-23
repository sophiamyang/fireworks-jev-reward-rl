"""Diagnostics and review gate, not additional reward terms."""

import re

from .rl_math import advantages

TECHNICAL = ("nonempty", "complete", "valid_response", "no_repetition_loop")


def nonanswer_flags(draft):
    # Conservative review hints; never alter the reward or silently strip text.
    flags = []
    if re.search(r"^\s*(thinking|analysis|let me think)\s*:", draft, re.I):
        flags.append("planning_preamble")
    if len(re.findall(r"\boption\s+[1-9]", draft, re.I)) >= 2:
        flags.append("option_menu")
    return flags


def diagnose(group, minimum_spread):
    details = []
    for row in group:
        details.append(
            {
                "sample": row["sample"],
                "technical_failure": not all(row["score"]["guards"][k] for k in TECHNICAL),
                "source_gate_zero": row["score"]["grounding"]["blocked"],
                "nonanswer_hints": nonanswer_flags(row["draft"]),
            }
        )
    passing = [
        r
        for r, d in zip(group, details, strict=True)
        if not d["technical_failure"]
        and not d["source_gate_zero"]
        and "planning_preamble" not in d["nonanswer_hints"]
    ]
    values = [r["score"]["reward"] for r in passing]
    numeric = advantages([r["score"]["reward"] for r in group], minimum_spread)
    valid_range = max(values) - min(values) if values else None
    return {
        "case_id": group[0]["case"]["id"],
        "numeric_eligible": numeric["eligible"],
        "all_output_range": max(r["score"]["reward"] for r in group)
        - min(r["score"]["reward"] for r in group),
        "passing_samples": [r["sample"] for r in passing],
        "passing_range": valid_range,
        "style_signal_candidate": len(passing) >= 3 and valid_range >= minimum_spread,
        "samples": details,
        "note": "Menu flags are review hints, not automatic failures under user-only requests. Full-draft review must use the actual requested constraints. Planning preambles remain excluded from candidate rankings.",
    }
