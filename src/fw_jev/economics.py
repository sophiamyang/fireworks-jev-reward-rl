"""Measured client latency and token-based estimates, never billing assertions."""

import math
import statistics

PRICE = {
    "model": "jev-1.13.0",
    "usd_per_million_input_tokens": 0.042,
    "usd_per_million_output_tokens": 0.0,
    "source": "https://docs.typesafe.ai/models",
    "verified_date": "2026-09-21",
    "kind": "published list-price estimate; excludes taxes, discounts, failed attempts and Fireworks",
}


def summarize(responses):
    if not responses:
        return {"requests": 0, "price": PRICE}
    latencies = [r["telemetry"]["latency_seconds"] for r in responses]
    # Missing or null usage is unknown cost, never zero.
    usages = [r.get("usage") if isinstance(r.get("usage"), dict) else {} for r in responses]
    known = [
        isinstance(u.get("input_tokens"), int)
        and not isinstance(u["input_tokens"], bool)
        and u["input_tokens"] >= 0
        for u in usages
    ]
    complete = all(known)
    tokens = sum(u["input_tokens"] for u in usages) if complete else None
    cost = tokens * PRICE["usd_per_million_input_tokens"] / 1e6 if complete else None
    n = len(responses)
    return {
        "requests": n,
        "questions": sum(r["telemetry"]["question_count"] for r in responses),
        "input_tokens": tokens,
        "usage_complete": complete,
        "requests_without_usage": known.count(False),
        "estimated_usd": cost,
        "estimated_usd_per_draft": cost / n if cost is not None else None,
        "latency_seconds": {
            "p50": statistics.median(latencies),
            "p95_nearest_rank": sorted(latencies)[math.ceil(0.95 * n) - 1],
            "min": min(latencies),
            "max": max(latencies),
            "first_request": latencies[0],
            "warm_p50": statistics.median(latencies[1:]) if n > 1 else None,
        },
        "sum_request_seconds": sum(latencies),
        # Failed attempts may still be billed; their usage is not reported.
        "failed_attempts_before_success": sum(len(r["telemetry"].get("failed_attempts", [])) for r in responses),
        "price": PRICE,
        "measurement": "Sequential client-side HTTP round trips of successful attempts; not server-only "
        "latency. Retried failures are counted separately.",
        "caveat": "Small samples are descriptive, not a throughput or competitor benchmark.",
    }
