"""Complete, HTML-escaped examples and descriptive prompt-paired changes."""

import html
import json
import statistics
from pathlib import Path

from .calibration import correlations
from .economics import summarize as economics
from .storage import save


def stats(rows):
    if not rows:
        return {}
    keys = set(rows[0]["score"]["components"])
    versions = {r["score"]["version"] for r in rows}
    if len(versions) != 1 or any(set(r["score"]["components"]) != keys for r in rows):
        raise ValueError("Cannot aggregate different reward protocols")
    return {
        "samples": len(rows),
        "reward_mean": statistics.mean(r["score"]["reward"] for r in rows),
        "writing_reward_before_grounding_mean": statistics.mean(r["score"]["raw_reward"] for r in rows),
        "grounding_support_factor_mean": statistics.mean(
            r["score"].get("grounding_support_factor", 1.0) for r in rows
        ),
        "words_mean": statistics.mean(r["score"]["words"] for r in rows),
        "tokens_mean": statistics.mean(len(r["tokens"]) for r in rows),
        "guard_failures": sum(not all(r["score"]["guards"].values()) for r in rows),
        "truncations": sum(r["truncated"] for r in rows),
        "rubric_threshold_fraction": {
            str(bar): sum(r["score"]["reward"] >= bar for r in rows) / len(rows) for bar in (0.5, 0.7, 0.9)
        },
        "hallucination_gate_failures": sum(r["score"]["grounding"]["blocked"] for r in rows),
        "components": {k: statistics.mean(r["score"]["components"][k] for r in rows) for k in sorted(keys)},
    }


def build(folder):
    folder = Path(folder)
    records = json.loads((folder / "records.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    before = [r for r in records if r["phase"] == "before"]
    after = [r for r in records if r["phase"] == "after"]
    paired = []
    for cid in sorted({r["case"]["id"] for r in before} & {r["case"]["id"] for r in after}):
        b = stats([r for r in before if r["case"]["id"] == cid])
        a = stats([r for r in after if r["case"]["id"] == cid])
        paired.append(
            {
                "case_id": cid,
                "before": b,
                "after": a,
                "reward_change": a["reward_mean"] - b["reward_mean"],
                "words_change": a["words_mean"] - b["words_mean"],
            }
        )
    judged = [r["judge"] for r in records]
    calibration_path = folder / "scorer-check" / "records.json"
    if calibration_path.exists():
        judged = [r["judge"] for r in json.loads(calibration_path.read_text())] + judged
    eco = economics(judged)
    # Each group latency is recorded on every sample; count once per generation call.
    group_times = {}
    for row in records:
        group_times[row["phase"], row["case"]["id"]] = row["generation_group_seconds"]
    measured_generation = sum(group_times.values())
    measured_judge = eco.get("sum_request_seconds", 0)
    summary = {
        "status": status,
        "before": stats(before),
        "after": stats(after),
        "prompt_paired": paired,
        "jev_economics": eco,
        "component_correlations": correlations(
            [
                {
                    "id": f"{r['phase']}/{r['case']['id']}/{r['sample']}",
                    "components": r["score"]["components"],
                }
                for r in records
            ]
        )
        if len(records) > 1
        else {},
        "generation_group_seconds": measured_generation,
        "judge_share_of_generation_plus_scoring_time": measured_judge / (measured_generation + measured_judge)
        if measured_generation + measured_judge
        else None,
        "quality_improvement_demonstrated": False,
        "caveats": [
            "Same training judge; held-out reward is not independent quality validation.",
            "Fresh stochastic samples, paired by prompt—not identical random draws.",
            "Inspect full drafts, content omissions, lengths, and component scores.",
            "Timing share excludes reference, optimizer, setup and checkpoint time.",
            "Threshold fractions are descriptive Jev rubric metrics, not calibrated writing success rates.",
        ],
    }
    budget_path = folder / "budget.json"
    if budget_path.exists():
        attempted = json.loads(budget_path.read_text())["attempted"]["jev"]
        # judge/ holds every response saved before scoring; calibration keeps only scored ones.
        saved = len(list((folder / "judge").glob("*.json")))
        calibrated = len(judged) - len(records)
        summary["jev_economics"]["attempted_requests"] = attempted
        summary["jev_economics"]["requests_without_saved_response"] = max(0, attempted - calibrated - saved)
        summary["jev_economics"]["saved_responses_without_record"] = max(0, saved - len(records))
        summary["jev_economics"]["includes_all_attempted_usage"] = attempted == len(judged)
    save(folder / "summary.json", summary)
    esc = html.escape
    cells = []
    for row in records:
        score = row["score"]
        cells.append(
            f"<article><h3>{esc(row['phase'])} · {esc(row['case']['id'])} · sample {row['sample']}</h3>"
            f"<p>Reward {score['reward']:.3f} · {score['words']} words · "
            f"{len(row['tokens'])} tokens</p>"
            f"<details><summary>Request and supplied context</summary><pre>{esc(row['request'])}</pre>"
            f"<p>Report-only content checks: {esc('; '.join(row['case']['content_checks']))}</p></details>"
            f"<pre>{esc(row['draft'])}</pre>"
            f"<details><summary>All scores and validity checks</summary>"
            f"<pre>{esc(json.dumps(score, indent=2))}</pre></details></article>"
        )
    title = "MOCK — not model results" if status["mock"] else "Fireworks + Jev RL"
    body = (
        f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
        "<style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.5 system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #ccc;padding:20px;"
        "margin:20px 0;border-radius:12px}summary{cursor:pointer}</style>"
        f"<h1>{title}</h1><p>Complete drafts, in recorded order. No best-sample filtering.</p>"
        f"<details open><summary>Results, limitations, cost and speed</summary>"
        f"<pre>{esc(json.dumps(summary, indent=2))}</pre></details>" + "".join(cells)
    )
    (folder / "comparison.html").write_text(body)
    if before or after:
        columns = []
        for cid in dict.fromkeys(r["case"]["id"] for r in before + after):
            case_rows = [r for r in before + after if r["case"]["id"] == cid]
            columns.append(
                f"<section><h2>{esc(cid)}</h2><details><summary>Prompt and supplied context</summary>"
                f"<pre>{esc(case_rows[0]['request'])}</pre></details><div class='pair'>"
            )
            for phase, label in (
                ("before", "Before · 0 updates"),
                ("after", f"After · {status['optimizer_updates']} updates"),
            ):
                columns.append(f"<div><h3>{label}</h3>")
                for row in [r for r in case_rows if r["phase"] == phase]:
                    score = row["score"]
                    columns.append(
                        f"<article><p>Draw {row['sample'] + 1} · reward {score['reward']:.3f} · "
                        f"{score['words']} words</p><pre>{esc(row['draft'])}</pre>"
                        f"<details><summary>Scores and guards</summary>"
                        f"<pre>{esc(json.dumps(score, indent=2))}</pre></details></article>"
                    )
                columns.append("</div>")
            columns.append("</div></section>")
        (folder / "paired.html").write_text(
            f'<!doctype html><meta charset="utf-8"><title>{title} · paired evaluation</title>'
            "<style>body{max-width:1400px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}"
            "section{border-top:2px solid #bbb;margin-top:40px;padding-top:20px}"
            "article{border:1px solid #ddd;padding:20px;margin:16px 0;border-radius:12px}"
            "summary{cursor:pointer}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>"
            f"<h1>{title} · all evaluation drafts</h1>"
            "<p>Same prompts, two independent draws per checkpoint. No best-sample selection. "
            "Jev reward is the training rubric, not independent evidence of better writing.</p>"
            "<p><a href='comparison.html'>Full report, costs, metrics and training drafts</a></p>"
            + "".join(columns)
        )
    return summary
