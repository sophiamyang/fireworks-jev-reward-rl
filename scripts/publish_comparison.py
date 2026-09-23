"""Offline presentation of a saved raw baseline and completed raw-base RL run.

No sampling, rescoring, checkpoint selection or source-record mutation. Only
synthetic validation prompts/drafts and descriptive metrics enter the HTML.
"""
import argparse
import html
import json
import re
from pathlib import Path

from fw_jev.config import load_rows, prompt
from fw_jev.report import stats
from fw_jev.reward import VARIANT, VERSION, calculate
from fw_jev.storage import digest, save

ROOT = Path(__file__).resolve().parents[1]
METHOD = "Compared against a saved untrained baseline on the same 24 prompts with the same reward."
TRAINING_GOAL = (
    "The task is No AI Slop: train Qwen3.8-27B to produce natural, specific writing instead of "
    "generic filler, canned hype and repetitive templates. "
    "Tasks include social posts, summaries, rewrites, replies and short creative pieces. "
    "The reward also checks usefulness and support for claims when source context is supplied. "
    "Jev is the judge, not the model being trained."
)
TRAINING_TOPICS = (
    "Training topics span AI and developer tools (MoE, RLVR, inference batching, caching and databases), "
    "product updates, arts and community events, work and personal posts, and everyday creative writing."
)
FAMILY_LABELS = {
    "social": "Social post / thread", "summary": "Summary / explainer",
    "rewrite": "Rewrite", "reply": "Contextual reply", "cut": "Shorten a draft",
    "notes": "Notes to post", "direct": "Direct writing request",
}
VALIDATION_TOPICS = {
    "adapter-review": "Small-adapter benchmark",
    "audio-beta": "Rehearsal-feedback app beta",
    "ceramic-exhibit": "Ceramics exhibition",
    "changelog-edit": "Footnote export bug fix",
    "festival-notes": "Book festival volunteer roundup",
    "funding-pilot": "Oral-history arts grant",
    "garden-night": "Neighborhood lantern evening",
    "membership-reply": "Club membership and ticket refund",
    "mentor-post": "Open-source mentor thank-you",
    "model-abstention": "AI assistant abstention experiment",
    "museum-returns": "Museum activity-box return policy",
    "museum-sign": "A museum's squeaky door",
    "onboarding-job": "First week as a technical writer",
    "orchard-walk": "Community orchard walk",
    "parser-reply": "CSV import: blanks versus zero",
    "policy-rewrite": "Studio booking cancellation policy",
    "queue-benchmark": "Job-queue batching benchmark",
    "rainwater-trial": "Greenhouse rain-barrel trial",
    "release-rollback": "Editor release rollback",
    "repair-pilot": "Community repair-counter pilot",
    "retrieval-release": "Document-search release",
    "roof-cut": "A ladder-and-tennis-ball mishap",
    "school-map": "School walking-route map",
    "speaker-bio": "Repair-workshop speaker bio",
}


def read(path):
    return json.loads(Path(path).read_text())


def escape(text, escape_html=html.escape):
    return re.sub(r"[ \t]+(?=\n|$)", lambda m: "".join(f"&#{ord(c)};" for c in m[0]), escape_html(text))


def code_spans(line):
    """Single-line code spans, or None when any backtick run could pair elsewhere."""
    if "\\`" in line:
        return None
    runs, spans, i = list(re.finditer(r"`+", line)), [], 0
    while i < len(runs):
        j = next((j for j in range(i + 1, len(runs)) if len(runs[j][0]) == len(runs[i][0])), None)
        if j is None or "|" in line[runs[i].start():runs[j].end()]:
            return None
        spans.append((runs[i].start(), runs[j].end()))
        i = j + 1
    return spans


def markdown_html(text):
    """HTML-escape, except inside code spans, where Markdown already shows text literally."""
    out = []
    for block in re.split(r"(\n[ \t]*\n)", text):
        found = [code_spans(line) for line in block.split("\n")]
        if None in found:  # An unpaired backtick keeps the conservative escape for its paragraph.
            out.append(html.escape(block))
            continue
        lines = []
        for line, spans in zip(block.split("\n"), found, strict=True):
            end, parts = 0, []
            for start, stop in [*spans, (len(line), len(line))]:
                parts.append(html.escape(line[end:start]) + line[start:stop])
                end = stop
            lines.append("".join(parts))
        out.append("\n".join(lines))
    return "".join(out)


def rubric(baseline, trained):
    manifests = [read(p / "manifest.json") if (p / "manifest.json").exists() else {} for p in (baseline, trained)]
    keys = [k for k in ("writer_protocol", "questions") if all(k in m for m in manifests)]
    if any(manifests[0][k] != manifests[1][k] for k in keys):
        raise ValueError("Incompatible comparison: writer protocol or judge questions")
    if len(keys) == 2:
        return {"rubric_text_verified": True, "rubric_text_note": "Writer protocol and judge questions match."}
    return {"rubric_text_verified": False,
            "rubric_text_note": "A manifest lacks writer_protocol/questions, so judge question text was not "
                                "verified; reward version, config keys and every saved score replay were checked."}


def prepare(baseline, trained):
    baseline, trained = Path(baseline), Path(trained)
    bc, tc = read(baseline / "config.json"), read(trained / "config.json")
    status = read(trained / "status.json")
    if read(baseline / "status.json")["state"] != "complete":
        raise ValueError("Baseline evaluation is incomplete")
    if status.get("mock") or status["state"] != "complete" or status["optimizer_updates"] != 24:
        raise ValueError("Requires the complete live 24-update run")
    recipe = read(ROOT / "experiments/raw-base-v1/config.json")
    if status.get("parent_updates", 0) != 0 or set(tc) != set(recipe) or tc["reward_variant"] != VARIANT:
        raise ValueError("The presentation must use the fresh raw-base recipe and its reward")
    for key in ("model", "tokenizer", "tokenizer_revision", "temperature", "max_tokens", "reward_variant", "weights"):
        if bc[key] != tc[key]:
            raise ValueError(f"Incompatible comparison: {key}")
    cases = {c["id"]: c for c in load_rows(ROOT / "experiments/scale-v1/validation.jsonl")}
    version = VERSION
    final = read(trained / "checkpoints.json")[-1]["sampler_path"]
    rows = []
    for folder, phase in ((baseline, "before"), (trained, "after")):
        selected = [r for r in read(folder / "records.json") if r["phase"] == phase]
        if len(selected) != 48 or {(r["case"]["id"], r["sample"]) for r in selected} != {
            (cid, i) for cid in cases for i in range(2)
        }:
            raise ValueError("All 24 tasks and both draws are required")
        for row in selected:
            case = cases[row["case"]["id"]]
            if row["case"] != case or row["request"] != prompt(case):
                raise ValueError("Only the frozen synthetic inputs may be published")
            before = {bc["model"], read(baseline / "status.json").get("initial_sampler")}
            if (row["sampler"] not in before) if phase == "before" else row["sampler"] != final:
                raise ValueError("Wrong baseline or final sampler")
            replay = calculate(row["judge"], row["draft"], truncated=row["truncated"],
                               malformed=row["malformed"], weights=tc["weights"])
            if row["score"] != replay or replay["version"] != version:
                raise ValueError("Saved reward does not replay")
        rows.extend(selected)
    summary = {p: stats([r for r in rows if r["phase"] == p]) for p in ("before", "after")}
    paired = []
    for cid in sorted(cases):
        pair = {p: stats([r for r in rows if r["case"]["id"] == cid and r["phase"] == p])
                for p in ("before", "after")}
        paired.append({"case_id": cid, **pair})
    summary["paired"] = paired
    summary["rubric"] = rubric(baseline, trained)
    return rows, summary


def directions(summary, noun):
    """Count prompt-mean changes at the three decimals shown in the tables."""
    pairs = [(round(p["before"]["reward_mean"], 3), round(p["after"]["reward_mean"], 3)) for p in summary["paired"]]
    up, same = sum(a > b for b, a in pairs), sum(a == b for b, a in pairs)
    return f"{up}/{len(pairs)} {noun} means increased, {same} unchanged at three decimals, {len(pairs) - up - same} decreased."


def render(rows, summary):
    b, a = summary["before"], summary["after"]
    social = len({r["case"]["id"] for r in rows if r["case"]["family"] == "social"})
    train = load_rows(ROOT / "experiments/scale-v1/train.jsonl")
    train_social = sum(c["family"] == "social" for c in train)
    shorter = 100 * (1 - a["words_mean"] / b["words_mean"])
    defects = []
    trained_text = "\n".join(r["draft"] for r in rows if r["phase"] == "after")
    if "licensed password manager" in trained_text:
        defects.append("A trained bio invents a nonsensical occupation.")
    if "I lentish a ladder" in trained_text:
        defects.append("One roof edit contains 'I lentish a ladder'.")
    failure_note = " ".join(defects)
    table = ["| Metric | Untrained | After 24 updates |", "|---|---:|---:|"]
    for label, old, new in (
        ("Combined Jev reward", b["reward_mean"], a["reward_mean"]),
        ("Style", b["components"]["style"], a["components"]["style"]),
        ("Quality", b["components"]["quality"], a["components"]["quality"]),
        ("Source support", b["grounding_support_factor_mean"], a["grounding_support_factor_mean"]),
        ("Mean words", b["words_mean"], a["words_mean"]),
    ):
        table.append(f"| {label} | {old:.3f} | {new:.3f} |")
    report = ["# No AI Slop: results after 24 RL updates", "", TRAINING_GOAL, "", METHOD, "",
              "Qwen3.8-27B, 96 training prompts and 24 evaluation prompts. Two responses per evaluation prompt per model; all 96 responses are included. No SFT.", "",
              *table, "", f"Mean reward gain: **{a['reward_mean'] - b['reward_mean']:+.3f}**. {directions(summary, 'prompt')} Responses were **{shorter:.1f}% shorter**.", "",
              "[Read every prompt and response](examples/comparison.md) · [Run the tutorial](TUTORIAL.md)", "", "## Method and limits", "",
              "The untrained responses and scores come from a saved untrained baseline evaluated on the same 24 prompts with the same reward; the trained responses and scores come from the completed 24-update run. They were not generated or rescored together. The prompts, model, sampling settings and reward version match, and every saved score replays with the reward calculation. This is a descriptive fixed-baseline comparison, not the run's own before-training evaluation or a checkpoint-selection test.", "",
              "Jev supplied both training rewards and evaluation scores. The evaluation prompts were excluded from training but had already been inspected. These are stochastic responses on an opened suite, not independent evidence of human preference or authorship. "
              f"The evaluation mix also differs from training: {social} of 24 evaluation prompts are social posts, versus {train_social} of {len(train)} training prompts.", "",
              f"Inspect quality and omissions alongside length: shortening may explain part of a reward gain. Hard-guard failures were {b['guard_failures']}/48 versus {a['guard_failures']}/48. {failure_note} All responses remain in the comparison.", "",
              "This page demonstrates the integration, not a best-model claim. Full operational records are retained privately; no saved baseline, score or training journal was overwritten.", "", "## All tasks", "",
              "| Task | Untrained reward | Trained reward |", "|---|---:|---:|"]
    for p in summary["paired"]:
        report.append(f"| {p['case_id'].removeprefix('scale1-validation-')} | {p['before']['reward_mean']:.3f} | {p['after']['reward_mean']:.3f} |")
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<title>Untrained vs 24 RL updates</title><style>body{max-width:1360px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17202b;background:#f7f8fa}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}article{background:white;border:1px solid #ddd;border-radius:12px;padding:22px;margin:16px 0}section{border-top:1px solid #ccd3db;margin-top:36px;padding-top:16px}summary{cursor:pointer}small{color:#52606d}@media(max-width:760px){.pair{grid-template-columns:1fr}}</style>",
             f"<h1>No AI Slop: untrained → 24 RL updates</h1><p><strong>Mean Jev reward: {b['reward_mean']:.3f} → {a['reward_mean']:.3f}</strong> · {directions(summary, 'task')}</p>",
             f"<h2>Training task: No AI Slop</h2><p>{escape(TRAINING_GOAL)}</p><p>{escape(TRAINING_TOPICS)}</p>",
             "<p>Training: 96 prompts. The examples below are separate validation tasks: 24 prompts × two responses per model, not training targets.</p>",
             f"<p><small>{METHOD} Saved baseline scores; no new baseline generation or rescoring.</small></p>",
             "<p>All 24 prompts and both responses per model, including failures. Jev is also the training judge; reward is not a probability of human authorship. <a href='../RESULTS.md'>Method and limitations</a></p>"]
    for pair in summary["paired"]:
        cid = pair["case_id"]
        selected = sorted([r for r in rows if r["case"]["id"] == cid], key=lambda r: r["sample"])
        topic = VALIDATION_TOPICS[cid.removeprefix('scale1-validation-')]
        family = FAMILY_LABELS[selected[0]["case"]["family"]]
        parts.append(f"<section id='{escape(cid)}'><h2>{escape(topic)}</h2><p>Task: {escape(family)}</p><details><summary>Prompt and source context</summary><pre>{escape(selected[0]['request'])}</pre></details><div class='pair'>")
        for phase, label in (("before", "Untrained · saved baseline"), ("after", "Trained · 24 RL updates")):
            parts.append(f"<div><h3>{label}</h3>")
            for row in [r for r in selected if r["phase"] == phase]:
                s = row["score"]
                parts.append(f"<article data-phase='{phase}' data-reward='{s['reward']}' data-sample='{row['sample']}'><p>Response {row['sample'] + 1} · reward {s['reward']:.3f} · {s['words']} words</p><pre>{escape(row['draft'])}</pre></article>")
            parts.append("</div>")
        parts.append("</div></section>")
    parts.append("</html>\n")
    return "\n".join(report) + "\n", "\n".join(parts)


def quote(text):
    """Keep model Markdown readable; escape HTML outside code, trailing whitespace and setext underlines."""
    lines, fenced, previous = [], False, ""
    for line in escape(text, markdown_html).split("\n"):
        fence = re.match(r" {0,3}(```|~~~)", line)
        fenced ^= bool(fence)
        if not fenced and not fence and previous.strip() and re.fullmatch(r" {0,3}(-+|=+)", line):
            line = re.sub(r"[-=]", r"\\\g<0>", line, count=1)  # Otherwise the previous line becomes a heading.
        previous = "" if fence else line
        lines.append("> " + line if line else ">")
    return "\n".join(lines)


def render_markdown(rows, summary):
    b, a = summary["before"], summary["after"]
    parts = [
        "# No AI Slop: untrained vs 24 RL updates", "",
        "## What are we training?", "", TRAINING_GOAL, "", TRAINING_TOPICS, "",
        "Training uses **96 prompts**. The examples on this page are **validation**, not training targets: "
        "**24 separate prompts × two responses = 48 outputs per model** (96 outputs shown in total).", "",
        f"**Mean Jev reward: {b['reward_mean']:.3f} → {a['reward_mean']:.3f}.**", "",
        f"*{METHOD}*", "",
        "Untrained: 48 saved responses and their original scores from the saved untrained baseline. "
        "Trained: 48 responses and scores from the 24-update run. "
        "All 24 prompts and both responses per model are included, including failures. "
        "The responses were not generated or rescored together.", "",
        "Open a task below, then expand its prompt and response comparisons. "
        "Jev is also the training judge; higher reward is not independent evidence of better writing.", "",
        "[Method and limitations](../RESULTS.md) · [Tutorial](../TUTORIAL.md) · "
        "[Optional HTML version](comparison.html) (download/open locally for side-by-side viewing)", "",
        "## Tasks", "",
        "Each score below averages both responses for that task.", "",
        "| Topic | Writing task | Untrained | Trained |", "|---|---|---:|---:|",
    ]
    for pair in summary["paired"]:
        short = pair["case_id"].removeprefix("scale1-validation-")
        case = next(r["case"] for r in rows if r["case"]["id"] == pair["case_id"])
        topic, family = VALIDATION_TOPICS[short], FAMILY_LABELS[case["family"]]
        parts.append(f"| [{topic}](#{short}) | {family} | {pair['before']['reward_mean']:.3f} | {pair['after']['reward_mean']:.3f} |")
    for pair in summary["paired"]:
        cid = pair["case_id"]
        selected = {(r["phase"], r["sample"]): r for r in rows if r["case"]["id"] == cid}
        parts.extend([
            "", f"## {cid.removeprefix('scale1-validation-')}", "",
            f"**Topic:** {VALIDATION_TOPICS[cid.removeprefix('scale1-validation-')]} · "
            f"**Writing task:** {FAMILY_LABELS[selected['before', 0]['case']['family']]}", "",
            "<details>", "<summary>Prompt and source context</summary>", "",
            quote(selected["before", 0]["request"]), "", "</details>", "",
        ])
        for sample in range(2):
            before, after = (selected[phase, sample] for phase in ("before", "after"))
            parts.extend([
                "<details>",
                f"<summary>Response {sample + 1} · reward {before['score']['reward']:.3f} → {after['score']['reward']:.3f}</summary>", "",
            ])
            for row, label in ((before, "Untrained · saved baseline"), (after, "Trained · 24 RL updates")):
                parts.extend([
                    f"### {label}", "",
                    f"Jev reward: **{row['score']['reward']:.3f}** · {row['score']['words']} words", "",
                    quote(row["draft"]), "",
                ])
            parts.extend(["</details>", ""])
        parts.extend(["[Back to tasks](#tasks)", ""])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True, help="Private receipt path under runs/")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.receipt.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Provenance receipt must stay private under runs/")
    for path in (args.output, args.receipt):
        if any(path.resolve().is_relative_to(p.resolve()) for p in (args.baseline, args.trained)):
            raise ValueError("Do not write into an original run")
    if args.write and args.receipt.exists():
        raise FileExistsError("Never overwrite an existing receipt")
    rows, summary = prepare(args.baseline, args.trained)
    report, gallery = render(rows, summary)
    receipt = {"baseline_records_sha256": digest(args.baseline / "records.json"),
               "trained_records_sha256": digest(args.trained / "records.json"),
               "baseline_reused": True, "rescored": False, **summary["rubric"], "summary": summary}
    if args.write:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.touch(exist_ok=False)  # Exclusive claim before any public file changes.
        save(args.receipt, receipt)
        (args.output / "examples").mkdir(parents=True, exist_ok=True)
        (args.output / "RESULTS.md").write_text(report)
        (args.output / "examples/comparison.html").write_text(gallery)
        (args.output / "examples/comparison.md").write_text(render_markdown(rows, summary))
    print(json.dumps({"written": args.write, "baseline_reused": True,
                      "before": summary["before"]["reward_mean"], "after": summary["after"]["reward_mean"]}))


if __name__ == "__main__":
    main()
