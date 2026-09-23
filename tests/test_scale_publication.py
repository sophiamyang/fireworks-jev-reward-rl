import copy
import html
import importlib.util
import json
import re
import statistics
from pathlib import Path

import pytest

from fw_jev.config import load_rows, prompt
from fw_jev.reward import calculate, make_judge
from fw_jev.storage import save

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_all_public_outputs_reproduce_the_headline_without_private_runs():
    body = (ROOT / "docs/examples/comparison.html").read_text()
    entries = re.findall(r"<article data-phase='(before|after)' data-reward='([^']+)' data-sample='([01])'>", body)
    assert len(entries) == 96 and body.count("<section ") == 24
    for phase, expected in (("before", 0.5828301346801347), ("after", 0.7590947048611111)):
        values = [float(reward) for p, reward, _ in entries if p == phase]
        assert len(values) == 48
        assert statistics.mean(values) == pytest.approx(expected)
    assert "saved untrained baseline" in body and "24 RL updates" in body
    assert "I lentish a ladder" in body
    assert "licensed password manager" in body
    for forbidden in ("<script", "/Users/", "logprobs", "probabilities", "api_key", "sampler_path"):
        assert forbidden not in body.lower()
    assert not list((ROOT / "docs").rglob("*.json"))


def test_headline_values_are_consistent_in_current_docs():
    for name in ("README.md", "docs/RESULTS.md"):
        body = (ROOT / name).read_text()
        assert "0.583" in body and "0.759" in body
        assert "saved untrained baseline" in body
    assert "not PPO-clipped" in (ROOT / "docs/TUTORIAL.md").read_text()


def test_readme_is_a_short_overview_with_links_to_details():
    readme = (ROOT / "README.md").read_text()
    assert len(readme.split()) < 900
    assert "toy task" in readme and "not a\nproduction-ready writing model" in readme
    assert "the trained adapter is not published" in readme
    assert not (ROOT / "docs/BLOG.md").exists()
    for target in ("docs/TUTORIAL.md", "docs/cost-and-speed.md", "docs/examples/comparison.md",
                   "experiments/scale-v1/README.md", "src/fw_jev/reward.py"):
        assert f"]({target})" in readme


def test_readme_example_quotes_preserve_the_published_responses():
    readme = (ROOT / "README.md").read_text()
    example = readme.split("### One example: cleaning up a release note\n", 1)[1].split(
        "### Cost and speed", 1
    )[0]
    comparison = (ROOT / "docs/examples/comparison.md").read_text()
    task = comparison.split("## changelog-edit\n", 1)[1].split("## festival-notes", 1)[0]
    quotes = re.findall(r"^>[^\n]*(?:\n>[^\n]*)*", example, re.M)
    assert len(quotes) == 2
    assert all(quote in task for quote in quotes)
    for score, words in (("0.544", 157), ("0.952", 56)):
        assert f"Jev reward {score}" in example
        assert f"{words} words" in example
        assert f"Jev reward: **{score}** · {words} words" in task
    assert "opening excerpt" in example and "full response" in example
    assert "selected" in example and "unchanged" in example


def test_topic_labels_cover_the_frozen_validation_set_and_task_mix_matches_dataset_guide():
    helper = module("publish_comparison")
    cases = load_rows(ROOT / "experiments/scale-v1/validation.jsonl")
    assert set(helper.VALIDATION_TOPICS) == {c["id"].removeprefix("scale1-validation-") for c in cases}
    assert all(c["family"] in helper.FAMILY_LABELS for c in cases)
    body = (ROOT / "docs/examples/comparison.md").read_text()
    assert helper.TRAINING_GOAL in body and helper.TRAINING_TOPICS in body
    assert body.startswith("# No AI Slop:")
    assert "validation**, not training targets" in body
    for c in cases:
        assert helper.VALIDATION_TOPICS[c["id"].removeprefix("scale1-validation-")] in body
    manifest = json.loads((ROOT / "experiments/scale-v1/manifest.json").read_text())
    assert (ROOT / "README.md").read_text().startswith("# No AI Slop:")
    dataset_guide = (ROOT / "experiments/scale-v1/README.md").read_text()
    for title, families in (
        ("Source-to-social posts and threads", ("social",)),
        ("Summaries and explainers", ("summary",)),
        ("Rewrites and cuts", ("rewrite", "cut")),
        ("Contextual replies", ("reply",)),
        ("Notes-to-post writing", ("notes",)),
        ("Direct writing requests, including creative pieces", ("direct",)),
    ):
        train, validation = (sum(manifest["families"][split].get(f, 0) for f in families)
                             for split in ("train", "validation"))
        assert f"| {title} | {train} | {validation} |" in dataset_guide


@pytest.fixture
def comparison_sources(tmp_path):
    helper = module("publish_comparison")
    config = json.loads((ROOT / "experiments/raw-base-v1/config.json").read_text())
    cases = load_rows(ROOT / config["validation_file"])
    judge = make_judge(mock=True)
    baseline, trained = tmp_path / "baseline", tmp_path / "trained"
    for folder, phase, sampler in ((baseline, "before", config["model"]), (trained, "after", "private/test-final")):
        save(folder / "config.json", config)
        save(folder / "status.json", {"state": "complete", "mock": False, "optimizer_updates": 24})
        rows = []
        for case in cases:
            for i in range(2):
                draft = f"MOCK {i} — synthetic unit fixture, not a model output.  \n<script>test</script>"
                raw = judge.score(prompt(case), draft, grounded=case["grounding_applicable"])
                rows.append({"phase": phase, "case": case, "request": prompt(case), "sample": i,
                             "draft": draft, "judge": raw, "score": calculate(raw, draft),
                             "sampler": sampler, "truncated": False, "malformed": False, "tokens": [1]})
        save(folder / "records.json", rows)
    save(trained / "checkpoints.json", [{"sampler_path": "private/test-final"}])
    return helper, baseline, trained


def test_export_reads_but_does_not_mutate_sources(comparison_sources):
    helper, baseline, trained = comparison_sources
    original = [(p / "records.json").read_bytes() for p in (baseline, trained)]
    rows, summary = helper.prepare(baseline, trained)
    snapshot = copy.deepcopy(rows)
    report, gallery = helper.render(rows, summary)
    assert rows == snapshot
    assert original == [(p / "records.json").read_bytes() for p in (baseline, trained)]
    assert "not generated or rescored together" in report
    assert gallery.count("<article ") == 96 and gallery.count("<section ") == 24
    assert "private/test-final" not in gallery and "<script>" not in gallery
    assert rows[0]["draft"] in html.unescape(gallery)


def test_markdown_contains_every_exact_draft_and_no_active_html(comparison_sources):
    helper, baseline, trained = comparison_sources
    rows, summary = helper.prepare(baseline, trained)
    snapshot = copy.deepcopy(rows)
    body = helper.render_markdown(rows, summary)
    assert rows == snapshot
    assert body.count("### Untrained · saved baseline") == 48
    assert body.count("### Trained · 24 RL updates") == 48
    assert body.count("<summary>Prompt and source context</summary>") == 24
    assert body.count("<details>") == body.count("</details>") == 72
    assert "saved untrained baseline" in body and "not generated or rescored together" in body
    assert "private/test-final" not in body and "<script>" not in body
    for row in rows:
        assert helper.quote(row["request"]) in body
        assert helper.quote(row["draft"]) in body
    assert not any(line.endswith((" ", "\t")) for line in body.splitlines())


def test_public_markdown_matches_html_responses_and_scores():
    body = (ROOT / "docs/examples/comparison.md").read_text()
    gallery = (ROOT / "docs/examples/comparison.html").read_text()
    helper = module("publish_comparison")
    articles = re.findall(r"<article data-phase='(before|after)' data-reward='([^']+)' data-sample='([01])'>(.*?)</article>", gallery, re.S)
    assert len(articles) == 96
    for _, reward, _, content in articles:
        draft = html.unescape(re.search(r"<pre>(.*?)</pre>", content, re.S)[1])
        assert helper.quote(draft) in body
        assert f"Jev reward: **{float(reward):.3f}**" in body
    assert body.count("### Untrained · saved baseline") == 48
    assert body.count("### Trained · 24 RL updates") == 48
    assert "0.583 → 0.759" in body


@pytest.mark.parametrize(
    "defect", ["reward", "missing", "sampler", "context", "temperature", "restored", "extra-setting"]
)
def test_export_rejects_incompatible_or_edited_evidence(comparison_sources, defect):
    helper, baseline, trained = comparison_sources
    rows = json.loads((trained / "records.json").read_text())
    if defect == "reward":
        rows[0]["score"]["reward"] += 0.1
    elif defect == "missing":
        rows.pop()
    elif defect == "sampler":
        rows[0]["sampler"] = "other-model"
    elif defect == "context":
        rows[0]["request"] = "Different prompt"
    elif defect == "temperature":
        config = json.loads((trained / "config.json").read_text())
        save(trained / "config.json", dict(config, temperature=0.5))
    elif defect == "restored":
        status = json.loads((trained / "status.json").read_text())
        save(trained / "status.json", dict(status, parent_updates=6))
    else:
        config = json.loads((trained / "config.json").read_text())
        save(trained / "config.json", dict(config, restore_state="accounts/example/state-6"))
    save(trained / "records.json", rows)
    with pytest.raises(ValueError):
        helper.prepare(baseline, trained)


def test_html_export_preserves_original_whitespace_without_diff_warnings():
    helper = module("publish_comparison")
    original = "<script>example</script>  \nsecond\t \nlast  "
    escaped = helper.escape(original)
    assert html.unescape(escaped) == original
    assert "<script>" not in escaped
    assert not any(line.endswith((" ", "\t")) for line in escaped.splitlines())


def test_markdown_keeps_code_spans_literal_and_blocks_setext_headings():
    helper = module("publish_comparison")
    quoted = helper.quote("Use `a && b` or `x<y>` & <b>no</b>\nTitle\n---\n\n---\nTop\n===")
    assert quoted.splitlines() == [
        "> Use `a && b` or `x<y>` &amp; &lt;b&gt;no&lt;/b&gt;",
        "> Title", "> \\---", ">", "> ---", "> Top", "> \\===",
    ]
    # An unpaired backtick could pair across lines, so its paragraph keeps the full escape.
    assert helper.quote("stray ` <b>\nnext ` <i>") == "> stray ` &lt;b&gt;\n> next ` &lt;i&gt;"
    assert helper.quote("```\ncode\n---\n```") == "> ```\n> code\n> ---\n> ```"


def test_committed_results_page_matches_the_renderer_text(comparison_sources):
    helper, baseline, trained = comparison_sources
    report, gallery = helper.render(*helper.prepare(baseline, trained))
    committed = (ROOT / "docs/RESULTS.md").read_text()
    method = report.split("## Method and limits\n\n", 1)[1].split("\n\n")[0]
    assert method in committed and "question text" not in committed
    assert "every saved score replays with the reward calculation" in method
    for text in (helper.METHOD, helper.TRAINING_GOAL):
        assert text in committed
        assert text in (ROOT / "docs/examples/comparison.md").read_text()
    assert f"<small>{helper.METHOD} Saved baseline scores" in gallery
    assert f"<small>{helper.METHOD} Saved baseline scores" in (ROOT / "docs/examples/comparison.html").read_text()
    body = helper.render_markdown(*helper.prepare(baseline, trained))
    intro = next(line for line in body.splitlines() if line.startswith("Untrained: 48 saved responses"))
    assert intro in (ROOT / "docs/examples/comparison.md").read_text().splitlines()


def test_prompt_counts_use_the_displayed_precision():
    helper = module("publish_comparison")
    paired = [{"before": {"reward_mean": b}, "after": {"reward_mean": a}}
              for b, a in ((0.53015, 0.530275), (0.4, 0.6), (0.9, 0.6))]
    assert helper.directions({"paired": paired}, "prompt") == (
        "1/3 prompt means increased, 1 unchanged at three decimals, 1 decreased."
    )
    assert "20/24 prompt means increased, 1 unchanged at three decimals, 3 decreased." in (
        ROOT / "docs/RESULTS.md").read_text()


def test_rubric_text_is_compared_when_both_manifests_record_it(comparison_sources):
    helper, baseline, trained = comparison_sources
    assert helper.prepare(baseline, trained)[1]["rubric"]["rubric_text_verified"] is False
    manifest = {"writer_protocol": "p", "questions": {"general": ["q"]}}
    for folder in (baseline, trained):
        save(folder / "manifest.json", manifest)
    assert helper.prepare(baseline, trained)[1]["rubric"]["rubric_text_verified"] is True
    save(trained / "manifest.json", dict(manifest, questions={"general": ["other"]}))
    with pytest.raises(ValueError, match="judge questions"):
        helper.prepare(baseline, trained)


def test_receipt_never_overwrites_records_or_an_existing_receipt(comparison_sources, tmp_path, monkeypatch):
    helper, baseline, trained = comparison_sources
    (tmp_path / "runs").mkdir()
    baseline, trained = baseline.rename(tmp_path / "runs/baseline"), trained.rename(tmp_path / "runs/trained")
    (tmp_path / "experiments").symlink_to(ROOT / "experiments")
    monkeypatch.setattr(helper, "ROOT", tmp_path)
    records = (trained / "records.json").read_bytes()

    def main(receipt):
        monkeypatch.setattr("sys.argv", ["publish", "--baseline", str(baseline), "--trained", str(trained),
                                         "--output", str(tmp_path / "out"), "--receipt", str(receipt), "--write"])
        helper.main()

    with pytest.raises(ValueError, match="original run"):
        main(trained / "records.json")
    assert (trained / "records.json").read_bytes() == records and not (tmp_path / "out").exists()
    receipt = tmp_path / "runs/receipt.json"
    main(receipt)
    assert json.loads(receipt.read_text())["rubric_text_verified"] is False
    (tmp_path / "out/RESULTS.md").unlink()
    with pytest.raises(FileExistsError):
        main(receipt)
    assert not (tmp_path / "out/RESULTS.md").exists()


def test_blank_smoke_review_never_approves_or_overwrites(tmp_path):
    helper = module("review_raw_smoke")
    rows = [{"phase": "smoke", "case": {"id": f"case-{i // 4}"}, "sample": i % 4} for i in range(24)]
    save(tmp_path / "records.json", rows)
    path = helper.init(tmp_path)
    body = json.loads(path.read_text())
    assert body["approved"] is False and body["accepted_rankings"] == []
    assert all(s["answers_task"] is None for s in body["samples"])
    with pytest.raises(FileExistsError):
        helper.init(tmp_path)
