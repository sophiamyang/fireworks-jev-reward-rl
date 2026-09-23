import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fw_jev.config import load
from fw_jev.runner import fingerprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("args", [["--config", "experiments/raw-base-v1/config.json"], []])
def test_cli_plan_default_and_explicit_match_published_recipe(args):
    result = subprocess.run(
        [sys.executable, "-m", "fw_jev.cli", "plan", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["experiment"] == "raw-simple-v1"
    assert plan["limits"] == {"generations": 864, "jev": 896, "references": 768, "optimizer": 24}
    assert plan["reward_variant"] == "style-quality-support-v1"
    assert plan["limits"]["optimizer"] == 24
    assert plan["training_prompts"] == 96
    assert plan["validation_prompts_used"] == 24
    assert plan["system_prompt"] is None


def markdown_files():
    files = [ROOT / "README.md"]
    for directory in ("docs", "scripts", "src", "experiments", "model", "tests"):
        files.extend((ROOT / directory).rglob("*.md"))
    return files


def anchors(path):
    # GitHub-style heading slugs plus explicit HTML ids.
    text = path.read_text()
    found = set(re.findall(r"""id=['"]([^'"]+)['"]""", text))
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        found.add(slug)
    return found


def test_documentation_links_and_anchors_resolve_without_private_run_folders():
    for path in markdown_files():
        for target in re.findall(r"\]\(([^)\s]+)\)", path.read_text()):
            assert "archive/" not in target, (path, target)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            local, _, anchor = target.partition("#")
            assert not local.startswith("/Users/"), (path, local)
            destination = (path.parent / local) if local else path
            assert destination.exists(), (path, local)
            if anchor and destination.suffix == ".md":
                assert anchor in anchors(destination), (path, target)


def test_main_folders_only_expose_the_tutorial_recipe():
    assert not (ROOT / "configs").exists() and not (ROOT / "data").exists()
    assert {p.name for p in (ROOT / "experiments/raw-base-v1").glob("*.json")} == {
        "config.json",
        "smoke.json",
        "admission-v2.json",
        "contract.json",
    }
    assert {p.name for p in (ROOT / "src/fw_jev").glob("reward*.py")} == {"reward.py"}
    assert {p.name for p in (ROOT / "scripts").glob("*.py")} == {
        "audit_run.py",
        "review_raw_smoke.py",
        "publish_comparison.py",
    }
    assert {p.name for p in (ROOT / "docs").glob("*.md")} == {
        "README.md",
        "TUTORIAL.md",
        "LESSONS.md",
        "RESULTS.md",
        "cost-and-speed.md",
    }


def test_current_checkout_has_no_archive_or_abandoned_recipes():
    for directory in ("docs", "scripts", "experiments"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                assert "archive" not in path.relative_to(ROOT).parts, path
    assert not (ROOT / "experiments/scale-v1/config.json").exists()
    assert not (ROOT / "experiments/scale-v1/contract.json").exists()


def test_local_secrets_and_run_artifacts_are_ignored_and_not_tracked():
    private_paths = [".env", ".env.production", "runs/private/records.json", "wandb/run/log"]
    ignored = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(private_paths) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(ignored.stdout.splitlines()) == set(private_paths)
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    for name in filter(None, tracked):
        path = Path(name)
        assert not ({"runs", ".venv", "wandb", "__pycache__"} & set(path.parts)), name
        assert not path.name.startswith(".env") or name == ".env.example", name
        assert path.suffix not in {".pem", ".key", ".pyc"}, name


@pytest.mark.parametrize(
    "path",
    [
        "experiments/raw-base-v1/config.json",
        "experiments/raw-base-v1/smoke.json",
    ],
)
def test_recipe_paths_are_repository_relative_even_outside_cwd(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, train, validation, root = load(ROOT / path)
    assert root == ROOT
    assert train and validation


def test_run_snapshot_includes_current_executable_inputs():
    hashes = fingerprint(ROOT)
    required = {
        str(path.relative_to(ROOT))
        for directory, pattern in (
            ("src/fw_jev", "*.py"),
            ("scripts", "*.py"),
            ("experiments/scale-v1", "*.json*"),
            ("experiments/scale-v1", "*.py"),
            ("experiments/raw-base-v1", "*.json"),
        )
        for path in (ROOT / directory).rglob(pattern)
    }
    assert required <= hashes.keys()
    assert all((ROOT / path).is_file() for path in hashes)
