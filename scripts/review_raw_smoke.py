"""Create a blank review or check a completed one; never infer approval."""
import argparse
import json
from pathlib import Path

from fw_jev.config import load
from fw_jev.raw_base import check_admission
from fw_jev.storage import digest

ROOT = Path(__file__).resolve().parents[1]

# Tell a fixable review-form problem apart from a smoke that shouldn't train.
FORM = ("full-draft review", "review required", "Every smoke sample", "labels", "scorer limitations",
        "reviewed mixed-objective rankings", "ranking evidence", "nonanswers", "defect",
        "Source-support ranking")
DISAGREE = ("frozen reward margin",)
SIGNAL = ("numeric signal", "technical failures", "fixed-draft checks")


def explain(message):
    if any(key in message for key in FORM):
        return "Review form: fix smoke-review.json and run the check again."
    if any(key in message for key in DISAGREE):
        return ("Jev disagrees with a ranking by less than 0.05. Pick another pair you and Jev agree on; "
                "if there aren't three, stop: the reward signal isn't ready.")
    if any(key in message for key in SIGNAL):
        return "Signal: this smoke shouldn't train. Stop; don't resample until it passes."
    return "Setup: run a new smoke in a new folder with the current code and config."


def init(folder):
    folder = Path(folder)
    rows = json.loads((folder / "records.json").read_text())
    policy = json.loads((ROOT / "experiments/raw-base-v1/admission-v2.json").read_text())
    if len(rows) != 24 or any(row["phase"] != "smoke" for row in rows):
        raise ValueError("Expected 24 zero-update smoke drafts")
    body = {
        "policy": policy["version"],
        "records_sha256": digest(folder / "records.json"),
        "approved": False,
        "reviewer": "",
        "samples": [
            {"case_id": row["case"]["id"], "sample": row["sample"],
             "answers_task": None, "source_faithful": None,
             "key_content_present": None, "ambiguous_request": None, "note": ""}
            for row in rows
        ],
        "accepted_rankings": [],
        "scorer_limitations": [],
    }
    path = folder / "smoke-review.json"
    with path.open("x") as output:
        output.write(json.dumps(body, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "check"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        print(init(args.folder))
    else:
        config, train, _, root = load(ROOT / "experiments/raw-base-v1/config.json")
        try:
            print(json.dumps(check_admission(args.folder, root, config, train, require_review=True), indent=2))
        except ValueError as exc:
            raise SystemExit(f"Check failed: {exc}\n{explain(str(exc))}") from None


if __name__ == "__main__":
    main()
