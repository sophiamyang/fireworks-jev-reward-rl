"""Create a blank review or check a completed one; never infer approval."""
import argparse
import json
from pathlib import Path

from fw_jev.config import load
from fw_jev.raw_base import check_admission
from fw_jev.storage import digest

ROOT = Path(__file__).resolve().parents[1]


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
        print(json.dumps(check_admission(args.folder, root, config, train), indent=2))


if __name__ == "__main__":
    main()
