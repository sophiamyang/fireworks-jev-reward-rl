"""Check (or re-materialize) the frozen 96/24 prompt set. TASK INPUTS only, never outputs.

Training rows come from three checked-in sources: 24 seed rows stored only in
train.jsonl, 60 rows in sources.json and 12 rows in authored.py; validation is
authored.py. Nothing outside this folder is read.
Check: python experiments/scale-v1/build_dataset.py --check
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from authored import TRAIN, VALIDATION

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cache = HERE / "sources.json"
    imported = [p["case"] for p in json.loads(cache.read_text())]
    frozen = json.loads((HERE / "manifest.json").read_text())["files"]
    if digest(HERE / "train.jsonl") != frozen["train.jsonl"]:
        raise ValueError("Committed train.jsonl differs from its manifest hash")
    # The seed rows live only in the committed file; keep them in their frozen order.
    known = {r["id"] for r in imported + TRAIN}
    seed = [r for r in map(json.loads, (HERE / "train.jsonl").read_text().splitlines()) if r["id"] not in known]
    train = seed + imported + TRAIN
    validation = VALIDATION
    if (len(seed), len(imported), len(TRAIN), len(validation)) != (24, 60, 12, 24):
        raise ValueError("Dataset must be 24 seed + 60 sources.json + 12 authored training, 24 validation")
    # Deterministic round-robin families distributes contexts/lengths over all 24 updates.
    buckets = {}
    for row in train:
        buckets.setdefault(row["family"], []).append(row)
    ordered = []
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:
                ordered.append(bucket.pop(0))
    train = ordered
    for field in ("id", "scenario_id", "source_id"):
        values = [r[field] for r in train + validation]
        if len(values) != len(set(values)):
            raise ValueError("Duplicate input identity: " + field)
    texts = [" ".join((r["request"] + "\n\n" + r["context"]).lower().split()) for r in train + validation]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate prompt")
    payloads = {
        "train.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train),
        "validation.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in validation),
    }
    manifest = {
        "version": "scale-context-v1",
        "train_count": 96,
        "validation_count": 24,
        "smoke_ids": [r["id"] for r in train[:6]],
        "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in payloads.items()},
        "builder_sha256": digest(__file__),
        "authored_sha256": digest(HERE / "authored.py"),
        "source_cache_sha256": digest(cache),
        "families": {
            "train": dict(Counter(r["family"] for r in train)),
            "validation": dict(Counter(r["family"] for r in validation)),
        },
        "rules": [
            "Task inputs only, not SFT or preference targets.",
            "All source material is synthetic.",
            "Do not adapt training or rewards to the validation outputs.",
        ],
    }
    payloads["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    for name, body in payloads.items():
        target = HERE / name
        if args.check:
            if target.read_text() != body:
                raise ValueError("Frozen expansion differs: " + name)
        else:
            target.write_text(body)
    print(
        json.dumps(
            {
                "train": len(train),
                "validation": len(validation),
                "families": manifest["families"],
                "check": args.check,
            }
        )
    )


if __name__ == "__main__":
    main()
