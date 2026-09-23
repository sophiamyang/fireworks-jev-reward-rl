"""Offline completeness/math audit. No APIs, model changes, or reward tuning."""

import argparse
import json
from pathlib import Path

from fw_jev.config import load_rows, prompt, training_batches
from fw_jev.reward import VARIANT, VERSION, calculate
from fw_jev.rl_math import advantages, alignment, datum
from fw_jev.storage import digest, save
from fw_jev.writing import messages


def read(path):
    return json.loads(Path(path).read_text())


def audit(folder, *, require_source_snapshot=True):
    folder = Path(folder)
    c, status = read(folder / "config.json"), read(folder / "status.json")
    rows, inputs = read(folder / "records.json"), read(folder / "inputs.json")
    issues, checked = [], {"rewards": 0, "references": 0, "groups": 0, "tensors": 0, "updates": 0}

    def require(condition, label):
        if not condition:
            issues.append(label)

    require(status["state"] == "complete", "Run did not complete")
    require(c["steps"] > 0, "Not a training run")
    require(c.get("reward_variant") == VARIANT, "Reward variant differs")
    require(read(folder / "manifest.json")["reward_version"] == VERSION, "Reward version differs")
    allowed = {x["id"]: x for x in inputs["train"] + inputs["validation"]}
    train_ids, val_ids = ({x["id"] for x in inputs[k]} for k in ("train", "validation"))
    schedule = training_batches(c, inputs["train"])
    require(not train_ids.intersection(val_ids), "Training/validation IDs overlap")
    require(
        len(rows)
        == sum(map(len, schedule)) * c["group_size"] + 2 * len(inputs["validation"]) * c["eval_samples"],
        "Incomplete generation count",
    )
    keys = [(r["phase"], r["case"]["id"], r["sample"]) for r in rows]
    require(len(set(keys)) == len(keys), "Duplicate generation records")
    expected_keys = {
        (phase, r["id"], sample)
        for phase in ("before", "after")
        for r in inputs["validation"]
        for sample in range(c["eval_samples"])
    } | {
        (f"train-{index:02}", r["id"], sample)
        for index, batch in enumerate(schedule, 1)
        for r in batch
        for sample in range(c["group_size"])
    }
    require(set(keys) == expected_keys, "Phase/case/sample coverage differs")
    for r in rows:
        label = f"{r['phase']}/{r['case']['id']}/{r['sample']}"
        require(r["case"] == allowed.get(r["case"]["id"]), label + ": input differs")
        require(r["request"] == prompt(r["case"]), label + ": request differs")
        require(r["writer_messages"] == messages(r["request"]), label + ": message protocol differs")
        require(
            r["case"]["id"] in (train_ids if r["phase"].startswith("train-") else val_ids),
            label + ": split differs",
        )
        score = calculate(
            r["judge"], r["draft"], truncated=r["truncated"], malformed=r["malformed"], weights=c["weights"]
        )
        require(score == r["score"], label + ": reward replay differs")
        checked["rewards"] += 1
    for index in range(1, c["steps"] + 1):
        step = folder / f"step-{index:02}"
        batch = schedule[index - 1]
        expected_arrays, lengths, expected_gates = [], [], []
        for case in batch:
            group = sorted(
                [r for r in rows if r["phase"] == f"train-{index:02}" and r["case"]["id"] == case["id"]],
                key=lambda r: r["sample"],
            )
            require(
                [r["sample"] for r in group] == list(range(c["group_size"])),
                case["id"] + ": incomplete group",
            )
            if len(group) < 2:
                continue
            gate = advantages(
                [r["score"]["reward"] for r in group], c["min_group_spread"], c["advantage_std_floor"]
            )
            expected_gates.append({"case_id": case["id"], **gate})
            checked["groups"] += 1
            for r, adv in zip(group, gate["advantages"] or [0.0] * len(group), strict=True):
                ref_path = folder / "references" / f"{r['phase']}-{case['id']}-{r['sample']}.json"
                if not ref_path.exists():
                    issues.append(str(ref_path) + ": missing reference")
                    continue
                ref = read(ref_path)
                require(
                    ref["fixed_sampler"] == status.get("reference_sampler", status["initial_sampler"]),
                    str(ref_path) + ": reference differs",
                )
                array, measured = datum(
                    r["prompt_tokens"], r["tokens"], r["logprobs"], ref["logprobs"], adv, c["kl_beta"]
                )
                require(all(ref[k] == v for k, v in measured.items()), str(ref_path) + ": KL differs")
                checked["references"] += 1
                if gate["eligible"]:
                    expected_arrays.append(array)
                    lengths.append(len(r["tokens"]))
        if not (step / "group-signal.json").exists():
            issues.append(str(step) + ": missing group results")
            continue
        require(read(step / "group-signal.json") == expected_gates, str(step) + ": group advantages differ")
        if not expected_arrays:
            require(
                not (step / "optimizer-attempt.json").exists(), str(step) + ": update without eligible groups"
            )
            continue
        if not (step / "training-arrays.json").exists():
            issues.append(str(step) + ": missing training tensors")
            continue
        arrays_match = read(step / "training-arrays.json") == expected_arrays
        require(arrays_match, str(step) + ": training tensors differ")
        if arrays_match:
            checked["tensors"] += len(expected_arrays)
        for operation in ("forward-backward", "optimizer", "checkpoint", "sampler"):
            require(
                (step / f"{operation}-attempt.json").exists(), str(step) + f": missing {operation} attempt"
            )
            require(
                (step / f"{operation}-result.json").exists(),
                str(step) + f": missing/uncertain {operation} result",
            )
        if arrays_match and (step / "forward-backward-result.json").exists():
            backward_result = read(step / "forward-backward-result.json")
            require(
                "loss_fn_outputs" in backward_result,
                str(step) + ": missing backward outputs; plain --mock does not provide SDK receipts",
            )
            if "loss_fn_outputs" in backward_result:
                outputs = [
                    {"logprobs": p["logprobs"]["data"] if isinstance(p["logprobs"], dict) else p["logprobs"]}
                    for p in backward_result["loss_fn_outputs"]
                ]
                try:
                    measured = alignment(outputs, expected_arrays, lengths)
                except ValueError as exc:  # e.g. a run halted by the alignment gate
                    issues.append(f"{step}: {exc}")
                else:
                    require(
                        measured == read(step / "alignment.json")["sampled_kl"],
                        str(step) + ": alignment differs",
                    )
        if (step / "optimizer-result.json").exists():
            checked["updates"] += 1
    checkpoints = read(folder / "checkpoints.json") if (folder / "checkpoints.json").exists() else []
    by_step = {x["step"]: x for x in checkpoints}
    require(len(by_step) == len(checkpoints), "Duplicate checkpoint step")
    current = status.get("initial_sampler")
    require(current is not None or not rows, "Run halted before a sampler was created")
    require(all(r["sampler"] == current for r in rows if r["phase"] == "before"), "Before sampler differs")
    for index in range(1, c["steps"] + 1):
        require(
            all(r["sampler"] == current for r in rows if r["phase"] == f"train-{index:02}"),
            "Training sampler differs: " + str(index),
        )
        if index in by_step:
            checkpoint = by_step[index]
            for operation, key in (("checkpoint", "state_path"), ("sampler", "sampler_path")):
                path = folder / f"step-{index:02}" / f"{operation}-result.json"
                if path.exists():
                    require(
                        read(path).get("path") == checkpoint[key],
                        "Checkpoint receipt path differs: " + str(index),
                    )
            current = checkpoint["sampler_path"]
    require(all(r["sampler"] == current for r in rows if r["phase"] == "after"), "After sampler differs")
    if require_source_snapshot:
        require((folder / "source").is_dir(), "Missing frozen source snapshot")
    if (folder / "source").is_dir():
        for name, sha in read(folder / "manifest.json")["sha256"].items():
            path = folder / "source" / name
            require(path.is_file() and digest(path) == sha, "Source snapshot differs: " + name)
        for split, file_key, limit_key in (
            ("train", "train_file", "train_limit"),
            ("validation", "validation_file", "validation_limit"),
        ):
            path = folder / "source" / c[file_key]
            try:
                expected = load_rows(path)[: c[limit_key]] if split == "train" or c["steps"] else []
            except (OSError, ValueError) as exc:
                issues.append(f"Source dataset unreadable: {c[file_key]}: {exc}")
                continue
            require(inputs[split] == expected, "Inputs differ from source dataset: " + split)
    require(
        len(checkpoints) == checked["updates"] == status["optimizer_updates"],
        "Checkpoint/update totals differ",
    )
    require(
        len(list(folder.glob("step-*/optimizer-attempt.json"))) == checked["updates"],
        "Unresolved optimizer attempt",
    )
    return {
        "passed": not issues,
        "mock": status["mock"],
        "checked": checked,
        "issues": issues,
        "records_sha256": digest(folder / "records.json"),
        "manifest_sha256": digest(folder / "manifest.json"),
        "source_snapshot_required": require_source_snapshot,
        "source_snapshot_available": (folder / "source").is_dir(),
        "warnings": []
        if (folder / "source").is_dir()
        else [
            "No full source snapshot; saved inputs were not compared with the frozen dataset files."
        ],
        "scope": "Offline local journal, reward and tensor assembly. Does not verify Fireworks gradients, judge truth or writing quality.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.folder)
    if args.output:
        save(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
