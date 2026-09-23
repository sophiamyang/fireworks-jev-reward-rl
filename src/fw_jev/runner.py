"""One bounded pass, a fixed reference, local artifacts first, optional W&B."""

import importlib.metadata
import os
import shutil
import statistics
import subprocess
import time

from . import calibration, report
from .config import load, plan, prompt, training_batches
from .reward import VERSION, calculate, make_judge, questions
from .rl_math import advantages, datum
from .smoke import TECHNICAL, diagnose
from .storage import Budget, digest, fresh, safe_error, save
from .writing import PROTOCOL, messages


def fingerprint(root):
    paths = [
        *sorted((root / "src/fw_jev").glob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "scripts").rglob("*.py")),
        *sorted((root / "experiments/scale-v1").glob("*.json*")),
        *sorted((root / "experiments/scale-v1").glob("*.py")),
        *sorted((root / "experiments/raw-base-v1").glob("*.json")),
        # Notebooks only shell out to the CLI; an autosave must not stale the smoke.
    ]
    return {str(p.relative_to(root)): digest(p) for p in paths}


def run(config_path, output, *, mock=False, wandb=False, smoke_from=None):
    c, train, validation, root = load(config_path)
    admission = None
    if c["steps"] and not mock:
        if smoke_from is None:
            raise ValueError("Run and inspect the zero-update smoke first; pass --smoke-from its folder")
        from .raw_base import check_admission

        admission = check_admission(smoke_from, root, c, train)
    if not mock:
        for key in ("FIREWORKS_API_KEY", "TYPESAFE_API_KEY"):
            if not os.environ.get(key) or os.environ[key] == "replace_locally":
                raise ValueError(f"Set {key} locally")
    if wandb and (mock or not os.environ.get("WANDB_API_KEY")):
        raise ValueError("--wandb requires a live run and a locally configured WANDB_API_KEY")
    out = fresh(output)
    started = time.perf_counter()
    status = {
        "state": "preflight",
        "mock": mock,
        "optimizer_updates": 0,
        "scored_drafts": 0,
        "generated_drafts": 0,
        "checkpoint_bundles": 0,
    }
    records, steps, checkpoints = [], [], []
    judge = backend = tracking = None
    try:
        save(out / "status.json", status)
        save(out / "config.json", c)
        if admission is not None:
            save(out / "admission.json", admission)
        save(
            out / "inputs.json",
            {
                "writer_protocol": PROTOCOL,
                "system": None,
                "train": train,
                "validation": validation if c["steps"] else [],
            },
        )
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        source_hashes = fingerprint(root)
        for name in source_hashes:
            destination = out / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
            if digest(destination) != source_hashes[name]:
                raise ValueError("Source changed during snapshot")
        save(
            out / "manifest.json",
            {
                "sha256": source_hashes,
                "git_commit": revision.stdout.strip() or None,
                "reward_version": VERSION,
                "writer_protocol": PROTOCOL,
                "questions": {"general": questions(), "source_grounded": questions(grounded=True)},
                "packages": {
                    k: importlib.metadata.version(k)
                    for k in ("fireworks-ai", "tinker", "transformers", "httpx", "wandb")
                },
                "reproducibility": "Frozen inputs and artifacts; stochastic service outputs are not bitwise reproducible.",
            },
        )
        budget = Budget(out, plan(c, train, validation)["limits"])
        save(out / "records.json", records)
        if wandb:
            import wandb as wb

            tracking = wb.init(
                entity=os.environ.get("WANDB_ENTITY") or None,
                project=os.environ.get("WANDB_PROJECT", "fireworks-rl-jev-scorer"),
                config=c,
                dir=str(out),
                name=out.name,
            )
            save(out / "wandb.json", {"url": tracking.url, "id": tracking.id})
        if mock:
            from .mock import Backend
        else:
            from .raw_base import VerifiedRawBase as Backend

        judge = make_judge(os.environ.get("TYPESAFE_API_KEY"), mock=mock)
        fixed = calibration.run(judge, out / "scorer-check", c["weights"], budget)
        if not fixed["passed"]:
            raise ValueError("Fixed-draft scorer check failed. No Fireworks session or optimizer created.")
        backend = Backend(c, out, train + (validation if c["steps"] else []))
        status.update(
            fireworks_url=backend.url,
            initial_sampler=backend.initial,
            reference_sampler=getattr(backend, "reference_id", backend.initial),
        )
        save(out / "status.json", status)

        def sample(cases, phase, count):
            groups = []
            for case in cases:
                status.update(state="sampling", phase=phase, case_id=case["id"])
                save(out / "status.json", status)
                budget.take("generations", count)
                pending = backend.sample(case, count)
                for i, row in enumerate(pending):
                    row.update(
                        case=case,
                        request=prompt(case),
                        writer_messages=messages(prompt(case)),
                        writer_protocol=PROTOCOL,
                        phase=phase,
                        sample=i,
                    )
                # Persist the complete provider result before the first judge or per-row write.
                save(out / "batches" / f"{phase}-{case['id']}.json", pending)
                status["generated_drafts"] += len(pending)
                save(out / "status.json", status)
                for i, row in enumerate(pending):
                    save(out / "trajectories" / f"{phase}-{case['id']}-{i}.json", row)
                if len(pending) != count:
                    raise ValueError("Incomplete sample group")
                group = []
                for i, row in enumerate(pending):
                    budget.take("jev")
                    raw = judge.score(
                        row["request"],
                        row["draft"],
                        grounded=case["grounding_applicable"],
                    )
                    save(out / "judge" / f"{phase}-{case['id']}-{i}.json", raw)
                    score = calculate(
                        raw,
                        row["draft"],
                        truncated=row["truncated"],
                        malformed=row["malformed"],
                        weights=c["weights"],
                    )
                    row.update(judge=raw, score=score)
                    records.append(row)
                    group.append(row)
                    save(out / "records.json", records)
                    status.update(state="judging", scored_drafts=len(records))
                    save(out / "status.json", status)
                    if tracking:
                        tracking.log(
                            {
                                f"{phase}/reward": score["reward"],
                                f"{phase}/words": score["words"],
                                "jev/latency_seconds": raw["telemetry"]["latency_seconds"],
                                f"{phase}/grounding_support_factor": score.get(
                                    "grounding_support_factor", 1.0
                                ),
                                f"{phase}/writing_reward_before_grounding": score["raw_reward"],
                                **{f"{phase}/component/{k}": v for k, v in score["components"].items()},
                            }
                        )
                groups.append(group)
                print(
                    f"{phase} {case['id']}: " + ", ".join(f"{r['score']['reward']:.3f}" for r in group),
                    flush=True,
                )
            return groups

        if c["steps"]:
            sample(validation, "before", c["eval_samples"])
        batches = training_batches(c, train)
        save(out / "training-schedule.json", [[x["id"] for x in batch] for batch in batches])
        for index, batch in enumerate(batches, 1):
            groups = sample(batch, f"train-{index:02}" if c["steps"] else "smoke", c["group_size"])
            folder = out / f"step-{index:02}"
            arrays, lengths, kl, gates = [], [], [], []
            reference_seconds = 0.0
            for group in groups:
                gate = advantages(
                    [r["score"]["reward"] for r in group], c["min_group_spread"], c["advantage_std_floor"]
                )
                gates.append({"case_id": group[0]["case"]["id"], **gate})
                adv = gate["advantages"] or [0.0] * len(group)
                for row, advantage in zip(group, adv, strict=True):
                    budget.take("references")
                    reference_started = time.perf_counter()
                    ref = backend.reference_probs(row)
                    reference_elapsed = time.perf_counter() - reference_started
                    reference_seconds += reference_elapsed
                    a, measured = datum(
                        row["prompt_tokens"], row["tokens"], row["logprobs"], ref, advantage, c["kl_beta"]
                    )
                    save(
                        out / "references" / f"{row['phase']}-{row['case']['id']}-{row['sample']}.json",
                        {
                            "fixed_sampler": status["reference_sampler"],
                            "logprobs": ref,
                            **measured,
                            "client_seconds": reference_elapsed,
                        },
                    )
                    kl.append(measured)
                    if gate["eligible"]:
                        arrays.append(a)
                        lengths.append(len(row["tokens"]))
            save(folder / "group-signal.json", gates)
            save(folder / "reference-kl.json", kl)
            mean_kl = sum(k["sampled_reference_kl"] * k["response_tokens"] for k in kl) / sum(
                k["response_tokens"] for k in kl
            )
            if mean_kl > min(c["max_reference_kl"], 0.05 if index == 1 else c["max_reference_kl"]):
                raise ValueError("Reference/behavior drift exceeds bound; no optimizer replay")
            metric = {
                "iteration": index,
                "eligible_groups": sum(g["eligible"] for g in gates),
                "groups": len(groups),
                "reference_sampled_kl": mean_kl,
                "reference_seconds": reference_seconds,
                "updated": False,
                "group_mean_reward": statistics.mean(g["mean"] for g in gates),
                "group_mean_std": statistics.mean(g["std"] for g in gates),
                "words_mean": statistics.mean(r["score"]["words"] for g in groups for r in g),
                "words_median": statistics.median(r["score"]["words"] for g in groups for r in g),
                "response_tokens_mean": statistics.mean(len(r["tokens"]) for g in groups for r in g),
                "guard_failure_fraction": statistics.mean(
                    not all(r["score"]["guards"].values()) for g in groups for r in g
                ),
                "truncation_fraction": statistics.mean(r["truncated"] for g in groups for r in g),
                "writing_reward_mean": statistics.mean(r["score"]["raw_reward"] for g in groups for r in g),
                "sampled_surprisal_nats": -sum(sum(r["logprobs"]) for g in groups for r in g)
                / sum(len(r["tokens"]) for g in groups for r in g),
            }
            if arrays and c["steps"]:
                save(folder / "training-arrays.json", arrays)
                backward_started = time.perf_counter()
                metric["behavior_alignment_kl"] = backend.backward(folder, arrays, lengths)
                metric["forward_backward_seconds"] = time.perf_counter() - backward_started
                budget.take("optimizer")

                def confirmed_optimizer():
                    status["optimizer_updates"] += 1
                    status["state"] = "saving-checkpoint"
                    save(out / "status.json", status)

                def confirmed_checkpoint(checkpoint):
                    # Called by the backend right after both saves; again after return is a no-op.
                    if checkpoint in checkpoints:
                        return
                    checkpoints.append(checkpoint)
                    save(out / "checkpoints.json", checkpoints)
                    status["checkpoint_bundles"] = len(checkpoints)
                    save(out / "status.json", status)

                optimizer_started = time.perf_counter()
                checkpoint = backend.optimize(
                    folder, index, on_optimizer=confirmed_optimizer, on_checkpoint=confirmed_checkpoint
                )
                metric["optimizer_and_checkpoint_seconds"] = time.perf_counter() - optimizer_started
                confirmed_checkpoint(checkpoint)
                metric["updated"] = True
            steps.append(metric)
            save(out / "step-metrics.json", steps)
            if tracking:
                metric_phase = "train" if c["steps"] else "smoke"
                tracking.log({f"{metric_phase}/{k}": v for k, v in metric.items()})
            label = "Training batch" if c["steps"] else "Smoke batch"
            print(
                f"{label} {index}/{len(batches)}: confirmed_updates={status['optimizer_updates']} "
                f"reward={metric['group_mean_reward']:.3f} words={metric['words_mean']:.1f} "
                f"KL={mean_kl:.5f}",
                flush=True,
            )
        if c["steps"]:
            sample(validation, "after", c["eval_samples"])
        else:
            # Hallucination zeros are valid negative training signals, not broken plumbing.
            no_bad = all(all(r["score"]["guards"][k] for k in TECHNICAL) for r in records)
            eligible = sum(g["eligible"] for g in gates)
            diagnostics = [diagnose(g, c["min_group_spread"]) for g in groups]
            style_candidates = sum(g["style_signal_candidate"] for g in diagnostics)
            minimum_eligible = min(3, len(gates))
            save(
                out / "smoke-summary.json",
                {
                    "passed": no_bad and style_candidates >= minimum_eligible,
                    "all_outputs_technically_valid": no_bad,
                    "eligible_groups": eligible,
                    "minimum_eligible_groups": minimum_eligible,
                    "style_signal_candidates": style_candidates,
                    "group_diagnostics": diagnostics,
                    "full_draft_review_required": True,
                    "hallucination_gate_zeros": sum(r["score"]["grounding"]["blocked"] for r in records),
                    "optimizer_updates": 0,
                    "note": "Sanity checks only; inspect full drafts and group advantages.",
                },
            )
        status["state"] = "complete"
    except BaseException as exc:
        status.update(state="halted", **safe_error(exc))
        save(out / "failure.json", safe_error(exc))
        raise
    finally:
        status["wall_seconds"] = time.perf_counter() - started
        finalize_errors = []
        for name, client in (("backend", backend), ("judge", judge)):
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    finalize_errors.append({"stage": name + "-close", **safe_error(exc)})
        save(out / "status.json", status)
        summary = None
        try:
            if (out / "records.json").exists():  # Absent when setup halted before any request.
                summary = report.build(out)
                status["report_state"] = "complete"
            else:
                status["report_state"] = "skipped"
        except Exception as exc:
            status["report_state"] = "failed"
            finalize_errors.append({"stage": "report", **safe_error(exc)})
        if tracking:
            try:
                tracking.summary["run/optimizer_updates"] = status["optimizer_updates"]
                tracking.summary["run/checkpoint_bundles"] = status["checkpoint_bundles"]
                if summary:
                    for k, v in summary["jev_economics"].items():
                        if isinstance(v, (int, float)):
                            tracking.summary[f"jev/{k}"] = v
                    for phase in ("before", "after"):
                        for k, v in summary[phase].items():
                            if isinstance(v, (int, float)):
                                tracking.summary[f"validation/{phase}_{k}"] = v
                        for k, v in summary[phase].get("components", {}).items():
                            tracking.summary[f"validation/{phase}_component/{k}"] = v
                        if summary[phase]:
                            tracking.summary[f"validation/{phase}_reward"] = summary[phase]["reward_mean"]
            except Exception as exc:
                finalize_errors.append({"stage": "wandb-summary", **safe_error(exc)})
            finally:
                try:
                    tracking.finish(
                        exit_code=0 if status["state"] == "complete" and not finalize_errors else 1
                    )
                except Exception as exc:
                    finalize_errors.append({"stage": "wandb-finish", **safe_error(exc)})
        status["finalization_errors"] = finalize_errors
        save(out / "status.json", status)
        if finalize_errors:
            save(out / "finalization-errors.json", finalize_errors)
            if status["state"] == "complete":
                raise RuntimeError(
                    "Training completed but finalization failed; inspect artifacts, do not replay updates"
                )
    return summary
