import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from . import calibration, report
from .config import load, plan
from .reward import PREFLIGHT_REQUESTS, make_judge
from .runner import run
from .storage import Budget, fresh, safe_error, save


def main():
    parser = argparse.ArgumentParser(
        description="Bounded Fireworks + Jev RL. Paid commands require --execute."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "scorer-check"):
        p = sub.add_parser(name)
        p.add_argument("--env-file", default=".env", help="Local credentials file; never committed")
        p.add_argument(
            "--config",
            default="experiments/raw-base-v1/config.json",
            help="Recipe configuration (default: experiments/raw-base-v1/config.json)",
        )
        if name != "plan":
            p.add_argument("--output", required=True)
            p.add_argument("--execute", action="store_true")
        if name == "run":
            p.add_argument("--mock", action="store_true")
            p.add_argument("--wandb", action="store_true")
            p.add_argument("--smoke-from")
            p.add_argument(
                "--require-review", action="store_true", help="Also require an approved smoke-review.json"
            )
    p = sub.add_parser("report")
    p.add_argument("folder")
    args = parser.parse_args()
    load_dotenv(getattr(args, "env_file", ".env"), override=False)
    try:
        if args.command == "report":
            result = report.build(Path(args.folder))
        elif args.command == "plan":
            c, t, v, _ = load(args.config)
            result = plan(c, t, v)
        elif not args.execute and not getattr(args, "mock", False):
            parser.error("No paid API calls made. Use --execute, or --mock for an offline plumbing test.")
        elif args.command == "run":
            summary = run(
                args.config,
                args.output,
                mock=args.mock,
                wandb=args.wandb,
                smoke_from=args.smoke_from,
                require_review=args.require_review,
            )
            # Keep evaluation scores off the console so the blind review stays blind.
            status = (summary or {}).get("status", {})
            result = {
                "output": args.output,
                "state": status.get("state"),
                "mock": status.get("mock"),
                "optimizer_updates": status.get("optimizer_updates"),
                "next": "Smoke: review it with scripts/review_raw_smoke.py"
                if not status.get("optimizer_updates")
                else "Do the blind review first; `fw-jev report` prints scores.",
            }
        else:
            c, _, _, _ = load(args.config)
            judge = make_judge(os.environ.get("TYPESAFE_API_KEY"))
            folder = fresh(args.output)
            budget = Budget(folder, {"jev": PREFLIGHT_REQUESTS})
            try:
                result = calibration.run(judge, folder, c["weights"], budget)
            except BaseException as exc:
                save(folder / "failure.json", safe_error(exc))
                raise
            finally:
                judge.close()
            if not result["passed"]:
                print(json.dumps(result, indent=2))
                raise SystemExit("Scorer check failed; stop before training.")
        print(json.dumps(result, indent=2))
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(safe_error(exc), indent=2))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
