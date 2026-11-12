"""Small command surface around native framework configs and artifacts."""

import argparse
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from finpipe_rl.manifest import freeze, load_manifest
from finpipe_rl.runtime import launch, prepare_config, write_json


def read_config(path):
    return (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else tomllib.loads(path.read_text())
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="finpipe-rl")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "freeze", help="Freeze an index of existing Harbor task directories"
    )
    prepare.add_argument("catalog", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    check = commands.add_parser(
        "check", help="Revalidate task contents and split isolation"
    )
    check.add_argument("manifest", type=Path)
    for kind in ("train", "eval"):
        run = commands.add_parser(kind)
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument(
            "--artifact",
            type=Path,
            required=True,
            help="Pinned local base snapshot or native LoRA broadcast",
        )
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--prime-root", type=Path, required=True)
        run.add_argument("--resume", action="store_true")
        run.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate with real framework schemas; do not launch",
        )
        if kind == "train":
            run.add_argument("--max-steps", type=int)
        else:
            run.add_argument("--served-model", required=True)
            run.add_argument("--base-url", required=True)
            run.add_argument("--training-run", type=Path, required=True)
            run.add_argument("--role", choices=("baseline", "trained"), required=True)
            run.add_argument(
                "--baseline-run",
                type=Path,
                help="Generate qualification report automatically after trained evaluation",
            )
    audit = commands.add_parser(
        "audit", help="Use native model-free Harbor setup/gold/noop validation"
    )
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--concurrency", type=int, default=4)
    report = commands.add_parser("report")
    for flag in ("manifest", "baseline", "trained", "training", "output"):
        report.add_argument(f"--{flag}", type=Path, required=True)
    report.add_argument("--gpu-hour-rate", type=float)
    report.add_argument("--audit", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "freeze":
        catalog = read_config(args.catalog)
        for task in catalog["tasks"]:
            path = Path(task["path"]).expanduser()
            task["path"] = str(
                path if path.is_absolute() else args.catalog.resolve().parent / path
            )
        result = freeze(catalog, args.output)
        print(json.dumps({"manifest": str(args.output), "sha256": result.digest}))
    elif args.command == "check":
        manifest = load_manifest(args.manifest)
        print(json.dumps({"sha256": manifest.digest, "tasks": len(manifest.tasks)}))
    elif args.command in ("train", "eval"):
        raw = read_config(args.config)
        if args.command == "train" and args.max_steps is not None:
            raw["max_steps"] = args.max_steps
        model = (
            str(args.artifact.resolve())
            if args.command == "train"
            else args.served_model
        )
        if args.command == "eval":
            raw.setdefault("client", {})["base_url"] = args.base_url
        config = prepare_config(args.command, raw, args.manifest, args.output, model)
        result = launch(
            args.command,
            config,
            args.manifest,
            args.artifact,
            args.output.resolve(),
            args.prime_root.resolve(),
            resume=args.resume,
            dry_run=args.dry_run,
            training_run=getattr(args, "training_run", None),
            role=getattr(args, "role", None),
        )
        if args.dry_run:
            print(json.dumps(result, indent=2))
        elif args.command == "eval" and args.role == "trained" and args.baseline_run:
            from finpipe_rl.report import build_report

            build_report(
                args.manifest,
                args.baseline_run,
                args.output,
                args.training_run,
                args.output / "reports",
            )
    elif args.command == "audit":
        from verifiers.v1.configs.cli.validate import ValidateConfig

        load_manifest(args.manifest)
        config = {
            "taskset": {
                "id": "finpipe-rl",
                "manifest": str(args.manifest.resolve()),
                "split": args.split,
            },
            "runtime": {"type": "docker"},
            "max_concurrent": args.concurrency,
            "timeout": {"setup": 600, "total": 2400},
            "output_dir": str(args.output.resolve().parent),
            "run": {"name": args.output.name},
        }
        ValidateConfig.model_validate(copy.deepcopy(config))
        config_file = args.output.with_suffix(".validate.json")
        if config_file.exists() or args.output.exists():
            raise FileExistsError("Use a new audit output directory")
        write_json(config_file, config)
        subprocess.run(
            [str(Path(sys.executable).parent / "vf-validate"), "@", str(config_file)],
            check=True,
        )
    else:
        from finpipe_rl.report import build_report

        build_report(
            args.manifest,
            args.baseline,
            args.trained,
            args.training,
            args.output,
            gpu_hour_rate=args.gpu_hour_rate,
            audit_dirs=args.audit,
        )


if __name__ == "__main__":
    main()
