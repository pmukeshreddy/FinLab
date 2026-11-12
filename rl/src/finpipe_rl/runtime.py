"""Validate experiments; delegate launches and checkpoint recovery to Prime-RL."""

import copy
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from finpipe_rl.manifest import digest, file_hash, load_manifest

PRIME_REVISION = "a3ee545b686189a860aba4ee3c9f230aa1ae4705"
SUBMODULES = {
    "verifiers": "ae5e56bc4e714c782817b4bde822fa084fc319d2",
    "renderers": "c3133fcc5bd577e4d0061398dc67b0dc15ff5308",
    "pydantic-config": "65b15dffba82d4be19efdaf8b2b9705cc1756be8",
}


def prepare_config(kind, raw, manifest_path, run_dir, model):
    load_manifest(manifest_path)
    cfg = copy.deepcopy(raw)
    if cfg.get("clean") or cfg.get("resume") is not None or cfg.get("dry_run"):
        raise ValueError("Use runner options for resume/dry-run; clean is forbidden")
    cfg["output_dir"] = str(run_dir.resolve().parent)
    cfg["run"] = {"name": run_dir.name, "dir": run_dir.name}
    cfg["monitors"] = {**cfg.get("monitors", {}), "file": {}, "prime": None}
    if kind == "train":
        cfg.setdefault("model", {})["name"] = model
        for component in ("trainer", "orchestrator"):
            if (
                cfg[component].get("resume") is not None
                or cfg[component].get("ckpt") is not None
            ):
                raise ValueError(
                    "Resume and checkpoints must use the shared run configuration"
                )
            component_model = cfg[component].get("model", {})
            if component_model.get("name", model) != model or component_model.get(
                "init_from"
            ):
                raise ValueError(
                    "Component model overrides differ from the recorded base artifact"
                )
            if any(component_model.get("debug", {}).values()):
                raise ValueError(
                    "Model debug overrides cannot qualify a base-model comparison"
                )
            cfg[component].setdefault("monitors", {})["prime"] = None
        if cfg.get("ckpt", {}).get("output_dir"):
            raise ValueError("Checkpoints must remain in the recorded run directory")
        if cfg.get("inference", {}).get("vllm", {}).get("model", model) != model:
            raise ValueError(
                "Inference model override differs from the training artifact"
            )
        cfg.setdefault("env_vars", {})["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1]
        )
        if not cfg.get("ckpt", {}).get("interval"):
            raise ValueError("Periodic checkpoints are required")
        if cfg["trainer"].get("data", {}).get("fake") is not None:
            raise ValueError("Fake training data cannot qualify a Harbor task run")
        if not cfg["trainer"].get("model", {}).get("lora"):
            raise ValueError("LoRA must be enabled")
        if cfg["trainer"]["model"].get("conversion_dir") is not None:
            raise ValueError(
                "Model conversion cache is managed inside the run directory"
            )
        cfg["trainer"]["model"]["conversion_dir"] = str(
            run_dir.resolve() / "model_conversion"
        )
        if cfg["orchestrator"]["train"].get("algo", {}).get("type") != "grpo":
            raise ValueError("This integration requires GRPO")
        groups = [
            (cfg["orchestrator"]["train"], "train"),
            (cfg["orchestrator"]["eval"], "validation"),
        ]
    else:
        cfg["model"] = model
        groups = [(cfg, "test")]
    for group, split in groups:
        if group.get("env", {}).get("taskset"):
            raise ValueError(
                "Shared taskset overrides would change the frozen task contract"
            )
        if group.get("select"):
            raise ValueError("Task selection must come exclusively from the manifest")
        sources = group.get("source", [])
        if len(sources) != 1:
            raise ValueError("Exactly one local Finpipe source per split is required")
        source = sources[0]
        if source.get("group_size", group.get("group_size", 1)) != group.get(
            "group_size", 1
        ):
            raise ValueError(
                "Per-source group size must equal the experiment group size"
            )
        if source.get("algo") is not None:
            raise ValueError(
                "Set the training algorithm only on the shared train block"
            )
        if source.get("select") or source.get("serve", {}).get("address"):
            raise ValueError(
                "External sources/task selection could bypass the manifest"
            )
        taskset = source.get("env", {}).get("taskset", {})
        if taskset.get("id") != "finpipe-rl" or taskset.get("split") != split:
            raise ValueError(f"Expected finpipe-rl {split} split")
        taskset["manifest"] = str(manifest_path.resolve())
        if set(taskset) - {"id", "split", "manifest"}:
            raise ValueError("Taskset overrides could alter the frozen task contract")
        agent = source.get("env", {}).get("agent", {})
        if any(key in agent for key in ("model", "client", "sampling")) or source.get(
            "env", {}
        ).get("id"):
            raise ValueError(
                "Use the single-agent policy and experiment-level model/sampling"
            )
        scoring = agent.get("scoring", {})
        if scoring.get("mode", "isolated") != "isolated" or any(
            k in scoring for k in ("runtime", "env", "network")
        ):
            raise ValueError(
                "Scoring overrides must preserve the declared isolated verifier"
            )
        if group.get("group_size", 1) < (2 if split == "train" else 1):
            raise ValueError("GRPO requires at least two rollouts per task")
    return cfg


def resume_config(previous, current, step):
    if step is None:
        raise ValueError("No complete checkpoint; refusing to restart from scratch")
    before, after = copy.deepcopy(previous), copy.deepcopy(current)
    for item in (before, after):
        item.pop("max_steps", None)
        item.pop("resume", None)
    if before != after:
        raise ValueError("Experiment configuration changed on resume")
    if current["max_steps"] <= step:
        raise ValueError("max_steps must exceed the recovered checkpoint step")
    return {**current, "resume": {"step": step}}


def artifact_identity(path: Path) -> dict:
    path = path.resolve()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    weights = [p for p in files if p.suffix == ".safetensors"]
    if not weights or not any(
        p.name in ("config.json", "adapter_config.json") for p in files
    ):
        raise ValueError(
            "Model artifact needs safetensors weights and a model/adapter config"
        )
    # Stream each file once. This deliberately includes weights, not just filenames.
    hashes = {str(p.relative_to(path)): file_hash(p) for p in files}
    return {"path": str(path), "sha256": digest(hashes), "files": hashes}


def check_framework(root: Path) -> dict:
    def head(path):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()

    expected = {"prime-rl": PRIME_REVISION, **SUBMODULES}
    actual = {"prime-rl": head(root)}
    actual.update({key: head(root / "deps" / key) for key in SUBMODULES})
    if actual != expected:
        raise ValueError(
            "Framework checkout differs from the reviewed pinned revisions"
        )
    for path in [root, *(root / "deps" / k for k in SUBMODULES)]:
        if subprocess.check_output(
            ["git", "-C", str(path), "diff", "HEAD", "--"], text=True
        ):
            raise ValueError(f"Framework has modified tracked files: {path}")
    import prime_rl
    import verifiers

    if not any(
        Path(p).resolve().is_relative_to(root.resolve()) for p in prime_rl.__path__
    ):
        raise ValueError(
            "Run with the pinned Prime-RL environment, not another installation"
        )
    if (
        not Path(verifiers.__file__)
        .resolve()
        .is_relative_to((root / "deps/verifiers").resolve())
    ):
        raise ValueError("Verifiers must come from the pinned Prime-RL submodule")
    return {
        "revisions": actual,
        "lock_sha256": file_hash(root / "uv.lock"),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("prime-rl", "verifiers", "renderers", "harbor", "torch", "vllm")
        },
    }


def metadata_dir(run_dir: Path) -> Path:
    return run_dir.parent / ".finpipe" / run_dir.name


def complete_checkpoint(run_dir):
    """Select a joint committed native checkpoint, not a half-written newer step."""
    steps = []
    for path in (run_dir / "checkpoints").glob("step_*"):
        if (
            path.name[5:].isdigit()
            and (path / "trainer/.metadata").is_file()
            and (path / "orchestrator/progress.pt").is_file()
        ):
            steps.append(int(path.name[5:]))
    return max(steps, default=None)


def validate_eval_resume(run_dir, receipt):
    from finpipe_rl.report import read_episodes

    if receipt["attempts"][-1]["status"] == "completed":
        raise ValueError("Cannot resume a completed evaluation and retry its failures")
    for monitor in (run_dir / "monitors").glob("file*"):
        if any((monitor / "traces/stream").glob("*.jsonl*")) and any(
            not e["ok"] for e in read_episodes(run_dir, monitor.name)
        ):
            raise ValueError(
                "Cannot replace failed evaluation episodes on resume; run a fresh pair"
            )


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def launch(
    kind,
    config,
    manifest_path,
    artifact,
    run_dir,
    prime_root,
    *,
    resume=False,
    dry_run=False,
    training_run=None,
    role=None,
):
    """One native subprocess. A receipt preserves attempt history and the exact protocol."""
    import fcntl

    from prime_rl.configs.eval import EvalConfig
    from prime_rl.configs.rl import RLConfig

    model_cls = RLConfig if kind == "train" else EvalConfig
    validated = model_cls.model_validate(copy.deepcopy(config))
    if dry_run:
        return validated.model_dump(mode="json")
    framework = check_framework(prime_root)
    manifest = load_manifest(manifest_path)
    identity = artifact_identity(artifact)
    meta = metadata_dir(run_dir)
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = meta / "run.json"
        protocol = {
            "kind": kind,
            "manifest": manifest.digest,
            "artifact": identity,
            "framework": framework,
            "config": config,
            "integration_sha256": digest(
                {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
            ),
        }
        if kind == "train":
            protocol["resolved_deployment"] = validated.deployment.model_dump(
                mode="json"
            )
        if training_run is not None:
            training = json.loads((metadata_dir(training_run) / "run.json").read_text())
            if (
                training["protocol"]["kind"] != "train"
                or training["protocol"]["manifest"] != manifest.digest
            ):
                raise ValueError("Evaluation training run/manifest mismatch")
            if (
                role == "baseline"
                and identity["sha256"] != training["protocol"]["artifact"]["sha256"]
            ):
                raise ValueError("Baseline weights differ from the training base")
            if role == "trained" and (
                artifact.resolve().parent != (training_run / "broadcasts").resolve()
                or not (artifact / ".finished").is_file()
            ):
                raise ValueError(
                    "Trained adapter must be a completed native broadcast of this training run"
                )
            if role == "trained" and not (artifact / "adapter_config.json").is_file():
                raise ValueError("Trained evaluation requires a saved LoRA adapter")
            protocol.update(
                training_run=str(training_run.resolve()),
                role=role,
                training_identity=digest(training["protocol"]),
            )
        elif kind == "eval":
            raise ValueError(
                "Evaluation requires a training run and baseline/trained role"
            )
        if resume:
            previous = json.loads(receipt_path.read_text())
            old_protocol = previous["protocol"]
            if {k: v for k, v in old_protocol.items() if k != "config"} != {
                k: v for k, v in protocol.items() if k != "config"
            }:
                raise ValueError("Run provenance changed on resume")
            if kind == "train":
                step = complete_checkpoint(run_dir)
                config = resume_config(old_protocol["config"], config, step)
            else:
                validate_eval_resume(run_dir, previous)
                if old_protocol["config"] != config:
                    raise ValueError("Evaluation configuration changed on resume")
                config = {**config, "resume": {}}
            attempts = previous["attempts"]
        else:
            if receipt_path.exists() or run_dir.exists():
                raise FileExistsError(
                    "Run already exists; choose a new name or --resume"
                )
            attempts = []
        native = meta / "launch.json"
        write_json(native, config)
        command = [
            str(Path(sys.executable).parent / ("rl" if kind == "train" else "eval")),
            "@",
            str(native.resolve()),
        ]
        attempt = {"started_at": time.time(), "command": command, "status": "running"}
        attempts.append(attempt)
        receipt = {"protocol": protocol, "attempts": attempts}
        write_json(receipt_path, receipt)
        started = time.monotonic()
        gpu_process = gpu_stream = None
        process_env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        if config.get("slurm"):
            # Slurm's native wait option propagates job failure, not submission success.
            process_env["SBATCH_WAIT"] = "1"
            attempt["timing_scope"] = (
                "submission through completion, including queue time"
            )
        elif shutil.which("nvidia-smi"):
            gpu_stream = (meta / f"gpu-{len(attempts)}.csv").open("w")
            gpu_command = [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "-l",
                "5",
            ]
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                gpu_command += ["-i", os.environ["CUDA_VISIBLE_DEVICES"]]
            try:
                gpu_process = subprocess.Popen(
                    gpu_command, stdout=gpu_stream, stderr=subprocess.DEVNULL
                )
            except OSError as exc:
                attempt["gpu_telemetry_error"] = str(exc)
        try:
            result = subprocess.run(
                command, cwd=prime_root, env=process_env, check=False
            )
            attempt.update(
                returncode=result.returncode,
                status="completed" if result.returncode == 0 else "failed",
            )
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
        except BaseException:
            if attempt["status"] == "running":
                attempt["status"] = "interrupted"
            raise
        finally:
            if gpu_process is not None:
                gpu_process.terminate()
                try:
                    gpu_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    gpu_process.kill()
                    gpu_process.wait()
            if gpu_stream is not None:
                gpu_stream.close()
            attempt.update(
                ended_at=time.time(), elapsed_seconds=time.monotonic() - started
            )
            write_json(receipt_path, receipt)
    return receipt
