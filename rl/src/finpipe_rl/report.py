"""Qualification from native artifacts. No synthetic measurements or inferred wins."""

import copy
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from finpipe_rl.evaluation import compare
from finpipe_rl.manifest import digest, file_hash, load_manifest
from finpipe_rl.runtime import metadata_dir, write_json


def read_episodes(run_dir: Path, monitor="file"):
    stream = run_dir / "monitors" / monitor / "traces/stream"
    chunks = {p.name.split(".")[0] for p in stream.glob("*.jsonl*")}
    if not chunks:
        raise ValueError(f"No completed native trace stream in {run_dir}")
    for number in sorted(chunks):
        plain = stream / f"{number}.jsonl"
        if plain.exists():
            source = plain.open("rb")
        else:
            # Reuse Prime's actual compression dependency, only for compressed runs.
            import pyzstd

            source = pyzstd.SeekableZstdFile(stream / f"{number}.jsonl.zst", "rb")
        with source:
            for line in source:
                if not line.endswith(b"\n"):
                    raise ValueError(
                        "Truncated trace record; report only a completed run"
                    )
                yield json.loads(line)


def validate_pair(baseline, trained):
    if baseline.get("role") != "baseline" or trained.get("role") != "trained":
        raise ValueError("Comparison requires baseline and trained run roles")
    for key in (
        "kind",
        "manifest",
        "training_run",
        "training_identity",
        "framework",
        "integration_sha256",
    ):
        if baseline.get(key) != trained.get(key):
            raise ValueError(f"Evaluation provenance mismatch: {key}")
    protocols = []
    for record in (baseline, trained):
        cfg = copy.deepcopy(record["config"])
        for key in ("model", "client", "run", "output_dir", "monitors"):
            cfg.pop(key, None)
        protocols.append(cfg)
    if protocols[0] != protocols[1]:
        raise ValueError(
            "Evaluation protocols differ (sampling, harness, task selection, or budgets)"
        )
    if baseline["artifact"]["sha256"] == trained["artifact"]["sha256"]:
        raise ValueError("Baseline and trained artifacts are identical")


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "samples": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[int(0.1 * (len(ordered) - 1))],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
    }


def summarize_metrics(run_dir):
    series, unavailable = defaultdict(list), 0
    files = sorted((run_dir / "monitors").glob("file*/metrics.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            for key, value in row.items():
                if key in ("step", "time"):
                    continue
                if not any(
                    s in key
                    for s in (
                        "loss",
                        "entropy",
                        "kl",
                        "grad_norm",
                        "perf/",
                        "reward",
                        "error",
                        "throughput",
                        "time/",
                    )
                ):
                    continue
                if value is None or (
                    isinstance(value, (float, int)) and not math.isfinite(value)
                ):
                    unavailable += 1
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    series[key].append(value)
    return {
        "status": "measured" if series else "not_measured",
        "unavailable_samples": unavailable,
        "series": {key: distribution(values) for key, values in sorted(series.items())},
        "sources": [str(p) for p in files],
    }


def summarize_gpu(run_dir):
    by_gpu = defaultdict(lambda: defaultdict(list))
    for path in metadata_dir(run_dir).glob("gpu-*.csv"):
        with path.open() as stream:
            for row in csv.reader(stream):
                if len(row) != 5:
                    continue
                for name, raw in zip(
                    ("utilization_percent", "memory_mib", "power_watts"), row[2:]
                ):
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        by_gpu[row[1].strip()][name].append(value)
    return {
        "status": "measured" if by_gpu else "not_measured",
        "scope": "nvidia-smi samples on the launcher host only; not cluster-wide",
        "gpus": {
            gpu: {k: distribution(v) for k, v in metrics.items()}
            for gpu, metrics in by_gpu.items()
        },
    }


def _receipt(run_dir, completed=True):
    path = metadata_dir(run_dir) / "run.json"
    receipt = json.loads(path.read_text())
    if completed and (
        not receipt["attempts"] or receipt["attempts"][-1]["status"] != "completed"
    ):
        raise ValueError(f"Run is not completed: {run_dir}")
    return receipt


def build_report(
    manifest_path,
    baseline_dir,
    trained_dir,
    training_dir,
    output,
    *,
    gpu_hour_rate=None,
    audit_dirs=(),
):
    manifest = load_manifest(manifest_path)
    base, trained = _receipt(baseline_dir), _receipt(trained_dir)
    training = _receipt(training_dir, completed=False)
    validate_pair(base["protocol"], trained["protocol"])
    for directory in (baseline_dir, trained_dir):
        for archived in (directory / "monitors").glob("file.attempt_*"):
            if any((archived / "traces/stream").glob("*.jsonl*")) and any(
                not e["ok"] for e in read_episodes(directory, archived.name)
            ):
                raise ValueError(
                    "Resumed evaluation replaced failed episodes; run a fresh paired comparison"
                )
    if base["protocol"]["manifest"] != manifest.digest:
        raise ValueError("Report manifest differs from evaluated manifest")
    if base["protocol"]["training_identity"] != digest(training["protocol"]):
        raise ValueError("Training run changed since evaluation")
    repeats = base["protocol"]["config"]["group_size"]
    result = compare(
        manifest,
        read_episodes(baseline_dir),
        read_episodes(trained_dir),
        repeats=repeats,
    )
    result["manifest_sha256"] = manifest.digest
    result["training"] = summarize_metrics(training_dir)
    result["gpu"] = summarize_gpu(training_dir)
    execution = {}
    for name, run, receipt in (
        ("train", training_dir, training),
        ("baseline", baseline_dir, base),
        ("trained", trained_dir, trained),
    ):
        attempts = receipt["attempts"]
        elapsed = sum(a.get("elapsed_seconds", 0) for a in attempts)
        count = None
        try:
            count = sum(1 for _ in read_episodes(run))
        except ValueError:
            if name != "train":
                raise
        execution[name] = {
            "attempts": len(attempts),
            "failed_or_interrupted_attempts": sum(
                a["status"] != "completed" for a in attempts
            ),
            "measured_launch_seconds": elapsed,
            "final_stream_episodes": count,
            "final_stream_episodes_per_second": count / elapsed
            if count is not None and elapsed > 0
            else None,
        }
    result["execution"] = execution
    result["resource_consumption"] = {
        "billed_cost": None,
        "allocation_cost_estimate": None,
        "limitations": "Slurm allocation time/cost requires scheduler accounting; launcher wall time includes queue time. External inference, CPU and storage costs are not collected.",
    }
    if gpu_hour_rate is not None and (
        not math.isfinite(gpu_hour_rate) or gpu_hour_rate < 0
    ):
        raise ValueError("GPU-hour price must be finite and nonnegative")
    if gpu_hour_rate is not None and not training["protocol"]["config"].get("slurm"):
        deploy = training["protocol"].get("resolved_deployment") or training[
            "protocol"
        ]["config"].get("deployment", {})
        if deploy.get("type", "single_node") == "single_node":
            count = deploy["num_train_gpus"] + deploy["num_infer_gpus"]
        else:
            count = deploy["gpus_per_node"] * (
                deploy["num_train_nodes"]
                + deploy.get("num_infer_nodes", 1) * deploy.get("num_infer_replicas", 1)
            )
        hours = execution["train"]["measured_launch_seconds"] * count / 3600
        result["resource_consumption"]["allocation_cost_estimate"] = {
            "configured_gpus": count,
            "gpu_hours": hours,
            "price_per_gpu_hour": gpu_hour_rate,
            "cost": hours * gpu_hour_rate,
            "scope": "training launch wall time × configured allocation; excludes external eval, CPU, storage, and idle reservation time",
        }
    result["environment_quality"] = audit_summary(manifest, audit_dirs)
    result["reproducibility"] = {
        "framework": training["protocol"]["framework"],
        "integration_sha256": training["protocol"].get("integration_sha256"),
        "training_config_sha256": digest(training["protocol"]["config"]),
        "model_artifacts": {
            "baseline": base["protocol"]["artifact"],
            "trained": trained["protocol"]["artifact"],
        },
        "protocol_sha256": digest(base["protocol"]["config"]),
        "training_seed_replicates": None,
        "endpoint_binding": "operator attests that the served model is the recorded artifact; OpenAI model listing cannot attest weight bytes",
        "source_receipts": {
            str(metadata_dir(p) / "run.json"): file_hash(metadata_dir(p) / "run.json")
            for p in (training_dir, baseline_dir, trained_dir)
        },
    }
    ci = result["success"]["ci95"]
    result["qualification"] = {
        "observed_improvement": result["success"]["improvement"] > 0,
        "positive_ci_with_20_groups": bool(
            ci and ci[0] > 0 and result["independent_groups"] >= 20
        ),
        "execution_failure_rate_at_most_2_percent": all(
            v["failed_episodes"] / v["episodes"] <= 0.02
            for v in result["reliability"].values()
        ),
        "training_stability": "review_measured_series"
        if result["training"]["status"] == "measured"
        else "not_measured",
        "verifier_reference_and_repeatability": "not_measured",
        "gpu_checkpoint_recovery": "not_measured",
        "overall": "requires_hardware_and_verifier_qualification",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "qualification.json", result)
    (output / "qualification.md").write_text(render_report(result))
    with (output / "tasks.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task",
                "baseline_success",
                "trained_success",
                "improvement",
                "baseline_failures",
                "trained_failures",
            ]
        )
        for row in result["task_breakdown"]:
            b, t = row["baseline"], row["trained"]
            writer.writerow(
                [
                    row["task"],
                    b["success_rate"],
                    t["success_rate"],
                    t["success_rate"] - b["success_rate"],
                    b["failures"],
                    t["failures"],
                ]
            )
    return result


def audit_summary(manifest, directories):
    runs = []
    for directory in directories:
        config = json.loads((directory / "configs/validate.json").read_text())
        if load_manifest(Path(config["taskset"]["manifest"])).digest != manifest.digest:
            raise ValueError("Audit manifest mismatch")
        runs.append(
            {
                "directory": str(directory),
                "summary": json.loads((directory / "summary.json").read_text()),
            }
        )
    return {
        "status": "measured_native_checks" if runs else "not_measured",
        "native_validation": runs,
        "limitations": "Native gold validation may be unchecked when tasks provide no reference solution. Native noop's reward>=1 rule is not a Finpipe metric-specific success threshold. These checks alone do not establish verifier accuracy or repeatability.",
    }


def render_report(result):
    success = result["success"]
    lines = [
        "# RL qualification report",
        "",
        f"Held-out tasks: {result['tasks']}; independent groups: {result['independent_groups']}; rollouts/task/model: {result['repeats']}.",
        "",
        f"Success: {success['baseline']:.2%} baseline → {success['trained']:.2%} trained. Improvement: {success['improvement']:+.2%} (percentage-point change).",
        f"95% paired interval: {success['ci95'] if success['ci95'] else 'unavailable: fewer than two independent groups'}.",
        "",
        "## Existing metrics",
        "",
        "| Metric | Baseline | Trained | Direction-adjusted improvement | 95% interval | Paired tasks |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for key, row in result["metrics"].items():
        lines.append(
            f"| {key} | {row['baseline']} | {row['trained']} | {row['improvement']} | {row['ci95']} | {row['paired_tasks']} |"
        )
    for title, key in (
        ("Environment and verifier quality", "environment_quality"),
        ("Execution reliability and throughput", "execution"),
        ("Training stability and reward distributions", "training"),
        ("GPU samples", "gpu"),
        ("Resource consumption", "resource_consumption"),
        ("Qualification gates", "qualification"),
    ):
        lines.extend(
            ["", f"## {title}", "", "```json", json.dumps(result[key], indent=2), "```"]
        )
    lines.extend(
        [
            "",
            "## Reproducibility and limitations",
            "",
            f"Manifest: `{result['manifest_sha256']}`. Exact artifact hashes, framework versions, protocol and source receipts are in qualification.json.",
            "",
            "The endpoint operator must attest the served weights. Repeated training-seed runs, fixed-artifact verifier repeatability, and GPU checkpoint recovery remain unmeasured unless separately qualified.",
            "",
            *[f"- {x}" for x in result["limitations"]],
            "",
            "Task-level results: tasks.csv and qualification.json.",
            "",
        ]
    )
    return "\n".join(lines)
