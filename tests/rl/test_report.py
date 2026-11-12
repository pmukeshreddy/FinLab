import copy
import json

import pytest
from finpipe_rl.report import read_episodes, summarize_metrics, validate_pair


def protocol(role):
    return {
        "kind": "eval",
        "manifest": "manifest-digest",
        "role": role,
        "training_run": "/runs/train",
        "training_identity": "train-digest",
        "artifact": {"sha256": role},
        "framework": {"version": "pinned"},
        "integration_sha256": "code-digest",
        "config": {
            "model": role,
            "run": {"name": role},
            "output_dir": "/runs",
            "client": {"base_url": "http://localhost:8000/v1"},
            "group_size": 2,
            "sampling": {"temperature": 0.8},
            "source": [{"name": "test"}],
        },
    }


def test_rejects_incompatible_evaluation_protocols():
    baseline, trained = protocol("baseline"), protocol("trained")
    validate_pair(baseline, trained)
    for field in ("manifest", "training_identity", "framework"):
        changed = copy.deepcopy(trained)
        changed[field] = "different"
        with pytest.raises(ValueError):
            validate_pair(baseline, changed)
    changed = copy.deepcopy(trained)
    changed["integration_sha256"] = "different"
    with pytest.raises(ValueError):
        validate_pair(baseline, changed)
    changed = copy.deepcopy(trained)
    changed["config"]["sampling"]["temperature"] = 0.1
    with pytest.raises(ValueError):
        validate_pair(baseline, changed)


def test_reads_native_stream_and_rejects_truncated_record(tmp_path):
    stream = tmp_path / "monitors/file/traces/stream"
    stream.mkdir(parents=True)
    (stream / "00000.jsonl").write_text('{"id":"first"}\n')
    assert list(read_episodes(tmp_path)) == [{"id": "first"}]
    (stream / "00001.jsonl").write_text('{"id":')
    with pytest.raises(ValueError):
        list(read_episodes(tmp_path))


def test_missing_and_nonfinite_metrics_are_not_success(tmp_path):
    assert summarize_metrics(tmp_path)["status"] == "not_measured"
    path = tmp_path / "monitors/file/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"step": 1, "loss/mean": None, "perf/mfu": 42}) + "\n")
    summary = summarize_metrics(tmp_path)
    assert summary["unavailable_samples"] == 1
    assert summary["series"]["perf/mfu"]["mean"] == 42
    assert "loss/mean" not in summary["series"]


def test_resume_rejects_failed_cohort_and_completed_runs(tmp_path):
    from finpipe_rl.runtime import validate_eval_resume

    receipt = {"attempts": [{"status": "completed"}]}
    with pytest.raises(ValueError, match="completed"):
        validate_eval_resume(tmp_path, receipt)
    receipt["attempts"][-1]["status"] = "interrupted"
    validate_eval_resume(tmp_path, receipt)  # no landed episodes yet
    stream = tmp_path / "monitors/file/traces/stream"
    stream.mkdir(parents=True)
    (stream / "00000.jsonl").write_text('{"id":"a", "ok":false}\n')
    with pytest.raises(ValueError, match="failed"):
        validate_eval_resume(tmp_path, receipt)


@pytest.mark.parametrize("slurm", [False, True])
def test_complete_report_uses_only_measured_evidence(catalog, tmp_path, slurm):
    from finpipe_rl.manifest import digest, freeze
    from finpipe_rl.report import build_report
    from finpipe_rl.runtime import metadata_dir, write_json
    from test_evaluation import episode

    path = tmp_path / "manifest.json"
    manifest = freeze(catalog, path)
    training, base, trained = [
        tmp_path / name for name in ("training-run", "baseline-run", "trained-run")
    ]
    train_protocol = {
        "kind": "train",
        "manifest": manifest.digest,
        "framework": {"version": "pinned"},
        "config": {
            "deployment": {
                "type": "single_node",
                "num_train_gpus": 4,
                "num_infer_gpus": 4,
            }
        },
    }
    if slurm:
        train_protocol["config"]["slurm"] = {"job_name": "test"}
    attempt = {"status": "completed", "elapsed_seconds": 3600}
    write_json(
        metadata_dir(training) / "run.json",
        {"protocol": train_protocol, "attempts": [attempt]},
    )
    for role, directory, score in (("baseline", base, 0.3), ("trained", trained, 0.1)):
        record = protocol(role)
        record.update(
            manifest=manifest.digest,
            training_run=str(training),
            training_identity=digest(train_protocol),
        )
        write_json(
            metadata_dir(directory) / "run.json",
            {"protocol": record, "attempts": [attempt]},
        )
        stream = directory / "monitors/file/traces/stream"
        stream.mkdir(parents=True)
        rows = [
            episode(eid=role + str(i), score=score, ok=i == 0, manifest=manifest)
            for i in range(2)
        ]
        (stream / "00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    output = tmp_path / "reports"
    result = build_report(path, base, trained, training, output, gpu_hour_rate=2)
    assert result["success"]["improvement"] == 0.5
    assert result["gpu"]["status"] == "not_measured"
    assert result["training"]["status"] == "not_measured"
    assert (
        result["qualification"]["verifier_reference_and_repeatability"]
        == "not_measured"
    )
    assert result["resource_consumption"]["billed_cost"] is None
    estimate = result["resource_consumption"]["allocation_cost_estimate"]
    assert estimate is None if slurm else estimate["cost"] == 16
    assert (output / "qualification.md").is_file()
    assert json.loads((output / "qualification.json").read_text())["tasks"] == 1
    assert "test,0.0,0.5" in (output / "tasks.csv").read_text()
    archived = base / "monitors/file.attempt_1/traces/stream"
    archived.mkdir(parents=True)
    (archived / "00000.jsonl").write_text('{"id":"old", "ok":false}\n')
    with pytest.raises(ValueError, match="replaced failed"):
        build_report(path, base, trained, training, output)
