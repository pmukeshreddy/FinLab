import copy
import tomllib
from pathlib import Path

import pytest
from finpipe_rl.manifest import freeze
from finpipe_rl.runtime import prepare_config, resume_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def prepared(catalog, tmp_path):
    manifest = tmp_path / "manifest.json"
    freeze(catalog, manifest)
    raw = tomllib.loads((ROOT / "rl/configs/qwen3.8-27b.toml").read_text())
    return prepare_config("train", raw, manifest, tmp_path / "run", "model")


def test_upstream_configuration_contract(prepared, tmp_path):
    pytest.importorskip("verifiers.v1")
    from prime_rl.configs.eval import EvalConfig
    from prime_rl.configs.rl import RLConfig

    parsed = RLConfig.model_validate(prepared)
    assert parsed.orchestrator.train.source[0].env.taskset.split == "train"
    assert parsed.orchestrator.eval.source[0].env.taskset.split == "validation"
    assert parsed.trainer.model.lora.rank == 16
    assert parsed.inference.vllm.enable_lora
    assert parsed.trainer.ckpt.interval == 10
    assert parsed.trainer.model.conversion_dir == tmp_path / "run/model_conversion"
    assert parsed.trainer.monitors.prime is None
    raw = tomllib.loads((ROOT / "rl/configs/eval.toml").read_text())
    data = prepare_config(
        "eval", raw, tmp_path / "manifest.json", tmp_path / "eval", "model"
    )
    assert EvalConfig.model_validate(data).source[0].env.taskset.split == "test"


def test_resume_does_not_change_experiment(prepared):
    updated = copy.deepcopy(prepared)
    updated["max_steps"] += 20
    assert resume_config(prepared, updated, step=10)["resume"] == {"step": 10}
    updated["trainer"]["optim"]["lr"] *= 2
    with pytest.raises(ValueError, match="changed"):
        resume_config(prepared, updated, step=10)


def test_resume_requires_a_checkpoint_and_does_not_restart(prepared):
    with pytest.raises(ValueError, match="checkpoint"):
        resume_config(prepared, prepared, step=None)


def test_native_distributed_topology_contract(prepared, tmp_path):
    pytest.importorskip("verifiers.v1")
    from prime_rl.configs.rl import RLConfig

    prepared["deployment"] = {
        "type": "multi_node",
        "num_train_nodes": 2,
        "num_infer_nodes": 1,
        "gpus_per_node": 8,
    }
    prepared["slurm"] = {"project_dir": str(tmp_path)}
    parsed = RLConfig.model_validate(prepared)
    assert parsed.deployment.num_train_nodes == 2
    assert parsed.deployment.num_infer_nodes == 1
    assert parsed.inference.deployment.type == "single_node"


def test_native_audit_configuration(catalog, tmp_path, monkeypatch):
    pytest.importorskip("verifiers.v1")
    import json

    from finpipe_rl import cli

    manifest = tmp_path / "manifest.json"
    freeze(catalog, manifest)
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    cli.main(
        [
            "audit",
            "--manifest",
            str(manifest),
            "--split",
            "test",
            "--output",
            str(tmp_path / "audit"),
        ]
    )
    data = json.loads(Path(calls[0][-1]).read_text())
    assert data["taskset"]["split"] == "test"
    assert data["runtime"]["type"] == "docker"
    assert calls[0][0].endswith("vf-validate")


def test_rejects_test_leakage_and_partial_eval(catalog, tmp_path):
    manifest = tmp_path / "manifest.json"
    freeze(catalog, manifest)
    raw = tomllib.loads((ROOT / "rl/configs/qwen3.8-27b.toml").read_text())
    raw["orchestrator"]["train"]["source"][0]["env"]["taskset"]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        prepare_config("train", raw, manifest, tmp_path / "train", "model")
    raw = tomllib.loads((ROOT / "rl/configs/eval.toml").read_text())
    raw["select"] = {"limit": 1}
    with pytest.raises(ValueError, match="selection"):
        prepare_config("eval", raw, manifest, tmp_path / "eval", "model")


def test_joint_checkpoint_selection_ignores_incomplete_newest(tmp_path):
    from finpipe_rl.runtime import complete_checkpoint

    for step in (1, 2, 3):
        path = tmp_path / "checkpoints" / f"step_{step}"
        (path / "trainer").mkdir(parents=True)
        (path / "orchestrator").mkdir()
        if step != 3:
            (path / "trainer/.metadata").write_bytes(b"committed")
        if step != 2:
            (path / "orchestrator/progress.pt").write_bytes(b"committed")
    assert complete_checkpoint(tmp_path) == 1


@pytest.mark.parametrize(
    "override",
    [
        "group_prompt",
        "source_group_size",
        "source_algo",
        "skip_optimizer",
        "component_resume",
        "init_from",
        "component_model",
        "checkpoint_directory",
        "fake_data",
        "debug_layers",
        "conversion_directory",
    ],
)
def test_rejects_hidden_framework_overrides(catalog, tmp_path, override):
    manifest = tmp_path / "manifest.json"
    freeze(catalog, manifest)
    raw = tomllib.loads((ROOT / "rl/configs/qwen3.8-27b.toml").read_text())
    source = raw["orchestrator"]["train"]["source"][0]
    if override == "group_prompt":
        raw["orchestrator"]["train"]["env"]["taskset"] = {
            "system_prompt": "/outside/prompt.txt"
        }
    elif override == "source_group_size":
        source["group_size"] = 1
    elif override == "source_algo":
        source["algo"] = {"type": "sft"}
    elif override == "skip_optimizer":
        raw["trainer"]["ckpt"] = {"skip_optimizer": True}
    elif override == "component_resume":
        raw["trainer"]["resume"] = raw["orchestrator"]["resume"] = {"step": 10}
    elif override == "init_from":
        raw["trainer"]["model"]["init_from"] = "/outside/checkpoint"
    elif override == "component_model":
        raw["trainer"]["model"]["name"] = "other/model"
    elif override == "fake_data":
        raw["trainer"]["data"] = {"fake": {}}
    elif override == "debug_layers":
        raw["trainer"]["model"]["debug"] = {"num_layers": 1}
    elif override == "conversion_directory":
        raw["trainer"]["model"]["conversion_dir"] = "/outside/base"
    else:
        raw["ckpt"]["output_dir"] = "/outside/checkpoints"
    with pytest.raises(ValueError):
        prepare_config("train", raw, manifest, tmp_path / "train", "model")


def test_launch_records_failure_and_resumes_same_native_run(
    prepared, tmp_path, monkeypatch
):
    pytest.importorskip("verifiers.v1")
    import json
    import subprocess

    from finpipe_rl import runtime

    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model.safetensors").write_bytes(b"unit fixture, not real weights")
    calls = []

    def fake_process(command, **kwargs):
        calls.append(json.loads(Path(command[-1]).read_text()))
        return subprocess.CompletedProcess(command, 1 if len(calls) == 1 else 0)

    monkeypatch.setattr(runtime, "check_framework", lambda root: {"pinned": True})
    monkeypatch.setattr(runtime.subprocess, "run", fake_process)
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    run = tmp_path / "run"
    kwargs = {
        "kind": "train",
        "config": prepared,
        "manifest_path": tmp_path / "manifest.json",
        "artifact": artifact,
        "run_dir": run,
        "prime_root": tmp_path,
    }
    with pytest.raises(subprocess.CalledProcessError):
        runtime.launch(**kwargs)
    receipt = json.loads((runtime.metadata_dir(run) / "run.json").read_text())
    assert receipt["attempts"][-1]["status"] == "failed"
    checkpoint = run / "checkpoints/step_10"
    (checkpoint / "trainer").mkdir(parents=True)
    (checkpoint / "orchestrator").mkdir()
    (checkpoint / "trainer/.metadata").touch()
    (checkpoint / "orchestrator/progress.pt").touch()
    receipt = runtime.launch(**kwargs, resume=True)
    assert calls[1]["resume"] == {"step": 10}
    assert calls[1]["run"] == calls[0]["run"]
    assert [a["status"] for a in receipt["attempts"]] == ["failed", "completed"]
    assert "resume" not in prepared
