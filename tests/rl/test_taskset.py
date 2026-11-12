"""Run against the pinned, CPU-installable real Verifiers API."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

vf = pytest.importorskip(
    "verifiers.v1", reason="Install the pinned RL contract environment"
)

from finpipe_rl.manifest import freeze, load_manifest
from finpipe_rl.taskset import FinpipeConfig, FinpipeTaskset


def test_native_loader_isolated_and_selects_only_requested_split(catalog, tmp_path):
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    tasks = list(FinpipeTaskset(FinpipeConfig(manifest=out, split="test")))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.data.id == "test"
    assert task.data.verifier is not None
    assert task.scoring.mode == "isolated"
    assert task.data.image == catalog["tasks"][-1]["image"]
    assert task.data.verifier_image == catalog["tasks"][-1]["verifier_image"]
    assert task.config.reward_key == "rmse"
    assert task.config.rewards["solved"].weight == -1
    assert task.data.timeout.agent == 900
    assert task.data.timeout.scoring == 300
    assert task.data.agent_user == "agent"
    assert load_manifest(out).digest == task.data.finpipe_manifest_sha256
    assert (
        task.deployment_spec() is None
    )  # build-only Compose is already pinned as an image


def test_loader_discovery_and_native_scoring(catalog, tmp_path):
    from verifiers.v1.utils.loaders import taskset_class

    assert taskset_class("finpipe-rl") is FinpipeTaskset
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    task = next(iter(FinpipeTaskset(FinpipeConfig(manifest=out))))
    metrics = {}

    class Runtime:
        async def run(self, command, *args, **kwargs):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def read(self, path, **kwargs):
            return b'{"rmse": 0.125, "mae": 0.1}'

    trace = SimpleNamespace(info={}, record_metrics=metrics.update)
    assert asyncio.run(task.run_verifier(Runtime(), trace)) == {"rmse": 0.125}
    assert metrics == {"mae": 0.1}


def test_native_invalid_reward_fails(catalog, tmp_path):
    from verifiers.v1.errors import TaskError

    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    task = next(iter(FinpipeTaskset(FinpipeConfig(manifest=out))))

    class Runtime:
        async def run(self, *args, **kwargs):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def read(self, *args, **kwargs):
            return b'{"rmse": NaN}'

    with pytest.raises(TaskError, match="invalid"):
        asyncio.run(task.run_verifier(Runtime(), SimpleNamespace(info={})))


def test_nonzero_exit_cannot_supply_a_training_reward(catalog, tmp_path):
    from verifiers.v1.errors import TaskError

    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    task = next(iter(FinpipeTaskset(FinpipeConfig(manifest=out))))

    class Runtime:
        async def run(self, command, *args, **kwargs):
            return SimpleNamespace(
                exit_code=1 if "bash" in command else 0, stdout="", stderr="failed"
            )

        async def read(self, *args, **kwargs):
            return b'{"rmse": 0.0}'

    with pytest.raises(TaskError, match="exit"):
        asyncio.run(
            task.run_verifier(
                Runtime(), SimpleNamespace(info={}, record_metrics=lambda v: None)
            )
        )


def test_worker_setup_and_reward_contract_survive_reconstruction(catalog, tmp_path):
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    config = FinpipeConfig(manifest=out)
    taskset = FinpipeTaskset(config)
    cls = taskset.task_type()
    cls.toolsets(config.task)  # Called by native Env.serving before any rollout.
    task = next(iter(taskset))
    data = cls.data_type().model_validate_json(task.data.model_dump_json())
    rebuilt = cls(data, config.task)  # Native worker reconstructs with shared config.
    assert rebuilt.config.reward_key == "rmse"
    assert rebuilt.config.rewards["solved"].weight == -1
    env = vf.SingleAgentEnv(
        vf.SingleAgentEnvConfig.model_validate(
            {
                "taskset": {"id": "finpipe-rl", "manifest": str(out)},
                "agent": {"harness": {"id": "bash"}, "runtime": {"type": "docker"}},
            }
        )
    )
    assert not env._requires_tunnel({})


def test_preserves_runtime_compose_services(catalog, tmp_path):
    path = Path(catalog["tasks"][0]["path"]) / "environment/docker-compose.yaml"
    path.write_text(
        "services:\n  main:\n    image: solver\n  database:\n    image: postgres\n"
    )
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    task = next(iter(FinpipeTaskset(FinpipeConfig(manifest=out))))
    assert task.deployment_spec() is not None


def test_rejects_unsupported_verifier_user(catalog, tmp_path):
    path = Path(catalog["tasks"][0]["path"]) / "task.toml"
    path.write_text(path.read_text().replace('user = "root"', 'user = "nobody"'))
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    with pytest.raises(ValueError, match="root verifier"):
        list(FinpipeTaskset(FinpipeConfig(manifest=out)))
