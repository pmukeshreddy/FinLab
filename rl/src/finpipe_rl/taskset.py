"""Local task references adapted to Verifiers' native Harbor task implementation."""

import os
import shutil
import tempfile
import tomllib
from pathlib import Path

import tomli_w
import verifiers.v1 as vf
import yaml
from verifiers.v1.errors import TaskError
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTaskConfig,
    parse_task,
)

from finpipe_rl.manifest import Split, load_manifest


class FinpipeConfig(vf.TasksetConfig):
    task: HarborTaskConfig = HarborTaskConfig()
    manifest: Path = Path("rl/manifest.json")
    split: Split = "train"


class FinpipeData(HarborData):
    finpipe_manifest_sha256: str
    finpipe_reward_key: str
    finpipe_reward_direction: str
    finpipe_build_only_compose: bool = False


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def parse_local_task(spec, idx):
    """A temporary metadata view bridges unsupported, redundant root-user syntax.

    Native parsing remains authoritative. Only temporary metadata changes; all
    original task files and verifier scripts remain unchanged.
    """
    config = tomllib.loads((spec.path / "task.toml").read_text())
    verifier = config["verifier"]
    if verifier.get("user") not in (None, "root", "0", 0):
        raise ValueError("The native runtime only supports the default root verifier")
    verifier.pop("user", None)
    config.setdefault("environment", {})["docker_image"] = spec.image
    verifier.setdefault("environment", {})["docker_image"] = spec.verifier_image
    with tempfile.TemporaryDirectory(prefix="finpipe-harbor-") as directory:
        view = Path(directory)
        for entry in spec.path.iterdir():
            if entry.name != "task.toml":
                # Harbor rejects external symlinks. Copy for parsing only; the
                # runtime uses the original frozen directory restored below.
                if entry.is_dir():
                    shutil.copytree(
                        entry, view / entry.name, copy_function=link_or_copy
                    )
                else:
                    link_or_copy(entry, view / entry.name)
        (view / "task.toml").write_text(tomli_w.dumps(config))
        data = parse_task(view, idx, HarborConfig(ignore_timeouts=False))
    return data.model_copy(update={"task_dir": str(spec.path)})


def build_only_compose(path):
    compose = path / "environment/docker-compose.yaml"
    if not compose.exists():
        return False
    config = yaml.safe_load(compose.read_text())
    services = config.get("services", {})
    return (
        set(config) <= {"services", "secrets"}
        and set(services) == {"main"}
        and set(services["main"]) <= {"build", "image"}
    )


class FinpipeTask(HarborTask, vf.Task[FinpipeData, vf.State, HarborTaskConfig]):
    def __init__(self, data, config=None):
        # Workers reconstruct tasks from wire data and shared taskset config.
        # Keep per-task reward selection in that data, not only the loader instance.
        config = HarborTaskConfig(
            reward_key=data.finpipe_reward_key,
            rewards={
                "solved": {
                    "weight": 1.0
                    if data.finpipe_reward_direction == "maximize"
                    else -1.0
                }
            },
        )
        super().__init__(data, config)

    def deployment_spec(self):
        # The exporter uses Compose only to BUILD one image with build secrets.
        # That image is already pinned; no runtime service is being removed.
        if self.data.finpipe_build_only_compose:
            return None
        return super().deployment_spec()

    async def run_verifier(self, runtime, trace):
        scores = await super().run_verifier(runtime, trace)
        if trace.info["scoring_diagnostics"]["harbor"]["exit_code"] != 0:
            raise TaskError("Verifier exited with a nonzero exit code; reward rejected")
        return scores


class FinpipeTaskset(vf.Taskset[FinpipeTask, FinpipeConfig]):
    def load(self):
        manifest = load_manifest(self.config.manifest)
        for idx, spec in enumerate(manifest.selected(self.config.split)):
            data = parse_local_task(spec, idx)
            if data.verifier is None:
                raise ValueError(f"Missing separate verifier: {spec.id}")
            data = data.model_copy(
                update={
                    "id": spec.id,
                    "name": spec.id,
                    "image": spec.image,
                    "verifier_image": spec.verifier_image,
                    "upload_environment": False,
                    "compose_build_dockerfile": False,
                }
            )
            data = FinpipeData.model_validate(
                {
                    **data.model_dump(),
                    "finpipe_manifest_sha256": manifest.digest,
                    "finpipe_reward_key": spec.reward_key,
                    "finpipe_reward_direction": spec.metrics[spec.reward_key],
                    "finpipe_build_only_compose": build_only_compose(spec.path),
                }
            )
            yield FinpipeTask(data)
