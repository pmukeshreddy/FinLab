"""Freeze references to existing tasks; never generate or modify task contents."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Split = Literal["train", "validation", "test"]
Direction = Literal["maximize", "minimize"]


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def task_files(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise ValueError(f"Task directory cannot be a symlink: {path}")
    required = ["task.toml", "instruction.md", "tests/test.sh"]
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"Missing {path / name}")
    config = tomllib.loads((path / "task.toml").read_text())
    if config.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError(f"Task {path} must retain a separate verifier")
    files = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ValueError(f"Task contains a symlink: {entry}")
        if entry.is_file():
            files[entry.relative_to(path).as_posix()] = file_hash(entry)
    return files


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    path: Path
    split: Split
    groups: list[str] = Field(min_length=2)
    image: str = Field(pattern=r"^.+@sha256:[a-f0-9]{64}$")
    verifier_image: str = Field(pattern=r"^.+@sha256:[a-f0-9]{64}$")
    reward_key: str
    success_key: str
    success_threshold: FiniteFloat
    metrics: dict[str, Direction]
    files: dict[str, str] = Field(default_factory=dict)
    prompt_hash: str = ""

    @model_validator(mode="after")
    def validate_contract(self):
        if self.reward_key not in self.metrics or self.success_key not in self.metrics:
            raise ValueError(
                "Reward and success keys must have declared metric directions"
            )
        for prefix in ("paper:", "dataset:"):
            if not any(
                g.startswith(prefix) and len(g) > len(prefix) for g in self.groups
            ):
                raise ValueError(f"Task provenance requires a nonempty {prefix} group")
        if any(not g.strip() or g != g.strip() for g in self.groups):
            raise ValueError("Provenance groups must be nonempty and trimmed")
        return self

    def succeeds(self, values: dict[str, float]) -> bool:
        value = values[self.success_key]
        return (
            value >= self.success_threshold
            if self.metrics[self.success_key] == "maximize"
            else value <= self.success_threshold
        )


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    tasks: list[TaskSpec] = Field(min_length=3)
    digest: str = ""

    @model_validator(mode="after")
    def validate_splits(self):
        if {t.split for t in self.tasks} != {"train", "validation", "test"}:
            raise ValueError("Provide nonempty train, validation, and test splits")
        ids, paths, groups, prompts = set(), set(), {}, {}
        directions = {}
        for task in self.tasks:
            if task.id in ids or task.path in paths:
                raise ValueError("Duplicate task identity or path")
            ids.add(task.id)
            paths.add(task.path)
            for group in task.groups:
                if group in groups and groups[group] != task.split:
                    raise ValueError(f"Provenance leakage across splits: {group}")
                groups[group] = task.split
            if task.prompt_hash:
                if task.prompt_hash in prompts:
                    raise ValueError("Duplicate normalized task prompt")
                prompts[task.prompt_hash] = task.id
            for key, direction in task.metrics.items():
                if key in directions and directions[key] != direction:
                    raise ValueError(f"Conflicting metric direction: {key}")
                directions[key] = direction
        return self

    def selected(self, split: Split) -> list[TaskSpec]:
        return [t for t in self.tasks if t.split == split]


def freeze(catalog: dict, destination: Path) -> Manifest:
    data = Manifest.model_validate(catalog).model_dump(mode="json")
    for task in data["tasks"]:
        path = Path(task["path"]).expanduser()
        task["files"] = task_files(path)
        task["path"] = str(path.resolve())
        prompt = " ".join((path / "instruction.md").read_text().split())
        if not prompt:
            raise ValueError("Empty task instruction")
        task["prompt_hash"] = digest(prompt)
    manifest = Manifest.model_validate(data)
    manifest.digest = digest(manifest.model_dump(mode="json", exclude={"digest"}))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a frozen experimental split.
    with destination.open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def load_manifest(path: Path) -> Manifest:
    manifest = Manifest.model_validate_json(path.read_text())
    if manifest.digest != digest(manifest.model_dump(mode="json", exclude={"digest"})):
        raise ValueError("Manifest digest mismatch")
    for task in manifest.tasks:
        if not task.files or task_files(task.path) != task.files:
            raise ValueError(f"Task contents changed: {task.id}")
    return manifest
