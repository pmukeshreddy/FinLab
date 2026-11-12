from pathlib import Path

import pytest
from jinja2 import Template


@pytest.fixture
def catalog(tmp_path):
    tasks = []
    for name, split in [("train", "train"), ("val", "validation"), ("test", "test")]:
        path = tmp_path / name
        (path / "environment").mkdir(parents=True)
        (path / "tests").mkdir()
        (path / "instruction.md").write_text(f"Solve {name}")
        templates = (
            Path(__file__).resolve().parents[2] / "src/finpipe/resources/templates"
        )
        (path / "task.toml").write_text(
            Template((templates / "task.toml.j2").read_text()).render(
                org_name="fixture",
                task_name=name,
                description="Fixture",
                keywords=[],
                difficulty_explanation="Fixture",
            )
        )
        (path / "environment/docker-compose.yaml").write_text(
            (templates / "docker-compose.yaml.j2").read_text()
        )
        (path / "environment/Dockerfile").write_text("FROM busybox\n")
        (path / "tests/test.sh").write_text("echo test\n")
        (path / "tests/verifier.py").write_text("# existing verifier\n")
        tasks.append(
            {
                "id": name,
                "path": str(path),
                "split": split,
                "groups": [f"paper:{name}", f"dataset:{name}"],
                "image": "registry/solver@sha256:" + "a" * 64,
                "verifier_image": "registry/verifier@sha256:" + "b" * 64,
                "reward_key": "rmse",
                "success_key": "rmse",
                "success_threshold": 0.2,
                "metrics": {"rmse": "minimize"},
            }
        )
    return {"schema_version": 1, "tasks": tasks}
