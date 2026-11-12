import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize("mode", ["batch", "paper"])
def test_commands_compose_configuration_without_running_services(mode):
    result = subprocess.run(
        [sys.executable, "-m", "finpipe", mode, "--cfg", "job"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert config["enable_task_critic"] is False
    assert config["task_exporter"]["export_path"] == "./tasks/"
    if mode == "batch":
        assert config["db_engine"] == "sqlite:///./pipeline.db"
        assert len(config["scopes"]) == 5
    else:
        assert config["enable_judge"] is False


def test_single_paper_command_accepts_existing_hydra_overrides():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "finpipe",
            "paper",
            "--cfg",
            "job",
            "scopes=alpha_trading",
            "arxiv_id=1706.03762v7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    assert config["arxiv_id"] == "1706.03762v7"
    assert list(config["scopes"]) == ["alpha_trading"]
