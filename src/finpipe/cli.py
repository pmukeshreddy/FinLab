"""Command-line dispatch; composing configuration never starts external services."""

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None, config_path="resources/config", config_name="pipeline_config"
)
def batch(cfg: DictConfig) -> None:
    from finpipe.app.workflow import run_workflow

    run_workflow(
        cfg, Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir), "batch"
    )


@hydra.main(
    version_base=None,
    config_path="resources/config",
    config_name="manual_pipeline_config",
)
def paper(cfg: DictConfig) -> None:
    from finpipe.app.workflow import run_workflow

    run_workflow(
        cfg, Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir), "paper"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="finpipe")
    parser.add_argument("command", choices=("batch", "paper"))
    arguments = parser.parse_args(sys.argv[1:2])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        {"batch": batch, "paper": paper}[arguments.command]()
    finally:
        sys.argv = original_argv
