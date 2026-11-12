"""Opt-in real GPU qualification; uses existing tasks and native optimizer state."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("FINPIPE_RL_GPU_CONFIG"),
    reason="Requires GPU allocation and FINPIPE_RL_GPU_CONFIG",
)
def test_real_training_checkpoint_recovery(tmp_path):
    import torch.distributed.checkpoint as dcp
    from finpipe_rl.cli import read_config
    from finpipe_rl.runtime import launch, metadata_dir, prepare_config, write_json

    # Config JSON points at an existing native config, frozen manifest, model snapshot,
    # and pinned Prime checkout. All are explicit; no synthetic task or verifier.
    inputs = json.loads(Path(os.environ["FINPIPE_RL_GPU_CONFIG"]).read_text())
    recipe = read_config(Path(inputs["config"]))
    recipe["max_steps"] = 2
    recipe["ckpt"] = {"interval": 1}
    recipe["log"] = {"level": "DEBUG"}
    manifest, artifact = Path(inputs["manifest"]), Path(inputs["artifact"])
    run = tmp_path / "gpu-recovery"
    config = prepare_config("train", recipe, manifest, run, str(artifact.resolve()))
    arguments = {
        "kind": "train",
        "config": config,
        "manifest_path": manifest,
        "artifact": artifact,
        "run_dir": run,
        "prime_root": Path(inputs["prime_root"]),
    }
    launch(**arguments)
    old = run / "checkpoints/step_2/trainer"
    metadata = dcp.FileSystemReader(old).read_metadata()
    for field in ("app.model.", "app.optimizers.", "app.scheduler.", "app.progress."):
        assert any(k.startswith(field) for k in metadata.state_dict_metadata)
    # Load only the native progress scalar without inventing a checkpoint format.
    state = {"app": {"progress": {"step": 0}}}
    dcp.load(state, checkpoint_id=old)
    assert state["app"]["progress"]["step"] == 2
    config["max_steps"] = 3
    launch(**arguments, resume=True)
    dcp.load(state, checkpoint_id=run / "checkpoints/step_3/trainer")
    assert state["app"]["progress"]["step"] == 3
    receipt = json.loads((metadata_dir(run) / "run.json").read_text())
    assert len(receipt["attempts"]) == 2
    write_json(
        metadata_dir(run) / "gpu-recovery.json",
        {
            "status": "passed",
            "scope": "native checkpoint reload and progress continuation; not bitwise equivalence or forced-kill qualification",
            "checkpoint_step": 2,
            "continued_step": 3,
        },
    )
