import copy
import json
from pathlib import Path

import pytest
from finpipe_rl.manifest import freeze, load_manifest


def test_freeze_detects_mutation_and_is_deterministic(catalog, tmp_path):
    out = tmp_path / "manifest.json"
    frozen = freeze(catalog, out)
    assert load_manifest(out).digest == frozen.digest
    with pytest.raises(FileExistsError):
        freeze(catalog, out)
    task = Path(catalog["tasks"][0]["path"])
    (task / "tests/verifier.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(out)


@pytest.mark.parametrize("kind", ["group", "prompt", "id", "path"])
def test_rejects_leakage(catalog, tmp_path, kind):
    a, _, b = catalog["tasks"]
    if kind == "group":
        b["groups"].append(a["groups"][0])
    elif kind == "prompt":
        (Path(b["path"]) / "instruction.md").write_text(" Solve   train ")
    else:
        b[kind] = a[kind]
    with pytest.raises(ValueError):
        freeze(catalog, tmp_path / "manifest.json")


@pytest.mark.parametrize(
    "change",
    [
        {"image": "registry/solver:latest"},
        {"groups": []},
        {"success_threshold": float("nan")},
        {"reward_key": "missing"},
        {"success_key": "missing"},
        {"split": "typo"},
    ],
)
def test_invalid_contract(catalog, tmp_path, change):
    catalog["tasks"][0].update(change)
    with pytest.raises(ValueError):
        freeze(catalog, tmp_path / "manifest.json")


def test_rejects_symlinks_and_unisolated_verifier(catalog, tmp_path):
    task = Path(catalog["tasks"][0]["path"])
    link = task / "external"
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        freeze(catalog, tmp_path / "manifest.json")
    link.unlink()
    (task / "task.toml").write_text('schema_version = "1.4"\n')
    with pytest.raises(ValueError, match="separate"):
        freeze(catalog, tmp_path / "manifest.json")


def test_tampered_manifest_rejected(catalog, tmp_path):
    out = tmp_path / "manifest.json"
    freeze(catalog, out)
    data = json.loads(out.read_text())
    data["tasks"][0]["success_threshold"] = 999
    out.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        load_manifest(out)


def test_shared_provenance_within_split_is_allowed(catalog, tmp_path):
    extra = copy.deepcopy(catalog["tasks"][-1])
    extra["id"] = "test2"
    extra["path"] = str(tmp_path / "test2")
    import shutil

    shutil.copytree(catalog["tasks"][-1]["path"], extra["path"])
    (Path(extra["path"]) / "instruction.md").write_text("A related task")
    catalog["tasks"].append(extra)
    assert len(freeze(catalog, tmp_path / "manifest.json").tasks) == 4
