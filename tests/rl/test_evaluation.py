import copy

import pytest
from finpipe_rl.evaluation import compare, normalize_episode
from finpipe_rl.manifest import freeze


def episode(task="test", eid="one", score=0.1, ok=True, *, manifest):
    return {
        "id": eid,
        "ok": ok,
        "task": {"data": {"id": task, "finpipe_manifest_sha256": manifest.digest}},
        "errors": [],
        "traces": [
            {
                "metrics": {},
                "rewards": {"rmse": {"score": score, "weight": -1}},
                "info": {"scoring_diagnostics": {"harbor": {"exit_code": 0}}},
            }
        ],
    }


def test_native_reward_is_raw_not_weighted(catalog, tmp_path):
    manifest = freeze(catalog, tmp_path / "manifest.json")
    result = normalize_episode(episode(manifest=manifest), manifest.selected("test")[0])
    assert result["metrics"] == {"rmse": 0.1}
    assert result["success"] == 1


def test_failed_rollout_stays_in_success_denominator(catalog, tmp_path):
    manifest = freeze(catalog, tmp_path / "manifest.json")
    base = [
        episode(manifest=manifest, eid="b1", score=0.3),
        episode(manifest=manifest, eid="b2", ok=False),
    ]
    trained = [
        episode(manifest=manifest, eid="t1"),
        episode(manifest=manifest, eid="t2", ok=False),
    ]
    report = compare(manifest, base, trained, repeats=2)
    assert report["success"]["baseline"] == 0
    assert report["success"]["trained"] == 0.5
    assert report["success"]["improvement"] == 0.5
    assert report["success"]["ci95"] is None  # only one independent group
    assert report["metrics"]["rmse"]["improvement"] == pytest.approx(0.2)
    assert report["metrics"]["rmse"]["paired_tasks"] == 1
    assert report["reliability"]["trained"]["failed_episodes"] == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "duplicate",
        "nonfinite",
        "unscored",
        "multitrace",
        "wrong_manifest",
    ],
)
def test_rejects_invalid_comparison(catalog, tmp_path, kind):
    manifest = freeze(catalog, tmp_path / "manifest.json")
    base, trained = (
        [episode(manifest=manifest, eid="base")],
        [episode(manifest=manifest, eid="trained")],
    )
    if kind == "missing":
        trained.clear()
    elif kind == "extra":
        trained[0]["task"]["data"]["id"] = "train"
    elif kind == "duplicate":
        trained *= 2
    elif kind == "nonfinite":
        trained[0]["traces"][0]["rewards"]["rmse"]["score"] = float("inf")
    elif kind == "unscored":
        trained[0]["traces"][0]["rewards"] = {}
    elif kind == "wrong_manifest":
        trained[0]["task"]["data"]["finpipe_manifest_sha256"] = "stale"
    else:
        trained[0]["traces"] *= 2
    with pytest.raises(ValueError):
        compare(manifest, base, trained, repeats=1)


def test_clustered_ci_is_reproducible(catalog, tmp_path):
    import shutil
    from pathlib import Path

    for n in range(8):
        extra = copy.deepcopy(catalog["tasks"][-1])
        extra.update(
            id=f"heldout{n}",
            path=str(tmp_path / f"heldout{n}"),
            groups=[f"paper:group{n // 2}", f"dataset:group{n // 2}"],
        )
        shutil.copytree(catalog["tasks"][-1]["path"], extra["path"])
        (Path(extra["path"]) / "instruction.md").write_text(f"Unique {n}")
        catalog["tasks"].append(extra)
    manifest = freeze(catalog, tmp_path / "manifest.json")
    base = [
        episode(t.id, f"b-{t.id}", 0.3, manifest=manifest)
        for t in manifest.selected("test")
    ]
    trained = [
        episode(t.id, f"t-{t.id}", 0.1 if i < 4 else 0.4, manifest=manifest)
        for i, t in enumerate(manifest.selected("test"))
    ]
    one = compare(manifest, base, trained, repeats=1, bootstrap_samples=500)
    assert one == compare(manifest, base, trained, repeats=1, bootstrap_samples=500)
    assert one["independent_groups"] == 5
    assert (
        one["success"]["ci95"][0]
        <= one["success"]["improvement"]
        <= one["success"]["ci95"][1]
    )


def test_nonzero_verifier_exit_counts_as_failure(catalog, tmp_path):
    manifest = freeze(catalog, tmp_path / "manifest.json")
    row = episode(manifest=manifest)
    row["traces"][0]["info"]["scoring_diagnostics"]["harbor"]["exit_code"] = 1
    normalized = normalize_episode(row, manifest.selected("test")[0])
    assert normalized["success"] == 0
    assert normalized["failed"]
