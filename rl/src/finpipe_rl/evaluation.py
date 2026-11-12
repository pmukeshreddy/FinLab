"""Task-macro evaluation with paired, provenance-clustered uncertainty."""

import math
import random
import statistics
from collections import defaultdict

from finpipe_rl.manifest import Manifest, TaskSpec


def finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Expected a finite numeric metric, got {value!r}")
    return float(value)


def normalize_episode(episode: dict, task: TaskSpec) -> dict:
    if type(episode.get("ok")) is not bool or not episode.get("id"):
        raise ValueError("Episode needs native id and ok fields")
    traces = episode.get("traces", [])
    if len(traces) > 1:
        raise ValueError("Expected the configured single-agent, single-trace harness")
    trace = traces[0] if traces else {}
    scores = {
        k: finite(v) for k, v in trace.get("metrics", {}).items() if v is not None
    }
    for key, reward in trace.get("rewards", {}).items():
        if reward is not None:
            if key in scores:
                raise ValueError(f"Ambiguous duplicate metric {key}")
            scores[key] = finite(reward["score"])
    diagnostic = trace.get("info", {}).get("scoring_diagnostics", {}).get("harbor", {})
    verifier_error = diagnostic.get("exit_code") not in (None, 0)
    failed = not episode["ok"] or verifier_error or bool(trace.get("error"))
    if not failed and (not traces or not set(task.metrics).issubset(scores)):
        raise ValueError(f"Completed episode lacks declared metrics: {task.id}")
    timeout = bool(episode.get("is_timeout") or trace.get("is_timeout"))
    failed = failed or timeout
    return {
        "id": episode["id"],
        "task": task.id,
        "failed": failed,
        "timeout": timeout,
        "verifier_error": verifier_error,
        "errors": len(episode.get("errors", [])),
        "metrics": {} if failed else scores,
        "success": int(not failed and task.succeeds(scores)),
    }


def clusters(tasks):
    """Connected provenance groups, including transitive paper/dataset overlap."""
    parent = {task.id: task.id for task in tasks}

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    groups = {}
    for task in tasks:
        for group in task.groups:
            if group in groups:
                parent[root(task.id)] = root(groups[group])
            groups[group] = task.id
    result = defaultdict(list)
    for task in tasks:
        result[root(task.id)].append(task.id)
    return list(result.values())


def interval(values, groups, *, samples, seed):
    selected = [[values[t] for t in group if t in values] for group in groups]
    selected = [g for g in selected if g]
    if len(selected) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draw = [
            value for group in rng.choices(selected, k=len(selected)) for value in group
        ]
        draws.append(statistics.fmean(draw))
    draws.sort()
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def paired_summary(baseline, trained, groups, samples, seed, direction="maximize"):
    keys = baseline.keys() & trained.keys()
    if not keys:
        return {
            "baseline": None,
            "trained": None,
            "improvement": None,
            "ci95": None,
            "paired_tasks": 0,
        }
    sign = 1 if direction == "maximize" else -1
    changes = {key: sign * (trained[key] - baseline[key]) for key in sorted(keys)}
    return {
        "baseline": statistics.fmean(baseline[k] for k in sorted(keys)),
        "trained": statistics.fmean(trained[k] for k in sorted(keys)),
        "improvement": statistics.fmean(changes.values()),
        "ci95": interval(changes, groups, samples=samples, seed=seed),
        "baseline_ci95": interval(
            {k: baseline[k] for k in keys}, groups, samples=samples, seed=seed
        ),
        "trained_ci95": interval(
            {k: trained[k] for k in keys}, groups, samples=samples, seed=seed
        ),
        "paired_tasks": len(keys),
    }


def compare(
    manifest: Manifest,
    baseline,
    trained,
    *,
    repeats: int,
    bootstrap_samples=10000,
    seed=42,
):
    if repeats < 1 or bootstrap_samples < 100:
        raise ValueError("Positive repeats and at least 100 bootstrap samples required")
    tasks = {t.id: t for t in manifest.selected("test")}
    groups = clusters(list(tasks.values()))
    arms, reliability = {}, {}
    for name, episodes in (("baseline", baseline), ("trained", trained)):
        rows, seen = defaultdict(list), set()
        for ep in episodes:
            task_id = ep.get("task", {}).get("data", {}).get("id")
            if task_id not in tasks:
                raise ValueError(f"Unexpected task in held-out evaluation: {task_id}")
            if ep["task"]["data"].get("finpipe_manifest_sha256") != manifest.digest:
                raise ValueError("Episode task data does not match the frozen manifest")
            row = normalize_episode(ep, tasks[task_id])
            if row["id"] in seen:
                raise ValueError("Duplicate episode id")
            seen.add(row["id"])
            rows[task_id].append(row)
        if set(rows) != set(tasks) or any(len(rows[t]) != repeats for t in tasks):
            raise ValueError(
                "Incomplete or excess evaluation coverage; every task needs equal repeats"
            )
        flat = [r for values in rows.values() for r in values]
        reliability[name] = {
            "episodes": len(flat),
            "failed_episodes": sum(r["failed"] for r in flat),
            "timeout_episodes": sum(r["timeout"] for r in flat),
            "verifier_error_episodes": sum(r["verifier_error"] for r in flat),
            "recorded_episode_errors": sum(r["errors"] for r in flat),
        }
        arms[name] = rows
    success = {
        name: {t: statistics.fmean(r["success"] for r in rows[t]) for t in tasks}
        for name, rows in arms.items()
    }
    metric_directions = {k: v for t in tasks.values() for k, v in t.metrics.items()}
    metrics = {}
    for key, direction in sorted(metric_directions.items()):
        scores = {}
        coverage = {}
        for name, rows in arms.items():
            scores[name], coverage[name] = {}, 0
            for task_id, entries in rows.items():
                if key not in tasks[task_id].metrics:
                    continue
                values = [r["metrics"][key] for r in entries if key in r["metrics"]]
                coverage[name] += len(values)
                if values:
                    scores[name][task_id] = statistics.fmean(values)
        metrics[key] = {
            "direction": direction,
            "scored_episodes": coverage,
            **paired_summary(
                scores["baseline"],
                scores["trained"],
                groups,
                bootstrap_samples,
                seed,
                direction,
            ),
        }
    breakdown = []
    for task_id, spec in tasks.items():
        row = {
            "task": task_id,
            "groups": spec.groups,
            "success_key": spec.success_key,
            "success_threshold": spec.success_threshold,
        }
        for name in arms:
            scored = arms[name][task_id]
            row[name] = {
                "success_rate": success[name][task_id],
                "failures": sum(r["failed"] for r in scored),
                "metrics": {
                    key: [r["metrics"].get(key) for r in scored] for key in spec.metrics
                },
            }
        breakdown.append(row)
    return {
        "tasks": len(tasks),
        "repeats": repeats,
        "independent_groups": len(groups),
        "ci_method": "paired percentile bootstrap of independent provenance clusters; task-macro means",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "success": paired_summary(
            success["baseline"], success["trained"], groups, bootstrap_samples, seed
        ),
        "metrics": metrics,
        "reliability": reliability,
        "task_breakdown": breakdown,
        "limitations": [
            "Metric means use scored episodes on tasks observed in both arms; missing metrics are not imputed.",
            "Success includes all planned episodes; execution failures count as unsuccessful.",
            "Intervals describe this task sample, conditional on the fitted model; they do not measure training-seed variation.",
            "Small or degenerate cluster samples may underestimate uncertainty; collect more independent tasks.",
        ],
    }
