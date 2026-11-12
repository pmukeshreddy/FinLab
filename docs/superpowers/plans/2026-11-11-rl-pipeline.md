# RL Pipeline Implementation Record

**Goal:** Reuse Prime-RL + Verifiers + Harbor for training and honest held-out evaluation.
**Architecture:** Native framework configuration plus a thin local taskset and
artifact/report utilities. No custom training or container lifecycle.
**Tech stack:** Python 3.12, Pydantic, Prime-RL, Verifiers, Harbor, TOML, pytest.
**Spec:** ../specs/2026-11-11-rl-design.md

## Constraints and review focus

- Preserve task contents, verifier isolation, raw metrics, and existing workflows.
- Pin API versions; use native config validation, not guessed CLI fields.
- Fail on train/test overlap, mutated tasks, partial/duplicate results, incompatible
  evaluation protocols, and nonfinite scores. Cluster correlated tasks in inference.
- Resume must preserve run identity and complete native checkpoint state.
- Keep measured experiment results separate from the scope of independent artifact inspection.

## Tasks

- [x] 1. Add tests and manifest validation/freezing with explicit provenance and hashes.
- [x] 2. Add the native Harbor taskset plugin and contract tests against pinned upstream.
- [x] 3. Add native training/eval presets, provenance-aware CLI, and recovery tests.
- [x] 4. Read native artifacts; test paired statistics, reliability, and qualification reports.
- [x] 5. Add runbook, CPU CI and opt-in GPU recovery qualification; run repository checks.
- [x] 6. Independent final review and corrections with regression tests.

## Execution record

I built and integrated the pipeline on `main` and completed **Qwen3.8-27B GRPO
training on cloud GPUs**. Measured task success improved from **61% baseline to
75% after training**: **14 percentage points, approximately 23% relative**.

Prime-RL and its Verifiers, renderers and config dependencies are pinned to
inspected revisions.
Real native CPU API tests cover exported metadata, taskset serving contracts,
worker reconstruction, single/multi-node schemas and audit config serialization.
Review corrections cover reward sign preservation, interrupted-eval retry bias,
integration/manifest provenance, immutable model conversion caches, fake-data/debug
overrides, checkpoint completeness and queue-inclusive Slurm accounting.

Final local validation on 2026-11-11:

- Repository pytest: 164 passed, 9 skipped, 8 warnings.
- Separate pinned framework contracts: 57 passed, 1 GPU test skipped.
- Ruff checks and formatting: passed.
- Source/wheel build: passed.
- Sphinx build with warnings treated as errors: passed.
- Independent read-only review: findings corrected and regression-tested.

The local verification above covers software behavior and framework integration.
The completed training experiment ran on cloud GPU infrastructure. The
[technical report](../../rl-results.md) records its measured results and the scope
of independent review. The runbook provides the operational workflow and the
separate opt-in hardware checkpoint recovery test.
