# Task-Aware Financial Environments and GRPO Training

I built a pipeline that turns financial research papers into executable agent
environments, then connects those environments to RL training and evaluation.
**I completed Qwen3.8-27B GRPO training on cloud GPUs, improving task success from
61% to 75% across 100 held-out evaluation tasks: 14 more problems solved, or
+14 percentage points.**

The environment design addresses a specific failure mode: an agent can produce a
valid output file while violating the task's statistical requirements. A forecast
assigned to the wrong input window is incorrect even if its values look plausible.
Conversely, a synthetic return path need not reproduce a particular held-out path.
Scoring those samples row by row would penalize differences between unrelated
realizations instead of measuring distributional accuracy.

My design makes the meaning of an output explicit before generating the
environment or its verifier. The
[forecasting contract](../src/finpipe/contracts/task_types/forecasting.py) and
[generation contract](../src/finpipe/contracts/task_types/generation.py) encode
different required dataset roles and row-correspondence rules:

| Task | Agent output and data contract | Assessment semantics |
|---|---|---|
| Forecasting | Predictions conditioned on supplied test inputs; output rows correspond to the same rows in the withheld targets. | Compare aligned predictions and targets using the specified metric, such as MSE. |
| Unconditional return generation | Synthetic return windows sampled from a fitted model; training targets and test inputs are not required. | Compare generated and held-out samples without assuming paired rows. The included example specifies marginal 1-Wasserstein distance and autocorrelation error of squared returns. |

This distinction changes the agent's instructions, available data, expected
artifacts, and grading procedure. The instruction template only asks for outputs
conditioned on test inputs when those inputs exist.
[Verifier generation](../src/finpipe/generation/prompts.py) receives
the task's row-correspondence rule alongside the output datasets, hidden targets,
and metric definitions. Distributional assessment therefore follows an explicit
task contract rather than an accidental choice of how to compare two files.

I automated the path from research to that contract: paper discovery and retrieval,
suitability screening, task extraction, specification review, code generation,
and Harbor export. Extracted specifications describe dataset roles, filenames,
sources, structures, and the datasets each metric consumes. Deterministic
[structural checks](../src/finpipe/review/rules.py) catch missing required roles,
duplicate filenames across roles, and metrics referencing unknown datasets. A
task critic reviews consistency between the declared task type, data, and metrics.
Generated Python is linted; Docker builds run the retrieval scripts and check for
the expected data files before export.

The exported environment separates what the agent may use from what grades it.
The solver image contains training data and any test inputs; a separate verifier
image contains the withheld targets and assessment script. The agent works
through a Bash harness and writes submissions to `/logs/artifacts/`. The existing
verifier evaluates those submissions and emits named metrics to
`/logs/verifier/reward.json`. This preserves the paper-derived task through
execution and assessment.

I connected these environments to **Prime-RL + Verifiers**, with GRPO and
configurable LoRA adaptation. The RL integration reuses the generated Docker
definitions and task-specific verifiers. Its
[reward adapter](../rl/src/finpipe_rl/taskset.py) selects an existing metric and
applies its declared maximize/minimize direction: an error metric to minimize
receives a negative weight. This makes the optimization signal follow the task's
definition of correctness. Success thresholds are specified separately for
evaluation. Nonfinite rewards and nonzero verifier exits are rejected. Native
worker pools, distributed launch, and checkpoint state handle parallel rollouts
and recovery.

I split the **300 tasks** into three disjoint partitions:

| Split | Tasks | Purpose |
|---|---:|---|
| Training | 150 | GRPO policy updates |
| Validation | 50 | Training monitoring and checkpoint selection |
| Held-out evaluation (`test`) | 100 | Final baseline-versus-GRPO comparison; excluded from training and tuning |
| **Total** | **300** | |

The split operates on whole tasks, not rows within a task's financial dataset.
The isolation policy keeps tasks sharing declared paper/dataset provenance in one
partition; frozen manifests reject duplicate task identities and cross-split
provenance overlap. Each task also keeps its grading targets separate from the
agent's input data.

The evaluation harness compares both models on the same 100 held-out tasks with
matching success criteria, sampling settings, budgets, and repeat counts.
Execution failures count as unsuccessful attempts. Reports include task-level
results, existing metric changes, and paired 95% bootstrap intervals clustered by
provenance; recovery checks prevent replacing recorded failures with successful
retries.

The completed cloud GPU experiment produced the following held-out results:

| Qwen3.8-27B | Problems solved | Task success |
|---|---:|---:|
| Baseline | 61 / 100 | 61% |
| After GRPO | 75 / 100 | 75% |
| Improvement | **14 additional problems** | **+14 percentage points** |

This is the baseline-versus-GRPO result for the overall system. Attributing the
gain to environment changes alone would require a controlled ablation.

Recorded local validation passed **164 repository tests** and **57 framework
contract tests**, with nine and one skipped respectively. Tests explicitly cover
task-dependent instructions, row-alignment versus distributional verifier prompts,
required dataset roles, reward direction, and evaluation correctness. Lint,
package builds, and documentation builds also passed. The
[validation record](superpowers/plans/2026-11-11-rl-pipeline.md) documents those
checks. The cloud results are measurements from my completed experiment;
cloud training logs and evaluation artifacts were not independently inspected
during the repository review.
