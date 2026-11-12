# Qwen3.8-27B GRPO Training and Evaluation

I built this integration and completed **Qwen3.8-27B GRPO training on cloud GPUs**.
Measured task success increased from **61% baseline to 75% after training**:
**+14 percentage points, approximately 23% relative improvement**. The
[technical report](../docs/rl-results.md) documents the results, engineering work,
and verification scope.

I integrated Prime-RL for GRPO, LoRA, distributed training, worker pools,
inference, retries, checkpoints, and experiment logging. Verifiers provides the
Harbor execution lifecycle. The integration indexes existing exports, adapts their
metadata, validates run provenance, launches native commands, and compares native
evaluation artifacts. Existing task generation, Docker definitions, verifiers,
and rewards are reused.

This runbook describes how to install, operate, and reproduce the workflow using
pinned task images, a model snapshot, and a frozen experiment manifest. The
example hardware topology and hyperparameters below are configuration defaults;
they are not a record of the cloud allocation used for the completed experiment.

## 1. Install the pinned framework

Use Linux, Python 3.12, a supported CUDA/NVIDIA stack and Docker on rollout hosts.
Keep this environment separate from the root finance application's environment.
Set these absolute paths for your installation:

```bash
export FINPIPE_REPO=/shared/finpipe
export PRIME_ROOT=/shared/prime-rl
export BASE_MODEL=/shared/models/qwen3.8-27b-snapshot
export RL_OUTPUT=/shared/experiments/finpipe

git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_ROOT"
git -C "$PRIME_ROOT" checkout a3ee545b686189a860aba4ee3c9f230aa1ae4705
git -C "$PRIME_ROOT" submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
uv sync --directory "$PRIME_ROOT" --frozen --all-extras
uv pip install --python "$PRIME_ROOT/.venv/bin/python" --no-deps -e "$FINPIPE_REPO/rl"
export PYTHONPATH="$FINPIPE_REPO/rl/src"
```

Use `"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl` for commands below. An ordinary
`uv run` can remove packages installed outside the upstream lock; the explicit
`PYTHONPATH` also keeps the plugin available after native Slurm templates run
`uv sync`. The runner propagates this shared path to native worker processes.
Both repositories, tasks, model snapshot and outputs must be accessible at the
same absolute paths on every node.

Download the model at an explicit Hugging Face **commit revision**, with all
tokenizer/config/weight files, into `BASE_MODEL`. Record the revision with the
experiment. The runner hashes the actual snapshot files, including weights;
mutable model names alone cannot establish reproducibility. Keep artifacts
read-only throughout a run. Framework revisions, lock hash, installed versions,
integration source hash, configs, and artifact hashes are recorded in receipts.
Framework checkout mismatches or tracked modifications are rejected.
Native weight-format conversion caches are directed to `run/model_conversion/`
so they cannot change the base snapshot's recorded identity.

## 2. Prepare existing task images and a frozen split

Use the repository's existing export workflow. Build each task's existing
`environment/` and `tests/` Docker context using its existing Compose file or
Dockerfile. Supply its existing build secrets through Docker's secret mechanism.
Publish images to your registry and record immutable `repository@sha256:...`
digests for **both** the solver and verifier. Building once also freezes downloaded
data in those images; audit any verifier that still downloads mutable data during
execution. Ensure workers can pull them. The existing verifier image runs as root.

Copy `catalog.example.toml` to a local catalog and replace every placeholder. Each
entry refers to an existing task directory; paths resolve relative to the catalog.
Declare the exact keys emitted by its existing `reward.json`, their directions,
and a domain-approved success threshold. For example, RMSE uses `minimize` and
success is `rmse <= threshold`. The training signal is the original selected
metric with native reward weight +1 or -1. No metric normalization or new reward
function is introduced. Existing nonfinite rewards and nonzero verifier exits
fail the rollout rather than supply a reward.

Choose splits **before training or viewing final model results**. Keep all tasks
derived from the same paper, dataset family, overlapping data window, or other
dependent source in one split. Use stable, conservative `paper:` and `dataset:`
group identifiers, plus additional groups when needed. The manifest rejects
cross-split groups and duplicate normalized instructions. It cannot discover
undisclosed related papers, semantically duplicated tasks, or pretraining exposure.
Do not put hidden evaluation labels or verifier answers in solver images.

```bash
"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl freeze "$FINPIPE_REPO/rl/catalog.toml" \
  --output "$RL_OUTPUT/manifest.json"
"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl check "$RL_OUTPUT/manifest.json"

"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl audit \
  --manifest "$RL_OUTPUT/manifest.json" --split train \
  --output "$RL_OUTPUT/audit-train" --concurrency 4
```

Run separate audits for validation and test. This delegates setup, noop and gold
checks to native `vf-validate`. Gold is unchecked when the export has no reference
solution; the native noop check uses its own reward>=1 convention, which is not a
metric-specific success threshold. Inspect the native summary and logs. Independently
regrade fixed known-good/bad artifacts repeatedly through the existing verifier
to establish false positives, false negatives and repeatability. Do not infer
verifier accuracy from successful setup or from model scores.

All task files are hashed and checked when loading. Original files remain untouched.
The adapter creates a temporary metadata view for native parsing: it removes the
redundant `verifier.user="root"` field that this pinned parser rejects, and supplies
the catalog's pinned image references. Files are linked where possible, otherwise
copied for parsing; runtime execution uses the original directory. Non-root
verifier declarations are rejected. Exported Compose files containing only a
single image build are replaced by the native single-image deployment. Tasks with
additional runtime Compose settings/services retain native Compose behavior and
require an explicitly reviewed `env.agent.deployment.trust_compose = true` in
each source. The supplied audit command supports the standard exported tasks.

## 3. Train and recover

```bash
"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl train \
  --config "$FINPIPE_REPO/rl/configs/qwen3.8-27b.toml" \
  --manifest "$RL_OUTPUT/manifest.json" --artifact "$BASE_MODEL" \
  --prime-root "$PRIME_ROOT" --output "$RL_OUTPUT/train-001" --dry-run
```

Remove `--dry-run` to launch. Dry-run validates real upstream config schemas; it
does not execute a model, pull images, or establish GPU compatibility. The preset
uses 4 training and 4 inference GPUs, TP=4, rank-16 LoRA, GRPO groups of 8, bounded
worker concurrency, and checkpoints every 10 steps. This is a starting topology,
not a memory-fit or throughput guarantee. Run a short training/validation smoke
test on the target allocation first; measure peak memory and queueing, then tune
batch size, context, worker concurrency, TP and shard topology. Each task also
retains its existing agent/verifier timeouts and resource requests.

Training uses only `train`; checkpoint selection and tuning use only `validation`.
Reserve `test` for the final frozen comparison. Record the validation-based
checkpoint selection rule before testing. Do not repeatedly tune against test
results or present a selected winning seed as a single unbiased experiment.

For a restart, repeat the same command with `--resume`, optionally increasing
`--max-steps`. The runner selects the newest joint native checkpoint with both
trainer DCP metadata and orchestrator progress; incomplete newer checkpoints are
ignored. No checkpoint means an error, not a restart from scratch. Artifact,
framework, integration, split and protocol changes are rejected; only max steps
may change, and must exceed the recovered step. Native state includes model,
optimizer, scheduler and progress.
Retain the entire run directory **and** the sibling `.finpipe/<run-name>/` receipt
directory. A filesystem lock prevents two local launchers resuming the same run.
Use shared storage with working locks. Corrupt native state is left to fail native
loading; it is not silently discarded. Recovery is not a bitwise-determinism claim.

### Multiple nodes / Slurm

Copy the preset and replace its deployment block with a native topology such as:

```toml
[deployment]
type = "multi_node"
num_train_nodes = 2
num_infer_nodes = 1
gpus_per_node = 8

[slurm]
project_dir = "/shared/prime-rl"
partition = "YOUR_PARTITION"
account = "YOUR_ACCOUNT"
time = "24:00:00"
```

Set inference parallelism to fit that allocation. Native Prime-RL generates the
Slurm script, rendezvous, torchrun and worker placement. The wrapper uses
`SBATCH_WAIT=1` to record job completion/failure rather than submission success.
Follow `launcher/logs/` and scheduler job state. If a launcher is interrupted,
check/cancel the existing scheduler job before resuming; do not create concurrent
jobs against one run. Slurm time in receipts includes queue time. Collect scheduler
accounting and cluster GPU telemetry separately; the report does not manufacture
Slurm GPU-hour or billed-cost numbers from launcher time.

## 4. Serve and compare the frozen models

Keep the selected native LoRA broadcast directory under
`train-001/broadcasts/step_N/`, including `.finished`, `adapter_config.json` and
`adapter_model.safetensors`. Choose/preserve it before retention deletes it. A DCP
trainer checkpoint is a recovery artifact, not a directly serveable HF adapter;
the pinned full-model checkpoint converter does not convert LoRA checkpoints.

Serve the same base snapshot for both arms using **native `inference`**, with the
same serving precision, engine configuration, tokenizer and sampling settings.
For example, save a standalone inference TOML, substituting the actual base path:

```toml
[vllm]
model = "/shared/models/qwen3.8-27b-snapshot"
served_model_name = ["finpipe-base"]
dtype = "bfloat16"
tensor_parallel_size = 4
enable_lora = true
max_lora_rank = 16
seed = 42
```

```bash
"$PRIME_ROOT/.venv/bin/inference" @ /shared/inference.toml
```

Load the selected adapter into the engine's admin port (default 8100; the router's
client port defaults to 8000). Create JSON with the actual absolute adapter path:

```json
{"lora_name":"finpipe-trained","lora_path":"/shared/experiments/finpipe/train-001/broadcasts/step_100"}
```

```bash
curl --fail-with-body -X POST 'http://localhost:8100/load_lora_adapter?weight_version=100' \
  -H 'Content-Type: application/json' --data-binary @/shared/adapter.json
```

Check the response and model availability before evaluating. Do not modify/reload
weights during either evaluation. The operator must attest that the endpoints
serve these artifacts: a model listing does not prove the bytes of served weights.
Archive the inference config and logs with the experiment.

```bash
"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl eval \
  --config "$FINPIPE_REPO/rl/configs/eval.toml" --manifest "$RL_OUTPUT/manifest.json" \
  --artifact "$BASE_MODEL" --served-model finpipe-base \
  --base-url http://localhost:8000/v1 --role baseline \
  --training-run "$RL_OUTPUT/train-001" --prime-root "$PRIME_ROOT" \
  --output "$RL_OUTPUT/baseline-001"

"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl eval \
  --config "$FINPIPE_REPO/rl/configs/eval.toml" --manifest "$RL_OUTPUT/manifest.json" \
  --artifact "$RL_OUTPUT/train-001/broadcasts/step_100" --served-model finpipe-trained \
  --base-url http://localhost:8000/v1 --role trained \
  --training-run "$RL_OUTPUT/train-001" --prime-root "$PRIME_ROOT" \
  --output "$RL_OUTPUT/trained-001" --baseline-run "$RL_OUTPUT/baseline-001"
```

The second command automatically writes `trained-001/reports/qualification.json`,
`qualification.md`, and `tasks.csv`. Both arms must cover every held-out task with
identical repeats/budgets/settings. Failed episodes count as unsuccessful. Raw
metric means are conditional on scored episodes and report their coverage; they
are not failure-penalized replacements for task success. CIs use a seeded paired
bootstrap over connected paper/dataset provenance groups with task-macro means.
Fewer than two independent groups produces no CI; small/degenerate samples need
more independent tasks. CIs condition on one fitted model and do not measure
training-seed variation. Run multiple predeclared seeds for that question.

Evaluation `--resume` only continues an interrupted run with no recorded failed
episodes. Upstream resume retains successful episodes and retries failures, so
the runner rejects that biasing case and rejects completed-run resume. If failures
landed, retain the interrupted attempt and run a fresh **pair** under a new name;
record the exclusion and do not select retries based on higher scores. Reports
also reject archived failed cohorts, duplicates, missing tasks, changed protocols
and task data carrying a different manifest digest.

## 5. Observability and qualification

Native structured process logs, resolved configs, metrics JSONL and episode traces
remain in the run directory. The receipt preserves launch attempts and elapsed
time. Add optional native tracking to the training TOML:

```toml
[monitors.wandb]
project = "finpipe-rl"
```

Provide credentials through the environment. Prime platform upload is disabled
by the integration; file logging is enabled. Review rewards (including degenerate
groups), loss, entropy, KL/mismatch, gradient norms, staleness, retry/error counts,
tokens/sec and native performance/MFU series. The report summarizes only series
actually emitted. On a local NVIDIA launcher, `nvidia-smi` samples utilization,
memory and power every 5 seconds; these are host-local samples, not measured
cluster efficiency or energy integration. Missing/nonfinite samples stay unavailable.

Regenerate with native audit evidence and an optional known hourly price:

```bash
"$PRIME_ROOT/.venv/bin/python" -m finpipe_rl report \
  --manifest "$RL_OUTPUT/manifest.json" --training "$RL_OUTPUT/train-001" \
  --baseline "$RL_OUTPUT/baseline-001" --trained "$RL_OUTPUT/trained-001" \
  --audit "$RL_OUTPUT/audit-train" --output "$RL_OUTPUT/qualification" \
  --gpu-hour-rate 2.00
```

The price above is an example input, not a quoted price. Local allocation cost
uses configured GPUs times measured launch duration, explicitly excludes external
inference/CPU/storage/reservation costs, and is never billed cost. Throughput from
trace counts is final-stream episodes divided by total attempt wall time, including
startup/recovery; native performance series provide the steady-state view.

The report distinguishes observed improvement from positive confidence bounds,
execution reliability, verifier evidence, and missing hardware qualification.
It never declares the system production-qualified solely from CPU tests or a
positive mean. Preserve raw artifacts; investigate reward hacking, verifier
repeatability, loss/gradient anomalies and resource saturation before promotion.

## 6. Tests

CPU tests install pinned **real** framework config/Harbor APIs without CUDA:

```bash
uv sync --project rl --frozen --group contract
uv run --project rl --frozen --group contract python -m pytest tests/rl --confcutdir tests/rl -q
uv build --project rl
```

They cover actual exported metadata parsing, worker reconstruction/reward sign,
native GRPO/LoRA schemas, split isolation, failure/recovery semantics, native trace
statistics, and report output. CI runs this same contract suite. Fixtures and
mocked launches verify software behavior; they are not experimental results.

For the opt-in hardware test, create a JSON file with absolute `config`, `manifest`,
`artifact`, and `prime_root` paths. Use a small existing-task configuration suited
to your allocation. It runs two steps, reads native model/optimizer/scheduler/
progress checkpoint metadata, resumes to step three, and checks persisted progress:

```bash
FINPIPE_RL_GPU_CONFIG=/shared/gpu-test-inputs.json \
  "$PRIME_ROOT/.venv/bin/python" -m pytest "$FINPIPE_REPO/tests/rl/test_gpu_recovery.py" \
  --confcutdir "$FINPIPE_REPO/tests/rl" --basetemp /shared/gpu-recovery-test -q
```

Use a disposable `--basetemp` path (pytest clears it). Install pytest in the GPU
environment if absent. The emitted `gpu-recovery.json` records this narrow test;
it does not establish forced-kill recovery, bitwise equivalence, or cluster-wide
resilience. Run fault-injection and Docker/registry failure qualification on the
production topology before promoting it. This opt-in recovery test is separate
from the completed cloud GRPO experiment described in the technical report.

## Reviewed upstream sources

- [Pinned Prime-RL source and native docs](https://github.com/PrimeIntellect-ai/prime-rl/tree/a3ee545b686189a860aba4ee3c9f230aa1ae4705)
- [Pinned Verifiers Harbor implementation](https://github.com/PrimeIntellect-ai/verifiers/blob/ae5e56bc4e714c782817b4bde822fa084fc319d2/verifiers/v1/tasksets/harbor/taskset.py)
- [Qwen3.8-27B model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Slurm sbatch waiting/accounting semantics](https://slurm.schedmd.com/sbatch.html)
