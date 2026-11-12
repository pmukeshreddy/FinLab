RL training and evaluation
==========================

I built the training and evaluation integration and completed **Qwen3.8-27B GRPO
training on cloud GPUs**. Measured task success improved from **61% baseline to
75% after training**, a **14-percentage-point improvement (approximately 23%
relative)**. The :download:`technical report <rl-results.md>` presents the
experiment results, engineering work, testing evidence, and verification scope.

The independent ``rl/`` package integrates Prime-RL, Verifiers, and exported Harbor
tasks without installing GPU dependencies in the finance discovery application.
See the :download:`RL runbook <../rl/README.md>` for installation,
configuration, distributed training, recovery, evaluation, and qualification.

Training uses Qwen3.8-27B, native GRPO and LoRA, and the original Docker images and
verifiers. A frozen manifest enforces explicit provenance groups and disjoint
train, validation and test membership. Optimization uses the training split;
online evaluation uses validation tasks. The final comparison runs both models
on the same held-out test tasks.

Reports include task-macro success, paired cluster bootstrap confidence intervals,
raw metric changes, task-level results, execution failures, native training
telemetry and local GPU samples. Native experiment artifacts supply the
operational measurements used in generated qualification reports.
