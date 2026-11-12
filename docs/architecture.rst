Architecture
============

The application is organized around the stages of task creation. Each feature
package owns its settings, service, and supporting models or transformations.
``finpipe.app.workflow`` sequences those services for both command modes.

.. code-block:: text

    discovery -> documents -> evaluation -> extraction -> review -> generation -> export

The batch command searches across enabled scopes and stores progress in SQLite.
The paper command loads one paper into an in-memory database. It can bypass
paper evaluation, marking that paper accepted before extraction. Both modes can
bypass task critique by automatically approving candidates.

State and contracts
-------------------

``contracts`` contains the task-type definitions and state values used across
features. ``storage`` maps those records to the existing database tables and
updates their status transactionally. Table names, columns, enum storage, and
relationships remain compatible with existing databases.

Services read their eligible records from storage. Their status transitions are
the handoff to the next service, which allows batch runs to resume without
repeating completed work. Training data and withheld verifier data remain
separate throughout generation and export.

External services
-----------------

Provider clients belong to ``discovery.sources`` and ``integrations``.
Market-data tools normalize inputs and retain their established response formats.
MCP tools are filtered through a read-tool allowlist. Optional authenticated
connections are enabled by their environment variables; the Alpaca server is
constructed only when its credentials are available.

Commands and resources
----------------------

``finpipe.cli`` handles command selection and Hydra configuration. Heavy workflow
imports occur only after configuration has been composed, so inspecting a command
with ``--cfg job`` does not start external services.

Default YAML files and Jinja templates are package resources. Their locations do
not depend on the shell's current directory. Explicit template paths still select
user-provided templates, while output paths remain relative to the run location.

The package name is now ``finpipe``. Python consumers should update imports;
legacy module aliases are not installed. The ``uv run task pipeline`` and
``uv run task paper`` aliases, configuration keys, database schema, and Harbor
artifact contract remain available.

RL training and evaluation
--------------------------

I extended the task pipeline with a separate ``rl/`` integration built on Prime-RL
and Verifiers. I completed cloud GPU GRPO training for Qwen3.8-27B, increasing
measured task success from 61% to 75%: 14 percentage points, approximately 23%
relative improvement.

The integration consumes existing Harbor exports. A frozen manifest records task
content hashes, provenance groups, split membership, solver/verifier image
digests, and metric contracts. Verifiers runs the Bash agent in a solver container
and grades submitted artifacts in a separate verifier container, preserving the
existing scripts and rewards.

Prime-RL handles GRPO, LoRA, distributed launch, inference, worker pools, and
native checkpoint state. The local integration validates experiment identities,
selects complete recovery checkpoints, and enforces compatible evaluation
protocols. The evaluation harness compares both models on the same held-out
tasks, includes execution failures in task success, and produces task breakdowns
and paired confidence intervals clustered by provenance.

The :download:`technical report <rl-results.md>` describes the completed
experiment and evidence scope. See :doc:`rl` for the operational workflow.
