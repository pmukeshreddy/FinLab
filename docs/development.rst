Development
===========

Install the locked environment with ``uv sync --frozen``. Feature code and its
tests live together by responsibility: for example, ``src/finpipe/documents``
is covered by ``tests/features/documents``.

Run the checks from the project root::

    uv run pytest
    uv run ruff check src tests
    uv run ruff format --check src tests
    uv run task docs
    uv build

Tests avoid live model and market-data calls. Docker-dependent script validation
is retained as integration coverage. Without a daemon those tests report a skip;
``uv run pytest --require-docker`` makes an unavailable daemon an error. CI uses
that flag after setting up Docker.

The schema fixture in ``tests/fixtures/sqlite_schema.json`` records the existing
SQLite contract. Changes to model declarations must preserve it unless a database
migration is explicitly intended. Workflow tests protect stage order and optional
gates; export tests protect the artifact layout and parse generated metadata.

Documentation source is in ``docs`` and generated HTML in ``build/docs``.
Configuration and template assets under ``src/finpipe/resources`` ship in the
wheel. Test installed commands from a directory outside the checkout when changing
resource loading or command entry points.

The RL integration uses a separate pinned environment. Run its native framework
contract suite from the project root::

    uv sync --project rl --frozen --group contract
    uv run --project rl --frozen --group contract python -m pytest tests/rl --confcutdir tests/rl -q
    uv build --project rl

My recorded local validation passed 164 repository tests and 57 framework contract
tests, with nine and one skipped respectively. Ruff, package builds, and the
Sphinx documentation build passed. These checks cover task loading, native
configuration contracts, reward direction, checkpoint selection, split isolation,
evaluation statistics, and report generation.

I completed the cloud GPU GRPO experiment separately, improving Qwen3.8-27B task
success from 61% to 75%. See the :download:`technical report <rl-results.md>` for
the experiment results and verification scope, and the
:download:`RL runbook <../rl/README.md>` for the opt-in hardware recovery test.
