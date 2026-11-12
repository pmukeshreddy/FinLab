# Finpipe Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow the shared ledger.

**Goal:** Rewrite and reorganize the finance task pipeline while preserving its behavior.

**Architecture:** Feature packages own their services and validation. Shared workflow, storage, integrations, and packaged resources replace the nested pipeline layout.

**Tech Stack:** Python 3.12+, Hydra, SQLAlchemy, Pydantic, LangChain, Jinja, uv.

**Spec:** `docs/superpowers/specs/2026-11-11-finpipe-design.md`

## Global Constraints

- Preserve commands, Hydra keys, database schema and values, output format, prompt content, and third-party versions.
- Preserve behavior of all existing tests; keep Docker integration requirements explicit.
- New package and distribution: `finpipe`; no legacy import shims.
- No live model or financial-service calls during tests.
- Keep implementation on `refactor/finpipe` until verification is complete.

## Review Focus

- Existing SQLite data must remain readable with identical enum storage and relationships.
- Batch and single-paper workflows must execute the same ordered stages and skip gates correctly.
- Custom templates must override defaults; packaged defaults must work outside the checkout.
- Retrieval fallback, collector budgets, and model failure states must be preserved.
- Generated task metadata must parse correctly with arbitrary model-produced prose.

### Task 1: Package and resource migration

**Files:** `src/finpipe/**`, `pyproject.toml`, `uv.lock`, `tests/**`.
**Interfaces:** Keep existing feature class and method contracts while changing module paths; add `finpipe.resources.template_directory(path=None)`.

- [x] Record baseline tests and schema characterization.
- [x] Add command/resource contract tests and verify the new package is absent.
- [x] Migrate modules, tests, configs, and templates to the feature layout.
- [x] Update package metadata and CLI task commands without changing third-party pins.
- [x] Run migrated behavior tests and commit the migration.

### Task 2: Feature implementation rewrite

**Files:** `src/finpipe/{discovery,documents,evaluation,extraction,review,generation,export}/**`; related `tests/features/**`.
**Interfaces:** Preserve existing class names, constructor arguments, methods, returned model shapes, and state transitions. Consume shared storage/models and `template_directory` for nullable default template paths.

- [x] Read the mapped modules and existing behavior tests before rewriting.
- [x] Add focused behavior coverage for untested branching before changing it.
- [x] Rewrite service control flow into focused helpers; remove repeated queries, rendering, and validation where semantics are identical.
- [x] Preserve prompt text and public structured-response fields.
- [x] Keep collector budgets, fallbacks, ranking, rejection, exception handling, and artifact formats intact.
- [x] Run all feature tests; report Docker-dependent outcomes separately.
- [x] Commit only owned feature paths and report concrete implementation changes.

### Task 3: Application, storage, integrations, and documentation

**Files:** `src/finpipe/{app,storage,contracts,integrations,resources}/**`, `src/finpipe/cli.py`, `tests/{app,storage,integrations}/**`, `README.md`, `docs/**`, `.github/**`.
**Interfaces:** `run_workflow(cfg, output_dir, mode)` routes the same seven stages; `finpipe batch` and `finpipe paper` preserve Hydra override syntax. Resource helpers resolve bundled defaults and user-provided paths.

- [x] Add tests for batch order, single-paper behavior, gate skips, and existing schema reads.
- [x] Replace duplicated entry-point logic with a shared workflow and explicit context.
- [x] Refactor persistence updates and integrations into focused units while preserving payloads and side effects.
- [x] Rewrite README and architecture/developer docs to describe the actual new layout and commands.
- [x] Verify CLI configuration output, package resource resolution, and schema compatibility.

### Task 4: Integration and final review

**Files:** all changed paths and build artifacts (untracked).
**Interfaces:** The installed wheel exposes the same command and task generation contracts as the source checkout.

- [x] Run the complete test suite, lint, formatting, documentation build, and wheel build.
- [x] Install the wheel in a clean environment and check configuration output outside the checkout.
- [x] Review the complete rewrite against baseline behavior and fix substantive regressions with reproducing tests.
- [x] Record remaining environment limits and commit the verified rewrite.


## Verification result

- Full suite: 118 passed; three real Docker integration cases skipped because
  this environment has no running daemon. Strict Docker mode reports an error.
- Repository pre-commit hooks, Ruff, formatting, documentation build, sdist,
  and wheel build passed.
- A separate environment installed the final wheel and composed both commands
  from outside the checkout, then exported and parsed Unicode/DEL metadata using
  bundled resources.
- All third-party dependency pins, six template files, and prompt string content
  match the baseline. The SQLite schema compatibility test passes.
- Independent feature and whole-branch reviews completed. The final review found
  a preexisting non-BMP JSON/TOML escaping defect; an exporter serializer and
  production round-trip tests fixed it without modifying template bytes. The
  focused fix review approved the result.
- Existing dependency deprecation/beta warnings remain. Live paid-model and
  remote provider calls were not used during validation.
