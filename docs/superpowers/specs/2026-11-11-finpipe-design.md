# Finance pipeline rewrite

The user requested a complete implementation rewrite and a different repository
layout, while preserving behavior. The implementation remains Python 3.12+.

## Design

Use `finpipe` as the package and distribution name. Organize features directly
under the package: discovery, documents, evaluation, extraction, review,
generation, and export. Each feature owns its settings, service, and supporting
models. Shared storage, task contracts, external integrations, and application
orchestration have explicit boundaries. Configuration and export templates ship
inside `finpipe.resources`, so an installed wheel can run outside the checkout.

The selected approach is a feature-oriented rewrite with a shared workflow. A
folder-only rename would not satisfy the requested implementation rewrite. A new
language or database would create unnecessary compatibility changes.

## Compatibility

- Keep `uv run task pipeline` and `uv run task paper` and their Hydra overrides.
- Add equivalent `finpipe batch` and `finpipe paper` command entry points.
- Preserve database table/column names, enum values, defaults, and relationships.
- Preserve stage order, state transitions, rejection rules, ranking, retries,
  request throttling, prompt content, and public tool names and payloads.
- Preserve Harbor export filenames, format, and train/verifier isolation.
- Preserve environment variable names and explicit custom template paths.
- Python import paths intentionally change; do not add legacy package aliases.
- Do not make live paid-model or finance API calls during validation.
- Do not change locked third-party dependency versions.

## Verification

Use the existing tests as behavior contracts and add tests for orchestration,
schema compatibility, packaged-resource loading, and missing external services.
The baseline at `b836eba` is 58 passes and three Docker connection failures;
those tests require a Docker daemon. Keep them as explicit integration tests.
Check a built wheel from an unrelated working directory and build the docs.

## Scope decisions

Naming, module boundaries, and internal helpers are implementation choices.
Keep the current repository on main intact while building on refactor/finpipe.
Do not publish the rewrite during implementation.
