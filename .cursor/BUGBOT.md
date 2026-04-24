# Bugbot Rules for vllm-cohere

This repository is a maintained fork of
[vllm-project/vllm](https://github.com/vllm-project/vllm). Changes fall into
two categories: **modifications to upstream files** (which require annotation)
and **new cohere-only files** (which do not).

---

## 1. Cohere Marker Annotations

Every modification to a file that exists in upstream must be annotated with
cohere marker comments so that diffs are auditable and merge conflicts are
easy to attribute.

### Required marker format

| File type | Inline | Block start | Block end |
|-----------|--------|-------------|-----------|
| Python (`.py`) | `# cohere` | `# cohere start` | `# cohere end` |
| C/C++/CUDA (`.c`, `.cpp`, `.h`, `.cu`, `.cuh`) | `// cohere` | `// cohere start` | `// cohere end` |
| Shell/YAML/TOML/Dockerfile/CMake | `# cohere` | `# cohere start` | `# cohere end` |
| JS/TS (`.js`, `.ts`) | `// cohere` | `// cohere start` | `// cohere end` |
| Markdown (`.md`) | `<!-- cohere -->` | `<!-- cohere start -->` | `<!-- cohere end -->` |

### Rules

- A single changed line should end with the appropriate inline comment.
- A contiguous block of 2+ changed lines should be wrapped with start/end
  markers on the lines immediately before and after the block.
- Every `cohere start` must have a matching `cohere end`.
- New files that do not exist in upstream (e.g. anything under `vllm/cohere/`,
  `tests/cohere/`, `docs/cohere/`, `.cursor/`) do **not** need markers.

**Flag any PR that adds or modifies lines in an upstream file without the
corresponding cohere marker.**

---

## 2. Documentation and Code-Notes Freshness

Cohere-specific documentation lives under `docs/cohere/`. Key subdirectories:

- `docs/cohere/code_notes/` — deep technical notes on fork-specific behavior,
  organized by topic (runtime, models, CI, build, tests). The entry point is
  `docs/cohere/code_notes/upstream-diff.md`.
- `docs/cohere/tests/` — per-test documentation entries.

### Rules

- If a PR changes runtime behavior, model integration, CI pipelines, build
  logic, or test infrastructure, check whether the relevant code-notes file
  should be updated to reflect the change. Flag when a behavioral change has
  no accompanying docs update.
- If a PR adds or substantially modifies a test under `tests/cohere/`, check
  whether a matching entry exists in `docs/cohere/tests/`. Flag if missing.

---

## 3. Error Handling and Defaults

Code in this repo should fail clearly and early when required inputs,
configuration, or dependencies are missing.

- **No synthetic defaults**: if a value is required, make it a required
  parameter. Do not substitute made-up fallbacks when a lookup returns no
  result — propagate the absence (`None`, raise, or skip).
- **Let real errors surface**: do not swallow exceptions that cannot be
  meaningfully handled. Prefer raising or propagating so the root cause gets
  fixed.
- **Optional types are intentional**: `Optional[T]` / `T | None` should only
  appear when "no value" is a valid domain state, not as a convenience default.
- **Explicit config**: missing env vars or config entries should fail at startup
  or first use, not silently degrade to empty strings, zeros, or placeholders.

**Flag patterns that hide real problems**: catch-all exception handlers that
swallow errors, synthetic defaults for required values, or silent fallbacks
that mask missing configuration.

---

## 4. Code Reuse and Simplicity

- **Flag duplication**: if a PR introduces logic that already exists in a shared
  helper (e.g. `vllm/utils.py`, common test fixtures), suggest reusing or
  extending the existing code.
- **Flag unnecessary indirection**: trivial wrappers that only forward arguments,
  needless abstraction layers, or dead code that should be removed.
- **Shared helpers belong in well-known locations**: `vllm/utils.py`,
  `vllm/cohere/`, or common test fixtures. New utilities hidden in
  deeply-nested modules are harder to discover and reuse.

---

## 5. Test Coverage and Test Documentation

Every PR that changes runtime behavior, adds a feature, or fixes a bug should
include tests. Test documentation should accompany the tests so that CI
coverage, assertions, and compatibility are reviewable.

### Rules

- **Flag PRs with no tests**: if a PR adds or changes runtime behavior but does
  not add or update tests under `tests/cohere/`, flag it and ask the author to
  add test coverage.
- **Flag tests without test docs**: if a PR adds or modifies tests under
  `tests/cohere/` but does not add or update a matching entry in
  `docs/cohere/tests/`, remind the author to use the `/add-test-docs` skill
  (backfill mode) to generate documentation for the new or changed tests.
- **Test docs should follow the standard format**: each test case documented in
  `docs/cohere/tests/` must include `How it runs`, `Checks`, `Measurements`,
  `Compatibility`, and `Implementation` sections as defined by the
  `/add-test-docs` skill. Flag entries that are missing required sections.
- **Observability matrix and feature matrix must stay in sync**: new test entries
  should be reflected in `docs/cohere/tests/observability_matrix.md` and
  compatibility classifications should be propagated to
  `docs/cohere/tests/feature_matrix.md`. Flag when these are out of date.

