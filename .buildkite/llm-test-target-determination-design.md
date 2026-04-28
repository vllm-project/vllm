# LLM-Assisted Test Target Determination

## Objective

Select the right set of tests to run for each PR, reducing CI waste without
missing regressions.

Today, vLLM CI uses manually maintained `source_file_dependencies` lists in
YAML configs to decide which tests to trigger. These lists are static, go stale
as code evolves, and are either too broad (wasting GPU hours) or too narrow
(missing affected tests). There is no systematic way to keep them in sync with
the codebase.

This design replaces that approach with a two-part system:

1. **Static import analysis** generates an objective, code-derived mapping from
   source directories to test directories. It runs fresh on every invocation,
   so it never goes stale. This produces the **candidate set** of tests.
2. **An LLM (Claude Haiku)** reads the actual diff content to **narrow down**
   the candidate set to only the tests affected by the specific change. It also
   handles judgment calls the mapping cannot: non-Python file changes, unmapped
   subprocess-based tests, and ambiguous edge cases.

## Architecture

```
PR opened / updated
        |
        v
+------------------+
|  select_tests.sh |  (orchestrator)
+------------------+
        |
        |--- Step 1: git diff → changed files list + full diff content
        |
        |--- Step 2: python3 build_test_mapping.py --files <changed>
        |            (AST-based import analysis → pre-filtered candidates
        |             for changed files only)
        |
        |--- Step 3: read TEST_SELECTION.md
        |            (static rules for edge cases the mapping can't cover)
        |
        |--- Step 4: claude -p --model haiku
        |            (LLM uses mapping as candidates, reads diff to narrow
        |             down to only tests affected by the specific change)
        |
        |--- Step 5: parse output, build PR comment
        |
        |--- Step 6: gh pr comment (post to PR for author review)
        |
        v
  PR comment with suggested test targets
```

## Components

### 1. `build_test_mapping.py` — Static Import Analysis

**Location:** `.buildkite/scripts/build_test_mapping.py`

Produces a mapping from source files to test files. When invoked with
`--files`, it outputs a pre-filtered table containing only the candidate
tests for the specified changed files — not the full mapping for the
entire codebase.

Resolves three layers of dependencies for each test file:

| Layer | What it captures | How |
|---|---|---|
| Direct imports | `test_foo.py` does `from vllm.config import X` | AST parsing of `import` / `from...import` statements |
| Conftest fixtures | `test_foo.py` uses a pytest fixture defined in a `conftest.py` that imports `vllm.*` | Walk the conftest chain (directory + parents), match fixture names to test function parameters |
| Transitive helpers | `test_foo.py` imports `tests.utils` which imports `vllm.entrypoints` | Build index of all `tests/*.py` helper modules, recursively follow `tests.*` imports (up to depth 10, with cycle detection) |

**Coverage:** 98% of test files (865 of 882). The remaining 17 are
subprocess/HTTP-based tests with no Python-level `vllm` imports.

**Maintenance:** None. The script reads the current source tree on every run.
When tests are added, removed, or refactored, the mapping updates
automatically.

### 2. `TEST_SELECTION.md` — Static Rules

**Location:** `.buildkite/TEST_SELECTION.md`

A small instruction file containing only rules that the auto-generated mapping
**cannot** derive from code:

| Rule | Purpose |
|---|---|
| Rule 1: Candidates are the starting set, diff narrows | Pre-filtered candidates are already provided; read the diff to drop tests unaffected by the specific change (comment-only changes, single function edits, etc.) |
| Rule 2: Manual entries | Maps 8 tests that have no `vllm.*` imports (subprocess/HTTP tests, `forward_context` gap) |
| Rule 3: Non-Python files | Maps `csrc/`, `CMakeLists.txt`, `pyproject.toml`, `requirements/`, docs, `.md` to appropriate test areas |
| Rule 4: When in doubt, include | Sets error bias toward inclusion (`tests/basic_correctness/` as default) |
| Rule 5: Large PRs | Heuristic: 20+ files across 5+ dirs triggers `tests/basic_correctness/` + `tests/v1/e2e/` |

Also contains a strict output format specification that constrains the LLM
response to parseable `test_path | reason` lines.

### 3. `select_tests.sh` — Orchestrator

**Location:** `.buildkite/scripts/select_tests.sh`

End-to-end shell script that ties everything together:

1. **Get changed files and diff** — `git diff origin/main...HEAD --name-only`
   for the file list, `git diff origin/main...HEAD -- '*.py'` for full patch
   content (truncated at 50KB to control prompt size)
2. **Generate pre-filtered candidates** — Runs
   `build_test_mapping.py --files <changed_files>` to produce a candidate
   table scoped to only the changed files (not the full codebase mapping)
3. **Read instructions** — Loads `TEST_SELECTION.md`
4. **Ask Claude** — Sends a single prompt containing instructions +
   pre-filtered candidates + changed files + diff content to
   `claude -p --model haiku`
5. **Parse output** — Filters response to valid `path | reason` lines via
   `grep -E '^[a-zA-Z_/.]+ *\|'`. Handles `NONE` case for docs-only PRs.
6. **Post or print** — Posts a formatted PR comment via `gh pr comment`, or
   prints to stdout in `--dry-run` mode. Deletes any previous test selection
   comment to avoid stacking.

Also outputs clean test paths to stdout for future CI pipeline consumption.

**Usage:**

```bash
# Post comment to PR #1234
.buildkite/scripts/select_tests.sh 1234

# Local testing (no comment posted)
.buildkite/scripts/select_tests.sh --dry-run

# Custom base branch
.buildkite/scripts/select_tests.sh 1234 origin/release
```

**Requirements:** `claude` CLI, `python3`, `gh` CLI (except in dry-run mode).

## Prompt Structure

The prompt sent to Claude Haiku has five sections:

```
1. System instruction ("You are selecting tests for a CI pipeline...")
2. TEST_SELECTION.md content (rules + output format)
3. Pre-filtered candidate table (changed source file → candidate test files)
4. Changed files list (from git diff --name-only)
5. Diff content (from git diff -- '*.py', truncated at 50KB)
```

Sections 1-2 form a stable prefix across invocations, which benefits from
Anthropic's automatic prompt caching (subsequent calls with the same prefix
pay 0.1x input token cost on the cached portion). Sections 3-5 are
PR-specific. Note: the candidate table (section 3) is now pre-filtered to
only the changed files, significantly reducing prompt size compared to the
full codebase mapping.

**Diff truncation:** If the Python diff exceeds 50KB, it is globally
truncated. Files within the limit get full diff (LLM can reason precisely
about what changed). Files beyond the limit still appear in the changed
files list — the LLM falls back to including all mapped tests for them.
This means truncation degrades precision but never correctness.

## Rollout Plan

### Phase 1: Shadow mode (current)

- Script runs on PRs and posts a comment with suggested test targets
- Each comment includes the LLM's **reasoning** (what it saw in the diff,
  why it selected or excluded tests) so PR authors can verify the logic
- Authors reply if the selection looks wrong; no response means it's fine
- No CI behavior changes — existing test selection continues as-is
- Feedback from comments is used to refine `TEST_SELECTION.md` rules

### Phase 2: CI integration (future)

Once confidence is established through Phase 1 feedback:

- `select_tests.sh` output is consumed by CI pipeline to control which test
  jobs are triggered
- Existing `source_file_dependencies` in YAML configs can be removed
- Fallback: if Claude returns an error or empty response, run the full test
  suite (fail-open)

## Cost

- **Model:** Claude Haiku (cheapest tier)
- **Per-invocation:** ~$0.01-0.05 depending on mapping size and changed file count
- **Estimated monthly:** ~$5-20 at typical PR volume
- **Caching:** The mapping + instructions prefix is stable across PRs, so
  repeat calls benefit from automatic prompt caching (0.1x input cost on
  cache hits)

## Key Design Decisions

**Why send the diff content, not just file names?**

Import analysis maps at the file level: "test X imports source file Y." But
changing a comment in file Y doesn't require running test X. Sending the
actual diff lets the LLM reason about *what* changed — a comment edit, a
specific function signature, a new code path — and narrow down from the
candidate set accordingly. The mapping provides recall (don't miss anything),
the diff provides precision (don't run unnecessary tests).

**Why an LLM instead of pure static analysis?**

Static import analysis covers 98% of test files but cannot handle:
non-Python files (C++/CUDA, build configs), subprocess-based tests, or
judgment calls like "this large PR has cross-cutting risk." The LLM fills
these gaps using a small rule set, while the mapping provides the objective
foundation.

**Why generate the mapping at runtime instead of committing it?**

A committed mapping file would go stale as tests are added and imports change.
Generating it fresh on every run means zero maintenance — the mapping is
always accurate for the current codebase.

**Why Haiku?**

Test selection is a structured, rule-following task — not a creative or
reasoning-heavy one. Haiku is the cheapest model and sufficient for matching
file paths against a mapping table and applying 5 simple rules.

**Why post as a PR comment instead of controlling CI directly?**

Phase 1 uses comments to collect human feedback without risk. If the selection
is wrong, the worst case is a reply saying "also run X" — not a missed
regression in production CI.

## Known Limitation: Package-Level Import Granularity

The current import analysis operates at the **module** level. When a test does:

```python
from vllm.config import ModelConfig, CacheConfig
```

the AST records this as an import of `vllm.config` (the package), not the
specific sub-module where `ModelConfig` is defined (`vllm/config/model_config.py`).
This is because Python resolves the name through `vllm/config/__init__.py`,
which re-exports symbols from sub-modules.

As a result, changing **any** file under `vllm/config/` produces the same
candidate set (~551 tests) — because every test that imports anything from
`vllm.config` matches. The LLM + diff narrows this down using judgment, but
the candidate set itself is broader than necessary.

### Alternative: `__init__.py` re-export resolution

This can be solved mechanically by resolving re-exports through `__init__.py`
files. The approach:

1. **Parse `__init__.py` files** to build a symbol-to-source-file map. For
   example, parse `vllm/config/__init__.py` and find:
   ```python
   from vllm.config.model_config import ModelConfig
   from vllm.config.cache_config import CacheConfig
   from vllm.config.pool_config import PoolConfig
   ```
   This produces: `ModelConfig → vllm/config/model_config.py`,
   `CacheConfig → vllm/config/cache_config.py`, etc.

2. **Track imported names in test files**, not just modules. Instead of
   recording that `test_foo.py` imports `vllm.config`, record that it imports
   `ModelConfig` from `vllm.config`.

3. **Resolve names to source files** using the `__init__.py` map. If
   `test_foo.py` imports `ModelConfig` and `ModelConfig` is defined in
   `vllm/config/model_config.py`, then `test_foo.py` depends on
   `vllm/config/model_config.py` specifically — not the entire
   `vllm/config/` package.

**Expected impact:** For packages like `vllm/config/` (10+ sub-modules,
500+ importing tests), this would reduce candidate sets from ~551 to the
actual importers of the changed sub-module — potentially 10-50 tests.

**Complexity:** Moderate. Requires parsing `__init__.py` re-export chains
(which can be multi-level: `__init__.py` re-exports from a sub-package
whose `__init__.py` re-exports from a module). Also needs handling for
wildcard re-exports (`from .model_config import *`) and conditional
re-exports.

**Maintenance:** None — like the rest of the mapping, it would be derived
from the current source tree at runtime.

This is a natural next step if Phase 1 feedback shows that the LLM
struggles with large candidate sets for widely-imported packages.
