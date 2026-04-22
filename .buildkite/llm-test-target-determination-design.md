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
   so it never goes stale.
2. **An LLM (Claude Haiku)** combines that mapping with a small set of static
   rules to produce a final test selection. The LLM handles judgment calls the
   mapping cannot: non-Python file changes, unmapped subprocess-based tests,
   and ambiguous edge cases.

## Architecture

```
PR opened / updated
        |
        v
+------------------+
|  select_tests.sh |  (orchestrator)
+------------------+
        |
        |--- Step 1: git diff → list of changed files
        |
        |--- Step 2: python3 build_test_mapping.py
        |            (AST-based import analysis → source→test mapping table)
        |
        |--- Step 3: read TEST_SELECTION.md
        |            (static rules for edge cases the mapping can't cover)
        |
        |--- Step 4: claude -p --model haiku
        |            (LLM combines mapping + rules + changed files → test list)
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

Produces a markdown table mapping source directories (e.g., `vllm/config/`) to
test directories (e.g., `tests/basic_correctness/`, `tests/v1/e2e/`, ...).

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
| Rule 1: Use mapping as baseline | Tells the LLM how to interpret the mapping table (match at directory level) |
| Rule 2: Manual entries | Maps 8 tests that have no `vllm.*` imports (subprocess/HTTP tests, `forward_context` gap) |
| Rule 3: Non-Python files | Maps `csrc/`, `CMakeLists.txt`, `pyproject.toml`, `requirements/`, docs, `.md` to appropriate test areas |
| Rule 4: When in doubt, include | Sets error bias toward inclusion (`tests/basic_correctness/` as default) |
| Rule 5: Large PRs | Heuristic: 20+ files across 5+ dirs triggers `tests/basic_correctness/` + `tests/v1/e2e/` |

Also contains a strict output format specification that constrains the LLM
response to parseable `test_path | reason` lines.

### 3. `select_tests.sh` — Orchestrator

**Location:** `.buildkite/scripts/select_tests.sh`

End-to-end shell script that ties everything together:

1. **Get changed files** — `git diff origin/main...HEAD --name-only`
2. **Generate mapping** — Runs `build_test_mapping.py` (fresh every time)
3. **Read instructions** — Loads `TEST_SELECTION.md`
4. **Ask Claude** — Sends a single prompt containing instructions + mapping +
   changed files to `claude -p --model haiku`
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

The prompt sent to Claude Haiku has four sections:

```
1. System instruction ("You are selecting tests for a CI pipeline...")
2. TEST_SELECTION.md content (rules + output format)
3. Auto-generated mapping table (source dir → test dirs)
4. Changed files list (from git diff)
```

The mapping table and instructions form a stable prefix across invocations,
which benefits from Anthropic's automatic prompt caching (subsequent calls with
the same prefix pay 0.1x input token cost on the cached portion).

## Rollout Plan

### Phase 1: Shadow mode (current)

- Script runs on PRs and posts a comment with suggested test targets
- PR authors review the suggestion and reply if it looks wrong
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
