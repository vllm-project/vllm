---
name: pre-pr-check
description: Check local changes against the LMCache coding standards and fix issues before creating a PR
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, Agent
argument-hint: "[--scope uncommitted|branch|staged] [--fix|--check-only]"
---

# Pre-PR Check: LMCache Coding Standard Compliance

Check the developer's local changes against the coding standards defined in
`docs/coding_standards.md` and modify the code to fix compliance issues, so the
PR is clean before it is opened.

This skill is meant to be run **before** `/create-pr`.

## Arguments

- `--scope` (default: `branch`)
  - `uncommitted` -- check only unstaged + staged changes vs HEAD
  - `staged` -- check only staged changes
  - `branch` -- check all commits on the current branch not yet in `dev` (the
    full PR diff). This is the default since that matches what a reviewer sees.
- `--fix` (default) -- automatically fix issues that can be fixed mechanically
- `--check-only` -- only report issues, do not modify files

## Workflow

### Step 0 -- Load the standard

Read `docs/coding_standards.md` at the repo root. This is the authoritative reference.

### Step 1 -- Detect the scope of changes

Run `git status --short` and `git branch --show-current` to orient yourself.

Then, based on `--scope`, compute the diff:

| Scope | Command |
|-------|---------|
| `uncommitted` | `git diff HEAD` |
| `staged` | `git diff --cached` |
| `branch` (default) | `git diff dev...HEAD` |

Also list changed files:

| Scope | Command |
|-------|---------|
| `uncommitted` | `git diff --name-only HEAD` |
| `staged` | `git diff --cached --name-only` |
| `branch` | `git diff --name-only dev...HEAD` |

If there are no changes in the selected scope, report that and stop.

### Step 2 -- Read the changed files

For each changed file, read the current content (not just the diff) so you can
make accurate edits. Focus on:
- `.py` files -- full coding-standard review
- `.md`, `.rst` -- documentation review (brief)
- `.cpp`, `.cu`, `.rs` -- light review (style only; clang-format / cargo fmt handle most of it)

For each changed file, also identify the **changed public symbols** (functions,
methods, classes, fields whose signature, return type, raised exceptions, or
behavior contract changed). You will use these in Step 4 to find and inspect callers.

For large changes, delegate to the Explore agent to summarize per-file
responsibilities before reviewing.

### Step 3 -- Run the checks

For each changed Python file, check the following (from `docs/coding_standards.md`):

#### A. Mechanical issues (fix automatically unless `--check-only`)

These are straightforward to fix:

1. **License header** (Section 1.3): Missing `# SPDX-License-Identifier: Apache-2.0` on line 1 of new Python files.
2. **`assert` used for runtime validation** (Section 2.2): Replace with `if <not cond>: raise ValueError(<clear msg>)`.
3. **`Optional` usage** (Section 2.1): Where a value is always initialized, remove the `Optional` wrapper and initialize directly. Where None is genuinely possible, rewrite as `X | None`.
4. **Bare generics** (Section 2.1): Replace bare `list`, `dict`, `tuple`, `set` in annotations with parameterized forms (`list[T]`, `dict[K, V]`).
5. **Lazy imports** (Section 7.2): Move imports to the top of the file, under the correct section heading. Preserve `torch`-before-native-C-extension ordering.
6. **Bare `logging.getLogger(__name__)`** (Section 7.4): Replace with `init_logger(__name__)` from `lmcache.logging`.
7. **Operational logs at `INFO` level** (Section 7.4): If a log is about store/retrieve/task progress, change `logger.info(...)` to `logger.debug(...)`.
8. **Inappropriate `WARNING`** (Section 7.4): For log messages about benign concurrent conditions (e.g., "key not found, this should not happen" in a code path documented as not holding the global lock), downgrade to `debug` and fix the misleading message.

#### B. Docstring issues (fix when content is clear; ask when content is unclear)

9. **Missing docstrings** on public/modified functions (Section 3.1-3.2):
   - If the function is simple and its purpose is obvious from the code, draft a full docstring (summary, args, returns, raises).
   - If behavior is non-obvious, draft a docstring with clear sections but add a `TODO: confirm` note where you are uncertain, and flag it in the report.
10. **Docstring accuracy** (Section 3.3): If you added a parameter but the docstring still describes the old signature, update it. If a parameter is currently a no-op (accepted but ignored), the docstring must say so explicitly.
11. **Missing type hints** on function args or return values (Section 2.1): Add them based on the function's body.

#### C. Design issues (flag; do NOT silently refactor)

These are judgment calls and should be reported, not auto-fixed:

12. **Cross-class private member access** (Section 4.4): Accessing `other._private` outside the defining class.
13. **Ambiguous return values** (Section 4.3): Function returns `None` for multiple distinct meanings.
14. **Boolean parameters on public APIs** (Section 4.3).
15. **New features without tests** (Section 5.2).
16. **Bug fixes without regression tests** (Section 5.2).
17. **Unbounded collection growth** (Section 7.6): A set/dict that is appended to without cleanup.
18. **Error paths that leave state inconsistent** (Section 7.5): Exceptions that skip lock release, FD close, or pool free.
19. **Missing design doc** for non-trivial new features (Section 5.1).
20. **PR scope too large** (Section 1.1): If the diff touches many unrelated modules, suggest how to split.
21. **Lock protocol violations** (Section 8): Debug methods not holding `self._lock`; shared state accessed without synchronization.

### Step 4 -- Apply fixes

For category A and B issues (unless `--check-only`):
- Use `Edit` to make precise, minimal changes.
- Preserve existing comments and code style.
- After fixes, re-read the file to confirm the edit applied cleanly.

For category C issues: do NOT auto-fix. Report them for the developer to address.

### Step 5 -- Caller Impact Analysis (Section 6.2 of the standard)

Even when the diff itself looks clean, a change can silently break unchanged
callers. Before declaring the changes ready for PR, verify the global view for
each changed public symbol identified in Step 2.

**Triggers -- run this step if the diff does any of:**
- Adds, removes, renames, or reorders parameters of a public function/method
- Changes parameter types, defaults, or accepted domains (e.g., adds `None` to accepted values, narrows a type)
- Changes the return type or the meaning of sentinel returns (e.g., what `None` means)
- Adds, removes, or changes raised exceptions
- Changes side effects, ordering, blocking behavior, or thread-safety guarantees
- Modifies a method on a base class or protocol (subclasses are callers)
- Changes an invariant of a class field accessed by public API

If none of the above applies, skip this step.

**How to run it:**

1. For each changed public symbol, `Grep` its name across the whole repo
   (production code, `tests/`, `benchmarks/`, `lmcache/integration/`, `docs/`).
   Do not limit to files in the diff.
2. For commonly-named symbols with many false matches, use `Grep -A 2 -B 2`
   for context, or delegate to the `Agent` (Explore) tool with a focused prompt
   (e.g., "find all callers of `LMCacheMPWorkerAdapter.submit_store_request`
   and summarize the arguments each caller passes").
3. For base-class / protocol changes, also grep for subclasses
   (`class \w+\(<BaseName>`).

**For each caller, decide:**

- **Broken and must be fixed in this PR** (category A-like, fix it):
  - Removed / renamed parameter that the caller passes by keyword
  - Reordered positional parameters the caller relies on
  - Return type change that the caller destructures (e.g., tuple shape change)
  - Newly raised exception the caller doesn't handle, when the caller is on the normal success path

  For these, update the caller in the same pass as the main change. Apply the
  edit using `Edit`, then re-read to confirm. If the caller is in an unrelated
  module and the fix is non-trivial, flag it for manual review instead.

- **Behaviorally risky** (category C-like, flag for the developer):
  - Signature is still compatible but semantics changed (e.g., a return value
    that used to always be non-empty can now be empty)
  - Caller passes through to another function whose contract may now be violated
  - Base-class change where a subclass may need updating even though the
    signature is backwards-compatible

  Do NOT silently edit these -- the developer needs to confirm the intent. Flag
  them in the output with `path:line` of the caller and a one-line explanation.

- **Safe** (no action):
  - Backwards-compatible signature extension the caller does not use
  - Change in an internal detail not observable at the call site

Track the caller-impact findings under the existing output sections:
- Auto-fixed callers go under "Auto-fixed issues" with a new category "Caller updates"
- Flagged risky callers go under "Manual review required" with severity
  `error` if the caller is clearly broken and `warning` otherwise

### Step 6 -- Final verification

Run these checks and report the outcome. Do NOT auto-run them; print the command the developer should run (respecting the user's preference to run these themselves):

```bash
pre-commit run --all-files
pytest -xvs tests/v1/distributed/        # if distributed/ changed
pytest -xvs tests/v1/mp_observability/   # if mp_observability/ changed
```

## Output Format

### Summary
- Scope: `<uncommitted|staged|branch>`
- Files changed: `<N>`
- Issues found: `<total>` (auto-fixed: `<N>`, manual: `<N>`)

### Auto-fixed issues
Grouped by category, with file:line references. E.g.:

**Typing (3 fixes)**
- `lmcache/v1/foo.py:42` -- replaced `Optional[int]` with `int | None`
- `lmcache/v1/foo.py:87` -- added type hint for `limit: int` and return type `-> list[str]`
- `lmcache/v1/bar.py:120` -- replaced `dict` with `dict[str, int]`

**Runtime validation (1 fix)**
- `lmcache/v1/config.py:548` -- replaced `assert config.url is not None` with `if config.url is None: raise ValueError("url is required")`

**Docstrings (2 fixes)**
- `lmcache/v1/baz.py:15` -- added full docstring for public method `submit_request`
- `lmcache/v1/baz.py:60` -- updated docstring for `lookup` to reflect new `cache_salt` parameter

### Manual review required (needs developer judgment)

Grouped by severity:

**error** (must fix before PR):
- `lmcache/v1/foo.py:230` -- new feature `BatchProcessor.process_async` has no tests in `tests/v1/`. Add unit tests that exercise the public `process_async` API.
- `lmcache/v1/bar.py:15` -- `BarAdapter.do_thing` accesses `self._cache._internal_map` directly; use a public accessor or expose a method on `Cache`.

**warning** (should fix):
- `lmcache/v1/baz.py:42` -- `register(name: str, force: bool)` uses a boolean parameter. Consider splitting into `register(name)` and `register_overwrite(name)`, or using an enum `RegisterMode`.

**info** (suggestion):
- `lmcache/v1/baz.py:78` -- filter parameter is named `filter`; consider `key_eligible_filter` for clarity.

### Commands to run next

```bash
pre-commit run --all-files
# then run relevant test suites
```

### Summary table

| Category | Auto-fixed | Manual | Total |
|----------|-----------|--------|-------|
| Typing | 3 | 0 | 3 |
| Runtime validation | 1 | 0 | 1 |
| Docstrings | 2 | 0 | 2 |
| Design | 0 | 2 | 2 |
| Tests | 0 | 1 | 1 |
| **Total** | **6** | **3** | **9** |

## Rules

- Only touch files in the selected scope, **except** when Step 5 identifies an
  unchanged caller that is clearly broken by a signature change in this PR. In
  that case, update the caller in the same pass. Do not otherwise "improve"
  unchanged files.
- Never add error handling, fallbacks, or validation that the task does not require.
- Never introduce new abstractions, helpers, or dependencies in fixes.
- When a fix requires judgment (e.g., "what should this docstring say?" or "is
  this caller's assumption still valid?"), prefer to flag it for manual review
  rather than guess.
- Never run `git add`, `git commit`, or `git push`. The developer handles git operations themselves.
- Never run `pre-commit` or the test suite automatically; print the command for the developer to run.
- If the scope is `branch` and the branch has no upstream or is far behind `dev`, warn the developer but proceed.
- If `docs/coding_standards.md` is missing, stop and ask the developer to check it out or pull the latest `dev`.
