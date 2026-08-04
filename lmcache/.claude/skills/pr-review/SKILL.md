---
name: pr-review
description: Review a GitHub pull request against the LMCache coding standards
allowed-tools: Bash, Read, Grep, Glob, Agent, mcp__github__pull_request_read, mcp__github__list_pull_requests, mcp__github__search_pull_requests, mcp__github__add_reply_to_pull_request_comment
argument-hint: "<PR number or URL>"
---

# PR Review: LMCache Code Quality Standard

Review a pull request against the LMCache coding standards defined in `docs/coding_standards.md`.

## Inputs

`$ARGUMENTS` is a PR number (e.g., `3032`) or a GitHub PR URL.

## Review Process

### Step 0 -- Load the standards

Read `docs/coding_standards.md` at the repo root. This is the authoritative reference for
all quality checks. Also read `AGENTS.md` for the quick-reference checklist.

### Step 1 -- Understand the PR

1. Fetch the PR metadata (`get`) to understand the title, description, author, and base branch.
2. Fetch the diff (`get_diff`) to see exactly what changed.
3. Fetch the list of changed files (`get_files`) for a structural overview.
4. Read any design docs referenced in the PR description. `docs/design/` mirrors
   the `lmcache/` package tree, so for each changed module `lmcache/<path>/`
   check `docs/design/<path>/` for relevant docs.
5. For each **changed public symbol** (function, method, class, field), note its fully qualified name. You will use these in Step 4 to find callers.

### Step 2 -- Design Doc Compliance

If a relevant design doc exists (check `docs/design/<path>/` for each changed
module `lmcache/<path>/`; `docs/design/` mirrors the `lmcache/` package tree):

- Check that the implementation conforms to documented contracts, invariants, and assumptions.
- Present compliance as a table:

| Requirement | Status | Notes |
|-------------|--------|-------|
| ... | pass/fail | ... |

If no design doc is relevant, state that and skip this section.

### Step 3 -- Coding Quality Review

Review the diff against these standards (from `docs/coding_standards.md`):

**Typing (Section 2)**:
- All new/modified functions have type hints for arguments and return values.
- No use of `Any` or bare generic containers (`list`, `dict`, `tuple`).
- No `assert` used for runtime validation -- must use `if/raise ValueError`.

**Docstrings (Section 3)**:
- All new/modified public functions have complete docstrings (summary, args, returns, raises, notes).
- Docstrings match actual current behavior (not planned future behavior).
- Modified functions have updated docstrings.

**Function and Interface Design (Section 4)**:
- Function names are self-explanatory.
- No ambiguous return values (e.g., `None` meaning two different things).
- No boolean parameters where an enum or function split would be clearer.
- Dict/container parameters have documented schemas.
- Caller assumptions documented in docstrings.
- Class interfaces are minimal (no unnecessary public methods).
- No cross-class private member access.

**New Functionality (Section 5)**:
- Design doc present for non-trivial features.
- Tests cover new features (error-severity if missing).
- Tests verify public interface, not implementation details.
- Tests do not access private members.

**Modifications (Section 6)**:
- Docstrings updated for changed functions.
- User/design docs updated if behavior changes.
- Checked for reusable existing code.

**Code Organization (Section 7)**:
- License header present on new Python files.
- Import ordering correct (Standard / Third Party / First Party / Local).
- All imports at file top (no lazy imports).
- `import torch` before native C extensions.
- Uses `init_logger(__name__)` from `lmcache.logging`, not bare `logging.getLogger()`.
- Operational logs at DEBUG level (not INFO).
- Log levels appropriate (no WARNING for expected concurrent conditions).
- Module-level helpers at top; private methods at end of class.

**Resource Management (Section 7.5-7.6)**:
- Error paths release locks, free pool entries, close file descriptors.
- No unbounded collection growth (sets/dicts cleaned up when resources freed).
- No unnecessary memory copies in hot paths.
- CUDA/GPU resources properly managed.

### Step 4 -- Caller Impact Analysis (Section 6.2)

This step is what distinguishes a real review from a diff-reader. For each changed
public symbol identified in Step 1, verify the change is safe for **every unchanged caller**.

**Triggers -- run this step whenever the PR:**
- Adds, removes, renames, or reorders parameters of a public function/method
- Changes parameter types, defaults, or accepted domains (e.g., adds `None` to accepted values, narrows a type)
- Changes return types or the meaning of sentinel returns (e.g., what `None` means)
- Adds, removes, or changes raised exceptions
- Changes side effects, ordering, blocking behavior, or thread-safety guarantees
- Modifies a method on a base class or protocol (subclasses are callers)
- Changes an invariant of a class field accessed by public API

**How to find callers:**

1. Use `Grep` with the symbol name across the repo (do not filter to only changed files).
   Include `tests/`, `benchmarks/`, `lmcache/integration/`, and `docs/` in the search.
2. For common function names that may have many false matches, use `Grep` with
   `-A 2 -B 2` to see context, or use the `Agent` (Explore) tool with a focused
   prompt (e.g., "find all callers of `L1Manager.is_key_evictable` and summarize
   what they do with the return value").
3. For base-class/protocol changes, also grep for subclasses
   (`class \w+\(<BaseName>`).

**For each caller, check:**
- Is the new signature compatible with this call site? (Keyword arg renames are especially dangerous.)
- Does the caller still handle the new return type / new exceptions correctly?
- Does the caller pass the output to something else whose contract would now be violated?
- If the change is backwards-compatible at the signature level, is it also backwards-compatible at the **semantic** level (e.g., does a now-optional param change default behavior)?

**Flag as error severity:**
- Any unchanged caller that is now broken and not updated in the PR.
- Any protocol/base-class change where at least one implementer is not updated.

**Flag as warning severity:**
- Callers whose assumptions are weakened by the change and whose correctness is not obviously preserved.

Report findings as part of the issues list, with the path:line of the affected caller and a one-line explanation of the impact.

### Step 5 -- Thread Safety (Section 8)

- Shared state protected by locks.
- Lock granularity appropriate.
- Concurrent access from multiple controller threads is safe.
- Even debug/test-only methods hold locks when accessing shared state.

### Step 6 -- Test Coverage (Section 5.2)

- New features have tests (error-severity if missing).
- Bug fixes have regression tests (error-severity if missing).
- Tests verify public API and docstring contract.
- Tests do not access private members or test private functions directly.
- Note what is tested and what is missing (especially failure paths and concurrent access).

### Step 7 -- PR Structure (Section 1.1)

- Is the PR appropriately scoped? If too many changed files, suggest how to break it down.

## Output Format

### Summary
One paragraph describing what the PR does.

### Design Doc Compliance
Table (if applicable) or "No relevant design doc."

### Issues

Group issues by severity, following the calibration from `docs/coding_standards.md` Section 9.2:

**error** (must fix before merge):
- Missing SPDX header, missing type hints, missing docstrings on public functions,
  new feature with zero tests, bug fix with no regression test, cross-class private
  member access, poor architectural decision, ambiguous return values, `assert` for
  runtime validation, docstring that misrepresents behavior.

**warning** (should fix):
- Docstring could be more detailed, tests that test implementation details, naming
  could be clearer, code could be more modular, missing `strict=True` on `zip` of
  parallel lists, log level too high for operational messages.

**info** (suggestion, non-blocking):
- Optional naming improvement, minor refactor for readability, style preference.

For each issue, include:
- File path and line number(s)
- What the issue is and why it matters
- Suggested fix (if straightforward)

### Summary Table

| Severity | Count | Key Items |
|----------|-------|-----------|
| error    | N     | ... |
| warning  | N     | ... |
| info     | N     | ... |

## Reviewer Rules

- The review is scoped to the diff, with **one exception**: unchanged callers of
  changed symbols (Step 4). Flag a caller in unchanged code only when the PR's
  changes demonstrably affect it; do not review unrelated issues in those files.
- Do NOT praise the code. Only report issues or confirm the PR is clean.
- Do NOT pad findings. If the PR is clean, say "PR is clean" and list what was checked.
- Be precise about file paths and line numbers.
- If unsure whether something is an issue, treat it as `info`, not `error`.
- Do NOT leave trivial or nitpick comments that linters would catch.
- Focus on design decisions, architectural soundness, correctness, caller impact, thread safety, code cleanliness, modularity, and maintainability.
