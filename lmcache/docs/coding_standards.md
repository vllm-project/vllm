# LMCache Code Quality and Review Standard

This document defines the coding quality standards and code review expectations for the LMCache project. All contributors (human and AI) should follow these standards. Reviewers should use this as the authoritative reference when reviewing PRs.

> **Source of truth**: This file is the canonical reference. `AGENTS.md` contains the
> quick-reference checklist; `CLAUDE.md` and `.gemini/styleguide.md` point here for details.

---

## 1. General Rules

### 1.1 PR Scope and Breakdown

- **PRs must be small and focused.** If the number of changed files is large, break it down conceptually into multiple PRs. Each PR should do one thing well.
- Prefer a series of small, reviewable PRs over one large omnibus PR.
- If a feature spans multiple modules, split by module boundary or by logical phase (e.g., interface change first, then implementation, then tests).

### 1.2 Documentation Requirements

- **User docs** (`docs/source/`) and **design docs** (`docs/design/`) must be updated whenever there are user-facing changes or new functionalities.
- **Docstrings** must be updated whenever there are changes to existing functions, methods, or class fields.
- If there are documents specifying schemas of arguments or return values, update them as well.
- **Code owners** must be updated if ownership changes. Reviewers should remind PR authors.

### 1.3 License Header

All Python files must have `# SPDX-License-Identifier: Apache-2.0` as the first line.

---

## 2. Typing

### 2.1 Strong Typing is Enforced

- All functions and methods must have type hints for arguments and return values.
- **Avoid `Any`**. If the type is not clear, redesign the interface until it is.
- Generic containers must be fully typed:
  - Use `list[SomeType]`, not bare `list`.
  - Use `dict[KeyType, ValueType]`, not bare `dict`.
  - Use `tuple[int, str]` or `tuple[int, ...]`, not bare `tuple`.
- **Avoid `Optional`**. Prefer always initializing objects (even if empty/unused) over wrapping in `Optional`. If a value can genuinely be absent, use `X | None` syntax rather than `Optional[X]`.

### 2.2 Runtime Validation

- **Never use `assert` for runtime-dependent checks** (e.g., config validation, input validation). Assertions are stripped by `python -O`. Use explicit `if ...: raise ValueError(...)` instead.
- Only use `assert` for development-time invariant checks that indicate programmer errors.

---

## 3. Docstrings

### 3.1 When Docstrings Are Required

| Scope | Requirement |
|-------|-------------|
| Public functions and methods | Full docstring (always required) |
| Module-level / global helper functions | Full docstring |
| Long private helper functions | Full docstring |
| Short, clear class-private helpers | Short docstring acceptable |
| Override methods with no new behavior | Short docstring acceptable |

### 3.2 Full Docstring Format

A full docstring must include:

1. **Summary**: A clear one-sentence description of what the function does.
2. **Details** (when needed): A few more sentences covering:
   - Thread safety guarantees or requirements
   - Assumptions about the caller/callee
   - Important invariants
3. **Args**: Detailed description of every argument, including:
   - Type and purpose
   - Assumptions or constraints (e.g., "length of `keys` must equal length of `values`")
4. **Returns**: Detailed description of the return value.
5. **Raises**: All exceptions the function may raise.
6. **Note** (when needed): Important assumptions or caveats that developers should be aware of.

### 3.3 Docstring Accuracy

- Docstrings must match the function's **actual current behavior**, not planned future behavior.
- When modifying a function, always update the docstring to reflect the change.
- If a parameter is currently a no-op (accepted but not yet used), the docstring must say so explicitly. Do not describe behavior that is not yet implemented.

---

## 4. Function and Interface Design

### 4.1 Naming

- Function names must be **self-explanatory**. A developer should understand the behavior without reading the implementation.
- Parameter names should imply semantics. For predicates/filters, name them to indicate what `True`/`False` means (e.g., `key_eligible_filter` rather than just `filter`).
- Variable naming must be consistent across files and modules.
- Private methods and private members start with `_`.

### 4.2 Self-Contained Functions

- Functions should be self-contained. If a function's input is expected to come from another specific function (or its output is expected to feed into another), this must be **documented explicitly** in the docstring.

### 4.3 Parameter and Return Value Design

- **Avoid ambiguity in return values.** For example, returning `None` for both "process not finished" and "request not found" makes it impossible for the caller to distinguish. Use distinct return types or raise exceptions.
- **Document schemas** if the parameter or return value includes `dict` or complex containers. Specify the expected keys, value types, and semantics.
- **Avoid boolean parameters.** They make call sites unreadable. Prefer:
  - Breaking into two separate functions, or
  - Using an `enum` to make intent explicit.
- **Clarify caller assumptions** in the docstring (e.g., "caller must hold the lock", "must be called from the main thread").

### 4.4 Class Design

- **Minimize the public interface.** Follow the principle of minimal surface area -- only add a new public method or property when there is a concrete need.
- **Class fields with internal state** must have:
  - A clear, documented meaning
  - A clear invariant (at any point in time, a developer should know what the value should be without running the code)
  - A docstring or inline comment explaining it
- **Private members must never be directly accessed by other classes.** Interact only through public APIs. This is enforced by CI in `lmcache/v1/multiprocess/` and `lmcache/v1/distributed/`, but applies project-wide.
- Prefer `@property` that derives from existing data over storing redundant fields.

### 4.5 Config and Dispatch

- Use `isinstance()` on config objects to determine behavior, not string-based type names.
- Config classes should follow existing conventions (e.g., `@dataclass`). Controllers and managers should always be created (even with empty/default config), never wrapped in `Optional`.

---

## 5. New Functionality

### 5.1 Design Documentation

- **A design doc is required** for non-trivial new features or architectural changes.
- **`docs/design/` mirrors the `lmcache/` package tree.** A design doc for code at
  `lmcache/<path>/` lives at `docs/design/<path>/`. For example, design docs for
  `lmcache/v1/distributed/l2_adapters/` go under
  `docs/design/v1/distributed/l2_adapters/`. A doc that spans several sibling
  submodules belongs at their common parent directory.
- Use descriptive file names (e.g., `overall.md`, `l2_eviction.md`, `query-command.md`)
  rather than mirroring module file names. See `docs/design/README.md` for the full
  convention.
- User-facing READMEs (`README.md` in module directories) stay next to the code and
  are symlinked from the corresponding `docs/design/<path>/` location — do not move
  them.

### 5.2 Testing

- All new features must include corresponding unit tests.
- All bug fixes must include regression tests.
- Tests must:
  - Verify the **public interface and docstring contract**, not implementation details.
  - Be written as if the function's implementation is unknown.
  - **Not access private members** (`_`-prefixed) of the class under test.
  - Not test private helper functions directly. Exercise them through the public API.
- Test files should mirror the source structure under `tests/`.

### 5.3 Reuse Existing Code

- When adding or updating functions, variables, or logic, check whether there is existing code that can be reused or existing state that can be exported instead of duplicating.

---

## 6. Modification to Existing Code

### 6.1 Documentation and Schema Updates

- **Must update docstrings** for any modified functions or fields.
- **Should update** user docs and design docs if the change affects behavior or APIs.
- Update schema documents if argument or return value structures change.
- Check for existing reusable code before introducing new helpers or utilities.

### 6.2 Caller Impact Analysis (Global View)

Any change to an existing function, class, or module must be evaluated in the
context of its **callers**, not just the file being edited. A local change that
looks correct in isolation can silently break unchanged code that depends on the
prior contract. Authors and reviewers must both take this global view.

**When caller analysis is required:**

- Adding, removing, renaming, or reordering function parameters
- Changing a parameter's type, default value, or accepted domain (e.g., allowing `None` where it wasn't before, or narrowing an accepted type)
- Changing the return type, return semantics, or the meaning of sentinel values (e.g., what `None` means)
- Adding, removing, or changing the exceptions a function raises
- Changing side effects, ordering guarantees, thread-safety guarantees, or blocking vs. non-blocking behavior
- Changing invariants of a class field that other classes read via a public accessor
- Modifying a public method that is part of an interface or base class (subclasses are callers too)

**How to perform the analysis:**

1. **Find every caller** of the changed symbol across the codebase, including:
   - Non-test production code
   - Unit tests and integration tests
   - Documentation snippets and examples
   - Downstream integrations (e.g., `lmcache/integration/vllm/`)
2. **For each caller, verify:**
   - The new signature is compatible (keyword-arg callers are especially sensitive to renames).
   - Caller assumptions about return values, exceptions, and side effects still hold.
   - If the caller passes the function's output to another function, that downstream contract still holds.
3. **Update callers in the same PR** when a breaking change is genuinely required. Do not leave callers broken with the intent to fix them later.
4. **Prefer backwards-compatible extensions** (new parameter with a default, new optional return field) over breaking changes unless there is a strong reason.
5. **If the change affects a base class or protocol**, check all implementers, not just direct callers.

**When in doubt, err on the side of reading one more caller.** The cost of
reading five call sites is small; the cost of a latent bug that ships because no
one looked is large.

---

## 7. Code Organization and Style

### 7.1 File Structure

- Module-level helper functions go **at the top** of the file (after imports, before classes).
- Private/helper methods within a class go **at the end** of the class, after all public methods.
- Code file length should not be too long. If a file is growing large, think about breaking it into smaller modules.

### 7.2 Import Ordering

Imports must follow this section-heading convention:

```python
# Standard
import os

# Third Party
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import helper
```

- All imports must be at the top of the file. No lazy imports.
- Always `import torch` before importing native C extensions (`lmcache.c_ops`, etc.).

### 7.3 Formatting and Linting

```bash
# Run all checks (mirrors CI)
pre-commit run --all-files

# Individual tools
ruff check .              # Lint (E, F, B, SLF rules)
ruff format .             # Format (line-length 88)
isort .                   # Import sorting (black profile, from_first=true)
mypy --config-file=pyproject.toml   # Type checking
codespell --toml pyproject.toml     # Spell checking
```

C++/CUDA files use clang-format (Google style, 80-col). Rust code uses `cargo fmt` and `cargo clippy`.

### 7.4 Logging

- Use the project's `init_logger(__name__)` from `lmcache.logging`, not bare `logging.getLogger()`.
- Operational logs (store/retrieve activity, background task progress) should be at `DEBUG` level, not `INFO`.
- Log levels must be appropriate: do not use `WARNING` for conditions that are expected during normal concurrent operation (e.g., a key not found due to a benign race condition).

### 7.5 Error Handling

- Replace `assert` with `if/raise` for all runtime validation (see Section 2.2).
- Error/failure paths must not leave the system in an inconsistent state. Check:
  - Locks are released
  - Pool entries are freed
  - File descriptors are closed
  - Partial state is cleaned up on failure

### 7.6 Resource Management

- Unbounded collections are memory leaks. If a set or dict grows over time (e.g., tracking seen keys), ensure entries are cleaned up when the corresponding resource is freed.
- CUDA/GPU resources must be properly managed (allocated, freed, synchronized).
- No unnecessary memory copies or allocations in hot paths.

---

## 8. Thread Safety

- All shared state access must be protected by appropriate locks.
- Lock granularity should be appropriate -- not too coarse (blocking unrelated operations) and not too fine (risking missed synchronization).
- Even test-only or debug methods must hold `self._lock` for thread safety.
- When code runs under multiple controller threads (e.g., StoreController + PrefetchController), verify that concurrent access is safe.
- Start with the simplest correct design. Only introduce threading, locks, and queues when there is a concrete need that cannot be met otherwise.

---

## 9. Code Review Process

### 9.1 Review Focus

Reviews should focus on:
1. **Design decisions and architectural soundness** -- is this the right approach?
2. **Correctness** -- logic bugs, error handling, resource management, edge cases.
3. **Caller impact (global view)** -- do the changes break any unchanged callers, tests, or subclasses? (See Section 6.2.)
4. **Thread safety** -- shared state, lock protocols, concurrent access patterns.
5. **Code cleanliness and maintainability** -- naming, modularity, readability.
6. **Documentation** -- docstrings, design docs, user docs.
7. **Test coverage** -- especially failure paths and concurrent access.
8. **Design doc compliance** -- does the implementation match documented contracts?

Reviews should **not** focus on:
- Security-related issues (out of scope for most changes)
- Trivial style matters already caught by linters
- Unrelated refactors of unchanged code (only flag unchanged code when it is a **caller of a changed symbol** and the change affects it; do not report unrelated issues in unchanged code)

### 9.2 Severity Levels

| Severity | Meaning | Examples |
|----------|---------|----------|
| **error** | Must fix before merge | Missing SPDX header, missing type hints on public function, missing docstring on public function, new feature with zero tests, cross-class private member access, poor architectural decision, ambiguous return values, `assert` used for runtime validation, **breaking change that leaves unchanged callers broken** |
| **warning** | Should fix | Docstring could be more detailed, tests that test implementation details, naming could be clearer, code could be more modular, missing `strict=True` on `zip` of parallel lists, **caller assumptions weakened without audit** |
| **info** | Suggestion only, non-blocking | Optional naming improvement, minor refactor for readability, style preference, **backwards-compatible extension where a cleaner API break might be justified** |

### 9.3 Reviewer Guidelines

- Only review lines in the diff. Do not report issues in unchanged code.
- Do not praise the code. Only report issues or confirm the PR is clean.
- Do not pad findings. If the PR is clean, say so.
- Be precise about file paths and line numbers.
- If unsure whether something is an issue, treat it as `info`, not `error`.
- Do not leave trivial or nitpick comments that do not affect correctness, maintainability, or readability.

### 9.4 Review Output Format

1. Start with a short summary of what the PR does.
2. Present **design doc compliance** as a table (requirement | pass/fail | notes) if a relevant design doc exists.
3. List issues grouped by severity (`error` > `warning` > `info`).
4. End with a summary table: severity | count | key items.

---

## 10. Quick Reference Checklist

Use this checklist before submitting a PR or during review:

### Correctness
- [ ] Code does what it claims; matches PR description
- [ ] Edge cases handled (empty inputs, `None` values, boundary conditions)
- [ ] Error/failure paths do not leave inconsistent state
- [ ] No regressions (existing tests still pass)

### Typing and Style
- [ ] `pre-commit run --all-files` passes
- [ ] All functions have type hints (arguments + return values)
- [ ] No use of `Any` or bare generic containers
- [ ] License header present on all Python files
- [ ] Import ordering follows section-heading convention

### Documentation
- [ ] All public functions have complete docstrings
- [ ] Docstrings match actual behavior
- [ ] Design docs updated for non-trivial changes
- [ ] User docs updated for user-facing changes

### Encapsulation and Design
- [ ] No cross-class private member access
- [ ] Public APIs minimal and well-defined
- [ ] Module-level helpers at top; private methods at end of class
- [ ] No boolean parameters (use enum or split into separate functions)
- [ ] No ambiguous return values

### Caller Impact (Global View)
- [ ] All callers of changed signatures still compile/pass type checks
- [ ] All callers still handle the new return type / new exceptions correctly
- [ ] Subclasses/implementers of modified base classes or protocols updated
- [ ] Tests and integration code (`tests/`, `lmcache/integration/`) checked, not just production paths

### Testing
- [ ] New features have corresponding tests
- [ ] Bug fixes have regression tests
- [ ] Tests target public interface, not implementation details
- [ ] Tests do not access private members

### Safety and Performance
- [ ] No `assert` for runtime validation (use `if/raise`)
- [ ] No unbounded collection growth
- [ ] Thread safety maintained for shared state
- [ ] No unnecessary memory copies in hot paths
- [ ] CUDA/GPU resources properly managed
