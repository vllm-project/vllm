# LMCache Code Review Style Guide

This is the style guide for the LMCache project — a KV cache management engine
for LLM serving (vLLM/SGLang integration, GPU/CPU/disk/S3 storage tiers, CUDA
kernels, Rust raw-block I/O).

## Review philosophy

Do NOT care about security-related issues or small trivial matters. ONLY focus
on design decisions, architectural soundness, technical debt, code cleanliness,
modularity, and future maintainability and readability.

## Project conventions

Read these files for project standards:
- `docs/coding_standards.md` — **authoritative** coding quality and review standard
- `AGENTS.md` — coding conventions, testing practices, review checklist
- `CONTRIBUTING.md` — contribution guidelines

## What to check

### Convention Compliance & Documentation

- All new/modified `.py` files have `# SPDX-License-Identifier: Apache-2.0` as line 1
- All new functions have type hints (arguments + return values)
- All new public functions have docstrings (what, args, return, exceptions)
- Docstrings match the function's actual behavior
- No private member access (`_`-prefixed attributes) across class boundaries
- Import order: Standard / Third Party / First Party / Local (with section heading comments)
- Code passes ruff rules: E (pycodestyle), F (pyflakes), B (bugbear), SLF (self/private access)
- Formatting consistent with ruff (line-length 88) and isort (black profile, from_first=true)
- User-facing changes reflected in `docs/source/` if applicable
- Breaking changes explicitly called out
- New docs placed in correct subdirectory and linked from a toctree
- Design documents updated or added for non-trivial new features or architectural changes

### Testing

- New features include corresponding tests
- Bug fixes include regression tests
- Tests verify public interface and docstring contract, not implementation details
- No tests for private methods
- Test files are in the correct location under `tests/` matching the source structure

### Architecture & Design

- Changes consistent with existing codebase patterns
- New abstractions justified (not premature)
- Public APIs minimal and well-defined — no exposed internals
- Module-level helpers at top of file; private methods at end of class
- SLF discipline followed (no cross-class private member access), especially in
  `lmcache/v1/multiprocess/` and `lmcache/v1/distributed/` where CI enforces it

## Severity calibration

- **error** — MUST fix before merge. Examples: missing SPDX header, missing type
  hints on public function, missing docstring on public function, missing or
  inaccurate documentation for user-facing changes, new feature with zero tests,
  poor architectural decision, cross-class private member access in enforced
  directories, premature abstraction, tightly coupled modules.
- **warning** — SHOULD fix. Examples: docstring could be more detailed, tests
  that test implementation details, code could be more modular, naming could
  be clearer for maintainability.
- **info** — suggestion only, non-blocking. Examples: could improve naming,
  optional refactor for readability, minor style preference.

## Rules for the reviewer

- Do NOT report issues in unchanged code — only review lines in the diff.
- Do NOT praise the code. Only report issues.
- Do NOT pad findings. If the PR is clean, say so and move on.
- Be precise about file paths and line numbers.
- If you are unsure whether something is an issue, treat it as info, not error.
- Do NOT leave trivial or nitpick comments. If it doesn't affect correctness,
  maintainability, or readability in a meaningful way, don't comment on it.
