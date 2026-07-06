# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test/cuda.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test/cuda.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test/cuda.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Adding tests

Before writing a test, search the area you changed for an existing file and
harness to extend. Prefer adding cases to an existing test module and reusing
`conftest.py` fixtures, parametrization, and helpers in `tests/` over creating
a new file or bespoke setup. Add a new test file only when no nearby suite covers
the behavior.

Write tests that protect **observable behavior and regression-prone paths**, not
trivial wiring or implementation details that refactors should be free to
change. A passing test is not meaningful by itself — coverage of code that cannot
break (simple getters, one-line passthroughs) adds CI cost without signal.

Principles borrowed from large open-source projects:

- **Linux kernel** ([testing overview](https://docs.kernel.org/dev-tools/testing-overview.html)):
  match test scope to the code — behavior and whole features through their
  public interface; isolated units only when a regression test needs direct
  access. Fix a bug, then add a test so it stays fixed.
- **Kubernetes** ([writing good e2e tests](https://github.com/kubernetes/community/blob/main/contributors/devel/sig-testing/writing-good-e2e-tests.md)):
  reuse shared framework utilities and constants; extend them when something is
  generally useful. Follow stable nearby tests. Flaky tests are worse than no
  tests — make failures actionable.
- **PyTorch** ([test infrastructure](https://pytorch.org/blog/understanding-pytorchs-test-infrastructure/)):
  parameterize over existing fixtures instead of copy-pasting cases. Test through
  public APIs; keep tests atomic with no order or global-state dependencies.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

Follow these rules for all code changes in this repository:

- Try to match existing code style.
- Code should be self-documenting and self-explanatory.
- Keep comments and docstrings minimal and concise.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
