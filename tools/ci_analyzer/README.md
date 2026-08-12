# ci_analyzer

Deterministic, derived CI test selection for vLLM. Give it a PR diff and it returns the Buildkite jobs to run, computed from vLLM's import graph and `.buildkite` config. No hand-maintained code-to-test map.

## Setup

Requires Python 3.12+ and a local vLLM checkout to analyze against.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

## Usage

### Select jobs for a diff

Point it at a vLLM checkout and give it a git range. `origin/main...HEAD` is the PR's merge-base diff (what CI sees):

```bash
ci-select --repo /path/to/vllm --diff origin/main...HEAD
```

It prints JSON: the steps to run, why each was selected, and any run-all fallbacks. Alongside the prose reasons, `selected_rules` gives the rule name behind each one (`graph`, `declared-deps`, `fail-open`, ...). Those names are a stable contract, pinned in `policy.RULES`, because downstream filtering routes on them. To pass changed paths directly instead of a git range, use `--files a.py b.py`.

### Validate

`ci-validate` bundles six confidence checks. They run against a vLLM checkout and are not part of the per-PR path. Each one exits 1 when it finds a problem, and also when it finds nothing at all: a detector that has stopped detecting looks identical to a clean result from the outside.

```bash
# Replay real PRs: our selection vs the jobs that actually ran and failed on CI.
# Needs `gh` authenticated.
ci-validate crosscheck --repo /path/to/vllm --prs 50378 47189

# Prove the import graph has no unmodeled dynamic imports (exits 1 if any).
ci-validate dynamic-sites --repo /path/to/vllm

# Flag docs cross-references that no longer resolve (exits 1 if any).
ci-validate docs-refs --repo /path/to/vllm

# List test files that no live CI job invokes.
ci-validate uninvoked --repo /path/to/vllm

# Audit config-gated plugin demotion; --census lists seam amplifiers.
# Exits 1 if a demoted plugin routes to zero tests (a routing gap).
ci-validate demoted-plugins --repo /path/to/vllm

# Prove the subtractive passes left every dropped edge another route.
ci-validate dropped-edges --repo /path/to/vllm
```
