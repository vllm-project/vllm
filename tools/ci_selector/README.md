# ci-selector

Works out which vLLM CI jobs a diff needs to run. It derives the answer from the source tree and the Buildkite config, then checks that against a record of what each CI step actually executed.

## How it works

**The code map** reads the repo and the CI config (import graph, registries, step targets, container-build DAG) and works out which steps a diff could affect. When it cannot work something out it selects more, never less.

**The coverage record** is a table of what each step actually ran on real CI builds, one row per step, produced by an instrumented build.

Neither is a stage of the other. `decide.py` reads both, per changed file:

| for file F and step S | decides |
| --- | --- |
| S has a row, and it shows S ran F | **select**, and the map gets no vote |
| S has a row, and it shows S ran none of F | **drop**, if every gate agrees |
| S has no row, or F is outside the recorder's root | **the map decides** |

Selecting takes one observation and carries no gate. Dropping carries all of them.

## Setup

Needs a local vLLM checkout to analyze against.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

For the coverage half, put a table at `coverage-data/table.json.gz`, which is gitignored because it is a build artifact. Override the location with `--table` or `$CI_SELECTOR_TABLE`. Without one the selector runs on the code map alone and says so on stderr.

## Commands

```bash
# Both inputs. This is the answer CI would use.
ci-select --repo /path/to/vllm --diff origin/main...HEAD

# What the code alone says, for comparison and debugging.
ci-select codemap --repo /path/to/vllm --diff origin/main...HEAD
```

A two-ended range is required. `origin/main...HEAD` is the PR's merge-base diff, which is what CI sees. Output is JSON: the steps to run, why each was selected, and any run-all fallbacks.

### Step keys for CI

`--emit-keys` prints the selection as the key list Buildkite consumes, spelled the way the pipeline generator spells it rather than the way this tool does.

```bash
ci-select --repo /path/to/vllm --diff origin/main...HEAD --emit-keys
```

**It prints, it does not send.** Nothing in `.buildkite/` or `.github/` calls it yet. If anything goes wrong it prints no list at all, which leaves the generator to run everything. You never need to name image builds or other prerequisites, since the generator adds those itself.

## Validate

Checks that need more than a plain checkout. Anything checkable from the checkout alone is a drift-marked test instead, see Tests below. It exits 1 on a problem, and also when it finds nothing at all, because a detector that has stopped detecting looks like a clean result from the outside.

```bash
# How our selection compares to what really ran, and failed, on CI.
# Run after any change to selection. Needs `gh`.
ci-validate crosscheck --repo /path/to/vllm --prs 50378 47189
```

## Tests

```bash
# All tests.
uv run pytest tests -q        # offline
uv run pytest tests -q --sync # online (downloads ci-infra generator files first)

# Just the drift guards: the ones that fail when vLLM or ci-infra moved, rather
# than when our code is wrong. This is the set worth running in vLLM's own CI.
uv run pytest tests -m drift -q        # offline
uv run pytest tests -m drift -q --sync # online
```

A `drift` failure means a hardcoded fact went stale, so the fix is editing `handwritten.py` or teaching a parser. `pytest tests -m drift --collect-only -q` lists what is watched.

`tests/` covers the code map and needs a real vLLM checkout. Set `VLLM_REPO` if it is not in the usual place. `tests/coverage/` covers the coverage half and builds throwaway repos, so it needs nothing.
