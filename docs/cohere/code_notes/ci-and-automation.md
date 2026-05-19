# Code Notes: CI and Automation

## 1) Pipeline Topology and Control Flow

Main execution chain:

- `build-and-test.yaml` decides whether to build images first (nvidia, amd, cpu), then fans out by GPU for **feature** coverage only (`benchmarks` fixed to `none`).
- `build-and-eval.yaml` reuses the same build + GPU fanout pattern for eval suites (`lm_eval`, `bee_eval`) and forces `features: none`.
- `build-and-bench.yaml` reuses the same build + GPU fanout pattern, but exposes only performance benchmark entrypoints (`perf_100`, `perf_1000`) and forces `features: none`.
- `dispatcher.yaml` computes the test matrix (feature tests + eval/perf groups). CPU tests are included when `features` is `fast_check` or `all`.
- `test-pipeline.yaml` runs each matrix entry on mapped runners and handles uploads. CPU tests run inside a pre-built `vllm-cpu` Docker image on `ubuntu-latest`.

Design intent:

- Separate image production from test orchestration, so tests can reuse either "just built" images or an explicit image tag.
- Keep matrix construction in JS for complex TP/model validation logic. The workflow uses `actions/github-script` as a thin wrapper that calls `.github/scripts/dispatcher-set-matrix.js`.
- Treat GPU as a first-level dimension and TP/model grouping as second-level dimensions.
- CPU tests use a dedicated `vllm-cpu` image built from `docker/Dockerfile.cpu` (AVX2-only, no AVX-512) to ensure compatibility with `ubuntu-latest` runners.

## 1b) Commit-SHA Pinning and `nightly-benchmark.yaml`

All entry-point workflows (`build-and-test.yaml`, `build-and-eval.yaml`,
`build-and-bench.yaml`, `nightly-benchmark.yaml`) pin the branch to a resolved
commit SHA in a `resolve-commit` job before any downstream work begins.
This prevents a push mid-run from causing the build and test steps to diverge
on different commits.

`/.github/workflows/nightly-benchmark.yaml` is the top-level nightly entry for
Cohere benchmark/eval coverage:

- triggers `build-and-push.yaml` once for `nvidia`, `amd`, and `cpu`,
- dispatches nightly eval/perf coverage by GPU through `dispatcher.yaml`,
- relies on `test-pipeline.yaml` for actual on-runner execution and reporting.

Current nightly routing behavior:

- H100 / GB200 / MI300X perf jobs dispatch with `features: none` and
  `benchmarks: perf_1000`,
- B200 dispatches with `features: all` and `benchmarks: perf_100`,
- A100 currently dispatches `features: fast_check` with `benchmarks: none`.

Reporting consequence:

- `nightly-benchmark.yaml` itself does not render a GitHub Actions test report;
  downstream reporting happens in `test-pipeline.yaml`,
- the existing `Test Report` step in `test-pipeline.yaml` currently runs for
  pytest-based `fast_check`, `cpu`, and `model_arch_*` groups,
- new nightly tests that should appear as a real GitHub Actions `Test Report`
  must run through a pytest/JUnit-producing path and may require widening that
  condition,
- the `model_arch` feature expands into two independent jobs
  (`model_arch_reward`, `model_arch_c5_3a30t`), each dispatched to its own
  runner via `runner_map.json`,
- the `quantization` feature expands into one independent job
  (`quantization_32bit_logits`), dispatched via `runner_map.json`, with JUnit
  test reporting and `unit_results_summary.json` upload,
- nightly benchmark/eval artifacts still flow through the result-upload path
  described below, rather than the `Test Report` UI.

## 1c) Incremental Build Safety Check

`build-and-push.yaml` supports an **incremental build** mode (enabled by
`incremental_build: true`). Instead of compiling vLLM from scratch, it finds
a previously built `base-<sha>` image in the container registry and layers
only the Cohere-specific `Dockerfile.cohere` on top, which reinstalls vLLM
Python sources with `pip install -e . --no-deps`.

Because the incremental path skips C++ compilation and dependency
installation, it is only safe when nothing outside pure Python sources has
changed since the base image was built. The "Find latest base image" step
enforces this by diffing each candidate base commit against HEAD for a set of
**rebuild-trigger patterns**:

```text
requirements/  pyproject.toml  setup.py  setup.cfg
docker/        CMakeLists.txt  cmake/    csrc/
Makefile
```

If any of these paths changed between the nearest base image commit and HEAD,
the build immediately falls back to a full rebuild (older base images are
guaranteed to be equally or more stale, so searching further is pointless).
If no base image exists within 200 commits, the build also falls back to a
full rebuild.

Callers that pass `incremental_build: true` (`build-and-test.yaml`,
`build-and-eval.yaml`, `build-and-bench.yaml`) rely on this guard to avoid
silently stale images. Manual `workflow_dispatch` defaults to
`incremental_build: true` but also exposes a `force_build` toggle to override
the image-exists check entirely.

When adding new files or directories whose changes should invalidate the base
image, add the path to the `REBUILD_PATTERNS` array in the "Find latest base
image for incremental build" step of `build-and-push.yaml`.

## 2) Dispatcher Matrix Logic (Important Behavioral Contract)

The matrix builder in `.github/scripts/dispatcher-set-matrix.js` (invoked by `dispatcher.yaml`) does more than routing:

- Validates requested GPU exists in `tests/cohere/configs/runner_map.json`.
- Resolves tensor parallel from the `models` CSV:
    - no `:tpN` suffix -> use per-model `recommended_tp` from `tests/cohere/configs/tp_model_map.json`
    - optional `model:tpN` suffix -> forces that TP for every listed model in the run (all entries must use the same `:tpN`); each model must still satisfy `minimum_tp`
- Performance-only runs may also pass public Hugging Face repo IDs directly (for example `org/model:tp1`). These must include an explicit `:tpN` because they are not present in `tp_model_map.json`.
- Groups models by selected TP so each job can batch model lists efficiently. Exception: `bee_eval` creates one matrix entry per model so multi-hour eval runs execute on separate runners in parallel.
- Expands feature tests independently from eval/perf tests.
- Adds a `cpu` matrix entry (with `gpu: "cpu"`, `runner_labels: ["ubuntu-latest"]`) when `features` is `fast_check` or `all`.

Why this matters:

- This creates an explicit compatibility gate between requested model/GPU/TP combos and real runner capacity.
- If rebasing loses this logic, failures shift from "early validation" to "late runtime OOM/timeouts".

## 3) Result Upload Semantics (CI Dump vs GH Pages)

`test-pipeline.yaml` routes uploads by explicit result-upload branch:

- default entry-workflow value -> `ci_dump`
- `nightly-benchmark.yaml` scheduled runs -> `gh-pages` (manual dispatch defaults to `ci_dump`)

`upload-results` action behavior:

- appends JSON record(s) to a per-path, per-GPU aggregate list,
- optionally annotates records with commit and CI metadata,
- stores run traceability fields (`run_id`, `run_attempt`, `run_number`, workflow, URL).

Operational effect:

- reporting dashboards can consume one append-only history per metric family,
- nightly and ad hoc data remain separated but generated by the same code path.

Risk note:

- Action currently performs branch switching and hard reset internally; if this action drifts during rebase, reporting can silently break while tests still pass.

## 4) Auto-Rebase / Rerere Workflow

`auto-rebase-upstream.yaml` is a full rebase automation pipeline, not just a simple `git rebase` wrapper:

- checks out `{source}-squashed`,
- recreates `{source}-synced`,
- restores rerere cache from Actions cache and optional branch artifact,
- rebases via shared script, handling "conflicts expected" as a first-class outcome,
- emits conflict-handling guidance and snapshots conflict state branch when unresolved.

Design intent:

- amortize repeated conflict resolution effort through rerere reuse,
- preserve a deterministic branch lifecycle (`-squashed` -> `-synced`),
- automate weekly upstream sync while still supporting manual intervention.

## 5) Cohere Test Entry Script as CI Contract

`tests/cohere/scripts/run_tests.sh` functions as the on-container execution contract:

- selects behavior by `TEST_GROUP`,
- reads `MODELS`, `TP_SIZE`, `GPU_TYPE` envs from workflow,
- skips checkpoint pre-download for performance runs when every requested model is a public Hugging Face repo ID, while still pre-downloading internal GCS-backed checkpoints,
- stages checkpoint and Hugging Face cache downloads in temporary directories before moving them into place, so workflow-level retries do not treat partial copies as complete artifacts,
- performs SHM cleanup to avoid stale buffer collisions,
- supports `bee_eval` and `lm_eval` as independent test groups (nightly
  dispatch should request each benchmark explicitly when both are needed),
- runs perf and feature test groups with shared conventions.

Important detail:

- CI workflow and this script are tightly coupled; this script itself calls out that it needs periodic sync with `test-pipeline.yaml`.

## 5b) Bee Eval Config Pipeline

`generate-serving-config.py` and `run-bee-eval.sh` form a two-stage config pipeline for bee eval runs:

1. **Config generation** (`generate-serving-config.py --mode eval`): reads `eval-config.json` to determine which tests each model runs, then emits per-model entries in `serving-cohere-tests.json` with `server_parameters` and `eval_parameters`. Model-conditional logic (C5 tool-call parsers, thinking budgets) lives here.
2. **Config consumption** (`run-bee-eval.sh`): iterates over generated configs, converts `server_parameters` to CLI args via `json2args`, reads `eval_parameters` for bee command construction including optional `--extra_body` from `thinking_token_budget`/`reasoning_thinking_token_budget` fields.

Design intent:

- All model-specific behavioral knobs are centralized in the Python generator; the shell script is a generic executor that reads whatever the config provides.

## 6) Change Hotspots and Verification Checklist

High-conflict files:

- `.github/workflows/dispatcher.yaml`
- `.github/workflows/test-pipeline.yaml`
- `.github/actions/upload-results/action.yaml`
- `.github/workflows/auto-rebase-upstream.yaml`
- `tests/cohere/scripts/run_tests.sh`
- `tests/cohere/scripts/generate-serving-config.py`

Post-rebase sanity checks:

1. Trigger a small `fast_check` matrix run.
2. Trigger one eval run with non-default TP (for example `models` including a shared `:tpN` suffix).
3. Verify result JSON append in target branch.
4. Confirm metadata annotation fields are present for performance uploads.
5. Validate dispatcher rejects invalid TP/model combinations.

## 7) Test Pipeline Integration

Reference this section when writing or reviewing tests, feature docs, or CI
workflows. It captures the conventions that all Cohere tests must follow to
integrate correctly with the CI pipeline.

### JUnit XML Reporting (pytest)

Tests whose `TEST_GROUP` is listed in the `dorny/test-reporter` condition in
[`/.github/workflows/test-pipeline.yaml`](../../../.github/workflows/test-pipeline.yaml)
**must** be invoked through `pytest` so that JUnit XML reports are generated
automatically.

Current condition (keep in sync with the workflow):

```text
fast_check | cpu | quantization_32bit_logits | model_arch_c5_3a30t
```

How the pipeline produces reports:

1. The Docker container mounts `${{ runner.temp }}` as `/root/output`
   (`OUTPUT_DIR`).
2. A `pytest_configure` hook in
   [`/tests/conftest.py`](../../../tests/conftest.py) detects `OUTPUT_DIR` and
   sets `config.option.xmlpath` to write a timestamped JUnit XML file inside
   that directory.
3. After the test step, `dorny/test-reporter@v2` looks for
   `${{ runner.temp }}/*.xml` on the host. If no XML files exist, **the step
   fails the entire job** even when the tests themselves passed.

Rules for new or modified tests:

- **Never run test scripts with plain `python3`** when the test group appears
  in the reporter condition. Always use `pytest` in `run_tests.sh`.
- If a test is a standalone script (argparse-based), add a `test_*` pytest
  function that reads configuration from environment variables and delegates to
  the existing logic. Keep the `__main__` entry point for local convenience.
- When adding a new `TEST_GROUP` to the reporter condition, verify the
  corresponding `run_tests.sh` function invokes `pytest`, not `python3`.
- The `conftest.py` hook is automatic -- no `--junitxml` flag is needed on the
  pytest command line as long as `OUTPUT_DIR` is set.

### Hardware Profiles

[`/vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml)
defines per-GPU runtime args and env vars for vLLM serving and benchmarks.
The YAML lives inside the package and is bundled into the wheel via
`setup.py` `package_data`, so the profiles travel with whatever vLLM is
installed in the container.

How profiles are applied:

1. The runtime hook lives at the end of `EngineArgs.__post_init__` in
   [`/vllm/engine/arg_utils.py`](../../../vllm/engine/arg_utils.py): when
   `VLLM_ENABLE_COHERE_AUTO_CONFIG` is truthy (`1` / `true`), it calls
   `apply_cohere_auto_config(self)` from
   [`/vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py),
   which (a) probes the HF config for a Cohere architecture, (b) resolves
   matching profiles via CEL `when:` clauses bound to `{server.type, gpu.name}`,
   (c) merges `vllm-default` with the GPU-specific overlay, then (d) mutates
   the live `EngineArgs` instance in place. User-supplied fields (live value
   != dataclass default) are preserved.
2. CI shells opt in once at the top of each script with
   `export VLLM_ENABLE_COHERE_AUTO_CONFIG=1`:
   [`run_tests.sh`](../../../tests/cohere/scripts/run_tests.sh),
   [`run-bee-eval.sh`](../../../tests/cohere/scripts/run-bee-eval.sh),
   [`run-performance-benchmarks.sh`](../../../tests/cohere/scripts/run-performance-benchmarks.sh).
   Every spawned `vllm serve` / `vllm bench latency` / `vllm bench throughput`
   / pytest subprocess inherits the env var and self-configures.
3. Python entry points that build `LLM(...)` / `AsyncEngineArgs(...)`
   directly call
   `os.environ.setdefault("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")` in their
   `main()` / `__main__` / pytest entry function so standalone invocations
   self-configure without depending on the shell wrapper.
4. Profiles are applied in YAML order; later profiles override earlier ones.
   The `vllm-default` profile is always applied as a baseline. Profile `env:`
   entries only land when the env var is currently unset (pre-existing
   `os.environ` values win and log at INFO).

For the runtime-side contract (gate semantics, env-cache rationale, error
swallowing) see
[Cohere Auto-Config](runtime-and-scheduling.md#8-cohere-auto-config-hardware-profile-application).
For CPU unit-test coverage see
[`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py)
([feature doc](../tests/features/auto_config.md)).

Rules for new or modified tests:

- **Respect hardware profiles by default.** Tests that use the vLLM engine
  for serving or benchmarking should opt in to auto-config (env var) rather
  than hard-coding GPU-specific settings; profile values fill in
  dataclass-default fields automatically.
- Prefer enabling `torch.compile` and CUDA graphs when compatible with the
  test. Call out any intentional deviation in the test doc's `## Setup`
  section.
- If a test needs to override a profile value (e.g. a specific attention
  backend), pass it explicitly to `LLM(...)` / `AsyncEngineArgs(...)` --
  user-set fields are preserved by `apply_cohere_auto_config`. Document the
  override and reason in the test doc.
- When adding a new GPU profile, update
  [`/vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml)
  and add coverage in
  [`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py)
  for any non-trivial CEL `when:` clause or new keys.

### Nightly Metric Emission

[`/.github/workflows/nightly-benchmark.yaml`](../../../.github/workflows/nightly-benchmark.yaml)
is the top-level nightly CI entry. It triggers image builds, then fans out
eval/perf/feature jobs per GPU via
[`dispatcher.yaml`](../../../.github/workflows/dispatcher.yaml) and
[`test-pipeline.yaml`](../../../.github/workflows/test-pipeline.yaml).

Metric upload path:

1. Nightly runs set `result_upload_branch: gh-pages`.
2. Ad-hoc / PR runs default to `ci_dump`.
3. [`test-pipeline.yaml`](../../../.github/workflows/test-pipeline.yaml) uses
   the [`.github/actions/upload-results`](../../../.github/actions/upload-results)
   action to append JSON records to per-path, per-GPU aggregate lists on the
   target branch.

Current upload triggers in `test-pipeline.yaml`:

| `test_group` | Artifact | Upload path on target branch |
| --- | --- | --- |
| `performance` | `benchmark_results_summary.json` | `data/summary` |
| `bee_eval` | `eval_results_summary.json` | `eval_data/summary` |
| `quantization_32bit_logits` | `unit_results_summary.json` | `unit_data/summary` |

Rules for new or modified tests:

- **Tests that emit benchmark metrics must write a summary JSON** to
  `$OUTPUT_DIR` so the upload step can find it.
- When adding a new uploadable `TEST_GROUP`, add a corresponding upload step
  in `test-pipeline.yaml` and document the artifact name and upload path here.
- Feature docs should reference the nightly reporting path (`gh-pages`) in
  their `## Measurements` section for any metric tracked over time.
- Prefer the shared metric pattern codes in
  [`observability_matrix.md`](../tests/observability_matrix.md#metric-pattern-codes)
  when defining expected values for benchmark metrics.

## 8) Manual Workflow Dispatch Guardrail

When manually triggering GitHub Actions with `gh`, always scope commands to this repository:

- use `-R cohere-ai/vllm-cohere` with `gh workflow run`, `gh run list`, and `gh run view`.

Why:

- without explicit repo scoping, `gh` can target upstream remotes (for example `vllm-project/vllm`),
- this can produce confusing 404 errors for valid workflow names or dispatch runs in the wrong repository.
