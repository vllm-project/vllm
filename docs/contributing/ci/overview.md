# vLLM CI Overview

This document describes the vLLM continuous integration (CI) system powered by
[Buildkite](https://buildkite.com/vllm). It covers how CI is triggered, what is
tested, how to add new tests, and how to investigate failures.

## Architecture

vLLM uses a **two-platform CI strategy**:

| Platform | Purpose |
|---|---|
| **GitHub Actions** | Fast pre-commit checks: linting (`ruff`, `clang-format`), type checking (`mypy`), shell/markdown linting, typo detection |
| **Buildkite** | All unit tests and integration/e2e tests that require GPU hardware |

The Buildkite pipeline is **dynamically generated** at runtime by the
[`pipeline-generator`](https://github.com/vllm-project/ci-infra/tree/main/buildkite/pipeline_generator)
tool. Rather than a static YAML file, the generator reads step definitions from
`.buildkite/` and produces a Buildkite pipeline tailored to exactly which files
changed in the PR.

### Pipeline Generator

The generator lives in the
[`vllm-project/ci-infra`](https://github.com/vllm-project/ci-infra/tree/main/buildkite/pipeline_generator)
repository. At pipeline startup it:

1. Reads step definitions from three directories in the vLLM repo:
    - `.buildkite/image_build/` — Docker image build jobs
    - `.buildkite/test_areas/` — GPU and standard test jobs
    - `.buildkite/hardware_tests/` — Alternative hardware (AMD, Intel, Ascend, Arm, GH200)
2. Computes the list of files changed in the PR (`git diff` against merge base).
3. For each step, evaluates whether it should run based on `source_file_dependencies`.
4. Groups steps, converts them to Buildkite format, and writes the pipeline YAML.

The pipeline configuration is controlled by `.buildkite/ci_config.yaml`:

```yaml
name: vllm_ci
job_dirs:
  - ".buildkite/image_build"
  - ".buildkite/test_areas"
  - ".buildkite/hardware_tests"
run_all_patterns:
  - "docker/Dockerfile"
  - "CMakeLists.txt"
  - "requirements/common.txt"
  - "requirements/cuda.txt"
  - "requirements/build.txt"
  - "requirements/test.txt"
  - "setup.py"
  - "csrc/"
  - "cmake/"
registries: public.ecr.aws/q9t5s3a7
repositories:
  main: "vllm-ci-postmerge-repo"
  premerge: "vllm-ci-test-repo"
```

### Documentation-Only Detection

If **all** changed files are under `docs/`, end with `.md`, or are `mkdocs.yaml`,
the generator skips all test steps and annotates the build as "doc-only". To
override this (e.g., for a doc change that also touches code), set
`DOCS_ONLY_DISABLE=1` in the build environment.

## Triggering CI

### Automatic Triggers

| Event | Behavior |
|---|---|
| PR labeled **`ready`** | Runs standard (non-optional) test suite |
| PR labeled **`ready-run-all-tests`** | Runs all tests including optional ones (`RUN_ALL=1`, `NIGHTLY=1`) |
| Merge to `main` | Post-merge test suite, filtered by diff between the new commit and its parent (same `source_file_dependencies` logic applies) |

!!! note
    CI does **not** run automatically when you open a PR. A maintainer must add
    the `ready` label. If your PR is ready for review and CI has not yet started,
    ask a maintainer to add the label or trigger a build.

### Changes That Always Trigger a Full Run

Modifications to any of the following always run the complete test suite,
regardless of which steps have `source_file_dependencies`:

- `docker/Dockerfile`
- `CMakeLists.txt`
- `requirements/common.txt`, `requirements/cuda.txt`, `requirements/build.txt`, `requirements/test.txt`
- `setup.py`
- `csrc/` (all CUDA/C++ sources, excluding `csrc/cpu/`, `csrc/rocm/`)
- `cmake/` (all CMake files, excluding `cmake/hipify.py`, `cmake/cpu_extension.cmake`)

## What Gets Tested

### Image Build (runs first)

All test jobs depend on a Docker image being built from the PR's code:

| Step | Hardware | Description |
|---|---|---|
| `image-build` | AWS builder | Main CUDA GPU image |
| `image-build-cpu` | AWS builder | CPU-only image |
| `image-build-hpu` | AWS builder | Intel Gaudi (HPU) image |

Images are pushed to AWS ECR (`public.ecr.aws/q9t5s3a7`):

- Pre-merge: `vllm-ci-test-repo:{commit}`
- Post-merge (main): `vllm-ci-postmerge-repo:{commit}`

### Test Areas (`.buildkite/test_areas/`)

These are the primary GPU test groups, each defined in a separate YAML file:

| Group | File | Key Tests |
|---|---|---|
| Attention | `attention.yaml` | V1 attention kernels (H100, B200) |
| Basic Correctness | `basic_correctness.yaml` | CUDA memory, basic correctness, CPU offload |
| Benchmarks | `benchmarks.yaml` | Latency/throughput benchmarks, CLI tests |
| Compile | `compile.yaml` | Sequence-parallel, AsyncTP, fusion tests |
| CUDA | `cuda.yaml` | Platform tests, CUDAGraph dispatch |
| Distributed | `distributed.yaml` | Collective ops, DP, 8×H100, multi-node |
| E2E Integration | `e2e_integration.yaml` | DeepSeek V2-Lite, Qwen3-30B accuracy (optional) |
| Engine | `engine.yaml` | Engine unit tests, V1 engine, scheduling |
| Entrypoints | `entrypoints.yaml` | OpenAI API (3 shards), RPC, pooling, responses API |
| Expert Parallelism | `expert_parallelism.yaml` | EPLB algorithm, execution, elastic EP scaling |
| Kernels | `kernels.yaml` | Core ops, attention, quantization, MoE, Mamba, DeepGEMM |
| LM Eval | `lm_eval.yaml` | GSM8K, GPQA accuracy evaluation |
| LoRA | `lora.yaml` | LoRA sharded (4× parallel), LoRA TP |
| Misc | `misc.yaml` | Spec decode, metrics/tracing, regression, examples |
| Model Executor | `model_executor.yaml` | Model executor, tensorizer |
| Model Runner V2 | `model_runner_v2.yaml` | Core, examples, distributed, spec decode |
| Models - Basic | `models_basic.yaml` | Initialization, CPU tests |
| Models - Distributed | `models_distributed.yaml` | Distributed model tests (2 GPUs) |
| Models - Language | `models_language.yaml` | Standard, hybrid (Mamba), pooling, perplexity, MTEB |
| Models - Multimodal | `models_multimodal.yaml` | Qwen2/3, Gemma, LLaVA, Whisper, multimodal processor |
| Plugins | `plugins.yaml` | Platform plugin, LoRA resolver, scheduler plugins |
| PyTorch | `pytorch.yaml` | Compile unit tests, H100 compile, fullgraph |
| Quantization | `quantization.yaml` | Quantization, Blackwell MoE (B200), quantized models |
| Ray Compat | `ray_compat.yaml` | Ray dependency compatibility (soft-fail) |
| Samplers | `samplers.yaml` | Samplers, FlashInfer sampler (with AMD mirror) |
| Spec Decode | `spec_decode.yaml` | Eagle, MTP, ngram+suffix, draft model |
| Weight Loading | `weight_loading.yaml` | Multi-GPU weight loading (optional) |

### Hardware Tests (`.buildkite/hardware_tests/`)

Tests for non-default hardware platforms:

| Group | File | Notes |
|---|---|---|
| CPU | `cpu.yaml` | Intel CPU (kernels, compat, generation, distributed) and ARM CPU |
| AMD | `amd.yaml` | ROCm Docker image build (gfx90a/942/950) |
| Intel HPU/GPU | `intel.yaml` | Intel Gaudi HPU and Intel GPU (XPU) tests (soft-fail) |
| Ascend NPU | `ascend_npu.yaml` | Ascend NPU tests (soft-fail) |
| GH200 | `gh200.yaml` | NVIDIA GH200 tests (soft-fail, optional) |

## Hardware Infrastructure

The pipeline routes jobs to different agent queues based on the `device` field in
each step definition:

### NVIDIA GPUs

| Device Value | Queue | Hardware |
|---|---|---|
| *(omitted)* | `GPU_1` | 1× NVIDIA L4 (AWS EC2, default) |
| `4_gpu` | `GPU_4` | 4× NVIDIA L4 (AWS EC2) |
| `a100` | `A100` | NVIDIA A100 (Kubernetes, Roblox/EKS) |
| `h100` | `MITHRIL_H100` | NVIDIA H100 (Kubernetes, Nebius/IBM) |
| `h200` | `H200` | NVIDIA H200 (Sky Lab) |
| `b200` | `B200` | NVIDIA B200 (standalone) |
| `gh200` | `GH200` | NVIDIA GH200 (standalone) |

### CPU and Other Hardware

| Device Value | Hardware |
|---|---|
| `cpu` | x86-64 CPU (AWS Elastic CI) |
| `cpu-small` / `cpu-medium` | Smaller CPU instances |
| `arm_cpu` | ARM64 CPU |
| `intel_cpu` | Intel CPU (standalone) |
| `intel_hpu` | Intel Gaudi HPU (standalone) |
| `intel_gpu` | Intel GPU / XPU (standalone) |
| `ascend_npu` | Ascend NPU (standalone) |

### AMD GPUs

AMD tests use `mi250_1/2/4/8`, `mi325_1/2/4/8`, or `mi355_1/2/4/8` device
values specifying both the GPU model and count. AMD mirrors can be added to
existing steps with the `mirror` field (see [Adding Tests](#adding-new-tests)).

## Step Definition Reference

Each test is defined as a YAML step inside a file in `.buildkite/test_areas/` or
`.buildkite/hardware_tests/`. Files are organized by group:

```yaml
group: "My Group"
depends_on: image-build
steps:
  - label: "My Test"
    timeout_in_minutes: 30
    source_file_dependencies:
      - vllm/my_module/
      - tests/my_tests/
    commands:
      - pytest -v -s tests/my_tests/test_foo.py
```

### Full Step Field Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `label` | str | required | Display name in Buildkite UI |
| `commands` | list[str] | `[]` | Shell commands to run |
| `device` | str | 1× L4 GPU | Hardware target (see [Hardware](#hardware-infrastructure)) |
| `num_devices` | int | — | Number of GPUs/devices required |
| `no_gpu` | bool | `False` | Run on CPU-only machine |
| `timeout_in_minutes` | int | — | Job timeout |
| `source_file_dependencies` | list[str] | `[]` (always runs) | File path prefixes that trigger this step |
| `working_dir` | str | `/vllm-workspace/tests` | Execution directory |
| `optional` | bool | `False` | Must be manually unblocked unless `NIGHTLY=1` |
| `soft_fail` | bool | `False` | Failure does not fail the pipeline |
| `parallelism` | int | — | Shard across N parallel agents; use `%N` in label |
| `depends_on` | list[str] | — | Step keys this step waits for |
| `key` | str | — | Unique step ID (referenced by `depends_on`) |
| `env` | dict | `{}` | Additional environment variables |
| `retry` | dict | — | Retry configuration |
| `no_plugin` | bool | `False` | Disable Docker/K8s plugins (standalone machines) |
| `mirror` | dict | — | Add an AMD mirror variant (see below) |

### `source_file_dependencies`

This field controls when a step is triggered. If **empty**, the step always runs.
If set, the step only runs when at least one changed file matches a listed prefix.

```yaml
source_file_dependencies:
  - vllm/attention/          # any change under this path triggers the step
  - tests/kernels/test_attn  # including partial filename matches
```

!!! tip
    Set `source_file_dependencies` as narrowly as possible to avoid running
    expensive tests for unrelated changes.

### AMD Mirror Steps

To run the same test on AMD hardware in parallel, add a `mirror` field:

```yaml
- label: "Samplers Test"
  commands:
    - pytest tests/samplers/
  mirror:
    amd:
      device: mi325_1
      depends_on: image-build-amd
```

This instructs the generator to create an identical step targeting AMD hardware.

## Required vs. Optional Tests

### Required Tests

Steps without `optional: true` run automatically whenever their
`source_file_dependencies` match changed files (or always, if empty). A failure
in a required step blocks the pipeline.

### Optional Tests

Steps with `optional: true` are **gated by a manual block** in Buildkite — they
do not run automatically on regular PRs. To unblock an optional step:

1. Open the Buildkite build for your PR (linked from GitHub CI checks).
2. Find the orange **block step** for the optional test.
3. Click **Unblock** to allow it to proceed.

Optional tests run automatically during nightly runs (when `NIGHTLY=1` is set,
triggered by the `ready-run-all-tests` label).

!!! note
    Optional tests are typically expensive, slow, or cover large model evals. Use
    the `ready-run-all-tests` label to run them all in one go, or manually unblock
    specific ones as needed.

### Soft-Fail Tests

Steps with `soft_fail: true` always run but their failure does **not** fail the
pipeline. These are typically experimental hardware tests (Intel HPU, Ascend NPU)
or informational tests where failures are expected.

## Adding New Tests

### Step 1: Find the Right YAML File

Look in `.buildkite/test_areas/` for the group that best matches your test.
For example, new attention kernel tests belong in `attention.yaml`; new model
tests belong in `models_basic.yaml` or `models_language.yaml`.

If your test covers a new area with no existing file, create a new YAML file
following the existing format.

### Step 2: Add the Step

```yaml
# In .buildkite/test_areas/my_area.yaml

group: "My Area"
depends_on: image-build
steps:
  - label: "My New Test"
    timeout_in_minutes: 30
    source_file_dependencies:
      - vllm/my_module/
      - tests/my_tests/
    commands:
      - pytest -v -s tests/my_tests/
```

### Step 3: Choose the Right Hardware

- **Default (omit `device`)**: 1× L4 GPU — suitable for most tests
- **`device: h100`**: Required for FP8, FlashAttention 3, DeepGEMM, or tests needing H100-specific features
- **`device: b200`**: For Blackwell (FP4/FP6, B200-specific kernels)
- **`device: 4_gpu`** + `num_devices: 4`: For tensor parallelism and multi-GPU tests
- **`no_gpu: true`**: For CPU-only tests

!!! warning
    Avoid over-requesting hardware. Use H100/B200 only when the test genuinely
    requires those capabilities — they are in higher demand and slower to schedule.

### Step 4: Set Dependencies Appropriately

```yaml
# Multi-GPU test
- label: "LoRA TP Test"
  num_devices: 4
  depends_on: image-build
  commands:
    - pytest tests/lora/test_lora_tp.py

# Test that requires another step to complete first
- label: "Downstream Test"
  depends_on:
    - image-build
    - upstream-test-key
  commands:
    - pytest tests/downstream/
```

### Step 5: Sharding Long Tests

For test files with many test cases, use `parallelism` to split them across
multiple agents:

```yaml
- label: "Models Language %N"    # %N is replaced with the shard index
  parallelism: 4
  timeout_in_minutes: 60
  commands:
    - pytest tests/models/language/ --shard-id=$BUILDKITE_PARALLEL_JOB
      --num-shards=$BUILDKITE_PARALLEL_JOB_COUNT
```

### Step 6: CPU / Alternative Hardware Tests

For CPU tests, add to `.buildkite/hardware_tests/cpu.yaml`:

```yaml
- label: "My CPU Test"
  device: intel_cpu
  no_plugin: true           # standalone machine, no Docker plugin
  timeout_in_minutes: 30
  commands:
    - pytest tests/my_module/test_cpu.py
```

For other hardware (Ascend, Intel HPU, GH200), add to the corresponding file in
`.buildkite/hardware_tests/` with `no_plugin: true` and `soft_fail: true`.

## Checking CI Results

### Buildkite Build Page

Every GitHub CI check links directly to the Buildkite build. The build page shows:

- **Green steps**: Passed
- **Red steps**: Failed (click to view logs)
- **Orange steps**: Blocked (optional steps awaiting manual unblock)
- **Grey steps**: Skipped (file dependencies did not match)
- **Annotations**: Benchmark results and special messages posted by the pipeline

!!! tip
    Click on a failed step, then **"View Log"** to see the full test output. Use
    `Ctrl+F` to search for `FAILED` in the log.

### CI Dashboard

The [vLLM CI Dashboard](https://vllm-ci-dashboard.vercel.app/) provides:

- **Builds**: Build history and status overview
- **Jobs**: Individual job details and durations
- **Queue**: Current queue depth and wait times
- **Performance**: Performance benchmark trends over time

### Test Analytics

Buildkite's test analytics tracks reliability per test over time:
👉 [Test Reliability on `main`](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&order=ASC&sort_by=reliability)

This is useful to identify tests that are consistently flaky.

## Investigating and Fixing CI Failures

See [CI Failures](failures.md) for a detailed guide on:

- Filing a CI failure issue
- Log wrangling and cleanup
- Bisecting failures on `main`
- Reproducing flaky test failures

### Quick Checklist

1. **Look at the `main` branch builds**: [Buildkite main](https://buildkite.com/vllm/ci/builds?branch=main) — does the failure also appear there?
2. **Reproduce locally**: Pull the CI Docker image for the exact commit and run the failing test.
   The commit SHA is visible in the Buildkite build page. For pre-merge builds use
   `vllm-ci-test-repo`; for post-merge (main) builds use `vllm-ci-postmerge-repo`:
    ```bash
    COMMIT=<commit-sha-from-buildkite>
    docker pull public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:$COMMIT
    docker run --gpus all --rm \
      -v $HOME/.cache/huggingface:/root/.cache/huggingface \
      public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:$COMMIT \
      pytest tests/failing/test_foo.py::test_name -v
    ```
3. **Check for flakiness** using `.buildkite/scripts/rerun-test.sh`:
    ```bash
    bash .buildkite/scripts/rerun-test.sh tests/failing/test_foo.py::test_name
    ```


## Common CI Patterns

### "Only My Files Changed, But Many Tests Run"

If you changed a file listed in `run_all_patterns` (e.g., `requirements/cuda.txt`,
`CMakeLists.txt`), the full test suite runs. This is intentional — changes to
build dependencies can affect all tests.

### "My Test Is Slow and Blocking PRs"

Mark it as `optional: true`. Optional tests run during nightly builds and can be
manually unblocked for specific PRs that need them.

### "I Need a Test to Run Only on Specific PRs"

Use narrow `source_file_dependencies`. The test only triggers when the listed
files are modified.

### "I Want to Run a Test on Both NVIDIA and AMD"

Add a `mirror` field to your step pointing to an AMD device:

```yaml
mirror:
  amd:
    device: mi325_1
    depends_on: image-build-amd
```

### "CI Passes But Tests Are Flaky"

1. Use `rerun-test.sh` to reproduce the flakiness.
2. File an issue using the
   [CI Failure Report template](https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml).

## Key Resources

| Resource | URL |
|---|---|
| vLLM CI on Buildkite | <https://buildkite.com/vllm> |
| vLLM CI Dashboard | <https://vllm-ci-dashboard.vercel.app/> |
| Test Analytics (reliability) | <https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&order=ASC&sort_by=reliability> |
| CI Infra Repo | <https://github.com/vllm-project/ci-infra> |
| Pipeline Generator | <https://github.com/vllm-project/ci-infra/tree/main/buildkite/pipeline_generator> |
| File a CI Failure | <https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml> |
