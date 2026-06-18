# Testing Guide (Cohere)

## TL;DR

- we have entry workflows [build-and-test](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-test.yaml) (feature tests), [build-and-eval](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-eval.yaml) (`lm_eval` / `bee_eval`), and [build-and-bench](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-bench.yaml) (perf benchmarks) to build docker images against a commit and run tests with those images
- the testing groups are: `cpu`, `fast_check`, `model_arch`, `quantization` (expands to `quantization_32bit_logits`), `GG` (expands to `guided_generation`), `thinking_budget` (expands to `thinking_budget`, `bee_sample_tb_check`), `lm_eval`, `bee_eval`, `performance`, `speculative_decoding`, `vision`, `asr` (details below)
- `fast_check` and `all` also trigger CPU tests in parallel on a `ubuntu-latest` runner
- CPU tests also run automatically on every PR to `cohere` via `pr-cpu-tests.yaml`
- to kick off tests, you can use the GitHub Actions UI -> `build-and-test` / `build-and-eval` / `build-and-bench` -> new workflow, or use `gh cli`
- runner hardware options: `h100, mi300x, a100, b200, gb200, all`
- tensor parallel: omit suffix on each model to use `recommended_tp` from `tests/cohere/configs/tp_model_map.json`, or add `model:tpN` (e.g. `command-r7b_fp8:tp1`). If any entry uses `:tpN`, every entry in the same run must use the same `:tpN`.

```bash
# Run eval benchmarks (lm_eval/bee_eval); default TP per model from tp_model_map.json
gh workflow run build-and-eval.yaml --ref <sha> -f evaluations=bee_eval -f models=command-r7b_fp8

# Same run with an explicit shared TP suffix on all listed models
gh workflow run build-and-eval.yaml --ref <sha> -f evaluations=bee_eval -f models=command-r7b_fp8:tp1

# Run perf benchmarks; optional :tpN on models the same way
gh workflow run build-and-bench.yaml --ref <sha> -f benchmarks=perf_100 -f models=command-r7b_fp8:tp1

# Run feature tests (fast_check, GG, thinking_budget, speculative_decoding, etc.)
gh workflow run build-and-test.yaml --ref <sha> -f features=all

# Run on AMD GPU
gh workflow run build-and-test.yaml --ref <sha> -f gpu=mi300x -f features=fast_check

# Override the checkpoint prefix (defaults to gs://cohere-model-efficiency-ci/engines/)
gh workflow run build-and-eval.yaml --ref <sha> -f model_path=gs://cohere-model-efficiency-ci/engines/
```

- `--ref` tells GHA where to look for the workflow file and which ref to run against.
- the job will fail if any of the tests fail; for pytest based test groups you can see a summary of the output in the job UI.

NOTE: if you specify a branch in `--ref`, the workflow may fail if you make changes to the branch in the middle (between image build and test start). It is safer to use the commit sha.

## Running Tests Locally

### Prerequisites

1. **Install dependencies** (from repo root):

   ```bash
   cd tests
   bash cohere/scripts/setup_tests.sh
   ```

   This will:
   - Auto-detect your GPU platform (NVIDIA/AMD)
   - Install test dependencies
   - Reinstall vLLM in editable mode using precompiled wheels for C++/CUDA extensions
   - This allows testing local source code changes while using optimized compiled extensions
   - Hardware profiles are applied automatically at engine boot via [`vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py) when `VLLM_ENABLE_COHERE_AUTO_CONFIG=1` is set; CI shells export it once at the top of [`run_tests.sh`](../../../tests/cohere/scripts/run_tests.sh) (and friends), and standalone Python entry points set it via `os.environ.setdefault`. See [Hardware Profiles](#hardware-profiles) below.

2. **Download model checkpoints** (optional, only needed for eval/performance/guided_generation/speculative_decoding/vision/asr tests):

   ```bash
   # Set environment variables
   export ENGINES_DIR=/root/repos/engines
   export MODELS=command-r7b_fp8,command-a_fp8,command-a-vision_fp8,command-a-reasoning_fp8,c4-25a218t_fp8,c4-25a218t_int4a16,c4-v2-25a218t_int4a16_gs32_gptq,c4-v2-25a218t_int4a16_gs32_awq_gptq,c4-25a218t_w4a8
   export HF_CACHE_DIR=/path/to/hf_cache

   # Download all models for all test groups
   bash cohere/scripts/download_checkpoints.sh models
   ```

   Note: This requires GCP authentication (`gcloud auth login`) as models are stored in `gs://cohere-model-efficiency-ci/`

### Environment Variables

Configure these environment variables to customize paths:

**Directory paths** (set in `run_tests.sh`, exported to child scripts):

- `ENGINES_DIR` - Directory for model checkpoints (default: `/root/engines`)
- `VLLM_WORKSPACE` - Path to vllm-cohere repo (default: `/vllm-workspace`)
- `BEE_DIR` - Path to bee evaluation tool (default: `/app/cohere/apiary/bee`)
- `OUTPUT_DIR` - Directory for test outputs and XML reports (default: `/root/output`)

**Test configuration:**

- `HF_CACHE_DIR` - HuggingFace cache directory (default: `/home/runner/_work/hf_cache`)
- `TP_SIZE` - Tensor parallel size for eval/performance tests (default: `1`)
- `MODELS` - Comma-separated list of models for eval/performance tests (default: `command-r7b_fp8`)
- `TEST_DATA_FILE` - Path to eval config YAML file (required for lm_eval tests)

**Hardware profile opt-in** (exported once by each CI shell):

- `VLLM_ENABLE_COHERE_AUTO_CONFIG` - When `1` / `true`, `EngineArgs.__post_init__` invokes `apply_cohere_auto_config` from [`vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py), which loads [`vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml) and fills in `EngineArgs` fields that still match their dataclass defaults. Fields a test explicitly passes to `LLM(...)` / `AsyncEngineArgs(...)` are preserved. See [Hardware Profiles](#hardware-profiles) below.

### Running Test Groups

All tests are run from the `tests/` directory using `run_tests.sh`:

You can also run specific pytest tests directly:

```bash
# Setup environment first
tests/cohere/scripts/setup_tests.sh

# Set environment variables as needed
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=DEBUG
```

### Examples

**Run lm_eval tests on command-a_fp8 with TP=2:**

```bash
MODELS=command-a_fp8 \
TP_SIZE=2 \
VLLM_WORKSPACE=/host/vllm-cohere \
ENGINES_DIR=/host/engines \
TEST_GROUP=lm_eval \
HF_ALLOW_CODE_EVAL=1 \
bash cohere/scripts/run_tests.sh
```

**Run multiple models in bee eval:**

```bash
MODELS=command-a_fp8,command-a-vision_fp8,c4-25a218t_fp8 \
TP_SIZE=4 \
VLLM_WORKSPACE=/host/vllm-cohere \
ENGINES_DIR=/host/engines \
TEST_GROUP=bee_eval \
bash cohere/scripts/run_tests.sh
```

**Run performance benchmarks on command-a_fp8 with TP=2:**

```bash
MODELS=command-a_fp8 \
TP_SIZE=2 \
VLLM_WORKSPACE=/host/vllm-cohere \
ENGINES_DIR=/host/engines \
TEST_GROUP=performance \
bash cohere/scripts/run_tests.sh
```

**Run fast_check on specific GPU:**

```bash
cd tests
CUDA_VISIBLE_DEVICES=2 TEST_GROUP=fast_check bash cohere/scripts/run_tests.sh
```

**Run ASR tests with the predownloaded Cohere ASR checkpoint:**

```bash
cd tests
DATA_DIR=/root/data ENGINES_DIR=/host/engines bash cohere/scripts/download_checkpoints.sh asr
ENGINES_DIR=/host/engines \
VLLM_WORKSPACE=/host/vllm-cohere \
TEST_GROUP=asr \
bash cohere/scripts/run_tests.sh
```

## Test Documentation

Test planning and compatibility tracking live in three connected documents:

| Document | Purpose |
| --- | --- |
| [`observability_matrix.md`](./observability_matrix.md) | Central registry of every test entry and benchmark metric, each with a unique `<cat>.<feat>.<seq>` ID. |
| [`feature_matrix.md`](./feature_matrix.md) | Cross-feature compatibility tables recording which test cases verify compatibility between features, inputs, hardware, etc. |
| [`features/*.md`](./features/) | Per-feature detail docs (How it runs, Checks, Measurements, Compatibility, Implementation). |

Start at the observability matrix to find a test entry, follow links to the
feature doc for details, and check the feature matrix for cross-cutting
compatibility.

## Test Groups

| Group | Duration | Description |
| ------- | ---------- | ------------- |
| `cpu` | ~5m | CPU-only tests run inside a pre-built `vllm-cpu` Docker image on `ubuntu-latest`. Triggered automatically by `fast_check` or `all`. Includes Cohere CPU tests, CPU-safe tool parsers, V1 core scheduler tests, and logits processor correctness tests. |
| `fast_check` | ~1h | Core functionality tests run on every PR. Includes V1 core tests, basic correctness, entrypoints. |
| `lm_eval` | ~15m | Run GSM8K, NIAH, metabench, RULER against Command R7B (default). Configurable via `TP_SIZE` and `MODELS`. |
| `bee_eval` | ~3h | Run bee eval tasks against multiple models. Includes lm_eval for long context tests. Configurable via `TP_SIZE` and `MODELS`. |
| `GG` | ~10m | Guided generation regression bucket. Expands into `guided_generation`. |
| ↳ `guided_generation` | ~10m | *(internal group, use `GG` or `all`)* Guided generation tests with Command 4 and Command A models (TP=4). |
| `thinking_budget` | ~20m | Thinking budget regression bucket. Expands into `thinking_budget`, `bee_sample_tb_check`. |
| ↳ `bee_sample_tb_check` | ~10m | *(internal group, use `thinking_budget` or `all`)* Bee samples with per-task thinking budgets enabled on C5 (TP=1). Validates that `thinking_token_budget` produces passing scores. |
| `speculative_decoding` | ~15m | EAGLE speculative decoding tests, validates mean acceptance length metrics. |
| `performance` | ~1.5h | Serving benchmarks for CR7B (TP=1) or Command A (TP=2+). Configurable via `TP_SIZE` and `MODELS`. |
| `vision` | ~10m | Vision model tests with Command-A Vision, verifies multi-image input handling. |
| `asr` | ~5m | Speech-to-text regression bucket for Cohere ASR. See [Cohere ASR](./features/asr.md). Currently routed only to the 1xH100 runner. Downloads `cohere-transcribe-03-2026`, covers ASR preprocess-worker config checks, short- and long-form WER regressions, the long-audio streaming WER gate, plus transcription spacing and cancellation endpoint tests. |
| `model_arch` | ~10m | Model architecture regression bucket combining reward model checks and C5 sanity checks. |
| `quantization` | ~20m | Quantization regression bucket. Expands into `quantization_32bit_logits` (LM-head fp32 microbenchmark and C5 fp32 logits consistency). |
| ↳ `quantization_32bit_logits` | ~20m | *(internal group, use `quantization` or `all`)* LM-head fp32 microbenchmark (`test_logits_processor.py`) and full C5 fp32 logits consistency check (`test_c5_fp32_logits.py`). Runs on H100, A100, B200, GB200; not supported on MI300x. |
| `template_tokenizer_parser_check` | ~5m | Diagnostic tool (not a correctness test). Boots `vllm serve` for `c5-3a30t_fp8` twice (`no_parsers` and `with_parsers`) and logs raw messages, rendered prompt, tokenization, and generation/reasoning per sample for engineer inspection. Fails only on server-side request errors; emitted JSON logs are not uploaded. See [`features/template_tokenizer_parser.md`](./features/template_tokenizer_parser.md). |
| `collective_rpc_reload` | ~3m | Hot-reload of model weights via `/collective_rpc reload_weights`. Boots server from a corrupted checkpoint, verifies broken score, reloads good weights + recaptures CUDA graphs, verifies recovery. See [`features/weight_reload.md`](./features/weight_reload.md). |

## Hardware Profiles

Engine args for vLLM tests (memory utilization, chunked prefill, cudagraph capture sizes, etc.) are centralized in [`vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml) — bundled into the wheel via `setup.py` `package_data` — rather than hardcoded in each test file.

**How it works:**

1. The runtime hook lives at the end of `EngineArgs.__post_init__` in [`vllm/engine/arg_utils.py`](../../../vllm/engine/arg_utils.py): when `VLLM_ENABLE_COHERE_AUTO_CONFIG` is truthy (`1` / `true`), it calls `apply_cohere_auto_config(self)` from [`vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py).
2. `apply_cohere_auto_config` (a) probes the HF config for a Cohere architecture, (b) resolves matching profiles via CEL `when:` clauses bound to `{server.type, gpu.name}` (lowercased so YAML patterns like `b200` and `mi300x` match), (c) merges `vllm-default` with the GPU-specific overlay, then (d) mutates the live `EngineArgs` instance in place. Fields whose live value still equals the dataclass default are filled from the profile; user-supplied overrides are preserved unchanged.
3. CI shells export the env var once: [`run_tests.sh`](../../../tests/cohere/scripts/run_tests.sh), [`run-bee-eval.sh`](../../../tests/cohere/scripts/run-bee-eval.sh), [`run-performance-benchmarks.sh`](../../../tests/cohere/scripts/run-performance-benchmarks.sh) — every spawned `vllm serve` / `vllm bench` / pytest subprocess inherits and self-configures. Standalone Python entry points set it via `os.environ.setdefault("VLLM_ENABLE_COHERE_AUTO_CONFIG", "1")` in their `main()` / `__main__` / pytest entry function.
4. Profile env vars (`env:` block) only land when the var is currently unset; pre-existing `os.environ` values win and log at INFO. Failures inside `apply_cohere_auto_config` are caught and logged but never raised.

For the runtime contract see [Cohere Auto-Config](../code_notes/runtime-and-scheduling.md#8-cohere-auto-config-hardware-profile-application). For CI integration see [Hardware Profiles](../code_notes/ci-and-automation.md#hardware-profiles). For CPU unit-test coverage see [`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py) ([feature doc](./features/auto_config.md)).

**Key files:**

| File | Purpose |
| ------ | --------- |
| [`vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml) | Canonical per-GPU engine args and env vars |
| [`vllm/cohere/auto_config.py`](../../../vllm/cohere/auto_config.py) | `apply_cohere_auto_config` -- profile resolution, type coercion, user-override detection |
| [`vllm/engine/arg_utils.py`](../../../vllm/engine/arg_utils.py) | `EngineArgs.__post_init__` opt-in gate (reads `VLLM_ENABLE_COHERE_AUTO_CONFIG` directly) |
| [`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py) | CPU unit suite covering arch detection, CEL `when:` clauses, type coercion, profile resolution, `__post_init__` integration |

**Adding or changing engine args:**

- To change a default for all tests, edit the `vllm-default` profile in [`vllm/cohere/hardware_profiles.yaml`](../../../vllm/cohere/hardware_profiles.yaml).
- To add a GPU-specific override, add/edit a profile with a CEL `when:` clause (e.g. `matches(gpu.name, "b200")`).
- List values (e.g. `cudagraph-capture-sizes: [1, 4, 16, 64]`) are coerced to the declared `EngineArgs` field type via `_coerce`.
- Add coverage for any non-trivial new CEL clause or coercion shape in [`tests/cohere/cpu/test_auto_config.py`](../../../tests/cohere/cpu/test_auto_config.py).

## Test Output

When `OUTPUT_DIR` is set, pytest automatically generates JUnit XML reports:

- For lm_eval tests: `${OUTPUT_DIR}/report_<config_name>_<timestamp>.xml`
- For other tests: `${OUTPUT_DIR}/report_<timestamp>.xml`

Other outputs written to `OUTPUT_DIR`:

- `eval_results_summary.json` - Bee eval results summary
- `benchmark_results_summary.json` - Performance benchmark results
- `unit_results_summary.json` - Model-architecture microbenchmark summary
- `speculative_decoding_test.log` - Speculative decoding test logs
- `co-bench` - Bee eval results in co-bench format

For GitHub Actions uploads (via `upload-results`), `result_upload_branch` controls branch routing:

- `build-and-test.yaml` default: `None` (disables uploads unless set)
- `build-and-eval.yaml` and `build-and-bench.yaml` default: `ci_dump`
- `nightly-benchmark.yaml` scheduled runs fall back to `gh-pages`; manual dispatch defaults to `ci_dump` (overridable via the `result_upload_branch` input)
- uploaded records include CI run metadata (`ci_run_id`, `ci_run_attempt`, `ci_run_number`, `ci_workflow`, `ci_run_url`)

## Runners

We currently use kubernetes for self-hosted GPU runners, where the GPUs are taken from `model-efficiency` quota. Currently there are max 1 2xH100 machine and 2 1xH100 machines (assuming quota available). The details on the machines are configured here in infra repo: [infra/k8s/actions-runners-cw-efficiency/values.yaml](https://github.com/cohere-ai/infra/blob/main/k8s/actions-runners-cw-efficiency/values.yaml)

One current limitation is that the CI runners don't have persistent storage, which means we need to save/load model checkpoints and other artifacts from cloud storage. The overall bucket for these artifacts is `gs://cohere-model-efficiency-ci` and is organized by the type of checkpoints:

1. for private models we copy the checkpoint directly to `/home/runner/_work/engines/<model-name>/` (e.g., `command-r7b_fp8`, `command-a_fp8`). The full GCS path is built as `<MODEL_PATH_PREFIX><model-name>`, where `MODEL_PATH_PREFIX` is required and passed in via workflows (typically `gs://cohere-model-efficiency-ci/engines/`). This keeps the path pattern consistent across tests.
2. for public/huggingface checkpoints we `tar` then copy the entire hf cache, then mount it in `$HF_HOME` before running tests. The reason for using `tar` is that the hf cache often has symlinks which are not handled well by `gcloud storage cp`.

When adding new checkpoints, for (1) you can just copy the artifacts to the gcp storage bucket under `<MODEL_PATH_PREFIX><model-name>`. This should be the case for majority Cohere-specific tests and changes. For (2) e.g. new tests from upstream which use new models, we need to run a one-off copy step after the test runs (reach out to Conway).

## Workflow

![test-architecture](../../../tests/test-architecture.png "Test Workflow Architecture")

- [build-and-test](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-test.yaml) triggers image build and feature-test dispatch
- [build-and-eval](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-eval.yaml) triggers image build and eval dispatch (`lm_eval`, `bee_eval`)
- [build-and-bench](https://github.com/cohere-ai/vllm-cohere/actions/workflows/build-and-bench.yaml) triggers image build and perf dispatch
- [dispatcher](https://github.com/cohere-ai/vllm-cohere/actions/workflows/dispatcher.yaml) decides which test groups to run on which runners (based on `runner_map.json` and `tp_model_map.json`), and calls (one or more instances of):
- [test-pipeline](https://github.com/cohere-ai/vllm-cohere/actions/workflows/test-pipeline.yaml) which contains shared test setup, runs the test script `run_tests.sh`, and does optional post-processing

## Adding Tests

For adding new tests, there are a few options:

1. Add to existing group (if it fits the category); just append the pytest command under the test group definition in `tests/cohere/scripts/run_tests.sh`
2. Create new group; follow the existing template and create a new test group. Make sure to add in `run_tests.sh` as well as workflow files so the dispatcher can see the new group. Also specify the test_group to runner mapping in `runner_map.json` and model compatibility in `tp_model_map.json`.
3. Add new model: upload checkpoint to GCS bucket under `<MODEL_PATH_PREFIX><model-name>`, update `tp_model_map.json` (specify per device type and per model TP constraints via `minimum_tp` and `recommended_tp`), and optionally `model_eval_map.json` for lm-eval tests; `eval-config.json` for bee eval tests. If you want to verify the bee eval results, add the ground truth values in `bee_tasks/ground_truths.json`.

We prefer to add new python/pytest file instead of modifying existing ones since that will make upstream merges easier, and have a clear separation between core vLLM tests and Cohere-specific tests.
