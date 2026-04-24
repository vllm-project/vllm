<!-- markdownlint-disable MD024 -->
# FP32 Logits Tests and Benchmarks

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 2.1.1, 2.1.2, 2.1.3 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section FP32 Logits

Validates the fp32-logits path for C5 through dedicated pytest coverage files
under `tests/cohere/unit/`.

<details>
<summary>Test case 1: C5 fp32 logits consistency</summary>

## How it runs

1. `run_fp32_logits_consistency_test` runs two sequential generations in one
   process with `VLLM_USE_LOGITS_FP32_COMPUTATION=0` and then `=1`, resetting
   cached env reads with `envs.disable_envs_cache()` between runs.
   - [`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py)
2. `_capture_generation_snapshot` generates with
   `SamplingParams(logprobs=1, prompt_logprobs=1)`, installs the logits dtype
   hook, and records token IDs, sampled logprobs, prompt token IDs, prompt
   logprobs, and fp32 debug info.
   - [`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py)
3. The dedicated pytest entrypoint runs with
   `C5_MODEL_DIR=<model_dir> pytest -v -s tests/cohere/unit/test_c5_fp32_logits.py`.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)

## Checks

1. Fails on **token ID mismatch** when the shared prefix between baseline and
   fp32 runs is shorter than `min_shared_prefix`.
   - `test_c5_fp32_logits_consistency`
2. Fails on **sampled logprob delta** above `max_logprob_abs_diff` (default
   0.5) within the shared-prefix region.
   - `test_c5_fp32_logits_consistency`
3. Fails on **prompt logprob delta** above `max_prompt_logprob_abs_diff`
   (default 5.0) across all prompt positions.
   - `test_c5_fp32_logits_consistency`
4. fp32 mode must materialize logits as **`torch.float32`**, via the dtype hook
   in the test and `_get_logits` in the runtime.
   - `test_c5_fp32_logits_consistency`

## Measurements

1. Successful runs print **logits dtypes**, **max sampled abs diff**,
   **max prompt abs diff**, **comparable sampled/prompt token counts**, and
   **token divergence count** to stdout.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible)
2. **Cohere Feature**:
3. **Model Architecture**: C5 Arch (compatible)
4. **Quantization**: FP32 Logits (compatible)
5. **Hardware**: MI300x (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- no `quantization_32bit_logits` runner for MI300x
6. **vLLM Feature**: Torch Compile, CUDA Graphs (not compatible)

## Implementation

Primary test:
[`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py)
Runtime path: [`vllm/model_executor/layers/logits_processor.py`](../../../../vllm/model_executor/layers/logits_processor.py)

### Setup

1. Toggles `VLLM_USE_LOGITS_FP32_COMPUTATION` between `0` and `1`, resets
   cached env reads with `envs.disable_envs_cache()` so both modes run in one
   process:
   [`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py)
2. Enables `VLLM_ALLOW_INSECURE_SERIALIZATION=1` because `apply_model()` sends
   Python callables over RPC for the dtype hook and debug-info capture:
   [`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py)
3. Uses the C5 model path, `tensor_parallel_size`, and
   `SamplingParams(temperature=0.0, max_tokens=32, logprobs=1, prompt_logprobs=1)`
   over the fixed multilingual/code prompt set in `C5_SANITY_PROMPTS`. Engine
   kwargs are merged with hardware profile defaults via
   `get_engine_kwargs_with_overrides` (`VLLM_HARDWARE_PROFILE_ARGS` env var):
   [`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py),
   [`tests/test_utils_c5.py`](../../../../tests/test_utils_c5.py),
   [`tests/cohere/test_utils_engine_args.py`](../../../../tests/cohere/test_utils_engine_args.py)
4. Tuning env vars: `C5_FP32_LOGITS_MAX_ABS_DIFF` (sampled logprob tolerance,
   default 0.5), `C5_FP32_LOGITS_MAX_PROMPT_ABS_DIFF` (prompt logprob
   tolerance, default 5.0), `C5_FP32_LOGITS_MIN_SHARED_PREFIX` (minimum
   shared token prefix, default 8).

</details>

<details>
<summary>Test case 2: LM-head-only fp32 vs bf16 projection diff</summary>

## How it runs

1. Builds a deterministic LM-head-only fixture with bf16 hidden states and a
   dense bf16 LM-head weight, then runs the projection twice with
   `VLLM_USE_LOGITS_FP32_COMPUTATION=0` and `=1`.
   - [`tests/cohere/unit/test_logits_processor.py`](../../../../tests/cohere/unit/test_logits_processor.py)
2. Resets cached env reads between the two executions so
   `VLLM_USE_LOGITS_FP32_COMPUTATION` is re-evaluated before each call into
   `LogitsProcessor._get_logits`.
   - [`tests/cohere/unit/test_logits_processor.py`](../../../../tests/cohere/unit/test_logits_processor.py)
3. Runs a microbenchmark for the no-bias projection path with explicit warmup
   iterations before measured iterations, and reports separate bf16-path and
   fp32-path timings from the same eager fixture.
   - [`tests/cohere/unit/test_logits_processor.py`](../../../../tests/cohere/unit/test_logits_processor.py)
4. CI routes the correctness check through JUnit XML via the pytest conftest
   hook, and the nightly job routes through `quantization_32bit_logits`; see
   [Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration).

## Checks

1. **fp32 vs bf16** outputs are not exactly identical but stay within
   **tolerance**.
   - `test_lm_head_fp32_projection_diff_is_small_but_nonzero`
2. fp32 path materializes logits as **`torch.float32`**, baseline stays on the
   **reduced-precision** projection path.
   - `test_lm_head_fp32_projection_diff_is_small_but_nonzero`
3. Benchmark writes summary file with **bf16/fp32 timings**.
   - `test_lm_head_fp32_projection_benchmark_writes_summary`

## Measurements

1. Writes **bf16/fp32 timings** and **relative slowdown** to
   `unit_results_summary.json`.
   - `bf16_median_ms` -- `LOWER-ANCHOR3+5%`
   - `fp32_median_ms` -- `LOWER-ANCHOR3+5%`
2. CI persists the structured benchmark data via the `quantization_32bit_logits`
   upload path; see
   [Nightly Metric Emission](../../code_notes/ci-and-automation.md#nightly-metric-emission).
   Pattern definitions: [Metric Pattern Codes](../observability_matrix.md#metric-pattern-codes).

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**:
2. **Cohere Feature**:
3. **Model Architecture**:
4. **Quantization**: FP32 Logits (compatible)
5. **Hardware**: A100, H100, B200, GB200 (compatible); MI300x (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `quantization_32bit_logits` runners for A100/H100/B200/GB200; none for MI300x
6. **vLLM Feature**: Torch Compile, CUDA Graphs (not compatible)

## Implementation

Primary test:
[`tests/cohere/unit/test_logits_processor.py`](../../../../tests/cohere/unit/test_logits_processor.py)
Runtime path:
[`vllm/model_executor/layers/logits_processor.py`](../../../../vllm/model_executor/layers/logits_processor.py)
CI routing:
[Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration)

The isolated LM-head microbenchmark intentionally keeps `compute_logits()`
eager so it matches the real v1 runtime boundary, where the model forward is
the compiled/cudagraph-managed callable and logits are computed afterward on the
raw model object. End-to-end `torch.compile` and CUDA-graph fp32-logits
coverage remains on
[`tests/cohere/unit/test_c5_fp32_logits.py`](../../../../tests/cohere/unit/test_c5_fp32_logits.py).

### Setup

1. The fixture uses deterministic bf16 hidden states and a dense bf16
   `lm_head.weight` so the test isolates `LogitsProcessor._get_logits` without a
   full model load: [`tests/cohere/unit/test_logits_processor.py`](../../../../tests/cohere/unit/test_logits_processor.py)
2. Toggles `VLLM_USE_LOGITS_FP32_COMPUTATION` between `0` and `1`,
   resets env caching between runs, and exercises the no-bias projection path so
   the benchmark targets the `torch.mm` branch in
   [`vllm/model_executor/layers/logits_processor.py`](../../../../vllm/model_executor/layers/logits_processor.py).

</details>
