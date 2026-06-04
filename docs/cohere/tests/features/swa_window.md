<!-- markdownlint-disable MD024 -->
# SWA Window Semantics Tests

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entries 6.2.1–6.2.5 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section Sliding Window

CPU unit suite for the NeMo-inclusive SWA window fix in
[`vllm/model_executor/models/commandr.py`](../../../../vllm/model_executor/models/commandr.py).
Background:
[SWA Window Semantics: NeMo Inclusive Convention](../../code_notes/models-and-inference.md#9-swa-window-semantics-nemo-inclusive-convention).

NeMo/FaxServer trains with an inclusive window `[pos - W, pos]` (W+1 tokens
visible). vLLM's FlashAttention backend applies a `(value - 1, 0)` offset
internally, so `commandr.py` passes `config.sliding_window + 1` to cancel
that offset. The same `+1` propagates into the KV-cache eviction formula in
`single_type_kv_cache_manager.py`.

<details>
<summary>Test case 1: SWA window CPU unit suite</summary>

## How it runs

1. `tests/cohere/cpu/test_cohere_swa_window.py` is collected by the `cpu_check`
   test group inside the prebuilt `vllm-cpu` image and runs without GPUs, model
   downloads, or a distributed process group. All GPU-bound dependencies
   (`QKVParallelLinear`, `RowParallelLinear`, `get_rope`, `Attention`,
   `get_tensor_model_parallel_world_size`) are patched out with `MagicMock`.
   - [`tests/cohere/cpu/test_cohere_swa_window.py`](../../../../tests/cohere/cpu/test_cohere_swa_window.py)
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh) -- `cpu_check` group dispatch
2. `TestCohereAttentionSlidingWindow` instantiates `CohereAttention` with a
   minimal `SimpleNamespace` stand-in for `Cohere2Config`, verifying the
   `self.sliding_window` value set during `__init__` and the
   `per_layer_sliding_window` keyword arg forwarded to the `Attention`
   constructor mock.
   - [`vllm/model_executor/models/commandr.py`](../../../../vllm/model_executor/models/commandr.py) -- `CohereAttention.__init__`
3. `TestSwaEvictionArithmetic` is pure arithmetic — it mirrors the formulas in
   `flash_attn.py` and `single_type_kv_cache_manager.py` and pins the
   before/after eviction behaviour as a regression guard without importing any
   vLLM attention code.
   - [`vllm/v1/attention/backends/flash_attn.py`](../../../../vllm/v1/attention/backends/flash_attn.py) -- `(sliding_window - 1, 0)` offset
   - [`vllm/v1/core/single_type_kv_cache_manager.py`](../../../../vllm/v1/core/single_type_kv_cache_manager.py) -- `get_num_skipped_tokens` formula
4. CI shape: dispatched as the `cpu_check` test group on `ubuntu-latest`
   (no GPU); triggered automatically on every PR via `pr-cpu-tests.yaml`.
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)

## Checks

1. A **Cohere v2 SWA layer** (`layer_types[i] == "sliding_attention"`) sets
   `self.sliding_window = config.sliding_window + 1`, not `config.sliding_window`.
   - `test_swa_layer_adds_one`
2. A **full-attention layer** and a **v1 model** (no SWA) leave
   `self.sliding_window = None`.
   - `test_full_attention_layer_has_no_window`
   - `test_v1_model_has_no_window`
3. The `+1` value is forwarded as **`per_layer_sliding_window`** to the
   `Attention` constructor, confirming end-to-end propagation.
   - `test_window_propagated_to_attention_layer`
4. With the fix applied (`value = W + 1`), the **KV-cache eviction formula**
   `max(0, N - value + 1)` produces zero skipped tokens at exactly `N = W`
   (window not yet full) and one skipped token at `N = W + 1` (first eviction).
   - `test_nemo_convention_no_eviction_at_exactly_w`

## Measurements

N/A — no CI-uploaded artifacts. `cpu_check` has no `upload-results` or
benchmark JSON step in `test-pipeline.yaml`. Pytest run produces a JUnit XML
report surfaced as a "CPU Test Report" check on the GitHub run via
`dorny/test-reporter@v2`.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: not applicable (CPU unit suite, no model inputs).
2. **Cohere Feature**: not applicable.
3. **Model Architecture**: tests `CohereAttention` for Cohere v2 models
   (C5 Arch, `Cohere2Config`). The v1 guard (`test_v1_model_has_no_window`)
   confirms no SWA window is set for Command-R v1 (`CohereConfig`).
   - [`tests/cohere/cpu/test_cohere_swa_window.py`](../../../../tests/cohere/cpu/test_cohere_swa_window.py)
4. **Quantization**: not applicable.
5. **Hardware**: CPU-only — runs on the `vllm-cpu` Docker image on
   `ubuntu-latest`. No GPU runner involvement.
   - [`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)
6. **vLLM Feature**: not applicable (no engine launch; patches
   `Attention` and linear layers directly).

## Implementation

Primary test:
[`tests/cohere/cpu/test_cohere_swa_window.py`](../../../../tests/cohere/cpu/test_cohere_swa_window.py)
Runtime paths:
[`vllm/model_executor/models/commandr.py`](../../../../vllm/model_executor/models/commandr.py) -- `CohereAttention.__init__` fix,
[`vllm/v1/attention/backends/flash_attn.py`](../../../../vllm/v1/attention/backends/flash_attn.py) -- downstream `(value - 1, 0)` offset,
[`vllm/v1/core/single_type_kv_cache_manager.py`](../../../../vllm/v1/core/single_type_kv_cache_manager.py) -- downstream eviction formula
CI entry: `cpu_check` group via
[`.github/workflows/pr-cpu-tests.yaml`](../../../../.github/workflows/pr-cpu-tests.yaml)

### Setup

1. **Fake config**: `_make_cohere2_config()` builds a `SimpleNamespace` with
   `sliding_window=4096`, a configurable `layer_types` list, and the minimal
   attributes read by `CohereAttention.__init__`. No real `Cohere2Config`
   or model download is required.
2. **GPU dep patching**: `_build_cohere_attention()` patches
   `QKVParallelLinear`, `RowParallelLinear`, `get_rope`, `Attention`, and
   `get_tensor_model_parallel_world_size` (returning `1`) so `CohereAttention`
   can be instantiated on CPU without a CUDA context or process group.
3. **Propagation capture**: `test_window_propagated_to_attention_layer` sets a
   `side_effect` on the `Attention` mock to record the
   `per_layer_sliding_window` keyword argument.
4. **Arithmetic regression**: `TestSwaEvictionArithmetic` uses only built-in
   Python — no imports. `_skipped()` mirrors
   `SlidingWindowManager.get_num_skipped_tokens` exactly.

</details>
