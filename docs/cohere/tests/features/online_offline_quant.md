<!-- markdownlint-disable MD024 -->
# Online/Offline Quant Equivalence

> **Registry**: [`observability_matrix.md`](../observability_matrix.md) entry 2.3.1 |
> **Compatibility**: [`feature_matrix.md`](../feature_matrix.md) section Online/Offline Quant

Validates that the online quantize-at-load path produces the same outputs as the
matching offline pre-quantized checkpoints for C5 FP8-family schemes.

<details>
<summary>Test case 1: C5 online/offline quant equivalence</summary>

## How it runs

1. CI pre-downloads offline/online checkpoint pairs for `fp8`, `mxfp8`, and
   `block_fp8` before the quantization logits group runs.
   - [`tests/cohere/scripts/download_checkpoints.sh`](../../../../tests/cohere/scripts/download_checkpoints.sh)
2. `run_model_arch_logits_checks` iterates over those three schemes and invokes
   `pytest -v -s tests/cohere/test_c5_online_vs_offline_quant.py` with
   `C5_OFFLINE_MODEL_DIR`, `C5_ONLINE_MODEL_DIR`, and `C5_QUANT_SCHEME` set per
   run under the `quantization_32bit_logits` test group.
   - [`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh)
3. The pytest entry loads the offline checkpoint first, tokenizes
   `C5_SANITY_PROMPTS` once, reuses those exact token IDs for the online
   checkpoint via `TokensPrompt`, and installs a `logits_processor` forward hook
   through `apply_model()` to capture one full-vocabulary next-token logits row
   per prompt.
   - [`tests/cohere/test_c5_online_vs_offline_quant.py`](../../../../tests/cohere/test_c5_online_vs_offline_quant.py)
   - [`tests/cohere/test_utils_c5.py`](../../../../tests/cohere/test_utils_c5.py)
4. CI routes `quantization_32bit_logits` through pytest/JUnit reporting; see
   [Test Pipeline Integration](../../code_notes/ci-and-automation.md#7-test-pipeline-integration).

## Checks

1. Fails when the captured offline/online logits matrices do not align
   position-wise.
   - `test_c5_online_vs_offline_quant_logits`
2. Fails when **max L1**, **mean L1**, or **top-1 agreement** violate the
   configured tolerances.
   - `test_c5_online_vs_offline_quant_logits`
3. Fails when the greedy rollout token IDs differ between the offline and
   online checkpoints for any prompt.
   - `test_c5_online_vs_offline_quant_logits`

## Measurements

N/A. This test does not emit CI-uploaded summary metrics; the
`quantization_32bit_logits` upload path is currently driven by the separate
LM-head fp32 benchmark.

## Compatibility

Features from [Feature Matrix](../feature_matrix.md)
([Compatibility Sources](../feature_matrix.md#compatibility-sources)):

1. **Input**: Basic (compatible), Multilingual (compatible)
2. **Cohere Feature**:
3. **Model Architecture**: C5 Arch (compatible)
4. **Quantization**: FP8 (compatible), MXFP8 (compatible), Online/Offline Quant (compatible)
5. **Hardware**: A100 (not compatible), H100 (compatible), B200 (compatible), GB200 (compatible), MI300x (not compatible)
   - [`tests/cohere/configs/runner_map.json`](../../../../tests/cohere/configs/runner_map.json) -- `quantization_32bit_logits` runners exist for H100/B200/GB200, not A100/MI300x
6. **vLLM Feature**: Chunked Prefill (compatible), Torch Compile (not compatible), CUDA Graphs (not compatible)
   - [`vllm/cohere/hardware_profiles.yaml`](../../../../vllm/cohere/hardware_profiles.yaml)

## Implementation

Primary test:
[`tests/cohere/test_c5_online_vs_offline_quant.py`](../../../../tests/cohere/test_c5_online_vs_offline_quant.py)
Runtime paths:
[`vllm/model_executor/layers/quantization/online/fp8.py`](../../../../vllm/model_executor/layers/quantization/online/fp8.py),
[`vllm/model_executor/layers/quantization/online/mxfp8.py`](../../../../vllm/model_executor/layers/quantization/online/mxfp8.py)
CI routing:
[`tests/cohere/scripts/run_tests.sh`](../../../../tests/cohere/scripts/run_tests.sh),
[`tests/cohere/scripts/download_checkpoints.sh`](../../../../tests/cohere/scripts/download_checkpoints.sh)

### Setup

1. The test is configured by `C5_OFFLINE_MODEL_DIR`, `C5_ONLINE_MODEL_DIR`,
   `C5_QUANT_SCHEME`, `C5_QUANT_TENSOR_PARALLEL_SIZE`,
   `C5_QUANT_MAX_TOKENS`, `C5_QUANT_MAX_L1`, `C5_QUANT_MEAN_L1`, and
   `C5_QUANT_MIN_TOP1_AGREE`.
2. `LLM(...)` runs with `enforce_eager=True` and
   `enable_prefix_caching=False`, then compares next-token logits captured from
   the shared prompt-token IDs plus the full greedy rollout token IDs.
3. `run_tests.sh` exports `VLLM_ENABLE_COHERE_AUTO_CONFIG=1`, so default
   hardware profiles remain active during CI; `mxfp8` additionally self-skips on
   non-Blackwell GPUs inside the pytest test.

</details>
