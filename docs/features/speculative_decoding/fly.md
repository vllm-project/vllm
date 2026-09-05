# FLy Verification

[FLy](https://arxiv.org/abs/2511.22972) is an approximate verification policy that can defer a native rejection at a high-entropy position when the following draft tokens remain aligned with native verification.

!!! warning
    FLy is intentionally lossy. Unlike standard speculative decoding, it does not preserve the target distribution and may degrade model outputs.

## Usage

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 8,
        "rejection_sample_method": "fly",
        "fly_window_size": 6,
        "fly_entropy_threshold": 0.3,
    },
)
```

FLy requires at least two speculative tokens. `fly_window_size` is the number of subsequent native acceptance decisions checked; it must be smaller than `num_speculative_tokens` and defaults to `min(6, num_speculative_tokens - 1)`. The entropy gate uses the three largest processed target probabilities by default. Set `VLLM_FLY_ENTROPY_TOP_K` to change that number.

FLy supports greedy draft sampling with target-only acceptance and probabilistic draft sampling with standard p/q acceptance. Token-Level Intersection can be used for cross-vocabulary draft models with greedy draft sampling, but is incompatible with `use_local_argmax_reduction`.

FLy is supported by both model runners. With ModelRunnerV2, use a supported proposer such as MTP, EAGLE, or DFlash; standalone `draft_model` proposing still requires ModelRunnerV1.

## Validated Configurations

The primary evaluation covers the V1 `draft_model` proposer. Configurations not listed below are not necessarily unsupported, but have not yet been experimentally validated.

FLy was faster than Standard SD in all 40 primary configurations. Across all four target/draft pairs, the 29 configurations with ratio-based quality measurements had a minimum quality retention of 97.9%.

| Accelerator | Model configuration | Workloads | Sampling | Measured speedup |
| --- | --- | --- | --- | ---: |
| AMD MI355X | Llama-3.1-70B-Instruct / Llama-3.1-8B-Instruct | 10-benchmark suite | Temperature 0, greedy draft | 1.092x–1.335x vs Standard SD |
| AMD MI355X | DeepSeek-R1-Distill-Llama-70B / DeepSeek-R1-Distill-Llama-8B | 10-benchmark suite | Temperature 0, greedy draft | 1.147x–1.467x vs Standard SD |
| AMD MI355X | Llama-3.1-405B-Instruct-FP8-KV / Llama-3.1-8B-Instruct | 10-benchmark suite | Temperature 0, greedy draft | 1.189x–1.938x vs Standard SD |
| AMD MI355X | Qwen3-235B-A22B-Thinking-2507-FP8 / Qwen3-8B | 10-benchmark suite | Temperature 0, greedy draft | 1.067x–1.478x vs Standard SD |
| NVIDIA B300 | DeepSeek-R1-Distill-Llama-70B / DeepSeek-R1-Distill-Llama-8B | MATH-500, HumanEval, HLE | Temperature 0, greedy draft | 1.26x–1.43x vs Standard SD |
| NVIDIA B300 | Qwen2.5-72B-Instruct / Qwen2.5-7B-Instruct | GSM8K, MGSM, GPQA-Diamond, Spec-Bench | Temperature 0, greedy draft | 1.07x–1.32x vs Standard SD |
| AMD MI300X | Kimi-K2.5 (1T) | Random 8K input / 1K output | DFlash comparison | >1.09x vs DFlash |

The 10-benchmark suite contains GSM8K, MGSM, MATH-500, AIME 2024, AIME 2025, GPQA-Diamond, HumanEval, MBPP, Spec-Bench, and HLE. Subsets of GSM8K and MATH-500 were also validated with temperature 1 and probabilistic draft sampling.
