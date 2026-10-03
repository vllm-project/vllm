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
        "fly_entropy_top_k": 3,
    },
)
```

FLy requires at least two speculative tokens. `fly_window_size` is the number of subsequent native acceptance decisions checked; it must be smaller than `num_speculative_tokens` and defaults to `min(6, num_speculative_tokens - 1)`. The entropy gate uses the three largest processed target probabilities by default. Set `fly_entropy_top_k` in `speculative_config` to change that number; it must be positive and is capped at the vocabulary size.

FLy supports greedy draft sampling with target-only acceptance and probabilistic draft sampling with standard p/q acceptance. Token-Level Intersection can be used for cross-vocabulary draft models with greedy draft sampling, but is incompatible with `use_local_argmax_reduction`.

FLy is supported by both model runners. With ModelRunnerV2, use a supported proposer such as MTP, EAGLE, or DFlash; standalone `draft_model` proposing still requires ModelRunnerV1.

## Validated Configurations

The following evaluation covers the V1 `draft_model` proposer; MRV2 validation is summarized below. Configurations not listed here are not necessarily unsupported, but have not yet been experimentally validated.

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

### ModelRunner V2

MRV2 runs used greedy draft sampling, `fly_window_size=2`, `fly_entropy_threshold=0.3`, and `fly_entropy_top_k=3`. Decode throughput was measured with 1,024 output tokens per request at batch sizes 1, 4, 8, 16, 32, and 64, excluding prefill.

| Accelerator | Model | Speculative method | Draft tokens (K) | Measured decode speedup |
| --- | --- | --- | ---: | ---: |
| AMD MI355X | DeepSeek-V4-Flash-0731 | DSpark | 7 | 1.103x–1.168x vs Standard SD |
| AMD MI355X | Qwen3.8-27B | MTP | 3 | 1.055x–1.116x vs Standard SD |
| AMD MI355X | GLM-5.2-FP8 | MTP | 5 | 1.110x–1.229x vs Standard SD |
| AMD MI355X | Qwen3.5-122B-A10B-FP8 | MTP | 8 | 1.251x–1.310x vs Standard SD |
| NVIDIA B300 | DeepSeek-V4-Flash-0731 | DSpark | 7 | 1.099x–1.253x vs Standard SD |
| NVIDIA B300 | Qwen3.8-27B | MTP | 3 | 1.041x–1.095x vs Standard SD |
| NVIDIA B300 | GLM-5.2-FP8 | MTP | 5 | 1.120x–1.207x vs Standard SD |
| NVIDIA B300 | Qwen3.5-122B-A10B-FP8 | MTP | 8 | 1.202x–1.273x vs Standard SD |

Single-pass quality evaluations compared Target-only, Standard SD, and FLy on GSM8K (100 examples), MATH-500 (100), AIME25 (30), and GPQA-Diamond (198). Relative to Target-only, the minimum measured accuracy retention was 96.43% on AMD and 96.30% on NVIDIA; both corresponded to one fewer correct answer on AIME25.
