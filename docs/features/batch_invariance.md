# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance currently requires NVIDIA GPUs with compute capability 9.0 or higher:

- **H-series**: H100, H200
- **B-series**: B100, B200

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## MoE Router GEMM Backend

Batch-invariant MoE Router gates use the existing deterministic persistent
kernel by default. On supported Hopper GPUs, an opt-in selector can use faster
deterministic backends after a dtype-specific correctness preflight:

```bash
export VLLM_BATCH_INVARIANT=1
export VLLM_BATCH_INVARIANT_ROUTER_GEMM=auto
```

`VLLM_BATCH_INVARIANT_ROUTER_GEMM` accepts the following values:

| Value | Behavior |
|---|---|
| `persistent` | Use the existing deterministic kernel. This is the default and rollback mode. |
| `auto` | Use DeepGEMM for eligible BF16 inputs, deterministic full-K for eligible FP16/FP32 inputs, and fall back to `persistent`. |
| `full_k` | Force deterministic Triton full-K. Unsupported signatures fail fast. Intended for testing. |
| `deepgemm` | Force DeepGEMM. Unsupported signatures or a missing optional dependency fail fast. Intended for testing. |

The optimized backends require SM90 CUDA tensors, contiguous two-dimensional
input and weight, no bias, and the Router shape
`A[M, 2048] @ W[128, 2048].T`. The BF16/FP16 full-K backend is limited to
`M <= 2048`; the FP32 full-K backend uses one row per CTA and is limited to
`M <= 128` to avoid a large-M regression. FP32 full-K preflight compares the
output against an FP64 reference with `rtol=1e-5` and `atol=1e-6`; a failed
precision, JIT, or Graph preflight falls back to `persistent` in `auto` mode.
DeepGEMM remains an optional dependency.

Backend selection is cached by the complete tensor signature. Import, JIT,
output validation, and a CUDA Graph smoke test run before Graph capture.
Failures in `auto` mode fall back during this preflight; errors during Graph
replay remain fail-fast and never trigger a dynamic backend switch.

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`
- **Llama 3**: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## Implementation Details

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism (such as custom all-reduce operations in tensor parallel mode)

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).
