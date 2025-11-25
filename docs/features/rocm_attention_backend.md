# ROCm Attention Backends

vLLM on the AMD ROCm platform supports multiple attention backends to optimize performance across a wide range of hardware architectures (e.g., AMD Instinct MI200/MI300 series, Radeon consumer cards) and model architectures (e.g., Llama, DeepSeek-V3/R1).

While vLLM attempts to select the best default backend for your hardware, explicitly selecting a backend allows for fine-grained performance tuning, especially on high-end datacenter GPUs like the MI300X.

## Why Multiple Backends?

The choice of attention backend affects **latency**, **throughput**, and **memory usage**. The primary difference lies in the underlying kernel implementation:

* **Triton Kernels**: Written in OpenAI Triton. They offer excellent portability across different ROCm versions and GPU architectures but might not always squeeze the last bit of performance out of specific hardware.
* **HIP C++ Kernels**: Native AMD ROCm implementations. Historically used for PagedAttention, providing stable performance.
* **AITER (Assembly/CK) Kernels**: Kernels from the **A**MD **I**nference **Te**nso**R** library. These are highly optimized, low-level kernels (often written in Assembly or Composable Kernel) specifically tuned for **Matrix Core** architectures like CDNA3 (MI300 series). They typically offer the highest performance but have stricter hardware requirements.

## Backend Architecture Explained

### Standard Attention Backends

These backends are used for standard Transformers models (Llama 3, Qwen 2, Mistral, etc.).

| Backend Name | Description | Use Case |
| :--- | :--- | :--- |
| **TRITON_ATTN** | **Default.** Uses vLLM's unified attention implementation written in Triton. Both prefill and decode phases use Triton kernels. | General purpose, high portability, ease of debugging. |
| **ROCM_ATTN** | Uses a hybrid approach: Triton for chunked prefill and custom HIP C++ kernels for paged attention decode. | Scenarios where specific HIP optimizations outperform Triton. |
| **ROCM_AITER_FA** | Uses **AITER** (AMD Inference TensoR) Flash Attention kernels. Highly optimized assembly kernels. | **Recommended for MI300 (gfx94x).** Requires `gfx9` architecture. |
| **ROCM_AITER_UNIFIED_ATTN** | Uses AITER's unified attention implementation. | Alternative AITER path for unified prefill/decode. |
| **FLEX_ATTENTION** | Uses vLLM's FlexAttention implementation. | Experimental or specific model requirements. |

### MLA Backends (DeepSeek-V3/R1)

DeepSeek models use **Multi-Head Latent Attention (MLA)**, which requires specialized memory handling.

| Backend Name | Description | Configuration |
| :--- | :--- | :--- |
| **ROCM_AITER_MLA** | **Recommended.** Optimized AITER Assembly kernels tuned for DeepSeek on AMD hardware. | Supports default `block_size` (16) and `block_size=1`. |
| **TRITON_MLA** | Portable Triton implementation. | Requires `block_size >= 16`. |

---

## Usage Examples

### Attention Backend on ROCm

You can control the backend selection using the `VLLM_ATTENTION_BACKEND` environment variable.

### 1. TRITON_ATTN (Default)

Uses vLLM's triton unified attention backend.

```bash
# Example 1: Default behavior (no flags needed)
vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 2: Explicitly set the backend
VLLM_ATTENTION_BACKEND="TRITON_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 3: Force Triton even if AITER is enabled
VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="TRITON_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 4: Fallback to Triton by disabling AITER MHA
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 vllm serve meta-llama/Llama-3.1-8B-Instruct

```

### 2. ROCM_ATTN (Hybrid)

Uses vLLM's chunked prefill (Triton) and paged decode (HIP) kernel.

```bash

# Example 1: Explicit selection
VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 2: When enable AITER but still want to use TRITON_ATTN
VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 3
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 4: Legacy flag combination
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 vllm serve meta-llama/Llama-3.1-8B-Instruct

```

### 3. ROCM_AITER_FA (Flash Attention)

Use the AITER Flash Attention backend. Requires gfx9 architecture.

```bash

# Example 1: Only use AITER FA backend without enabling other AITER kernels
VLLM_ATTENTION_BACKEND="ROCM_AITER_FA" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 2: Auto-selection via convenience flag
VLLM_ROCM_USE_AITER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct

```

### 4. ROCM_AITER_UNIFIED_ATTN

Use AITER unified attention backend.

```bash

# Example 1: Only use AITER FA backend without enabling other AITER kernels
VLLM_ATTENTION_BACKEND="ROCM_AITER_UNIFIED_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 2: Force Unified even if AITER flag is set
VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="ROCM_AITER_UNIFIED_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

# Example 3:
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 vllm serve meta-llama/Llama-3.1-8B-Instruct

```

### MLA Backend

On AMD ROCm, there are TRITON_MLA and ROCM_AITER_MLA

### 5. DeepSeek MLA (TRITON_MLA)

Uses vLLM's triton MLA backend. The prefill uses triton flash attention/ CK flash attention varlen, and decode uses triton mla decode kernel. Requires block_size >= 16.

```bash

# Example 1: Explicit selection (Requires block_size >= 16)
VLLM_ATTENTION_BACKEND="TRITON_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8

# Example 2:
VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="TRITON_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8

```

### 6. DeepSeek MLA (ROCM_AITER_MLA)

For DeepSeek models, vLLM defaults to ROCM_AITER_MLA on supported hardware.

```bash

# Example 1: Explicit selection
VLLM_ATTENTION_BACKEND="ROCM_AITER_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8

# Example 2: Implicit selection via AITER flag
VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-R1 -tp 8

# Example 3: Legacy support (block-size 1)
VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-R1 \
  --tp 8 \
  --block-size 1
  
```

## Selection Priority

When multiple configurations are provided, vLLM follows this strict priority order:

1. **Explicit Selection**:
    * `VLLM_ATTENTION_BACKEND` takes the highest precedence. If set, other flags are ignored.

2. **Implicit Flags** (only checked if no explicit backend is set):
    * **Priority 1**: `ROCM_AITER_UNIFIED_ATTN` (if `VLLM_ROCM_USE_AITER=1` AND `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`).
    * **Priority 2**: `ROCM_AITER_FA` (if `VLLM_ROCM_USE_AITER=1` AND hardware is `gfx9`).
    * **Priority 3**: `ROCM_ATTN` (if `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`).

3. **Default**:
    * Falls back to `TRITON_ATTN`.

!!! note
    Explicitly setting `VLLM_ATTENTION_BACKEND` is the recommended way to debug backend issues, as it bypasses complex flag combinations.
