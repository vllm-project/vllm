# AITER (AMD Inference Transformer) Options

AITER is an experimental, optimized backend for running Inference Transformer operations on AMD ROCm devices. It includes a suite of highly tuned kernels such as GEMM, Flash Attention, RoPE, RMSNorm, and MoE logic that can be enabled selectively using environment variables.

All AITER-related flags are prefixed with `VLLM_ROCM_USE_AITER`.

## Master Toggle

- **`VLLM_ROCM_USE_AITER`**
  - **Default:** `False`
  - **Description:** The main toggle for enabling AITER operations. By default, it acts as a parent switch for the other sub-components. Setting this to `True` enables the AITER pipeline globally, but specific features can still be turned on/off using their respective flags.

## Component Flags

### Matrix Multiplication (GEMM) & Linear Layers
- **`VLLM_ROCM_USE_AITER_LINEAR`** (Default: `True`)
  - Enables AITER optimized GEMMs for standard unquantized GEMMs and linear operations.
- **`VLLM_ROCM_USE_AITER_TRITON_GEMM`** (Default: `True`)
  - Enables AITER triton kernels for generic GEMM ops.
- **`VLLM_ROCM_USE_AITER_FP8BMM`** (Default: `True`)
  - Enables Triton FP8 Batched Matrix Multiply kernel.
- **`VLLM_ROCM_USE_AITER_FP4BMM`** (Default: `True`)
  - Enables Triton FP4 Batched Matrix Multiply kernel.
- **`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM`** (Default: `False`)
  - Enables AITER FP4 assembly GEMM optimizations.

### Attention Mechanisms
- **`VLLM_ROCM_USE_AITER_PAGED_ATTN`** (Default: `False`)
  - Enables the AITER paged attention backend.
- **`VLLM_ROCM_USE_AITER_MHA`** (Default: `True`)
  - Enables AITER Multi-Head Attention operations.
- **`VLLM_ROCM_USE_AITER_MLA`** (Default: `True`)
  - Enables AITER Multi-head Latent Attention operations.
- **`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`** (Default: `False`)
  - Uses AITER Triton unified attention logic for V1 Attention APIs.

### Mixture of Experts (MoE)
- **`VLLM_ROCM_USE_AITER_MOE`** (Default: `True`)
  - Enables AITER MoE kernel operations.
- **`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`** (Default: `False`)
  - Enables AITER fusion ops for shared experts inside MoE models.

### Normalization and Embeddings
- **`VLLM_ROCM_USE_AITER_RMSNORM`** (Default: `True`)
  - Enables AITER optimized RMSNorm operations.
- **`VLLM_ROCM_USE_AITER_TRITON_ROPE`** (Default: `False`)
  - Enables AITER Triton kernels for Rotary Positional Embeddings (RoPE).

## Best Practices

For users trying to maximize performance on MI300X or recent AMD hardware, the recommended starting point is enabling the master toggle:

```bash
export VLLM_ROCM_USE_AITER=1
```

If you encounter issues or numerical instability (e.g., NaNs during inference), try disabling specific modules like `VLLM_ROCM_USE_AITER_PAGED_ATTN=0` to isolate the offending kernel.
