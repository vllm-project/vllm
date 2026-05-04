# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    All environment variables used by vLLM are prefixed with `VLLM_`. **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## ROCm-Specific Environment Variables

For users running vLLM on AMD GPUs with ROCm, the following environment variables are available to control specific optimizations and backends (such as AITER operations):

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `VLLM_ROCM_USE_AITER` | `False` | Main switch to enable AITER (AMD Inference Transformer) operations. |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `False` | Whether to use AITER paged attention. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `True` | Use AITER tuned GEMMs for unquantized GEMMs and linear ops. |
| `VLLM_ROCM_USE_AITER_MOE` | `True` | Whether to use AITER MoE (Mixture of Experts) ops. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `True` | Use AITER RMS norm op if AITER ops are enabled. |
| `VLLM_ROCM_USE_AITER_MLA` | `True` | Whether to use AITER MLA ops. |
| `VLLM_ROCM_USE_AITER_MHA` | `True` | Whether to use AITER MHA ops. |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `False` | Whether to use AITER fp4 gemm asm. |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `False` | Whether to use AITER triton rope. |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` | Whether to use AITER triton fp8 BMM kernel. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `True` | Whether to use AITER triton fp4 BMM kernel. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `False` | Use AITER triton unified attention for V1 attention. |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`| `False` | Whether to use AITER fusion shared experts ops. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `True` | Whether to use AITER triton kernels for gemm ops. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `True` | Use ROCm skinny GEMMs. |
| `VLLM_ROCM_FP8_PADDING` | `1` | Pad the fp8 weights to 256 bytes for ROCm. |
| `VLLM_ROCM_MOE_PADDING` | `1` | Pad the weights for the MoE kernel. |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`| `False` | Whether to use the shuffled KV cache layout. |

## Full Variable Definitions

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
