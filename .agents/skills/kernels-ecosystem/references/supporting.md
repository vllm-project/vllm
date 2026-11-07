# Normalization, quantization, RoPE and sampling

[Primitive index](../SKILL.md)

These are reusable primitive/backend choices, not every model's local glue
kernel. Framework imports of the same FlashInfer operation are omitted.
Quantized output format and scale layout are part of the contract; NVFP4,
MXFP4, FP8 and MXFP8 are not interchangeable.

## Normalization

| Implementation / entry | Language | Origin and source | Operation / constraints |
| --- | --- | --- | --- |
| Native RMSNorm / residual RMSNorm | CUDA | FlashInfer; [kernels][norm-native] | `rmsnorm`, `gemma_rmsnorm`, `fused_add_rmsnorm`, `gemma_fused_add_rmsnorm`; ordinary versus Gemma weight-offset semantics, in-place residual variants. |
| RMSNorm: `RMSNormKernel`, `QKRMSNormKernel` | CuTe DSL | FlashInfer; [kernels][rms] | `rmsnorm_cute`, `qk_rmsnorm_cute`; row/headwise variants. |
| Residual RMSNorm: `fused_add_rmsnorm_cute` | CuTe DSL | FlashInfer; [kernel][add-norm] | Add/residual fusion, separate from the unfused kernel. |
| RMSNorm / residual RMSNorm | Triton | FlashInfer; [kernels][triton-norm] | `rms_norm`, `rms_norm_add_residual`; actual Triton backend, not a framework import. |
| RMSNorm / residual RMSNorm | cuTile | TRTLLM; [RMSNorm][tile-rms], [residual][tile-residual] | `rms_norm_kernel`, `rms_norm_fuse_residual_kernel`; gather/static-persistent variants. |
| Native LayerNorm / quantized LayerNorm | CUDA | FlashInfer; [device kernels][norm-native] | `layernorm`, `layernorm_quant`; affine mean/variance normalization. `layernorm_quant` has no CuTe backend at this snapshot. |
| LayerNorm: `layernorm_cute`, `LayerNormKernel` | CuTe DSL | FlashInfer; [kernel][layer-norm] | Mean/variance reduction, not RMSNorm. |
| RMSNorm + FP8 / residual quantization | CUDA | FlashInfer; [device kernels][norm-native] | `rmsnorm_quant`, `fused_add_rmsnorm_quant`, `fused_add_rmsnorm_fp8_block_quant`; scale/output contracts differ. |
| RMSNorm + FP8 / residual quantization | CuTe DSL | FlashInfer; [RMSNorm][rms], [residual][add-norm] | `RMSNormQuantKernel`, `FusedAddRMSNormQuantKernel`; scalar/device scale and quantized output. |
| RMSNorm + NVFP4 | CuTe DSL | FlashInfer; [norm][rms4], [residual][add-rms4] | `rmsnorm_fp4quant`, `add_rmsnorm_fp4quant`; packed FP4 plus scale factors. |
| Residual RMSNorm + FP4: `residualRmsNormFp4Quant` | CUDA | TRTLLM; [kernel][trt-rms4] | Native fused residual/norm/FP4 output. |
| Warp-specialized LayerNorm + FP4: `invokeWSLayerNorm` | CUDA | TRTLLM; [kernels][trt-ws] | Distinct warp-specialized algorithm, not RMSNorm. |
| Gated RMSNorm quant: `invokeFusedGatedRMSNormQuant` | CUDA | TRTLLM; [kernels][trt-gated] | Gate/norm/quant fusion. |
| RMSNorm quantization fusions | Helion | vLLM; [kernels][helion] | `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`; dynamic per-token versus block scales. |

FlashInfer normally selects CuTe norm when available; its native CUDA path
can be selected with `FLASHINFER_USE_CUDA_NORM=1`. This does not extend CuTe
support to the CUDA-only `layernorm_quant` API.

## Quantization and gated activation

| Implementation / entry | Language | Origin and source | Operation / constraints |
| --- | --- | --- | --- |
| Native NVFP4 / MXFP4 quantization | CUDA | FlashInfer, including vendor-derived kernels; [API/JIT entry][fp4] | `nvfp4_quantize`, `nvfp4_batched_quantize`, `mxfp4_quantize`; E2M1 values, distinct scale encodings and layouts. |
| NVFP4 direct-load quantization | CuTe DSL | FlashInfer; [kernels][nv4] | `NVFP4QuantizeLinearKernel`, `NVFP4QuantizeSwizzledKernel`; scale-vector size 16. |
| NVFP4 per-token quantization | CuTe DSL | FlashInfer; [kernel][nv4] | `NVFP4QuantizePerTokenKernel`; per-token scaling rather than ordinary global scale. |
| NVFP4 TMA quantization | CuTe DSL | FlashInfer; [kernel][nv4] | `NVFP4QuantizeTMAKernel`; TMA pipeline distinct from direct loads. |
| MXFP4 quantization | CuTe DSL | FlashInfer; [kernels][mx4] | `mxfp4_quantize_cute_dsl`; linear/swizzled E8M0 block scales. |
| SVDQuant smoothing + NVFP4 | C++ CUTLASS/CUDA | FlashInfer; [entry][smooth] | `nvfp4_quantize_smooth(backend="cutlass")`; SM100/103, BF16 per-channel smoothing, swizzled scales. |
| SVDQuant smoothing + NVFP4 | CuTe DSL | FlashInfer; [kernel][nv4] | `nvfp4_quantize_smooth_cute_dsl`; fused smoothing with aligned BF16 operands. |
| Native MXFP8 quantization | CUDA | FlashInfer/vendor lineage; [API/JIT entry][fp8] | `mxfp8_quantize(backend="cuda")`; native source selected by manifest. |
| MXFP8 quantization | CuTe DSL | FlashInfer; [kernels][mx8] | `mxfp8_quantize_cute_dsl`; linear/swizzled scale layouts. |
| Grouped MXFP8 quantization | cuTile | FlashInfer; [kernel][mx8-tile] | `mxfp8_grouped_quantize(backend="cutile")`; group-shape contract. |
| Grouped MXFP8 quantization | CUDA, generated | FlashInfer Cake; [manifest][mx8-cake] | `mxfp8_grouped_quantize(backend="cake")`; separate generated backend. |
| Per-token group 8-bit quantization | cuTile | FlashInfer; [kernel][group8] | `per_token_group_quant_8bit`; per-token groups, cuTile-only API at this snapshot. |
| Dynamic/token-group FP8 quantization | Helion | vLLM; [kernels][helion] | `dynamic_per_token_scaled_fp8_quant`, `per_token_group_fp8_quant`; separate scale granularity. |
| Native gated SiLU / GELU | CUDA | FlashInfer; [kernels][activation] | `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`; split gate/up multiply. |
| Gated SiLU | Triton | FlashInfer; [kernel][act-triton] | `triton.activation.silu_and_mul`; independent library backend. |
| Gated SiLU + NVFP4 | CUDA | FlashInfer/vendor-derived family; [entry][fp4] | `silu_and_mul_nvfp4_quantize`, `silu_and_mul_scaled_nvfp4_experts_quantize`; expert scale variants. |
| Gated SiLU + MXFP8 | CUDA | FlashInfer; [device sources][act-mx8] | `silu_and_mul_mxfp8_quantize`; forward activation/quantization fusion. |
| Gated SiLU + FP8/block quant | Helion | vLLM; [kernels][helion] | `silu_mul_fp8`, `silu_and_mul_per_block_quant`; fused alternatives with explicit output-scale contracts. |

GEMM-integrated activation/quantization is in [GEMM](gemm.md) or
[MoE](moe.md), not repeated as a standalone quantization kernel here.

## RoPE and KV cache

| Implementation / entry | Language | Origin and source | Operation / constraints |
| --- | --- | --- | --- |
| Native RoPE | CUDA | FlashInfer; [kernels][rope-native] | `apply_rope`, `apply_rope_pos_ids`, `apply_llama31_rope`, `apply_rope_with_cos_sin_cache`; in-place/scaling/interleave variants grouped. |
| GQA/MLA RoPE + FP8 quantization | CUDA | FlashInfer; [API/JIT entry][rope] | `rope_quantize_fp8`, `mla_rope_quantize_fp8`; ordinary Q/K/V and MLA latent/positional layouts differ. |
| GQA RoPE + FP8 quantization | cuTile | FlashInfer; [kernel][rope-tile] | `rope_quantize_fp8(backend="cutile")`; distinct implementation. |
| RoPE + FP8 + paged append | CUDA | FlashInfer; [entry][rope] | `rope_quantize_fp8_append_paged_kv_cache`; fused positional transform, quantization and cache write. |
| Fused QK norm + RoPE | CUDA | FlashInfer; [device kernel][qk-rope] | `fused_qk_rmsnorm_rope`; headwise normalization plus positional transform. |
| Fused QK norm + RoPE | Helion | vLLM; [kernel][helion] | `fused_qk_norm_rope`; independent implementation, not FI import. |
| Paged GQA/MLA cache append | CUDA | FlashInfer; [kernels][page] | `append_paged_kv_cache`, `append_paged_mla_kv_cache`; different dense versus latent cache contracts. |
| NVFP4 KV dequantization | CUDA | FlashInfer; [device kernels][kv-dequant] | `nvfp4_kv_dequantize`, `nvfp4_kv_dequantize_paged`; packed values/scales to floating point. |

Sparse compression and index selection belong to
[sparse attention](sparse-attention.md). Model-specific Triton cache glue
and metadata-only transforms are deliberately not enumerated.

## Sampling

| Implementation / entry | Language | Origin and source | Operation / constraints |
| --- | --- | --- | --- |
| Categorical: `sampling_from_logits`, `sampling_from_probs` | CUDA | FlashInfer; [device kernels][sampling] | Seed/offset and optional deterministic mode. |
| Top-k / top-p / min-p sampling | CUDA | FlashInfer; [device kernels][sampling] | `top_k_sampling_from_probs`, `top_p_sampling_from_probs`, `min_p_sampling_from_probs`; distinct filtering distributions. |
| Joint top-k/top-p sampling | CUDA | FlashInfer; [device kernels][sampling] | `top_k_top_p_sampling_from_logits`, `top_k_top_p_sampling_from_probs`. |
| AIR top-p renormalization: `top_p_renorm_probs` | CUDA | TRTLLM-AIR-derived implementation in FlashInfer; [kernel][air] | Radix threshold finding plus renormalization; filtering, not random sampling. |
| Top-k renormalization/masking | CUDA | FlashInfer; [device kernels][sampling] | `top_k_renorm_probs`, `top_k_mask_logits`; produces filtered probabilities/logits, not samples. |
| Chain speculative verification: `chain_speculative_sampling` | CUDA | FlashInfer; [device kernels][sampling] | Draft/target probabilities, acceptance and recovered-token outputs. |

## Hyperconnection fusions

| Implementation / entry | Language | Origin and source | Operation / constraints |
| --- | --- | --- | --- |
| mHC pre/post fusion | CUDA | FlashInfer; [device sources][mhc-fi] | `mhc_pre_big_fuse`, `mhc_pre_big_fuse_with_prenorm`, `mhc_post`; distinct pre/post phases. |
| mHC fused pre/post/norm/broadcast | TileLang | SGLang-derived implementation adapted in vLLM; [vLLM kernels][mhc-tile] | `mhc_pre_big_fuse_tilelang`, `mhc_fused_tilelang`, `mhc_post_tilelang`; grouped substantial pipeline, not local Triton reduction helpers. |

[norm-native]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/norm.cuh
[qk-rope]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh
[kv-dequant]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/fp4_kv_dequantization.cu
[rms]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/norm/kernels/rmsnorm.py
[add-norm]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/norm/kernels/fused_add_rmsnorm.py
[triton-norm]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/triton/kernels/norm.py
[tile-rms]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cuda_tile_kernels/rms_norm.py
[tile-residual]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cuda_tile_kernels/rms_norm_fuse_residual.py
[layer-norm]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/norm/kernels/layernorm.py
[rms4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/rmsnorm_fp4quant.py
[add-rms4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/add_rmsnorm_fp4quant.py
[trt-rms4]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/rmsNormFp4QuantKernels.cu
[trt-ws]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fusedLayernormKernels
[trt-gated]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant
[helion]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/kernels/helion/ops
[fp4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/fp4_quantization.py
[nv4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/nvfp4_quantize.py
[mx4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/mxfp4_quantize.py
[smooth]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/gemm_svdquant.py
[fp8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/fp8_quantization.py
[mx8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/mxfp8_quantize.py
[mx8-tile]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/cutile/mxfp8_grouped_quantize_cutile.py
[mx8-cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_grouped_mxfp8_quantize.py
[group8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/cutile/per_token_group_quant_8bit_cutile.py
[activation]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/activation.cuh
[act-triton]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/triton/activation.py
[act-mx8]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/gated_act_mxfp8
[rope-native]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/pos_enc.cuh
[rope]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/rope.py
[rope-tile]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/quantization/kernels/cutile/rope_quantize_fp8_cutile.py
[page]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/page.cuh
[sampling]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/sampling.cuh
[air]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/air_top_p.cuh
[mhc-fi]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/mhc
[mhc-tile]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/mhc/tilelang_kernels.py
