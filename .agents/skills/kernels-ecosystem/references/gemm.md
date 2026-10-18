# GEMM: dense, batched, grouped and skinny

Return to the [primitive index](../SKILL.md). Expert FC1/activation/FC2
pipelines belong in [MoE](moe.md); GEMM/communication fusions belong in
[collectives](collectives.md).

Start with the operand format and scheduling problem below. **CuTe DSL is
Python DSL; CUDA/C++ CUTLASS/CuTe is native C++.** Scale granularity, packed
weights and output dtype are part of each contract, not interchangeable
between implementations. Architecture/tile variants are grouped.

Prefer upstream device source/project, then vLLM's implementation or vendored
copy. Keep other frameworks' material adaptations at their own source.
"Provider evidence" and integration links establish an external callable, not
audited device source. Re-exports are not additional candidates.

## BF16/FP16 dense and low-latency algorithms

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| CUTLASS dense/batched - FlashInfer | `bf16_gemm_sm100`, `mm_bf16`, `bmm_bf16` with `backend="cutlass"` | CUDA/C++ CUTLASS; BF16 operands, output types depend on entry. [Native templates][fi-gemm-src], [JIT/source manifest][fi-jit], [contract][fi-base]. |
| Persistent tensor-core dense/BMM - NVIDIA TRTLLM | `PersistentDenseGemmKernel`, `SM107PersistentDenseGemmKernel` | CuTe DSL; Blackwell BF16/FP8 and Rubin BF16. Rubin preferred-cluster BMM uses `PersistentDenseGemmKernelPreferredCluster`; same family, distinct schedule. [Blackwell][trt-dense], [Rubin][trt-rubin]. |
| TGV low-latency - FlashInfer | `tgv_gemm_sm100`, `TgvGemmCuteExtKernel` | Two implementations: CUDA/C++ CuTe and CuTe DSL `cute_ext`; Blackwell BF16, bias/PDL, 1-/2-CTA variants. [C++ source][tgv-cpp], [DSL source][tgv-dsl]. |
| TinyGEMM2 - NVIDIA TRTLLM, also exposed by FlashInfer | `launch_tinygemm2`, `tinygemm2_cuda_forward`; `tinygemm_bf16` exposure | CUDA small-M family, not TGV or tensor-core split-K. [Native source][tiny-trt], [FlashInfer manifest][tiny-fi]. |
| CTA-local direct/dot-product - vLLM-derived | `LLBf16Dotprod`; FlashInfer `DirectDenseGemmKernel` adaptation | CuTe DSL small-M BF16. TokenSpeed vendors the vLLM body; FlashInfer adds its own direct-kernel dispatch/fallback, M<=32. [vLLM source directory][v-cute], [vendored copy][ts-dot], [adaptation][fi-direct]. |
| Clustered tensor-core split-K - vLLM-derived | `LLBf16SplitK` | CuTe DSL BF16 small-M router projection; TokenSpeed vendors the kernel with its own tuning. Not assumed identical to FlashInfer's split-K family. [vLLM `_ll_bf16_splitk.py`][v-cute], [copy][ts-split]. |
| Tensor-core and warp split-K - FlashInfer | `SplitKDenseGemmKernel`, `CpAsyncWarpSplitKKernel` | Two CuTe DSL algorithms: SM100 tensor-core split-K and cp.async warp split-K with optional bias. SGLang's `run_splitk_dense_silu`/`run_splitk_dense_gate` add fused epilogues to the SM100 family. [SM100][fi-split], [warp][fi-warp], [epilogue variant][sgl-split]. |
| Shape-dynamic skinny - vLLM-derived | `ShapeDynamicSkinnyGemm`, `CuteSkinnyGemm` | CuTe DSL FP16/BF16 small-M. TokenSpeed's vendored family includes `skinny_gemv_add3`; not another generic GEMM provider. [vLLM `_skinny_gemm.py`][v-cute], [copy][ts-skinny]. |
| TGV-style tiny/skinny - SGLang | `cutedsl_bf16_gemm`, `cutedsl_bf16_gemm_out` | In-tree CuTe DSL implementation with its own eligibility/tactics. Similar naming is not evidence of identity with FlashInfer TGV. [Source][sgl-cute]. |
| Skinny-N / skinny-K projection - vLLM | `KdaSkinnyGemm`, `_KdaSkinnyNGemm`, `_KdaSkinnyKGemm` | CuTe DSL model-shaped KDA projections; N- and K-skinny algorithms grouped, not generic dense replacements. [Source][v-kda]. |
| BF16x3 accurate router projection - vLLM | `bf16x3_router_gemm` | CuTe DSL FP32-oriented projection accuracy; not top-k routing. [Source][v-router]. |
| Dense/BMM and masked/ragged matrix multiplication - FlashInfer | `mm_bf16`, `bmm_bf16` with `backend="cutile"`; `masked_bmm`, `ragged_bmm`, `gemm_alpha_beta` | cuTile, not CuTe. Includes explicitly scaled masked/ragged and ragged block-scaled variants; their metadata contracts differ. [Kernel directory][fi-cutile]. |
| Task-scheduled batched GEMM - FlashInfer prims-ts | `batched_gemm_run.compile`, `batched_gemm_run.run` | Experimental CuTe DSL task scheduling; BF16/FP8/FP4 configured token layouts and epilogue tasks. [Kernel][fi-ts]. |

## FP8, MXFP8 and INT8 scaled GEMM

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| CUTLASS FP8 tensor/group/block scaling - FlashInfer | `bmm_fp8`, `fp8_gemm_sm100`, `fp8_blockscale_gemm_sm90`, `gemm_fp8_nt_groupwise` | CUDA/C++ CUTLASS; per-entry scales and SM90/100/120 specializations. [Native templates][fi-gemm-src], [manifest][fi-jit], [contract][fi-base]. |
| CUTLASS rowwise/blockwise FP8 - NVIDIA TRTLLM | `FP8RowwiseGemmRunner`, `fp8_block_scaling_gemm` | Native C++ CUTLASS; row/column versus block-scale contracts. Not TRTLLM-GEN artifacts. [Rowwise source][trt-row], [block source][trt-block]. |
| CUTLASS scaled INT8/FP8 - vLLM and SGLang source branches | `cutlass_scaled_mm`, `cutlass_scaled_mm_azp`; `fp8_scaled_mm`, `int8_scaled_mm`, `fp8_blockwise_scaled_mm` | CUDA/C++ CUTLASS family; retain branch-specific zero-point, scaling and SM120 JIT contracts rather than claiming identical bodies. [vLLM source][v-scaled], [SGLang AOT binding][sgl-scaled], [SGLang JIT source][sgl-gemm]. |
| SmoothQuant INT8 tensor-core - NVIDIA TRTLLM | `CutlassInt8GemmRunner::gemm` | C++ CUTLASS INT8 with activation/weight scale granularity. [Source][trt-int8]. |
| SmoothQuant INT8 CUDA-core - NVIDIA TRTLLM | `int8_sq_launcher`, `int8_sq_kernel` | Separate small-M CUDA-core algorithm, not the CUTLASS implementation. [Source][trt-int8-cuda]. |
| FP8 low-latency CUTLASS - NVIDIA TRTLLM | `CutlassLowLatencyFp8GemmRunner::gemm` | Native CUDA/C++ CUTLASS small-M FP8 specialization. **Not** the generated-artifact implementation exposed by FlashInfer's `trtllm_low_latency` tag. [Native templates][trt-low]. |
| Blockwise FP8 persistent - NVIDIA TRTLLM | `Sm100BlockwiseGemmKernel`, `SM107BlockwiseGemmKernel` | CuTe DSL Blackwell/Rubin blockwise scaling; distinct from tensor-scaled dense/BMM. [Source][trt-block-dsl]. |
| FP8 batched / masked grouped - FlashInfer | `bmm_fp8(backend="cute-dsl")`, `grouped_gemm_masked_wrapper` | CuTe DSL Blackwell/Rubin implementations; ordinary BMM and masked expert-major buffers are different contracts. [BMM source][fi-bmm], [masked source][fi-masked]. |
| Low-latency blockscaled - FlashInfer | `mm_fp8`, `mm_mxfp8`, `mm_fp4` with `backend="cutedsl_low_latency"` | CuTe DSL small-M algorithm, format-specific scales; FP8 additionally accepts E5M2. [Source][fi-low]. |
| FP8 groupwise persistent - TileGym-derived FlashInfer cuTile | `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise(backend="cutile")` | cuTile dense/grouped NT; grouped persistent launch uses device `m_indptr`. [Source][fi-tile-fp8]. |
| MXFP8 tensor-core - CUTLASS | `mxfp8_gemm_sm100`, `bmm_mxfp8(backend="cutlass")`; TRTLLM `mxfp8_mxfp8_gemm_autotuned` | CUDA/C++ blockscaled family; E8M0 scales, architecture-specific padded/swizzled layouts. [FlashInfer manifest][fi-jit], [TRTLLM template][trt-mxfp8]. |
| TRTLLM-GEN dense/batched - NVIDIA | `gemm/KernelRunner`, `fp8_batched_gemm_trtllmgen`; FlashInfer `gemm_fp8_nt_groupwise`, `mm_mxfp8`, `mm_fp4` with `backend="trtllm"` | Generated CUDA family, listed once across native/exposed APIs. Includes `mm_fp8(backend="trtllm_low_latency")`, which loads TRTLLM-GEN GEMM artifacts. FP8/MXFP8/FP4 have different ABIs; not every dtype supports every entry. [Native boundary][trt-gen], [FlashInfer contract][fi-base], [low-latency exposure][fi-low-trt]. |
| DeepGEMM blockscaled dense/grouped/batched - DeepGEMM | `fp8_gemm_nt`, `m_grouped_fp8_gemm_nt_contiguous`, `m_grouped_fp8_gemm_nt_masked`; FlashInfer `group_deepgemm_fp8_nt_groupwise` | CUDA provider, one family across direct and FlashInfer integrations. Contiguous/masked expert layouts and MXFP8 scale routes are not interchangeable. [Upstream device kernels][dg-source] at [vLLM's dependency pin][dg-pin], [vLLM integration][dg-evidence], [FlashInfer integration][fi-dg]. |

## Grouped GEMM without a complete expert pipeline

For fused expert activation/finalization, use [MoE](moe.md).

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| Segmented CUTLASS - FlashInfer | `SegmentGEMMWrapper` | CUDA/C++ CUTLASS SM80/90 packed segments, not full MoE. [Implementation/contract][fi-base]. |
| Native grouped low-precision CUTLASS - FlashInfer | `group_gemm_fp8_nt_groupwise`, `group_gemm_mxfp8_mxfp4_nt_groupwise`, `group_gemm_nvfp4_nt_groupwise` | CUDA/C++ grouped FP8 and mixed/blockscaled FP4. FP8 `backend="trtllm"` builds local SM100/120 kernels: **not** TRTLLM-GEN downloaded GEMM. [Native templates][fi-gemm-src], [source manifest][fi-jit]. |
| SM120 groupwise FP8 / MXFP8 - FlashInfer | `moe_gemm_fp8_nt_groupwise`, `moe_gemm_mxfp8_nt_groupwise` in `grouped_mm/cute_sm120_*` | **CUDA/C++ CuTe**, not Python DSL despite module names. Distinct scale-format specializations in `csrc/cute_sm12x_gemm`. [JIT/source boundary][fi-group120]. |
| Tensor-core grouped / split-K - NVIDIA TRTLLM | `groupedGemm`, `splitkGroupedGemm` | CUDA/C++ grouped problem arrays and split-K workspace/reduction. `cudaGraphGroupedGemm`/`cudaGraphSplitKGroupedGemm` are execution variants, not new arithmetic. [Grouped source][trt-group], [split-K source][trt-split-group]. |
| LoRA grouped projection - NVIDIA TRTLLM | `loraGroupGEMMParamFillRowReorderFusion` and LoRA GEMM runner | CUDA/C++ CUTLASS grouped low-rank projection; rank/adapter metadata and row reorder are part of this family. [Source][trt-lora]. |

## FP4 and weight-only / mixed-precision algorithms

NVFP4 (block 16), MXFP4 (block 32), integer INT4 and W4A16 dequantized
compute are different formats/algorithms. Do not substitute by bit width.

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| Blockscaled FP4 tensor-core - CUTLASS | `mm_fp4(backend="cutlass")`; TRTLLM `FP4GemmRunner`; vLLM `cutlass_scaled_fp4_mm` | CUDA/C++ family, packed E2M1 and backend-specific scales. SM100/103/120 specializations; MXFP8xMXFP4 mixed inputs use a separate type/scale contract. [FlashInfer manifest][fi-jit], [TRTLLM source][trt-fp4], [vLLM source][v-fp4]. |
| Persistent blockscaled - TRTLLM/CUTLASS-derived | `Sm100BlockScaledPersistentDenseGemmKernel`, `Sm107BlockScaledPersistentDenseGemmKernel`; FlashInfer `mm_fp4`/`mm_mxfp8(backend="cute-dsl")` | CuTe DSL family: FlashInfer SM100 is ported from TRTLLM; SM103/Rubin branches derive from CUTLASS examples. Group architecture variants, not duplicate framework entries. [TRTLLM source][trt-bs], [ported source and attribution][fi-bs]. |
| Split-K blockscaled - FlashInfer | `Sm100BlockScaledSplitKGemmKernel` | CuTe DSL SM100 FP4 with separate reduction epilogue; not merely a persistent-kernel tile. [Source][fi-bs-split]. |
| SM12x blockscaled - CUTLASS-example-derived FlashInfer | `mm_fp4`, `mm_mxfp8` with `backend="b12x"` | CuTe DSL SM120/121 port of a CUTLASS example. API tag alone does not establish identity with external B12x GEMM. [Source][fi-bs120]. |
| CUDA-core NVFP4 - NVIDIA TRTLLM | `CudaCoreNVFP4Runner` | Small-M dequantize/accumulate alternative to tensor-core FP4. [CUDA source][trt-fp4-core]. |
| Native / tiled BF16xFP4 - FlashInfer | `mm_bf16_fp4(backend="blackwell-native")`, `backend="blackwell-tiled"` | Two CUDA algorithms; native uint8 weights/linear E4M3 scales versus tiled int32 weights/S0E5M3 scales, N multiple of 64. [Implementation boundary][fi-w4-cuda]. |
| Mixed-input W4A16 - FlashInfer | `mm_bf16_fp4`; `dense_gemm_bf16_fp4_sm12x`, `gemv_bf16_fp4_sm12x` | CuTe DSL SM100/SM12x dense kernels plus distinct small-M GEMV; BF16 activations, prepared FP4 weights. [Dense source][fi-w4-dsl], [SM12x source directory][fi-w4-dir]. |
| Weight-only mixed-input tensor-core - NVIDIA TRTLLM | `weight_only_quant_gemm`, `finegrained_mixed_dtype_gemm` | C++ CUTLASS FP16/BF16xINT4/INT8; per-channel versus groupwise scale/zero contracts. [Source][trt-mixed]. |
| Weight-only batched GEMV - NVIDIA TRTLLM | `weight_only::dispatcher` in `kernelDispatcher.h` | CUDA-core INT4/INT8 dequantization; column/interleaved layouts and dtype instantiations grouped. [Source][trt-weight]. |
| Marlin - IST-DASLab lineage and adapted copies | `gptq_marlin_gemm`, `marlin_gemm`, `marlin_nvfp4_gemm` | CUDA packed-weight family covering supported GPTQ/AWQ integer, FP8, NVFP4/MXFP4/MXFP8 routes. TRTLLM NVFP4 is W4A16 on SM89-99; formats require distinct repacking/scales. [Original project][marlin-origin]; [vLLM adapted source][v-marlin] and [TRTLLM adaptation][trt-marlin] establish the extended formats, not the original project. SGLang/TokenSpeed copies are not extra families. |
| Machete - vLLM | `machete_mm`, `machete_prepack_B` | CUDA/C++ CUTLASS mixed precision with prepacked weights; not Marlin. [Source][v-machete]. |
| GPTQ / ExLlama - adapted CUDA | `gptq_gemm` | Packed GPTQ weights and optional shuffled representation; separate from GPTQ-on-Marlin. [vLLM source][v-gptq]. |
| AWQ native - adapted CUDA | `awq_gemm`, `awq_dequantize` | Native AWQ path, distinct from AWQ-on-Marlin. [vLLM source][v-awq]. |
| AllSpark - adapted CUDA | `allspark_w8a16_gemm` | W8A16 with its own repacked weight/scale layout. [vLLM source][v-allspark]. |
| QServe - TRTLLM adaptation | `QServeGemmRunner::gemmPerChannel`, `gemmPerGroup` | CUDA W4A8 family; channel/group scales and zero contracts differ. [Per-channel source][trt-qserve], [per-group source][trt-qserve-group]. |
| GGML/GGUF - adapted CUDA | `ggml_mul_mat_a8`, `ggml_mul_mat_vec_a8` | Quantized matrix/matrix and matrix/vector consumers; one GGML family rather than each framework's import. [SGLang retained source][ggml]. |

## Fused projection algorithms

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| Gated GEMM tensor-core - NVIDIA TRTLLM | `CutlassFusedGatedGemmRunner::gemm`; `gemmGatedAct/KernelRunner` | Distinct C++ CUTLASS Hopper and TRTLLM-GEN generated gated-activation implementations. [CUTLASS source][trt-gated], [generated boundary][trt-gen-gated]. |
| GEMM + SwiGLU + FP4 output quantization - LightSeek | `nvfp4_gemm_swiglu_nvfp4_quant`, `Sm100BlockScaledPersistentDenseGemmKernel` | Experimental CuTe DSL SM100 fused epilogue; packed operands/scales and interleaved even-N FC1. Shared SGLang/TokenSpeed implementation, one entry. [SGLang source][sgl-fused4], [TokenSpeed copy][ts-fused4]. |
| NVFP4 + low-rank SVDQuant - FlashInfer | `mm_nvfp4_svdquant` | Separate `cutlass` (C++ CUTLASS), `cake` (generated CUDA), `cute-dsl` fused and `cute-dsl-unfused` branches. The unfused composition is an explicit experimental comparison, not a new low-bit format. [Dispatch and implementation boundaries][fi-svd]. |
| Fused attention A projection - NVIDIA TRTLLM / SGLang | `invokeFusedAGemm`; CuTe `dsv3_fused_a_gemm` | Shape-specialized CUDA fused projection and separate SGLang CuTe DSL implementation; neither is the router GEMM. [CUDA source][trt-a], [DSL source][sgl-a]. |

## AMD dense and skinny algorithms

| Implementation family / origin | Concrete entry | Contract and pinned evidence |
| --- | --- | --- |
| Warp-reduction / MFMA-LDS dense - TokenSpeed AMD | `gluon_mm_a16w16_warp_reduce_smallm_gfx950`, `gluon_mm_a16w16_mfma_lds_smallm_gfx950` | Two Gluon algorithms: warp dot/reduction and LDS/MFMA tensor-core. Medium-M, batched and add3 epilogues stay with this A16W16 family. [Source][ts-amd950]. |
| Large-M MFMA dense - TokenSpeed AMD | `gluon_mm_a16w16_largem_gfx950` | Gluon dedicated large-M pipeline, not a small-M GEMV tile. [Source][ts-amd-large]. |
| WMMA/TDM dense - TokenSpeed AMD | `gluon_wmma_tdm_dense_gfx1250`, `gluon_wmma_dense_gemv_gfx1250` | Gluon gfx1250 matrix/decode family; model-shaped QKV and add3 projection epilogues grouped. [Source][ts-amd1250]. |
| FP8 split-K skinny - vLLM | `wvSplitKQ` | Native HIP small-M algorithm; higher-M PyTorch fallback is not another local implementation. [HIP source][v-rocm]. |

## Dense and quantized provider implementations

These are **providers**, not inference-framework adapter classes. Citations
below are pinned dependency contracts unless explicitly described as source;
they do not establish every package's internal kernel language or eligibility.

| Provider family | Concrete provider entry | Scope / evidence |
| --- | --- | --- |
| cuBLAS / cuBLASLt | cuBLAS-backed `bmm_fp8`; `mm_bf16(backend="cublaslt")`, `nvfp4_gemm_cublaslt` | Vendor CUDA-library dense/BMM alternatives; BF16, FP8 and NVFP4 availability depends on API/library version. [FlashInfer provider boundary][fi-base], [TRTLLM provider boundary][trt-ops]. |
| cuDNN | `mm_bf16`, `bmm_fp8`, `mm_fp4`, `mm_bf16_fp4` with `backend="cudnn"`; `grouped_mm_bf16/fp8/mxfp8/fp4` | Graph/workspace/tactic provider, not a CUTLASS kernel. Per-format graph/scale contracts. [Dense evidence][fi-base], [grouped evidence][fi-grouped], [W4A16 evidence][fi-w4]. |
| OpenAI `triton_kernels` | `matmul` | Substantive upstream Triton GEMM/ragged grouped family; unquantized, FP8 QDQ and MXFP4 precision contexts. [Upstream device source][openai-source] at the release recorded by [TRTLLM's vendoring provenance][openai-pin]; [integration contract][trt-openai]. This is TRTLLM's retained v3.7.0 source, not vLLM's different dependency pin. |
| B12x | `mm_block_fp8`, `tensor_fp8_linear` | External block/tensor FP8 provider plus NVFP4/MX formats; do not equate all B12x names with FlashInfer's example-derived GEMM. [Provider evidence][v-b12x]. |
| Humming | `HummingMethod`, `humming_forward` | External quantized dense/indexed/grouped family, INT8/FP8 and supported low-bit/MX formats. [Upstream project][humming-origin]; [vLLM integration contract][v-humming]. The adapter alone does not establish device language or every upstream version's eligibility. |
| AITER | `gemm_a8w8`, `gemm_a8w8_blockscale` | ROCm HIP/ASM provider and separate tuned Triton routes; scale/layout and architecture predicates matter. [Provider evidence][v-aiter]. |
| FBGEMM | `torch.ops.fbgemm.f4f4bf16` | External NVFP4 GEMM to BF16. [Provider evidence][v-fbgemm]. |
| Conch | `conch.ops.quantization.gemm.mixed_precision_gemm` | External mixed-precision GEMM; not a local vLLM implementation. [Provider evidence][v-conch]. |
| BitsAndBytes | `bitsandbytes.matmul`, `bitsandbytes.matmul_4bit` | External 8-/4-bit weight consumers; separate packed contracts. [Provider evidence][sgl-bnb]. |
| Petit | `petit_kernel.mul_nvfp4_a16` | External ROCm NVFP4 W4A16 provider with architecture verification. [Provider evidence][sgl-petit]. |

Isolated framework-local Triton GEMMs, linear adapter subclasses,
emulation/reference fallbacks and packing-only helpers are deliberately
omitted. This is a candidate inventory, not a complete dtype/export census.

[fi-base]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/gemm_base.py
[fi-jit]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/gemm
[fi-gemm-src]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/gemm
[tgv-cpp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/gemm/tgv_gemm.cuh
[tgv-dsl]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/tgv_gemm_cute_ext.py
[tiny-fi]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/tinygemm2.py
[fi-direct]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_bf16_gemm_direct.py
[fi-split]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
[fi-warp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py
[fi-cutile]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/cutile
[fi-ts]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/prims_ts/batched_gemm/batched_gemm_kernel.py
[fi-low-trt]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/trtllm_low_latency_gemm.py
[fi-bmm]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/bmm_fp8_blackwell.py
[fi-masked]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py
[fi-low]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py
[fi-tile-fp8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/cutile/gemm_fp8_nt_groupwise_cutile.py
[fi-dg]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/deep_gemm.py
[fi-group120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cute_sm12x_gemm.py
[fi-bs]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100.py
[fi-bs-split]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py
[fi-bs120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py
[fi-w4-cuda]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/gemm_bf16_fp4_blackwell.py
[fi-w4-dsl]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/cute_dsl/dense_gemm_bf16_fp4_sm100.py
[fi-w4-dir]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/kernels/cute_dsl
[fi-svd]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/gemm_svdquant.py
[fi-grouped]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/grouped_mm/core.py
[fi-w4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gemm/gemm_bf16_fp4.py
[trt-dense]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_gemm_persistent.py
[trt-rubin]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/rubin
[tiny-trt]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/tinygemm2/tinygemm2_cuda.cu
[trt-row]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_rowwise_gemm
[trt-block]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm
[trt-int8]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm
[trt-int8-cuda]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/int8SQ.cu
[trt-low]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/low_latency_gemm
[trt-block-dsl]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockwise_gemm/blockwise_gemm.py
[trt-mxfp8]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/mxfp8_mxfp8_gemm_template_sm100.h
[trt-gen]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/gemm
[trt-group]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/groupGemm.cu
[trt-split-group]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/splitkGroupGemm.cu
[trt-lora]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/lora
[trt-fp4]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm
[trt-bs]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dense_blockscaled_gemm_persistent.py
[trt-fp4-core]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemmNVFP4.cu
[trt-mixed]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm
[trt-weight]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv
[trt-marlin]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/marlin
[trt-qserve]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/qserveGemmPerChannel.cu
[trt-qserve-group]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/qserveGemmPerGroup.cu
[trt-gated]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/fused_gated_gemm
[trt-gen-gated]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/gemmGatedAct
[trt-a]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
[trt-ops]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py
[trt-openai]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/modules/triton_linear.py
[v-cute]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/linear/cute_dsl
[v-kda]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py
[v-router]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_cutedsl.py
[v-scaled]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/w8a8/cutlass
[v-fp4]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/fp4
[v-marlin]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/marlin
[v-machete]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/machete
[v-gptq]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/gptq
[v-awq]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/awq
[v-allspark]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/quantization/gptq_allspark
[v-rocm]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/rocm
[v-b12x]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/linear/scaled_mm/b12x.py
[v-aiter]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/linear/scaled_mm/aiter.py
[v-fbgemm]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/linear/nvfp4/fbgemm.py
[v-conch]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/kernels/linear/mixed_precision/conch.py
[sgl-scaled]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/python/sgl_kernel/gemm.py
[sgl-gemm]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/gemm
[sgl-cute]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py
[sgl-split]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py
[sgl-fused4]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py
[sgl-a]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/gemm/cutedsl_dsv3_fused_a_gemm.py
[ggml]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/csrc/quantization/gguf
[dg-source]: https://github.com/deepseek-ai/DeepGEMM/tree/8b1392b978f5a03c828dd1711090d7fb50958b8a/deep_gemm/include/deep_gemm/impls
[dg-pin]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/tools/install_deepgemm.sh
[dg-evidence]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/utils/deep_gemm.py
[openai-source]: https://github.com/triton-lang/triton/blob/5f3f125e8f63c24613f1f73b937442864f263f94/python/triton_kernels/triton_kernels/matmul_details/_matmul.py
[openai-pin]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/triton_kernels/README.md
[marlin-origin]: https://github.com/IST-DASLab/marlin
[humming-origin]: https://github.com/vllm-project/humming
[v-humming]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/quantization/utils/humming_utils.py
[sgl-bnb]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/quantization/bitsandbytes.py
[sgl-petit]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/quantization/petit_utils.py
[ts-dot]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py
[ts-split]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_splitk_kernel.py
[ts-skinny]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/skinny_gemm/__init__.py
[ts-fused4]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/nvfp4_gemm_swiglu_nvfp4_quant.py
[ts-amd950]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/gemm/fp16/mm.py
[ts-amd-large]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/gemm/fp16/largem.py
[ts-amd1250]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/gemm/fp16/mm.py
