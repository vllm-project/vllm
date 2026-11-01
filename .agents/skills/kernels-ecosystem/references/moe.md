# MoE: expert compute by format and algorithm

Return to the [primitive index](../SKILL.md). For standalone dense/grouped
matrix multiplication, see [GEMM](gemm.md). For EP dispatch/combine protocols,
see [collectives](collectives.md#expert-alltoall), not the expert table.

An entry represents expert GEMMs or an FC1/activation/FC2 pipeline, not a
framework's backend adapter. Gather, activation, weighted finalize, tile
choices and routing configurations stay with their implementation family.
CuTe DSL, CUDA/C++ CuTe, cuTile, Gluon and Triton are different technologies.
Prefer upstream device source/project, then vLLM source; preserve other
frameworks' material adaptations rather than relabeling imports as kernels.

## Grouped tensor-core and generated expert pipelines

| Implementation family / origin | Concrete entry | Format / algorithm and pinned evidence |
| --- | --- | --- |
| CUTLASS fused experts - NVIDIA TRTLLM lineage, exposed by FlashInfer | `cutlass_fused_moe`, `CutlassBf16Config`, `CutlassFp8PerTensorConfig`, `CutlassFp8BlockConfig` | CUDA/C++ CUTLASS BF16 and tensor-/block-scaled FP8. One native/vendored family, not one per framework; routing scales and TP/EP metadata are separate contracts. [TRTLLM math][trt-cutlass], [FlashInfer source manifest][fi-jit], [format contracts][fi-api]. |
| CUTLASS blockscaled / mixed-input experts - same lineage | `CutlassNvfp4Config`, `CutlassMxfp8Config`, `CutlassMxfp8Mxfp4Config`, `CutlassW4A16Config`, `CutlassW4A8Config` | CUDA/C++ CUTLASS format branches: NVFP4 W4A4, MXFP8, mixed MXFP8xMXFP4 and weight-only/mixed-input modes. Not interchangeable scales or activation dtypes. [Native math][trt-cutlass], [concrete format contracts][fi-api]. |
| Humming-style CUTLASS mixed-input fusion - NVIDIA TRTLLM/FlashInfer | `CutlassHummingConfig` | CUDA/C++ CUTLASS SM90: BF16-facing activation, MXFP4 weights, pre-MMA E8M0 fusion and mixed-input interleave. Distinct algorithm from the external Humming package provider below. [Contract][fi-api], [source manifest][fi-jit]. |
| TRTLLM-GEN fused experts - NVIDIA | `trtllm_bf16_moe`, `trtllm_fp8_per_tensor_scale_moe`, `trtllm_fp8_per_channel_scale_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp4_block_scale_moe`, `trtllm_mxint4_block_scale_moe` | Generated CUDA family, one entry for native TRTLLM and FlashInfer exposure. Routed/unrouted are contracts, not duplicate algorithms. BF16, FP8 tensor/channel/block scaling, NVFP4/MXFP4 and integer MXINT4 retain distinct ABIs. [Native generated boundary][trt-gen], [native format callables][trt-gen-ops], [FlashInfer exposure][fi-core]. |
| CUTLASS grouped experts - vLLM source branch | `cutlass_moe_mm`, `cutlass_fp4_moe_mm`, `cutlass_mxfp4_moe_mm`, `cutlass_w4a8_moe_mm` | CUDA/C++ FP8, NVFP4, MXFP4 and W4A8 expert kernels. These retained source branches are not imports of FlashInfer; scale/shape contracts remain branch-specific. [Native source directory][v-moe], [call contracts][v-cutlass]. |
| CUTLASS grouped FP8 / W4A8 - SGLang source branch | `fp8_blockwise_scaled_grouped_mm`, `cutlass_w4a8_moe_mm` | CUDA/C++ blockwise FP8 and mixed-input W4A8. Compare the retained branch, not its framework runner enumeration. [FP8 binding][sgl-moe], [W4A8 binding][sgl-w4]. |
| Expert-specialized CUTLASS - SGLang | `es_fp8_blockwise_scaled_grouped_mm`, `es_sm100_mxfp8_blockscaled_grouped_mm` | Expert-specialized FP8/MXFP8 grouped algorithms, not merely the ordinary grouped runner renamed. MXFP8 JIT body is **C++ CUTLASS**, not Python CuTe. [Bindings][sgl-es], [SM100 source][sgl-es-src]. |

## CuTe DSL gather/activation/finalize pipelines

| Implementation family / origin | Concrete entry | Format / algorithm and pinned evidence |
| --- | --- | --- |
| Hopper contiguous BF16 - CUTLASS-example-derived FlashInfer | `cute_dsl_fused_moe_bf16`, `CuteDslBf16MoEWrapper` | SM90 CuTe DSL gather+activation GEMM1 and weighted-finalize GEMM2; stages are one pipeline. [Source][fi-hopper]. |
| Blackwell blockscaled grouped - NVIDIA TRTLLM, adapted by FlashInfer | `Sm100BlockScaledContiguousGroupedGemmKernel`, `BlockScaledContiguousGatherGroupedGemmKernel`; `cute_dsl_fused_moe_nvfp4`, `cute_dsl_fused_moe_mxfp8_mxfp4` | CuTe DSL SM100/103 NVFP4 W4A4 or mixed W4A8. Gather/SwiGLU/SiTU and weighted-finalize stages remain grouped; SiTU and W4A8 restrictions are explicit. [TRTLLM grouped source][trt-group], [FlashInfer pipeline][fi-cute], [port attribution][fi-port]. |
| Rubin contiguous grouped - NVIDIA TRTLLM, adapted by FlashInfer | `Sm107ContiguousGatherGroupedGemmSwigluFusionKernel`, `Sm107BlockScaledContiguousGatherGroupedGemmActFusionKernel` | CuTe DSL BF16 and blockscaled expert pipelines with matching finalize kernels; not a Blackwell tile choice. [Native BF16 source][trt-rubin], [ported family][fi-rubin]. |
| Blackwell mixed-input W4A16 - FlashInfer | `Sm100W4A16GroupedGemmKernel` | CuTe DSL NVFP4 weights with BF16 activations; distinct from W4A4/W4A8. [Source][fi-w4a16]. |
| Experts-as-dense NVFP4 - NVIDIA TRTLLM | `cute_dsl_nvfp4_dense_gemm_swiglu_moe_blackwell`, `cute_dsl_nvfp4_dense_gemm_fc2_blackwell` | CuTe DSL SM100/103 dense expert FC1+SwiGLU and FC2+combine rather than ordinary grouped GEMM. [FC1 source][trt-dense1], [FC2 source][trt-dense2]. |
| B12x static/dynamic NVFP4 - B12x port in FlashInfer | `MoEStaticKernel`, `MoEDynamicKernel`, `b12x_fused_moe` | CuTe DSL SM120/121 routed-block static and dynamic work scheduling; generic/gated forms grouped. No extra entry for framework imports. [Static source][fi-b12-static], [dynamic source][fi-b12-dynamic], [port attribution][fi-b12-port]. |
| B12x micro/direct-micro NVFP4 - same origin | `MoEMicroKernel`, `MoEDirectMicroKernel` | CuTe DSL small-token micro and direct-micro algorithms, distinguished from full static/dynamic scheduling. [Micro source][fi-b12-micro], [direct source][fi-b12-direct]. |
| B12x W4A16 - same origin | `moe_w4a16_kernel`, `B12xW4A16Config` | CuTe DSL BF16 activation/NVFP4 weight compute; route compaction is support, not another candidate. [Source][fi-b12-w4]. |

## Alternative scheduling and decode-specialized compute

| Implementation family / origin | Concrete entry | Format / algorithm and pinned evidence |
| --- | --- | --- |
| cuTile experts - FlashInfer | `CuTileBf16Config`, `CuTileNvfp4Config` | Separate cuTile BF16 and packed NVFP4 expert GEMMs/activation/indexing; not CuTe DSL. [BF16 source][fi-tile], [NVFP4 source][fi-tile4]. |
| prims-ts task-scheduled MoE - FlashInfer | `prims_ts_bf16_moe`, `prims_ts_fp8_per_tensor_scale_moe`, `prims_ts_fp8_block_scale_moe`, `prims_ts_fp4_block_scale_moe` | Experimental CuTe DSL task pipeline; tensor/block scaling have separate predicates, not a universal dtype switch. [BF16][fi-ts16], [FP8][fi-ts8], [FP4][fi-ts4]. |
| MonoMoE - FlashInfer | `mono_moe` | Single-kernel CUDA/C++ CuTe SM90a FP8 blockscale decode; fixed E=256, hidden=2048, intermediate-half=512, M<=8. [Source boundary][fi-mono]. |
| BGMV expert / low-rank projection - FlashInfer | `bgmv_moe_shrink`, `bgmv_moe_expand`, `bgmv_moe` | CUDA pointer-based expert projections; shrink/expand dtype instantiations grouped. SM100 `prepare_bgmv_moe` + `bgmv_moe` provide a generated full expert route, not just LoRA composition. [Projection source][fi-bgmv], [Blackwell manifest][fi-bgmv100]. |
| AlphaMoE - FlashInfer | `alphamoe_fp8_block_scale_aligned_moe` | CUDA/generated SM100 FP8 blockscale, aligned weights and gated-row interleave; distinct from BGMV/MonoMoE. [Source boundary][fi-alpha]. |
| Cake warp-decode - FlashInfer | `CakeWarpDecodeConfig` | CUDA/generated exact-shape SM100/103 NVFP4 warp-decode alternative. [Manifest][fi-cake]. |
| Llama4 min-latency experts - NVIDIA TRTLLM | `run_moe_llama4_tp8ep1_min_latency`, `llama4_fp8_fp8_gemm_swiglu_op` | Model-specific CUDA FP8 FC13/SwiGLU/FC2 pipeline, TP8/EP1 constraints; not generic TRTLLM-GEN MoE. [Pipeline][trt-llama], [gated source][trt-llama-gated]. |

## Weight-only and external expert providers

Upstream device links identify kernel bodies; project links establish origin.
Integration citations identify the callable and its reviewed dependency
contract, **not** an audit of that package's device source.
The same provider appearing in multiple inference repositories is listed once.

| Implementation family / origin | Concrete entry | Format / algorithm and pinned evidence |
| --- | --- | --- |
| Marlin - IST-DASLab lineage and adapted copies | `moe_wna16_marlin_gemm`, `fused_marlin_moe`, `marlin_nvfp4_moe_gemm` | CUDA packed low-bit expert family; standard/degree-of-batching variants and SGLang/TokenSpeed copies share the lineage. Integer/MXFP4/NVFP4 routes require distinct repacks/scales; TRTLLM NVFP4 W4A16 is SM89-99. [Original GEMM project][marlin-origin]; expert adaptations are established by [vLLM CUDA source][v-moe] and [TRTLLM source][trt-marlin], not the original project. |
| Native WNA16 expert GEMM - vLLM | `moe_wna16_gemm` | CUDA weight-only expert implementation in `moe_wna16.cu`; not automatically the Marlin algorithm. [Source][v-moe]. |
| GGML/GGUF experts - adapted CUDA | `ggml_moe_a8`, `ggml_moe_a8_vec` | Quantized expert matrix/matrix and matrix/vector consumers; same GGML origin as [dense GEMM](gemm.md). [SGLang retained source][ggml]. |
| DeepGEMM grouped experts - DeepGEMM | `m_grouped_fp8_gemm_nt_contiguous`, `m_grouped_fp8_gemm_nt_masked` | CUDA provider: contiguous throughput versus masked expert-major buffers; FP8 block scales and separate supported FP4 routes. Direct and FlashInfer integrations are not independent math. [Upstream device kernels][dg] at [vLLM's dependency pin][dg-pin], [vLLM FP4/integration contract][v-dg]. |
| OpenAI `triton_kernels` experts | `matmul` | Upstream Triton ragged grouped implementation with unquantized, FP8 QDQ and MXFP4 contexts; fused/unfused activation orchestration does not create another provider. [Upstream device source][openai-source] at [TRTLLM's retained v3.7.0 release][openai-pin], [integration contract][trt-openai]. This pin is not vLLM's different Triton dependency. |
| Humming indexed / grouped experts | `HummingMethod`, `humming_forward` | External quantized GEMM provider; indexed, grouped and batched contracts. [Upstream project][humming-origin], [vLLM indexed/grouped integration][v-humming]. Language is not established by the adapter. |
| AITER experts | `fused_moe` | ROCm HIP/ASM provider; W4A16/W4A8 MXFP4 and MXFP8 routes depend on quantization contract. FlyDSL is a distinct implementation option within AITER, not assembly. [vLLM integration/format contracts][v-aiter]. |
| HPC Ops experts | `hpc.fuse_moe`, `hpc.fuse_moe_blockwise` | External FP8/BF16, tensor or 128x128 block scaling, gated SiLU only; rejects shared experts/no-combine. Evidence establishes provider availability, not a local kernel body. [Provider boundary][sgl-hpc]. |

## Gluon staged and warp-decode algorithms

These are substantive TokenSpeed AMD kernel families, not its one-off Triton
packing/routing helpers. Group FC1/reduction/activation/FC2 stages together.

| Implementation family / origin | Concrete entry | Format / algorithm and pinned evidence |
| --- | --- | --- |
| Staged BF16/FP16 - TokenSpeed AMD | `gluon_bf16_moe`, `invoke_stage1_splitk` | Gluon gfx950 tensor-core expert FC1/FC2; split-K reduction/SwiGLU and atomic output combine are pipeline variants. [Pipeline][ts-bf], [split-K math][ts-bf-split]. |
| Warp-decode BF16/FP16 - TokenSpeed AMD | `invoke_stage1_warp_decode_gluon`, `invoke_stage2_warp_decode_gluon` | Gluon gfx950 low-row-count warp-GEMV algorithm, distinct from staged tensor-core MoE. [Source][ts-bf-warp]. |
| FP8 exact-MFMA / warp-decode - TokenSpeed AMD | `gluon_fp8_block_exact_mfma_moe`, `gluon_fp8_block_warp_decode_moe` | Two Gluon gfx950 algorithms: blockscaled tensor-core versus warp-GEMV/dequantization. [Exact-MFMA source][ts-f8], [warp source][ts-f8-warp]. |
| Staged low-bit experts - TokenSpeed AMD | `gluon_mxfp4_moe_apply`, `gluon_a16w4_situ_grouped_ep_gfx950` | Gluon gfx950 MXFP4 A16W4; SwiGLU/SiTU and grouped/warp-decode variants. MXFP8-activation A8W4 SiTU and gfx1250 packed-weight branches have separate eligibility/layouts. [Contracts][ts-mx], [SiTU math][ts-situ]. |
| Pipelined/fused low-bit experts - TokenSpeed AMD | `gluon_mxfp_ragged_matmul`, `MoEPipelinedProgram`, `gluon_mxfp_dynamic_mxfp4_fused_moe` | Gluon gfx950 ragged pipelined GEMMs and fused precomputed/dynamic route ownership. Medium-decode scheduling is a variant of this pipeline. [GEMM source][ts-mx-gemm], [fused pipeline][ts-mx-fused]. |
| FP8-activation low-bit warp decode - TokenSpeed AMD | `_gluon_mxfp4_fp8_warp_decode_moe` | Gluon gfx950 cooperative warp algorithm with SiTU/top-k FC1 and FP8/MXFP4 FC2; not merely a tile choice for staged A16W4. [Source][ts-mx-warp]. |

## Fused distributed expert compute

These entries fuse expert **math** with movement. They are not an AllToAll
transport inventory: use [EP collectives](collectives.md#expert-alltoall)
for dispatch/combine choices and topology. A matching dtype does not imply
identical distributed algorithms or communication eligibility.

All eight FlashInfer registered EP mega implementations are represented:
the DeepGEMM provider exposure and seven native CUDA/CuTe DSL routes below.
The experimental Cake EP16 executor is outside that eight-entry registry.

| Implementation family / origin | Concrete entry | Compute contract and pinned evidence |
| --- | --- | --- |
| DeepGEMM MegaMoE - DeepGEMM | `forward_mega_moe`, `fp8_mega_moe`; FlashInfer `DeepGemmMegaKernelBackend` | External fused compute/communication: SM100 mixed FP8/FP4->BF16 and separate Hopper FP8 path. Direct TRTLLM/SGLang/TokenSpeed and FlashInfer integration are one provider, not four kernels. [SM100 upstream device source][dg-mega] at [vLLM's dependency pin][dg-pin]; [Hopper integration fallback][dg-mega90] does not establish Hopper source at that pin. [FlashInfer registered exposure][fi-ep-deep]. |
| Persistent NVFP4 FC12 MegaMoE - NVIDIA TRTLLM | `Sm100MegaMoEKernel`, `Sm100SwapABSwigluFp4Fc12Kernel` | CuTe DSL fused SwiGLU FC1/FC2 with symmetric buffers and grid synchronization; `TopkReduce` is a stage. [Mega kernel][trt-mega], [FC12 source][trt-fc12]. |
| Hopper FP8 push MegaMoE - FlashInfer | `Sm90PushFp8MegaKernelBackend` | Native CUDA FP8 weights/activations->BF16 with push transport; single-node EP<=32, top-k 1/2/4/6/8. [Implementation boundary][fi-ep90-cuda]. |
| Hopper FP8 pull MegaMoE - FlashInfer | `Sm90PullFp8MegaKernelBackend` | CuTe DSL FP8/FP8->BF16 with pull protocol; distinct compute/movement algorithm, not the CUDA push pipeline. [Implementation boundary][fi-ep90-dsl]. |
| BF16 distributed experts - FlashInfer | `Bf16CutedslMegaKernelBackend` | CuTe DSL SM100 BF16 weights/activations/output with distributed buffers. [Implementation boundary][fi-ep-bf]. |
| Rank-major BF16 distributed experts - FlashInfer | `Bf16RankMajorCudaMegaKernelBackend` | Native SM100 CUDA/C++ CUTLASS with rank-major distributed buffers; not the CuTe DSL implementation. [Implementation boundary][fi-ep-rank]. |
| MXFP8 distributed experts - FlashInfer | `Mxfp8CutedslMegaKernelBackend` | CuTe DSL SM100 MXFP8 weights/activations->BF16, per-block scales. [Implementation boundary][fi-ep-mx8]. |
| NVFP4 distributed experts - FlashInfer | `Nvfp4CutedslMegaKernelBackend` | CuTe DSL SM100 packed NVFP4 weights/activations->BF16. No assertion of identity with TRTLLM FC12 merely from "MegaMoE". [Implementation boundary][fi-ep-fp4]. |
| SM120 MXFP8 distributed experts - FlashInfer | `Sm120Mxfp8CutedslMegaKernelBackend` | CuTe DSL SM120 MXFP8/MXFP8->BF16 with separate architecture-specific kernel source/staging. [Implementation boundary][fi-ep120]. |
| Cake MXFP8 EP16 MegaMoE - FlashInfer | `preprocess_cake_mxfp8_megamoe_ep16_weights` and EP16 executor | Experimental generated CUDA fixed-EP16 compute family with prepared weights; not a general transport. [Source boundary][fi-ep-cake]. |

Excluded: framework import wrappers, reference/emulated experts, router
top-k and token packing inventories, local one-off Triton experts/helpers,
and LoRA delta compositions that merely invoke an already listed GEMM.
This does not exclude the substantive upstream `triton_kernels` provider.

[trt-cutlass]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm
[trt-gen]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe
[trt-gen-ops]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py
[trt-group]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py
[trt-rubin]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/rubin/moe/rubin_contiguous_gather_grouped_gemm_swiglu_fusion.py
[trt-dense1]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc1.py
[trt-dense2]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/moe_as_dense_gemm/fc2.py
[trt-llama]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4MinLatencyMoEOp.cu
[trt-llama-gated]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Fp8GemmSwiGLU.cu
[trt-marlin]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/marlin
[trt-openai]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/moe/fused_moe/fused_moe_triton.py
[trt-mega]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/megamoe_kernel.py
[trt-fc12]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/kernel_fc12.py
[fi-jit]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/fused_moe.py
[fi-api]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/api.py
[fi-core]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/core.py
[fi-hopper]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/sm90_fused_moe.py
[fi-cute]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/fused_moe.py
[fi-port]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell/__init__.py
[fi-rubin]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/rubin
[fi-w4a16]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16.py
[fi-b12-static]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py
[fi-b12-dynamic]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py
[fi-b12-port]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/__init__.py
[fi-b12-micro]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py
[fi-b12-direct]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_direct_micro_kernel.py
[fi-b12-w4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py
[fi-tile]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cutile/moe.py
[fi-tile4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/cutile/fp4.py
[fi-ts16]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/backends/prims_ts/bf16_op.py
[fi-ts8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/backends/prims_ts/fp8_op.py
[fi-ts4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/backends/prims_ts/fp4_op.py
[fi-mono]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/monomoe.py
[fi-bgmv]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/bgmv_moe.py
[fi-bgmv100]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/blackwell_bgmv_moe.py
[fi-alpha]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/fused_moe/alphamoe_sm100.py
[fi-cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_fused_moe_warp_decode.py
[fi-ep90-cuda]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_fp8_bf16_push_cuda/backend.py
[fi-ep90-dsl]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm90/fp8_fp8_bf16_pull_cutedsl/backend.py
[fi-ep-bf]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_cutedsl/backend.py
[fi-ep-rank]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm100/bf16_bf16_bf16_rank_major_cuda/backend.py
[fi-ep-deep]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm100/fp8_fp4_bf16_deepgemm/backend.py
[fi-ep-mx8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm100/mxfp8_mxfp8_bf16_cutedsl/backend.py
[fi-ep-fp4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/backend.py
[fi-ep120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/backend.py
[fi-ep-cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/moe_ep/cake_mxfp8_megamoe_ep16.py
[v-moe]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/moe
[v-cutlass]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py
[v-dg]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py
[v-humming]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py
[v-aiter]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/fused_moe/experts
[sgl-moe]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/python/sgl_kernel/moe.py
[sgl-w4]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/python/sgl_kernel/cutlass_moe.py
[sgl-es]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/python/sgl_kernel/expert_specialization.py
[sgl-es-src]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/jit/csrc/moe/expert_specialization
[ggml]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/csrc/quantization/gguf
[dg]: https://github.com/deepseek-ai/DeepGEMM/tree/8b1392b978f5a03c828dd1711090d7fb50958b8a/deep_gemm/include/deep_gemm/impls
[dg-pin]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/tools/install_deepgemm.sh
[openai-source]: https://github.com/triton-lang/triton/blob/5f3f125e8f63c24613f1f73b937442864f263f94/python/triton_kernels/triton_kernels/matmul_details/_matmul.py
[openai-pin]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/triton_kernels/README.md
[marlin-origin]: https://github.com/IST-DASLab/marlin
[humming-origin]: https://github.com/vllm-project/humming
[sgl-hpc]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/moe/moe_runner/hpc_ops.py
[dg-mega]: https://github.com/deepseek-ai/DeepGEMM/blob/8b1392b978f5a03c828dd1711090d7fb50958b8a/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh
[dg-mega90]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/moe/mega_moe_sm90.py
[ts-bf]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/bf16.py
[ts-bf-split]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/fp16/stage1_splitk_kernel.py
[ts-bf-warp]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/fp16/warp_decode_gluon_kernel.py
[ts-f8]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/fp8/exact_mfma.py
[ts-f8-warp]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/fp8/warp_decode.py
[ts-mx]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/gluon/mxfp4.py
[ts-situ]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/situ_grouped.py
[ts-mx-gemm]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/gemm_api.py
[ts-mx-fused]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/moe.py
[ts-mx-warp]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/moe/mxfp4/fused/warp_decode.py
