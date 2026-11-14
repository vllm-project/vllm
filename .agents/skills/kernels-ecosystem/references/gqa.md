# GQA, MQA and MHA implementations

[Kernel index](../SKILL.md) | [Dense MLA](mla.md) |
[Sparse attention and indexers](sparse-attention.md)

Compare attention math, not serving-framework registrations. GQA shares KV
heads; MQA is its single-KV-head case. Compatibility still depends on head
ratio, dimensions, dtype, mask and cache layout. Source locations below are
pinned snapshots unless explicitly marked unpinned. Prefer the kernel's upstream
source, then vLLM's copy, then other framework adaptations; a hosting framework
does not establish authorship.
CuTe DSL means Python DSL; CUDA/C++ CuTe is a different implementation language.

## Prefill and cached context

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| FlashAttention-4: `FlashAttentionForwardSm80`, `FlashAttentionForwardSm90`, `FlashAttentionForwardSm100` | CuTe DSL | FlashAttention; [upstream kernels][fa4] | Forward/prefill and cached attention; architecture-specific scheduling. Relative-bias adaptations are cited separately below, not attributed to this upstream snapshot. |
| FlashAttention-4 SM120 adaptation: `FlashAttentionForwardSm120`, `FlashAttentionForwardSm120DecodeTranspose` | CuTe DSL | FlashAttention-derived; [SGLang SM120 sources][fa120] | Separate SM120 prefill/decode scheduling and paged-KV support; do not inherit SM100 contracts. |
| Native FA2-style attention: `single_prefill_with_kv_cache` | CUDA | FlashInfer; [prefill kernels][fi-prefill] | Prefill and tensor-core decode; single, paged and ragged plans with distinct masking contracts. This is not a framework importing upstream FlashAttention. |
| Hopper FA3-style attention: `gen_single_prefill_module`, `gen_batch_prefill_module` FA3 specializations | CUDA/C++ CuTe | FlashInfer; [FA3 source-generation manifest][fi-jit] | SM90 prefill/tensor-core decode; BF16/FP16 and separate FP8 source path. |
| CUTLASS Blackwell FMHA: `gen_fmha_cutlass_sm100a_module` | CUDA/C++ CUTLASS | CUTLASS-based FlashInfer implementation; [source manifest][fi-jit] | SM100 prefill with planner and compiled FMHA sources; not Python CuTe DSL. |
| FMHA-v2: `FusedMHARunnerV2` | CUDA | NVIDIA TensorRT-LLM; [native source][fmha-source], [runner][fmha-v2]; FlashInfer [generated exposure][fi-jit] | Packed/padded context QKV. FlashInfer's `gen_fmha_v2_module` packages SM120 BF16/E4M3 specializations of this family, not a separate owner. |
| Modular Blackwell FMHA: `BlackwellFusedMultiHeadAttentionForward` | CuTe DSL | FlashInfer; [modular prefill][fi-cute-prefill] | Separate paged/ragged prefill adapters. |
| Primitives FMHA: `cute_dsl_fmha_ragged_prefill` | CuTe DSL | NVIDIA/FlashInfer; [source/export entry][fi-fmha] | Blackwell ragged prefill, FP16/BF16/E4M3; D=192 is FP8-only. Export loading may use a compiled artifact. |
| Blockscaled primitives FMHA: `cute_dsl_fmha_blockscaled_prefill` | CuTe DSL | NVIDIA/FlashInfer; [blockscaled entry][fi-blockscaled] | Explicit scale tensors and packed blockscaled inputs; not ordinary FP8 FMHA. |
| SM120 FP8 primitives FMHA: `sm120_fmha_fp8_ragged_prefill`, `sm120_fmha_fp8_paged_prefill` | CuTe DSL | FlashInfer; [SM120 implementation][fi-fmha120] | FP8 TMA pipeline with separate paged/ragged contracts. |
| BSR cuTile prefill: `prefill_attention_kv_paged_cutile`, `prefill_attention_kv_ragged_cutile` | cuTile Python | FlashInfer; [kernel source][cutile-prefill] | Ragged adapter requires equal Q/KV prefix sums; paged mode uses BSR metadata. |
| Task-scheduled context: `FmhaTs`, `batch_prefill`, `batch_prefill_with_paged_kv_cache` | CuTe DSL, prims-ts | FlashInfer family; [FlashInfer entry][ts-context], [TensorRT-LLM source adaptation][trt-ts-context] | Ragged/paged context plan geometries; TRTLLM integration targets SM100/103 HND paged cache. One family, not one candidate per host framework. |
| Visual FMHA: `cute_dsl_fmha_fwd`, `BlackwellFusedMultiHeadAttentionForward` | CuTe DSL | NVIDIA; [TensorRT-LLM visual source][visual-fmha] | SM100/103/107 visual sequence layout, not paged LLM attention. Shared class spelling alone does not establish identity with modular prefill above. |
| Visual blockscaled FMHA: `BlackwellFusedMultiHeadBlockScaledAttentionForward` | CuTe DSL | NVIDIA; [TensorRT-LLM visual source][visual-blockscaled] | Separate low-bit/blockscale QK/PV pipeline and visual layout. |
| NVFP4 attention: `nvfp4_attention_sm120_fwd` | CUDA | FlashInfer; [SM120 implementation entry][nvfp4] | Dense forward with packed NVFP4 QKV/scales; `nvfp4_attention_sm120_quantize_qkv` is its preparation stage, not another consumer. |

## Decode and mixed-phase scheduling

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| CUDA-core decode: `single_decode_with_kv_cache` | CUDA | FlashInfer; [decode kernels][fi-decode] | Single-request and batched paged decode, NHD/HND caches; not tensor-core prefill reused for decode. |
| Masked multihead attention: `masked_multihead_attention`, `mmha_supported` | CUDA | TensorRT-LLM; [MMHA source][mmha] | Generation with paged-cache, beam and sequence metadata. |
| XQA HMMA/QGMMA: `DecoderXQARunner`, `supportConfigHMMA`, `supportConfigQGMMA` | CUDA, NVRTC/JIT | NVIDIA TensorRT-LLM; [native source][xqa], [eligibility][xqa-runner]; FlashInfer [vendored exposure][fi-xqa] | Grouped decode; head ratio, cache dtype and generation length have separate guards. FlashInfer's `xqa_batch_decode_with_kv_cache` exposes this family; its vendor-auto route uses XQA on SM90/120/121. |
| TRTLLM-GEN FMHA: `TllmGenFmhaRunner`, `FmhaDispatcher::run` | Generated CUDA kernels/artifacts | NVIDIA TensorRT-LLM; [native generated family][gen], [dispatcher][dispatcher]; FlashInfer [exposure manifest][fi-jit] | Distinct context/decode tactics. FlashInfer's `trtllm_batch_context_with_kv_cache` and `trtllm_batch_decode_with_kv_cache` expose this owner; SM100/103 auto-decode routing does not imply all native tactics are exposed. |
| Cascade split/reduction attention | CUDA/C++ tensor-core kernels | TensorRT-LLM; [cascade source][cascade] | Multi-block generation; split attention and reduction form one algorithm, not context FMHA. |
| Modular GQA decode: `GroupedQueryAttentionDecode`, `GroupedQueryAttentionDecodePaged` | CuTe DSL | FlashInfer; [contiguous][cute-gqa], [paged][cute-paged] | Blackwell; distinct contiguous/paged variants. Paged wrapper requires equal QK/VO dimensions and does not implement RoPE/ALiBi/soft-cap. |
| Task-scheduled SwapsMmaAb decode: `FmhaDecodeTs`, `fmha_decode_launch` | CuTe DSL, prims-ts | FlashInfer; [entry][ts-decode], [same kernel family in TensorRT-LLM][trt-ts-decode] | SM100a/103a HND paged cache; FI head dims 64/128/256 and pages 16/32/64/128. No FI wrapper-managed graphs; TRTLLM adapter is single-token. |
| BSR cuTile decode: `_decode_attention_kv_paged_kernel` | cuTile Python | FlashInfer; [kernel source][cutile-decode] | Paged KV and BSR index metadata; split-K reduction is part of the implementation. |
| Cake FMHA: `cake_batch_context_with_kv_cache`, `cake_batch_decode_with_kv_cache` | CUDA, generated specializations | FlashInfer; [source/route manifest][cake] | Explicit Cake context/decode choice, not the vendor `auto` route. Architecture and shape select generated implementations. |
| Cake request-ordered decode: `plan_cake_fmha_request_ordered_paged_decode` | CUDA | FlashInfer; [request-ordered source][cake-request] | Preplanned request-ordered paged-cache access; separate generated decode design. |
| Experimental SM110 decode: `sm110_gqa_decode` | CUDA | FlashInfer; [experimental source][sm110] | SM110-specific experimental path, not a generic CUDA candidate. |
| POD fused prefill/decode: `get_pod_module` | CUDA | FlashInfer; [POD implementation entry][pod] | Co-launches prefill and decode; not two independently scheduled API calls. |
| Holistic batch attention: `get_holistic_attention_module` | CUDA | FlashInfer; [implementation entry][holistic] | Per-request prefill/decode choice in one paged kernel; NHD/HND with query/KV indptr. |
| Draft/block attention: `DSparkAttentionKernel` | CuTe DSL | TensorRT-LLM; [DSpark math][dspark] | Blackwell draft/block mask and cache contract; not unrestricted paged GQA. |

## Relative attention and AMD-native alternatives

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| Relative-bias prefill/extend: `rel_mha_varlen_tsmha` | CuTe DSL | FlashAttention-4-derived; [TokenSpeed adaptation][relative-extend] | SM100-oriented varlen bias attention. Bias evaluation and split-output combine belong to this family. |
| Relative split decode: `rel_mha_decode_tsmha`, `rel_mha_decode_tsmha_v2` | CuTe DSL | FA4-derived v1 and TokenSpeed v2 adaptation; [v1][relative-decode], [v2][relative-decode2] | Paged KV and partial-output combine; v2 has dedicated buffers/prediction-width specialization, not just another tile. |
| Relative MXFP8 cached attention: `fa4_rel_mha_decode_with_kvcache_mxfp8`, `fa4_rel_mha_extend_with_kvcache_mxfp8` | CuTe DSL | FA4-derived TokenSpeed relative family; [quantized registrations][relative-reg] | Head width 128 and blockscaled cache arguments; distinct quantized route. |
| AMD MHA prefill/extend/decode: `gluon_mha_prefill_gfx950`, `gluon_mha_extend_gfx950`, `gluon_mha_decode_gfx950`; gfx1250 prefill/decode variants | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-mha], [gfx1250 kernels][amd-mha1250] | Dedicated gfx950/gfx1250 families; no separate gfx1250 extend kernel asserted. |
| AMD relative attention: `gluon_rel_mha_prefill_gfx950`, `gluon_rel_mha_extend_gfx950`, `gluon_rel_mha_decode_gfx950` | Gluon | TokenSpeed AMD package; [relative implementations][amd-relative] | gfx950 relative-bias math, not generic MHA aliases. |
| AMD Wave attention: `prefill_attention_wave`, `extend_attention_wave`, `decode_attention_wave` | Wave DSL | Wave; [upstream attention templates][wave], [SGLang integration only][wave-integration] | Separate AMD context, prefix-extend and decode templates; SGLang imports their kernel factories. Triton fallback is not Wave compute. |
| Verification attention/reduction: `vattn3_core.s`, `vred.s` | AMD assembly | SGLang; [assembly source][vattn] | Optional gfx950-only verification path. |

Framework imports of FA2/FA3/FA4, FlashInfer, AITER, cuDNN or PyTorch do not add
candidates here. The source-backed FA4 adaptation is retained; serving backend
classes, per-framework local Triton attention fallbacks, cache plumbing and
distributed compositions are omitted. Sparse masks with dedicated consumers
belong in [sparse attention](sparse-attention.md), not a dense-backend bucket.

The FA4 upstream revision is the sync target recorded by vLLM's retained
`flash-attention` dependency commit `506341a143fcabd4bb79052a7605ada727d6b3f5`;
it is not asserted to be SGLang's exact vendoring base. Wave uses the retained
SGLang dependency's `wave-lang==3.8.2` tag, resolved to its source commit.

[fa4]: https://github.com/Dao-AILab/flash-attention/tree/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute
[fa120]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/fa4_sm120
[fi-prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/prefill.cuh
[fi-jit]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/attention/modules.py
[fmha-source]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/kernels/fmha_v2/src
[fmha-v2]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp
[fi-cute-prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/prefill.py
[fi-fmha]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/cute_dsl/fmha.py
[fi-blockscaled]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/cute_dsl/fmha_blockscaled.py
[fi-fmha120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/cute_dsl/sm120_fmha.py
[cutile-prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/kernels/cutile/fmha_prefill_bsr_cutile.py
[ts-context]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/prims_ts/context.py
[trt-ts-context]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_context/fmha_kernel.py
[visual-fmha]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/fmha.py
[visual-blockscaled]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/fmha_blockscaled.py
[nvfp4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/nvfp4_attention_sm120.py
[fi-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/decode.cuh
[mmha]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderMaskedMultiheadAttentionTemplate.h
[xqa]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/kernels/xqa
[xqa-runner]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.cpp
[fi-xqa]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/xqa/mha.cu
[gen]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha
[dispatcher]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
[cascade]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cascadeAttentionKernel.cu
[cute-gqa]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/gqa_decode.py
[cute-paged]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/gqa_decode_paged.py
[ts-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/prims_ts/decode.py
[trt-ts-decode]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/attention/backends/prims_ts/kernels/fmha_decode/fmha_decode_kernel.py
[cutile-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/kernels/cutile/fmha_decode_bsr_cutile.py
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cake_fmha.py
[cake-request]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/cake_fmha/request_ordered_paged_decode
[sm110]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/experimental/sm110_gqa_decode
[pod]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/pod.py
[holistic]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/_core.py
[dspark]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/dspark
[relative-extend]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/rmha/_cute_dsl/rel_extend.py
[relative-decode]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/rmha/_cute_dsl/rel_decode.py
[relative-decode2]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/rmha/_cute_dsl/rel_decode_v2.py
[relative-reg]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/rmha/cuda.py
[amd-mha]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/mha
[amd-mha1250]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/mha
[amd-relative]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/rmha
[wave]: https://github.com/iree-org/wave/tree/1cdf9e30be9961b48e846441d942811de98b8971/wave_lang/kernel/wave/templates
[wave-integration]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/attention/wave_ops
[vattn]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/vattn_asm_gfx950
