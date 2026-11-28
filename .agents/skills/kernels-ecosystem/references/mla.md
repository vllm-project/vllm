# Dense MLA implementations

[Kernel index](../SKILL.md) | [GQA](gqa.md) |
[Sparse MLA and indexers](sparse-attention.md)

MLA consumes latent KV and positional components, not an ordinary GQA cache.
Prefill may use an expanded representation; decode usually uses absorbed
latent KV. A prefill implementation does not establish decode support.
Origin and source location are separate below: an imported serving backend
does not create an implementation.

## Prefill and planned latent attention

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| Native FA2-style MLA: `get_batch_mla_module` with `fa2` specialization | CUDA | FlashInfer; [native plan/launch contract][fi-fa2] | Planned paged latent + positional attention; separate query/KV indptr and split-width contracts. Not an import of upstream FlashAttention. |
| Hopper FA3-style MLA: `get_batch_mla_module` with `fa3` specialization | CUDA/C++ CuTe | FlashInfer; [Hopper plan/launch contract][fi-fa3] | Separate Hopper kernel and capability checks; do not inherit dense GQA FA3 shapes. |
| CUTLASS planned MLA: `gen_mla_module` | CUDA/C++ CUTLASS | NVIDIA CUTLASS-based FlashInfer implementation; [source-generation entry and contract][fi-cutlass] | Latent/positional dimension and dtype capability table; compact-stride restrictions belong to this path. |
| TokenSpeed packaged prefill: `tokenspeed_mla_prefill`, `BlackwellFusedMultiHeadAttentionForward` | CuTe DSL | NVIDIA-derived FMHA with TokenSpeed adaptations; [kernel][token-fmha], [entry][token-prefill] | Blackwell FMHA prefill; `tokenspeed-mla` contains actual kernel source, not just the engine registration. |
| FlashAttention-4 MLA: `FlashAttentionMLAForwardSm100` | CuTe DSL | FlashAttention; [upstream MLA kernel][fa4-mla] | Dedicated SM100 MLA forward implementation; not FlashMLA and not the generic FA4 GQA kernel. Uses the upstream sync target recorded in vLLM's retained FA4 dependency, not an asserted SGLang vendoring base. |
| TRTLLM-GEN MLA: `TllmGenFmhaRunner` | Generated CUDA kernels/artifacts | NVIDIA TensorRT-LLM; [generated FMHA family][gen], [dispatcher][dispatcher]; FlashInfer [MLA exposure][fi-mla] | Context and dense decode have different tactics/cache contracts. FI exposes `trtllm_prefill_with_kv_cache_mla` and `trtllm_batch_decode_with_kv_cache_mla`, with latent rank 512; these are not two additional owners. |

Expanded-cache context attention uses the [GQA/FMHA candidates](gqa.md).
Native RoPE, cache reconstruction and chunked-prefill output transforms are
preparation stages, not additional MLA attention algorithms.

## Absorbed-cache decode

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| FlashMLA split-KV: `run_flash_splitkv_mla_kernel`, adapted `run_mha_fwd_splitkv_mla` | CUDA/C++ | DeepSeek FlashMLA; [upstream dense kernel][flashmla], [vLLM fork source][vllm-flashmla], [TensorRT-LLM FP8 adaptation][trt-flashmla] | Hopper dense MLA: upstream FP16/BF16; the retained TRTLLM adaptation also instantiates FP8. Preserve each implementation's split-metadata/cache contract. Framework imports of `flash_mla_with_kvcache` add no candidates. |
| Legacy SM80 MLA: `gen_batch_decode_mla_module` | CUDA/C++ CuTe | FlashInfer; [SM80 kernel][fi-sm80] | SM80-oriented absorbed-cache decode; not Python CuTe DSL. |
| SM100 CUTLASS MLA: `sm100_cutlass_mla_decode` | CUDA/C++ CUTLASS | NVIDIA/SGLang-derived; [vLLM adaptation source][vllm-cutlass] | Compute-capability major 10; separate packaged implementation from FI's planned MLA. Shared CUTLASS language alone does not establish identical algorithms. |
| Modular MLA FP16/BF16: `BlackwellMultiLatentAttentionForward` | CuTe DSL | FlashInfer; [modular kernel][fi-modular], [dispatch contract][fi-selection] | Blackwell decode; supports sinks, rejects compact-variable-Q/DCP-only modes. |
| Modular MLA FP8: `BlackwellMultiLatentAttentionForwardFP8` | CuTe DSL | FlashInfer; [FP8 kernel][fi-modular8] | Separate FP8 cache/scaling pipeline, not just another launch tile for FP16 math. |
| Monolithic MLA FP16/BF16: `BlackwellMultiHeadLatentAttentionForwardFP16`, `cute_dsl_mla_decode_fp16_blackwell` | CuTe DSL | NVIDIA family; [TensorRT-LLM source][trt-mono16], [FlashInfer adaptation][fi-mono16], [FI selection][fi-selection] | TRTLLM: SM100/103, latent 512 + RoPE64, <=128 heads, generation-only. FI monolithic adaptation supports compact-variable-Q/DCP but rejects sinks. |
| Monolithic MLA FP8: `BlackwellMultiHeadLatentAttentionForwardFP8`, `cute_dsl_mla_decode_fp8_blackwell` | CuTe DSL | NVIDIA family; [TensorRT-LLM FP8 source][trt-mono8], [FlashInfer adaptation][fi-mono8] | Separate FP8 loads, MMA and scaling; preserve each integration's cache/scale ABI rather than assuming interchangeability. |
| TokenSpeed packaged MLA: `tokenspeed_mla_decode`, `BlackwellMultiHeadLatentAttentionForwardFP16`, `BlackwellMultiHeadLatentAttentionForwardFP8` | CuTe DSL | NVIDIA-derived, materially adapted TokenSpeed package; [FP16/BF16][token16], [FP8][token8], [entry][token-decode] | Blackwell, latent 512 + RoPE64, query `[B,Q,H,576]`, paged KV. FP16/BF16 and FP8 math remain separate variants within this packaged alternative; serving imports add no candidates. |
| Task-scheduled MLA: `MlaDecodeTs`, `ThroughputLatencyMlaDecodeTs`, `prims_ts_batch_mla_decode_with_kv_cache` | CuTe DSL, prims-ts | FlashInfer; [FI entry][ts-mla], same kernel family in TensorRT-LLM [2CTA][ts-2cta] / [1CTA][ts-1cta] | Two-CTA throughput and one-CTA throughput/latency are grouped schedule variants. SM100/103, BF16/E4M3 -> BF16, paged 512+64 latent attention with explicit workspace. |
| cuTile MLA: `decode_mla_kv_paged_cutile` | cuTile Python | FlashInfer; [decode math][cutile], [MLA contract][cutile-contract] | Planned paged latent attention; separate cuTile capabilities and query/cache packing, not the CUDA/C++ MLA path. |
| XQA MLA: `jit::supportConfigMLA`, `xqa_batch_decode_with_kv_cache_mla` | CUDA | NVIDIA TensorRT-LLM; [native SM120 source][trt-xqa], FlashInfer [vendored exposure][fi-xqa] | Absorbed latent/rotary cache. FI SM120/121 path requires paired BF16/BF16 or FP8/FP8 query/cache; do not apply MHA XQA constraints blindly. |

The NVIDIA-derived monolithic implementations share a recognizable family,
but adaptation-specific query layouts, DCP, sinks and scaling still matter.
TokenSpeed's packaged decode is retained as a substantial adaptation, not
counted again for its SGLang/vLLM imports. Modular MLA and task-scheduled MLA
are separate algorithm designs, even though all use CuTe DSL.

## AMD prefill, decode and fused projected-value alternatives

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| MLA prefill: `gluon_mla_prefill_gfx950`, `gluon_mla_prefill_gfx1250` | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-mla], [gfx1250 kernels][amd-mla1250] | Variable-length prefill with separate gfx950/gfx1250 kernels. |
| BF16 MLA decode: `gluon_mla_decode_bf16xbf16_gfx950_*` | Gluon | TokenSpeed AMD package; [decode family][amd-mla] | gfx950, page 64; small-head, multiblock and 64/128-head tile variants are grouped. |
| Mixed/FP8 MLA decode: `gluon_mla_decode_bf16xfp8_gfx950_bh16bn128`, `gluon_mla_decode_fp8xfp8_gfx950_bh16bn128` | Gluon | TokenSpeed AMD package; [quantized decode family][amd-mla] | gfx950, page 64, 1-16 query heads; query dtype distinguishes mixed-input and FP8-input pipelines. |
| MLA cached decode/extend: `gluon_mla_decode_gfx1250`, `gluon_mla_extend_gfx1250` | Gluon | TokenSpeed AMD package; [gfx1250 kernels][amd-mla1250] | Page 64 and 1-128 query heads; not an alias of gfx950's decode tiles. |
| Fused projected-value decode: `gluon_mla_decode_projected_value_gfx950`, `gluon_mla_decode_projected_value_gfx1250` | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-mla], [gfx1250 kernels][amd-mla1250] | Fuses latent reduction with output projection; page 64 and query heads 12/16. Standalone projection helpers are not separate attention consumers. |

Excluded: framework-local one-off Triton MLA fallbacks; backend classes;
optional binary loaders without a retained implementation artifact; KV packing,
normalization/projection-only helpers; and model-specific cache glue.
Sparse top-k consumers and indexers are in [sparse attention](sparse-attention.md).

FlashMLA's upstream citation is a retained ancestor of vLLM's dependency;
the vLLM fork citation uses its current CMake pin. Neither establishes that
TensorRT-LLM's FP8 changes exist in the original DeepSeek kernel.

[fi-fa2]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_batch_mla/_backends/fa2_backend.py
[fi-fa3]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_batch_mla/_backends/fa3_backend.py
[fi-cutlass]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_batch_mla/_backends/cutlass_backend.py
[token-fmha]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-mla/python/tokenspeed_mla/fmha.py
[token-prefill]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-mla/python/tokenspeed_mla/mla_prefill.py
[fa4-mla]: https://github.com/Dao-AILab/flash-attention/blob/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/flash_attn/cute/flash_fwd_mla_sm100.py
[gen]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha
[dispatcher]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
[fi-mla]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_core.py
[flashmla]: https://github.com/deepseek-ai/FlashMLA/blob/42f3c5789db65b5ff1eadea0fe4ce3805483a8e8/csrc/sm90/decode/dense/splitkv_mla.cu
[vllm-flashmla]: https://github.com/vllm-project/FlashMLA/tree/6bc49418c5ead572ff0339191ddf3b155749e183/csrc/sm90/decode/dense
[trt-flashmla]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/flashMLA
[fi-sm80]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/decode_mla_cute_sm80.cuh
[vllm-cutlass]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu
[fi-modular]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/mla_decode.py
[fi-modular8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/mla_decode_fp8.py
[fi-selection]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/mla_dispatch.py
[trt-mono16]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla/mla_decode_fp16.py
[fi-mono16]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py
[trt-mono8]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla/mla_decode_fp8.py
[fi-mono8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/monolithic/mla_decode_fp8.py
[token16]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp16.py
[token8]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-mla/python/tokenspeed_mla/mla_decode_fp8.py
[token-decode]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-mla/python/tokenspeed_mla/mla_decode.py
[ts-mla]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/prims_ts/mla_decode.py
[ts-2cta]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_2cta/kernel.py
[ts-1cta]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/attention/backends/prims_ts/kernels/mla_decode/throughput_latency_1cta/kernel.py
[cutile]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/kernels/cutile/fmha_decode_bsr_cutile.py
[cutile-contract]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_batch_mla/_backends/cutile_backend.py
[trt-xqa]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/kernels/xqa/mla_sm120.cu
[fi-xqa]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/xqa/mla_sm120.cu
[amd-mla]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/mla
[amd-mla1250]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/mla
