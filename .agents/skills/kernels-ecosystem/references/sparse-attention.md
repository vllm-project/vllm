# Sparse attention, indexers and top-k

[Kernel index](../SKILL.md) | [Dense GQA](gqa.md) | [Dense MLA](mla.md)

Choose the operation before choosing the implementation:
**score/index** produces logits; **select** produces token/block IDs;
**consume** reduces selected KV into attention output. A score kernel is not a
sparse MLA implementation. Cache compression, RoPE, packing and index-namespace
conversion alone are not candidates here.

Tables prefer actual upstream kernel source, then vLLM source or vendored
copies, then other framework adaptations. A dependency-boundary citation
establishes the named API/contract, not a review of every dependency kernel or
architecture. Unpinned citations are explicitly labeled.

## Sparse MLA consumers

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| FlashMLA sparse: `flash_mla_sparse_fwd`, sparse `flash_mla_with_kvcache` | CUDA/C++ | DeepSeek FlashMLA; [upstream kernel sources][flashmla-dsa], [vLLM fork API][flashmla-api], [vLLM V4 integration only][flashmla-v4] | Sparse forward/prefill and paged decode. The upstream citation is a retained dependency ancestor, not evidence for the fork's V4 extensions. V4 compressed C4/C128 + SWA metadata is not the V3.2 cache ABI; framework-specific imports add no candidates. |
| Q8KV8 sparse prefill: `sparse_mla_q8kv8_prefill_sm90` | CUDA/C++ CUTLASS/CuTe | SGLang; [native JIT source entry][q8-prefill] | Hopper FP8 Q/KV consumer with explicit query preparation; not inferred to be FlashMLA just because its build flags derive from FlashMLA. |
| TRTLLM-GEN sparse MLA: `TllmGenFmhaRunner` | Generated CUDA kernels/artifacts | NVIDIA TensorRT-LLM; [generated family][gen], [native dispatcher][dispatcher]; FlashInfer [exposure][fi-mla] | FI SM100/103 exposes V3.2 via `trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=...)` and V4 via `trtllm_batch_decode_sparse_mla_dsv4`. V4 separates compressed/SWA indices and active lengths; native epilogues can include inverse RoPE/FP8. |
| SM120 sparse MLA: `sparse_mla_decode_dsv3_2_kernel`, `sparse_mla_prefill_mg_kernel` | CUDA | FlashInfer; [DSv3.2 decode math][fi-sparse32], [multi-group prefill][fi-sparse-prefill] | SM120/121 BF16 queries and packed uint8 KV; prefill multi-group scheduling and decode are variants of the sparse family. |
| SM120 V4 sparse MLA: FP8 path and `sparse_mla_decode_dsv4_nvfp4_kernel` | CUDA | FlashInfer; [FP8 contract][fi-sparse-v4], [NVFP4 math][fi-sparse4] | BF16 query; separate compressed/SWA segments. NVFP4 packing is a different cache ABI from FP8, not a dtype alias. |
| Hierarchical compressed attention: `BlackwellHeavilyCompressedAttentionForwardFP8` | CuTe DSL | FlashInfer; [HCA math][hca] | Blackwell sparse V4 consumer; CuTe route of `trtllm_batch_decode_sparse_mla_dsv4`, not the TRTLLM-GEN CUDA implementation despite sharing an API. |
| TileLang sparse MLA: `tilelang_sparse_fwd`, `sparse_mla_fwd_decode_partial`, `sparse_mla_fwd_decode_partial_fp8` | TileLang | SGLang; [sparse math][tile] | BF16 and FP8 sparse forward/decode; v1/v2 tiles and partial/combine stages are grouped, while FP8 remains a distinct compute variant. |
| TileLang V4 attention: `dpsk_v4_fp8_attention_fwd` | TileLang | SGLang; [V4 math][tile] | V4 FP8 partial/combine used by HIP FlashMLA compatibility path; do not mistake compatibility API naming for CUDA FlashMLA ownership. |
| AMD sparse MLA: `gluon_dsa_prefill_gfx950/gfx1250`, `gluon_dsa_decode_gfx950/gfx1250` | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-dsa], [gfx1250 kernels][amd-dsa1250] | Selected-cache-slot attention, with dedicated prefill/decode implementations on each architecture. |
| AMD FP8 dense-cache sparse prefill: `gluon_dsa_prefill_fp8_dense_gfx950/gfx1250` | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-dsa], [gfx1250 kernels][amd-dsa1250] | Sparse selection over FP8 dense cache; not merely FP8 index scoring. |
| AMD V4 compressed/window attention: `gluon_dsv4_prefill_gfx950`, `gluon_dsv4_decode_split_gfx950` | Gluon | TokenSpeed AMD package; [V4 family][amd-v4] | gfx950 compressed-cache + sliding-window prefill/split decode; separate V4 metadata. |

## Block-sparse, hierarchical and selected GQA consumers

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| BSA: `bsa_attn_sm100_blk64`, `bsa_attn_sm100_blk128`, `bsa_attn_sm120` families | CuTe DSL | Block-Sparse-Attention/FlashAttention-derived; [upstream source/dispatch (unpinned moving branch)][bsa-upstream], FlashInfer [block-64][bsa64], [block-128][bsa128], [SM120][bsa120] adaptations | SM100 block 64/128 and SM120 block 64; forward/split/combine variants grouped. Retained comments name the upstream file but no base revision; use pinned FI sources for its JIT-only dispatch and tuning changes. The moving link locates the source, not a re-audit of its current contents. |
| Task-scheduled block-sparse FMHA: `FmhaTs` sparse scheduling | CuTe DSL, prims-ts | FlashInfer; [sparse entry][ts-bsa], [TensorRT-LLM kernel family][trt-ts] | Inspect/prepare and block-sparse schedules over task-scheduled FMHA; TRTLLM SM100/103 block/page alignment constraints. |
| Cake variable sparse attention: `plan_cake_vsa`, `run_cake_vsa` | CUDA | FlashInfer; [Cake VSA][cake-vsa] | Explicit sparse mask plan/run; not an alias for dense FMHA. |
| Cake Sage BSA: `bsa_attn_sm120_blk64_sage_fwd` | CUDA | FlashInfer Cake implementation; [generated source manifest][cake-sage], [contract][bsa120] | SM120a, MHA D=128; INT8 BHSD Q/K, E4M3 HDS V, BF16 output; noncausal, no LSE. Distinct from CuTe BSA. |
| Video sparse attention: `block_sparse_attn_from_indices_cute` | CuTe DSL | NVIDIA TensorRT-LLM; [VSA math][vsa] | Fine top-k cube attention, D=128, 4x4x4 cubes, <=4096 cubes; coarse branch is dense. |
| QSA mixed-input attention: `MixedInputFusedMultiHeadAttentionDecode`, `cute_dsl_blackwell_qsa_sparse_attention` | CuTe DSL | TokenSpeed; [mixed-input math][qsa-kernel], [contract][qsa-entry] | Blackwell Q/K/V width 256, Q heads 6/12/24 and KV heads 1/2/4; fixed selected-width gate. |
| Compressed attention + online selection: `compressed_attention_tilelang`, `_fused_attn_pooling_online_topk` | TileLang | SGLang MiniCPM implementation; [fused math][minicpm] | Fuses compressed attention, pooling and online top-k; not just metadata preparation. |
| MiniMax sparse FMHA: `fmha_sm100`, `sparse_fmha`, `fmha_sm100_plan` | CUDA/C++ CUTLASS | MiniMax MSA; [upstream CUDA sources][msa-cuda], [entry/contract][msa-api] | SM100 sparse FMHA; tile instantiations, planning and split reduction belong to one family. Imported `fmha_sm100` in vLLM/TRTLLM is not another implementation or proof of CuTe DSL math. |
| MiniMax sparse forward/decode: `sparse_atten_func`, `sparse_decode_atten_func`, `sparse_atten_nvfp4_kv_func` | CuTe DSL | MiniMax MSA; [upstream kernel sources][msa-cute] | SM100 D=128; dense/FP8 MMA and separate packed NVFP4-cache path. Forward/decode schedules, CSR construction and combines are stages, not extra candidates. |
| SM12x MSA: `msa_sparse_attention`, `msa_sparse_decode_attention` | CuTe DSL | FlashInfer; [prefill][msa-prefill], [decode][msa-decode] | Sparse selected-cache consumers, distinct from proxy scoring and from MiniMax's SM100 package. |
| Blackwell native MSA: `blackwell_msa_sparse_attention`, `blackwell_msa_sparse_decode_attention` | CUDA | FlashInfer; [SM100 implementation entry][msa100] | SM100a/103a BF16 and FP8 route plans; do not equate this CUDA source path with a CuTe DSL consumer. |

## Logits and index scoring

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| Paged MQA FP8 logits: `FP8MQALogitsKernel`, `fp8_paged_mqa_logits` | CuTe DSL | NVIDIA TensorRT-LLM; [original family][trt-mqa8], [FlashInfer adaptation][fi-mqa8] | Blackwell FP8 paged keys and index-head scores; shared origin, not an independent score algorithm per framework port. |
| Paged MQA FP4 logits: `FP4MQALogitsKernel`, `fp4_paged_mqa_logits` | CuTe DSL | NVIDIA TensorRT-LLM; [original family][trt-mqa4], [FlashInfer adaptation][fi-mqa4] | Packed FP4 keys/scales; distinct FP4 tensor-core pipeline rather than an FP8 alias. |
| DeepGEMM MQA logits: `fp8_mqa_logits`, `fp8_paged_mqa_logits`, `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` | CUDA/C++ | DeepGEMM; [upstream FP8 kernel][deepgemm8], [mixed-format kernel][deepgemm4], [vLLM integration only][deepgemm-vllm] | Ragged/paged DSA scoring; V4 FP8/FP4 mixed-format path has separate cache/scaling contracts. Source revision is vLLM's retained DeepGEMM pin. Top-k selection is downstream, not supplied by these logits calls. |
| TileLang MQA scores: `tilelang_fp8_paged_mqa_logits`, `fp8_index` | TileLang | SGLang; [math][tile] | Paged/ragged FP8 scoring, separate from TileLang sparse value accumulation. |
| QSA compressed MQA scores: `tilelang_qsa_mqa_prefill`, `tilelang_qsa_mqa_decode` | TileLang | SGLang; [QSA scoring math][qsa-score] | Distinct prefill/decode shape contracts; not the QSA attention consumer. |
| MiniMax decode block scores: `IndexDecodeScoreKernel`, `minimax_index_decode_score` | CuTe DSL | vLLM; [original kernel][vllm-minimax-score], [TokenSpeed epilogue adaptation][token-minimax-score] | Blackwell block maxima over causally valid index-key dot products. TokenSpeed additionally applies score scaling and forced init/local-block selection in its epilogue; those changes are not attributed to vLLM. |
| MSA proxy scores: `msa_proxy_score`, `msa_proxy_score_fp4` | CuTe DSL | FlashInfer; [BF16/FP8][proxy], [FP4][proxy4] | SM12x proxy scoring; FP4 uses quantized operands. No attention output. |
| MiniMax FP4 block indexer: `fp4_indexer_block_scores` | CuTe DSL | MiniMax MSA; [upstream indexer kernel][msa-fp4] | Blackwell packed FP4 Q/K plus validated scales; scale reorder belongs to the same stage. |
| AMD DSA scores: `gluon_dsa_*_topk_standard_gfx950/gfx1250`, `gluon_dsa_*_topk_fp8_gfx950/gfx1250` | Gluon | TokenSpeed AMD package; [gfx950 kernels][amd-dsa], [gfx1250 kernels][amd-dsa1250] | Standard-cache and FP8 prefill/decode scores; distinct cache variants grouped by operation. |
| AMD V4 MXFP4 scores: `gluon_dsv4_prefill_topk_mxfp4_gfx950`, `gluon_dsv4_decode_topk_mxfp4_gfx950` | Gluon | TokenSpeed AMD package; [V4 indexer family][amd-v4] | gfx950 MXFP4 query/key scores with dedicated plan metadata. |
| AMD K-pool indexer: `gluon_kpool_prefill_topk_fp8_gfx950` | Gluon | TokenSpeed AMD package; [K-pool family][amd-kpool] | gfx950 FP8 prefill over pool-compressed keys, not ordinary token-key indexing. |
| TriAttention score/normalize/union: `_TriAttentionScoreKernel`, `_TriAttentionNormalizeUnionKernel`, `build_score_pipeline` | CuTe DSL | TensorRT-LLM; [scoring][tri-score], [selection][tri-select] | Compression scoring followed by grouped normalization/selected-token union. These stages are one selection pipeline, not FMHA consumers. |

## Top-k and block selection

| Implementation | Language | Origin and source (pinned citation) | Phase and constraints |
| --- | --- | --- | --- |
| Native radix top-k: `top_k`, `top_k_page_table_transform`, `top_k_ragged_transform` | CUDA | FlashInfer; [top-k family][fi-topk] | Dense scores to values/indices or fused paged/ragged transforms; default algorithm. Framework imports add no rows. |
| Exact cluster top-k | CUDA | FlashInfer; [cluster math][fi-cluster] | SM100 cluster selection through the same top-k APIs; `FLASHINFER_TOPK_ALGO=clusters`, including fused transforms. |
| CUB batched top-k: `DeviceBatchedTopK::MaxPairs` | CUDA/C++ CUB | NVIDIA CCCL/CUB; [upstream implementation][cub-topk], FlashInfer [native adapter][fi-cub] | `FLASHINFER_TOPK_ALGO=cub`; separate workspace and transform tails. Uses FI's retained CCCL gitlink, not the custom native radix implementation. |
| Single-pass multi-CTA radix: `SinglePassMultiCTARadixTopKKernel` | CuTe DSL | NVIDIA TensorRT-LLM; [upstream kernel][trt-radix], [FlashInfer adaptation][fi-radix] | Varlen rows split over CTAs with global coordination. Per-framework vendored runners add no algorithms. |
| Cluster-cooperative radix: `SinglePassMultiCTARadixTopKClusterKernel` | CuTe DSL | NVIDIA TensorRT-LLM; [upstream cluster kernel][trt-cluster] | Cluster/shared-memory communication rather than global-only radix; returns local column offsets, not physical cache slots. |
| Filtered radix: `run_filtered_topk_decode`, `run_filtered_topk_prefill`, `cute_dsl_radix_filter_topk_wrapper` | CuTe DSL | NVIDIA family; TensorRT-LLM [decode][trt-filter-decode] / [prefill][trt-filter-prefill], FlashInfer [adaptation][fi-filter] | Valid score ranges, per-row starts/ends and filter-first selection; prefill/decode and multi-CTA variants grouped. |
| GVR guess/verify/refine: `GvrTopKKernel`, `GvrTopKLBKernel`, `GvrRegClusKernel` | CuTe DSL | NVIDIA/FlashInfer family; FI [cluster][fi-gvr], [load-balanced][fi-gvr-lb], [GVR2 register/cluster][fi-gvr2]; [TensorRT-LLM variants][trt-gvr] | Direct/register/TP/self-sampling and load-balanced preparation/emission are variants/stages. GVR2 has distinct host-managed resources; not a new candidate per exported helper. |
| Native indexer top-k: `invokeIndexerTopKDecode` and prefill launches | CUDA | TensorRT-LLM; [native math][trt-topk] | Separate decode/prefill selection from the CuTe radix/GVR implementations. |
| Fast top-k/transform family: `fast_topk`, `fast_topk_v2`, `fast_topk_transform_fused`, `fast_topk_transform_ragged_fused` | CUDA; separate HIP source | SGLang; [AOT entry][sg-topk], [CUDA/HIP source tree][sg-topk-source], [JIT variant][sg-jit-topk] | AOT/JIT and fused page/ragged transforms grouped. HIP is separate source; V4 `deepseek_v4_topk_transform_512` selects 512. |
| V4 planned radix selection: `topk_transform_paged`, `plan_topk_v2`, `topk_transform_ragged_v2`, `topk_transform_paged_v2` | CUDA | SGLang; [V4 source entry][sg-v4] | V1 and planned V2 selection/transform algorithms; grouped here rather than listing each cache-format wrapper. |
| Persistent V4 top-k: `indexer_topk_prefill`, `persistent_topk` | CUDA | TokenSpeed; [native source entry][token-topk-v4] | Separate prefill/persistent selection; `ragged_decode_topk` exposes the same family, with int32 local indices and uint8 workspace. |
| MiniMax block selection: `sparse_topk_select`, packaged `minimax_prefill_score_topk` | CUDA | MiniMax MSA; [upstream selection kernel][msa-topk], [package API][msa-api] | Sparse block IDs; planning/reduction and framework score-dispatch wrappers are not additional attention consumers. |
| MiniMax native select-blocks: `invokeMinimaxM3SelectBlocks` | CUDA | TensorRT-LLM; [selection math][trt-blocks] | Index score to selected block IDs; distinct from the score GEMM. |
| MiniMax decode top-k: `minimax_decode_topk` | CUDA | SGLang; [kernel entry][sg-minimax-topk] | Model-specific decode block selection, not generic DSA token top-k. |
| MSA radix, chunked and count-rank selection: `msa_topk_select` | CuTe DSL | FlashInfer; [radix][msa-radix], [chunked][msa-chunk], [count-rank][msa-count] | Three explicit SM12x selection algorithms, grouped under one MSA operation rather than opaque backend tags. |
| Blackwell MSA native selection: `blackwell_msa_topk_select` | CUDA | FlashInfer; [SM100 native family][msa100] | SM100a/103a route; distinct from SM12x CuTe selection. |

Excluded: per-framework FI/FA/AITER import rows; one-off local Triton sparse
fallbacks; Q/K normalization, RoPE, quantization-only helpers; compression/cache
controllers; SWA/index conversion and speculative metadata; output-combine and
schedule exports promoted to independent kernels. Large AMD Gluon families and
TileLang consumers/indexers are retained because they supply substantial math.
No support or performance is inferred solely from a shared language or name.

MSA source links use MiniMax's initial upstream commit, an ancestor of the
`890aaa1a37a598ad17ccff0827fea21540d381fa` vLLM-fork pin recorded in TokenSpeed's
retained vendor README. That package is otherwise upstream-identical apart
from formatting and its local CUTLASS-header lookup; these links do not claim
to contain later fork compatibility fixes. No dependency pins were refreshed.

[flashmla-dsa]: https://github.com/deepseek-ai/FlashMLA/tree/42f3c5789db65b5ff1eadea0fe4ce3805483a8e8/csrc
[flashmla-api]: https://github.com/vllm-project/FlashMLA/blob/6bc49418c5ead572ff0339191ddf3b155749e183/flash_mla/flash_mla_interface.py
[flashmla-v4]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/models/deepseek_v4/nvidia/flashmla.py
[q8-prefill]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py
[gen]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha
[dispatcher]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp
[fi-mla]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_core.py
[fi-sparse32]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh
[fi-sparse-prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/sparse_mla_sm120/prefill_mg_kernel.cuh
[fi-sparse-v4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mla/_sparse_mla_sm120.py
[fi-sparse4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_nvfp4_kernel.cuh
[hca]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/attention/dsa/hca_fp8.py
[tile]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
[amd-dsa]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsa
[amd-dsa1250]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/dsa
[amd-v4]: https://github.com/lightseekorg/tokenspeed/tree/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsv4
[bsa64]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/sparse/bsa_attn_sm100_blk64.py
[bsa-upstream]: https://github.com/NVIDIA-JerryChen/Block-Sparse-Attention/blob/master/bsa_attn_interface.py
[bsa128]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/sparse/bsa_attn_sm100_blk128.py
[bsa120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py
[ts-bsa]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attention/prims_ts/block_sparse.py
[trt-ts]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/attention/backends/prims_ts/kernels
[cake-vsa]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cake_vsa.py
[cake-sage]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_sage_block_sparse_attention.py
[vsa]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/video_sparse_attention
[qsa-kernel]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/qsa_sparse.py
[qsa-entry]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/qsa/cute_dsl.py
[minicpm]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/attention/minicpm/fuse_kernel.py
[msa-cuda]: https://github.com/MiniMax-AI/MSA/tree/9175bebf2b623ca0ea8ad5247bb615cd424598a0/python/fmha_sm100/csrc
[msa-api]: https://github.com/MiniMax-AI/MSA/blob/9175bebf2b623ca0ea8ad5247bb615cd424598a0/python/fmha_sm100/api.py
[msa-cute]: https://github.com/MiniMax-AI/MSA/tree/9175bebf2b623ca0ea8ad5247bb615cd424598a0/python/fmha_sm100/cute/src/sm100
[msa-prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/sparse_prefill_sm12x.py
[msa-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/sparse_decode_sm12x.py
[msa100]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/_blackwell_sm100.py
[trt-mqa8]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/fp8_paged_mqa_logits.py
[fi-mqa8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attn_scores/kernels/fp8_paged_mqa_logits.py
[trt-mqa4]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/paged_mqa_logits/fp4_paged_mqa_logits.py
[fi-mqa4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/attn_scores/kernels/fp4_paged_mqa_logits.py
[deepgemm8]: https://github.com/deepseek-ai/DeepGEMM/blob/8b1392b978f5a03c828dd1711090d7fb50958b8a/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh
[deepgemm4]: https://github.com/deepseek-ai/DeepGEMM/blob/8b1392b978f5a03c828dd1711090d7fb50958b8a/deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh
[deepgemm-vllm]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/utils/deep_gemm.py
[qsa-score]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/srt/layers/attention/qsa/mqa.py
[vllm-minimax-score]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/models/minimax_m3/nvidia/ops/index_decode_score.py
[token-minimax-score]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/msa/_cute_dsl/decode_score.py
[proxy]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/proxy_score_sm12x.py
[proxy4]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/proxy_score_fp4_sm12x.py
[msa-fp4]: https://github.com/MiniMax-AI/MSA/blob/9175bebf2b623ca0ea8ad5247bb615cd424598a0/python/fmha_sm100/cute/src/sm100/fp4_indexer.py
[amd-kpool]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/dsa/sparse_mla.py
[tri-score]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/kv_cache_compression/triattention/triattention_cute_score_fused.py
[tri-select]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/kv_cache_compression/triattention/triattention_cute_selection.py
[fi-topk]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk.py
[fi-cluster]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/fast_topk_clusters_exact.cuh
[fi-cub]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/cub_topk.cu
[cub-topk]: https://github.com/NVIDIA/cccl/blob/16bd510c9b712e82b0ab6cbb630d8e29ba1f7116/cub/cub/device/device_batched_topk.cuh
[trt-radix]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/single_pass_multi_cta_radix_topk.py
[fi-radix]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk_varlen/kernels/radix_topk.py
[trt-cluster]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/single_pass_multi_cta_radix_topk_cluster.py
[trt-filter-decode]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/filtered_top_k_decode_varlen.py
[trt-filter-prefill]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/filtered_top_k_prefill_varlen.py
[fi-filter]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk_varlen/kernels/filtered_topk_decode.py
[fi-gvr]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk_varlen/kernels/gvr_topk_decode.py
[fi-gvr-lb]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk_varlen/kernels/gvr_topk_decode_lb.py
[fi-gvr2]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/topk_varlen/kernels/gvr2_topk_decode.py
[trt-gvr]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py
[trt-topk]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/indexerTopK.cu
[sg-topk]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/python/sgl_kernel/top_k.py
[sg-topk-source]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/csrc/elementwise
[sg-jit-topk]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/elementwise/fast_topk.py
[sg-v4]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/dsv4
[token-topk-v4]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/dsv4_attention.py
[msa-topk]: https://github.com/MiniMax-AI/MSA/blob/9175bebf2b623ca0ea8ad5247bb615cd424598a0/python/fmha_sm100/csrc/sparse_topk_select.cu
[trt-blocks]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/minimaxM3SelectBlocks.cu
[sg-minimax-topk]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/minimax_decode_topk.py
[msa-radix]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/topk_select_radix_sm12x.py
[msa-chunk]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/topk_select_chunked_sm12x.py
[msa-count]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/msa_ops/cute_dsl/topk_select_countrank_sm12x.py
