# Gated DeltaNet: prefill, decode and MTP

Return to the [primitive index](../SKILL.md).

GDN's scalar-gated delta recurrence is not [KDA](kda.md)'s channel-gated
recurrence or [Mamba2](mamba2.md)'s selective state-space update.

## Chunked prefill

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| FLA chunk GDN - `chunk_gated_delta_rule` | Triton | Flash Linear Attention [upstream v0.5.1][fla-up]; [vLLM adaptation][fla-v] | One implementation family. Gate scan, KKT/triangular solve, WY factors, chunk-state propagation and output form one pipeline. Packed-sequence and initial/final-state variants; upstream is a family reference, not a claim that the vLLM copy exactly matches this release. |
| Chunked delta rule - `chunk_gated_delta_rule` | CuTe DSL | FlashInfer: [SM90][fi90], [Blackwell][fi100], [SM120][fi120] | Architecture-specific chunk schedules grouped as one family; flattened varlen inputs and explicit state output. SM120 HMMA is not the Blackwell tcgen05 path. |
| CP delta rule - `cp_delta_rule_dsl_sm90`, `gdn_cp_prefill`, `cp_delta_rule_dsl_sm120` | CuTe DSL | FlashInfer: [SM90][cp90], [Blackwell][cp100], [SM120][cp120] | CP precompute/fixup/prefill algorithm, distinct from ordinary chunked GDN. T/MN preparation and HMMA/SIMT fixup are constituent stages, not extra implementations. |
| Blackwell chunk pipeline - `chunk_gated_delta_rule_cutedsl` | CuTe DSL | vLLM [source][v-cute] | KKT inverse/UW, state propagation and output grouped. SM10.x, K=128 in the reviewed vLLM selection. This is not the FlashInfer import route. |

## Recurrent decode and multi-token verification

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| FLA fused recurrence - `fused_recurrent_gated_delta_rule`, `fused_recurrent_gated_delta_rule_update` | Triton | Flash Linear Attention [upstream v0.5.1][fla-rec-up]; [vLLM adaptation][fla-rec-v] | Decode/multi-token recurrence with indexed state and sequence offsets in adapted copies. Same provider family as FLA prefill, different recurrent algorithm; framework-local gating/replay helpers are not additional candidates. |
| FP32-state decode - `gated_delta_rule_decode_pretranspose`, `gated_delta_rule_decode` | CuTe DSL | FlashInfer: [V-major][fi-vk], [K-major][fi-kv] | Two material memory-access variants: `[B,HV,V,K]` versus `[B,HV,K,V]` state. Pretransposed path documents SM90+; do not transpose state implicitly when substituting. |
| FP32-state MTP - `gated_delta_rule_mtp` | CuTe DSL | FlashInfer [source][fi-mtp] | Multi-token recurrence with inline verification/intermediate-state handling; not the single-token launch contract. |
| BF16-state decode/MTP - `run_gdn_decode_bf16state_mtp_ilp4`, `gated_delta_rule_t1_wide_vec`, `gated_delta_rule_mtp_wide_vec` | CuTe DSL | FlashInfer [source][fi-bf16]; SGLang [ring-write adaptation][s-ring] | ILP4 and wide-vector schedules grouped, not exported aliases counted separately. SGLang's vendored variant adds fused ReplaySSM ring writes; SM100 BF16 pooled state, K=V=128 and split-pool writes. |
| Fused gate + recurrent update - `cutedsl_fused_sigmoid_gating_delta_rule_update` | CuTe DSL | SGLang [in-tree kernel][s-decode] | SM90+ pooled-state decode, including packed/varlen input paths. Distinct local CuTe implementation, not a FlashInfer serving adapter. |
| Post-convolution short MTP - `fused_gdn_decode_post_conv_mtp` | CUDA | vLLM [source][v-cuda] | Qwen-specific fused short multi-token decode after convolution; not generic chunk prefill. |
| Fused SM120 decode with PDL - `gdn_fused_decode_step` | CuTe DSL | FlashInfer [experimental kernel][fi-fused-cute] | SM120 fused decode with programmatic dependent launch. |
| Persistent SM120 fused decode - `gdn_fused_decode_step` | CUDA | FlashInfer [experimental device source][fi-fused-cuda] | Separate persistent CUDA implementation, not a CuTe launch specialization. |
| AITER fused GDN - `fused_rearrange_sigmoid_gated_delta_rule` | Triton | AITER [device source][aiter-up]; [vLLM integration][aiter] | AMD fused rearrangement/gate/recurrence provider, adapted from FLA/vLLM. Upstream revision resolves the retained vLLM Dockerfile's v0.1.21.post2 pin; no separate vLLM implementation is implied. |

## Replay and checkpoint materialization

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| WY output-only / cached-update GDN - `gated_delta_rule_mtp`, `gated_delta_rule_mtp_ucache`, `gated_delta_rule_mtp_ucache_flush` | CuTe DSL | FlashInfer: [output-only][wy], [U-cache][ucache], [flush][flush] | BF16-state MTP/replay variants with different state-write policies. Output-only avoids full state writes; U-cache retains low-rank updates and flush materializes them. Flush is part of the cached-update family, not another recurrence provider. |

## Additional provider backends

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Cake GDN - `chunk_gated_delta_rule(backend="cake")`, recurrent GDN APIs | CUDA, source-built/generated | Cake in FlashInfer: [manifest][cake], [prefill dispatch][prefill], [decode dispatch][decode] | Optional prefill/decode/MTP backend. Source and state-slot eligibility are route-specific; not every CuTe option transfers to Cake. |
| cuDNN GDN / related delta recurrences - `cudnn_chunk_gated_delta_rule`, `cudnn_chunk_gated_delta_product`, `cudnn_chunk_gated_delta_rule2` | External cuDNN | NVIDIA cuDNN; FlashInfer [dependency graph integration][cudnn] | Packed THD prefill with integer sequence offsets. GDN, multiple-Householder DeltaProduct and channel-gated GDN2 are distinct equations within the provider, not aliases. Explicit CP/indexing/checkpoint options have separate restrictions. |

## Consolidation boundary

FLA copies preserve provider attribution, not a claim of bitwise-identical
framework behavior. The SGLang ring-write variant remains visible because
it changes speculative-state storage. Import-only FlashInfer routes in
vLLM, TRTLLM, SGLang and TokenSpeed do not add implementations.
Standalone local Triton scans, gate/norm/projection helpers and state
gather/scatter/replay plumbing are deliberately omitted.

FLA v0.5.1 is the release identified by the retained TokenSpeed
[FLA attribution](kda.md#decode-and-fused-recurrence), not a refreshed upstream
checkout. SGLang citations remain for its local CuTe recurrence and ring-write
adaptation, rather than redundant copies of generic FLA/vLLM pipelines.

[fla-up]: https://github.com/fla-org/flash-linear-attention/blob/2e38c1fab332174d056928feaf29f8c5fd5ac550/fla/ops/gated_delta_rule/chunk.py
[fla-rec-up]: https://github.com/fla-org/flash-linear-attention/blob/2e38c1fab332174d056928feaf29f8c5fd5ac550/fla/ops/gated_delta_rule/fused_recurrent.py
[fla-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/third_party/flash_linear_attention/ops/chunk.py
[fi90]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py
[fi100]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/blackwell/gdn_prefill.py
[fi120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py
[cp90]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py
[cp100]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/blackwell/gdn_cp_prefill.py
[cp120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py
[v-cute]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl
[fla-rec-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
[fi-vk]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_pretranspose.py
[fi-kv]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_nontranspose.py
[fi-mtp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_mtp.py
[fi-bf16]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_bf16_state.py
[s-ring]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/cutedsl_gdn_mtp_ring.py
[s-decode]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/cutedsl_gdn.py
[v-cuda]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu
[fi-fused-cute]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/experimental/kernel/gdn_fused_decode_cutedsl_sm120_pdl.py
[fi-fused-cuda]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/experimental/kernel/gdn_fused_decode_sm120.cu
[aiter-up]: https://github.com/ROCm/aiter/blob/2e38b2405bd688fd2d8e9fea9e7ba8f3eb9abee9/aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_rearrange_sigmoid_gdr.py
[aiter]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
[wy]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py
[ucache]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache.py
[flush]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_gdn.py
[prefill]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_prefill.py
[decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/gdn_decode.py
[cudnn]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cudnn/linear_attention.py
