# FlashInfer Attention Backends Support

Date: 2026-07-06

TODO: Remove this file before submitting the branch to `vllm-project/main`.
This is a local review aid, not intended upstream PR content.

This document summarizes FlashInfer attention backend support as used by vLLM.
Source code is treated as the evidence for support. Tests are listed as vLLM
coverage only; they are not used as proof that a capability is implemented.

## Terms

- **FP8 compute** means the backend performs attention with FP8 Q/K/V inputs
  natively. If the source casts FP8 inputs to FP16 before running the kernel,
  this is recorded as **False**.
- **FP8 KV cache** means FP8 K/V input or KV cache is accepted and scale
  plumbing exists.
- **FP4 KV cache** means FlashInfer's packed `uint8` NVFP4 KV cache with
  `kv_cache_sf`; this is not generic FP4 KV.
- **DCP support** is a vLLM integration property unless explicitly stated
  otherwise. FlashInfer source evidence for `return_lse` is necessary but not
  sufficient for vLLM DCP.

## Support Matrix

| Kernel / Path | Architectures Supported | DCP Support | FP8 Compute | FP8 KV Cache | SWA Attention | FP4 / NVFP4 KV Cache | FlashInfer Source Evidence | vLLM Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BatchPrefillWithPagedKVCacheWrapper`, backend `auto` | SM80, SM90, SM100 | vLLM wrapper DCP only; FlashInfer has no native DCP mode [vllm-dcp-prefill] | FA3: True; FA2: False for native FP8 compute because non-FA3 ragged prefill casts FP8 inputs to FP16, and paged FA2 source only proves FP8 scale plumbing [fa3-backend-selection], [non-fa3-fp8-cast], [paged-prefill-fp8-scale-plumbing] | True when backend accepts dtype; FA3 requires FP8 Q for FP8 KV [fa3-fp8-kv-constraint], [paged-prefill-fp8-scale-plumbing] | True for FA2/auto causal paged prefill: source passes `window_left` into module generation and run args [paged-prefill-swa-source] | `auto` cannot choose FA3 for `torch.uint8` KV; packed NVFP4 goes to FA2 path and requires `kv_cache_sf` [fa3-nvfp4-rejection], [paged-prefill-nvfp4-source] | [auto-backend-selection], [paged-prefill-plan-backend], [paged-prefill-swa-source], [paged-prefill-nvfp4-source] | [vllm-cov-flashinfer-causal], [vllm-cov-context-parallel-flashinfer]; no targeted vLLM coverage for explicit FA2/FA3/NVFP4/SWA wrapper knobs |
| `BatchPrefillWithPagedKVCacheWrapper`, backend `fa2` | SM80, SM90, SM100 | vLLM wrapper DCP only [vllm-dcp-prefill] | False for native FP8 compute; source proves scale tensors are passed, not native FP8 arithmetic [paged-prefill-fp8-scale-plumbing] | True as accepted/scaled input; native FP8 compute not proven [paged-prefill-fp8-scale-plumbing] | True: `window_left` is passed into plan and run args [paged-prefill-swa-source] | True: `uint8` KV requires `kv_cache_sf`, scales are unpacked and passed into run args [paged-prefill-nvfp4-source] | [paged-prefill-plan-backend], [paged-prefill-fp8-scale-plumbing], [paged-prefill-swa-source], [paged-prefill-nvfp4-source] | No targeted vLLM coverage found for explicit `backend="fa2"` paged prefill |
| `BatchPrefillWithPagedKVCacheWrapper`, backend `fa3` | SM90 | vLLM wrapper DCP only if FA3 constraints pass [vllm-dcp-prefill] | True for source-selected FA3 FP8 path [non-fa3-fp8-cast], [fa3-backend-selection] | True only when Q is FP8 [fa3-fp8-kv-constraint] | True only when FA3 support checks pass; FA3 rejects custom mask, RoPE/ALiBi mode, and FP16 QK reductions [fa3-backend-selection] | False: FA3 rejects `torch.uint8` KV [fa3-nvfp4-rejection] | [fa3-backend-selection], [fa3-fp8-kv-constraint], [fa3-nvfp4-rejection] | No targeted vLLM coverage found for explicit `backend="fa3"` paged prefill |
| `BatchPrefillWithPagedKVCacheWrapper`, backend `cudnn` | SM80, SM90, SM100 | Not wired by vLLM DCP wrapper path; raw cuDNN prefill can return LSE [cudnn-paged-prefill-call] | BF16 source path exists; FP8 source path exists behind cuDNN version check [cudnn-prefill-fp8-version] | True for FP8 input path in cuDNN graph source when cuDNN version allows it [cudnn-prefill-fp8-version] | False / not source-evidenced: wrapper accepts `window_left` but cuDNN call does not pass it [cudnn-paged-prefill-call] | False / not source-evidenced: cuDNN call has no `kv_cache_sf` argument [cudnn-paged-prefill-call] | [cudnn-paged-prefill-wrapper], [cudnn-paged-prefill-call], [cudnn-prefill-fp8-version] | No targeted vLLM coverage found for FlashInfer cuDNN paged prefill |
| `BatchPrefillWithPagedKVCacheWrapper`, backend `trtllm-gen` | SM100, SM103 | Source has LSE return, but vLLM direct TRTLLM DCP is blocked [trtllm-context-lse-source], [vllm-trtllm-dcp-block] | True [trtllm-context-source] | True [trtllm-context-source] | Causal: True; non-causal + `window_left >= 0`: rejected [trtllm-context-swa-rejection] | True with `kv_cache_sf`; FP4 output requires FP8 query, but that is output not KV [trtllm-context-nvfp4-source] | [trtllm-context-source], [trtllm-context-lse-source], [trtllm-context-swa-rejection], [trtllm-context-nvfp4-source] | [vllm-cov-trtllm-full-attn], [vllm-cov-trtllm-nvfp4-kv]; no vLLM DCP coverage for direct TRTLLM path |
| `BatchPrefillWithRaggedKVCacheWrapper`, backend `auto/fa2/cutlass` | SM80, SM90, SM100 | vLLM wrapper DCP new-token path only [vllm-dcp-prefill] | FA2/cutlass: False for native FP8 compute; non-FA3 FP8 inputs are cast to FP16 [non-fa3-fp8-cast] | Accepted/scaled input only; native FP8 compute is not source-proven for non-FA3 [ragged-prefill-nvfp4-source], [non-fa3-fp8-cast] | True: source passes `window_left` into backend generation and run args [ragged-prefill-swa-source] | True for ragged NVFP4 scale plumbing, except FA3 rejects `torch.uint8` [ragged-prefill-nvfp4-source], [fa3-nvfp4-rejection] | [ragged-prefill-backends], [ragged-prefill-swa-source], [ragged-prefill-nvfp4-source], [non-fa3-fp8-cast] | [vllm-cov-flashinfer-causal], [vllm-cov-context-parallel-flashinfer]; no targeted vLLM coverage for explicit ragged backend knobs |
| `BatchPrefillWithRaggedKVCacheWrapper`, backend `fa3` | SM90 | vLLM wrapper DCP only if FA3 constraints pass [vllm-dcp-prefill] | True [non-fa3-fp8-cast], [fa3-backend-selection] | True only when Q is FP8 [fa3-fp8-kv-constraint] | True only when FA3 checks pass [fa3-backend-selection], [ragged-prefill-swa-source] | False: FA3 rejects `torch.uint8` KV [fa3-nvfp4-rejection] | [fa3-backend-selection], [fa3-fp8-kv-constraint], [fa3-nvfp4-rejection], [ragged-prefill-swa-source] | No targeted vLLM coverage found for explicit `backend="fa3"` ragged prefill |
| `BatchPrefillWithRaggedKVCacheWrapper`, backend `cudnn` | SM80, SM90, SM100 | Not wired by vLLM DCP wrapper path; raw cuDNN prefill can return LSE [cudnn-ragged-prefill-call] | BF16 source path exists; FP8 ragged cuDNN source evidence not established in this report [cudnn-ragged-prefill-call] | Not source-established for ragged cuDNN | False / not source-evidenced: wrapper accepts `window_left` but cuDNN call does not pass it [cudnn-ragged-prefill-call] | False / not source-evidenced: cuDNN call has no `kv_cache_sf` argument [cudnn-ragged-prefill-call] | [cudnn-ragged-prefill-wrapper], [cudnn-ragged-prefill-call] | No targeted vLLM coverage found for FlashInfer cuDNN ragged prefill |
| `BatchDecodeWithPagedKVCacheWrapper`, backend `auto` | SM80, SM90, SM100 | vLLM wrapper DCP only [vllm-dcp-decode] | FA3: True when selected; FA2 tensor-core path source proves FP8 scale plumbing, not native FP8 compute [decode-auto-tensor-core-selection], [decode-fp8-scale-plumbing] | True as accepted/scaled input; FA3 requires FP8 Q for FP8 KV [decode-auto-tensor-core-selection], [fa3-fp8-kv-constraint] | True: source passes `window_left` into tensor-core plan and run args [decode-swa-source] | For decode wrapper, NVFP4 source is tensor-core path, not legacy decode kernel; `uint8` KV requires `kv_cache_sf` [decode-nvfp4-source], [decode-tensor-core-source] | [decode-auto-tensor-core-selection], [decode-swa-source], [decode-nvfp4-source], [decode-tensor-core-source] | [vllm-cov-flashinfer-causal], [vllm-cov-context-parallel-flashinfer]; no targeted vLLM coverage for explicit FA2/FA3/NVFP4/SWA wrapper knobs |
| `BatchDecodeWithPagedKVCacheWrapper`, backend `fa2` with `use_tensor_cores=True` | SM80, SM90, SM100 | vLLM wrapper DCP only [vllm-dcp-decode] | False for native FP8 compute; source proves scale plumbing, not native FP8 arithmetic [decode-fp8-scale-plumbing] | True as accepted/scaled input [decode-fp8-scale-plumbing] | True: `window_left` reaches plan/run args [decode-swa-source] | True through tensor-core wrapper path; do not describe as legacy decode-kernel support [decode-nvfp4-source], [decode-tensor-core-source] | [decode-fp8-scale-plumbing], [decode-swa-source], [decode-nvfp4-source], [decode-tensor-core-source] | No targeted vLLM coverage found for explicit `backend="fa2"` tensor-core decode wrapper |
| `BatchDecodeWithPagedKVCacheWrapper`, backend `fa3` | SM90 | vLLM wrapper DCP only if FA3 constraints pass [vllm-dcp-decode] | True when selected [fa3-backend-selection], [decode-auto-tensor-core-selection] | True only when Q is FP8 [fa3-fp8-kv-constraint] | True only when FA3 checks pass [fa3-backend-selection], [decode-swa-source] | False: FA3 rejects `torch.uint8` KV [fa3-nvfp4-rejection] | [fa3-backend-selection], [fa3-fp8-kv-constraint], [fa3-nvfp4-rejection], [decode-auto-tensor-core-selection] | No targeted vLLM coverage found for explicit `backend="fa3"` decode wrapper |
| `BatchDecodeWithPagedKVCacheWrapper`, backend `trtllm-gen` | SM100, SM103 | Source has LSE; vLLM direct TRTLLM DCP is blocked [decode-trtllm-lse-source], [vllm-trtllm-dcp-block] | True [decode-trtllm-source] | True [decode-trtllm-source] | True: direct TRTLLM decode accepts `window_left`; source proof is argument plumbing [decode-trtllm-source] | True for `trtllm-gen`; source rejects relevant NVFP4/LSE combinations on XQA [decode-trtllm-nvfp4-source], [decode-xqa-restrictions] | [decode-trtllm-source], [decode-trtllm-lse-source], [decode-trtllm-nvfp4-source], [decode-xqa-restrictions] | [vllm-cov-trtllm-full-attn], [vllm-cov-trtllm-nvfp4-kv]; no vLLM DCP coverage for direct TRTLLM path |
| `trtllm_batch_context_with_kv_cache` direct API | SM100, SM103 | Source has LSE; vLLM direct TRTLLM DCP is blocked [trtllm-context-lse-source], [vllm-trtllm-dcp-block] | True [trtllm-context-source] | True [trtllm-context-source] | Causal: True; non-causal + `window_left >= 0`: rejected [trtllm-context-swa-rejection] | True with `kv_cache_sf` [trtllm-context-nvfp4-source] | [trtllm-context-source], [trtllm-context-lse-source], [trtllm-context-swa-rejection], [trtllm-context-nvfp4-source] | [vllm-cov-trtllm-full-attn], [vllm-cov-trtllm-nvfp4-kv]; no vLLM LSE-return or DCP coverage for direct context API |
| `trtllm_batch_decode_with_kv_cache` direct API | SM90, SM100, SM103, SM120, SM121 | Source has LSE only for `trtllm-gen`; vLLM direct TRTLLM DCP is blocked [decode-trtllm-lse-source], [vllm-trtllm-dcp-block] | `trtllm-gen`: True; `xqa`: rejects FP8 Q [decode-xqa-restrictions] | True for `trtllm-gen`; XQA restrictions apply [decode-trtllm-source], [decode-xqa-restrictions] | True: `window_left` is direct API argument and is passed to backend dispatch [decode-trtllm-source] | `trtllm-gen`: True; XQA rejects NVFP4 output and LSE [decode-trtllm-nvfp4-source], [decode-xqa-restrictions] | [decode-direct-auto-arch], [decode-trtllm-source], [decode-trtllm-lse-source], [decode-xqa-restrictions] | [vllm-cov-trtllm-full-attn], [vllm-cov-trtllm-nvfp4-kv], [vllm-cov-flashinfer-xqa-decode], [vllm-cov-flashinfer-xqa-scale] |
| `flashinfer.decode.cudnn_batch_decode_with_kv_cache` direct API | SM80, SM90, SM100 | Not vLLM wrapper DCP | BF16 source path only in this report [cudnn-decode-source] | Not source-established | Not source-established | Not source-established | [cudnn-decode-source] | No targeted vLLM coverage found for FlashInfer cuDNN decode direct API |
| `MultiLevelCascadeAttentionWrapper` | SM80, SM90, SM100 | No native DCP; source is cascade-level LSE merge, not DCP communication [cascade-lse-merge] | Not source-established at cascade wrapper surface | `kv_data_type` is forwarded into inner paged prefill plans, but scale plumbing is not exposed at cascade `run` [cascade-plan-forward] | `window_left` is forwarded into inner paged prefill plans; capability depends on inner backend [cascade-plan-forward] | Not source-established; no `kv_cache_sf` at cascade `run` [cascade-run-surface] | [cascade-plan-forward], [cascade-lse-merge], [cascade-run-surface] | No targeted vLLM coverage found for FlashInfer `MultiLevelCascadeAttentionWrapper` |
| `fast_decode_plan` | SM80, SM90, SM100 | N/A | N/A | N/A | N/A | N/A | Planner helper only; source builds plan metadata and stores `window_left`, no attention compute [fast-decode-plan-source] | N/A |

## FlashInfer Source Evidence

[auto-backend-selection]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/utils.py#L476-L518  
**auto-backend-selection-{flashinfer/flashinfer/utils.py}**: `determine_attention_backend` returns FA3 only when SM90a and FA3 checks pass; otherwise FA2.

[fa3-backend-selection]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/utils.py#L387-L432  
**fa3-backend-selection-{flashinfer/flashinfer/utils.py}**: FA3 support rejects custom masks, non-NONE positional encoding, FP16 QK reductions, FP8 KV without FP8 Q, and `torch.uint8` packed NVFP4 KV.

[fa3-fp8-kv-constraint]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/utils.py#L423-L428  
**fa3-fp8-kv-constraint-{flashinfer/flashinfer/utils.py}**: FA3 FP8 KV requires FP8 query.

[fa3-nvfp4-rejection]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/utils.py#L429-L431  
**fa3-nvfp4-rejection-{flashinfer/flashinfer/utils.py}**: FA3 rejects `torch.uint8` packed NVFP4 KV.

[paged-prefill-plan-backend]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2071-L2096  
**paged-prefill-plan-backend-{flashinfer/flashinfer/prefill.py}**: paged prefill `auto` resolves through `determine_attention_backend`, then creates the selected module.

[paged-prefill-swa-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2080-L2096  
**paged-prefill-swa-source-{flashinfer/flashinfer/prefill.py}**: paged prefill passes `window_left >= 0` into module construction.

[paged-prefill-fp8-scale-plumbing]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2521-L2540  
**paged-prefill-fp8-scale-plumbing-{flashinfer/flashinfer/prefill.py}**: paged prefill extracts FP8 scale tensors when Q is FP8 and passes them to the module run args; this proves scale plumbing, not native FA2 FP8 compute.

[paged-prefill-nvfp4-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2363-L2371  
**paged-prefill-nvfp4-source-{flashinfer/flashinfer/prefill.py}**: paged prefill requires `kv_cache_sf` for `torch.uint8` KV and unpacks K/V block scales.

[non-fa3-fp8-cast]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L3522-L3531  
**non-fa3-fp8-cast-{flashinfer/flashinfer/prefill.py}**: ragged prefill casts FP8 Q/K/V to FP16 whenever backend is not FA3; this is direct source evidence against native FA2/cutlass FP8 compute on that path.

[cudnn-paged-prefill-wrapper]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L1668-L1670  
**cudnn-paged-prefill-wrapper-{flashinfer/flashinfer/prefill.py}**: paged prefill accepts explicit `backend="cudnn"` and asserts NHD layout.

[cudnn-paged-prefill-call]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2456-L2484  
**cudnn-paged-prefill-call-{flashinfer/flashinfer/prefill.py}**: paged prefill cuDNN path calls `cudnn_batch_prefill_with_kv_cache` with Q/K/V, sequence lengths, block tables, scales, output and LSE; it does not pass `window_left` or `kv_cache_sf`.

[cudnn-prefill-fp8-version]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/cudnn/prefill.py#L177-L183  
**cudnn-prefill-fp8-version-{flashinfer/flashinfer/cudnn/prefill.py}**: cuDNN prefill source gates FP8 on cuDNN backend version >= 9.17.1.

[trtllm-context-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L4106-L4213  
**trtllm-context-source-{flashinfer/flashinfer/prefill.py}**: direct TRTLLM context API exposes query/KV cache, FP8/NVFP4 scales, output dtype, `window_left`, and layout controls.

[trtllm-context-lse-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L4228-L4241  
**trtllm-context-lse-source-{flashinfer/flashinfer/prefill.py}**: direct TRTLLM context API documents `lse` and `return_lse`.

[trtllm-context-swa-rejection]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L4246-L4250  
**trtllm-context-swa-rejection-{flashinfer/flashinfer/prefill.py}**: direct TRTLLM context rejects sliding-window non-causal attention.

[trtllm-context-nvfp4-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L4265-L4299  
**trtllm-context-nvfp4-source-{flashinfer/flashinfer/prefill.py}**: direct TRTLLM context requires `kv_cache_sf` for `torch.uint8` KV and checks FP4 output constraints.

[ragged-prefill-backends]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L2765-L2771  
**ragged-prefill-backends-{flashinfer/flashinfer/prefill.py}**: ragged prefill lists explicit backends, including `cudnn`, `cutlass`, and `cute-dsl`; `auto` remains a backend value, not proof that cuDNN is auto-selected.

[ragged-prefill-swa-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L3181-L3212  
**ragged-prefill-swa-source-{flashinfer/flashinfer/prefill.py}**: ragged prefill passes `window_left >= 0` into module construction for non-cuDNN/non-cute paths.

[ragged-prefill-nvfp4-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L3409-L3437  
**ragged-prefill-nvfp4-source-{flashinfer/flashinfer/prefill.py}**: ragged prefill fuses scales for NVFP4, unpacks `kv_cache_sf`, and adjusts output shape for packed `uint8` KV.

[cudnn-ragged-prefill-wrapper]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L3004-L3011  
**cudnn-ragged-prefill-wrapper-{flashinfer/flashinfer/prefill.py}**: ragged cuDNN requires max token/sequence lengths and V/O indptr metadata.

[cudnn-ragged-prefill-call]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py#L3487-L3520  
**cudnn-ragged-prefill-call-{flashinfer/flashinfer/prefill.py}**: ragged cuDNN calls `cudnn_batch_prefill_with_kv_cache` with Q/K/V, sequence lengths, scales, offsets, output and LSE; it does not pass `window_left` or `kv_cache_sf`.

[decode-auto-tensor-core-selection]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L1181-L1201  
**decode-auto-tensor-core-selection-{flashinfer/flashinfer/decode.py}**: tensor-core decode `auto` resolves via `determine_attention_backend` for FP8-involved types, otherwise sets FA2.

[decode-swa-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L1215-L1239  
**decode-swa-source-{flashinfer/flashinfer/decode.py}**: tensor-core decode passes `window_left` into plan args.

[decode-fp8-scale-plumbing]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L1633-L1655  
**decode-fp8-scale-plumbing-{flashinfer/flashinfer/decode.py}**: decode extracts FP8 scale tensors when Q is FP8 and passes them into run args; this proves scale plumbing, not native FA2 FP8 compute.

[decode-nvfp4-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L1460-L1468  
**decode-nvfp4-source-{flashinfer/flashinfer/decode.py}**: decode requires `kv_cache_sf` for `torch.uint8` KV and unpacks K/V block scales.

[decode-tensor-core-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L1599-L1615  
**decode-tensor-core-source-{flashinfer/flashinfer/decode.py}**: decode tensor-core path uses prefill-style run args; this distinguishes NVFP4 decode wrapper support from legacy decode-kernel support.

[decode-trtllm-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L2489-L2517  
**decode-trtllm-source-{flashinfer/flashinfer/decode.py}**: direct TRTLLM decode API signature exposes `window_left`, backend selection, scales, `kv_cache_sf`, LSE, and return controls.

[decode-direct-auto-arch]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L2592-L2596  
**decode-direct-auto-arch-{flashinfer/flashinfer/decode.py}**: direct TRTLLM decode `auto` chooses `trtllm-gen` on SM100/SM103 and XQA on SM90/SM120/SM121.

[decode-trtllm-lse-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L2645-L2656  
**decode-trtllm-lse-source-{flashinfer/flashinfer/decode.py}**: direct TRTLLM decode documents LSE support only for `trtllm-gen`.

[decode-xqa-restrictions]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L2713-L2728  
**decode-xqa-restrictions-{flashinfer/flashinfer/decode.py}**: XQA rejects NVFP4 output, cumulative query lengths, shared-page-index false, and LSE.

[decode-trtllm-nvfp4-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L2681-L2708  
**decode-trtllm-nvfp4-source-{flashinfer/flashinfer/decode.py}**: direct TRTLLM decode requires `kv_cache_sf` for `torch.uint8` KV and validates FP8 scale tensor format.

[cudnn-decode-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/cudnn/decode.py#L75-L180  
**cudnn-decode-source-{flashinfer/flashinfer/cudnn/decode.py}**: cuDNN decode direct API builds a BF16 SDPA graph with paged attention tables; no FP8, SWA, LSE, or NVFP4 evidence is present in these lines.

[cascade-plan-forward]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/cascade.py#L401-L517  
**cascade-plan-forward-{flashinfer/flashinfer/cascade.py}**: cascade plan forwards `window_left`, `q_data_type`, and `kv_data_type` to inner paged prefill wrappers.

[cascade-lse-merge]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/cascade.py#L547-L556  
**cascade-lse-merge-{flashinfer/flashinfer/cascade.py}**: cascade run gets `(out, lse)` from inner wrappers and merges attention states.

[cascade-run-surface]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/cascade.py#L522-L556  
**cascade-run-surface-{flashinfer/flashinfer/cascade.py}**: cascade run accepts only `q` and `paged_kv_cache`, so scale and `kv_cache_sf` surfaces are not exposed there.

[fast-decode-plan-source]: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/decode.py#L3100-L3265  
**fast-decode-plan-source-{flashinfer/flashinfer/decode.py}**: `fast_decode_plan` is a planner helper that builds plan metadata and stores `window_left`; it does not execute attention.

[vllm-dcp-prefill]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L218-L316  
**vllm-dcp-prefill-{vllm/v1/attention/backends/flashinfer.py}**: vLLM DCP prefill wraps paged-context and ragged-new-token FlashInfer wrappers and combines LSE states.

[vllm-dcp-decode]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L2175-L2198  
**vllm-dcp-decode-{vllm/v1/attention/backends/flashinfer.py}**: vLLM wrapper decode DCP all-gathers query, requests LSE, and performs DCP combine.

[vllm-trtllm-dcp-block]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py#L1843-L1869  
**vllm-trtllm-dcp-block-{vllm/v1/attention/backends/flashinfer.py}**: local vLLM branch blocks DCP on direct `trtllm_batch*` API paths.

## vLLM Coverage

This section lists vLLM tests only. FlashInfer upstream tests are intentionally
not used as coverage evidence in this document.

**[vllm-cov-flashinfer-backend-list-{tests/v1/attention/test_attention_backends.py}](https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_attention_backends.py#L38-L52)**: vLLM includes `AttentionBackendEnum.FLASHINFER` in the generic attention backend correctness suite when FlashInfer is importable.

[vllm-cov-flashinfer-causal]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_attention_backends.py#L577-L623  
**vllm-cov-flashinfer-causal-{tests/v1/attention/test_attention_backends.py}**: generic causal attention correctness covers decode, prefill, mixed batches, and TP head partition simulation through the vLLM `FLASHINFER` backend. It does not pin FlashInfer wrapper backend values such as `fa2`, `fa3`, `cudnn`, or `trtllm-gen`.

[vllm-cov-flashinfer-xqa-scale]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_attention_backends.py#L633-L650  
**vllm-cov-flashinfer-xqa-scale-{tests/v1/attention/test_attention_backends.py}**: verifies vLLM XQA decode scale selection for BF16 versus FP8 query dtype.

[vllm-cov-flashinfer-xqa-decode]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_attention_backends.py#L657-L742  
**vllm-cov-flashinfer-xqa-decode-{tests/v1/attention/test_attention_backends.py}**: on SM90, verifies vLLM routes FlashInfer decode metadata to `FlashInferTrtllmAPIDecode` with XQA and compares output to the generic SDPA reference harness.

[vllm-cov-sliding-excludes-flashinfer]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_attention_backends.py#L747-L813  
**vllm-cov-sliding-excludes-flashinfer-{tests/v1/attention/test_attention_backends.py}**: vLLM's sliding-window backend correctness test enumerates FlashAttention, FlexAttention, and Triton backends, but not `FLASHINFER`.

**[vllm-cov-trtllm-helper-{tests/v1/attention/test_trtllm_attention_integration.py}](https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_trtllm_attention_integration.py#L283-L508)**: vLLM integration helper sets `use_trtllm_attention=True`, builds FlashInfer metadata, asserts `TRTLLMPrefill` / `FlashInferTrtllmAPIDecode` with `TRTLLM_GEN`, runs `FlashInferImpl.forward`, and compares against an SDPA reference.

[vllm-cov-trtllm-full-attn]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_trtllm_attention_integration.py#L515-L524  
**vllm-cov-trtllm-full-attn-{tests/v1/attention/test_trtllm_attention_integration.py}**: parameterized vLLM test invokes the TRTLLM integration helper over all batch specs.

[vllm-cov-trtllm-nvfp4-kv]: https://github.com/vllm-project/vllm/blob/main/tests/v1/attention/test_trtllm_attention_integration.py#L527-L539  
**vllm-cov-trtllm-nvfp4-kv-{tests/v1/attention/test_trtllm_attention_integration.py}**: parameterized vLLM test invokes the TRTLLM integration helper with `kv_cache_dtype="nvfp4"` and an NVFP4 model.

**[vllm-cov-context-parallel-flashinfer-params-{tests/distributed/test_context_parallel.py}](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_context_parallel.py#L137-L153)**: DCP GSM8K parameterization includes `Qwen/Qwen2.5-1.5B-Instruct` with `attn_backend="FLASHINFER"`.

[vllm-cov-context-parallel-flashinfer]: https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_context_parallel.py#L220-L311  
**vllm-cov-context-parallel-flashinfer-{tests/distributed/test_context_parallel.py}**: vLLM DCP end-to-end coverage passes `--decode-context-parallel-size`, optional `--attention-backend=FLASHINFER`, runs GSM8K through a remote OpenAI server, and asserts minimum accuracy. This is end-to-end DCP coverage, not per-wrapper backend selection coverage.

**[vllm-cov-dcp-lse-combine-{tests/distributed/test_dcp_a2a.py}](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_dcp_a2a.py#L250-L311)**: unit coverage for `_lse_weighted_combine(..., return_lse=True)` in base-e and base-2 LSE modes.

**[vllm-cov-dcp-pack-unpack-{tests/distributed/test_dcp_a2a.py}](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_dcp_a2a.py#L322-L377)**: CUDA coverage for packing LSE into A2A send buffers and unpacking/combining against a reference.

**[vllm-cov-dcp-distributed-a2a-{tests/distributed/test_dcp_a2a.py}](https://github.com/vllm-project/vllm/blob/main/tests/distributed/test_dcp_a2a.py#L379-L498)**: distributed 4-GPU coverage for `dcp_a2a_lse_reduce`, including workspace-manager mode.

## Open Gaps

- FA2 FP8 compute is not supported by source evidence in this document. The
  source shows FP8 scale plumbing, but ragged non-FA3 path explicitly casts FP8
  inputs to FP16.
- cuDNN SWA is not source-evidenced through the wrapper paths inspected here;
  `window_left` is not passed into the cuDNN prefill calls.
- cuDNN NVFP4 KV is not source-evidenced through the wrapper paths inspected
  here; `kv_cache_sf` is not passed into the cuDNN calls.
- vLLM has no targeted FlashInfer SWA correctness coverage in
  `test_attention_backends.py`; the sliding-window backend list excludes
  `FLASHINFER` [vllm-cov-sliding-excludes-flashinfer].
- vLLM has TRTLLM full-attention and NVFP4-KV integration coverage, but no vLLM
  test found here that requests `return_lse=True` from direct
  `trtllm_batch_context_with_kv_cache` / `trtllm_batch_decode_with_kv_cache`.
- Cascade support should not be generalized from inner wrappers for FP8 scale or
  NVFP4 KV because cascade `run` does not expose scale or `kv_cache_sf`.
