# Coverage matrix for an attention backend

Copy this into the backend's test PR (or the module docstring of its suite) and
mark each row *covered (test id)*, *N/A (reason)* or *gap (issue)*.

## A. Capability contract (L0)

| # | Case |
| --- | --- |
| A1 | Each declared `supported_dtypes` value runs; an undeclared dtype is rejected by the selector. |
| A2 | Each declared KV-cache dtype runs (`auto`, `bfloat16`, `float16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, ...); undeclared ones are rejected. |
| A3 | Each value from `get_supported_kernel_block_sizes` runs; a non-supported block size is rejected or remapped by the kernel-block-size logic as intended. |
| A4 | Each `get_supported_head_sizes` value runs; the next unsupported head size used by a real model (e.g. 512) is rejected. |
| A5 | `supports_attn_type`: decoder accepted; encoder, encoder-only, encoder-decoder rejected (or tested if accepted). |
| A6 | Sinks, sliding window, non-causal, alibi, logits soft cap, mm-prefix: declared ones run, undeclared ones are rejected. |
| A7 | `supports_pcp` / `supports_dcp` / KV connector: selector agrees with the declaration. |
| A8 | `customize_spec` and `supported_kv_cache_layouts` for every env/layout combination; capability attributes computed at import do not go stale when env toggles are refreshed. |
| A9 | `get_cudagraph_support` value matches what the runner captures (see F). |
| A10 | Env gates that select the backend (e.g. `VLLM_ROCM_USE_AITER`, `..._MHA`) and the default priority order on each platform. |

## B. Metadata builder (L2)

| # | Case |
| --- | --- |
| B1 | Decode / extend / prefill split for every phase mixture, with the reorder threshold the builder sets (including spec-as-decode widening). |
| B2 | Each metadata field equals an independently computed value (query start locs per phase, max lens, seq lens, block tables). |
| B3 | Context chunking: no context, context < one chunk, exactly one chunk, many chunks, uneven contexts across requests. |
| B4 | Sliding-window tables: trimmed block table and rebased seq lens at window edges and block edges. |
| B5 | `build_for_cudagraph_capture` and `build_for_drafting` produce metadata the forward path accepts, without GPU syncs. |
| B6 | No GPU sync in decode-only builds; CPU copies only when the phase needs them. |
| B7 | Persistent buffers: allocated once, sized from scheduler limits, reused across builds (same `data_ptr`). |
| B8 | Padded batches: `num_actual_tokens` < padded tokens, and requests with query length 0 placed after the real prefills/extends, in two forms: seq_len 0 (FULL CUDA graph request padding) and seq_len > 0 (a KV connector loading asynchronously; vllm-project/vllm#51486). Run each form through every prefill/context chunking path the builder has (B3). Real rows must match the reference, and no kernel may get an empty tensor (vllm-project/vllm#55695). |
| B9 | Builder rejects unsupported layer mixes with a clear error (e.g. two different sliding windows in one group). |

## C. Forward dispatch (L2)

| # | Case |
| --- | --- |
| C1 | One test per row of the dispatch table, asserting the kernel that ran and the numerical output. |
| C2 | Profiling run (`attn_metadata is None`) returns without touching the cache. |
| C3 | Fused-output-quant arguments are rejected if unsupported. |
| C4 | KV-sharing layers / KV-update-bypass layers read K/V from the cache for every phase. |
| C5 | Non-causal and causal both reach the right kernel in every phase that allows them. |
| C6 | Every declared feature (window, sinks, soft cap, ALiBi, and their combinations) x every phase (prefill, extend, decode, spec decode, mixed) x KV dtype gives the reference output. |
| C7 | Branch coverage of the backend module under the unit suite; every uncovered branch is explained. |

## D. KV cache (L2)

| # | Case |
| --- | --- |
| D1 | `do_kv_cache_update` writes the expected bytes for every layout and KV dtype; read back and compare. |
| D2 | Non-unit per-tensor K/V scales applied on write and read. |
| D3 | Slot mapping with padding (`-1`) slots is skipped. |
| D4 | Fused RoPE+cache and fused qk-norm+RoPE+cache hooks match the unfused path. |
| D5 | Layout swap (block-major vs head-major) gives identical attention output. |

## E. Numerics and shapes (L2)

| # | Case |
| --- | --- |
| E1 | Model-shape matrix (head size, H/KV, TP-sharded heads) x phases x KV dtype x block size, covering set pre-merge, product nightly. |
| E2 | Tokens per request 1, 2048, 4096, 8192; batch sizes 1, 16, 128. |
| E3 | Boundaries: block, window, partition, chunk sizes (at, -1, +1). |
| E4 | Long context crossing several kernel partitions. |
| E5 | Sliding window with context much longer than the window, in each phase. |
| E6 | Sinks / soft cap / alibi values from real models where declared. |
| E7 | No NaN/Inf; padded output rows not read downstream. |

## F. CUDA graphs (L2/L3)

| # | Case |
| --- | --- |
| F1 | `NONE`, `PIECEWISE`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` (and `FULL` if supported) produce the same logprobs as eager. |
| F2 | Replay with different seq lens and block tables than at capture. |
| F3 | Uniform-decode batches with spec tokens (q_len > 1) under full graphs. |
| F4 | Mode downgrade behavior when the builder's support level is below the requested mode is the documented one. |

## G. Workspace and memory (L2/L3)

| # | Case |
| --- | --- |
| G1 | Each persistent buffer allocated for a config is used by that config. |
| G2 | Per-forward allocations in the hot path are listed and justified. |
| G3 | Peak memory for a fixed workload stays within a budget. |

## H. Runtime integration (L3)

| # | Case |
| --- | --- |
| H1 | Scheduler: chunked prefill, prefix caching, preemption and recompute. |
| H2 | Model runner v1 and v2 (`VLLM_USE_V2_MODEL_RUNNER`). |
| H3 | Async scheduling on and off. |
| H4 | Speculative decoding (EAGLE/MTP/ngram) with the backend as target and draft. |
| H5 | PCP / DCP: tested, or rejected by the selector. |
| H6 | Hybrid models: shares the KV pool with linear-attention layers. |
| H7 | Mixed sliding and full layers, and mixed head sizes across layers. |
| H8 | Prompt and sample logprobs match a reference backend within tolerance. |

## I. Task accuracy and performance

| # | Case |
| --- | --- |
| I1 | GSM8K (or equivalent) with real weights, one model per supported head size, fp16/bf16 and fp8 KV. |
| I2 | Benchmark config covering decode, prefill, mixed and spec-decode shapes; results stored per main commit and compared per PR. |
