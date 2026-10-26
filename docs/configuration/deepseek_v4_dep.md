# DeepSeek V4 DEP on MI355X

This review branch supports DeepSeek-V4-Pro-0813 on eight MI355X GPUs with
TP1, DP8 attention, EP8, AITER MegaMoE, MoRI and the Rust frontend. The native
sparse-attention and E8M0 optimizations are opt-in and disabled by default.

Set these variables before starting the server:

```bash
export VLLM_USE_RUST_FRONTEND=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FP8BMM=1
export VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA=1
export VLLM_ROCM_USE_AITER_DSV4_FP8_MXSCALE=1
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export AITER_MEGA_FIXED_SLOT_MAX_MTPR=511
export MORI_SHMEM_HEAP_SIZE=16G
```

Use the following options with your DeepSeek V4 model path and other serving
settings:

```bash
vllm serve /path/to/DeepSeek-V4-Pro-0813 \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --enable-expert-parallel --enable-ep-weight-filter \
  --moe-backend aiter_mega_moe --all2all-backend mori_high_throughput \
  --async-scheduling --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"mxfp4"}' \
  --max-num-batched-tokens 16384 --max-num-seqs 16 \
  --long-prefill-token-threshold 16384 \
  --prefill-schedule-interval 1 \
  --dp-prefill-delayer --dp-prefill-delayer-protected-steps 10 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[7,14,21,28,35,42,49,56,63,70,77,84,91,98,105,112]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":6,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 --enable-auto-tool-choice \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --gpu-memory-utilization 0.86
```

This is the real-verification serving form. The recorded performance runs used
synthetic acceptance length 3.77; they do not measure real draft acceptance.
The configuration was validated through c128 balanced decode and c64 AgentX.
The per-rank request limit is 16. Higher concurrency and CPU KV offload require
separate validation. This branch's ASM cache layout rejects KV connectors and
CPU offload because its separate RoPE pool is not transferred.

## Native sparse attention and E8M0

`VLLM_ROCM_USE_AITER_ASM_SPARSE_MLA=1` requires gfx950, DeepSeek V4, TP1,
128 local heads and `fp8_ds_mla` caches (selected by `--kv-cache-dtype fp8`).
Decode quantizes queries to FP8. The native cache stores BF16 RoPE in a separate
pool, which the planner accounts for; this reduces cache capacity. Prefill uses
the existing attention path over the same cached values.

`VLLM_ROCM_USE_AITER_DSV4_FP8_MXSCALE=1` enables native E8M0 activation scales
for compatible shared-expert, attention and DSpark draft projections. Weight
scales must be exactly representable and satisfy the kernel layout restrictions;
otherwise the layer retains its configured path. Query RMSNorm/quantization is
fused only when both query consumers support the result. Power-of-two activation
quantization changes numerical behavior, so evaluate quality on your workload.

The gfx950 MXFP4 indexer automatically selects AITER TopK for K=1024 and a
262144-column logits plane, including short batches and captured graph replay.
It does not add another environment variable.

## DP prefill delayer

`--dp-prefill-delayer` is disabled by default. When enabled, ranks exchange
prefill/decode state once per step through one CPU all-reduce and make the same
prefill release decision. `--dp-prefill-delayer-protected-steps` defaults to 10
and protects that many decode steps after a prefill. Aligning prefills can
increase time to first token.

The delayer requires `--prefill-schedule-interval 1`; it replaces the ordinary
cadence. Elastic EP and custom scheduler classes are unsupported. This branch
adopts [upstream PR #61834](https://github.com/vllm-project/vllm/pull/61834).
The long-prefill threshold retains its existing exception for a sole local
request, so setting a cap does not prove mixed-step duration is bounded.

## Measured limits

Balanced full decode cycles remain 1.9-3.0% above the same-host ATOM control at
c64/c96/c128. The selected c64 AgentX result is 1881.75 output tok/s/node and
59.56 P90 tok/s/user; it does not meet the published ATOM target. The real-model
quality comparison returned NIAH 360/360, AIME 30/30 and GPQA 330/396, compared
with 360/360, 28/30 and 338/396 for the earlier ASM baseline. Quality equivalence
is unproven. The review bundle under `benchmarks/kernels/dsv4_dep_review/` contains
per-stage timings, complete accounting and source hashes.
