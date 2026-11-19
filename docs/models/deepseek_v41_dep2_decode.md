# DeepSeek-V4.1 specialized decode with data and expert parallelism

The experimental DEP executor keeps attention at TP1. Each DP rank owns its
request rows, while experts are sharded across 2, 4 or 8 GPUs on one NVLink
node. Local routed experts retain the original TRTLLM W13/W2 kernels.

## Enable

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --enable-expert-parallel --all2all-backend allgather_reducescatter \
  --additional-config '{"decode_backend":"latency_specialized_dep"}' \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
```

Other model limits, tokenizer and memory settings remain deployment choices.
Set `decode_backend` back to `auto` to disable the executor.

For a different group size, change `--data-parallel-size`. The same
`latency_specialized_dep` backend selects the matching DEP executor. Keep
`--tensor-parallel-size 1` and `--enable-expert-parallel`; these are not the
TP4/TP8 executors.

| GPUs | `--data-parallel-size` | Routed experts per rank (384 total) |
| ---: | ---: | ---: |
| 2 | 2 | 192 |
| 4 | 4 | 96 |
| 8 | 8 | 48 |

The explicit `latency_specialized_dep2`, `latency_specialized_dep4` and
`latency_specialized_dep8` names remain accepted for existing launch files;
their size must match the DP group. Unsupported layouts fail at startup.
Global expert IDs are preserved through dispatch.

## Supported Scope

- SM103 GPUs, TP1 with equal DP and EP sizes as above, CP1/PP1, and modular TRTLLM MoE with linear
  expert placement. No sequence parallelism, EPLB or microbatching.
- DSpark5 with adaptive verification disabled. Sampling is unchanged; the
  TP-sharded Markov sampler is not used across independent DP requests.
- FULL decode graphs, FP8 KV, and the sparse MXFP4 indexer shown above.
- No LoRA, KV transfer, sleep, weight updates or batch-invariant mode.

Unsupported configurations fail at initialization. Pure decode graphs up to
128 token rows per rank use the executor; prefill, eager execution and larger
graphs retain the original layer loop. Request counts and token rows differ:
DSpark5 can produce six target rows and five draft rows per request.

The executor reuses the specialized attention/indexer and Engram prefetch
paths, adds local mHC boundaries without TP collectives, and fuses top-k with
activation quantization before expert dispatch.

For uniform, nonempty decode graphs, expert contributions are published
directly to the rank that owns each request. The consumer fuses the reduction,
local shared-expert addition, mHC, normalization and activation quantization.
Eligible target layers also write routing and quantization outputs directly
into a symmetric dispatch buffer, and fuse expert finalize with publication.
The target and draft share three rotating owner buffers on the model stream.
Accumulation is in FP32 in rank order before rounding to BF16; it is not
bitwise identical to NCCL's reduction tree.

Before Engram and at the final layer, the original EP combine is retained.
Direct dispatch additionally requires an even row count. Empty or uneven
rank layouts use the standard modular all-gather/reduce-scatter path, with
every rank participating. No request-local gate skips a required collective.

Performance depends on the per-rank row count and context distribution. This
does not change the default backend. Two-rank timings do not predict a four-
or eight-rank speedup: compare against `auto` with the same DP/EP width.
In benchmark tables, C2/C4/C8 denote requests **per DP rank**, not total
concurrency. Total concurrency is the per-rank count multiplied by the DP size.

## Experimental C8 attention

`dep_wide_c8_attention: true` in `additional_config` enables an experimental
64-head sparse attention kernel for the target's 48-row, 640-column decode
case. It is off by default and allocates about 30 MiB of fixed scratch per
rank. Other shapes, the draft, and unsupported layouts retain their existing
dispatch. Component correctness has been exercised through 384 rows, but the
larger shapes regressed and are not selected. This does not increase the
executor's 128-row limit. Full-model validation is still required before
deployment.
