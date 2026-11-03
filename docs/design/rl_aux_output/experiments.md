# Aux Output Experiments

End-to-end measurements behind the [protocol](protocol.md). The designs are described in [design space](design_space.md).

## Experiment matrix

| ID | What | Pass criterion |
|---|---|---|
| X0 | Model output sanity; KV token capacity and bytes per token; where an R3 KV layer can attach | Correct, diverse outputs; real capacity numbers |
| X1 | Correctness per design: multi-turn agent, GRPO and forks; delivered rows vs inline rows | 0 missing, 0 wrong |
| X1b | Row determinism across schedules: the same prompt alone, as 8 copies, mixed with long prompts, decode vs re-prefill | Bitwise identical |
| X2 | Throughput with aux outputs on | Measured per design |
| X3 | Host memory and pool peaks | Extrapolated to frontier scale |
| X4 | Requested means delivered: small pool, continuous sessions | 0 missing, no deadlock |
| X5 | Freed once used: release latency, leaks at the end | 0 leaks |
| X6 | Indexer compression on real selections | Ratio and codec speed |
| X7 | Hidden states: recomputed logits vs vLLM, fp8 error on full-vocab KL | Choose bf16 or fp8 |
| X9 | Trainer-side logprob recompute with R3 replay from the connector (forward only) | Rows aligned; replay effect measured |

## Status snapshot

| ID | c1 | c0 | P-S | P-B | P-A | Shared |
|---|---|---|---|---|---|---|
| X0 | | | | | | pass |
| X1 | pass | pass in X4 (not rerun as X1) | pass | pass | pass (hashes equal to c1) | |
| X1b | | | | | | `auto` pass; `deep_gemm_mega_moe` fail, fixed |
| X4/X5 | pass | pass | pass | — | — | |
| X6 | | | | | | pass |
| X7 | | | | | | pass |
| X9 | | | | | | pass (alignment verified) |
| Indexer/hidden through the delivery path | | | pass | | | |

## Setup

- Model: DeepSeek-V4.1-Flash (40 layers + 3 MTP, 384 experts with top-6, 8 indexer source layers with top-512), public config.
- vLLM at commit `da8922bf0f`, Model Runner V2, one node with 4× GB200, TP4 + EP4.
- Batch invariance on: `VLLM_BATCH_INVARIANT=1`, with `VLLM_BI_MXFP8_GEMM=tensor_core`, `VLLM_BI_MXFP8_PREQUANT=1`, `VLLM_BI_BF16_GEMM=tensor_core`.
- Engine arguments: `kv_cache_dtype=nvfp4_ds_mla`, `attention_backend=FLASHMLA_SPARSE_DSV41`, prefix caching on. MoE backend `auto` (resolves to `FLASHINFER_TRTLLM_MXFP4_MXFP8`) unless noted.
- Store: one Mooncake master plus one segment holder on the same node.
- Delivery: block-wise writes keyed by `<job>/<session>/<block_hash>`, a tail and a manifest at request end, and release by session prefix. The delivery path was implemented on top of the block-keyed aux output connector (#45635).
- Workloads (X1):
  - multi-turn agent, 16 sessions × 4 turns;
  - GRPO, 4 groups × 8;
  - forks: 8 parents and 8 children.

  That is 112 requests in total. Each request is checked three ways:
  - **delivered**: rows read back through the manifest;
  - **inline**: rows returned by the engine;
  - **reference**: a no-cache full prefill.

## Designs compared

All designs share the delivery path. They differ in where rows for prefix-cache hits come from:

| Design | Rows for prefix-cache hits come from |
|---|---|
| c1 | an engine-local block-row cache, allocated and evicted with the KV; blocks missing from the session are written at request end |
| c0 | nowhere: prefix hits are capped at `aux_start` and recomputed |
| P-S | the delivery pool doubles as the session's cache: hits are capped at the longest prefix this session already published |
| P-B | R3 is an extra KV-cache layer (RFC #64383) |
| P-A | Mooncake keyed by the global block hash (#57968), keys-only output |

## Results

### Capacity

- KV capacity of a TP4 engine at 0.65 GPU memory utilization: 3.39M tokens.
- A block-row cache sized to the KV:
  - R3: ~1.6 GB per engine (~1% of the KV);
  - indexer: ~54 GB raw, ~10 GB compressed.
- P-B (R3 as a KV layer) reduced KV capacity by 0.21%.

### Correctness (X1)

With the "first computation wins" rule:

| Design | Missing rows | Delivered = inline | Left after release |
|---|---|---|---|
| c1 | 0 | 112/112 | 0 |
| P-S | 0 | 112/112 | 0 |
| P-B | 0 | 112/112 | 0 |
| P-A | 0 | 112/112 (R3 row hashes equal to c1 for all 112 requests) | 0 |

Without that rule, c0, P-B and c1 each had a few requests where 1–177 rows differed. The rows were exactly `[start of the last unfinished block, aux_start)`, re-prefilled by the next turn after a prefix hit.

A full prefill from position 0 matched decode bitwise in these tests. A prefill resuming after a cache hit did not always match. For the same reason, generated tokens can differ between designs (41/112 between P-S and c1): each design changes the hit pattern, and the output depends on where a prefill resumes. Within each run, delivery was exact.

### Always delivered, freed after use (X4/X5)

The workload was 96 sessions × 4 turns. The trainer reads each finished session and then releases it.

| Design | Pool | In-flight cap | Missing | Wrong | Leaked keys | Release p50 |
|---|---|---|---|---|---|---|
| c1 | 64 MB | 46 sessions | 0 | 0 | 0 | 0.4 ms |
| c0 | 64 MB | 46 sessions | 0 | 0 | 0 | 0.4 ms |
| P-S | 64 MB | 46 sessions | 0 | 0 | 0 | 0.3 ms |
| P-S | 16 MB | 11 sessions | 0 | 0 | 0 | 0.2 ms |

The in-flight cap kept the pool from ever filling, so backpressure never triggered.

### Throughput

Output tokens/s for X1 (agent / GRPO / fork):

| Design | Agent | GRPO | Fork |
|---|---|---|---|
| c1 | 654 | 474 | 321 |
| P-S | 624 | 428 | 348 |
| P-B | 675 | 394 | 375 |
| P-A | 662 | 434 | 353 |

The workload is small, and the differences are within run-to-run noise.

### Indexer replay (X6)

- Selections are bitwise identical across runs under batch invariance.
- Raw size: 8 layers × 512 × int32 = 16 KB/token.

| Encoding | Size per token | Ratio |
|---|---|---|
| 20-bit packing | 10 KB | 1.6× |
| zstd-3 | — | 4.5–6.1× |
| 20-bit packing + zstd-3 | 2.5–3.6 KB | 4.6–6.7× |

- Codec speed: 300–380 MB/s per core.
- Delta encoding on top of zstd does not help.

### Indexer and hidden states through the delivery path

Indexer selections and hidden states were added to the same block-wise path as R3, with the P-S design on 112 requests:
- indexer: one object per full block, 20-bit packing + zstd, same deduplication and "first computation wins" rule as R3;
- hidden states: one object per 64 generated positions, fp8 e4m3 with a per-token scale;
- the manifest describes every field.

| Output | Check | Result | Bytes stored per token |
|---|---|---|---|
| R3 | delivered vs inline rows | 112/112 equal, 0 missing | 480 (raw) |
| Indexer | delivered vs an independent in-engine capture of the same requests | 112/112 bitwise equal, 0 missing, 0 holes | 551 (16,384 raw) |
| Hidden | delivered vs the captured hidden state | every row is the exact fp8 requantization | 5,124 (10,240 as bf16) |

Stored indexer bytes are far below the per-token compressed size (2.5–3.6 KB) because blocks are written once per session.

Full-vocabulary check for hidden states: the delivered fp8 hidden state times `lm_head`, compared with vLLM's distribution.

| Metric | Mean | Max |
|---|---|---|
| KL | 7.7e-4 | 0.012 |
| Logprob difference | 0.019 | 0.24 |

All objects were removed by session release, with 0 keys left.

**Throughput.** Total output tokens/s over 4 alternating passes per arm (first pass excluded as warm-up):

| Arm | Median | Range |
|---|---|---|
| R3 only | 552 | 520–565 |
| R3 + indexer + hidden | 495 | 443–533 |

That is ~10% lower. The ranges overlap on this small workload, and the split between compression, device-to-host copies and puts has not been measured.

### Hidden states (X7)

| Comparison | Result |
|---|---|
| fp32 `lm_head(hidden)` vs vLLM logits | max abs diff 2.1e-4; not bitwise |
| bf16 `lm_head(hidden)` vs vLLM logits | max abs diff 0.062 |
| fp8 e4m3 with a per-token scale | full-vocabulary KL mean 4.4e-4 / 8.7e-4, max 1.9e-3 / 2.8e-3; argmax agreement 98–100% |

### Trainer-side logprob recompute with replay (X9)

R3 rows for 16 sessions were read back from the connector and replayed in a BF16 trainer (TP1/EP16), forward only, with no optimizer and no gradients.

- **Alignment check:** the fraction of tokens whose expert set differs between trainer and rollout is:
  - 2% at layer 0 when aligned;
  - ~100% when shifted by ±1 position or compared against the wrong layer.
- **Real routing gap:** quantized rollout vs BF16 trainer grows with depth: 16% at layer 1, 40–85% in deep layers.
- **Effect of replay on trainer logprobs:**

| | Mean abs diff | Max abs diff |
|---|---|---|
| Replay off | 0.175 | 8.15 |
| Replay on | 0.082 | 3.09 |

## Finding: `deep_gemm_mega_moe` broke batch invariance

With `moe_backend="deep_gemm_mega_moe"`, the same prompt run alone vs batched gave different R3 rows and different greedy tokens. The backend enables MoE sequence parallelism: each TP rank owns a 1/TP slice of the step's tokens, and attention partial sums are combined with a reduce-scatter. On CUDA under `VLLM_BATCH_INVARIANT=1`, the batch-invariant branch of `CudaCommunicator.reduce_scatter` is gated on `current_platform.is_rocm()`, so the call falls through to a plain NCCL ring reduce-scatter, whose summation order depends on which rank owns the output slice. A token's attention output therefore depends on which slice it lands in. Per-layer instrumentation confirms it: the four per-rank inputs to the layer-0 post-attention reduce-scatter are bitwise equal while its output is not; growing the batch with the probe token on the same rank changes nothing. Dropping the `is_rocm()` condition (so CUDA uses `reduce_scatter_batch_invariant`) makes alone / 8 copies / mixed batches bitwise identical in R3 rows, greedy tokens and logprobs. The cost of the extra all-to-all is not measured; any configuration that enables MoE sequence parallelism (e.g. DP>1 with TP>1 and EP) should be affected.

## Bugs found while building the delivery path

- **Manifest deferred to the next engine step:** an idle offline engine never wrote it. Write it when the request finishes.
- **Keys built from the engine-internal request id:** the id has a random suffix, so the caller could not find the objects. Use the caller-visible id.
- **Duplicate puts from every TP rank retried forever:** treat "key already exists" as success.
- **One extra row:** async scheduling runs a forward for the last sampled token. Readers truncate to `L−1`.
- **Replay hook bypassed in the trainer:** some MoE implementations call `route` directly, bypassing a router-level replay hook. Hook replay at the call that computes routing.
