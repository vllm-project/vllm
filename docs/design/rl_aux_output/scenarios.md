# Sizing Scenarios

Two scenarios bound the design: a single-node setup and a frontier-scale agentic RL run. All numbers assume batch invariance (BI) is on. Sizes come from model configs and formulas; assumptions are marked. The last section replaces the per-token sizes with measurements on DeepSeek-V4.1-Flash.

## 1. Scenarios

| | Single node | Frontier scale |
|---|---|---|
| Model | Qwen3-30B-A3B | a ~500B-A23B MoE (49 MoE layers, 256 experts, 1M context) |
| Workload | GRPO, short multi-turn agents | long-horizon agents with sub-agent forks |
| Mean / max context | 16K / 32K (assumed) | 200K (assumed) / 1M |
| Share of generated tokens | 50% (assumed) | 10%; multi-turn agents are prefill-dominated (assumed) |
| Trajectories in flight (cluster) | 256 (assumed) | ≈110,000 (public report) |
| In flight per replica | 32 (8 engines) | ≈51 (110,000 ÷ 2,160 replicas) |
| Staleness | 0 to a few steps | up to ~107 weight versions (~1 day) |
| Weight-sync interval | every step | ≈807 s (86,400 s ÷ 107) |
| Inference | 1 node × 8 H200, 8 engines (TP1) | ≈2,160 GB300 nodes, one replica each (assumed) |
| Training | colocated | ≈465 GB300 nodes (inference : training ≈ 4.65 : 1) |
| Completion rate | — | ≈41 trajectories/s |

By Little's law, a frontier-scale trajectory lives ≈110,000 ÷ 41.3 ≈ 2,660 s and spans about 3 weight updates.

## 2. Bytes per position and per trajectory

| Output | Positions recorded | Per position | 16K trajectory | 200K trajectory | 1M trajectory |
|---|---|---|---|---|---|
| R3 | all | 384 B / 392 B (MoE layers × top-k × 1 B) | 6.3 MB | 80 MB | 411 MB |
| top-k 256 | generated only | 2 KB (256 × (int32 id + fp32 logprob)) | 17 MB | 42 MB | 215 MB |
| top-p mask (≤ 256 ids) | generated only | ≤ 1 KB | 8.4 MB | 21 MB | 107 MB |
| indexer, 12 layers × 2048 | all | 96 KiB | 1.6 GB | 20 GB | 103 GB |
| indexer, 61 layers × 2048 | all | 488 KiB | 8.2 GB | 102 GB | 524 GB |

The two indexer rows assume a model of the same size with a DSA-style indexer. Bounding the top-p mask at 256 ids requires `top_k ≤ 256` at sampling time.

## 3. Cache segment as a KV layer: host memory

Only outputs recorded at every position need a cache segment (R3, indexer); top-k and mask cover generated positions, which are never prefix hits. The cache segment's token capacity equals the full-attention KV capacity.

| Output | Single node, per engine (589K tokens) | Single node, per node (8 engines) | Frontier, per replica (12.8M tokens) | Ratio to KV (single node / frontier) |
|---|---|---|---|---|
| R3 | 0.23 GB | 1.8 GB | 5.0 GB | 0.39% / 1.06% |
| indexer, 12 layers | 58 GB | 463 GB | 1.26 TB | 100% / 267% |
| indexer, 61 layers | 294 GB | 2.35 TB | 6.4 TB | 508% / 1356% |

At frontier scale each replica has ≈51 × 200K ≈ 10.4M live tokens, 81% of its 12.8M capacity. R3's cache segment is cheap; an uncompressed indexer cache segment exceeds host memory.

## 4. Delivery pool

Pool size = per-node in-flight bytes ÷ 0.75 (in-flight cap at 75%).

| Output | Single node: in flight | Single node: pool | Frontier: per node in flight | Frontier: cluster in flight | Frontier: per-node pool (512K start / 1M bound) |
|---|---|---|---|---|---|
| R3 | 1.6 GB | 4.3 GB | 4.1 GB | 8.8 TB | 14 / 28 GB |
| top-k 256 | 4.3 GB | 11.5 GB | 2.1 GB | 4.6 TB | 7 / 15 GB |
| top-p mask | 2.1 GB | 5.6 GB | 1.1 GB | 2.3 TB | 3.6 / 7.3 GB |
| indexer, 12 layers | 206 GB | 1.1 TB | 1.0 TB | 2.2 PB | 3.5 / 7.0 TB |
| indexer, 61 layers | 1.0 TB | 5.6 TB | 5.2 TB | 11.3 PB | 18 / 36 TB |

Deduplication across turns is mandatory at frontier scale. Rewriting the full context every turn would write ~2 GB of R3 for a 200K, 50-turn trajectory, 25× the deduplicated 80 MB.

## 5. Write traffic and object counts (frontier scale)

| Output | Cluster bandwidth | Per inference node | Object (16-position block) | Live objects (cluster) |
|---|---|---|---|---|
| R3 | 3.3 GB/s | 1.5 MB/s | 6.3 KB | 1.4B |
| top-k 256 | 1.7 GB/s | 0.8 MB/s | 33 KB | 141M |
| top-p mask | 0.87 GB/s | 0.4 MB/s | 16 KB | 141M |
| indexer, 12 layers | 831 GB/s | 385 MB/s | 1.6 MB | 1.4B |
| indexer, 61 layers | 4.2 TB/s | 1.96 GB/s | 8.0 MB | 1.4B |

Bandwidth is not the constraint for R3, top-k or mask. Object count is:
- 16-position objects mean ~530K objects/s, more than a single Mooncake master handles unbatched (54K objects/s measured).
- Larger blocks (64–256 tokens), one object per block holding every output kind, and batched puts (337K objects/s with batches of 16) bring this within budget.

## 6. Training side: reads per step

A frontier-scale step consumes ≈33K trajectories (41/s × 807 s) across ≈465 training nodes. With pipeline parallelism, each stage reads only its own layers of R3 and indexer; top-k and mask are read by the last stage only.

| Output | Single node: per step | Frontier: per training node per step |
|---|---|---|
| R3 | 3.2 GB | 5.8 GB |
| top-k 256 | 8.6 GB | 3.0 GB |
| top-p mask | 4.3 GB | 1.5 GB |
| indexer, 12 layers | 825 GB | 1.4 TB |
| indexer, 61 layers | 4.2 TB | 7.3 TB |

## 7. Recomputing in-flight work at weight sync

```
per-replica in-flight tokens × 2 × 23B active params / (4 GPUs × 5 PFLOPS FP8 × 40% MFU)
= 51 × 200K × 46 GFLOP / 8 PFLOPS ≈ 60 s
```

That is ≈7.4% of each 807 s interval (MLP only; attention adds more). Frontier-scale runs therefore keep generating across weight versions and reuse older KV rather than recomputing in-flight requests. Prefix hits on blocks computed by an older version are normal.

## 8. Constraints on the design

- R3 and top-k are the same order of magnitude (2–4 GB in flight per node, 7–28 GB pools) and can share one delivery protocol. Top-k needs no cache segment.
- A cache segment in a KV layer is affordable for R3, but not for an uncompressed indexer.
- The indexer needs its own treatment: compression, a subset of layers or positions, or an SSD tier.
- Deduplication is mandatory at frontier scale, and release must be per trajectory: trajectories span many weight versions, so deleting by generation is not possible.
- With BI on, row differences between computations come only from weight-version changes, except for the cases in the [experiments](experiments.md). Rows must stay tied to the KV that produced them.

## 9. Scenario coverage vs. existing systems

Compared with the SGLang output store and Miles (which reject several of these configurations at startup), the protocol explicitly handles:
- multimodal prompts: positions counted on the engine-side expanded sequence, and image hashes included in the prefix hash;
- multiple policies or LoRAs sharing an engine: a new session per policy and in-flight caps per policy;
- streaming responses: references are returned in the final chunk;
- chat-message APIs: the engine-side `prompt_token_ids` are returned;
- tool-observation positions: explicit fill values, so an empty support set never yields NaN losses;
- training-step retry and checkpoint rollback: release only after the step commits, and regenerate after a rollback;
- offline saving of rollouts: read inline first;
- MoE backends that bypass top-k: reject R3 at startup;
- routers that drop unknown fields: a missing `aux` is an error;
- requests aborted while queued: they produce an empty set of rows;
- leak checks at job end: a sweep that reports the count.

## 10. DeepSeek-V4.1-Flash, measured

Model (public config):
- 40 layers + 3 MTP layers;
- 384 experts, top-6;
- 8 indexer source layers, top-512.

Setup: TP4 + EP4, BI on. The frontier column keeps the workload from section 1 and swaps in DeepSeek-V4.1-Flash per-token sizes.

| Output | Positions | Per position | 200K trajectory | Frontier: per replica in flight (≈10.4M positions) | Frontier: per-replica pool (÷0.75) |
|---|---|---|---|---|---|
| R3 | all | 480 B (40 × 6 × uint16) | 96 MB | 5.0 GB | 6.7 GB |
| top-k 256 | generated | 2 KB | 41 MB | 2.1 GB | 2.8 GB |
| hidden (`lm_head` input, fp8 + per-token scale) | generated | ≈5 KB (hidden 5120) | 102 MB | 5.2 GB | 7.0 GB |
| indexer, raw (8 × 512 × int32) | all | 16 KB | 3.3 GB | 170 GB | 227 GB |
| indexer, 20-bit + zstd (measured) | all | 2.5–3.6 KB | 0.5–0.7 GB | 26–37 GB | 35–50 GB |

- **KV capacity, measured.** A TP4 engine at 0.65 GPU memory utilization holds 3.39M tokens.
  - R3 as a block-row cache (or KV layer): ≈1.6 GB per engine, ~1% of the KV.
  - Indexer: ≈54 GB raw, ≈10 GB compressed.
- **Indexer compression.** Selections are bitwise reproducible under BI. Compressed, the indexer is 5–7× R3, and fits in the same delivery pool.
- **Hidden states.** fp8 hidden times the teacher's `lm_head` is within full-vocabulary KL < 1e-3 of vLLM's distribution. The behavior logprob must still come from the inline per-token logprob, because recomputed logits are not bitwise equal.
