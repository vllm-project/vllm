# Sizing Scenarios

Two scenarios bound the design: a single-node setup and a frontier-scale agentic RL run. Sizes come from model configs and formulas; assumptions are marked. Section 9 gives per-position sizes for DeepSeek-V4.1-Flash.

A session request stores the rows of every position it computes, in full objects and one tail ([protocol §5](protocol.md#5-objects-and-writes)), and a later turn reuses the objects of the positions it hits. All per-trajectory and per-pool numbers below therefore scale with positions, not with generated tokens. Every enabled field is stored at every position.

## 1. Scenarios

| | Single node | Frontier scale |
| --- | --- | --- |
| Model | Qwen3-30B-A3B | a ~500B-A23B MoE (49 MoE layers, 256 experts, 1M context) |
| Workload | GRPO, short multi-turn agents | long-horizon agents with sub-agent forks |
| Mean / max context | 16K / 32K (assumed) | 200K (assumed) / 1M |
| Trajectories in flight (cluster) | 256 (assumed) | ≈110,000 (public report) |
| In flight per replica | 32 (8 engines) | ≈51 (110,000 ÷ 2,160 replicas) |
| Staleness | 0 to a few steps | up to ~107 weight versions (~1 day) |
| Weight-sync interval | every step | ≈807 s (86,400 s ÷ 107) |
| Inference | 1 node × 8 GPUs, 8 engines (TP1) | ≈2,160 nodes, one replica each (assumed) |
| Training | colocated | ≈465 nodes (inference : training ≈ 4.65 : 1) |
| Completion rate | — | ≈41 trajectories/s |

By Little's law a frontier-scale trajectory lives ≈110,000 ÷ 41.3 ≈ 2,660 s and spans about 3 weight updates. A session's segments grow by at most one entry per update it spans.

## 2. Bytes per position and per trajectory

| Output | Per position | 16K trajectory | 200K trajectory | 1M trajectory |
| --- | --- | --- | --- | --- |
| R3 (49 MoE layers × top-8 × 1 B) | 392 B | 6.4 MB | 80 MB | 411 MB |
| top-k logprobs, k = 128 (int32 id + fp32 logprob) | 1 KB | 17 MB | 210 MB | 1.07 GB |
| indexer, 12 layers × 2048, int32 | 96 KiB | 1.6 GB | 20 GB | 103 GB |
| indexer, 61 layers × 2048, int32 | 488 KiB | 8.2 GB | 102 GB | 524 GB |
| hidden states | `hidden_size` × dtype bytes | | | |

R3 also stores a zero row for every dense layer, 8 B per position each. The two indexer rows assume a model of the frontier size with a DSA-style indexer. Hidden states depend on the model: 10 KB per position at `hidden_size` 5120 in bfloat16, 20 KB in float32 (section 9).

Reuse across turns keeps these numbers per trajectory rather than per turn: a turn writes only positions no earlier turn of the session wrote under the same name. Rewriting the full context every turn would write about 2 GB of R3 for a 200K, 50-turn trajectory, 25× the 80 MB above.

## 3. KV tier capacity across weight updates

When GPU memory is released at a weight update, the GPU prefix cache is empty afterwards. A session continues on its stored KV only if a KV connector tier holds it, including the blocks that decoding completed. When the tier has lost it, the next turn computes the prefix again with the new weights and writes it again under new names ([protocol §6](protocol.md#6-hits-kv-tiers-and-weight-updates)). The names stay true, but the recompute costs time and the second copy costs pool space:

```text
tier capacity per replica ≥ live session positions per replica × KV bytes per position
```

At frontier scale each replica has ≈51 × 200K ≈ 10.4M live positions, 81% of an assumed GPU KV capacity of 12.8M positions. The tiers must hold all of it at every update to avoid recomputing it. Sessions that move between replicas need a KV store shared by the replicas, or the new replica computes the prefix again.

## 4. Object pool

Objects stay until the consumer releases them. A memory-only pool holds every object of every live session, plus headroom; on the default SSD-first store removal frees no disk, so the SSD holds every object written in the job. A session is released after a pause that follows its last request ([protocol §7](protocol.md#7-pause-and-release)), so in memory it occupies the pool for up to one weight-sync interval after it ends; on SSD its bytes stay until the job ends.

By default the pool is the SSD of the holders, with memory as a cache ([protocol §8](protocol.md#8-deployment)): removal does not free disk, so the free disk of the holders must hold every byte written in the job, not only the in-flight bytes below, and new sessions are admitted by free disk. A memory-only store hard-pins objects, and its memory must hold the in-flight bytes ÷ 0.75.

| Output | Single node: in flight | Frontier: per replica in flight | Frontier: cluster in flight |
| --- | --- | --- | --- |
| R3 | 1.6 GB | 4.1 GB | 8.8 TB |
| top-k 128 | 4.2 GB | 11 GB | 23 TB |
| indexer, 12 layers | — | 1.0 TB | 2.2 PB |
| indexer, 61 layers | — | 5.2 TB | 11.3 PB |

The single-node model has no indexer. Tails add one object per request, about half a block of positions on average. Prefixes computed again after a KV loss across a weight update (section 3) add a copy under new names, from the object that holds the first position computed again whose stated version is not current. In flight, R3 and top-k fit in host memory at frontier scale, and an uncompressed indexer or hidden states at every position need terabytes per replica; on SSD the bytes written grow with the job's length at the rates of section 5. Enable only the outputs the trainer uses.

## 5. Object counts and write rate (frontier scale)

The cluster computes about 8.4M new positions per second (41 trajectories/s × 200K).

| Output | Cluster bandwidth | Per inference node | Per inference node, per hour |
| --- | --- | --- | --- |
| R3 | 3.3 GB/s | 1.5 MB/s | 5.4 GB |
| top-k 128 | 8.6 GB/s | 4.0 MB/s | 14 GB |
| indexer, 12 layers | 831 GB/s | 385 MB/s | 1.4 TB |
| indexer, 61 layers | 4.2 TB/s | 1.96 GB/s | 7.1 TB |

On the default SSD-first store, which frees no disk on removal, the holders together need the cluster bandwidth times the job's length of SSD; the hourly column is one inference node's share.

All enabled outputs of a block share one object, so the object count depends only on the block size `B` of the full-attention KV cache group:

| `B` | Objects/s, cluster | Objects/s, per replica | Live objects, cluster |
| --- | --- | --- | --- |
| 16 | ≈530K | ≈240 | 1.4B |
| 256 | ≈33K | ≈15 | ≈88M |

Plus one tail object per request. Bandwidth is not the constraint for R3 or top-k; at small block sizes the object rate is. A single Mooncake master's rate at this scale has not been measured.

A replica queues at most `2 × max_num_seqs` batches. That bounds what a pause waits for, not how long: at the rates above the queue drains quickly, but a slow or full store delays the pause until writes land, or until the stall deadline when none does, after which each queued batch gets one attempt.

## 6. Training side: reads per step

A frontier-scale step consumes ≈33K trajectories (41/s × 807 s) across ≈465 training nodes; each node reads the objects of its share.

| Output | Single node: per step | Frontier: per training node per step |
| --- | --- | --- |
| R3 | 3.2 GB | 5.8 GB |
| top-k 128 | 8.6 GB | 15 GB |
| indexer, 12 layers | — | 1.4 TB |
| indexer, 61 layers | — | 7.3 TB |

## 7. Recomputing in-flight work at weight sync

```text
per-replica in-flight positions × 2 × 23B active params / (4 GPUs × 5 PFLOPS FP8 × 40% MFU)
= 51 × 200K × 46 GFLOP / 8 PFLOPS ≈ 60 s
```

That is ≈7.4% of each 807 s interval (MLP only; attention adds more). Frontier-scale runs therefore keep generating across weight versions on their stored KV rather than recomputing at every update, and hits on KV computed with an older version are normal. This is why versions are part of the names and why the client carries them between turns.

## 8. Constraints on the design

- **Versions must travel with the KV, everywhere.** A trajectory spans ~3 updates and may be ~107 versions stale. Its prefix KV survives updates only in connector tiers, on this replica or another, so the version has to be part of what every tier uses as a key: the block hash.
- **Reuse is mandatory and scoped to the session.** Rewriting the full context every turn writes ~25× more. Sharing across sessions would need reference counts for little gain.
- **KV tiers are sized for throughput.** Losing a live session's prefix KV recomputes it and, after a weight update, writes its objects a second time under new names (section 3).
- **The pool holds every object it must serve.** Objects stay until release. Size a memory pool with hard pins for the live objects, and an SSD-first pool, which frees no disk on removal, for the bytes written in the job at the rates of section 5 (section 4).
- **Object rate, not bandwidth, limits R3 and top-k** at small block sizes (section 5).

## 9. DeepSeek-V4.1-Flash

Model (public config): 40 layers + 3 MTP layers; 384 experts, top-6; 8 indexer source layers, top-512; `hidden_size` 5120.

| Output | Per position | 200K trajectory | Frontier: per replica in flight (≈10.4M positions) |
| --- | --- | --- | --- |
| R3 (40 × 6 × uint16) | 480 B | 96 MB | 5.0 GB |
| top-k 128 | 1 KB | 205 MB | 10.6 GB |
| hidden states, bfloat16 | 10 KB | 2.0 GB | 107 GB |
| indexer, int32 (8 × 512) | 16 KB | 3.3 GB | 170 GB |

- **KV capacity** (measured). A TP4 engine at 0.65 GPU memory utilization holds 3.39M positions, about a third of the frontier per-replica live positions. The rest must be in tiers at every update (section 3).
- **Indexer compression** (earlier implementation, not in the stack): 20-bit packing plus zstd gave 2.5–3.6 KB per position up to 128K context and about 5.6 KB at 512K. Selections spread over the whole sequence at long contexts, so zstd gains less there.
- **Hidden states** (earlier implementation): logprobs recomputed from them had an error of at most 7.8e-4, from the trainer's own matmul. The behavior logprob must still come from the inline sampled logprob.
