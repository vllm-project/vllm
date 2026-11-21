# Sizing Scenarios

This page sizes the products the protocol delivers: what each granularity shares and stores in the common RL patterns, how large the products are, how fast they are written and read, and where they fit. Two scenarios bound the numbers: a single-node setup and a frontier-scale agentic RL run. Sizes come from model configs and formulas; assumptions are marked.

A request stores the products of the positions it computes ([protocol §5](protocol.md#5-objects-and-writes)), and a later request of its namespace reuses the objects of the positions it hits. Per-trajectory and per-pool numbers therefore scale with positions, not with generated tokens. Every enabled field is stored at every position.

## 1. Scenarios

Staleness is how many weight versions a trajectory can fall behind the current weights. The frontier-scale numbers come from a public report; the replica count is assumed.

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

By Little's law a frontier-scale trajectory lives ≈110,000 ÷ 41.3 ≈ 2,660 s and spans about 3 weight updates, so later turns routinely reuse KV computed at earlier versions. No name carries the version, so this costs nothing in names or objects.

## 2. Bytes per position and per trajectory

The fields: R3 holds the routed expert ids of every MoE layer; top-k logprobs use k = 128; the indexer holds the positions a DSA-style sparse attention selects in each layer; hidden states are the last layer's output.

For the frontier-scale model, assumed to have a DSA-style indexer:

| Output | Per position | 16K trajectory | 200K trajectory | 1M trajectory |
| --- | --- | --- | --- | --- |
| R3 (49 MoE layers × top-8 × 1 B) | 392 B | 6.4 MB | 80 MB | 411 MB |
| top-k logprobs, k = 128 (int32 id + fp32 logprob) | 1 KB | 17 MB | 210 MB | 1.07 GB |
| indexer, 12 layers × 2048, int32 | 96 KiB | 1.6 GB | 20 GB | 103 GB |
| indexer, 61 layers × 2048, int32 | 488 KiB | 8.2 GB | 102 GB | 524 GB |
| hidden states | `hidden_size` × dtype bytes | | | |

R3 also stores a zero entry for every dense layer, 8 B per position each. Hidden states take 10 KB per position at `hidden_size` 5120 in bfloat16.

For DeepSeek-V4.1-Flash (public config: 40 layers, all MoE, plus 3 MTP layers; 384 experts, top-6; 8 indexer source layers, top-512; `hidden_size` 5120). The last column is one frontier-scale replica's live positions, 51 trajectories × 200K ≈ 10.4M:

| Output | Per position | 200K trajectory | Per replica in flight (≈10.4M positions) |
| --- | --- | --- | --- |
| R3 (40 × 6 × uint16) | 480 B | 96 MB | 5.0 GB |
| top-k 128 | 1 KB | 205 MB | 10.6 GB |
| hidden states, bfloat16 | 10 KB | 2.0 GB | 107 GB |
| indexer, int32 (8 × 512) | 16 KB | 3.3 GB | 170 GB |

R3, top-128 logprobs and hidden states together take 11,744 B per position, 2.35 GB per 200K positions; R3 alone 96 MB.

Reuse across turns keeps these numbers per trajectory rather than per turn: a turn writes only positions no earlier turn of the trajectory wrote, apart from the block holding the previous turn's tail (section 3). Rewriting the full context every turn would write about 2 GB of R3 for a 200K, 50-turn trajectory, 25× the 80 MB above.

## 3. What each granularity shares and stores

| Pattern | Group namespace | Trajectory namespace |
| --- | --- | --- |
| GRPO group of `n` samples of one prompt | A sample reuses the prompt blocks another sample has already scheduled on the same engine rank when it starts, and computes the rest. Samples sent at once to different replicas or data-parallel ranks, or started during another's prefill, each compute the prompt but write the same names, so the store keeps one copy of every full prompt block and all but one of them list another sample's computation. Sending one sample first and the others once its prefill has finished computes the prompt once. The block holding the last prompt position also holds generated tokens and is stored per sample. | Each sample computes and stores the prompt in its own namespace: `n` copies. |
| Multi-turn | The next turn hits up to the last full block of the previous turn and computes the previous tail's block again. That block is stored twice: as the previous turn's tail and as the next turn's full object. | The same. |
| Fork, sub-agent | Stays in the group's namespace and hits the KV of the prefix it shares. | Stays in its parent's namespace and hits the parent's KV, or gets a nested namespace and computes the shared prefix again. |
| Retry | Hits what the failed request cached and lists the same names for it. Its last prompt position is computed again and listed: the trajectory takes the retry's own products for it, unless the prompt ends on a block boundary, where the listing names the block stored earlier ([protocol §12](protocol.md#12-limits)). | The same, within the sample's namespace. |
| Failed request, no response | Its objects stay until the group is released. | Its objects stay until the sample is released. |
| Weight update, prefix cache kept | Later turns hit KV of earlier versions. | The same. |
| Weight update, prefix cache reset | Later turns compute their prefix again under the same names; the puts keep the first copies, so nothing new is stored. A sample that computes the group's prompt again lists the copy another sample stored. | Later turns compute their prefix again under the same names; nothing new is stored. |

**Prompt copies in numbers.** A group of 8 samples whose shared prompt is 200K positions, on DeepSeek-V4.1-Flash: a group namespace stores the prompt once, 96 MB with R3 only and 2.35 GB with R3, top-128 logprobs and hidden states. Trajectory namespaces store it 8 times: 768 MB and 18.8 GB.

**Lifetime.** A group namespace holds a group's objects until every sample of the group is settled and a pause follows, so the slowest sample sets the lifetime. A trajectory namespace is released when its sample is settled.

## 4. Object pool

Objects stay until the trainer releases their namespace ([protocol §8](protocol.md#8-pause-and-release)), on one of the store deployments of [protocol §11](protocol.md#11-deployment).

Objects alive at the same time:

| Output | Single node: in flight | Frontier: per replica in flight | Frontier: cluster in flight |
| --- | --- | --- | --- |
| R3 | 1.6 GB | 4.1 GB | 8.8 TB |
| top-k 128 | 4.2 GB | 11 GB | 23 TB |
| indexer, 12 layers | — | 1.0 TB | 2.2 PB |
| indexer, 61 layers | — | 5.2 TB | 11.3 PB |

The single-node model has no indexer. Tails add one object per request, about half a block of positions on average. On top of this, every turn stores one extra block (section 3); group namespaces hold groups waiting for their slowest sample, and trajectory namespaces hold a group's prompt once per sample.

### 4.1 SSD-first store (default)

Objects are not pinned. Each holder's memory segment is a cache in front of a local NVMe directory: when memory is full, objects are offloaded on eviction, and reads find them on SSD without the reader knowing. Two consequences size the SSD:

- **SSD is planned by bytes written in the job, not by bytes in flight.** On Mooncake 0.3.14 a removal deletes only metadata; bucket files on SSD are neither reclaimed nor reused, so each holder's disk use is the total written to it in the job. Removing half of the objects and writing the same amount again grew disk use from 6.28 to 9.42 GB; releasing namespaces and continuing to write grew it from 4.52 to 8.39 GB. Each holder uses a new directory per job, and deleting it at the end of the job is the only point where disk comes back. Section 5 gives the bytes written per hour.
- **Requests are admitted by free disk.** Every open namespace is charged the bytes its responses listed, each key once, plus the worst case of each request in flight: the positions from one block before its `aux_prompt_start` to its longest response, times bytes per position, plus headers and a tail. Before each request, the driver checks on every holder that `free disk − memory segment − reserve ≥ charges + the request's worst case`. A holder that fails the check because its disk filled stays failed for the rest of the job: read and release the admitted namespaces, then continue in new directories under a new key prefix. Disk eviction must be off, since it deletes unreleased objects by bucket age.

A unit test of the Mooncake backend with a 1 GB memory segment and a 4 s burst of 6 GiB of writes, with objects not pinned, SSD offload on and bounded retries, wrote everything in 116 s and read it all back with 0 missing. Reads from SSD took 0.99 ms at p50 and 4.07 ms at p99.

### 4.2 Trajectories per node

How many 200K-token trajectories of DeepSeek-V4.1-Flash one node's SSD holds, at 11.8 TB of usable SSD (14 TiB less a 20% reserve and a local copy of the weights):

| Fields stored | Per position | One 200K trajectory | Trajectories in 11.8 TB |
| --- | --- | --- | --- |
| R3 (40 layers × top-6 × uint16) | 480 B | 96 MB | ≈120K |
| R3 + top-128 logprobs | 1,504 B | 301 MB | ≈39K |
| R3 + top-128 logprobs + hidden states (bfloat16) | 11,744 B | 2.35 GB | ≈5K |

Top-k takes 128 × 8 B per position (an `int32` id and a `float32` logprob); hidden states take 5,120 × 2 B. While Mooncake does not reclaim disk on removal, these are the trajectories the job can write in total, not the trajectories alive at once.

### 4.3 When writes fail

A put the store refuses, because memory is full or the store reports an error, is retried every 0.2 s. When no put has succeeded for the stall deadline (half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, whose default is 300 s, so 150 s by default), the writer drops the objects it cannot write. Their keys stay listed in responses, the trainer masks the samples that need them, and the engine keeps running; the first successful put ends the dropping ([protocol §10](protocol.md#10-failures)).

### 4.4 Memory-only store

Without SSD, objects are hard-pinned in memory and never evicted, so the pool holds every object written between two releases. One rule sets both admission and the pool size:

- **Admission.** The driver admits a request only while the charges of the open namespaces, counted as in section 4.1, plus the request's worst case stay at or below 0.75 × pool.
- **Pool size.** Every request that writes between two releases has to be admitted, so the pool is at least 1/0.75 ≈ 1.33 × the bytes written between two releases.

How much of the pool the writes actually fill was measured, with an earlier implementation, by setting the pool against the bytes each run wrote:

| Pool ÷ bytes written | Model | Result |
| --- | --- | --- |
| 1.2× | — | passed |
| 1.095× | — | passed |
| 0.955× | DeepSeek-V4 | pool full at 89.6% of the writes |
| 0.943× | gpt-oss | pool full at 97.2% of the writes |

A pool slightly larger than the bytes written held them, and a smaller one filled. A pool sized by the rule above is 1.33 × an upper-bound estimate, so it is above these measured needs.

For comparison, Miles' output store sizes its capacity in advance at (maximum staleness + 1) batches × bytes per batch and pins it, without admission. Its measured peak was 7.17 GB against 7.0 GB from the formula.

### 4.5 Store issues that affect capacity

Three Mooncake 0.3.14 issues affect capacity and failures: removed objects that reappear when a removal overlaps an offload to SSD, a dead holder's segments kept registered for `client_ttl`, and metadata lost when the master restarts. They are described, with their handling, in [protocol §10](protocol.md#10-failures).

## 5. Object counts and write rate (frontier scale)

The cluster computes about 8.4M new positions per second (41 trajectories/s × 200K).

| Output | Cluster bandwidth | Per inference node | Per inference node, per hour |
| --- | --- | --- | --- |
| R3 | 3.3 GB/s | 1.5 MB/s | 5.4 GB |
| top-k 128 | 8.6 GB/s | 4.0 MB/s | 14 GB |
| indexer, 12 layers | 831 GB/s | 385 MB/s | 1.4 TB |
| indexer, 61 layers | 4.2 TB/s | 1.96 GB/s | 7.1 TB |

On the default SSD-first store, which frees no disk on removal, the holders together need the cluster bandwidth times the job's length of SSD; the hourly column is one inference node's share.

All enabled outputs of a block share one object, so the object count depends only on the block size `B` of the full-attention KV cache group.

| `B` | Objects/s, cluster | Objects/s, per replica | Live objects, cluster |
| --- | --- | --- | --- |
| 16 | ≈530K | ≈240 | 1.4B |
| 256 | ≈33K | ≈15 | ≈88M |

Plus one tail object per request, and the full object over each previous tail. Bandwidth is not the constraint for R3 or top-k; at small block sizes the object rate is. A single Mooncake master's rate at this scale has not been measured.

A replica queues at most `2 × max_num_seqs` batches. That bounds what a pause waits for, not how long: a slow or full store delays the pause until writes land, or until the stall deadline when none does.

## 6. Training side: reads per step

A frontier-scale step consumes ≈33K trajectories (41/s × 807 s) across ≈465 training nodes; each node reads the objects of its share. A group's shared prompt objects are read once per node under group granularity, and once per sample under trajectory granularity.

| Output | Single node: per step | Frontier: per training node per step |
| --- | --- | --- |
| R3 | 3.2 GB | 5.8 GB |
| top-k 128 | 8.6 GB | 15 GB |
| indexer, 12 layers | — | 1.4 TB |
| indexer, 61 layers | — | 7.3 TB |

## 7. Memory or SSD

For DeepSeek-V4.1-Flash with one copy of the objects per engine, on two node types:

| Node | Node A (measured) | Node B (assumed) |
| --- | --- | --- |
| GPUs | 4 | 8 |
| Host memory | 956.8 GiB | 2–4 TB |
| Local NVMe | 11–14 TiB | 30.7 TB |
| Engines per node | 1 | 2 |

A replica needs room for its live positions: 51 trajectories × 200K tokens ≈ 10.4M positions. The table gives how many times a medium holds that. About 30% more is taken by finished trajectories that the trainer has not read yet, so 1.3× is the least that is safe.

| Per replica in flight (10.4M positions) | R3 + top-k | Hidden states, bfloat16 | Indexer, int32 | All fields |
| --- | --- | --- | --- | --- |
| Per position | 1,504 B | 10 KB | 16 KB | ≈28 KB |
| Bytes | 15.6 GB | 107 GB | 170 GB | 293 GB |

| Medium (usable) | R3 + top-k | Hidden states | Indexer | All fields |
| --- | --- | --- | --- | --- |
| Node A memory (≈650 GB for the object pool) | 42× | 6.1× | 3.8× | 2.2× |
| Node B memory, 2 TB (≈656 GB per engine) | 42× | 6.1× | 3.9× | 2.2× |
| Node A SSD (≈11.8 TB) | — | 111× | 69× | 40× |
| Node B SSD (≈12 TB per engine) | — | 113× | 71× | 41× |

Usable memory is what is left for the object pool after the engine's resident memory and the OS. Usable SSD keeps a 20% reserve and a local copy of the weights.

- **R3 and top-k fit in memory.** 15.6 GB in flight, 42× the usable memory of an inference node.
- **Hidden states and the indexer need SSD.** In memory they fit only 4 to 6 times; on SSD 69 to 111 times.
- **Long runs are sized by bytes written.** Mooncake 0.3.14 does not free disk on removal (section 4.1), so a long run needs the job's total writes: about 0.5 TB per node per day for R3 + top-k, which fits; tens of TB per node per day for hidden states or the indexer, which needs a Mooncake release that reclaims removed objects.

## 8. Overhead

Overhead against products off, on DeepSeek-V4.1-Flash without batch invariance, the base scenario of the validation matrix, four runs on one node alternating products on and off ([experiments](experiments.md#group-and-trajectory-sharing-on-earlier-code)):

- **Group sharing** (earlier session-wise code, one GRPO group per session): −0.9% generated tokens per second, within the ±2.8% spread of the off runs.
- **Per-trajectory isolation** (request-wise prototype, every trajectory salted with its own owner): −11.2%, from lost prefix hits: each sample computes the group's prompt again, 14% more positions computed per second.
- **Earlier implementations, top scenario.** On node A without sleeps: +1.7% (521.9 s against 513.2 s over 8 arms; 51 sessions × 12 turns growing to 406K tokens, 106 version updates). On a second cluster: +1.1% without sleeps and +4.4% with a sleep at every update.

Generated tokens per second from single runs do not measure overhead: two products-off runs of the same build on the same node differed by 16%. The code of this protocol has not been measured yet.

## 9. Storage on the earlier protocols

An earlier session-wise stack and a request-wise prototype ran the same workload: DeepSeek-V4.1-Flash without batch invariance, one seed, 120 versions, keep and abort updates alternating, 6 to 12 turns per trajectory, 30% of trajectories consumed late.

| Metric | Session-wise, version segments | Request-wise prototype |
| --- | --- | --- |
| Stored objects, peak / median | 342.8K / 334.1K | 79.1K / 25.6K |
| Stored objects ÷ prediction (median) | 2.94× | 1.00× |
| SSD at the end | 613 GB | 211 GB |
| Objects written (estimated) | ≈411K | ≈373K |

The session-wise excess was objects of positions computed again under other version histories, written under new names and kept until their session was released. In the top scenario the keys in the store reached 4.5 times the keys predicted from the objects in flight, and 56 times with a sleep at every update.

**What changes with this protocol.** Without the version in the salt, a position computed again gets the same name and the store keeps the first copy, so that excess is not written. What remains is one extra block per turn, the prompt stored once per sample under trajectory granularity, and, under group granularity, the wait for a group's slowest sample. Not measured on this protocol yet.

## 10. Writing across clusters

Engines in one cluster can write objects to a store next to the trainer in another. One inference node over TCP is enough; across clusters, the limit is the total bandwidth between the two clusters.

**Measured transport.** One node wrote objects as an engine would, and another ran the store and the reader. The two shared one 400 Gb/s port, with RDMA over RoCE v2 and TCP over the same port. Mooncake 0.3.14, a new master and holder per protocol, 30 s per cell:

| Cross-node write | Concurrent writes | Write throughput | One write, p50 / p99 | Single-thread read on the store node, p50 |
| --- | --- | --- | --- | --- |
| RDMA, 2.7 MB objects | 8 | 45.7 GB/s | 0.36 / 0.46 ms | 0.16 ms |
| RDMA, 2.7 MB objects | 24 | 48.8 GB/s | 1.17 / 1.83 ms | 0.17 ms |
| RDMA, 25 MB objects | 8 | 48.9 GB/s | 3.95 / 4.14 ms | 0.75 ms |
| RDMA, 25 MB objects | 24 | 48.7 GB/s | 12.1 / 13.2 ms | 0.72 ms |
| TCP, 2.7 MB objects | 8 | 3.15 GB/s | 6.7 / 7.3 ms | 1.5 ms |
| TCP, 2.7 MB objects | 24 | 4.08 GB/s | 15.4 / 18.4 ms | 1.5 ms |
| TCP, 25 MB objects | 8 | 3.66 GB/s | 55 / 56 ms | 12.4 ms |
| TCP, 25 MB objects | 24 | 3.65 GB/s | 163 / 173 ms | 12.1 ms |

RDMA fills the port at about 49 GB/s. TCP over the same port reaches 3.2 to 4.1 GB/s, limited by the protocol stack and the CPU, not the link. Concurrency past 8 adds queueing, not throughput. The link was within one rack, with a round trip under 1 ms.

**How fast objects must be written.**

```text
required write rate    = new positions per second × object bytes per position
measured write rate    = bytes written per step ÷ step time
bytes written per step = trajectories × tokens per trajectory × bytes per token
```

Measured small workloads, on one node with Qwen3-30B-A3B and R3 only. The last column is the slowest single-node TCP cell, 3.15 GB/s, over the write rate:

| Workload | Derivation | Average write rate | Single-node TCP ÷ rate |
| --- | --- | --- | --- |
| W1, trainer integration, synchronous | 256 trajectories × ≈7K tokens × 384 B (48 layers × top-8 × uint8) ≈ 0.7 GB per step; 280 s per step | 2.5 MB/s | ≈1,260× |
| W2, trainer integration, fully asynchronous multi-turn | 9.1 GB over 30 steps; about 91 s per step | 3.3 MB/s | ≈950× |
| W1, Miles' output store, synchronous | the same workload with R3 as `int32` (1,536 B per token): 27.7 GB over 10 steps; about 254 s per step | ≈11 MB/s | ≈290× |

At frontier scale the cluster computes about 8.2M new positions per second on about 2,160 inference nodes. With DeepSeek-V4.1-Flash bytes per position:

| Fields stored | Cluster total | Per inference node | Single-node TCP ÷ per-node rate |
| --- | --- | --- | --- |
| R3 only (480 B) | 4.0 GB/s | 1.8 MB/s | ≈1,700× |
| R3 + top-128 logprobs (1,504 B) | 12.6 GB/s | 5.8 MB/s | ≈540× |
| All fields, with hidden states (11,744 B) | 98.6 GB/s | 46 MB/s | ≈69× |

- **One node is not the bottleneck at low round-trip times.** With all fields an inference node writes 46 MB/s on average, tens of times below one node's TCP rate.
- **A long round trip can be.** A writer that waits for one put of one object before the next, at a 100 ms wide-area round trip and objects of about 122 KB, writes about 1.2 MB/s, below the 1.8 MB/s a node writes with R3 only. Putting objects in batches avoids this; neither has been measured across a wide-area link.
- **Across clusters the limit is the total bandwidth between them.** R3 alone needs 4 GB/s across the link, with hidden states about 100 GB/s. Whether a link that wide exists between two clusters has not been measured.

Not verified: one holder serving TCP writers and RDMA readers at once. Reading the 0.7 GB of one W1 step took about 0.4 s over TCP and 0.06 s over RDMA.
