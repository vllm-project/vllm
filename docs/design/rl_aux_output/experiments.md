# Aux Output Experiments

Measurements behind the [protocol](protocol.md). The designs are described in [design space](design_space.md).

## Setup

- Model: DeepSeek-V4.1-Flash (40 layers + 3 MTP, 384 experts with top-6, 8 indexer source layers with top-512, 9 KV cache groups), public config.
- vLLM at commit `da8922bf0f`, Model Runner V2, TP4 + EP4 on 4× GB200 per replica.
- Batch invariance on: `VLLM_BATCH_INVARIANT=1`, `VLLM_BI_MXFP8_GEMM=tensor_core`, `VLLM_BI_MXFP8_PREQUANT=1`, `VLLM_BI_BF16_GEMM=tensor_core`. Note this is batch invariance, not the full train-inference consistent configuration of #62365.
- Engine: `kv_cache_dtype=nvfp4_ds_mla`, `attention_backend=FLASHMLA_SPARSE_DSV41`, prefix caching on, MoE backend `auto` unless noted.
- Store: Mooncake (one master; segments on the same node, or one shared pool over RDMA).
- Delivery path: block-wise writes keyed by `<job>/<session>/<name>`, tail and manifest at request end, release by session prefix; built on the block-keyed aux output connector (#45635).

## How correctness is checked

Each request is read back through its manifest and compared three ways:

- **Inline:** the rows the engine returned with the response.
- **Reference:** each trajectory's own first computation, dumped by the engine per step after the rejected speculative suffix is dropped. For a fork, positions before the fork point come from the parent. Rows of drafts the scheduler later truncated are skipped.
- **Version segments:** the manifest's segments must match the weight versions the trajectory saw.

The inline check is blind at prefix-hit positions: the engine reads those inline rows back from the same store, so "delivered = inline" holds even if both are another trajectory's rows. Errors in the hardening section below were only visible against the reference.

Engine events that do not occur on their own in a short test are triggered through the engine's real code paths:

- **Preemption:** one decode-time block allocation fails every 20 scheduler steps, so the scheduler takes its own KV-shortage branch (`running.pop()` + `_preempt_request`). Natural KV pressure does not preempt on this model: admission requires the whole prompt to fit.
- **In-flight weight update:** a thread calls `pause_scheduler("keep", clear_cache=True)` then `resume_scheduler` every 20 s while `generate` runs.
- **Write failures:** the writer fails one write in N.
- **Weight versions:** an in-place weight update after every step perturbs routers, norms and indexer weights; a probe confirms R3, indexer and hidden rows change between versions.

## Workloads

- X1: multi-turn agent (16 sessions × 4 turns), GRPO (4 groups × 8), forks (8 parents, 8 children): 112 requests.
- GRPO regression: 4 groups of 4 trajectories sharing a session, 8 turns, 64K context: 128 requests, 4.65M rows.
- Fork regression: trajectories 1–3 fork from trajectory 0 after its first turn (or after 2 more parent turns, "delayed"): 116 requests, 4.54M rows.
- Preemption regression: 12 sessions × 6 turns, 64K context: 72 requests, 2.71M rows.

## Experiment matrix

| ID | What | Pass criterion | Result |
|---|---|---|---|
| X0 | Output sanity; KV capacity; where an R3 KV layer can attach | real numbers | pass |
| X1 | Correctness per design | 0 missing, 0 wrong | all five designs pass |
| X1b | Row determinism across schedules | bitwise identical | `auto` pass; `deep_gemm_mega_moe` failed, fixed |
| X2 | Throughput with aux outputs on; engine-local store vs Mooncake at 8 replicas; each hop vs its ceiling | measured | as built about −41% at the top scenario; with partial capture, CUDA graph `FULL` and capture off the output lock: one node at 512K −2.2% (all outputs), top scenario −2.1% to −6.1% across runs (all outputs) |
| X4/X5 | Requested means delivered; freed once used | 0 missing, 0 leaks | pass down to a 16 MB pool |
| X6 | Indexer encoding | ratio, order, bytes | 4.6–6.7× up to 128K; rows restored in engine order; deep slots out of blocks: 51.2 → 14.1 GB per run |
| X7 | Hidden states: exact vs fp8 | logprob error, bytes, time | exact (`raw`) is the default: error max 7.8e-4 vs ~0.3 for fp8, no measurable time cost |
| X9 | Trainer logprob recompute with R3 and indexer replay | aligned rows; gap vs a self-perturbation floor | delivered = exact; remaining gap is trainer-engine numerics |
| E0 | KV offload to an NVMe tier | recompute, time, aux complete | −23% time and 9× less recompute with a 4 GB CPU tier; 0 missing |
| Scale | 512K, 108 turns, shared pool | 0 missing, 0 wrong | pass |
| Hardening | Preemption, GRPO, forks, delayed forks, in-flight weight updates, write failures, indexer gaps, writer memory, speculative decoding | 0 missing, 0 wrong vs the reference | pass (P-S) |

## Designs compared (X1)

| Design | Rows for prefix-cache hits come from |
|---|---|
| c1 | an engine-local block-row cache allocated and evicted with the KV |
| c0 | nowhere: hits capped at `aux_start` and recomputed |
| P-S | the delivery pool, as the session's cache |
| P-B | R3 as an extra KV-cache layer (#64383) |
| P-A | Mooncake keyed by the global block hash (#57968) |

| Design | Missing rows | Delivered = inline | Left after release |
|---|---|---|---|
| c1 | 0 | 112/112 | 0 |
| P-S | 0 | 112/112 | 0 |
| P-B | 0 | 112/112 | 0 |
| P-A | 0 | 112/112 (R3 row hashes equal to c1) | 0 |

Without "first computation wins", c0, P-B and c1 each had requests where 1–177 rows differed: exactly `[start of the last unfinished block, aux_start)`, re-prefilled by the next turn. A full prefill from position 0 matched decode bitwise; a prefill resuming after a hit did not always. Generated tokens can therefore differ between designs (41/112 between P-S and c1), since each design changes where prefills resume.

## Capacity (X0)

- KV capacity of a TP4 engine at 0.65 GPU memory utilization: 3.39M tokens.
- Block-row cache sized to the KV: R3 ~1.6 GB per engine (~1% of the KV); indexer ~54 GB raw, ~10 GB compressed.
- R3 as a KV layer reduced KV capacity by 0.21%.

## Always delivered, freed after use (X4/X5)

96 sessions × 4 turns; the trainer reads each finished session and releases it.

| Design | Pool | In-flight cap | Missing | Wrong | Leaked keys | Release p50 |
|---|---|---|---|---|---|---|
| c1 | 64 MB | 46 | 0 | 0 | 0 | 0.4 ms |
| c0 | 64 MB | 46 | 0 | 0 | 0 | 0.4 ms |
| P-S | 64 MB | 46 | 0 | 0 | 0 | 0.3 ms |
| P-S | 16 MB | 11 | 0 | 0 | 0 | 0.2 ms |

## Throughput and overhead

### Designs on one node (X2)

| Design | Agent | GRPO | Fork |
|---|---|---|---|
| c1 | 654 | 474 | 321 |
| P-S | 624 | 428 | 348 |
| P-B | 675 | 394 | 375 |
| P-A | 662 | 434 | 353 |

Output tokens/s on X1; differences are within run-to-run noise.

### Engine overhead relative to R3 off

Every overhead is measured against R3 off, with a fixed workload, CUDA graph `FULL`, batch invariance on and no preemption. One run per arm unless stated. Arms compared on one node share that node: another node of the same type runs 4.3% slower, and one build repeated on one node differs by 2–4%, so single-node differences of a few points between single runs are not conclusions.

There were two separate costs. R3 forced every new prompt row through the replay layers (GPU work). All outputs on top of R3 then cost several more points through host work on one TP rank (CPU work that stalled one GPU).

#### R3: rows through the replay layers

On DeepSeek-V4.1 the layers past the last KV source (rest of 20, and 21–39) run in eager prefill steps only on each request's trailing rows. The build reads R3 as needing exact rows at every prompt position, so every new row of every request runs through those layers. Removing only the capture hook leaves the cost unchanged; removing that rule removes it. No process shows a Python hot spot; GPU utilization rises from 79% to 90%.

**Top scenario.** 5 TP4 replicas plus a pool node (700 GB, and 160 GB per replica node), 512K context, 16 turns, 51 sessions in flight per replica, a weight update every step, 4× GB300 per replica. Arms run in sequence on the same nodes; median replica generation time; each arm once.

| Capture | CUDA graph | R3 in the response | R3 + indexer + hidden through Mooncake |
|---|---|---|---|
| R3 off | `FULL` | 1,977.7 s | — |
| every row through layers 20–39 (as built) | `FULL` | −41.2% | −41.4% |
| replay trim: only rows not yet in the pool run through those layers | `FULL` | −39.5% | −40.9% |
| partial capture, blocks only (§6.11 without cone patches) | `FULL` | −0.6% | −3.8% |
| partial capture, blocks only | `FULL_AND_PIECEWISE` | −40.5% | −41.0% |

At this scale nearly all prompt rows are new, so trimming rows already in the pool saves almost nothing. Both conditions are needed: partial capture, and CUDA graph `FULL` (with piecewise graphs the eager stages cost as much as the full rows). Generated tokens per arm differ by a few percent at this scale. With partial capture R3 costs little from here on: −0.1% in the response with cone patches (below).

#### All outputs: host work on the output rank

With R3 settled, the remaining cost is that of the indexer and hidden states. Its cause was not the bytes but where the host work ran.

**Bytes are not the bottleneck.** On the one-node 14.6K workload (512 agents, below), dropping the deep indexer slots from blocks and compressing the cone patches cut the indexer from 51.2 to 14.1 GB per run (cone patches 0.99 GB). The all-outputs arm took 1,238.5 s, no faster than before.

**What a profile shows.** All outputs against no indexer, same node, sampled at the turn plateau: GPU 0 runs at 82% against 93% (the other GPUs 91–93% in both). TP rank 0 is the output rank. It captures and assembles every request's rows while holding the connector lock, 17.6 s per 60 s with all outputs against 6.7 s without the indexer, and the engine's main thread waits on the same lock (begin step 3.0 s against 0.9 s, prepare output 2.0 s against 0.2 s). Rank 0's GPU idles while the others wait for it in collectives. The work under the lock was cone patch encoding (concatenation, masked copies, zstd), indexer block encoding and per-request capture of the indexer and hidden rows: many small per-request operations in every decode step. The writer thread was not involved: it was blocked 1.6 s in the run, and compressed indexer objects were already encoded on its thread pool.

**Shrinking the work under the lock.** Each change keeps the stored bytes identical. Same node, one run each, against no indexer at 1,094.9 s:

| Change, cumulative | All outputs | Over no indexer |
|---|---|---|
| cone patch encoding copies only the kept rows (29.3 → 17.4 ms per request) | 1,166.3 s | 71 s |
| indexer block buffers zero-allocated; only shallow slots copied | 1,158.0 s | 63 s (within noise) |
| hidden rows indexed per request; chunk bytes built on the writer thread | 1,148.3 s | 53 s |
| hidden rows kept as references to the step's host arrays and stacked on the writer thread; short cone steps kept as views | 1,134.0 s | 39 s |

On the first build, an arm without indexer cone patches splits the indexer's cost: 1,166.3 / 1,129.2 / 1,094.9 s (all / no indexer cones / no indexer) on this node and 1,237.3 / 1,166.9 / 1,114.6 s on a second node, so cone encoding is about half of it and block capture and encoding the rest, all on rank 0. After the last change GPU 0 runs at 88% (78% before it) and the output path holds the lock 12.3 s per 60 s (20.0 s before).

**Longer contexts, same node per workload.** At 128K the indexer's share is larger than at 14.6K: 16 sessions × 6 turns gave R3 off 76.2 s, no indexer 77.3 s (−1.4%), all outputs 81.3 s (−6.3%); 48 sessions × 8 turns gave 215.9 s, 220.7 s (−2.2%), 228.4 s (−5.5%). The profile of the larger run shows GPU 0 at 90% against 96% without the indexer. Turning zstd off changed nothing (229.8 s against 228.4 s), and the cores were not contended (load 5.7 on 144 cores). What remained was the lock: capture 3.5 s and block publishing 4.3 s per 60 s.

**Capture off the critical path.** Capture and per-request assembly moved to a dedicated thread, fed by one ordered queue from inside the lock, so the order of work is the order in which the output path handled requests and first-computation-wins is unchanged (protocol §6.9). The same build can run the work inline under the lock for comparison. 128K, 48 sessions × 8 turns, one run each:

| Node | Threaded | Inline | No indexer |
|---|---|---|---|
| A | 223.5 s | 227.4 s | 222.9 s |
| B | 222.9 s | 223.0 s | 218.7 s |

All outputs now cost 0.3% and 1.9% over no indexer, against 3.3 points before. Threaded against inline is 1.7% on one node and equal on the other, so this pair alone does not establish the gain of the thread; the profiles below do. Exact hidden states (X7) on node B: 223.7 s.

**512K, one node.** 8 sessions × 6 turns, all arms on the same node, one run each: R3 off 200.5 s, no indexer 201.6 s (−0.5%), all outputs 205.0 s (−2.2%). In the profile GPU 0 runs at 97% against 98% for the others, and the main thread's lock time is gone (process output 1.2 s, begin step 0.1 s per 60 s). Most of the extra time of all outputs is in turn 1, the large prefill (+1.4 s); later turns add 0.3–0.5 s each. The first attempt of this run failed in the checker, not the engine: decoding one 512K request's indexer on the GPU ran out of memory, and checkers now decode on the CPU above 100,000 rows.

| Context | Workload | Capture | R3 off | No indexer | All outputs |
|---|---|---|---|---|---|
| 14.6K | 512 agents × 16 turns | under the lock, after the four changes | — | 1,094.9 s | 1,134.0 s |
| 128K | 48 sessions × 8 turns | under the lock | 215.9 s | 220.7 s | 228.4 s |
| 128K | 48 sessions × 8 turns | own thread, nodes A / B | — | 222.9 / 218.7 s | 223.5 / 222.9 s |
| 512K | 8 sessions × 6 turns | own thread | 200.5 s | 201.6 s | 205.0 s |

#### Top scenario: runs and their spread

Same scenario as above (5 replicas, 512K, 51 sessions in flight per replica, a weight update every step). Each arm once per run, arms in sequence on the same nodes, 5 replicas per arm.

| Run | R3 off | R3 in the response | All outputs without the indexer | All outputs |
|---|---|---|---|---|
| cone patches, GB300 | 1,986.6 s | 1,989.1 s (−0.1%) | — | 2,028.7 s (−2.1%) |
| cone patches, GB200 | 2,056.7 s | 2,052.6 s (no cost) | — | 2,168.5 s (−5.2%; −4.0% median of per-node pairs) |
| cone patches, GB300, earlier build | — | — | −0.4% | indexer 3.3 points more |
| host work under the lock reduced, GB300 | 1,968.4 s | — | 2,013.3 s (−2.2%) | 2,096.4 s (−6.1%) |
| capture on its own thread, GB300 | 1,999.8 s | — | −2.9% (median of per-replica pairs) | 2,084.9 s (−4.1%); −5.1% median of per-replica pairs |

Times are the median replica. In the run with reduced host work one replica sat on an outlier node (−18% all outputs, −30% without the indexer); the other replicas' pairs were −3.6 to −6.2% for all outputs and −0.5 to −3.3% without the indexer, an indexer cost of about 3 points, as in the earlier build. In the capture-thread run the per-replica pairs were −5.8 / −5.4 / −4.5 / −3.8 / −5.1% for all outputs and −2.8 / −2.9 / −3.8 / −1.9 / −3.0% without the indexer, about 2.2 points for the indexer. Every run had 0 missing and 0 wrong rows.

The arm without the indexer ran nearly the same code in the earlier build and in the capture-thread run, and cost −0.4% in one and −2.9% in the other. Run-to-run spread at this scale is therefore at least 2 points, and a single top run cannot resolve the effect of one change. The one-node gains (71 → 39 s over no indexer at 14.6K, the lock time gone at 512K) did not show at the top. The writer is not the cause there: in the capture-thread run puts took 0.71 ms at p99, the capture queue held at most one item, and the writer was blocked about 0 s, except on the slowest replica, which had one 4 s stall and 11.6 s blocked in all. All outputs write 53 GB per replica, against 15 GB without the indexer. Factors present only at the top, such as five replicas writing to one pool node over RDMA, are not yet tested. A repeated run with the latest build, alternating the arms twice each on the same nodes, is in progress.

On the GB200 cluster the checker's readback of one replica ran at about 25 MB/s and 6,831 keys still failed after three retries; they are reported missing, with 0 wrong rows, while the pool holds every object (no eviction). The same readback on GB300 takes 130–210 s with nothing missing. Reads of large objects that outlive the 1 s read lease are the suspect; this is the trainer-side path and is still open. In that GB200 run each replica wrote 106.6 GB in 427,573 puts with no retry: indexer 91.1 GB (3,419 bytes per token, before deep slots were dropped from blocks), R3 13.2 GB, hidden 2.1 GB, cone patches 0.33 GB.

#### One node, 14.6K context

512 closed-loop agents, 16 turns growing to 14.6K tokens, 256 output tokens per turn, same prompts and prefix hits in every arm. All arms on the same node, one run each, before the host-work changes above:

| Arm | Time | vs R3 off |
|---|---|---|
| R3 off | 1,031.4 s | — |
| R3 in the response | 1,044.5 s | −1.3% |
| R3 through Mooncake, cone patches | 1,066.6 s | −3.3% |
| R3 + indexer + hidden through Mooncake | 1,146.3 s | −10.0% |
| No trim (every row through layers 20–39) | 1,370.6 s | −24.7% |

Cone patches for R3 are about 1.7 GB for 512 sessions (2.1M generated tokens). The 7 points between R3 and all outputs were mostly the host work above, not the indexer's 7× larger bytes.

#### Exactness

Against a run without trimming, same prompts, no weight updates: every cell a sampled token depends on, plus all columns below layer 20, is bitwise identical for R3 and the indexer, in 512/512 plain sessions, 16/16 forks and 16/16 forks inside a parent's rows. 0 of 137.6M R3 cells and 0 of 36.0M indexer cells missing. Every build in the host-work and capture-thread series repeats these checks (512/512, 16/16) with 0 missing. The reader library, run against the writer's own overlay, differs in 0 of 286M replay cells and reports no needed cell missing, in plain runs and in the group, fork, delayed-fork, forced-preemption, write-failure and cut regressions. With speculative decoding and a weight update every step, groups, forks, delayed forks, forced preemption and injected write failures stay at the same counts as without partial capture.

### Engine-local store vs Mooncake, 8 replicas

Eight TP4 replicas and one pool node; arms in sequence on the same nodes, same workload (128K, 32 turns, 24 sessions per replica, a weight update every step). All three arms have R3 on, so this compares delivery backends only, not the cost of R3.

| Arm | Generation time per replica (median) | Output tok/s, 8 replicas | Put p50 / p99 | Trainer read per replica |
|---|---|---|---|---|
| Engine-local store, R3 returned in the response | 804.7 s | 3,876 | — | in the response |
| Mooncake, R3 | 798.6 s | 3,845 | 0.3 / <1.8 ms | 1.0–1.2 s |
| Mooncake, R3 + indexer + hidden | 799.1 s | 3,846 | 0.3–0.4 / <0.71 ms | 99–108 s |

Choosing Mooncake over the engine-local store costs nothing measurable here (±3% noise), but this workload is GPU-bound: eager mode, at most 64 requests per step. The full outputs are 5.9 KB per token, about 23 MB/s per replica.

### Each hop against its ceiling

Two nodes, 4× ConnectX-7 NDR (400 Gb/s) per node.

| Hop | Measured | Ceiling | Assessment |
|---|---|---|---|
| NIC (`ib_write_bw`, 1 MB) | 48 GB/s per port, 192 GB/s for 4 ports | 50 GB/s per port | line rate |
| GPU → pinned host | 67 GB/s at 4 MB, 182 GB/s at 134 MB | NVLink-C2C | tens of µs per step for R3 |
| Host block assembly | 19–23 GB/s | one-core memcpy | at the ceiling |
| Mooncake put, 4–16 MB, 8 threads | 57–70 GB/s | 192 GB/s | 30–36% of the node, above one port |
| Mooncake put, one thread | ≤7 GB/s | one port | not zero-copy: the payload is copied into the registered buffer first |
| Mooncake get | ≤14 GB/s at any size or thread count | 192 GB/s | bound by copying into Python bytes |
| Mooncake op, 60 KB (one R3 block) | 200 µs put, 110 µs get | ~30–50 µs (two RPCs + 1 µs transfer) | 4–6× above, irrelevant at ~300 objects/s per replica |
| zstd-3, one thread | 180 MB/s | — | the hop that matters on the write side |
| Indexer decode on the trainer, numpy | 110–210 MB/s | memory bandwidth | the hop that matters overall |

At the top scenario (512K, 51 sessions in flight per replica, 8 replicas) each replica writes ~45 MB/s, 0.2% of the pool node's NICs. Per replica, measured inside the run:

- **One TP rank writes:** put calls equal successes; there are no redundant puts.
- **Writer thread:** moving indexer encoding to a 4-thread pool and batching ready puts cut the writer's wait on encoding from 1,517 s to 310 s, R3 queueing from 1.5 s to 0.056 s, and scheduling pauses on writer backlog from 78–102 s to 0.
- **Engine host time:** in the 512-agent workload above, committing outputs on the output thread averages 29 ms per step, and the full arm moves about 70 MB per step from GPU to host. The copy is cheap; what cost throughput was the per-request capture work done while holding the connector lock on TP rank 0, which stalled that rank's GPU and the main thread (engine overhead, above). It now runs on its own thread.
- **Trainer read:** 126–207 s to read back the 51 sessions (down from 857–914 s), with indexer decoding on the GPU where replay runs (protocol §7).

## Indexer encoding (X6)

A token's indexer row is 8 source layers × top-512 = 4,096 positions.

| Indexer encoding | Size per token | Ratio vs int32 |
|---|---|---|
| int32 | 16 KB | 1× |
| 20-bit packing (on the GPU) | 10 KB | 1.6× |
| 20-bit packing + zstd-3, contexts up to 128K | 2.5–3.6 KB | 4.6–6.7× |
| 20-bit packing + zstd-3, 512K context | ~5.6 KB | ~2.9× (zstd itself ~1.8×) |

Compression depends on context length. At long contexts the selected positions spread over the whole compressed sequence, so zstd gains less, but it still halves the stored bytes at 512K. The encoder stores the packed bytes uncompressed when zstd does not reach 90% of them.

**Sorted deltas in byte planes.** The current codec stores each cell's positions as deltas from the previous position and splits the deltas into byte planes before zstd. The positions of a cell are already ascending, so the deltas are small and their high planes compress well. The codec is named in the manifest and in each object's header (protocol §3).

**Pads stay last.** The engine's top-k, and the trainer's top-k kernel that replay must match, emit a cell as ascending positions with the `-1` pads at the end; this held for every row of a captured data set. A decoder must restore exactly that order. One delta decoder put the pads first, so 9–12% of rows started with `-1`. Every check and review compared cells as sets and passed, and trainer indexer replay gained nothing until the decoders were fixed to rotate the pads back to the end (X9). Tests now compare decoded cells byte for byte in engine order.

**Fewer bytes under partial capture.** Deep slots (layers 20 and up) of block objects are never read under partial capture, since deep cells come from cone patches, so blocks no longer store them, and cone patches are compressed. On the one-node 14.6K workload this cut the indexer from 51.2 to 14.1 GB per run, with 0.99 GB of cone patches. At the top scenario all outputs now write 53 GB per replica (15 GB without the indexer). None of this changed throughput; the cost was elsewhere (engine overhead).

Selections are bitwise identical across runs under batch invariance. Through the delivery path (112 requests) the indexer matched an independent in-engine capture bitwise (551 B stored per token thanks to block sharing).

## Hidden states (X7)

The hidden state is the input of `lm_head`; the trainer multiplies it by the LM head to get full-vocabulary logits. Comparisons against vLLM's logits:

| Hidden states | Result |
|---|---|
| fp32 `lm_head(hidden)` vs vLLM logits | max abs diff 2.1e-4, not bitwise |
| bf16 | max abs diff 0.062 |
| fp8 e4m3 + per-token scale | full-vocabulary KL mean 4.4e-4–8.7e-4, max 1.9e-3–2.8e-3; argmax agreement 98–100% |

Through the delivery path every fp8 row was the exact requantization; KL mean 7.7e-4, max 0.012.

**Exact delivery is the default.** Codec `raw` stores the rows in the dtype that enters the LM head, float32 on this build: 16 KB per generated token, against 4.1 KB for fp8. Recomputing the sampled tokens' logprobs from the delivered rows gives an error of max 7.8e-4 and mean 3e-6, against a max of about 0.3 with fp8; the residual is the checker's own matmul, which is not bitwise equal to vLLM's (first row of the table). In every later regression the max stays at or below 6.7e-4 with 0 bad rows, including across forced preemption. Cost on the 128K one-node workload (same node, one run each): 223.7 s against 222.9 s with fp8, within noise, with 1.0 GB of hidden states in the run. fp8 stays available and produces the same bytes as before.

## Trainer-side replay and training-inference mismatch (X9)

**R3 replay, first test.** R3 rows for 16 sessions were replayed in a BF16 trainer (TP1/EP16), forward only.

- Alignment: 2% of tokens differ in their expert set at layer 0 when aligned, ~100% when shifted by ±1 position or compared against the wrong layer.
- The real routing gap grows with depth (16% at layer 1, 40–85% deep).

| | Mean abs logprob diff | Max |
|---|---|---|
| Replay off | 0.175 | 8.15 |
| Replay on | 0.082 | 3.09 |

**Delivered rows are the exact rows.** Under partial capture a BF16 trainer replays the delivered rows (no-replay cells run their own routing) and, separately, the exact rows of an untrimmed run. Over 8 forked samples (14,848 generated tokens), plain and interior forks, the two replays give bitwise identical logprobs, and a repeat is bitwise identical too. Without indexer replay the trainer's own indexer top-k overlaps the delivered cells by 0.90 on average (0.67 against the previous row, which checks alignment).

**Indexer replay.** The trainer replays delivered indexer cells too, and already emulates the engine's NVFP4 KV cache (quantize on insert, dequantize on gather, 528 B per token). On fork data decoded with pads first (X6), adding indexer replay gave nothing: mean |Δlogprob| 0.0294 with R3, 0.0303 with R3 + indexer. On new fork data decoded in engine order:

| Replay | Mean abs logprob diff |
|---|---|
| none | 0.0489 and 0.0505 (one run each) |
| R3 | 0.03413 |
| R3 + indexer | 0.03285 (−3.8%) |

Delivered equals exact here as well, and repeats are bitwise identical. The two data sets are different rollouts, so their numbers are not comparable with each other.

**Self-perturbation floor.** To judge the remaining 0.033, the same delivered data was scored twice with one rounding change in the trainer: the LM head in fp32 and in bf16. KL is the k3 estimator per token.

| | Mean abs diff | Median | p99 | KL mean | KL median |
|---|---|---|---|---|---|
| floor: trainer fp32 head vs bf16 head | 0.0066 | 8e-5 | 0.068 | 1.4e-4 | 1.7e-8 |
| trainer vs rollout | 0.0329 | 4.2e-4 | 0.44 | 4.8e-3 | 9.2e-8 |

**Median and mean tell different stories.** The median KL between trainer and rollout is 9.2e-8, the mean 4.8e-3. The 10% of tokens with |Δ| > 0.1 carry 75% of the mean, spread evenly over positions in the turn. A median alone hides that tail, so both are reported. Another system's published report gives a median KL of 2.2e-4 and a mean |Δ| of 0.0085, on a different model; numbers from different models, and medians against means, are not comparable.

**What the remaining gap is.** Not sampling parameters: the rollout logprobs are raw, at temperature 1. Not cells the trainer computes itself: replay of the exact rows gives the same logprobs bitwise. Not delivery: delivered rows equal the exact rows. The gap between trainer and rollout, about 5× the floor in mean and median, is trainer-engine numerics outside the delivery (expert precision and similar), and belongs to the train-inference consistency work, not to this protocol.

## Scale

P-S, R3 + indexer (GPU packing) + hidden, a weight-version bump after every step, one TP4 replica per 4× GB200, all replicas on one shared Mooncake pool.

| Run | Scale | Requests | R3 / indexer rows | Missing / wrong | Version segments | Hidden | Released |
|---|---|---|---|---|---|---|---|
| 512K | 13 replicas × 51 sessions × 108 turns, 768 GB pool, RDMA over InfiniBand | 71,604 | 3.48e8 each | 0 / 0 | 71,604, 0 failures | last request per session: 42,432 rows, logprob error 0.041 | 663/663 |
| 512K, high fidelity (real text, weights perturbed every step) | 5 replicas × 51 sessions × 108 turns | 27,540 | 1.33e8 each | 0 / 0 | 27,540, 0 failures | 119,630 rows, 0 bad | 255/255 |
| 512K, hardened code, 51 sessions in flight | 8 replicas × 51 sessions × 16 turns, 700 GB pool node + 160 GB per replica node, hard-pinned | 816 per replica | — | 0 / 0 per session, 0 evictions | — | — | 51/51 per replica |

The first two runs predate the hardening below. Their sessions hold one trajectory each; branches and engine events are covered by the regression tests.

The third run exposed two store requirements, now in protocol §9:

- Without pinning, an in-flight cap sized for short-context indexer bytes let 1.3 TB into a 700 GB pool, and Mooncake evicted 1.81M keys.
- Taking down a finished replica's segment removed objects of the replicas still reading.

## KV offload to an SSD tier (E0)

A trajectory that waits for many weight versions loses its KV from the GPU, and its next turn recomputes the whole context unless a lower tier still holds it. E0 tests a tier on the node's local NVMe below the CPU tier: does the resumed context load instead of recomputing, and does aux delivery stay complete when KV comes from the tier?

**Setup.** vLLM's offloading connector with a tiering spec: a CPU tier, and a filesystem tier on local NVMe. 128K context, all outputs with partial capture. Each pair of arms ran on one node, one run per arm.

**Forcing a spill.** With the full GPU KV (10.3M tokens) the workload (48 sessions × 12 turns, 6.3M tokens) never left the GPU: 34.6M of 40.8M tokens were prefix hits on the GPU and the tiers served nothing (267.8 s CPU only, 269.7 s with the tier). Restricting the KV to about a quarter, with at most 8 running requests (48 × 6 turns), still kept every prior context on the GPU (294.4 s, 297.4 s; the tier wrote 26 GB and read nothing). The runs below use about 1/16 of the KV and at most 4 running requests, 48 sessions × 4 turns.

| CPU tier | NVMe tier | Generation time | Old context recomputed |
|---|---|---|---|
| 20 GB | no | 304.0 s | 0.115M tokens |
| 20 GB | yes | 313.0 s | 0.017M tokens |
| 4 GB | no | 411.9 s | 5.69M tokens |
| 4 GB | yes | 317.9 s (−23%) | 0.61M tokens |

A 20 GB CPU tier already holds nearly all of it; the NVMe tier then removes the rest of the recompute and adds 9 s in this run. With a 4 GB CPU tier the NVMe tier is what saves the run: it wrote 47.8 GB and read 38.7 GB in 27 s, with 32,826 of 33,021 chunk lookups hitting (99.4%). The residual 0.61M tokens of recompute come with 18 promotion failures: a chunk read from NVMe goes through the CPU tier, and 4 GB had no room for it, so the engine recomputed. Aux delivery stayed complete in every arm: 0 missing rows, 0 uncovered cone cells, 0 preemptions, also with KV loaded from the tier.

**Configuration.**

- **Offload generated tokens too.** The next turn's prompt contains the previous turn's output; with prompt-only offload that part is always recomputed.
- **There is no backpressure.** Under the default policy a store the tier cannot take is dropped, and its blocks are recomputed later. A full SSD degrades the same way, silently, so drops and tier fill need monitoring.
- **Size the CPU tier for promotion,** not only as a cache: every NVMe hit passes through it.

## Aux objects on SSD (E-aux)

Before E-aux, the store's memory had to hold every session in flight, pinned, and sessions were admitted by that memory. E-aux asks whether the store can keep aux objects on each holder's local NVMe instead, with memory only a cache, at the same speed and with nothing lost. It is the basis of the SSD-first default in protocol §9.

**Setup.** 14.6K context × 512 sessions × 16 turns, all outputs with partial capture, one TP4 replica per node. The memory arm: a 96 GB hard-pinned segment. The SSD arm: an 8 GB segment, no pinning, Mooncake offload to a fresh directory on local NVMe (`--enable_offload --offload_on_evict --offload_force_evict=false --enable_disk_eviction=false --client_ttl=600`), sessions admitted by free disk. Each pair ran on one node, in both orders.

**Speed.** Eight same-node pairs:

| Node | First | Second | SSD vs memory |
|---|---|---|---|
| A | memory 1247 s | SSD 1225 s, 1250 s | −1.8%, +0.2% |
| B | SSD 1220 s | memory 1226 s | −0.5% |
| C | memory 1150 s | SSD 1237 s | +7.6% |
| D | SSD 1130 s | memory 1143 s | −1.1% |
| E | memory 1199 s | SSD 1238 s | +3.3% |
| E | SSD 1224 s | memory 1177 s | +4.0% |
| F | SSD 1214 s | memory 1187 s | +2.3% |

Rollouts on SSD are 1–2% slower (median +1.3%, mean +1.8%); reading every session back is about 2% slower (median 1621 s against 1583 s). Every arm delivered every row: 7.44M R3 and indexer rows and 131K hidden rows, 0 missing, 0 wrong. The SSD arm held about 7 of its 8 GB and spilled 36 GB.

**Which settings fail.** The same workload with the alternatives:

- **Offload at put (`--offload_on_evict=false`)** is not usable. Turns took 1467–2056 s instead of about 90 s, and every session read back had missing rows. In isolation, 100K objects of 32 KB through a 256 MB segment lost 2,976 (3.0%) with no error and all eviction counters at zero; a 1M offload queue and 10-hour task timeouts lost the same 2,976. The store evicts objects whose disk write has not finished.
- **No SSD, no pinning, 8 GB:** the writers backed up and the run did not finish in 4 hours.
- **Admission by memory with an SSD:** 8 GB admitted 57 sessions at a time, 9 waves, 20% slower. Admission has to follow the disk.

**Fault tests** (Mooncake alone, 256 MB segment, 4 MB objects unless noted):

| Test | Result |
|---|---|
| 1 GB through 256 MB with a retrying writer | 0 failed, 0 corrupt; puts first fail with −200, then land as offload frees memory |
| 100K × 32 KB, offload on eviction | 100,000 of 100,000 intact |
| Remove right after a read | non-force removal removes nothing (read lease, −706); force removal removes all |
| Disk after removal | not reclaimed after 300 s; reused by later writes (6 × 256 MB written and removed left 512–588 MB) |
| Holder restarted on its old directory | every removed key is listed again; a fresh directory lists none |
| Holder paused 30 s | `client_ttl` 10 s: all memory objects lost; 600 s: none |
| Holder killed | the master still lists every key, every get returns empty; puts fail |
| Disk writes failing (file size limit) | `FILE_WRITE_FAIL` per bucket, partial files left (15 GB for 2 GB of data), memory stays full, every later put fails; no acked object lost |
| `--quota_bytes=1 GB` | no effect on the offload directory |

These are the rules of protocol §9: offload on eviction only, a long `client_ttl`, force removal, one fresh directory per holder and job deleted at job end, an abort on a dead holder or on puts that stop landing, and disk headroom monitored outside the store.

## Correctness hardening

Each row is a defect found by test or review, the test that exposes it, and the result before and after the fix, on the same workload. "Wrong" is against the reference unless noted.

| Defect | Test | Before | After |
|---|---|---|---|
| A preemption was handled as a finish | forced preemption, 323 preemptions | 57,678 missing, 753 wrong of 2.65M; hidden 14 bad | 0 / 0 of 2.71M; 323 stashed, 323 resumed |
| GRPO samples of one session spliced each other's tails | GRPO, weights updated every step | 19,135 wrong (inline) | 0 |
| In-flight weight update (pause + reset) | reset every 20 s | 511 wrong, hidden 26 bad | 0 / 0 over 59 resets and 598 preemptions (4.65M rows) |
| Write failures: a lost block counted once per later turn; a session-wide error marker failed every sibling; a recompute republished in place of a lost block carried another version's rows under the lost one's labels | 1 in 500 and 1 in 97 writes fail | up to 5,504 wrong; whole sessions failed | 0 wrong; failed manifests fail only their request, at once; lost blocks reported missing |
| Siblings of one session computing a block at another version received the first copy | fork 2 parent turns later | 440,160 wrong | 0 |
| Indexer blocks with rows the request never captured (a hit ending mid-block) were published with pad as valid | GRPO | 118 such blocks per run; invisible to range checks | not published; the computing request publishes them |
| A fork took the parent's latest tail instead of the tail at the fork point | delayed fork | 5,376 wrong (inline) | 0 |
| Writer bookkeeping grew without bound | GRPO with session release | 7,422 version records left | 184 (sessions released after the last snapshot) |
| Speculative decoding: hidden states dropped (1 + num_draft logits rows per request), positions taken from the scheduler's run-ahead token start, hash-lagging blocks and a capture ending on a block boundary left out of the tail, multi-block tails not spliced, the previous request's computation of `aux_start` replaced, a fork's parent-computed block replaced | GRPO and forks with DSpark (`num_speculative_tokens=4`) | up to 504 missing, 46,243 wrong; hidden 127 requests bad | R3 0 missing, 0 wrong of 4.65M; indexer 0 missing; hidden 0 bad; version segments 0 failures |
| A block starting before `aux_start` was shared without a version check, so a block spanning the fork point could hand a fork another trajectory's rows | spec-decoding fork, per-trajectory copies forced | 861 wrong | 0 |
| The fork-point tail was looked up by start position; a later, shorter parent turn overwrote it | spec-decoding fork, per-trajectory copies forced | 7 wrong | 0 |
| Same-version sharing compared sets of versions, so a mixed v0/v1 block matched an all-v1 computation; identical bytes skipped the version check | review; exact per-position comparison | — | unit cases; regression unchanged |
| A failed own copy (`hash~tag`) could be rewritten under its name; a failed indexer object blocked the block's R3 sharing | review; write-failure injection | — | 166 injected failures, all reported, 0 wrong |
| A delayed fork's own copy of a block whose first rows came from a prefix hit was withheld (pad rows), and no other request published it | delayed fork | 3,384,265 indexer rows missing | 0; 112 blocks filled from the session's object |
| Rows a fork served from a prefix hit were labelled with the version inherited from the parent | fork | 10 version-segment failures; 1.69M indexer rows missing | 0 |
| `invalid_from` was lost across a preemption; a parked finish used only the first block of a multi-block tail; hidden chunk metadata assumed full chunks; an abort and a preemption in the same step left a stash | review; forced preemption | — | 89–167 preemptions per run, 0 missing, 0 wrong; 72/72 hidden chunk records exact |
| In-flight weight update combined with preemption | one weight change per step while paused, ~110 preemptions | — | 0 wrong of 4.53M with per-trajectory copies |
| Partial capture: a patch floor at `aux_start` lost cells of forks inside a parent's rows and of shrinking prompts; earlier patches past a fork point shadowed the fork's own cells; one version per patch lost per-cell provenance | review; interior fork arm | cells lost silently in unit cases | patches clipped per request, per-cell versions; interior forks 16/16 bitwise |
| Partial capture: the position `aux_start` computed by the previous request, preemption before the first sampled token, holes before a long generation, a second release batch, more than 256 versions in one request | review, two rounds, four reviewers each | unit cases | fixed; each fix has a unit case; regressions unchanged |

KV-load rewind (block-level failure recovery) cannot occur on this model: vLLM refuses block-level recovery with 9 KV cache groups. For single-group models the protocol reports rows from the rewind point as invalid.

## Finding: same-version rows differ without the consistent build

On this build (batch invariance, not #62365), GRPO siblings prefilling the same prompt in different chunks route differently: in one run 2,253 of 5,842 positions differed, and ~4,800 blocks per GRPO run had same-hash, different-content copies. Against the reference, sharing the first copy leaves ~224K rows of a GRPO run (and ~243K of a fork run) delivered from a sibling's computation. Under the train-inference consistent build these computations are bitwise identical and the protocol shares them; keeping a copy per trajectory instead (`OC_ALT=1` in the overlay) delivers every trajectory its own computation.

## Finding: `deep_gemm_mega_moe` broke batch invariance

With `moe_backend="deep_gemm_mega_moe"`, the same prompt alone vs batched gave different R3 rows and greedy tokens. The backend enables MoE sequence parallelism; on CUDA under `VLLM_BATCH_INVARIANT=1`, the batch-invariant branch of `CudaCommunicator.reduce_scatter` is gated on `current_platform.is_rocm()`, so a plain NCCL ring reduce-scatter runs, whose summation order depends on which rank owns the output slice. The four per-rank inputs to the layer-0 post-attention reduce-scatter were bitwise equal while its output was not. Dropping the `is_rocm()` condition made alone / 8 copies / mixed batches bitwise identical in R3 rows, greedy tokens and logprobs. Any configuration enabling MoE sequence parallelism (e.g. DP>1 with TP>1 and EP) should be affected.

## Smaller bugs found building the path

- The manifest was deferred to the next engine step; an idle engine never wrote it.
- Keys used the engine-internal request id (random suffix); use the caller-visible id.
- Duplicate puts from every TP rank retried forever; "key already exists" is success.
- Async scheduling computes one more position after the last sampled token; readers truncate to `L−1`.
- Some MoE implementations call `route` directly, bypassing a router-level replay hook.
- numpy integers in the manifest broke JSON encoding, and the writer only logged the error, so readers timed out; manifests are built from Python ints.
