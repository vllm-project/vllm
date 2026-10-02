# Layer-sharded KV cache (KVPP)

KVPP increases the capacity of replicated MLA caches by assigning each layer
bundle to one persistent owner in a PCP × TP replica group. Every rank executes
its layers using stable attention tensor views. NCCL broadcasts materialize
historical KV in two reusable receiver slots, with one-bundle-ahead prefetch
before attention.

The target workload is multi-prefix serving where a limited KV memory budget
causes eviction. The benefit depends on cache residency, broadcast cost, and
available computation overlap.

## Enablement and compatibility

Enable `--enable-kvpp` with `VLLM_USE_V2_MODEL_RUNNER=1`. Choose eager execution
with `--enforce-eager`, or compiled PIECEWISE:

```text
--enable-kvpp -cc.mode=3 -cc.cudagraph_mode=PIECEWISE -cc.use_inductor_graph_partition=false
```

| Component | Requirement |
| --- | --- |
| Device and runner | NVIDIA CUDA, NCCL, Model Runner V2 |
| Replica group | PCP × TP ≥ 2, DCP=1; at least one target bundle per rank |
| Cache layout | Replicated full-attention MLA, common block size, layer-compact storage |
| Execution | Eager, compiled PIECEWISE with Dynamo splitting of attention and KV updates, or PIECEWISE breakable CUDA graphs |
| Offload | `SimpleCPUOffloadConnector`, using persistent owner views |

Keep the default `splitting_ops` so cache acquisition, prefetch, and release run
on every replay. Models that default to breakable CUDA graphs (for example
GLM-5.x and DeepSeek-V3.2 sparse attention) run acquisition and release in eager
breaks. KVPP overrides FULL and FULL_AND_PIECEWISE to PIECEWISE because FULL
replay bypasses the forward context. GPU MRV1, Inductor graph partitioning, DBO,
HiSparse, and cross-layer KV sharing are unsupported. Other connectors must
implement the layer-sharded storage contract.

KVPP broadcasts on a side stream while compute continues, so it must not run
next to all-reduce kernels that spin on peers. `--enable-kvpp` therefore sets
`NCCL_LAUNCH_ORDER_IMPLICIT=1`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`, and
`VLLM_ALLREDUCE_USE_SYMM_MEM=0` unless they are already set, and disables custom
all-reduce. Without both the launch ordering and NCCL-only all-reduce, GLM-5.3 on
4×GB300 deadlocks at the first broadcast with history.

For TP2 × PCP2, set `--tensor-parallel-size 2` and
`--prefill-context-parallel-size 2`. PCP retains its MRV2 capability limits,
including the restriction on PCP + PP execution. PCP MoE requires
`VLLM_MOE_SKIP_PADDING=0` because token gathering leaves routing padding masks
rank-local; the GSM8K configuration includes this setting.

## Cache allocation and management

A layer declares its replicated components through
`AttentionLayerBase.get_kv_cache_bundle()`. DeepSeek MLA bundles its main cache
with its sparse-indexer cache when present. Draft caches remain local. PCP's
existing prefill gather and cache update maintain full replicas when DCP=1.

The allocation flow extends the common KV cache pipeline:

1. Discover bundles in model order and assign contiguous, count-balanced owner
   partitions. Check agreement on logical specs, bundle order, and ownership
   across workers.
2. Convert each worker's physical memory budget into logical block capacity.
   The layout builder accounts for owned bundles and two alternating scratch
   slots, using the same packed layout as final allocation.
3. Run the existing override, auto-fit, null-block reservation, admission checks,
   config builder, layout validation, and cross-worker minimum-block flow.
4. Apply physical placement after the common block count is final.
   `KVCacheStoragePlan` records regions in the ordinary `KVCacheConfig`; the
   allocator builds stable views into a packed byte arena using the existing
   allocation and view-construction helpers.
5. Use that storage plan for broadcast and offload. Register persistent owner
   views with the offload connector. A common per-block byte budget keeps pool
   block IDs aligned across ranks with unequal owner counts.

Logical block IDs, prefix hashes, scheduler reference counts, and request
admission retain their existing meanings. Physical allocation includes packing
requirements from cache specs; block-count overrides must fit the worker budget.

## Broadcast and cache lifetime

`maybe_prepare_kvpp` prepares the runtime after connector pre-forward setup.
It determines history from scheduled request state before PCP partitioning,
where global context lengths distinguish previous KV from current-prefill
queries. A batch with history broadcasts only the unique blocks that hold its
requests' computed tokens: the owner gathers them from each bundle component,
broadcasts one packed buffer, and receivers scatter it into their scratch views.
`BlockTables` keeps a host mirror of logical block IDs for this when KVPP is
enabled. Dummy runs, which have no request state, broadcast complete bundles. A
batch without history uses its cache views directly.

For a batch with history, the first component access to bundle `i` calls
`acquire`, which:

1. Starts its broadcast if needed and makes the compute stream wait for completion.
2. Marks the bundle active and starts bundle `i+1`, when present, on the transfer stream.
3. Provides the same views to the cache update, sparse indexer, and attention.

The main attention's `release` records completion for scratch reuse. CUDA events
order preceding-forward cache writes before broadcast reads, broadcast completion
before computation, and the previous attention use before reusing a scratch
slot. All ranks traverse bundles in the same order and release them before the
next forward. Failed forwards retain their state; runner shutdown synchronizes
the device before releasing cache storage.

`set_forward_context` scopes the platform-selected `KVPPRuntime` to the forward.
`parallel_state` owns the KVPP group from `initialize_model_parallel` through
`destroy_model_parallel`. Each group spans PCP × TP within one DP replica and PP
stage, with local rank `pcp_rank * tp_size + tp_rank`. The GPU worker calls
`warmup_process_group` before its initial memory snapshot. The runtime uses the
group's device communicator for asynchronous broadcasts and manages its streams
and events.

## Validation

### Accuracy and execution

DeepSeek-V2-Lite-Chat on 4×H100, TP2 × PCP2, MRV2, compiled PIECEWISE:

| Check | Replicated baseline | KVPP |
| --- | ---: | ---: |
| GSM8K, 200 questions / 5-shot | 64.0% | 63.0% |
| Invalid answers | 0 | 0 |
| Short/long and repeated-prefix smoke | Reference | 8/8 token-for-token matches |

Both scores pass the existing 0.65 ± 0.08 threshold (max output 256, concurrency
8). The evaluation used `08cccd2edf` with its eager-only guard removed.
Commit `74a2ceb3a9` passed a fresh startup/capture, the eight-request comparison,
and nine configuration tests. Observers confirmed graph replay with KVPP
prefetch executing outside capture.

Module coverage includes placement agreement, capacity, scratch aliases,
persistent offload views, distributed lifetimes, and group reinitialization.
The full distributed suite and helper tests were not rerun at `74a2ceb3a9`.
`tests/evals/gsm8k/configs/DeepSeek-V2-Lite-Chat-KVPP.yaml` provides the model
accuracy guard. Extend validation across prefill, decode, prefix reuse, and pool
reload as execution support expands.

### Capacity-constrained prefix pooling

The best observed attention-before result used DeepSeek-V2-Lite-Chat on
4×RTX 5090, TP4, MRV2 eager, with 512 MiB KV per rank. Four families each supplied
4,096 shared prefix tokens and a 1,024-token unique suffix. Each of two measured
rounds ran 256 requests at concurrency 4, generating one token per request after
warmup.

| Metric | Replicated baseline | KVPP |
| --- | ---: | ---: |
| Logical KV capacity (tokens) | 17,248 | 51,776 |
| Prefix-cache hit rate, rounds 1 / 2 | 37.8% / 38.3% | 80.0% / 80.0% |
| Input tokens/s, rounds 1 / 2 | 67,988 / 68,586 | 124,651 / 124,638 |
| Mean input tokens/s | 68,287 | 124,645 (+82.5%) |

This GPU-resident prefix pool used no CPU offload. Input throughput includes
cached tokens, so the gain combines cache residency and broadcast scheduling.
Both services used `NCCL_P2P_DISABLE=1` and disabled FlashInfer autotune. Maximum
model length was 8,192 for baseline and 16,384 for KVPP; both served the same
5,120-token requests.

The performance data predates the current refactor and PIECEWISE support; its
scope is the earlier eager implementation and this one-token-output workload.
Current-revision throughput and long-output decode gains remain unmeasured.
Device and workload matter: the H100 cropped-DSv3.2 CPU-offload experiment showed
a throughput regression with attention-before broadcast. Source provenance and
validation commands are recorded in [PR #59059](https://github.com/vllm-project/vllm/pull/59059).

### GLM-5.3 prefill pooling on 4×GB300

GLM-5.3 FP8 (DSA sparse attention), MRV2, default breakable CUDA graphs
(PIECEWISE under KVPP; TP2 × PCP2 runs with `-cc.cudagraph_mode=NONE`), prefix
caching, `--max-num-batched-tokens 16384`, 0.9 GPU memory utilization, one
output token per request. Agent workloads use 32 (64K) or 16 (120K) shared
prefixes with a 10% unique suffix, about 1.8M tokens of prefixes, shuffled.

| Configuration | KV capacity (tokens) | GSM8K | Cold 64K tok/s | Agent 64K tok/s (hit) | Agent 120K tok/s (hit) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4 | 778,688 | 0.940 | 16,619 | 24,344 (32.7%) | 20,887 (27.1%) |
| TP4 + KVPP | 2,713,216 | 0.935 | 16,573 | 71,157 (78.7%) | 66,741 (78.7%) |
| TP2 × PCP2 | 698,624 | 0.920 | 22,810 | 32,425 (29.8%) | 28,278 (23.4%) |
| TP2 × PCP2 + KVPP | 2,443,712 | 0.910 | 22,552 | 95,896 (77.6%) | 91,118 (78.0%) |

KVPP keeps the whole prefix working set resident, which replicated KV cannot,
and its broadcast cost is under 1% on cold prefill. PCP speeds up the
remaining computation, so the two compose. Larger
`--max-num-batched-tokens` values gain at most 4% on cold prefill but reserve
more activation memory and shrink KV capacity. Decode was not measured; it
broadcasts every step and gives up the FlashInfer all-reduce.
