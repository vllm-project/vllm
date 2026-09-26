# UMBP Embedded DRAM Offloading Status

This document describes the scope, current implementation status, validation
coverage, and remaining work for the UMBP embedded runtime.

## Scope

The embedded runtime is responsible only for single-engine, single-node KV
cache offloading to MORI-managed host DRAM.

The other deployment modes are separate runtime responsibilities:

```text
embedded runtime
└── In-process, single-node, UMBP-managed DRAM offloading

standalone runtime
└── A separate service and DRAM sharing across engines

distributed runtime
└── Cross-node metadata, RDMA, and distributed tier management
```

Standalone services, cross-engine sharing, cross-node access, RDMA, and SSD
tiers are therefore not completion requirements for embedded mode.

Embedded mode currently rejects pipeline parallelism (`PP > 1`). PP stages need
a stage-specific KV ownership map before logical object completeness can be
implemented safely.

## Data Path

Embedded mode registers each rank's GPU KV buffers with MORI and transfers KV
objects between GPU memory and the UMBP DRAM tier.

```text
vLLM GPU KV cache
  └── batch_put_ranges_from_ptr()
      └── UMBP-managed host DRAM
          └── batch_get_ranges_into_ptr()
              └── vLLM GPU KV cache
```

vLLM owns request scheduling, GPU KV blocks, block hashes, and transfer plans.
MORI owns DRAM allocation, object lookup, publication, capacity management, and
eviction.

## Current Features

- [x] GPU-to-DRAM KV store.
- [x] DRAM-to-GPU KV load.
- [x] UMBP object lookup and publication.
- [x] Configurable per-rank DRAM capacity.
- [x] Configurable high and low eviction watermarks.
- [x] Configurable NUMA placement.
- [x] Configurable huge pages and huge-page size.
- [x] Configurable memory prefaulting.
- [x] Configurable shared-memory backing and name.
- [x] Asynchronous transfer execution.
- [x] Free-queue-driven lazy offload with rank-aggregated store events.
- [x] Transfer timeout and per-object failure reporting.
- [x] Request preemption and transfer cancellation lifecycle.
- [x] GPU block pinning until store completion.
- [x] Cache reset.
- [x] Ranged I/O over multi-layer KV objects.
- [x] Partial hash hits and partial physical ranges.
- [x] Non-block-aligned partial-tail plans.
- [x] Hybrid and Mamba boundary-state planning.
- [x] Layer-wise load.
- [x] TP, PP, PCP, and DCP rank-local object identities.
- [x] Logical object completeness across model-parallel ranks.
- [x] Separate physical rank keys from logical completion identities.
- [x] Store/load statistics and transferred-byte counters.
- [x] KV stored and removed events.
- [x] UMBP eviction detection.
- [x] Single-GPU Llama DRAM restore correctness.
- [x] Four-GPU TP=4 Llama DRAM restore correctness.

Layer-wise store remains disabled. MORI ranged put currently requires the ranges
in one put to tile the complete object. Independent per-layer puts do not meet
that contract and must not be advertised as supported.

## DRAM Configuration

The following embedded options are supported:

```json
{
  "mode": "embedded",
  "total_capacity_bytes": 68719476736,
  "dram_high_watermark": 0.9,
  "dram_low_watermark": 0.7,
  "dram_use_hugepages": true,
  "dram_hugepage_size": 2097152,
  "dram_numa_node": -1,
  "dram_prefault": true,
  "dram_use_shared_memory": false,
  "dram_shm_name": "umbp-dram"
}
```

`capacity_bytes` configures capacity per rank. `total_capacity_bytes` configures
a total budget that is divided by the participating TP/PP/PCP/DCP rank count.
The two options are mutually exclusive.

## Comparison with Simple CPU Offloading

This comparison covers only the CPU backend of `SimpleCPUOffloadConnector`.
Disk offloading is outside the embedded DRAM runtime's scope.

| Capability | UMBP embedded DRAM | Simple CPU offloading |
| --- | --- | --- |
| Host-memory owner | MORI/UMBP | vLLM worker |
| Data organization | Hash-keyed KV objects | Fixed CPU block slots |
| GPU-to-CPU transfer | MORI ranged pointer API | vLLM DMA copy backend |
| CPU allocation | UMBP DRAM allocator | PyTorch CPU tensors with pinning |
| Capacity control | Per-rank bytes and watermarks | Fixed slot capacity |
| NUMA placement | Configurable | No equivalent connector option |
| Huge pages | Configurable | No equivalent connector option |
| Prefaulting | Configurable | No equivalent connector option |
| Rank completeness | Explicit TP/PP/PCP/DCP objects | Internal block ownership |
| Partial ranged objects | Supported | Primarily full-block copies |
| Layer-wise load | Supported | Whole-load asynchronous submission |
| Layer-wise store | Disabled | Not a true per-layer store path |
| Eager offload | Supported and validated | Supported |
| Lazy offload | Free-queue based and tested | Free-queue based and tested |
| Preemption | Implemented, needs stress E2E | More mature coverage |
| HMA integration | Supported | Supported |
| MultiConnector coverage | Needs system validation | NIXL combinations tested |
| Model coverage | Llama, single GPU and TP=4 | Broader model/eval coverage |
| Performance baselines | Not complete | Existing latency/performance tests |

## Remaining Development Work

- [x] Document `capacity_bytes` explicitly as per-rank capacity in the public
  connector configuration documentation and startup logs.
- [x] Add `total_capacity_bytes` and divide it across participating workers.
- [x] Add authoritative per-rank lookup diagnostics for logical misses.
- [x] Add structured diagnostics for unavailable rank sockets and missing keys.
- [ ] Add configurable transfer concurrency and queue-depth tuning if MORI
  measurements show the current worker pool is limiting throughput.
- [ ] Tune `lazy_offload_max_blocks` against GPU pressure and transfer latency.
- [x] Verify that DRAM watermark eviction produces timely and complete KV
  removal events under sustained load.
- [ ] Harden cancellation when GPU blocks are reused immediately after
  preemption.
- [ ] Validate memory cleanup and socket cleanup across repeated engine startup
  and shutdown.
- [ ] Keep layer-wise store disabled until MORI supports atomic staged partial
  object assembly, or another full-object-safe protocol is implemented.

## Remaining Correctness Tests

- [x] Single-GPU Llama store, eviction, DRAM restore, and token equality.
- [x] TP=4 Llama store, eviction, logical completeness, DRAM restore, and token
  equality.
- [x] Four-GPU raw MORI range store/load round trip.
- [x] Rank-local physical keys with canonical logical completion aggregation.
- [x] Lazy offload with a real model on one GPU.
- [x] Lazy offload with TP=4.
- [ ] Store cancellation followed by immediate GPU block reuse.
- [ ] Partial failure where one TP rank fails to publish an object.
- [x] Partial eviction where one TP rank loses an object before lookup.
- [x] Repeated eviction and restore cycles under a deliberately small DRAM
  capacity.
- [ ] PP=2 serving-level restore (unsupported in embedded mode).
- [ ] TP=2 plus PP=2 serving-level restore (unsupported in embedded mode).
- [ ] DCP serving-level restore.
- [ ] Hybrid attention and Mamba model restore.
- [ ] Multiple prefix-cacheable KV groups.
- [ ] Long-running concurrent request stress test.
- [ ] Engine restart and cleanup test for lookup sockets and registered memory.
- [ ] MultiConnector integration where UMBP embedded is combined with another
  connector.

## Remaining Performance Tests

- [ ] Measure GPU-to-DRAM bandwidth by transfer size and batch size.
- [ ] Measure DRAM-to-GPU bandwidth by transfer size and batch size.
- [ ] Compare restore latency with `SimpleCPUOffloadConnector`.
- [ ] Compare time-to-first-token after an external-cache hit.
- [ ] Measure transfer overlap with model compute.
- [ ] Sweep embedded `num_workers`.
- [ ] Compare local and remote NUMA placement.
- [ ] Compare huge pages enabled and disabled.
- [ ] Compare prefaulting enabled and disabled.
- [ ] Measure eviction overhead at different high and low watermarks.
- [ ] Measure p50, p95, and p99 latency under concurrent load.

## Development and Test Image

The current local development image with `total_capacity_bytes` support is:

```text
rocm/pytorch-private:vllm_umbp_embedded_lazy_gfx950_dev_20260918
```

This image also includes Ruff 0.14.0 for source and test lint checks.

The reusable MI355X development image contains the current UMBP embedded
source, MORI bindings, and vLLM ROCm extensions compiled for `gfx950`:

```text
rocm/pytorch-private:vllm_umbp_embedded_dramcfg_gfx950_20260918
```

Registry digest:

```text
sha256:b5bf4e18b2e64588178999969257db55a6de376c8fee77fcb2eeee05c1633ed6
```

The image has been used on Mi355 GPU 0-3. The TP=4 Llama validation restored
192 cached prompt tokens and produced the same output tokens as the cold run.

Example launch options:

```bash
docker run --rm --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v /data/models:/data/models:ro \
  rocm/pytorch-private:vllm_umbp_embedded_dramcfg_gfx950_20260918 \
  <test-command>
```

## Current Assessment

Within the intentionally narrow scope of local UMBP-managed DRAM offloading:

- Core store/load correctness: approximately 90% complete.
- Basic parity with the Simple CPU backend: approximately 85% complete.
- Production validation and performance maturity: approximately 65-70%
  complete.

The primary remaining work is validation and hardening around lazy offload,
eviction, preemption, additional parallel layouts, hybrid KV caches, and
performance under sustained concurrent load.
