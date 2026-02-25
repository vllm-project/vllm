# vLLM Extensions: KV Cache Tiering

This package extends vLLM with **tiered KV cache management**: GPU memory is the primary tier and CPU (pinned) memory is a secondary tier. When GPU block capacity is full, blocks are evicted to CPU and fetched back on demand (or via prefetch).

## Overview

| Module | Purpose |
|--------|--------|
| **tiered_block_manager** | GPU/CPU block allocators, eviction to CPU, fetch from CPU, async transfers with CUDA streams |
| **eviction_policies** | Policies for choosing which blocks to evict (LRU, attention-weighted, hybrid) |
| **prefetcher** | Sequential prefetcher: predicts next blocks and async-prefetches from CPU to GPU |
| **instrumentation** | Eviction/fetch stats and optional access tracing (CSV) for analysis |

## Design

- **TieredBlockSpaceManager**  
  Proxies vLLM’s block space: when `allocate()` would fail (GPU full), it evicts one or more blocks to CPU via the chosen **EvictionPolicy**, then allocates a GPU block.  
  `get_block(block_id)` returns block metadata; if the block is on CPU, it is fetched to GPU (possibly evicting another block first) and metadata is updated.

- **Eviction policies**  
  - **LRUEvictionPolicy**: evict least recently used.  
  - **AttentionWeightedEvictionPolicy**: evict blocks with lowest cumulative attention score.  
  - **HybridEvictionPolicy**: combines attention, recency, and access frequency with configurable weights.

- **Prefetcher**  
  **SequentialPrefetcher** predicts the next blocks from the request’s block table and calls `TieredBlockSpaceManager.async_transfer_to_gpu()` so data is moved CPU→GPU in parallel with compute. `check_ready(block_id)` is used to see if a prefetched block is ready.

- **Instrumentation**  
  - **EvictionStats**: counts and latencies for evictions and fetches; `summary()` returns a small dict (totals, averages, eviction/fetch ratio).  
  - **AccessTracer** (optional): records per-request, per-block operations (allocate/access/evict/fetch) and location (gpu/cpu/in_transit) to CSV for replay or analysis.

## Integration notes

- The package uses a **BlockSpaceManagerProxy** (and mock `PhysicalTokenBlock` / GPU tensors) so it can be developed and tested without full vLLM linkage. In production, `TieredBlockSpaceManager` would inherit from vLLM’s real `BlockSpaceManager` and use real block IDs and tensors.
- **BlockMetadata** carries `block_id`, `location`, `last_access_time`, `access_count`, `ref_count`, `cumulative_attention_score`, `is_last_in_sequence`, `is_being_accessed`, and addresses for GPU/CPU and transfer stream.
- Eviction is only from **evictable** blocks: `ref_count == 1`, not last in sequence, and not currently being accessed.
- Multiple **CUDA streams** are used for overlapping transfers (round-robin per transfer).

## Usage (conceptual)

```python
from vllm_extensions import TieredBlockSpaceManager
from vllm_extensions.eviction_policies import LRUEvictionPolicy
from vllm_extensions.prefetcher import SequentialPrefetcher

manager = TieredBlockSpaceManager(
    num_gpu_blocks=100,
    num_cpu_blocks=1000,
    eviction_policy=LRUEvictionPolicy(),
)
prefetcher = SequentialPrefetcher(prefetch_distance=3)

# On allocate (e.g. from scheduler): block = manager.allocate(request_id)
# On access: meta = manager.get_block(block_id); prefetcher.prefetch(next_ids, manager)
# When block needed: prefetcher.check_ready(block_id); use block
# Stats: manager.stats.summary(); prefetcher.stats.hit_rate
```

## Relation to vLLM core

- vLLM’s **paged attention** uses fixed-size KV cache **blocks** on GPU; the scheduler and **KVCacheManager** manage block allocation and mapping.  
- These extensions add a **second tier (CPU)** and policies/prefetch so that more logical blocks can exist than fit in GPU memory, at the cost of eviction and fetch latency, mitigated by prefetching and attention-aware eviction.

For how these ideas relate to the JAX/XLA/TPU stack (memory tiering, async data movement, tiling), see [RELATION_TO_JAX_XLA_TPU.md](RELATION_TO_JAX_XLA_TPU.md).
