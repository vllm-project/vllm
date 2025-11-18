# Tutorial 06: KV Cache Management

## Learning Objectives

1. Understand the complete KV cache lifecycle in vLLM
2. Learn memory allocation strategies and their trade-offs
3. Master eviction policies for memory-constrained scenarios
4. Explore prefix caching and cache reuse optimization
5. Debug cache-related issues and optimize memory usage

## Overview

KV Cache Management is central to vLLM's efficiency. This tutorial covers the complete lifecycle of KV cache from allocation to eviction, focusing on how vLLM's block-based approach enables efficient memory usage and high throughput.

## KV Cache Fundamentals

### What is KV Cache?

In transformer models, attention requires computing K (Key) and V (Value) for all previous tokens. Instead of recomputing these every time, we cache them:

```
Without Cache (Recompute everything):
Step 1: "The" → Compute K₁, V₁ for "The"
Step 2: "The cat" → Compute K₁, V₁ for "The" again + K₂, V₂ for "cat"
Step 3: "The cat sat" → Recompute K₁, V₁, K₂, V₂ + compute K₃, V₃
❌ Wasteful O(n²) computation!

With Cache (Store and reuse):
Step 1: "The" → Compute K₁, V₁, cache them
Step 2: "The cat" → Reuse cached K₁, V₁ + compute K₂, V₂, cache K₂, V₂
Step 3: "The cat sat" → Reuse K₁, V₁, K₂, V₂ + compute K₃, V₃
✓ Efficient O(n) computation!
```

### Memory Requirements

For a single token in a 7B parameter model:

```
Per token KV cache size:
  = 2 (K and V)
  × 32 (layers)
  × 4096 (hidden dimension)
  × 2 (bytes for fp16)
  = 524,288 bytes
  ≈ 512 KB per token

For 2048-token sequence:
  = 2048 × 512 KB
  = 1 GB per sequence!

For 100 concurrent sequences:
  = 100 GB of KV cache needed!
```

This is why efficient KV cache management is critical.

## KV Cache Architecture in vLLM

### Block-Based Storage

```
Traditional Contiguous Storage:
┌────────────────────────────────────────────────┐
│ Seq 1: [KV][KV][KV][KV][░░][░░][░░][░░][░░]  │ ← Wasted space
├────────────────────────────────────────────────┤
│ Seq 2: [KV][KV][░░][░░][░░][░░][░░][░░][░░]  │ ← More waste
└────────────────────────────────────────────────┘

vLLM Block-Based Storage:
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ Blk │ Blk │ Blk │ Blk │ Blk │ Blk │ Blk │ Blk │
│  0  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ▲     ▲           ▲     ▲           ▲     ▲
  │     │           │     │           │     │
Seq 1   Seq 1     Seq 2   Seq 2     Seq 3   Seq 3

No wasted space! Blocks allocated on-demand.
```

## Core Components

### 1. KV Cache Manager

**File**: `/vllm/v1/core/kv_cache_manager.py`

```python
class KVCacheManager:
    """
    Manages KV cache allocation and deallocation.
    Coordinates with BlockPool for physical memory management.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        enable_caching: bool = False,
    ):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks

        # Create block pool
        self.block_pool = BlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
        )

        # Allocate physical KV cache tensor
        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        self.kv_cache = torch.zeros(
            num_gpu_blocks,
            2,  # K and V
            block_size,
            num_kv_heads,
            head_size,
            dtype=dtype,
            device='cuda'
        )

        # Request tracking: request_id -> list[KVCacheBlock]
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}

    def can_allocate(self, request: Request) -> bool:
        """Check if we have enough blocks for request"""

        num_required_blocks = self._calculate_num_blocks(request)

        # Check with prefix caching
        if self.block_pool.enable_caching:
            cached_blocks = self._count_cached_blocks(request)
            num_new_blocks = num_required_blocks - cached_blocks
        else:
            num_new_blocks = num_required_blocks

        return self.block_pool.num_free_blocks() >= num_new_blocks

    def allocate(self, request: Request) -> list[KVCacheBlock]:
        """Allocate KV cache blocks for a request"""

        # Calculate blocks needed
        num_blocks = self._calculate_num_blocks(request)

        # Try to reuse cached blocks (prefix caching)
        cached_blocks = self._find_cached_blocks(request)

        # Allocate remaining blocks
        num_new_blocks = num_blocks - len(cached_blocks)
        new_blocks = self.block_pool.allocate(num_new_blocks)

        # Combine cached and new blocks
        all_blocks = cached_blocks + new_blocks

        # Store for request
        self.request_blocks[request.request_id] = all_blocks

        return all_blocks

    def free(self, request_id: str) -> None:
        """Free KV cache blocks for a finished request"""

        blocks = self.request_blocks.pop(request_id, [])

        for block in blocks:
            # Decrement reference count
            block.decr_ref()

            if block.is_free():
                # Return to free pool (or cache if enabled)
                if self.block_pool.enable_caching and block.block_hash:
                    self.block_pool.add_to_cache(block)
                else:
                    self.block_pool.free_block_queue.append(block)

    def _calculate_num_blocks(self, request: Request) -> int:
        """Calculate number of blocks needed for request"""

        total_tokens = len(request.prompt_tokens) + request.max_new_tokens
        return (total_tokens + self.block_size - 1) // self.block_size
```

### 2. Block Table Management

Maps logical blocks to physical blocks:

```python
class BlockTableManager:
    """
    Manages block tables for requests.
    Block table: logical_block_idx → physical_block_id
    """

    def __init__(self):
        # request_id → block_table tensor
        self.block_tables: dict[str, torch.Tensor] = {}

    def create_block_table(
        self,
        request_id: str,
        blocks: list[KVCacheBlock]
    ) -> torch.Tensor:
        """Create block table for request"""

        # Extract physical block IDs
        block_ids = [block.block_id for block in blocks]

        # Create tensor
        block_table = torch.tensor(
            block_ids,
            dtype=torch.int32,
            device='cuda'
        )

        self.block_tables[request_id] = block_table

        return block_table

    def get_block_table(self, request_id: str) -> torch.Tensor:
        """Get block table for request"""
        return self.block_tables[request_id]

    def append_block(
        self,
        request_id: str,
        new_block: KVCacheBlock
    ) -> None:
        """Append a new block to request's block table"""

        current_table = self.block_tables[request_id]

        # Append new block ID
        new_table = torch.cat([
            current_table,
            torch.tensor([new_block.block_id], device='cuda')
        ])

        self.block_tables[request_id] = new_table
```

## Cache Lifecycle

### Complete Lifecycle Flow

```
1. Request Arrives
   │
   ├─▶ Calculate blocks needed
   │
   └─▶ Check if sufficient free blocks
       │
       ├─ Yes ─▶ Continue
       │
       └─ No ─▶ Try eviction or reject

2. Allocation Phase
   │
   ├─▶ Search for cached prefix blocks
   │   └─▶ Reuse if found (prefix caching)
   │
   └─▶ Allocate new blocks for remaining tokens
       └─▶ Update block table

3. Execution Phase
   │
   ├─▶ Model reads K, V from blocks using block table
   │
   └─▶ Write new K, V to allocated blocks

4. Growth Phase (generation)
   │
   ├─▶ Generate new token
   │
   ├─▶ Check if current block has space
   │   ├─ Yes ─▶ Use current block
   │   └─ No ─▶ Allocate new block
   │
   └─▶ Update block table

5. Completion Phase
   │
   ├─▶ Request finishes
   │
   ├─▶ Decrement block reference counts
   │
   └─▶ Free blocks
       ├─▶ If cacheable → Add to cache
       └─▶ Otherwise → Return to free pool
```

### Code Walkthrough: Prefill and Decode

```python
# Prefill Phase: Process prompt
def prefill_phase(request: Request, kv_cache_manager: KVCacheManager):
    """Process prompt and fill initial KV cache"""

    # Allocate blocks for entire prompt
    blocks = kv_cache_manager.allocate(request)

    # Create block table
    block_table = [b.block_id for b in blocks]

    # Run model forward pass
    for layer_idx in range(num_layers):
        # Compute K, V for prompt tokens
        key, value = model_layer_forward(
            request.prompt_tokens,
            layer_idx
        )

        # Store K, V in blocks
        for token_idx, (k, v) in enumerate(zip(key, value)):
            block_idx = token_idx // block_size
            offset = token_idx % block_size

            physical_block_id = block_table[block_idx]

            # Write to cache
            kv_cache[physical_block_id, 0, offset] = k  # Key
            kv_cache[physical_block_id, 1, offset] = v  # Value

    return blocks, block_table


# Decode Phase: Generate tokens
def decode_phase(
    request: Request,
    blocks: list[KVCacheBlock],
    block_table: list[int],
    kv_cache_manager: KVCacheManager
):
    """Generate tokens one at a time"""

    generated_tokens = []

    for step in range(request.max_new_tokens):
        # Current sequence length
        seq_len = len(request.prompt_tokens) + len(generated_tokens)

        # Check if we need a new block
        if seq_len % block_size == 0:
            new_block = kv_cache_manager.allocate_single_block()
            blocks.append(new_block)
            block_table.append(new_block.block_id)

        # Run model forward for next token
        next_token_logits = model_forward_decode(
            last_token=generated_tokens[-1] if generated_tokens else request.prompt_tokens[-1],
            kv_cache=kv_cache,
            block_table=block_table,
            position=seq_len
        )

        # Sample next token
        next_token = sample(next_token_logits)
        generated_tokens.append(next_token)

        # Compute and store K, V for new token
        for layer_idx in range(num_layers):
            key, value = compute_kv_for_token(next_token, layer_idx)

            # Find block and offset for current position
            block_idx = seq_len // block_size
            offset = seq_len % block_size
            physical_block_id = block_table[block_idx]

            # Write to cache
            kv_cache[physical_block_id, 0, offset] = key
            kv_cache[physical_block_id, 1, offset] = value

        # Check for EOS
        if next_token == eos_token_id:
            break

    return generated_tokens
```

## Memory Allocation Strategies

### Strategy 1: Greedy Allocation

Always allocate when blocks are available:

```python
def greedy_allocation(request: Request, kv_cache_manager: KVCacheManager):
    """Greedy allocation: allocate if possible"""

    if kv_cache_manager.can_allocate(request):
        return kv_cache_manager.allocate(request)
    else:
        return None  # Reject or wait
```

**Pros**: Simple, low overhead
**Cons**: Can lead to starvation of large requests

### Strategy 2: Reserved Blocks

Reserve blocks for ongoing generation:

```python
class ReservedBlockStrategy:
    """Reserve blocks for ongoing requests"""

    def __init__(self, reserve_ratio: float = 0.1):
        self.reserve_ratio = reserve_ratio

    def can_allocate(
        self,
        request: Request,
        kv_cache_manager: KVCacheManager,
        num_running_requests: int
    ) -> bool:
        """Check allocation with reservation"""

        total_blocks = kv_cache_manager.num_gpu_blocks
        reserved_blocks = int(total_blocks * self.reserve_ratio)

        available_for_new = kv_cache_manager.num_free_blocks() - reserved_blocks

        num_required = kv_cache_manager._calculate_num_blocks(request)

        return num_required <= available_for_new
```

**Pros**: Prevents starvation
**Cons**: May underutilize memory

### Strategy 3: Watermark-Based

Use high/low watermarks:

```python
class WatermarkAllocationStrategy:
    """Allocate based on memory watermarks"""

    def __init__(
        self,
        high_watermark: float = 0.9,
        low_watermark: float = 0.5
    ):
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.throttle_mode = False

    def should_allocate(
        self,
        request: Request,
        kv_cache_manager: KVCacheManager
    ) -> bool:
        """Decide whether to allocate"""

        total_blocks = kv_cache_manager.num_gpu_blocks
        free_blocks = kv_cache_manager.num_free_blocks()
        utilization = 1.0 - (free_blocks / total_blocks)

        # Update throttle mode
        if utilization > self.high_watermark:
            self.throttle_mode = True
        elif utilization < self.low_watermark:
            self.throttle_mode = False

        # In throttle mode, only allow small requests
        if self.throttle_mode:
            max_tokens = request.max_new_tokens
            return max_tokens < 100  # Small requests only

        return kv_cache_manager.can_allocate(request)
```

**Pros**: Adaptive to memory pressure
**Cons**: More complex logic

## Eviction Policies

### When to Evict?

```python
def should_evict(kv_cache_manager: KVCacheManager) -> bool:
    """Determine if eviction is needed"""

    # No free blocks and no cached blocks to evict
    if kv_cache_manager.num_free_blocks() == 0:
        if kv_cache_manager.num_cached_blocks() > 0:
            return True  # Evict cached blocks

    return False
```

### Policy 1: LRU (Least Recently Used)

```python
class LRUEvictionPolicy:
    """Evict least recently used cached blocks"""

    def evict(
        self,
        kv_cache_manager: KVCacheManager,
        num_blocks_needed: int
    ) -> list[KVCacheBlock]:
        """Evict blocks using LRU"""

        evicted_blocks = []
        cached_blocks = kv_cache_manager.block_pool.get_cached_blocks()

        # Sort by last access time
        sorted_blocks = sorted(
            cached_blocks,
            key=lambda b: b.last_accessed
        )

        # Evict oldest blocks
        for block in sorted_blocks:
            if len(evicted_blocks) >= num_blocks_needed:
                break

            # Remove from cache
            kv_cache_manager.block_pool.remove_from_cache(block)

            # Clear block data
            block.block_hash = None
            block.ref_cnt = 0

            evicted_blocks.append(block)

        return evicted_blocks
```

**Pros**: Simple, works well for temporal locality
**Cons**: Doesn't consider block utility

### Policy 2: LFU (Least Frequently Used)

```python
class LFUEvictionPolicy:
    """Evict least frequently used cached blocks"""

    def evict(
        self,
        kv_cache_manager: KVCacheManager,
        num_blocks_needed: int
    ) -> list[KVCacheBlock]:
        """Evict blocks using LFU"""

        evicted_blocks = []
        cached_blocks = kv_cache_manager.block_pool.get_cached_blocks()

        # Sort by access count
        sorted_blocks = sorted(
            cached_blocks,
            key=lambda b: b.access_count
        )

        # Evict least frequent blocks
        for block in sorted_blocks:
            if len(evicted_blocks) >= num_blocks_needed:
                break

            kv_cache_manager.block_pool.remove_from_cache(block)
            block.block_hash = None
            block.ref_cnt = 0
            evicted_blocks.append(block)

        return evicted_blocks
```

**Pros**: Keeps frequently reused blocks
**Cons**: Can keep old but once-popular blocks

### Policy 3: Size-Aware

Prioritize evicting larger cached sequences:

```python
class SizeAwareEvictionPolicy:
    """Evict larger cached block chains first"""

    def evict(
        self,
        kv_cache_manager: KVCacheManager,
        num_blocks_needed: int
    ) -> list[KVCacheBlock]:
        """Evict by size"""

        # Group blocks by prefix hash (chain)
        chains = self._group_blocks_by_chain(kv_cache_manager)

        # Sort chains by size (descending)
        sorted_chains = sorted(
            chains,
            key=lambda chain: len(chain),
            reverse=True
        )

        evicted_blocks = []

        # Evict largest chains first
        for chain in sorted_chains:
            if len(evicted_blocks) >= num_blocks_needed:
                break

            for block in chain:
                kv_cache_manager.block_pool.remove_from_cache(block)
                evicted_blocks.append(block)

        return evicted_blocks
```

## Prefix Caching Deep Dive

### How Prefix Caching Works

```
Request A: "Translate to French: Hello world"
Request B: "Translate to French: Goodbye world"

Common Prefix: "Translate to French:"

Step 1: Process Request A
  ┌───────────────────┐
  │ "Translate to     │ ← Compute K, V, hash block
  │  French:"         │   Hash: 0x7a3f
  └───────────────────┘
  ┌───────────────────┐
  │ "Hello world"     │ ← Compute K, V, hash block
  └───────────────────┘

  Cache: {0x7a3f: Block 5}

Step 2: Process Request B
  ┌───────────────────┐
  │ "Translate to     │ ← Hash matches! Reuse Block 5
  │  French:"         │   Hash: 0x7a3f ✓
  └───────────────────┘
  ┌───────────────────┐
  │ "Goodbye world"   │ ← Compute K, V for unique part
  └───────────────────┘

  Cache hit: Saved computation for prefix!
```

### Implementation

```python
def allocate_with_prefix_cache(
    request: Request,
    kv_cache_manager: KVCacheManager
) -> tuple[list[KVCacheBlock], int]:
    """
    Allocate blocks with prefix caching.

    Returns:
        blocks: Allocated blocks
        num_cached_blocks: Number of blocks from cache
    """

    tokens = request.prompt_tokens
    allocated_blocks = []
    num_cached = 0

    # Process tokens in block-sized chunks
    for start_idx in range(0, len(tokens), block_size):
        end_idx = min(start_idx + block_size, len(tokens))
        block_tokens = tokens[start_idx:end_idx]

        # Compute hash for this block
        parent_hash = allocated_blocks[-1].block_hash if allocated_blocks else None
        block_hash = compute_block_hash(block_tokens, parent_hash)

        # Look up in cache
        cached_block = kv_cache_manager.block_pool.get_cached_block(block_hash)

        if cached_block is not None:
            # Cache hit! Reuse block
            cached_block.incr_ref()
            cached_block.last_accessed = time.time()
            cached_block.access_count += 1
            allocated_blocks.append(cached_block)
            num_cached += 1
        else:
            # Cache miss, allocate new block
            new_block = kv_cache_manager.block_pool.allocate_single_block()

            # Will compute K, V and store
            new_block.block_hash = block_hash
            new_block.incr_ref()

            allocated_blocks.append(new_block)

            # Add to cache once full
            if len(block_tokens) == block_size:
                kv_cache_manager.block_pool.add_to_cache(new_block)

    return allocated_blocks, num_cached


def compute_block_hash(
    tokens: list[int],
    parent_hash: int | None = None
) -> int:
    """
    Compute hash for a block of tokens.
    Includes parent hash for chain consistency.
    """

    import hashlib

    # Combine parent hash (if exists) with tokens
    if parent_hash is not None:
        data = (parent_hash, tuple(tokens))
    else:
        data = (tuple(tokens),)

    # Compute hash
    hash_obj = hashlib.md5(str(data).encode())
    hash_value = int.from_bytes(hash_obj.digest()[:8], 'big')

    return hash_value
```

## Hands-On Exercises

### Exercise 1: Measure Cache Hit Rate

**Objective**: Evaluate prefix caching effectiveness

```python
class CacheMetrics:
    """Track cache hit/miss statistics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_blocks_saved = 0

    def record_hit(self):
        self.hits += 1
        self.total_blocks_saved += 1

    def record_miss(self):
        self.misses += 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def print_stats(self):
        print(f"Cache Statistics:")
        print(f"  Hits: {self.hits}")
        print(f"  Misses: {self.misses}")
        print(f"  Hit Rate: {self.hit_rate()*100:.1f}%")
        print(f"  Blocks Saved: {self.total_blocks_saved}")

# Use in cache manager
metrics = CacheMetrics()

def allocate_with_tracking(request, kv_cache_manager):
    blocks, num_cached = allocate_with_prefix_cache(request, kv_cache_manager)

    for _ in range(num_cached):
        metrics.record_hit()

    for _ in range(len(blocks) - num_cached):
        metrics.record_miss()

    return blocks
```

**Task**: Run workloads with varying prefix overlap and measure hit rates.

### Exercise 2: Implement Custom Eviction Policy

**Objective**: Create a hybrid LRU+LFU policy

```python
class HybridEvictionPolicy:
    """
    Hybrid policy: LRU for recently accessed, LFU for old blocks
    """

    def __init__(self, recency_threshold: float = 60.0):
        self.recency_threshold = recency_threshold  # seconds

    def evict(
        self,
        kv_cache_manager: KVCacheManager,
        num_blocks_needed: int
    ) -> list[KVCacheBlock]:
        """Evict using hybrid policy"""

        # TODO: Implement
        # 1. Split blocks into recent and old
        # 2. For recent blocks, use LRU
        # 3. For old blocks, use LFU
        # 4. Evict from both groups as needed

        pass
```

**Task**: Implement and benchmark against pure LRU.

### Exercise 3: Visualize Memory Utilization

**Objective**: Monitor memory usage over time

```python
import matplotlib.pyplot as plt

class MemoryTracker:
    """Track memory utilization over time"""

    def __init__(self):
        self.timestamps = []
        self.free_blocks = []
        self.cached_blocks = []
        self.used_blocks = []

    def record(self, kv_cache_manager):
        self.timestamps.append(time.time())
        self.free_blocks.append(kv_cache_manager.num_free_blocks())
        self.cached_blocks.append(kv_cache_manager.num_cached_blocks())

        total = kv_cache_manager.num_gpu_blocks
        used = total - self.free_blocks[-1] - self.cached_blocks[-1]
        self.used_blocks.append(used)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        times = [t - self.timestamps[0] for t in self.timestamps]

        ax.fill_between(times, 0, self.used_blocks, label='Active', alpha=0.7)
        ax.fill_between(
            times,
            self.used_blocks,
            [u + c for u, c in zip(self.used_blocks, self.cached_blocks)],
            label='Cached',
            alpha=0.7
        )

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Number of Blocks')
        ax.set_title('KV Cache Memory Utilization')
        ax.legend()
        plt.tight_layout()
        plt.show()
```

**Task**: Track and visualize memory during a request workload.

## Common Pitfalls and Solutions

### Pitfall 1: Cache Hash Collisions

**Problem**: Different prefixes producing same hash.

**Solution**: Use high-quality hash and verify tokens:

```python
def safe_cache_lookup(
    tokens: list[int],
    block_hash: int,
    kv_cache_manager: KVCacheManager
) -> KVCacheBlock | None:
    """Cache lookup with collision detection"""

    block = kv_cache_manager.get_cached_block(block_hash)

    if block is not None:
        # Verify tokens actually match
        if block.tokens != tokens:
            # Hash collision!
            return None  # Don't use this block

    return block
```

### Pitfall 2: Memory Leaks from Unclosed Requests

**Problem**: Forgetting to free blocks when requests finish.

**Solution**: Use context managers or explicit cleanup:

```python
class RequestContext:
    """Context manager for request lifecycle"""

    def __init__(self, request, kv_cache_manager):
        self.request = request
        self.kv_cache_manager = kv_cache_manager
        self.blocks = None

    def __enter__(self):
        self.blocks = self.kv_cache_manager.allocate(self.request)
        return self.blocks

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always free blocks on exit
        self.kv_cache_manager.free(self.request.request_id)

# Usage
with RequestContext(request, kv_cache_manager) as blocks:
    # Process request
    process_request(request, blocks)
# Blocks automatically freed
```

### Pitfall 3: Inefficient Eviction Triggering

**Problem**: Evicting too early or too late.

**Solution**: Use predictive eviction:

```python
def predictive_eviction(
    kv_cache_manager: KVCacheManager,
    incoming_request: Request
):
    """Evict preemptively before allocation fails"""

    # Predict future memory need
    blocks_needed = kv_cache_manager._calculate_num_blocks(incoming_request)
    current_free = kv_cache_manager.num_free_blocks()

    # Add safety margin
    safety_margin = 10
    target_free = blocks_needed + safety_margin

    if current_free < target_free:
        # Evict proactively
        num_to_evict = target_free - current_free
        eviction_policy.evict(kv_cache_manager, num_to_evict)
```

## Performance Optimization

### 1. Batch Cache Operations

```python
def batch_allocate(
    requests: list[Request],
    kv_cache_manager: KVCacheManager
) -> dict[str, list[KVCacheBlock]]:
    """Allocate for multiple requests efficiently"""

    # Calculate total blocks needed
    total_needed = sum(
        kv_cache_manager._calculate_num_blocks(req)
        for req in requests
    )

    # Check and evict once
    if kv_cache_manager.num_free_blocks() < total_needed:
        eviction_policy.evict(kv_cache_manager, total_needed)

    # Allocate for all requests
    allocations = {}
    for req in requests:
        allocations[req.request_id] = kv_cache_manager.allocate(req)

    return allocations
```

### 2. Lazy Eviction

```python
class LazyEvictionManager:
    """Evict only when absolutely necessary"""

    def __init__(self, eviction_threshold: float = 0.95):
        self.threshold = eviction_threshold

    def maybe_evict(self, kv_cache_manager):
        """Evict only if utilization is high"""

        utilization = 1.0 - (kv_cache_manager.num_free_blocks() /
                            kv_cache_manager.num_gpu_blocks)

        if utilization > self.threshold:
            # High utilization, evict some cached blocks
            num_to_evict = int(kv_cache_manager.num_cached_blocks() * 0.1)
            eviction_policy.evict(kv_cache_manager, num_to_evict)
```

### 3. Prefetch Cached Blocks

```python
async def prefetch_cached_blocks(
    request: Request,
    kv_cache_manager: KVCacheManager
):
    """Prefetch cached blocks asynchronously"""

    # Compute hashes for request
    hashes = compute_prefix_hashes(request.prompt_tokens)

    # Prefetch cached blocks in parallel
    tasks = [
        async_load_cached_block(hash_val)
        for hash_val in hashes
    ]

    cached_blocks = await asyncio.gather(*tasks)

    return cached_blocks
```

## Advanced Topics

### Distributed KV Cache

Share cache across multiple nodes:

```python
class DistributedKVCache:
    """KV cache distributed across nodes"""

    def __init__(self, local_cache, remote_caches):
        self.local_cache = local_cache
        self.remote_caches = remote_caches

    async def get_cached_block(self, block_hash):
        """Look up block locally then remotely"""

        # Try local cache first
        block = self.local_cache.get_cached_block(block_hash)
        if block is not None:
            return block

        # Try remote caches
        for remote_cache in self.remote_caches:
            block = await remote_cache.get_cached_block_async(block_hash)
            if block is not None:
                # Copy to local cache
                self.local_cache.add_to_cache(block)
                return block

        return None
```

### Hierarchical Cache

Use multiple cache tiers:

```python
class HierarchicalCache:
    """Multi-tier cache (GPU → CPU → Disk)"""

    def __init__(self):
        self.gpu_cache = GPUKVCache()
        self.cpu_cache = CPUKVCache()
        self.disk_cache = DiskKVCache()

    def get(self, block_hash):
        # Try GPU cache
        block = self.gpu_cache.get(block_hash)
        if block:
            return block

        # Try CPU cache
        block = self.cpu_cache.get(block_hash)
        if block:
            # Promote to GPU
            self.gpu_cache.put(block_hash, block)
            return block

        # Try disk cache
        block = self.disk_cache.get(block_hash)
        if block:
            # Promote to CPU (and GPU if space)
            self.cpu_cache.put(block_hash, block)
            return block

        return None
```

## References

### Source Code Files

- **KV Cache Manager**: `/vllm/v1/core/kv_cache_manager.py`
- **Block Pool**: `/vllm/v1/core/block_pool.py`
- **KV Cache Utils**: `/vllm/v1/core/kv_cache_utils.py`
- **Block Table**: `/vllm/v1/worker/block_table.py`

### Configuration

```python
@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = None
    enable_prefix_caching: bool = False
    cache_eviction_policy: str = "lru"
    cache_size_ratio: float = 0.5
```

## Summary

In this tutorial, you learned:

- Complete KV cache lifecycle from allocation to eviction
- Memory allocation strategies and their trade-offs
- Eviction policies (LRU, LFU, size-aware)
- Prefix caching for computation and memory savings
- Debugging and optimization techniques

KV cache management is crucial for vLLM's performance. Understanding these internals helps you optimize memory usage and throughput for your workload.

## Next Steps

- **Tutorial 07**: Request Batching Strategies
- **Tutorial 08**: Memory Management Techniques
- **Module 3**: Understanding PagedAttention fundamentals
