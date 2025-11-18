# Tutorial 02: Block Manager Walkthrough

## Learning Objectives

1. Understand the block-based memory management design in vLLM
2. Learn how BlockPool allocates and manages KV cache blocks
3. Master copy-on-write optimization for efficient memory sharing
4. Explore prefix caching and block reuse mechanisms
5. Debug memory-related issues and optimize block allocation

## Overview

The Block Manager is vLLM's memory management system that handles allocation of KV cache blocks. It implements sophisticated techniques like copy-on-write and prefix caching to maximize memory efficiency and throughput.

## Block-Based Memory Architecture

### Why Blocks?

Traditional LLM serving allocates contiguous memory for each request's KV cache:

```
Traditional Approach (Wasteful):
Request A: [████████████████░░░░░░░░] ← Pre-allocated, wasted space
Request B: [██████░░░░░░░░░░░░░░░░░░] ← More waste
Request C: [████████████░░░░░░░░░░░░] ← Fragmentation
```

vLLM's block-based approach:

```
Block-Based Approach (Efficient):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ Blk │ Blk │ Blk │ Blk │ Blk │ Blk │
│  0  │  1  │  2  │  3  │  4  │  5  │
└─────┴─────┴─────┴─────┴─────┴─────┘
  ▲     ▲           ▲     ▲
  │     │           │     │
Req A   Req A     Req B   Req B

Benefits:
✓ No pre-allocation waste
✓ Dynamic growth
✓ Easy sharing (prefix caching)
✓ Reduced fragmentation
```

### Block Structure

Each block has a fixed size (e.g., 16 tokens) and stores KV states:

```
Block Layout (block_size = 16):
┌─────────────────────────────────────────┐
│  Block ID: 42                           │
├─────────────────────────────────────────┤
│  K: [k₀, k₁, k₂, ..., k₁₅]            │  ← Key states
│  V: [v₀, v₁, v₂, ..., v₁₅]            │  ← Value states
├─────────────────────────────────────────┤
│  Reference Count: 2                     │  ← For CoW
│  Block Hash: 0x7f3a...                 │  ← For caching
│  Group ID: 0                            │  ← For prefix groups
└─────────────────────────────────────────┘
```

## Core Components

### 1. KVCacheBlock Class

**File**: `/vllm/v1/core/kv_cache_utils.py`

```python
class KVCacheBlock:
    """
    A single KV cache block that can store key-value states
    for block_size tokens.
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 0  # Number of references to this block
        self.block_hash: BlockHash | None = None  # Hash for prefix caching
        self.num_tokens_total = 0  # Total tokens in block
        self.last_accessed = 0  # For LRU eviction

    def incr_ref(self) -> None:
        """Increment reference count (copy-on-write)"""
        self.ref_cnt += 1

    def decr_ref(self) -> None:
        """Decrement reference count"""
        assert self.ref_cnt > 0
        self.ref_cnt -= 1

    def is_free(self) -> bool:
        """Check if block is free"""
        return self.ref_cnt == 0
```

### 2. BlockPool Class

**File**: `/vllm/v1/core/block_pool.py` (lines 125-150)

The BlockPool manages all KV cache blocks:

```python
class BlockPool:
    """
    BlockPool manages KVCacheBlocks.
    Provides allocation, freeing, and caching capabilities.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching

        # All KV cache blocks
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # Free block queue (FIFO for eviction)
        self.free_block_queue = FreeKVCacheBlockQueue(
            blocks=self.blocks
        )

        # Cache mapping: hash -> blocks
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

### 3. FreeKVCacheBlockQueue

Manages free blocks in eviction order:

```python
class FreeKVCacheBlockQueue:
    """
    Queue of free blocks in eviction order.
    When caching is enabled, uses LRU for eviction.
    """

    def __init__(self, blocks: list[KVCacheBlock]):
        # All blocks start as free
        self._free_blocks = deque(blocks)

    def append(self, block: KVCacheBlock) -> None:
        """Add block to end (most recently used)"""
        self._free_blocks.append(block)

    def popleft(self) -> KVCacheBlock | None:
        """Get least recently used free block"""
        if not self._free_blocks:
            return None
        return self._free_blocks.popleft()

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)
```

## Block Allocation Flow

### Sequence Diagram

```
Request     Scheduler   KVCacheManager   BlockPool
   │            │             │              │
   │   Need     │             │              │
   │  Blocks    │             │              │
   ├───────────▶│             │              │
   │            │  can_allocate(req)         │
   │            ├────────────▶│              │
   │            │             │ get_num_free_blocks()
   │            │             ├─────────────▶│
   │            │             │◀─────────────┤
   │            │             │  (num_free)  │
   │            │◀────────────┤              │
   │            │   (True)    │              │
   │            │  allocate(req)             │
   │            ├────────────▶│              │
   │            │             │  allocate(n) │
   │            │             ├─────────────▶│
   │            │             │              │
   │            │             │  [Get blocks]│
   │            │             │              │
   │            │             │◀─────────────┤
   │            │             │  (blocks)    │
   │            │◀────────────┤              │
   │◀───────────┤             │              │
   │  (blocks)  │             │              │
```

### Code Walkthrough

```python
# Step 1: Scheduler checks if allocation is possible
def can_allocate(self, request: Request) -> bool:
    """Check if we have enough free blocks"""

    num_required = self._get_num_required_blocks(request)
    num_free = self.block_pool.num_free_blocks()

    return num_free >= num_required

# Step 2: Allocate blocks for request
def allocate(self, request: Request) -> list[KVCacheBlock]:
    """Allocate blocks for a request"""

    num_required = self._get_num_required_blocks(request)

    # Try to find cached blocks first (prefix caching)
    cached_blocks = self._get_cached_blocks(request)

    # Allocate remaining blocks
    num_new_blocks = num_required - len(cached_blocks)
    new_blocks = self.block_pool.allocate(num_new_blocks)

    # Combine cached and new blocks
    all_blocks = cached_blocks + new_blocks

    # Store in request
    request.kv_cache_blocks = all_blocks

    return all_blocks

# Step 3: BlockPool allocates from free queue
def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
    """Allocate num_blocks from the free pool"""

    allocated = []

    for _ in range(num_blocks):
        block = self.free_block_queue.popleft()

        if block is None:
            # Need to evict cached blocks
            block = self._evict_one_block()

        if block is None:
            raise RuntimeError("Out of memory: no free blocks")

        # Initialize block
        block.incr_ref()
        allocated.append(block)

    return allocated
```

## Copy-on-Write (CoW) Optimization

### The Problem: Naive Copying

Without CoW, when multiple requests share a prefix, we waste memory:

```
Request A: "Translate to French: Hello"
Request B: "Translate to French: Goodbye"

Naive approach:
┌──────────────────────────────────┐
│ Request A's blocks:               │
│ [Translate to French:][Hello]    │  ← Duplicate prefix!
└──────────────────────────────────┘

┌──────────────────────────────────┐
│ Request B's blocks:               │
│ [Translate to French:][Goodbye]  │  ← Duplicate prefix!
└──────────────────────────────────┘
```

### CoW Solution: Share Until Modified

```
Copy-on-Write:
┌─────────────────────┐
│ [Translate to French:] │ ◀─┐  Shared block
└─────────────────────┘   │  (ref_cnt = 2)
          ▲              │
          │              │
    ┌─────┴──────┐  ┌────┴─────┐
    │ [Hello]    │  │ [Goodbye] │  ← Private blocks
    │            │  │           │
    └────────────┘  └───────────┘
     Request A       Request B
```

### CoW Implementation

```python
class BlockPool:
    def share_block(
        self,
        src_request: Request,
        dst_request: Request,
        block_idx: int
    ) -> None:
        """
        Share a block from src to dst (CoW).
        Both requests now point to the same block.
        """

        # Get the block to share
        block = src_request.kv_cache_blocks[block_idx]

        # Increment reference count
        block.incr_ref()

        # Add to destination request
        dst_request.kv_cache_blocks.append(block)

    def fork_block(
        self,
        request: Request,
        block_idx: int
    ) -> KVCacheBlock:
        """
        Fork a shared block (copy-on-write).
        Called when request needs to modify a shared block.
        """

        old_block = request.kv_cache_blocks[block_idx]

        # If not shared, can modify in-place
        if old_block.ref_cnt == 1:
            return old_block

        # Block is shared, need to copy
        new_block = self.allocate(1)[0]

        # Copy KV states from old to new
        self._copy_block_data(src=old_block, dst=new_block)

        # Update reference counts
        old_block.decr_ref()
        new_block.incr_ref()

        # Update request's block table
        request.kv_cache_blocks[block_idx] = new_block

        return new_block
```

### CoW Example Walkthrough

```python
# Example: Two requests sharing a prefix

# Request A: "Translate to French: Hello world"
req_a = Request(tokens=tokenize("Translate to French: Hello world"))

# Allocate blocks for Request A
blocks_a = block_pool.allocate(2)  # 2 blocks
req_a.kv_cache_blocks = blocks_a

# Blocks after Request A:
# Block 0: [Translate to French:] (ref_cnt=1)
# Block 1: [Hello world]          (ref_cnt=1)

# Request B: "Translate to French: Goodbye"
req_b = Request(tokens=tokenize("Translate to French: Goodbye"))

# Share prefix block from A to B (CoW)
block_pool.share_block(req_a, req_b, block_idx=0)

# Blocks after sharing:
# Block 0: [Translate to French:] (ref_cnt=2) ← SHARED!
# Block 1: [Hello world]          (ref_cnt=1)

# Allocate unique block for B's continuation
new_block = block_pool.allocate(1)[0]
req_b.kv_cache_blocks.append(new_block)

# Final state:
# Request A: [Block 0, Block 1]
# Request B: [Block 0, Block 2]  ← Block 0 is shared!
```

## Prefix Caching

### Block Hashing

Each full block gets a hash of its KV states:

```python
def compute_block_hash(block: KVCacheBlock) -> BlockHash:
    """
    Compute hash of a block's KV cache contents.
    Used for prefix caching to find identical blocks.
    """

    # Hash is computed from the token IDs that generated this block
    token_ids = block.token_ids
    hash_value = hash(tuple(token_ids))

    return BlockHash(hash_value)

# In practice, from kv_cache_utils.py:
def get_block_hash(
    block_id: int,
    token_ids: list[int],
    parent_block_hash: BlockHash | None = None
) -> BlockHash:
    """
    Compute cumulative hash including parent blocks.
    This allows matching prefix chains.
    """

    if parent_block_hash is not None:
        # Include parent hash for chain
        combined = (parent_block_hash, tuple(token_ids))
    else:
        combined = (tuple(token_ids),)

    return hash(combined)
```

### Prefix Cache Lookup

```python
class BlockHashToBlockMap:
    """
    Cache mapping from block hash to KVCacheBlock(s).
    From block_pool.py lines 29-123.
    """

    def __init__(self):
        self._cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Get any block with the given hash"""

        blocks = self._cache.get(key)

        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            elif isinstance(blocks, dict):
                return next(iter(blocks.values()))

        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """Insert block into cache"""

        blocks = self._cache.get(key)

        if blocks is None:
            # First block with this hash
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # Already have one block, convert to dict
            self._cache[key] = {
                blocks.block_id: blocks,
                block.block_id: block
            }
        elif isinstance(blocks, dict):
            # Add to existing dict
            blocks[block.block_id] = block
```

### Prefix Caching Workflow

```
┌─────────────────────────────────────────────────┐
│  New Request: "Translate to French: Bonjour"   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ Compute expected hashes    │
    │ for request's token prefix │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ Look up in cache:          │
    │ Hash("Translate to French")│
    └────────────┬───────────────┘
                 │
         ┌───────┴────────┐
         │                │
         ▼                ▼
    ┌────────┐      ┌──────────┐
    │ Found! │      │ Not found│
    │ (Hit)  │      │ (Miss)   │
    └────┬───┘      └─────┬────┘
         │                │
         ▼                ▼
   ┌──────────┐    ┌─────────────┐
   │Share block│    │Allocate new │
   │(CoW)      │    │block        │
   └──────────┘    └─────────────┘
```

### Implementation

```python
def allocate_with_prefix_cache(
    self,
    request: Request
) -> list[KVCacheBlock]:
    """
    Allocate blocks for request, using prefix cache when possible.
    """

    allocated_blocks = []
    token_idx = 0

    # Process tokens block by block
    while token_idx < len(request.tokens):
        # Get next block's worth of tokens
        block_tokens = request.tokens[token_idx:token_idx + self.block_size]

        # Compute hash for this block
        parent_hash = allocated_blocks[-1].block_hash if allocated_blocks else None
        block_hash = get_block_hash(
            block_id=-1,  # Not assigned yet
            token_ids=block_tokens,
            parent_block_hash=parent_hash
        )

        # Look up in cache
        cached_block = self.cached_blocks.get_one_block(block_hash)

        if cached_block is not None:
            # Cache hit! Share the block
            cached_block.incr_ref()
            allocated_blocks.append(cached_block)
        else:
            # Cache miss, allocate new block
            new_block = self.free_block_queue.popleft()

            if new_block is None:
                new_block = self._evict_cached_block()

            new_block.block_hash = block_hash
            new_block.incr_ref()

            # Add to cache for future requests
            self.cached_blocks.insert(block_hash, new_block)

            allocated_blocks.append(new_block)

        token_idx += self.block_size

    return allocated_blocks
```

## Block Eviction

### When to Evict

Eviction happens when:
1. Free blocks are exhausted
2. A new allocation request arrives
3. No cached blocks can be freed

### LRU Eviction Policy

```python
def _evict_one_block(self) -> KVCacheBlock | None:
    """
    Evict one cached block using LRU policy.
    Returns the evicted block or None if nothing to evict.
    """

    if not self.enable_caching:
        return None

    # Find least recently used cached block
    candidate_block = None
    oldest_access = float('inf')

    for block_hash, blocks in self.cached_blocks._cache.items():
        if isinstance(blocks, KVCacheBlock):
            blocks = [blocks]
        else:
            blocks = blocks.values()

        for block in blocks:
            # Skip blocks in use
            if block.ref_cnt > 0:
                continue

            # Check if older than current candidate
            if block.last_accessed < oldest_access:
                oldest_access = block.last_accessed
                candidate_block = block

    if candidate_block is None:
        return None  # Nothing to evict

    # Remove from cache
    self.cached_blocks.pop(candidate_block.block_hash, candidate_block.block_id)

    # Reset block
    candidate_block.block_hash = None
    candidate_block.num_tokens_total = 0

    return candidate_block
```

### Eviction Visualization

```
Free Block Queue (before eviction):
┌─────┐
│Empty│  ← No free blocks!
└─────┘

Cached Blocks:
┌─────────────────────────────────────┐
│ Block 5: hash=0x7f3a, ref=0, age=10 │ ← Oldest
│ Block 8: hash=0x2e1b, ref=0, age=5  │
│ Block 3: hash=0x9c4d, ref=2, age=1  │ ← In use (skip)
│ Block 9: hash=0x5a2f, ref=0, age=3  │
└─────────────────────────────────────┘
                  │
                  │ Evict Block 5 (oldest)
                  ▼
Free Block Queue (after eviction):
┌─────────┐
│ Block 5 │  ← Now available
└─────────┘
```

## Hands-On Exercises

### Exercise 1: Trace Block Allocation

**Objective**: Follow block allocation for a single request

```python
def trace_allocation():
    """Trace block allocation step by step"""

    # Setup
    block_pool = BlockPool(num_gpu_blocks=10, enable_caching=True)

    # Request with 50 tokens, block_size=16
    # Needs: ceil(50/16) = 4 blocks
    request = Request(
        request_id="trace_req",
        tokens=[i for i in range(50)]
    )

    print(f"Initial free blocks: {block_pool.num_free_blocks()}")

    # Allocate
    blocks = block_pool.allocate(request)

    print(f"Allocated {len(blocks)} blocks for request")
    for i, block in enumerate(blocks):
        print(f"  Block {i}: id={block.block_id}, ref_cnt={block.ref_cnt}")

    print(f"Free blocks remaining: {block_pool.num_free_blocks()}")
```

**Task**: Run this and verify the block count calculation.

### Exercise 2: Test Copy-on-Write

**Objective**: Verify CoW sharing and forking

```python
def test_copy_on_write():
    """Test that CoW correctly shares and forks blocks"""

    block_pool = BlockPool(num_gpu_blocks=20, enable_caching=False)

    # Request A
    req_a = Request(request_id="A", tokens=list(range(32)))
    blocks_a = block_pool.allocate(req_a)

    print(f"Request A blocks: {[b.block_id for b in blocks_a]}")
    print(f"Request A block[0] ref_cnt: {blocks_a[0].ref_cnt}")

    # Request B shares first block
    req_b = Request(request_id="B", tokens=list(range(16)))
    block_pool.share_block(req_a, req_b, block_idx=0)

    print(f"\nAfter sharing:")
    print(f"Request A block[0] ref_cnt: {blocks_a[0].ref_cnt}")  # Should be 2
    print(f"Request B block[0] is same: {req_b.kv_cache_blocks[0].block_id == blocks_a[0].block_id}")

    # Fork the shared block
    forked = block_pool.fork_block(req_b, block_idx=0)

    print(f"\nAfter forking:")
    print(f"Request A block[0] ref_cnt: {blocks_a[0].ref_cnt}")  # Back to 1
    print(f"Request B block[0] ref_cnt: {forked.ref_cnt}")  # Should be 1
    print(f"Blocks are different: {forked.block_id != blocks_a[0].block_id}")
```

**Task**: Implement the share_block and fork_block methods, then run the test.

### Exercise 3: Measure Prefix Cache Hit Rate

**Objective**: Evaluate prefix caching effectiveness

```python
def measure_cache_hit_rate():
    """Measure prefix cache hit rate for common prompts"""

    block_pool = BlockPool(num_gpu_blocks=100, enable_caching=True)

    # Common prefix prompt
    prefix = "Translate the following to French: "

    requests = [
        prefix + "Hello",
        prefix + "Goodbye",
        prefix + "Good morning",
        prefix + "Thank you",
    ]

    hits = 0
    misses = 0

    for i, prompt in enumerate(requests):
        tokens = tokenize(prompt)
        req = Request(request_id=f"req_{i}", tokens=tokens)

        # Allocate with cache tracking
        blocks = block_pool.allocate_with_prefix_cache(req)

        # Count hits (blocks with ref_cnt > 1 were cached)
        for block in blocks:
            if block.ref_cnt > 1:
                hits += 1
            else:
                misses += 1

    hit_rate = hits / (hits + misses) * 100
    print(f"Cache hit rate: {hit_rate:.1f}%")
    print(f"Hits: {hits}, Misses: {misses}")
```

**Task**: Run with different prompt sets and compare hit rates.

## Common Pitfalls and Solutions

### Pitfall 1: Memory Leaks from Unreleased Blocks

**Problem**: Forgetting to decrement reference counts causes memory leaks.

```python
# BAD: Block not released
def bad_request_cleanup(request: Request):
    # Request finished, but blocks not freed
    request.status = RequestStatus.FINISHED
    # ❌ Forgot to free blocks!
```

**Solution**: Always free blocks when request completes:

```python
# GOOD: Proper cleanup
def good_request_cleanup(request: Request):
    # Free all blocks
    for block in request.kv_cache_blocks:
        block.decr_ref()
        if block.is_free():
            block_pool.free_block_queue.append(block)

    request.kv_cache_blocks = []
    request.status = RequestStatus.FINISHED
```

### Pitfall 2: Premature Block Eviction

**Problem**: Evicting blocks that will be needed soon.

**Solution**: Implement smarter eviction heuristics:

```python
def smart_eviction_score(block: KVCacheBlock) -> float:
    """
    Calculate eviction score (lower = better candidate).
    Consider both recency and future utility.
    """

    score = 0.0

    # Recency component (LRU)
    time_since_access = current_time() - block.last_accessed
    score += time_since_access

    # Frequency component
    score -= block.access_count * 10  # High access = keep

    # Prefix length component (longer prefixes = keep)
    score -= block.num_tokens_total * 0.1

    return score
```

### Pitfall 3: Hash Collisions

**Problem**: Different token sequences produce same hash.

**Solution**: Use high-quality hash function and handle collisions:

```python
def robust_block_hash(token_ids: list[int]) -> BlockHash:
    """
    Compute hash with low collision probability.
    """

    import hashlib

    # Convert tokens to bytes
    token_bytes = bytes(token_ids)

    # Use SHA256 for low collision rate
    hash_obj = hashlib.sha256(token_bytes)
    hash_value = int.from_bytes(hash_obj.digest()[:8], 'big')

    return BlockHash(hash_value)

# When using cache, verify tokens match
def cache_lookup_with_verification(
    block_hash: BlockHash,
    expected_tokens: list[int]
) -> KVCacheBlock | None:
    """Look up block and verify tokens match"""

    block = cached_blocks.get_one_block(block_hash)

    if block is not None:
        # Verify tokens actually match
        if block.token_ids != expected_tokens:
            # Hash collision! Don't use this block
            return None

    return block
```

## Debugging Block Manager Issues

### Enable Block Tracking

```python
class DebugBlockPool(BlockPool):
    """Block pool with debug tracking"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allocation_history = []

    def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
        blocks = super().allocate(num_blocks)

        # Track allocation
        self.allocation_history.append({
            'timestamp': time.time(),
            'num_blocks': num_blocks,
            'blocks': [b.block_id for b in blocks],
            'free_remaining': self.num_free_blocks()
        })

        return blocks

    def print_allocation_summary(self):
        print(f"Total allocations: {len(self.allocation_history)}")
        total_allocated = sum(h['num_blocks'] for h in self.allocation_history)
        print(f"Total blocks allocated: {total_allocated}")
```

### Visualize Block Usage

```python
def visualize_block_pool(block_pool: BlockPool):
    """Create a visual representation of block pool state"""

    print("\nBlock Pool State:")
    print("=" * 60)

    for block in block_pool.blocks:
        if block.ref_cnt > 0:
            status = f"IN USE (ref={block.ref_cnt})"
        elif block.block_hash is not None:
            status = "CACHED"
        else:
            status = "FREE"

        print(f"Block {block.block_id:3d}: {status:20s} | tokens={block.num_tokens_total}")

    print("=" * 60)
    print(f"Free blocks: {block_pool.num_free_blocks()}/{block_pool.num_gpu_blocks}")
```

### Memory Leak Detection

```python
class LeakDetector:
    """Detect memory leaks in block allocation"""

    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        self.expected_free = block_pool.num_gpu_blocks

    def check_for_leaks(self):
        """Check if blocks are leaked"""

        # Count blocks by state
        in_use = sum(1 for b in self.block_pool.blocks if b.ref_cnt > 0)
        cached = sum(1 for b in self.block_pool.blocks if b.block_hash is not None and b.ref_cnt == 0)
        free = self.block_pool.num_free_blocks()

        total = in_use + cached + free

        if total != self.block_pool.num_gpu_blocks:
            print(f"⚠️  LEAK DETECTED!")
            print(f"   Total blocks: {self.block_pool.num_gpu_blocks}")
            print(f"   Accounted: {total} (in_use={in_use}, cached={cached}, free={free})")
            print(f"   Missing: {self.block_pool.num_gpu_blocks - total}")
        else:
            print(f"✓ No leaks detected (all {total} blocks accounted for)")
```

## Performance Optimization

### 1. Block Size Tuning

```python
def find_optimal_block_size(avg_sequence_length: int) -> int:
    """
    Find optimal block size based on workload.

    Smaller blocks: Better memory utilization, more overhead
    Larger blocks: Less overhead, potential waste
    """

    # Rule of thumb: block_size = sqrt(avg_sequence_length)
    optimal = int(avg_sequence_length ** 0.5)

    # Clamp to reasonable range
    optimal = max(8, min(128, optimal))

    # Round to power of 2 for alignment
    return 2 ** int(optimal.bit_length() - 1)
```

### 2. Cache Size Tuning

```python
def calculate_cache_budget(
    total_blocks: int,
    avg_concurrent_requests: int,
    avg_blocks_per_request: int
) -> int:
    """
    Calculate how many blocks to reserve for prefix cache.
    """

    # Blocks needed for running requests
    active_blocks = avg_concurrent_requests * avg_blocks_per_request

    # Reserve 20% for active requests
    active_budget = int(active_blocks * 1.2)

    # Remaining can be used for cache
    cache_budget = total_blocks - active_budget

    return max(0, cache_budget)
```

### 3. Batch Block Operations

```python
def allocate_batch(
    block_pool: BlockPool,
    requests: list[Request]
) -> dict[str, list[KVCacheBlock]]:
    """
    Allocate blocks for multiple requests at once.
    More efficient than one-by-one allocation.
    """

    # Calculate total blocks needed
    total_needed = sum(
        (len(req.tokens) + block_pool.block_size - 1) // block_pool.block_size
        for req in requests
    )

    # Check if enough blocks available
    if block_pool.num_free_blocks() < total_needed:
        # Evict enough blocks
        to_evict = total_needed - block_pool.num_free_blocks()
        for _ in range(to_evict):
            block_pool._evict_one_block()

    # Allocate for all requests
    allocations = {}
    for req in requests:
        num_blocks = (len(req.tokens) + block_pool.block_size - 1) // block_pool.block_size
        allocations[req.request_id] = block_pool.allocate(num_blocks)

    return allocations
```

## Advanced Topics

### Multi-GPU Block Management

```python
class MultiGPUBlockPool:
    """Block pool spanning multiple GPUs"""

    def __init__(self, num_gpus: int, blocks_per_gpu: int):
        self.pools = [
            BlockPool(num_gpu_blocks=blocks_per_gpu, enable_caching=True)
            for _ in range(num_gpus)
        ]

    def allocate_distributed(
        self,
        request: Request,
        gpu_ids: list[int]
    ) -> dict[int, list[KVCacheBlock]]:
        """Allocate blocks across multiple GPUs"""

        blocks_per_gpu = {}

        # Distribute blocks evenly
        total_blocks = self._calculate_blocks_needed(request)
        blocks_per_gpu_count = total_blocks // len(gpu_ids)

        for gpu_id in gpu_ids:
            blocks = self.pools[gpu_id].allocate(blocks_per_gpu_count)
            blocks_per_gpu[gpu_id] = blocks

        return blocks_per_gpu
```

### Dynamic Block Resizing

```python
def resize_block_pool(
    block_pool: BlockPool,
    new_size: int
) -> None:
    """
    Dynamically resize the block pool.
    Useful for autoscaling scenarios.
    """

    current_size = block_pool.num_gpu_blocks

    if new_size > current_size:
        # Grow pool
        new_blocks = [
            KVCacheBlock(idx)
            for idx in range(current_size, new_size)
        ]
        block_pool.blocks.extend(new_blocks)
        block_pool.free_block_queue.extend(new_blocks)
        block_pool.num_gpu_blocks = new_size

    elif new_size < current_size:
        # Shrink pool (only if blocks are free)
        blocks_to_remove = current_size - new_size

        # Remove from end if free
        for _ in range(blocks_to_remove):
            if block_pool.blocks[-1].is_free():
                removed = block_pool.blocks.pop()
                block_pool.free_block_queue.remove(removed)
            else:
                raise RuntimeError("Cannot shrink: blocks in use")

        block_pool.num_gpu_blocks = new_size
```

## References

### Source Code Files

- **Block Pool**: `/vllm/v1/core/block_pool.py`
- **KV Cache Block**: `/vllm/v1/core/kv_cache_utils.py`
- **KV Cache Manager**: `/vllm/v1/core/kv_cache_manager.py`
- **Block Table**: `/vllm/v1/worker/block_table.py`

### Configuration Parameters

```python
@dataclass
class CacheConfig:
    block_size: int = 16  # Tokens per block
    num_gpu_blocks: int | None = None  # Auto-calculated if None
    enable_prefix_caching: bool = False
    cache_eviction_policy: str = "lru"  # or "fifo"
```

### Related Documentation

- Tutorial 01: Scheduler Deep Dive
- Tutorial 06: KV Cache Management
- Module 3: Memory Management Fundamentals

## Summary

In this tutorial, you learned:

- Why block-based memory management is superior to contiguous allocation
- How BlockPool allocates and manages KV cache blocks
- Copy-on-write optimization for memory-efficient prefix sharing
- Prefix caching mechanism using block hashing
- Block eviction strategies and LRU policy
- Common pitfalls and debugging techniques
- Performance optimization strategies

The Block Manager is foundational to vLLM's efficiency. Understanding its internals allows you to tune memory usage, optimize cache hit rates, and debug allocation issues.

## Next Steps

- **Tutorial 03**: Model Executor Architecture - How blocks are used during execution
- **Tutorial 06**: KV Cache Management - Higher-level cache lifecycle
- **Tutorial 08**: Memory Management Techniques - Advanced memory optimization
