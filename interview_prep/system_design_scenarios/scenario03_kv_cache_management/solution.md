# Scenario 03: KV Cache Management System - Solution

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   KV Cache Manager                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Block Allocator (Paged Memory)              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │ Block 0  │  │ Block 1  │  │ Block 2  │    ...    │  │
│  │  │ 16 tokens│  │ 16 tokens│  │ 16 tokens│           │  │
│  │  └──────────┘  └──────────┘  └──────────┘           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Sequence Manager (Block Tables)               │  │
│  │  Seq 1: [Block 0, Block 1, Block 2]                  │  │
│  │  Seq 2: [Block 0, Block 3, Block 4]  <- Shares Block 0 │
│  │  Seq 3: [Block 5, Block 6]                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Prefix Cache (Hash Table)                     │  │
│  │  hash(prompt) → [Block IDs]                          │  │
│  │  "Translate to French:" → [Block 0]                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Eviction Policy (LRU/Priority-based)          │  │
│  │  - Track access time                                  │  │
│  │  - Reference counting for shared blocks              │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Physical GPU Memory Layout                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  K Cache Blocks (num_layers × num_blocks × ...)     │   │
│  │  [Layer 0][Layer 1]...[Layer N]                     │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  V Cache Blocks (num_layers × num_blocks × ...)     │   │
│  │  [Layer 0][Layer 1]...[Layer N]                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Design: PagedAttention

### Block-Based Memory Management

**Key Insight:** Treat KV cache like virtual memory in OS - use fixed-size blocks instead of contiguous allocation.

```python
class BlockAllocator:
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size  # tokens per block
        self.num_blocks = num_blocks

        # Free block pool
        self.free_blocks = set(range(num_blocks))

        # Block reference counting (for sharing)
        self.block_refcount = defaultdict(int)

    def allocate_blocks(self, num_tokens):
        """Allocate blocks for sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            # Trigger eviction
            self.evict_blocks(num_blocks_needed)

        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            self.block_refcount[block_id] = 1
            allocated.append(block_id)

        return allocated

    def free_blocks(self, block_ids):
        """Free blocks (with reference counting)"""
        for block_id in block_ids:
            self.block_refcount[block_id] -= 1
            if self.block_refcount[block_id] == 0:
                self.free_blocks.add(block_id)
                del self.block_refcount[block_id]

    def share_blocks(self, block_ids):
        """Increment reference count for shared blocks"""
        for block_id in block_ids:
            self.block_refcount[block_id] += 1
```

### Memory Calculation

```python
# For Llama-70B model:
num_layers = 80
num_heads = 64
head_dim = 128
block_size = 16  # tokens per block

# Memory per block (K or V)
bytes_per_block = (
    num_layers *
    block_size *    # tokens
    num_heads *
    head_dim *
    2              # FP16
)
# = 80 × 16 × 64 × 128 × 2 = 20.97 MB per block

# Total blocks for 80GB GPU:
# Reserve 35GB for model weights, 5GB for activations
# Available: 40GB for KV cache
# = 40,000 MB / 20.97 MB = ~1,900 blocks
# = 1,900 × 16 = 30,400 tokens total capacity

# For 100 concurrent sequences with 2K tokens each:
# Required: 100 × 2048 / 16 = 12,800 blocks
# This exceeds capacity! Need eviction or smaller batch.
```

## Sequence Manager with Block Tables

```python
class SequenceManager:
    def __init__(self, block_allocator):
        self.block_allocator = block_allocator

        # Map sequence ID to block table
        self.block_tables = {}  # seq_id → [block_ids]

        # Map sequence ID to metadata
        self.sequences = {}  # seq_id → SequenceMetadata

    def create_sequence(self, seq_id, prompt_tokens):
        """Create new sequence"""
        num_tokens = len(prompt_tokens)

        # Try to find matching prefix in cache
        prefix_blocks = self.prefix_cache.lookup(prompt_tokens)

        if prefix_blocks:
            # Share prefix blocks (copy-on-write)
            block_ids = prefix_blocks.copy()
            self.block_allocator.share_blocks(prefix_blocks)

            # Allocate new blocks for non-prefix part
            remaining_tokens = num_tokens - len(prefix_blocks) * self.block_size
            if remaining_tokens > 0:
                new_blocks = self.block_allocator.allocate_blocks(remaining_tokens)
                block_ids.extend(new_blocks)
        else:
            # Allocate all new blocks
            block_ids = self.block_allocator.allocate_blocks(num_tokens)

        self.block_tables[seq_id] = block_ids
        self.sequences[seq_id] = SequenceMetadata(
            seq_id=seq_id,
            num_tokens=num_tokens,
            created_at=time.time(),
            last_accessed=time.time()
        )

        return block_ids

    def append_token(self, seq_id):
        """Append new token to sequence (may need new block)"""
        block_ids = self.block_tables[seq_id]
        seq_meta = self.sequences[seq_id]

        seq_meta.num_tokens += 1
        seq_meta.last_accessed = time.time()

        # Check if we need a new block
        if seq_meta.num_tokens % self.block_size == 1:
            # Need new block
            new_block = self.block_allocator.allocate_blocks(1)[0]

            # Copy-on-write: if last block is shared, don't append
            if self.block_allocator.block_refcount[block_ids[-1]] > 1:
                # Fork: copy last block before modifying
                new_last_block = self.copy_block(block_ids[-1])
                block_ids[-1] = new_last_block

            block_ids.append(new_block)

    def delete_sequence(self, seq_id):
        """Delete sequence and free blocks"""
        block_ids = self.block_tables.pop(seq_id)
        self.block_allocator.free_blocks(block_ids)
        del self.sequences[seq_id]
```

## Prefix Caching

```python
class PrefixCache:
    """Cache KV for common prefixes"""

    def __init__(self, block_allocator, block_size=16):
        self.block_allocator = block_allocator
        self.block_size = block_size

        # Map token sequence hash to block IDs
        self.cache = {}  # hash(tokens) → [block_ids]

        # LRU tracking
        self.access_time = {}  # hash → timestamp

    def lookup(self, tokens):
        """Find longest matching prefix in cache"""
        # Try progressively shorter prefixes
        for length in range(len(tokens), 0, -self.block_size):
            # Only check block-aligned lengths
            if length % self.block_size != 0:
                continue

            prefix = tuple(tokens[:length])
            prefix_hash = hash(prefix)

            if prefix_hash in self.cache:
                # Found match!
                self.access_time[prefix_hash] = time.time()
                return self.cache[prefix_hash].copy()

        return None

    def store(self, tokens, block_ids):
        """Store prefix in cache"""
        # Only cache block-aligned prefixes
        aligned_length = (len(tokens) // self.block_size) * self.block_size
        if aligned_length == 0:
            return

        prefix = tuple(tokens[:aligned_length])
        prefix_hash = hash(prefix)

        # Increment reference count for these blocks
        num_blocks = aligned_length // self.block_size
        cached_blocks = block_ids[:num_blocks]
        self.block_allocator.share_blocks(cached_blocks)

        self.cache[prefix_hash] = cached_blocks
        self.access_time[prefix_hash] = time.time()

    def evict_lru(self, num_entries=1):
        """Evict least recently used prefix"""
        if not self.cache:
            return

        # Sort by access time
        sorted_entries = sorted(
            self.access_time.items(),
            key=lambda x: x[1]
        )

        for prefix_hash, _ in sorted_entries[:num_entries]:
            block_ids = self.cache.pop(prefix_hash)
            self.block_allocator.free_blocks(block_ids)
            del self.access_time[prefix_hash]
```

## Eviction Policies

### Policy 1: LRU (Least Recently Used)

```python
class LRUEvictionPolicy:
    def select_victim(self, sequences, num_blocks_needed):
        """Select sequences to evict based on LRU"""
        # Sort by last access time
        sorted_seqs = sorted(
            sequences.items(),
            key=lambda x: x[1].last_accessed
        )

        victims = []
        blocks_freed = 0

        for seq_id, seq_meta in sorted_seqs:
            if blocks_freed >= num_blocks_needed:
                break

            num_blocks = len(seq_meta.block_ids)
            victims.append(seq_id)
            blocks_freed += num_blocks

        return victims
```

### Policy 2: Priority-Based Eviction

```python
class PriorityEvictionPolicy:
    def select_victim(self, sequences, num_blocks_needed):
        """Evict based on priority score"""
        scores = {}

        for seq_id, seq_meta in sequences.items():
            # Score factors:
            # 1. Priority level (user-defined)
            # 2. Progress (tokens generated / max_tokens)
            # 3. Age (time since creation)
            # 4. Memory usage

            priority_score = seq_meta.priority * 10
            progress_score = seq_meta.num_tokens / seq_meta.max_tokens
            age_score = 1.0 / (time.time() - seq_meta.created_at + 1)
            memory_score = -len(seq_meta.block_ids)  # Prefer evicting large seqs

            scores[seq_id] = (
                0.4 * priority_score +
                0.3 * progress_score +
                0.2 * age_score +
                0.1 * memory_score
            )

        # Sort by score (lower = more likely to evict)
        sorted_seqs = sorted(scores.items(), key=lambda x: x[1])

        victims = []
        blocks_freed = 0

        for seq_id, _ in sorted_seqs:
            if blocks_freed >= num_blocks_needed:
                break

            num_blocks = len(sequences[seq_id].block_ids)
            victims.append(seq_id)
            blocks_freed += num_blocks

        return victims
```

## Copy-on-Write Implementation

```python
class CopyOnWriteManager:
    """Manage copy-on-write for shared blocks"""

    def __init__(self, k_cache, v_cache):
        self.k_cache = k_cache  # Physical K cache tensor
        self.v_cache = v_cache  # Physical V cache tensor

    def write_with_cow(self, seq_id, block_idx, position, k_new, v_new):
        """Write to block with copy-on-write"""
        block_id = self.block_tables[seq_id][block_idx]

        # Check if block is shared
        if self.block_allocator.block_refcount[block_id] > 1:
            # Need to copy
            new_block_id = self.copy_block(block_id)

            # Decrement old block refcount
            self.block_allocator.block_refcount[block_id] -= 1

            # Update block table
            self.block_tables[seq_id][block_idx] = new_block_id

            # Use new block for write
            block_id = new_block_id

        # Perform write
        offset = position % self.block_size
        self.k_cache[block_id, offset] = k_new
        self.v_cache[block_id, offset] = v_new

    def copy_block(self, src_block_id):
        """Copy block to new location"""
        # Allocate new block
        dst_block_id = self.block_allocator.allocate_blocks(1)[0]

        # Copy data
        self.k_cache[dst_block_id] = self.k_cache[src_block_id].clone()
        self.v_cache[dst_block_id] = self.v_cache[src_block_id].clone()

        return dst_block_id
```

## Distributed KV Cache

```python
class DistributedKVCache:
    """KV cache distributed across tensor parallel ranks"""

    def __init__(self, tp_rank, tp_size, num_blocks, block_size):
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        # Each rank manages subset of attention heads
        self.num_heads_per_rank = num_total_heads // tp_size

        # Local block allocator
        self.block_allocator = BlockAllocator(num_blocks, block_size)

        # Shared block table (replicated across ranks)
        self.block_tables = {}  # seq_id → [block_ids]

    def allocate_sequence(self, seq_id, num_tokens):
        """Allocate blocks for sequence (synchronized across ranks)"""
        if self.tp_rank == 0:
            # Leader allocates
            block_ids = self.block_allocator.allocate_blocks(num_tokens)

            # Broadcast to other ranks
            dist.broadcast_object_list([block_ids], src=0)
        else:
            # Followers receive
            block_ids = [None]
            dist.broadcast_object_list(block_ids, src=0)
            block_ids = block_ids[0]

            # Allocate same blocks locally
            self.block_allocator.mark_allocated(block_ids)

        self.block_tables[seq_id] = block_ids
        return block_ids

    def get_kv(self, seq_id, layer_idx, positions):
        """Retrieve K, V for positions (local to this rank)"""
        block_ids = self.block_tables[seq_id]

        k_list = []
        v_list = []

        for pos in positions:
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = block_ids[block_idx]

            k = self.k_cache[layer_idx, block_id, offset]
            v = self.v_cache[layer_idx, block_id, offset]

            k_list.append(k)
            v_list.append(v)

        return torch.stack(k_list), torch.stack(v_list)
```

## Performance Optimizations

### 1. Block Size Selection

```python
# Trade-off analysis:
block_size = 16  # tokens

# Pros:
# - Less internal fragmentation
# - Finer granularity for sharing

# Cons:
# - More blocks to manage
# - Higher metadata overhead

# Calculation:
avg_seq_len = 1024
internal_fragmentation = avg_seq_len % block_size / block_size
# = 1024 % 16 / 16 = 0 (no fragmentation)

# For block_size = 64:
# = 1024 % 64 / 64 = 0 (also no fragmentation)

# But for random lengths:
# block_size = 16: avg fragmentation = 8 tokens = 0.5%
# block_size = 64: avg fragmentation = 32 tokens = 2%
```

### 2. Attention Kernel Integration

```python
def paged_attention_kernel(
    q,                    # Query: [batch, num_heads, head_dim]
    k_cache,              # K cache: [num_blocks, block_size, num_heads, head_dim]
    v_cache,              # V cache: [num_blocks, block_size, num_heads, head_dim]
    block_tables,         # Block tables: [batch, max_num_blocks]
    context_lens,         # Context lengths: [batch]
    block_size=16
):
    """Attention with paged KV cache"""
    batch_size, num_heads, head_dim = q.shape

    outputs = []

    for i in range(batch_size):
        context_len = context_lens[i]
        num_blocks = (context_len + block_size - 1) // block_size

        # Gather K, V from blocks
        k_seq = []
        v_seq = []

        for block_idx in range(num_blocks):
            block_id = block_tables[i, block_idx]
            k_seq.append(k_cache[block_id])  # [block_size, num_heads, head_dim]
            v_seq.append(v_cache[block_id])

        k = torch.cat(k_seq, dim=0)[:context_len]  # Trim to actual length
        v = torch.cat(v_seq, dim=0)[:context_len]

        # Standard attention
        attn_output = flash_attention(q[i], k, v)
        outputs.append(attn_output)

    return torch.stack(outputs)
```

## Monitoring & Metrics

```python
class CacheMetrics:
    def __init__(self):
        self.total_blocks = 0
        self.free_blocks = 0
        self.shared_blocks = 0

        # Prefix cache stats
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0

        # Eviction stats
        self.evictions_total = 0
        self.blocks_evicted = 0

    def report(self):
        return {
            'memory_utilization': 1 - (self.free_blocks / self.total_blocks),
            'sharing_ratio': self.shared_blocks / (self.total_blocks - self.free_blocks),
            'prefix_cache_hit_rate': self.prefix_cache_hits / (self.prefix_cache_hits + self.prefix_cache_misses),
            'eviction_rate': self.evictions_total / time_window
        }
```

## Trade-offs Summary

| Approach | Memory Efficiency | Complexity | Sharing Support |
|----------|-------------------|------------|-----------------|
| Contiguous Allocation | Low (fragmentation) | Simple | No |
| Fixed-size Blocks | High (>90%) | Moderate | Yes (COW) |
| Variable-size Chunks | Moderate | High | Difficult |
| **PagedAttention** | **High** | **Moderate** | **Yes** |

## Key Takeaways

1. **Paged memory eliminates fragmentation** - Use fixed-size blocks (16 tokens)
2. **Prefix caching dramatically improves efficiency** - Share system prompts
3. **Copy-on-write enables safe sharing** - Reference counting is critical
4. **Eviction policy affects fairness** - Priority-based better than pure LRU
5. **Block size is a key parameter** - 16 tokens balances granularity and overhead
