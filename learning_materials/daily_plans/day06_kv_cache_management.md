# Day 6: KV Cache Management & Memory Optimization

> **Goal**: Master KV cache block management, memory optimization techniques, calculate requirements
> **Time**: 6-8 hours
> **Prerequisites**: Day 1-5 completed, PagedAttention understanding, block concepts clear
> **Deliverables**: Memory calculator tool, block manager analysis, optimization recommendations

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Block Manager Deep Dive

**9:00-9:45** - KV Cache Fundamentals Review
**9:45-11:00** - Block Manager Architecture
**11:00-11:30** - Break + Memory Calculations
**11:30-12:30** - Block Allocation Algorithms

### Afternoon Session (3-4 hours): Optimization & Tools

**14:00-15:00** - Memory Optimization Techniques
**15:00-16:00** - Profiling Memory Usage
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Build Memory Calculator

### Evening (Optional, 1-2 hours): Advanced Topics

**19:00-21:00** - Prefix caching, memory sharing, advanced optimizations

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain KV cache structure and memory layout
- [ ] Understand block manager implementation in detail
- [ ] Calculate memory requirements for any model/config
- [ ] Identify memory bottlenecks and optimization opportunities
- [ ] Use profiling tools to analyze memory usage
- [ ] Implement memory-efficient configurations
- [ ] Understand advanced techniques (prefix caching, sharing)

---

## 📚 Morning: Block Manager Internals (9:00-12:30)

### Task 1: KV Cache Fundamentals (45 min)

**What is KV Cache?**

```
In transformer self-attention:
  - Query (Q): Current token we're processing
  - Key (K): All previous tokens (for attention scores)
  - Value (V): All previous tokens (for weighted sum)

During autoregressive generation:
  Step 1: Process "Hello" → Compute K₁, V₁
  Step 2: Process "world" → Compute K₂, V₂
          Need K₁ to compute attention!

KV Cache: Store K and V from previous steps
  Avoid recomputing them every time
```

**Memory Layout - Per Token**:

```
Single token KV cache size:
  = 2 (K and V)
    × num_layers
    × num_kv_heads  (for GQA: can be < num_heads)
    × head_dim
    × bytes_per_element (2 for fp16, 4 for fp32)

Example (OPT-13B):
  = 2 × 40 layers × 40 heads × 128 head_dim × 2 bytes
  = 819,200 bytes
  = 800 KB per token!

Sequence of 1000 tokens:
  = 800 KB × 1000
  = 800 MB per sequence
```

**Total Memory Budget**:

```
GPU Memory Breakdown (A100 40GB):

1. Model Weights: ~26 GB (OPT-13B in fp16)
   - Frozen, loaded once
   - Cannot reduce (unless quantization)

2. Activation Memory: ~2-4 GB
   - Temporary tensors during forward pass
   - Depends on batch size and model

3. KV Cache: Remaining memory
   = 40 GB - 26 GB - 3 GB
   = ~11 GB available

4. OS/Driver overhead: ~1 GB

Usable for KV cache: ~10 GB
```

**Block-Based Organization**:

```
Instead of per-token allocation, use BLOCKS:

Block size: 16 tokens (configurable)
Block memory: 16 × 800 KB = 12.8 MB

Available KV cache: 10 GB
Number of blocks: 10 GB / 12.8 MB = ~780 blocks

Sequences don't allocate individual tokens,
they allocate blocks!

Sequence with 42 tokens:
  Needs: ⌈42 / 16⌉ = 3 blocks
  Actual memory: 3 × 12.8 MB = 38.4 MB
  Waste: (48 - 42) × 800 KB = 4.8 MB (12.5%)

Much better than pre-allocating max_len!
```

### Task 2: Block Manager Architecture (75 min)

**File**: `vllm/core/block_manager_v2.py`

**Core Classes**:

```python
# vllm/core/block_manager_v2.py (simplified)

class Block:
    """
    Represents a single memory block.

    Each block can store KV cache for block_size tokens.
    """

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.ref_count = 0  # For copy-on-write
        self.filled_tokens = 0  # How many tokens currently stored

    def is_full(self) -> bool:
        return self.filled_tokens == self.block_size

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.filled_tokens

    def allocate_slot(self) -> int:
        """Allocate one slot in this block."""
        if self.is_full():
            raise ValueError("Block is full")
        slot = self.filled_tokens
        self.filled_tokens += 1
        return slot


class BlockTable:
    """
    Maps logical block indices to physical blocks.

    One BlockTable per sequence.
    """

    def __init__(self):
        self.blocks: List[Block] = []

    def append_block(self, block: Block) -> None:
        """Append a block to this table."""
        self.blocks.append(block)

    def get_physical_block_id(self, logical_idx: int) -> int:
        """Get physical block ID for logical index."""
        return self.blocks[logical_idx].block_id

    def get_num_blocks(self) -> int:
        return len(self.blocks)


class BlockAllocator:
    """
    Manages free block pool.

    Allocates and frees blocks.
    """

    def __init__(self, num_blocks: int, block_size: int):
        # Create all blocks
        self.blocks = [Block(i, block_size) for i in range(num_blocks)]

        # Free list
        self.free_blocks: List[Block] = self.blocks.copy()

    def allocate(self) -> Optional[Block]:
        """Allocate one free block."""
        if not self.free_blocks:
            return None  # Out of memory

        block = self.free_blocks.pop(0)
        block.ref_count = 1
        block.filled_tokens = 0
        return block

    def free(self, block: Block) -> None:
        """Free a block (return to free list)."""
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        return self.get_num_free_blocks() >= num_blocks


class BlockSpaceManager:
    """
    High-level block management for sequences.

    Maintains:
    - GPU block allocator
    - CPU block allocator (for swapping)
    - Block tables for each sequence
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ):
        self.block_size = block_size

        # GPU allocator
        self.gpu_allocator = BlockAllocator(num_gpu_blocks, block_size)

        # CPU allocator (for swapping)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks, block_size)

        # Block tables: seq_id -> BlockTable
        self.block_tables: Dict[int, BlockTable] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> None:
        """
        Allocate blocks for a sequence.

        Creates block table with num_blocks GPU blocks.
        """
        # Create block table
        block_table = BlockTable()

        # Allocate GPU blocks
        for _ in range(num_blocks):
            block = self.gpu_allocator.allocate()
            if block is None:
                # Out of memory - free what we allocated
                self.free(seq_id)
                raise ValueError("Out of GPU blocks")
            block_table.append_block(block)

        self.block_tables[seq_id] = block_table

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate num_blocks."""
        return self.gpu_allocator.can_allocate(num_blocks)

    def append_slot(self, seq_id: int) -> None:
        """
        Append one slot for growing sequence.

        May allocate new block if current is full.
        """
        block_table = self.block_tables[seq_id]

        # Get last block
        if block_table.get_num_blocks() == 0:
            # No blocks yet - allocate first one
            block = self.gpu_allocator.allocate()
            block_table.append_block(block)
            return

        last_block = block_table.blocks[-1]

        # Check if full
        if last_block.is_full():
            # Need new block
            block = self.gpu_allocator.allocate()
            if block is None:
                raise ValueError("Out of GPU blocks")
            block_table.append_block(block)
            last_block = block

        # Allocate slot in current block
        last_block.allocate_slot()

    def free(self, seq_id: int) -> None:
        """Free all blocks for a sequence."""
        if seq_id not in self.block_tables:
            return

        block_table = self.block_tables[seq_id]

        # Free all blocks
        for block in block_table.blocks:
            self.gpu_allocator.free(block)

        # Remove block table
        del self.block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get physical block IDs for a sequence."""
        block_table = self.block_tables[seq_id]
        return [b.block_id for b in block_table.blocks]

    def get_num_free_gpu_blocks(self) -> int:
        """Get number of free GPU blocks."""
        return self.gpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_id: int) -> Dict[int, int]:
        """
        Swap sequence from GPU to CPU.

        Returns mapping: gpu_block_id -> cpu_block_id
        """
        gpu_blocks = self.block_tables[seq_id].blocks
        cpu_blocks = []

        mapping = {}

        for gpu_block in gpu_blocks:
            # Allocate CPU block
            cpu_block = self.cpu_allocator.allocate()
            if cpu_block is None:
                raise ValueError("Out of CPU blocks")

            # Record mapping (for actual data transfer)
            mapping[gpu_block.block_id] = cpu_block.block_id

            cpu_blocks.append(cpu_block)

            # Free GPU block
            self.gpu_allocator.free(gpu_block)

        # Update block table to use CPU blocks
        self.block_tables[seq_id].blocks = cpu_blocks

        return mapping

    def swap_in(self, seq_id: int) -> Dict[int, int]:
        """
        Swap sequence from CPU to GPU.

        Returns mapping: cpu_block_id -> gpu_block_id
        """
        cpu_blocks = self.block_tables[seq_id].blocks
        gpu_blocks = []

        mapping = {}

        for cpu_block in cpu_blocks:
            # Allocate GPU block
            gpu_block = self.gpu_allocator.allocate()
            if gpu_block is None:
                raise ValueError("Out of GPU blocks")

            # Record mapping
            mapping[cpu_block.block_id] = gpu_block.block_id

            gpu_blocks.append(gpu_block)

            # Free CPU block
            self.cpu_allocator.free(cpu_block)

        # Update block table
        self.block_tables[seq_id].blocks = gpu_blocks

        return mapping
```

**📝 Exercise: Trace Block Operations**

```
Initial State:
  GPU blocks: 10 total, all free
  Block size: 4 tokens

Operation 1: allocate(seq_1, 2 blocks)
  - Allocate block 0, block 1
  - Block table for seq_1: [0, 1]
  - Free blocks: [2, 3, 4, 5, 6, 7, 8, 9]

Operation 2: Seq 1 grows from 8 → 9 tokens
  - Current: 2 blocks (8 tokens)
  - After: need 3 blocks (9 tokens)
  - append_slot(seq_1)
    - Last block (block 1) is full (4 tokens)
    - Allocate new block 2
    - Block table: [0, 1, 2]
  - Free blocks: [3, 4, 5, 6, 7, 8, 9]

Operation 3: allocate(seq_2, 3 blocks)
  - Allocate block 3, 4, 5
  - Block table for seq_2: [3, 4, 5]
  - Free blocks: [6, 7, 8, 9]

Operation 4: free(seq_1)
  - Free blocks 0, 1, 2
  - Free blocks: [6, 7, 8, 9, 0, 1, 2]
  - Block table for seq_1: deleted

Operation 5: swap_out(seq_2)
  - Allocate CPU blocks: cpu_0, cpu_1, cpu_2
  - Mapping: {3→cpu_0, 4→cpu_1, 5→cpu_2}
  - Free GPU blocks 3, 4, 5
  - Block table for seq_2: [cpu_0, cpu_1, cpu_2]
  - Free GPU blocks: [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
```

### Task 3: Memory Calculation (60 min)

**Calculate KV Cache Size**:

```python
#!/usr/bin/env python3
"""
Calculate KV cache memory requirements
"""

def calculate_kv_cache_size(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype_bytes: int = 2,  # fp16
) -> dict:
    """
    Calculate memory per block and total capacity.

    Returns:
        Dictionary with memory breakdown
    """

    # Memory per token
    bytes_per_token = (
        2  # K and V
        * num_layers
        * num_kv_heads
        * head_dim
        * dtype_bytes
    )

    # Memory per block
    bytes_per_block = bytes_per_token * block_size

    return {
        'bytes_per_token': bytes_per_token,
        'kb_per_token': bytes_per_token / 1024,
        'bytes_per_block': bytes_per_block,
        'mb_per_block': bytes_per_block / (1024 ** 2),
    }

def calculate_capacity(
    gpu_memory_gb: float,
    model_size_gb: float,
    activation_memory_gb: float,
    kv_cache_size: dict,
) -> dict:
    """
    Calculate how many blocks and sequences can fit.
    """

    # Available for KV cache
    available_gb = gpu_memory_gb - model_size_gb - activation_memory_gb - 1.0  # OS overhead

    available_bytes = available_gb * (1024 ** 3)

    # Number of blocks
    num_blocks = int(available_bytes / kv_cache_size['bytes_per_block'])

    return {
        'available_gb': available_gb,
        'num_blocks': num_blocks,
        'max_tokens': num_blocks * 16,  # Assuming block_size=16
    }

# Example: OPT-13B
print("OPT-13B Configuration:")
print("=" * 60)

config = {
    'num_layers': 40,
    'num_kv_heads': 40,
    'head_dim': 128,
    'block_size': 16,
    'dtype_bytes': 2,  # fp16
}

kv_size = calculate_kv_cache_size(**config)

print(f"KV Cache Size:")
print(f"  Per token: {kv_size['kb_per_token']:.2f} KB")
print(f"  Per block (16 tokens): {kv_size['mb_per_block']:.2f} MB")

capacity = calculate_capacity(
    gpu_memory_gb=40,  # A100 40GB
    model_size_gb=26,  # OPT-13B fp16
    activation_memory_gb=3,
    kv_cache_size=kv_size,
)

print(f"\nCapacity on A100 40GB:")
print(f"  Available for KV cache: {capacity['available_gb']:.2f} GB")
print(f"  Number of blocks: {capacity['num_blocks']}")
print(f"  Max tokens (if one sequence): {capacity['max_tokens']:,}")

# Batch size calculations
print(f"\nBatch Size Analysis:")
for avg_len in [128, 256, 512, 1024]:
    blocks_per_seq = (avg_len + config['block_size'] - 1) // config['block_size']
    max_seqs = capacity['num_blocks'] // blocks_per_seq
    print(f"  Avg length {avg_len:4d} tokens → {blocks_per_seq:3d} blocks/seq → max batch size: {max_seqs:3d}")
```

**Expected Output**:

```
OPT-13B Configuration:
============================================================
KV Cache Size:
  Per token: 800.00 KB
  Per block (16 tokens): 12.50 MB

Capacity on A100 40GB:
  Available for KV cache: 10.00 GB
  Number of blocks: 800
  Max tokens (if one sequence): 12,800

Batch Size Analysis:
  Avg length  128 tokens →   8 blocks/seq → max batch size: 100
  Avg length  256 tokens →  16 blocks/seq → max batch size:  50
  Avg length  512 tokens →  32 blocks/seq → max batch size:  25
  Avg length 1024 tokens →  64 blocks/seq → max batch size:  12
```

---

## 🔬 Afternoon: Optimization & Profiling (14:00-18:00)

### Task 4: Memory Optimization Techniques (60 min)

**1. Grouped Query Attention (GQA)**:

```
Standard Multi-Head Attention:
  - num_heads = 40
  - Each head has its own K, V
  - KV cache: 2 × 40 heads × ...

Grouped Query Attention:
  - num_kv_heads = 8 (shared across query heads)
  - num_heads = 40 (for queries)
  - KV cache: 2 × 8 heads × ... (5x smaller!)

Memory savings:
  Standard: 800 KB/token
  GQA (8 KV heads): 160 KB/token
  Reduction: 80%!

Used in: Llama-2, Mistral models
```

**2. Quantization**:

```
FP16 KV Cache:
  - 2 bytes per element
  - Full precision

INT8 KV Cache:
  - 1 byte per element
  - 50% memory reduction
  - Minimal accuracy loss

INT4 KV Cache (experimental):
  - 0.5 bytes per element
  - 75% memory reduction
  - May hurt accuracy

Example:
  FP16: 800 KB/token
  INT8: 400 KB/token → 2x more sequences
```

**3. Block Size Tuning**:

```
Small block_size (e.g., 4):
  ✓ Less waste in partial blocks
  ✗ More metadata overhead
  ✗ More pointer indirection

Large block_size (e.g., 64):
  ✗ More waste in partial blocks
  ✓ Less metadata overhead
  ✓ Fewer indirections

Optimal: 16 tokens
  - Good balance
  - ~6% average waste
  - Reasonable metadata size
```

**4. Prefix Caching**:

```
Problem:
  Multiple requests with same prefix (e.g., system prompt)

  Req 1: "You are a helpful assistant. USER: Help me"
  Req 2: "You are a helpful assistant. USER: Write code"
  Req 3: "You are a helpful assistant. USER: Explain"

  Each allocates blocks for "You are a helpful assistant"
  Wasteful!

Solution: Prefix Caching
  - Detect common prefixes
  - Share blocks for prefix
  - Only allocate for unique suffix
  - Use Copy-on-Write (ref counting)

Memory savings:
  - 3 requests × 100 token prefix = 300 tokens
  - With sharing: 100 tokens + 3 × suffix
  - Reduction: ~2x for this example
```

**5. Dynamic Block Allocation**:

```
Eager allocation (naive):
  - Allocate all blocks upfront
  - Wastes memory if sequence ends early

Lazy allocation (vLLM):
  - Allocate blocks on-demand
  - Only allocate when block is full
  - Better utilization

Example:
  Request: max_tokens=100
  Actually generates: 30 tokens

  Eager: 7 blocks allocated (100/16 rounded up)
  Lazy: 2 blocks allocated (30/16 rounded up)
  Savings: 5 blocks freed for other requests
```

### Task 5: Profiling Memory Usage (60 min)

**Using PyTorch Memory Profiler**:

```python
#!/usr/bin/env python3
"""
Profile GPU memory usage during inference
"""

import torch
from vllm import LLM, SamplingParams

def profile_memory():
    """Profile memory allocation patterns."""

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Initial GPU memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Create model
    print("\nLoading model...")
    llm = LLM(
        model="facebook/opt-125m",
        max_model_len=512,
        gpu_memory_utilization=0.9,
    )

    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Get KV cache info
    # (vLLM allocates KV cache during initialization)
    cache_blocks = llm.llm_engine.cache_config.num_gpu_blocks
    block_size = llm.llm_engine.cache_config.block_size

    print(f"\nKV Cache Configuration:")
    print(f"  Blocks: {cache_blocks}")
    print(f"  Block size: {block_size} tokens")
    print(f"  Total capacity: {cache_blocks * block_size} tokens")

    # Run inference
    prompts = ["Hello world"] * 10
    sampling_params = SamplingParams(max_tokens=50)

    print(f"\nBefore inference:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    outputs = llm.generate(prompts, sampling_params)

    print(f"\nAfter inference:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Detailed memory stats
    print(f"\nMemory breakdown:")
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    profile_memory()
```

**Using vLLM's Built-in Stats**:

```python
# Access internal statistics

from vllm import LLM

llm = LLM(model="facebook/opt-125m")

# Get scheduler stats
scheduler = llm.llm_engine.scheduler
block_manager = scheduler.block_manager

print("Block Manager Stats:")
print(f"  Total GPU blocks: {block_manager.gpu_allocator.num_blocks}")
print(f"  Free GPU blocks: {block_manager.get_num_free_gpu_blocks()}")
print(f"  Used GPU blocks: {block_manager.gpu_allocator.num_blocks - block_manager.get_num_free_gpu_blocks()}")

# Calculate utilization
utilization = 1.0 - (block_manager.get_num_free_gpu_blocks() / block_manager.gpu_allocator.num_blocks)
print(f"  GPU block utilization: {utilization * 100:.1f}%")
```

### Task 6: Build Memory Calculator Tool (90 min)

```python
#!/usr/bin/env python3
"""
Day 6 Exercise: Complete memory calculator and optimizer
"""

import argparse
from typing import Dict, List, Tuple

class MemoryCalculator:
    """Calculate memory requirements for vLLM deployment."""

    def __init__(
        self,
        model_config: dict,
        gpu_memory_gb: float,
        block_size: int = 16,
    ):
        self.model_config = model_config
        self.gpu_memory_gb = gpu_memory_gb
        self.block_size = block_size

    def calculate_model_memory(self) -> float:
        """Calculate model weight memory in GB."""
        # Approximate: num_params * bytes_per_param
        num_params = self.model_config.get('num_params_billions', 0)
        dtype_bytes = self.model_config.get('dtype_bytes', 2)

        return num_params * dtype_bytes

    def calculate_activation_memory(self, batch_size: int) -> float:
        """Estimate activation memory in GB."""
        # Rough estimate: scales with model size and batch
        hidden_size = self.model_config['hidden_size']
        num_layers = self.model_config['num_layers']

        # Activation per token (rough)
        activation_per_token = (
            hidden_size * num_layers * 4  # Intermediate activations
            * 4  # FP32 for some ops
            / (1024 ** 3)  # Convert to GB
        )

        return activation_per_token * batch_size * 100  # Assume 100 tokens

    def calculate_kv_cache_memory(self) -> Dict:
        """Calculate KV cache memory per token and block."""
        num_layers = self.model_config['num_layers']
        num_kv_heads = self.model_config.get('num_kv_heads',
                                             self.model_config['num_heads'])
        head_dim = self.model_config['head_dim']
        dtype_bytes = self.model_config.get('dtype_bytes', 2)

        bytes_per_token = (
            2  # K and V
            * num_layers
            * num_kv_heads
            * head_dim
            * dtype_bytes
        )

        bytes_per_block = bytes_per_token * self.block_size

        return {
            'bytes_per_token': bytes_per_token,
            'kb_per_token': bytes_per_token / 1024,
            'mb_per_block': bytes_per_block / (1024 ** 2),
            'gb_per_1k_tokens': bytes_per_token * 1000 / (1024 ** 3),
        }

    def calculate_capacity(
        self,
        gpu_memory_utilization: float = 0.9
    ) -> Dict:
        """Calculate maximum capacity."""

        # Model memory
        model_gb = self.calculate_model_memory()

        # Activation memory (conservative estimate)
        activation_gb = 3.0  # Conservative

        # OS overhead
        overhead_gb = 1.0

        # Available for KV cache
        available_gb = (
            self.gpu_memory_gb * gpu_memory_utilization
            - model_gb
            - activation_gb
            - overhead_gb
        )

        if available_gb < 0:
            raise ValueError(f"Model too large! Needs {model_gb + activation_gb + overhead_gb:.2f} GB")

        # Calculate blocks
        kv_memory = self.calculate_kv_cache_memory()
        mb_per_block = kv_memory['mb_per_block']

        num_blocks = int(available_gb * 1024 / mb_per_block)

        return {
            'model_memory_gb': model_gb,
            'activation_memory_gb': activation_gb,
            'overhead_gb': overhead_gb,
            'available_for_kv_gb': available_gb,
            'num_blocks': num_blocks,
            'max_tokens': num_blocks * self.block_size,
            'kv_memory': kv_memory,
        }

    def recommend_batch_size(
        self,
        avg_prompt_len: int,
        avg_output_len: int,
        gpu_memory_utilization: float = 0.9,
    ) -> Dict:
        """Recommend optimal batch size for workload."""

        capacity = self.calculate_capacity(gpu_memory_utilization)

        # Average sequence length
        avg_seq_len = avg_prompt_len + avg_output_len

        # Blocks per sequence
        blocks_per_seq = (avg_seq_len + self.block_size - 1) // self.block_size

        # Max sequences that fit
        max_batch_size = capacity['num_blocks'] // blocks_per_seq

        # Calculate throughput estimate
        # (very rough - actual depends on hardware)
        estimated_throughput = max_batch_size * avg_output_len / avg_output_len  # tokens/step

        return {
            'avg_seq_len': avg_seq_len,
            'blocks_per_seq': blocks_per_seq,
            'recommended_batch_size': max_batch_size,
            'estimated_concurrent_requests': max_batch_size,
            'capacity': capacity,
        }

    def print_report(
        self,
        avg_prompt_len: int = 256,
        avg_output_len: int = 128,
    ):
        """Print comprehensive memory report."""

        print("=" * 70)
        print("vLLM Memory Analysis Report")
        print("=" * 70)

        # Model info
        print(f"\nModel Configuration:")
        print(f"  Name: {self.model_config.get('name', 'Unknown')}")
        print(f"  Layers: {self.model_config['num_layers']}")
        print(f"  Heads: {self.model_config['num_heads']}")
        print(f"  KV Heads: {self.model_config.get('num_kv_heads', self.model_config['num_heads'])}")
        print(f"  Hidden dim: {self.model_config['hidden_size']}")
        print(f"  Head dim: {self.model_config['head_dim']}")
        print(f"  Params: {self.model_config.get('num_params_billions', 'Unknown')} B")

        # GPU info
        print(f"\nGPU Configuration:")
        print(f"  Memory: {self.gpu_memory_gb} GB")
        print(f"  Block size: {self.block_size} tokens")

        # KV cache
        kv_memory = self.calculate_kv_cache_memory()
        print(f"\nKV Cache Memory:")
        print(f"  Per token: {kv_memory['kb_per_token']:.2f} KB")
        print(f"  Per block: {kv_memory['mb_per_block']:.2f} MB")
        print(f"  Per 1K tokens: {kv_memory['gb_per_1k_tokens']:.3f} GB")

        # Capacity
        capacity = self.calculate_capacity()
        print(f"\nMemory Breakdown:")
        print(f"  Model weights: {capacity['model_memory_gb']:.2f} GB")
        print(f"  Activations: {capacity['activation_memory_gb']:.2f} GB")
        print(f"  OS overhead: {capacity['overhead_gb']:.2f} GB")
        print(f"  Available for KV: {capacity['available_for_kv_gb']:.2f} GB")

        print(f"\nKV Cache Capacity:")
        print(f"  Total blocks: {capacity['num_blocks']}")
        print(f"  Max tokens (single seq): {capacity['max_tokens']:,}")

        # Batch size recommendations
        print(f"\nBatch Size Recommendations:")
        print(f"  (Assuming avg prompt: {avg_prompt_len}, avg output: {avg_output_len})")

        for pct in [0.7, 0.8, 0.9]:
            rec = self.recommend_batch_size(avg_prompt_len, avg_output_len, pct)
            print(f"\n  GPU utilization {pct*100:.0f}%:")
            print(f"    Recommended batch size: {rec['recommended_batch_size']}")
            print(f"    Concurrent requests: {rec['estimated_concurrent_requests']}")

        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # OPT-13B configuration
    opt_13b_config = {
        'name': 'OPT-13B',
        'num_params_billions': 13,
        'num_layers': 40,
        'num_heads': 40,
        'num_kv_heads': 40,  # MHA (same as num_heads)
        'hidden_size': 5120,
        'head_dim': 128,
        'dtype_bytes': 2,  # FP16
    }

    # Llama-2-13B with GQA
    llama2_13b_config = {
        'name': 'Llama-2-13B',
        'num_params_billions': 13,
        'num_layers': 40,
        'num_heads': 40,
        'num_kv_heads': 8,  # GQA - only 8 KV heads!
        'hidden_size': 5120,
        'head_dim': 128,
        'dtype_bytes': 2,
    }

    print("Comparing OPT-13B vs Llama-2-13B:")
    print("\n" + "="*70)
    print("OPT-13B (Multi-Head Attention)")
    print("="*70)

    calc_opt = MemoryCalculator(opt_13b_config, gpu_memory_gb=40)
    calc_opt.print_report()

    print("\n\n" + "="*70)
    print("Llama-2-13B (Grouped Query Attention)")
    print("="*70)

    calc_llama = MemoryCalculator(llama2_13b_config, gpu_memory_gb=40)
    calc_llama.print_report()

    # Show GQA benefit
    opt_kv = calc_opt.calculate_kv_cache_memory()
    llama_kv = calc_llama.calculate_kv_cache_memory()

    print(f"\n{'='*70}")
    print("GQA Memory Savings:")
    print(f"{'='*70}")
    print(f"OPT-13B KV cache: {opt_kv['kb_per_token']:.2f} KB/token")
    print(f"Llama-2-13B KV cache: {llama_kv['kb_per_token']:.2f} KB/token")
    print(f"Reduction: {(1 - llama_kv['kb_per_token']/opt_kv['kb_per_token'])*100:.1f}%")
    print(f"Can fit {opt_kv['kb_per_token']/llama_kv['kb_per_token']:.1f}x more sequences!")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **KV Cache Fundamentals**
- Memory layout and structure
- Per-token and per-block calculations
- GPU memory budget breakdown

✅ **Block Manager**
- Block allocation and deallocation
- Block tables and mapping
- Swap operations (GPU ↔ CPU)
- Reference counting for sharing

✅ **Memory Optimization**
- Grouped Query Attention (GQA)
- Quantization techniques
- Prefix caching and sharing
- Block size tuning

✅ **Capacity Planning**
- Calculate memory requirements
- Determine optimal batch sizes
- Analyze trade-offs
- Build planning tools

### Knowledge Check (Quiz)

**Question 1**: Why does OPT-13B use 800 KB per token for KV cache?
<details>
<summary>Answer</summary>
2 (K and V) × 40 layers × 40 heads × 128 head_dim × 2 bytes (FP16) = 819,200 bytes = 800 KB
</details>

**Question 2**: How does Grouped Query Attention reduce KV cache memory?
<details>
<summary>Answer</summary>
GQA shares KV heads across multiple query heads. Example: 40 query heads share 8 KV heads instead of each having its own. Reduction: 40 → 8 heads = 5x less KV memory. Llama-2 uses this.
</details>

**Question 3**: What happens during swap_out operation?
<details>
<summary>Answer</summary>
1. Allocate CPU blocks (same count as GPU blocks)
2. Transfer KV cache data: GPU → CPU (actual memory copy)
3. Free GPU blocks (return to free pool)
4. Update block table to point to CPU blocks
Reverse process for swap_in.
</details>

**Question 4**: Why is block_size=16 a good default?
<details>
<summary>Answer</summary>
Balance between:
- Waste in partial blocks: ~6% average (last block rarely full)
- Metadata overhead: 1/16 = 6.25% overhead for block table
- Memory fragmentation: Smaller blocks = more flexibility
Larger blocks waste more; smaller blocks have more overhead.
</details>

**Question 5**: On A100 40GB with OPT-13B, why can we only fit ~800 blocks?
<details>
<summary>Answer</summary>
Available = 40 GB - 26 GB (model) - 3 GB (activation) - 1 GB (overhead) = 10 GB
Block size = 16 tokens × 800 KB = 12.8 MB
Blocks = 10 GB / 12.8 MB = ~800 blocks
Total capacity = 800 × 16 = 12,800 tokens
</details>

### Daily Reflection

**What went well?**
- [ ] Understood block manager implementation
- [ ] Calculated memory requirements accurately
- [ ] Built useful planning tools

**What was challenging?**
- [ ] Memory calculation complexity
- [ ] Swap operation details
- [ ] Optimization trade-offs

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 7

Tomorrow - Week 1 Review:
- **Complete Week Summary**: All concepts covered
- **Comprehensive Quiz**: Test your knowledge
- **Practice Problems**: Hands-on challenges
- **Integration Exercise**: Build end-to-end understanding
- **Preparation for Week 2**: CUDA kernels deep dive

**Preparation**:
- Review all notes from Days 1-6
- Identify unclear concepts
- Prepare questions
- Get ready to dive deeper!

---

## 📚 Additional Resources

**Code Reading**:
- [ ] `vllm/core/block_manager_v2.py` (complete)
- [ ] `vllm/core/block/block_table.py`
- [ ] `csrc/cache_kernels.cu` (KV cache CUDA ops)

**Advanced Topics**:
- [ ] Prefix caching implementation
- [ ] Copy-on-write for sharing
- [ ] Automatic prefix detection
- [ ] Multi-tenant memory isolation

**Tools**:
- [ ] NVIDIA nvidia-smi (GPU memory monitoring)
- [ ] PyTorch memory profiler
- [ ] vLLM internal metrics

---

**Congratulations on mastering KV cache management! 🎉**

**You can now plan and optimize vLLM deployments!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
