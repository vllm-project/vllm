# PagedAttention Deep Dive - Part 2: Implementation Details

> **Learning Objective**: Understand how PagedAttention is implemented in vLLM's C++/CUDA codebase
> **Prerequisites**: Part 1 (Theory), CUDA kernel programming basics
> **Time to Complete**: 4-6 hours
> **Files to Study**: `csrc/attention/`, `vllm/attention/`, `vllm/core/block_manager_v2.py`

---

## 🎯 Learning Goals

- [ ] Understand vLLM's block manager implementation
- [ ] Read and comprehend paged attention CUDA kernels
- [ ] Trace block allocation and deallocation
- [ ] Analyze memory access patterns in kernels
- [ ] Identify optimization techniques used
- [ ] Implement a simplified version yourself

---

## 🏗️ Implementation Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Python Layer                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ BlockManagerV2 (block_manager_v2.py)             │  │
│  │ - Allocates/frees blocks                         │  │
│  │ - Maintains block tables                         │  │
│  │ - Handles swapping (CPU ↔ GPU)                   │  │
│  └──────────────────────┬───────────────────────────┘  │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Attention Layer (Python)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ PagedAttention (ops/paged_attn.py)               │  │
│  │ - Prepares inputs for kernels                    │  │
│  │ - Calls C++ extensions                           │  │
│  └──────────────────────┬───────────────────────────┘  │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              C++ Extension Layer                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ _custom_ops.py → C++ bindings                    │  │
│  │ - paged_attention_v1                             │  │
│  │ - paged_attention_v2                             │  │
│  └──────────────────────┬───────────────────────────┘  │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                 CUDA Kernels                             │
│  ┌──────────────────────────────────────────────────┐  │
│  │ csrc/attention/attention_kernels.cu              │  │
│  │ - Kernel implementations                         │  │
│  │ - Template specializations for dtypes           │  │
│  │ - Optimized memory access                        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Key Files to Study

| File | Purpose | LOC | Priority |
|------|---------|-----|----------|
| `vllm/core/block_manager_v2.py` | Block allocation logic | ~800 | **HIGHEST** |
| `vllm/core/block/block_table.py` | Block table abstraction | ~200 | **HIGH** |
| `vllm/attention/ops/paged_attn.py` | Python wrapper for kernels | ~300 | **HIGH** |
| `csrc/attention/attention_kernels.cu` | CUDA kernel implementation | ~2000 | **HIGHEST** |
| `csrc/attention/attention_generic.cuh` | Generic attention logic | ~500 | **HIGH** |
| `csrc/attention/dtype_float16.cuh` | FP16 specialization | ~300 | **MEDIUM** |
| `vllm/_custom_ops.py` | PyTorch custom op registration | ~100 | **MEDIUM** |

---

## 🔍 Block Manager Deep Dive

### Block Manager Responsibilities

Located in: `vllm/core/block_manager_v2.py:BlockManager`

```python
class BlockManager:
    """
    Manages GPU and CPU block allocation for sequences.

    Key responsibilities:
    1. Allocate blocks for new sequences
    2. Free blocks when sequences complete
    3. Swap blocks between GPU and CPU
    4. Track block usage and availability
    """

    def __init__(self, block_size: int, num_gpu_blocks: int, num_cpu_blocks: int):
        self.block_size = block_size  # e.g., 16 tokens

        # GPU block allocator
        self.gpu_allocator = BlockAllocator(num_gpu_blocks)

        # CPU block allocator (for swapping)
        self.cpu_allocator = BlockAllocator(num_cpu_blocks)

        # Block tables for each sequence
        self.block_tables: Dict[int, BlockTable] = {}
```

### Block Allocation Algorithm

**vllm/core/block_manager_v2.py:246 - `allocate()` method**

```python
def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
    """
    Allocate blocks for a sequence.

    Args:
        seq_id: Sequence identifier
        num_blocks: Number of blocks to allocate

    Returns:
        List of physical block IDs
    """
    # Allocate physical blocks
    physical_blocks = []
    for _ in range(num_blocks):
        block_id = self.gpu_allocator.allocate()
        if block_id is None:
            # Out of memory - trigger swapping or preemption
            raise RuntimeError("Out of GPU blocks")
        physical_blocks.append(block_id)

    # Create or extend block table
    if seq_id not in self.block_tables:
        self.block_tables[seq_id] = BlockTable(self.block_size)

    self.block_tables[seq_id].extend(physical_blocks)
    return physical_blocks
```

**Implementation Details**:

File: `vllm/core/block/block_table.py:BlockTable`

```python
class BlockTable:
    """
    Mapping from logical blocks to physical blocks.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._blocks: List[int] = []  # Physical block IDs

    def extend(self, physical_blocks: List[int]) -> None:
        """Add new physical blocks to table."""
        self._blocks.extend(physical_blocks)

    def get_physical_block_id(self, logical_block_id: int) -> int:
        """Map logical to physical block ID."""
        return self._blocks[logical_block_id]

    def get_num_blocks(self) -> int:
        """Number of allocated blocks."""
        return len(self._blocks)

    def get_last_block_size(self) -> int:
        """Number of tokens in last block (may be partial)."""
        # Computed from sequence length
        return self._last_block_size
```

### Memory Pool (GPU Block Allocator)

**vllm/core/block/block_allocator.py**

```python
class BlockAllocator:
    """
    Simple free list allocator for blocks.
    """

    def __init__(self, num_blocks: int):
        # Initialize all blocks as free
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()

    def allocate(self) -> Optional[int]:
        """Allocate a single block."""
        if not self.free_blocks:
            return None

        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int) -> None:
        """Free a block back to pool."""
        assert block_id in self.allocated_blocks
        self.allocated_blocks.remove(block_id)
        self.free_blocks.append(block_id)

    def get_num_free_blocks(self) -> int:
        """Query available blocks."""
        return len(self.free_blocks)
```

### Example: Block Allocation Trace

```python
# Initialization
block_manager = BlockManager(
    block_size=16,
    num_gpu_blocks=100,
    num_cpu_blocks=50
)

# Sequence 1: Prompt with 30 tokens
seq_1_blocks = block_manager.allocate(seq_id=1, num_blocks=2)
# Returns: [0, 1] (physical block IDs)
# Free blocks: 98

# Sequence 2: Prompt with 50 tokens
seq_2_blocks = block_manager.allocate(seq_id=2, num_blocks=4)
# Returns: [2, 3, 4, 5]
# Free blocks: 94

# Generation: Sequence 1 needs one more block (token 32 generated)
new_block = block_manager.allocate(seq_id=1, num_blocks=1)
# Returns: [6]
# Free blocks: 93
# Sequence 1 blocks: [0, 1, 6]

# Sequence 2 completes
block_manager.free(seq_id=2)
# Free blocks: 97 (93 + 4)
# Blocks [2, 3, 4, 5] now available for reuse
```

---

## 🔧 CUDA Kernel Implementation

### Kernel Entry Point

**File: csrc/attention/attention_kernels.cu**

```cuda
// Template kernel for different data types
template<typename scalar_t, int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,              // Output: [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ q,          // Query: [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ k_cache,    // Key cache: [num_blocks, num_heads, block_size, head_dim]
    const scalar_t* __restrict__ v_cache,    // Value cache: [num_blocks, num_heads, block_size, head_dim]
    const int* __restrict__ block_tables,    // Block tables: [num_seqs, max_num_blocks]
    const int* __restrict__ context_lens,    // Context lengths: [num_seqs]
    const float scale,                        // Attention scale: 1/sqrt(head_dim)
    const int max_num_blocks_per_seq
) {
    // Kernel implementation (explained below)
}
```

### Kernel Design: One Thread Block per Attention Head

**Thread organization**:
```
Grid: (num_tokens, num_heads)
Block: (block_threads)  // e.g., 256 threads per block

Each thread block computes attention for ONE head of ONE token
```

**Rationale**:
- Parallelism: Across tokens and heads
- Shared memory: Store partial results for one head
- Warp efficiency: Reduce within warps

### Kernel Algorithm

```cuda
__global__ void paged_attention_v1_kernel(...) {
    // 1. Identify which token and which head this block processes
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    // 2. Get sequence information
    const int seq_idx = /* map token_idx to sequence */;
    const int context_len = context_lens[seq_idx];
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // 3. Load query for this token and head
    const scalar_t* q_ptr = q + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;
    scalar_t q_vec[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        q_vec[i] = q_ptr[i];
    }

    // 4. Iterate over blocks (logical blocks of the sequence)
    __shared__ float shared_qk[BLOCK_SIZE];  // Store attention scores
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;

    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // 5. Get physical block ID from block table
        const int physical_block_id = block_table[block_idx];

        // 6. Compute attention scores for this block
        // Each thread computes score for one token in the block
        for (int local_token_idx = threadIdx.x;
             local_token_idx < BLOCK_SIZE;
             local_token_idx += blockDim.x) {

            const int global_token_idx = block_idx * BLOCK_SIZE + local_token_idx;
            if (global_token_idx >= context_len) break;

            // Load key from k_cache
            const scalar_t* k_ptr = k_cache +
                physical_block_id * num_heads * BLOCK_SIZE * HEAD_DIM +
                head_idx * BLOCK_SIZE * HEAD_DIM +
                local_token_idx * HEAD_DIM;

            // Compute Q·K^T
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                qk += (float)q_vec[d] * (float)k_ptr[d];
            }
            qk *= scale;

            // Store score in shared memory
            shared_qk[local_token_idx] = qk;

            // Update max for numerical stability
            max_score = fmaxf(max_score, qk);
        }
        __syncthreads();

        // 7. Compute softmax (numerically stable)
        for (int i = threadIdx.x; i < BLOCK_SIZE && (block_idx * BLOCK_SIZE + i) < context_len; i += blockDim.x) {
            shared_qk[i] = expf(shared_qk[i] - max_score);
            sum_exp += shared_qk[i];
        }
        __syncthreads();
    }

    // 8. Normalize softmax weights
    // (warp reduction to sum across threads)
    sum_exp = warp_reduce_sum(sum_exp);
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_sum, sum_exp);
    }
    __syncthreads();

    // 9. Compute weighted sum of values
    scalar_t out_vec[HEAD_DIM] = {0};

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block_id = block_table[block_idx];

        for (int local_token_idx = threadIdx.x;
             local_token_idx < BLOCK_SIZE;
             local_token_idx += blockDim.x) {

            const int global_token_idx = block_idx * BLOCK_SIZE + local_token_idx;
            if (global_token_idx >= context_len) break;

            // Load value from v_cache
            const scalar_t* v_ptr = v_cache +
                physical_block_id * num_heads * BLOCK_SIZE * HEAD_DIM +
                head_idx * BLOCK_SIZE * HEAD_DIM +
                local_token_idx * HEAD_DIM;

            // Weighted sum: out += attn_weight * value
            float attn_weight = shared_qk[local_token_idx] / shared_sum;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                out_vec[d] += attn_weight * v_ptr[d];
            }
        }
    }

    // 10. Write output (reduction across threads)
    // (details omitted - uses warp reduction)
}
```

### Memory Layout of KV Cache

```
K cache shape: [num_blocks, num_heads, block_size, head_dim]

Example: num_blocks=100, num_heads=32, block_size=16, head_dim=128

Physical Block 0:
  Head 0: [16 tokens × 128 dim]
  Head 1: [16 tokens × 128 dim]
  ...
  Head 31: [16 tokens × 128 dim]

Physical Block 1:
  Head 0: [16 tokens × 128 dim]
  ...

Accessing token i of head h in block b:
offset = b * (num_heads * block_size * head_dim) +
         h * (block_size * head_dim) +
         i * head_dim
```

### Optimization Techniques

#### 1. Shared Memory for Attention Scores
```cuda
__shared__ float shared_qk[BLOCK_SIZE];

// Store scores in shared memory (fast)
// Avoids recomputation for value phase
```

#### 2. Vectorized Loads (for head_dim = 128)
```cuda
// Load 4 floats at once (128-bit transaction)
float4* q_vec_ptr = reinterpret_cast<float4*>(q_ptr);
float4 q_chunk = q_vec_ptr[i / 4];
```

#### 3. Warp Reduction for Softmax Sum
```cuda
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

#### 4. Template Specialization for Different HEAD_DIM
```cuda
// Compile-time optimization for different head dimensions
template void paged_attention_v1<float, 64, 16>(...);
template void paged_attention_v1<float, 128, 16>(...);
template void paged_attention_v1<half, 128, 16>(...);
```

---

## 🔄 Complete Flow: Request to Kernel

### Step-by-Step Execution

```python
# 1. User submits request
from vllm import LLM
llm = LLM(model="meta-llama/Llama-2-7b")
output = llm.generate("Tell me a story")

# Internal flow:
# ───────────────────────────────────────────────────────

# 2. Engine creates sequence
seq = Sequence(prompt="Tell me a story", seq_id=123)

# 3. Block manager allocates initial blocks (for prefill)
num_tokens = len(tokenized_prompt)  # e.g., 5 tokens
num_blocks = math.ceil(num_tokens / 16)  # 1 block
blocks = block_manager.allocate(seq_id=123, num_blocks=1)
# blocks = [7]  (physical block ID)

# 4. Prefill: Compute KV cache for prompt
model_input = ModelInput(
    input_ids=tokenized_prompt,
    positions=[0, 1, 2, 3, 4]
)
hidden_states, kv_cache = model(model_input)

# 5. Store KV cache in allocated block
# kv_cache shape: [num_layers, 2, num_heads, 5, head_dim]
# Store in block 7
for layer in range(num_layers):
    k, v = kv_cache[layer]  # [num_heads, 5, head_dim]
    store_in_block(k, v, physical_block_id=7, offset=0)

# 6. Decode: Generate tokens one by one
for step in range(max_new_tokens):
    # 6a. Check if need new block
    current_tokens = num_tokens + step
    required_blocks = math.ceil(current_tokens / 16)
    if required_blocks > len(blocks):
        new_block = block_manager.allocate(seq_id=123, num_blocks=1)
        blocks.append(new_block)

    # 6b. Prepare block table (for attention kernel)
    block_table = torch.tensor(blocks)  # [0, 7, 42, ...]

    # 6c. Compute attention (using PagedAttention kernel)
    from vllm.attention.ops import paged_attention_v1

    attn_output = paged_attention_v1(
        query=current_query,           # [1, num_heads, head_dim]
        key_cache=global_k_cache,      # [num_blocks, num_heads, 16, head_dim]
        value_cache=global_v_cache,    # [num_blocks, num_heads, 16, head_dim]
        block_tables=block_table,      # [num_allocated_blocks]
        context_lens=torch.tensor([current_tokens]),
        scale=1.0 / math.sqrt(head_dim)
    )

    # 6d. Continue model forward pass
    next_token = model.sample(attn_output)

    # 6e. Store new KV in cache
    store_in_block(new_k, new_v, blocks[-1], offset=step % 16)

# 7. Sequence completes
block_manager.free(seq_id=123)
# Blocks [7, 42, ...] returned to free pool
```

---

## 📊 Performance Analysis

### Profiling Checklist

Use Nsight Compute to analyze:

```bash
ncu --kernel-name paged_attention_v1 --set full python bench_attention.py
```

**Metrics to Check**:

| Metric | Target | Typical |
|--------|--------|---------|
| Memory Throughput | > 80% of peak | 75-85% |
| Compute Throughput | > 60% of peak | 40-70% |
| Occupancy | > 50% | 60-80% |
| Shared Memory Conflicts | < 5% | 2-10% |
| Warp Execution Efficiency | > 90% | 85-95% |

### Bottleneck Analysis

**Compute-bound** (good for Tensor Cores):
```
- High arithmetic intensity
- Use FP16/BF16 for Tensor Core acceleration
- Optimization: Increase arithmetic per memory load
```

**Memory-bound** (typical for attention):
```
- Limited by HBM bandwidth
- Optimization: Better memory coalescing, shared memory usage
- Prefetching, vectorized loads
```

---

## 🎯 Implementation Exercises

### Exercise 1: Simple Block Manager (Python)

Implement a basic block allocator:

```python
class SimpleBlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # TODO: Initialize free block list
        pass

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        # TODO: Allocate blocks
        pass

    def free(self, seq_id: int) -> None:
        # TODO: Free all blocks for sequence
        pass

    def get_block_table(self, seq_id: int) -> List[int]:
        # TODO: Return block table for sequence
        pass

# Test your implementation
manager = SimpleBlockManager(num_blocks=10, block_size=16)
blocks_1 = manager.allocate(seq_id=1, num_blocks=3)
blocks_2 = manager.allocate(seq_id=2, num_blocks=2)
manager.free(seq_id=1)
blocks_3 = manager.allocate(seq_id=3, num_blocks=3)
assert blocks_3 == blocks_1  # Reused freed blocks
```

### Exercise 2: Simplified Attention Kernel (CUDA)

Implement a basic paged attention kernel:

```cuda
// Simplified single-head attention
__global__ void simple_paged_attention(
    float* out,                    // [num_tokens, head_dim]
    const float* q,                // [num_tokens, head_dim]
    const float* k_cache,          // [num_blocks, block_size, head_dim]
    const float* v_cache,          // [num_blocks, block_size, head_dim]
    const int* block_table,        // [max_blocks]
    int context_len,
    int head_dim,
    int block_size
) {
    // TODO: Implement
    // Hints:
    // 1. One thread block per token
    // 2. Compute Q·K^T for all blocks
    // 3. Softmax
    // 4. Weighted sum of V
}
```

### Exercise 3: Benchmark Memory Layouts

Compare performance of different memory layouts:

```python
# Layout 1: [num_blocks, block_size, num_heads, head_dim]
# Layout 2: [num_blocks, num_heads, block_size, head_dim]
# Layout 3: [num_heads, num_blocks, block_size, head_dim]

import torch
import time

def benchmark_layout(layout):
    # Create tensors
    # Run attention
    # Measure time
    pass

# Find optimal layout for your hardware
```

---

## 🚀 Next Steps

**Part 3: Advanced Optimizations** covers:
- Kernel fusion (fused attention + sampling)
- Multi-query attention (MQA) and grouped-query attention (GQA)
- FlashAttention integration with paging
- Quantization (INT8, FP8) with PagedAttention
- Multi-GPU block management

---

**Further Reading**:
- vLLM source code: `csrc/attention/`
- Flash Attention paper and code
- CUDA C++ Programming Guide: Shared Memory

**Ready for Part 3!** 🚀
