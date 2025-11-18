# Tutorial 03: Attention Kernel Walkthrough

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand the attention mechanism and its computational complexity
2. Analyze vLLM's paged attention implementation in detail
3. Identify memory access patterns in attention kernels
4. Understand the partitioning strategy for long sequences
5. Implement optimizations for attention computation in CUDA

## Prerequisites

- Completion of Tutorials 01 and 02
- Understanding of transformer attention mechanism
- Familiarity with matrix operations and dot products
- Knowledge of softmax operation and numerical stability

## Table of Contents

1. [Attention Mechanism Review](#attention-mechanism-review)
2. [vLLM Paged Attention Overview](#vllm-paged-attention-overview)
3. [Kernel Architecture](#kernel-architecture)
4. [Detailed Code Walkthrough](#detailed-code-walkthrough)
5. [Memory Access Patterns](#memory-access-patterns)
6. [Partitioning Strategy](#partitioning-strategy)
7. [Performance Analysis](#performance-analysis)
8. [Hands-on Exercises](#hands-on-exercises)
9. [Best Practices](#best-practices)
10. [References](#references)

## Attention Mechanism Review

### Mathematical Definition

Attention computes weighted sums of values based on query-key similarity:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Where:
- Q: Query matrix [batch, heads, 1, head_dim]  (decoding)
- K: Key matrix   [batch, heads, seq_len, head_dim]
- V: Value matrix [batch, heads, seq_len, head_dim]
- d_k: Head dimension (typically 64, 128, or 256)
```

### Computational Steps

```
Step 1: Compute attention scores
┌───────────────────────────────────┐
│  QK^T = Q · K^T                   │
│  Shape: [batch, heads, 1, seq_len]│
│  Cost: O(seq_len × head_dim)      │
└───────────────────────────────────┘
           ↓
Step 2: Scale scores
┌───────────────────────────────────┐
│  S = QK^T / √head_dim             │
│  Prevents saturation of softmax   │
└───────────────────────────────────┘
           ↓
Step 3: Apply softmax
┌───────────────────────────────────┐
│  P = softmax(S)                   │
│  For each position:               │
│    exp_sum = Σ exp(s_i)           │
│    p_i = exp(s_i) / exp_sum       │
└───────────────────────────────────┘
           ↓
Step 4: Weighted sum of values
┌───────────────────────────────────┐
│  Output = P · V                   │
│  Shape: [batch, heads, 1, head_dim]│
│  Cost: O(seq_len × head_dim)      │
└───────────────────────────────────┘
```

### Memory Complexity Challenge

For a sequence of length N:
- Naive attention requires O(N²) memory for attention matrix
- For N=100K tokens: 100K × 100K = 10B elements = 40GB for FP32
- **vLLM solution**: Compute attention on-the-fly, never materialize full matrix

## vLLM Paged Attention Overview

### Paged KV Cache

vLLM stores K and V tensors in paged memory blocks:

```
Physical Memory (Paged):
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ Block 4 │
│  K, V   │  K, V   │  K, V   │  K, V   │  K, V   │
└─────────┴─────────┴─────────┴─────────┴─────────┘

Logical Sequence:
Seq 1: [Block 2] → [Block 0] → [Block 4]  (3 blocks)
Seq 2: [Block 1] → [Block 3]              (2 blocks)

Block Table (indirection):
Seq 1: [2, 0, 4, ...]
Seq 2: [1, 3, ...]
```

### Benefits of Paging

1. **Memory efficiency**: No fragmentation, share blocks across sequences
2. **Dynamic allocation**: Add blocks as sequence grows
3. **Flexible batching**: Different sequence lengths in same batch
4. **Cache reuse**: Share prefixes (e.g., system prompts)

### Paged Attention V2

Two-stage computation for long sequences:

```
Stage 1: Compute partial attention for each partition
┌──────────────────────────────────────────────┐
│  Per partition p:                            │
│    - Compute QK scores for tokens in p       │
│    - Find local max: max_p                   │
│    - Compute local exp_sum: Σ exp(qk - max_p)│
│    - Store partial output: out_p             │
└──────────────────────────────────────────────┘

Stage 2: Reduce partitions to final result
┌──────────────────────────────────────────────┐
│  Global max: max_global = max(max_p)         │
│  Corrected exp_sums:                         │
│    exp_sum'_p = exp_sum_p × exp(max_p - max_global)│
│  Global exp_sum = Σ exp_sum'_p               │
│  Final output = Σ (out_p × exp_sum'_p) / global_exp_sum│
└──────────────────────────────────────────────┘
```

## Kernel Architecture

### File Structure

```
/home/user/vllm-learn/csrc/attention/
├── paged_attention_v2.cu        # Launcher (CPU-side)
├── attention_kernels.cuh        # Kernel implementation (GPU-side)
└── attention_utils.cuh          # Helper functions
```

### Grid/Block Configuration

**File**: `/home/user/vllm-learn/csrc/attention/paged_attention_v2.cu` (Lines 81-93)

```cpp
const int NUM_WARPS = NUM_THREADS / WARP_SIZE;  // Typically 128/32 = 4 warps
int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
int logits_size = PARTITION_SIZE * sizeof(float);
int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

// For paged attention v2 kernel.
dim3 grid(num_heads, num_seqs, max_num_partitions);
int shared_mem_size = std::max(logits_size, outputs_size);

// For paged attention v2 reduce kernel.
dim3 reduce_grid(num_heads, num_seqs);
int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
```

**Grid organization**:
```
Grid: (num_heads, num_seqs, max_num_partitions)
      ↓          ↓         ↓
   Each head  Each seq  Each partition
   processes  computed  of sequence
   independently  separately  (512 tokens default)

Example: 32 heads, 4 sequences, seq_len=2048
- Partitions per seq: 2048/512 = 4
- Total blocks: 32 × 4 × 4 = 512 blocks
```

## Detailed Code Walkthrough

### Part 1: Initialization and Setup

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 105-151)

```cpp
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(...) {
  // Line 106-114: Determine work for this thread block
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int seq_len = seq_lens[seq_idx];

  // Early exit if partition is beyond sequence length
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    return;
  }

  // Line 116-131: Calculate block and token ranges
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int start_block_idx = partition_idx * num_blocks_per_partition;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition,
                                 num_seq_blocks);

  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE,
                                 seq_len);

  // Line 133-143: Thread organization
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;

  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  // Line 145-150: Head configuration (for Multi-Query/Grouped-Query Attention)
  const int head_idx = blockIdx.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
}
```

**Key Insights**:
1. **3D grid**: Parallelism across heads, sequences, and partitions
2. **Thread groups**: Multiple threads cooperate on each QK dot product
3. **GQA support**: Maps multiple query heads to one KV head

### Part 2: Load Query Vector

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 152-184)

```cpp
// Line 157-163: Vectorization for efficient memory access
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

// Line 168-182: Load query to shared memory
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

#pragma unroll
for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
  const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
  q_vecs[thread_group_offset][i] =
      *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
}
__syncthreads();  // Ensure all threads have loaded query
```

**Memory Layout**:
```
Query vector split across thread group:
HEAD_SIZE = 128, THREAD_GROUP_SIZE = 4, VEC_SIZE = 2

Thread 0: [q[0:2],   q[8:10],   q[16:18],  ... ] (vectors 0, 4, 8, ...)
Thread 1: [q[2:4],   q[10:12],  q[18:20],  ... ] (vectors 1, 5, 9, ...)
Thread 2: [q[4:6],   q[12:14],  q[20:22],  ... ] (vectors 2, 6, 10, ...)
Thread 3: [q[6:8],   q[14:16],  q[22:24],  ... ] (vectors 3, 7, 11, ...)
```

### Part 3: Compute QK Scores

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 196-300)

```cpp
// Line 187-191: Allocate shared memory for logits
extern __shared__ char shared_mem[];
float* logits = reinterpret_cast<float*>(shared_mem);
__shared__ float red_smem[2 * NUM_WARPS];

float qk_max = -FLT_MAX;

// Line 202: Get block table for this sequence
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// Line 222-253: Iterate over KV blocks (each warp processes different blocks)
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {

  const int64_t physical_block_number =
      static_cast<int64_t>(block_table[block_idx]);

  // Line 260-285: For each token in the block
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset =
        (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    // Load key vector (vectorized, from paged cache)
    K_vec k_vecs[NUM_VECS_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const cache_t* k_ptr =
          k_cache + physical_block_number * kv_block_stride +
          kv_head_idx * kv_head_stride + physical_block_offset * x;
      const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
      // ... load key vector ...
    }

    // Line 287-292: Compute dot product Q·K
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                           q_vecs[thread_group_offset], k_vecs);
    // Add ALiBi positional bias if present
    qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

    // Line 294-300: Store logit and track maximum
    if (thread_group_offset == 0) {
      const bool mask = token_idx >= seq_len;
      logits[token_idx - start_token_idx] = mask ? 0.f : qk;
      qk_max = mask ? qk_max : fmaxf(qk_max, qk);
    }
  }
}
```

**Data Flow**:
```
Warp-level parallelism:
┌────────────────────────────────────────────────────┐
│ Warp 0: Blocks 0, 4, 8, ...  (stride = NUM_WARPS) │
│ Warp 1: Blocks 1, 5, 9, ...                       │
│ Warp 2: Blocks 2, 6, 10, ...                      │
│ Warp 3: Blocks 3, 7, 11, ...                      │
└────────────────────────────────────────────────────┘
              ↓
Thread group within warp processes one token:
┌────────────────────────────────────────────────────┐
│ For token T:                                       │
│   - Each thread loads part of K[T]                 │
│   - Compute partial dot products                   │
│   - Reduce across thread group → QK score         │
└────────────────────────────────────────────────────┘
```

### Part 4: Softmax Computation

After computing all QK scores:

```cpp
// Find global maximum across all threads (numerically stable softmax)
qk_max = block_max<NUM_WARPS>(red_smem, qk_max);

// Compute exp(qk - qk_max) and sum
float exp_sum = 0.0f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
  float val = __expf(logits[i] - qk_max);
  logits[i] = val;  // Reuse logits array for softmax values
  exp_sum += val;
}

// Reduce exp_sum across block
exp_sum = block_sum<NUM_WARPS>(red_smem, exp_sum);
```

### Part 5: Compute Attention Output

```cpp
// Each thread accumulates part of the output
float accs[NUM_VECS_PER_THREAD];
zero_vec<scalar_t, NUM_VECS_PER_THREAD>(accs);

// Iterate over blocks again, multiply softmax by values
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {

  const int64_t physical_block_number = block_table[block_idx];

  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    // Load value vector
    K_vec v_vecs[NUM_VECS_PER_THREAD];
    // ... (similar to key loading) ...

    // Get softmax probability for this token
    const float prob = logits[token_idx - start_token_idx] / exp_sum;

    // Accumulate: out += prob * V
    #pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      accs[j] += prob * v_vecs[j];
    }
  }
}

// Write accumulated results to output
// (reduction across thread group, then write)
```

## Memory Access Patterns

### Query Access Pattern

```
Query (in global memory):
[Seq0_Head0][Seq0_Head1]...[Seq1_Head0][Seq1_Head1]...

Each block reads ONE query vector:
- Block (head=H, seq=S, part=P) reads Q[S, H, :]
- Entire query loaded to shared memory cooperatively
- Reused for all tokens in partition

Access: O(head_size) per block
Reuse factor: O(partition_size) → Excellent locality
```

### Key/Value Access Pattern

```
KV Cache (paged):
Physical blocks in arbitrary order
Block table provides indirection

Key access:
- Each warp processes different blocks (parallel)
- Within warp, threads load different parts of key (vectorized)
- Each key read once per partition

Access: O(partition_size × head_size) per block
Pattern: Sequential within block, random across blocks
```

### Logits Access Pattern

```
Logits (in shared memory):
[logit_0][logit_1][logit_2]...[logit_511]  (512 = PARTITION_SIZE)

Two-phase access:
Phase 1 (QK): Write logits, track max (write-heavy)
Phase 2 (PV): Read logits for prob, compute output (read-heavy)

Shared memory size: PARTITION_SIZE × sizeof(float)
For PARTITION_SIZE=512: 2 KB (well within shared memory limits)
```

## Partitioning Strategy

### Why Partition?

For long sequences (e.g., 100K tokens):
- **Memory**: 100K × 4 bytes = 400 KB shared memory (exceeds limits)
- **Solution**: Process in partitions of 512 tokens

### Partition Processing

```
Sequence of 2048 tokens, PARTITION_SIZE=512:

Partition 0: Tokens [0:512]
┌────────────────────────────┐
│ 1. Compute QK scores       │
│ 2. Local max: max_0        │
│ 3. Local exp_sum: sum_0    │
│ 4. Partial output: out_0   │
└────────────────────────────┘

Partition 1: Tokens [512:1024]
┌────────────────────────────┐
│ 1. Compute QK scores       │
│ 2. Local max: max_1        │
│ 3. Local exp_sum: sum_1    │
│ 4. Partial output: out_1   │
└────────────────────────────┘

Partition 2: Tokens [1024:1536]
Partition 3: Tokens [1536:2048]

Reduce stage:
- Global max = max(max_0, max_1, max_2, max_3)
- Corrected sums = sum_i × exp(max_i - global_max)
- Final output = Σ (out_i × corrected_sum_i) / Σ corrected_sum_i
```

### Mathematical Correctness

The partitioning preserves softmax correctness:

```
Standard softmax:
  p_i = exp(x_i - max) / Σ exp(x_j - max)

Partitioned softmax:
  Partition k: max_k, sum_k, out_k
  Global: max_global = max(max_k)

  Correction factor: α_k = exp(max_k - max_global)
  Global sum = Σ (α_k × sum_k)
  Final output = Σ (α_k × out_k) / global_sum

Proof: This equals standard softmax by exponential properties
```

## Performance Analysis

### Theoretical Performance

For attention with:
- Sequence length: N = 2048
- Head size: D = 128
- Number of heads: H = 32

**Operations per head**:
- QK computation: N × D multiplies + N × D adds ≈ 2ND FLOPs
- Softmax: 2N FLOPs (exp + divide)
- PV computation: N × D multiplies + N × D adds ≈ 2ND FLOPs
- **Total**: ≈ 4ND FLOPs = 4 × 2048 × 128 = 1M FLOPs/head

**Memory traffic**:
- Load Q: D floats
- Load K: N × D floats
- Load V: N × D floats
- Store output: D floats
- **Total**: (2N + 2) × D floats ≈ 2ND for large N

**Arithmetic intensity**:
```
AI = FLOPs / Bytes = (4ND) / (2ND × 4 bytes) = 0.5 FLOPs/byte

This is memory-bound! (Much less than GPU peak AI of ~50-100)
```

### Profiling with Nsight Compute

```bash
# Profile attention kernel
ncu --set full --target-processes all \
    --kernel-name "paged_attention_v2_kernel" \
    python benchmark.py

# Key metrics:
# - Memory Throughput: Should be 70-90% of peak
# - Compute Throughput: Typically 10-30% (memory bound)
# - L2 Cache Hit Rate: Important for KV cache reuse
# - Shared Memory Bank Conflicts: Should be minimal
```

## Hands-on Exercises

### Exercise 1: Analyze Block Table Lookups

Given this block table:
```cpp
// Sequence 0: Uses blocks [5, 2, 7, 1]
// Sequence 1: Uses blocks [3, 6]
int block_tables[] = {5, 2, 7, 1, 0, 0, 0, 0,  // Seq 0 (padded to 8)
                      3, 6, 0, 0, 0, 0, 0, 0}; // Seq 1 (padded to 8)
int seq_lens[] = {140, 60};  // Actual lengths
int BLOCK_SIZE = 16;
```

**Tasks**:
1. Calculate number of blocks needed for each sequence
2. Determine physical block addresses for each token
3. Draw the memory access pattern for Warp 0 processing Seq 0

### Exercise 2: Calculate Shared Memory Usage

For this configuration:
```cpp
HEAD_SIZE = 128
PARTITION_SIZE = 512
NUM_THREADS = 128
NUM_WARPS = 4
```

**Tasks**:
1. Calculate size of `logits` array
2. Calculate size of `q_vecs` shared array
3. Calculate size of `red_smem` array
4. Determine total shared memory requirement
5. Check if it fits in 48 KB shared memory limit

### Exercise 3: Implement Simplified QK Kernel

Write a simplified version that computes QK scores:

```cpp
__global__ void simple_qk_kernel(
    float* __restrict__ logits,      // [num_seqs, seq_len]
    const float* __restrict__ q,     // [num_seqs, head_dim]
    const float* __restrict__ k,     // [num_seqs, seq_len, head_dim]
    const float scale,
    const int seq_len,
    const int head_dim) {

  const int seq_idx = blockIdx.x;
  const int token_idx = threadIdx.x;  // Assume seq_len <= 1024

  // TODO: Implement
  // 1. Load query for this sequence
  // 2. Load key for this token
  // 3. Compute dot product
  // 4. Scale and store in logits
}
```

**Bonus**: Add shared memory to load query once per block.

## Best Practices

### Attention Kernel Optimization

1. **Partition wisely**: Balance shared memory usage vs kernel launches
   - Too small: Many partitions, reduction overhead
   - Too large: Exceeds shared memory, reduced occupancy

2. **Vectorize memory access**: Use vector loads (float2, float4)
   - Reduces memory transactions
   - Better coalescing
   - vLLM uses VEC_SIZE template parameter for this

3. **Reuse query vector**: Load once to shared memory
   - Query reused for all tokens in partition
   - Much smaller than K/V (single vector vs sequence)

4. **Online softmax**: Never materialize attention matrix
   - Compute max on-the-fly
   - Compute exp_sum on-the-fly
   - Apply probabilities directly to values

5. **Numerical stability**: Always use max subtraction
   ```cpp
   // BAD: Can overflow
   prob = exp(qk) / sum(exp(qk_all))

   // GOOD: Numerically stable
   max_qk = max(qk_all)
   prob = exp(qk - max_qk) / sum(exp(qk_all - max_qk))
   ```

### vLLM-Specific Patterns

1. **Paged memory**: Design for non-contiguous KV cache
2. **GQA/MQA support**: Map multiple query heads to fewer KV heads
3. **Variable sequence lengths**: Handle different lengths in same batch
4. **FP8 quantization**: Support quantized KV cache with online dequantization

## References

### Papers

1. **Attention Mechanism**:
   - Vaswani et al. (2017). "Attention Is All You Need"
   - Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention" (next tutorial)

2. **vLLM**:
   - Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention"

3. **Optimizations**:
   - Rabe & Staats (2021). "Self-attention Does Not Need O(n²) Memory"
   - Pope et al. (2022). "Efficiently Scaling Transformer Inference"

### Code References

1. **vLLM Attention Implementation**:
   - Main kernel: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh`
   - Launcher: `/home/user/vllm-learn/csrc/attention/paged_attention_v2.cu`
   - Utils: `/home/user/vllm-learn/csrc/attention/attention_utils.cuh`

2. **External References**:
   - [FasterTransformer attention](https://github.com/NVIDIA/FasterTransformer)
   - [xFormers memory-efficient attention](https://github.com/facebookresearch/xformers)

### Tools

1. **Profiling**: NVIDIA Nsight Compute
2. **Visualization**: CUDA occupancy calculator
3. **Debugging**: cuda-gdb, compute-sanitizer

## Summary

vLLM's paged attention implementation demonstrates several advanced CUDA optimization techniques:

1. **Paged memory management**: Enables flexible batching and memory efficiency
2. **Partitioned computation**: Handles arbitrary sequence lengths within shared memory constraints
3. **Warp-level parallelism**: Each warp processes different KV blocks independently
4. **Thread group cooperation**: Multiple threads collaborate on each QK dot product
5. **Numerically stable softmax**: Online computation with max subtraction
6. **Vectorized memory access**: Maximizes memory bandwidth utilization

These patterns are applicable to many sequence-processing kernels in LLM inference.

---

**Next Tutorial**: [04_flash_attention_explained.md](04_flash_attention_explained.md) - Dive into FlashAttention's tiling and recomputation strategies.
