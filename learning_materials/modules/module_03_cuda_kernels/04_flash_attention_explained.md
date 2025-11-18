# Tutorial 04: Flash Attention Explained

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand the Flash Attention algorithm and its memory optimizations
2. Explain the tiling strategy and IO-aware implementation
3. Compare Flash Attention with standard attention implementations
4. Understand how vLLM integrates Flash Attention concepts
5. Implement tiling strategies for memory-efficient kernels

## Prerequisites

- Completion of Tutorial 03: Attention Kernel Walkthrough
- Understanding of attention mechanism mathematics
- Knowledge of SRAM vs HBM memory hierarchy
- Familiarity with GPU memory bandwidth bottlenecks

## Table of Contents

1. [The Memory Problem](#the-memory-problem)
2. [Flash Attention Algorithm](#flash-attention-algorithm)
3. [Tiling Strategy](#tiling-strategy)
4. [Online Softmax with Recomputation](#online-softmax-with-recomputation)
5. [Performance Analysis](#performance-analysis)
6. [vLLM Integration](#vllm-integration)
7. [Flash Attention 2 Improvements](#flash-attention-2-improvements)
8. [Hands-on Exercises](#hands-on-exercises)
9. [Best Practices](#best-practices)
10. [References](#references)

## The Memory Problem

### Standard Attention Implementation

Traditional attention computes in 3 stages:

```
Standard Attention Memory Access:

Stage 1: Compute S = QK^T
┌─────────────────────────────────┐
│ Input:  Q [N × d], K [N × d]    │
│ Output: S [N × N]               │
│ Memory: O(N²) ← Problem!        │
└─────────────────────────────────┘
         ↓ Write S to HBM

Stage 2: Compute P = softmax(S)
┌─────────────────────────────────┐
│ Input:  S [N × N]               │
│ Output: P [N × N]               │
│ Memory: O(N²) ← Problem!        │
└─────────────────────────────────┘
         ↓ Write P to HBM

Stage 3: Compute O = PV
┌─────────────────────────────────┐
│ Input:  P [N × N], V [N × d]    │
│ Output: O [N × d]               │
│ Read from HBM: O(N²)            │
└─────────────────────────────────┘

Total HBM Access: O(N²) reads + O(N²) writes
```

### The Bottleneck

For modern GPUs:
- **SRAM** (shared memory): ~20 TB/s bandwidth, ~20 MB size
- **HBM** (global memory): ~1-2 TB/s bandwidth, ~80 GB size

**Problem**: Attention is bottlenecked by HBM bandwidth, not computation!

```
Example: N=1024, d=64, on A100 GPU

Memory access:
- S matrix: 1024² × 4 bytes = 4 MB
- P matrix: 1024² × 4 bytes = 4 MB
- Total HBM traffic: ~20 MB

Computation:
- FLOPs: 2 × 1024² × 64 = 134M FLOPs
- A100 peak: 312 TFLOPS

Time breakdown:
- Computation: 134M / 312T = 0.4 μs
- Memory transfer: 20 MB / 1.5 TB/s = 13 μs

Result: 97% of time spent on memory transfers!
```

## Flash Attention Algorithm

### Key Insight

**Never materialize the full N×N attention matrix in HBM**

Instead:
1. Tile the computation to fit in SRAM
2. Recompute attention on-the-fly during backward pass
3. Use online softmax algorithm for numerical stability

### Algorithm Overview

```
Flash Attention Strategy:

┌────────────────────────────────────────────────────────┐
│  Divide Q, K, V into blocks that fit in SRAM           │
│  Process blocks iteratively:                           │
│    1. Load Q_block, K_block into SRAM                  │
│    2. Compute S_block = Q_block @ K_block^T in SRAM    │
│    3. Compute P_block = softmax(S_block) in SRAM       │
│    4. Load V_block into SRAM                           │
│    5. Update output with P_block @ V_block             │
│    6. Update running statistics for global softmax     │
│  Never write intermediate S or P to HBM                │
└────────────────────────────────────────────────────────┘

HBM Access: O(N × d) ← Linear in sequence length!
```

### Detailed Algorithm

**Input**: Q, K, V ∈ ℝ^(N×d)
**Output**: O = softmax(QK^T/√d)V ∈ ℝ^(N×d)

**Parameters**:
- B_r: Row block size (typically 128)
- B_c: Column block size (typically 128)
- T_r = ⌈N/B_r⌉: Number of row blocks
- T_c = ⌈N/B_c⌉: Number of column blocks

**Pseudocode**:
```python
# Divide Q into T_r blocks Q_1, ..., Q_Tr of size B_r × d
# Divide K, V into T_c blocks K_1, ..., K_Tc, V_1, ..., V_Tc of size B_c × d

# Initialize output O and statistics
O = zeros(N, d)
l = zeros(N)  # row_sum for softmax
m = -inf * ones(N)  # row_max for softmax

# Process each row block of Q
for i in 1 to T_r:
    # Load Q_i into SRAM
    Load Q_i from HBM to SRAM

    # Initialize block outputs
    O_i = zeros(B_r, d)
    l_i = zeros(B_r)
    m_i = -inf * ones(B_r)

    # Process each column block of K, V
    for j in 1 to T_c:
        # Load K_j, V_j into SRAM
        Load K_j, V_j from HBM to SRAM

        # Compute attention scores (in SRAM)
        S_ij = Q_i @ K_j^T / sqrt(d)  # [B_r × B_c]

        # Online softmax update
        m_new = max(m_i, rowmax(S_ij))

        # Reweight previous output
        α = exp(m_i - m_new)
        O_i = O_i * α

        # Compute current block contribution
        P_ij = exp(S_ij - m_new)  # [B_r × B_c]
        l_new = α * l_i + rowsum(P_ij)

        # Accumulate weighted values
        O_i = O_i + P_ij @ V_j

        # Update statistics
        l_i = l_new
        m_i = m_new

    # Normalize and write output
    O_i = O_i / l_i
    Write O_i to HBM

# Return O
```

## Tiling Strategy

### Memory Layout

```
Q, K, V matrices tiled for SRAM access:

Q (N × d):                  K, V (N × d):
┌─────┬─────┬─────┐        ┌─────┬─────┬─────┐
│ Q_1 │     │     │        │ K_1 │ K_2 │ K_3 │
├─────┤     │     │        └─────┴─────┴─────┘
│ Q_2 │ ... │     │        V_1   V_2   V_3
├─────┤     │     │
│ Q_3 │     │     │
└─────┴─────┴─────┘

B_r = Row block size (Q blocks)
B_c = Column block size (K, V blocks)
```

### Computation Pattern

```
Outer loop: Iterate over Q blocks (rows)
  Inner loop: Iterate over K, V blocks (columns)

        K_1    K_2    K_3
      ┌─────┬─────┬─────┐
Q_1   │ S11 │ S12 │ S13 │  → O_1
      ├─────┼─────┼─────┤
Q_2   │ S21 │ S22 │ S23 │  → O_2
      ├─────┼─────┼─────┤
Q_3   │ S31 │ S32 │ S33 │  → O_3
      └─────┴─────┴─────┘

Each S_ij computed in SRAM, never written to HBM

For Q_i:
  1. Load Q_i once
  2. For each K_j, V_j:
     - Compute S_ij = Q_i @ K_j^T
     - Apply softmax (online algorithm)
     - Accumulate O_i += softmax(S_ij) @ V_j
  3. Write final O_i
```

### Block Size Selection

Block sizes chosen to maximize SRAM utilization:

```cpp
// SRAM budget: 164 KB on A100
// Need to store: Q_block, K_block, V_block, S_block

// For FP16 (2 bytes per element):
// Q_block: B_r × d
// K_block: B_c × d
// V_block: B_c × d
// S_block: B_r × B_c

// Total SRAM: 2 × (B_r×d + B_c×d + B_c×d + B_r×B_c)

// For d=64:
// B_r = B_c = 128
// SRAM = 2 × (128×64 + 128×64 + 128×64 + 128×128)
//      = 2 × (8192 + 8192 + 8192 + 16384)
//      = 81,920 bytes = 80 KB
// Fits comfortably in 164 KB!
```

## Online Softmax with Recomputation

### The Challenge

Computing softmax across blocks requires:
1. Global maximum across all blocks
2. Global sum of exponentials

**Standard approach**: Two passes (find max, then compute softmax)
**Flash Attention**: Online algorithm in one pass

### Online Softmax Algorithm

```
Given: Streaming blocks of scores S_1, S_2, ..., S_n

Maintain:
- m: running maximum
- l: running sum of exp
- d: denominator

For each new block S_i:
  m_new = max(m_old, max(S_i))

  # Reweight previous sum
  α = exp(m_old - m_new)
  l_new = α × l_old + sum(exp(S_i - m_new))

  # Reweight previous output
  O_new = α × O_old + exp(S_i - m_new) @ V_i

  m = m_new
  l = l_new
  O = O_new

Final: O = O / l
```

### Mathematical Correctness

**Claim**: Online algorithm produces same result as batch softmax

**Proof sketch**:
```
Standard softmax:
  p_i = exp(s_i - m) / Σ_j exp(s_j - m)
  where m = max(s_all)

Online algorithm after seeing blocks 1..k:
  m_k = max(s_1, ..., s_k)
  l_k = Σ_{j=1..k} exp(s_j - m_k)
  O_k = Σ_{j=1..k} exp(s_j - m_k) @ V_j

When adding block k+1:
  m_{k+1} = max(m_k, max(s_{k+1}))

  If m_{k+1} = m_k:
    l_{k+1} = l_k + Σ_j exp(s_{k+1,j} - m_k)
    O_{k+1} = O_k + exp(s_{k+1} - m_k) @ V_{k+1}

  If m_{k+1} > m_k:
    α = exp(m_k - m_{k+1})
    l_{k+1} = α × l_k + Σ_j exp(s_{k+1,j} - m_{k+1})
    O_{k+1} = α × O_k + exp(s_{k+1} - m_{k+1}) @ V_{k+1}

Correctness: By induction, maintains invariant that
  O_k = Σ_{j≤k} exp(s_j - m_k) @ V_j
```

### Example Walkthrough

```
Sequence length N=256, d=64, B_r=B_c=128

Block 1 (tokens 0-127):
  Load Q[0:128, :], K[0:128, :], V[0:128, :]
  S_11 = Q[0:128] @ K[0:128]^T  # [128 × 128]
  m_1 = rowmax(S_11)
  l_1 = rowsum(exp(S_11 - m_1))
  O_1 = exp(S_11 - m_1) @ V[0:128] / l_1

Block 2 (tokens 128-255):
  Load K[128:256, :], V[128:256, :]  # Q_1 still in SRAM
  S_12 = Q[0:128] @ K[128:256]^T  # [128 × 128]

  m_new = max(m_1, rowmax(S_12))
  α = exp(m_1 - m_new)

  # Reweight previous output
  O_1 = α × O_1  # Scale down if new max higher

  # Add new contribution
  P_12 = exp(S_12 - m_new)
  l_new = α × l_1 + rowsum(P_12)
  O_1 = O_1 + P_12 @ V[128:256]
  O_1 = O_1 / l_new  # Final normalization

  Write O[0:128, :] = O_1

Process Q_2 similarly...
```

## Performance Analysis

### Memory Complexity

| Algorithm | HBM Reads | HBM Writes | Total HBM I/O |
|-----------|-----------|------------|---------------|
| Standard | O(Nd + N²) | O(N²) | O(N² + Nd) |
| Flash Attention | O(Nd) | O(Nd) | O(Nd) |

**Improvement**: From quadratic to linear in sequence length!

### Computational Complexity

Both algorithms: O(N²d) FLOPs (same)

**Key difference**: Flash Attention recomputes attention during backward pass
- Forward: No intermediate storage
- Backward: Recompute S, P from Q, K (still faster due to reduced I/O)

### Speedup Analysis

```
For N=1024, d=64 on A100:

Standard Attention:
- HBM I/O: ~20 MB
- Time: 20 MB / 1.5 TB/s ≈ 13 μs
- Compute: 134M FLOPs / 312 TFLOPS ≈ 0.4 μs
- Total: ~13.4 μs (I/O bound)

Flash Attention:
- HBM I/O: N×d = 1024×64×4 bytes = 256 KB
- Time: 0.26 MB / 1.5 TB/s ≈ 0.17 μs
- Compute: 134M FLOPs / 312 TFLOPS ≈ 0.4 μs
- Total: ~0.6 μs

Speedup: 13.4 / 0.6 ≈ 22×!

For N=16K (longer sequences), speedup > 50×
```

### Scaling Behavior

```
Time vs Sequence Length:

Standard Attention: T ∝ N²
Flash Attention: T ∝ N

Time
 │
 │     ╱ Standard (quadratic)
 │   ╱
 │  ╱
 │ ╱_________ Flash (linear)
 │╱
 └────────────────── Sequence Length
  1K    4K    16K   64K
```

## vLLM Integration

### PagedAttention and Flash Attention

vLLM combines concepts from both:

**From Flash Attention**:
1. **Tiling**: Partition computation to fit in shared memory
2. **Online softmax**: Numerically stable incremental computation
3. **No materialization**: Never store full attention matrix

**From PagedAttention**:
1. **Paged KV cache**: Non-contiguous memory layout
2. **Block tables**: Indirection for memory access
3. **Variable length sequences**: Different lengths in same batch

### Implementation in vLLM

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 85-105)

vLLM's attention kernel implements Flash Attention principles:

```cpp
// Partitioning (similar to Flash Attention tiling)
const int PARTITION_SIZE = 512;  // Like Flash's B_c
const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);

// Online softmax tracking
float qk_max = -FLT_MAX;  // Running maximum
float exp_sum = 0.0f;     // Running sum

// Process in partitions
for (int partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
  // Load Q (stays in shared memory)
  // Iterate over K, V blocks in this partition

  // Compute S = QK^T in shared memory
  // Update running max and exp_sum (online softmax)
  // Accumulate output weighted by softmax

  // Never write S or P to global memory
}
```

### Differences from Pure Flash Attention

| Aspect | Flash Attention | vLLM PagedAttention |
|--------|----------------|---------------------|
| Memory layout | Contiguous Q, K, V | Paged K, V cache |
| Use case | Training (batch processing) | Inference (auto-regressive) |
| Sequence handling | Fixed-length batches | Variable-length, dynamic |
| Backward pass | Recompute gradients | Not needed (inference) |
| Optimization target | Maximize throughput | Minimize latency + maximize batch |

## Flash Attention 2 Improvements

### Key Enhancements

1. **Better parallelism**:
   - Flash-1: Parallelize over batch × heads
   - Flash-2: Parallelize over batch × heads × sequence

2. **Reduced synchronization**:
   - Fewer atomic operations
   - Better work distribution

3. **Improved occupancy**:
   - Different block sizes for Q and K/V
   - More flexible tiling

### Algorithm Changes

```
Flash Attention 2:

Outer loop: Parallelize over ALL output positions
  - No sequential dependency between output rows
  - Each CUDA block computes one row of output

Within each block:
  - Warp-level parallelism for K, V iteration
  - Reduced shared memory usage
  - Better register utilization

Key insight: Partition work by output elements, not input blocks
```

### Performance Improvements

```
Speedup over Flash Attention 1:

Sequence Length  | FA1 | FA2 | Speedup
512             | 1.0 | 1.4 | 1.4×
2048            | 1.0 | 1.8 | 1.8×
8192            | 1.0 | 2.2 | 2.2×

Longer sequences benefit more due to better parallelism
```

## Hands-on Exercises

### Exercise 1: Calculate Block Sizes

Given SRAM budget and problem dimensions, calculate optimal block sizes:

```python
def calculate_block_sizes(sram_bytes, d, dtype_bytes=2):
    """
    Calculate B_r and B_c for Flash Attention

    Args:
        sram_bytes: Available SRAM (e.g., 164 * 1024 for A100)
        d: Head dimension
        dtype_bytes: Bytes per element (2 for FP16)

    Returns:
        (B_r, B_c): Block sizes
    """
    # TODO: Implement
    # Constraint: dtype_bytes × (B_r×d + B_c×d + B_c×d + B_r×B_c) ≤ sram_bytes
    # Maximize B_r × B_c subject to constraint
    pass

# Test
print(calculate_block_sizes(164 * 1024, d=64))
print(calculate_block_sizes(164 * 1024, d=128))
```

### Exercise 2: Implement Online Softmax

Implement the online softmax update:

```cpp
__device__ void online_softmax_update(
    float* __restrict__ output,       // [d]
    float* __restrict__ row_max,      // scalar
    float* __restrict__ row_sum,      // scalar
    const float* __restrict__ scores, // [block_size]
    const float* __restrict__ values, // [block_size, d]
    int block_size, int d) {

    // TODO: Implement
    // 1. Find max of new scores
    // 2. Update global max
    // 3. Compute correction factor α
    // 4. Reweight previous output and row_sum
    // 5. Add contribution from new block
    // 6. Update running statistics
}
```

### Exercise 3: Analyze Memory Traffic

Calculate HBM traffic for different implementations:

```
Problem: N=4096, d=128, FP16 (2 bytes/element)

a) Standard Attention:
   - S matrix writes: ?
   - P matrix writes: ?
   - P matrix reads: ?
   - Q, K, V reads: ?
   - Output writes: ?
   Total HBM traffic: ?

b) Flash Attention (B_r = B_c = 128):
   - Number of blocks: ?
   - Q reads per block: ?
   - K, V reads per block: ?
   - Output writes: ?
   Total HBM traffic: ?

c) Speedup from memory reduction: ?
```

## Best Practices

### Implementing Flash Attention Principles

1. **Tile to fit SRAM**:
   ```cpp
   // Calculate block sizes based on SRAM availability
   constexpr int SRAM_LIMIT = 48 * 1024;  // 48 KB
   constexpr int ELEM_SIZE = sizeof(scalar_t);
   constexpr int MAX_BLOCK = compute_max_block_size(SRAM_LIMIT, HEAD_SIZE);
   ```

2. **Online softmax for numerical stability**:
   ```cpp
   float m = -FLT_MAX;  // Running max
   float l = 0.0f;      // Running sum

   for (each block) {
     float m_new = max(m, block_max);
     float alpha = exp(m - m_new);
     l = alpha * l + block_sum;
     O = alpha * O + block_contribution;
     m = m_new;
   }
   O = O / l;
   ```

3. **Reuse loaded data**:
   ```cpp
   // Load Q once, reuse for all K/V blocks
   __shared__ scalar_t Q_shared[BLOCK_SIZE][HEAD_SIZE];

   for (each KV block) {
     // Q_shared already loaded - no redundant reads
   }
   ```

4. **Minimize HBM writes**:
   - Accumulate results in registers/shared memory
   - Write final output only once
   - Never materialize intermediate matrices

### When to Use Flash Attention

**Good fit**:
- Long sequences (> 512 tokens)
- Memory-constrained scenarios
- Training with large batch sizes
- Inference with batching

**May not help**:
- Very short sequences (< 128 tokens) - overhead not worth it
- Already compute-bound workloads
- When you have abundant HBM bandwidth

## References

### Papers

1. **Original Flash Attention**:
   - Dao, T., Fu, D., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS 2022.
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. **Flash Attention 2**:
   - Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning".
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

3. **Related Work**:
   - Rabe, M. N., & Staats, C. (2021). "Self-attention Does Not Need O(n²) Memory"
   - Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). "Reformer: The Efficient Transformer"

### Code

1. **Official Flash Attention**:
   - [GitHub: flash-attention](https://github.com/Dao-AILab/flash-attention)
   - CUDA implementation: `csrc/flash_attn/`

2. **vLLM Integration**:
   - PagedAttention V2: `/home/user/vllm-learn/csrc/attention/paged_attention_v2.cu`
   - Attention kernels: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh`

3. **Other Implementations**:
   - [xFormers](https://github.com/facebookresearch/xformers)
   - [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Resources

1. **Tutorials**:
   - [Flash Attention Explained](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
   - [Understanding Flash Attention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)

2. **Profiling Tools**:
   - NVIDIA Nsight Compute
   - PyTorch Profiler with Flash Attention

## Summary

Flash Attention revolutionizes attention computation through:

1. **IO-Aware Tiling**: Partition computation to fit in fast SRAM
2. **Online Softmax**: Compute softmax incrementally without storing intermediate results
3. **Recomputation Strategy**: Trade redundant computation for reduced memory I/O
4. **Linear Memory Complexity**: O(Nd) instead of O(N²) HBM access

**Key Principle**: On modern GPUs, memory bandwidth is more precious than FLOPs

vLLM adapts these principles for inference:
- Combines tiling with paged memory management
- Uses online softmax for variable-length sequences
- Optimizes for auto-regressive decoding patterns

These techniques are essential for scaling LLM inference to long contexts.

---

**Next Tutorial**: [05_kernel_fusion_techniques.md](05_kernel_fusion_techniques.md) - Learn how to fuse multiple operations into single kernels for better performance.
