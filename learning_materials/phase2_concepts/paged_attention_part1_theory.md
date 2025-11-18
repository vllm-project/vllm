# PagedAttention Deep Dive - Part 1: Theory & Foundations

> **Learning Objective**: Understand the problem PagedAttention solves and its theoretical foundations
> **Prerequisites**: Understanding of transformer attention mechanism, GPU memory management basics
> **Time to Complete**: 3-4 hours
> **Next**: Part 2 - Implementation Details

---

## 🎯 Learning Goals

By the end of this tutorial, you will:
- [ ] Understand the memory management problem in LLM inference
- [ ] Explain how PagedAttention solves memory fragmentation
- [ ] Describe the page table abstraction for KV cache
- [ ] Compare PagedAttention with traditional approaches
- [ ] Calculate memory savings with concrete examples

---

## 📚 Background: The LLM Inference Memory Problem

### Traditional Attention in Transformers

In transformer models, the attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q (Query)**: Current token's representation `[1, hidden_dim]`
- **K (Keys)**: All previous tokens' keys `[seq_len, hidden_dim]`
- **V (Values)**: All previous tokens' values `[seq_len, hidden_dim]`

### The KV Cache Concept

During **autoregressive generation** (generating one token at a time):

1. **Without KV Cache** (Inefficient):
   ```
   Step 1: Generate token 1 using prompt
   Step 2: Generate token 2 using prompt + token 1 (recompute ALL attention)
   Step 3: Generate token 3 using prompt + token 1 + token 2 (recompute ALL)
   ...
   ```
   ❌ **Problem**: Recomputing attention for all previous tokens is wasteful!

2. **With KV Cache** (Standard Approach):
   ```
   Step 1: Generate token 1, cache K₁ and V₁
   Step 2: Generate token 2, cache K₂ and V₂, reuse K₁, V₁
   Step 3: Generate token 3, cache K₃ and V₃, reuse K₁, K₂, V₁, V₂
   ...
   ```
   ✅ **Better**: Only compute attention for the new token!

### Memory Requirement Calculation

For a single request:
```
KV cache size = 2 × num_layers × num_heads × seq_len × head_dim × dtype_size

Example (Llama-2 7B):
- num_layers = 32
- num_heads = 32
- max_seq_len = 4096
- head_dim = 128
- dtype = float16 (2 bytes)

Per token: 2 × 32 × 32 × 1 × 128 × 2 = 524,288 bytes ≈ 0.5 MB
Full sequence: 0.5 MB × 4096 = 2048 MB = 2 GB per request!
```

**Key Insight**: For long contexts, KV cache dominates memory usage!

---

## 🔥 The Memory Management Problem

### Problem 1: Over-Allocation and Waste

**Traditional Approach**: Pre-allocate maximum sequence length for each request.

```
┌─────────────────────────────────────────────┐
│ Request 1: "Hi" → Pre-allocated 4096 tokens │
│ Used: 50 tokens                             │
│ Wasted: 4046 tokens (98.8% waste!)         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Request 2: "Hello" → Pre-allocated 4096 tokens│
│ Used: 80 tokens                             │
│ Wasted: 4016 tokens (98% waste!)           │
└─────────────────────────────────────────────┘

Total GPU Memory: 40 GB
Wasted: ~35 GB
Actual Usage: ~5 GB
Efficiency: 12.5% 😱
```

**Why over-allocate?**
- Don't know final sequence length in advance
- Need contiguous memory for efficient GPU operations
- Variable length sequences complicate memory management

### Problem 2: Memory Fragmentation

Even with dynamic allocation, fragmentation occurs:

```
Time T1: Three requests
┌────────┬────────┬────────┐
│ Req A  │ Req B  │ Req C  │
│ 1000   │ 500    │ 800    │
└────────┴────────┴────────┘

Time T2: Request B completes
┌────────┬────────┬────────┐
│ Req A  │ [FREE] │ Req C  │
│ 1000   │ 500    │ 800    │
└────────┴────────┴────────┘
         ↑ Can't use for new 1000-token request!

Time T3: New request D needs 1000 tokens
❌ Not enough contiguous space!
✅ But total free space = 500 tokens available
```

**Result**: Can't accept new requests even with plenty of free memory!

### Problem 3: Unpredictable Sequence Lengths

```python
# Can't predict ahead of time:
request_1 = "Explain quantum computing"  # Might generate 50 or 500 tokens
request_2 = "Hi"                          # Might generate 5 or 50 tokens
request_3 = "Write a story"               # Might generate 1000+ tokens
```

**Dilemma**:
- Allocate too little → Out of memory errors, truncation
- Allocate too much → Wasted memory, fewer concurrent requests

---

## 💡 PagedAttention Solution

### Core Idea: Virtual Memory for KV Cache

Inspired by **operating system virtual memory**:

```
OS Virtual Memory              PagedAttention
─────────────────              ───────────────
Virtual address space     →    Logical KV cache blocks
Physical pages            →    Physical GPU memory blocks
Page table                →    Block table
Page size (4KB)           →    Block size (16 tokens)
```

### Key Innovation: Break Contiguity Requirement

**Traditional**:
```
Request A: Must allocate contiguous 4096 tokens
┌──────────────────────────────────────────────┐
│ [0][1][2][3]...[4095]                        │
└──────────────────────────────────────────────┘
Contiguous in memory!
```

**PagedAttention**:
```
Request A: Allocate in small blocks (e.g., 16 tokens each)
Block 0: [0-15]    → Physical location: 0x1000
Block 1: [16-31]   → Physical location: 0x5000 (non-contiguous!)
Block 2: [32-47]   → Physical location: 0x2000
Block 3: [48-63]   → Physical location: 0x8000
...
```

**Mapping via Block Table**:
```
Logical Block ID → Physical Block ID
0               → 16
1               → 83
2               → 24
3               → 140
...
```

### Benefits

#### 1. No More Over-Allocation
```
Request: "Hi" (generates 50 tokens)

Traditional:
Pre-allocate 4096 tokens = 2 GB
Use: 50 tokens = 25 MB
Waste: 2048 - 25 = 1975 MB (98.8% waste)

PagedAttention:
Allocate: 4 blocks × 16 tokens = 64 tokens = 32 MB
Use: 50 tokens = 25 MB
Waste: 32 - 25 = 7 MB (21.9% waste)

Savings: 1975 MB → 7 MB waste (282x improvement!)
```

#### 2. No Fragmentation
```
Time T1: Allocate blocks
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │ B0 │ A1 │ C0 │ B1 │ A2 │ C1 │ D0 │
└────┴────┴────┴────┴────┴────┴────┴────┘

Time T2: Request B finishes (frees blocks B0, B1)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │[F]│ A1 │ C0 │[F]│ A2 │ C1 │ D0 │
└────┴────┴────┴────┴────┴────┴────┴────┘

Time T3: New request E needs 2 blocks
✅ Can use blocks B0 and B1! (non-contiguous is OK)
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │ E0 │ A1 │ C0 │ E1 │ A2 │ C1 │ D0 │
└────┴────┴────┴────┴────┴────┴────┴────┘
```

#### 3. Dynamic Growth
```
Request A: Starts with 1 block, grows as needed
T1: [Block 0]
T2: [Block 0][Block 1]  ← Allocate new block
T3: [Block 0][Block 1][Block 2]  ← Allocate another
...
```

---

## 🏗️ PagedAttention Architecture

### Components

#### 1. Block (Physical Memory Unit)
```
Block:
- Fixed size (e.g., 16 tokens)
- Stores K and V for those tokens
- Located anywhere in GPU memory

Block Structure for 16 tokens:
┌──────────────────────────────────────┐
│ Keys: [num_layers, 16, hidden_dim]   │
│ Values: [num_layers, 16, hidden_dim] │
└──────────────────────────────────────┘

Memory: ~64KB for Llama-7B (16 tokens × 32 layers × 128 × 2 × 2 bytes)
```

#### 2. Block Table (Mapping)
```
Block Table for Request A:
Logical Block → Physical Block
0            → 23
1            → 87
2            → 5
3            → 142

This means:
- Tokens 0-15 are in physical block 23
- Tokens 16-31 are in physical block 87
- Tokens 32-47 are in physical block 5
- Tokens 48-63 are in physical block 142
```

#### 3. Block Manager (Allocator)
```
Block Manager:
- Tracks free blocks
- Allocates blocks on demand
- Frees blocks when sequences complete
- Implements allocation policies (LRU, etc.)

Free Block List: [5, 12, 23, 45, 67, 89, ...]
```

### Complete Example

**Scenario**: Process 3 requests concurrently

```
Request A: "Translate this text" (will generate ~100 tokens)
Request B: "Hi" (will generate ~10 tokens)
Request C: "Write a poem" (will generate ~200 tokens)

Block size: 16 tokens
```

**Step 1**: Initial allocation (prefill phase)
```
Request A: 30 tokens → Needs 2 blocks (rounds up)
Request B: 5 tokens → Needs 1 block
Request C: 20 tokens → Needs 2 blocks

Block Allocation:
Physical Blocks: [0][1][2][3][4][5][6][7][8][9]...
Allocated:       [A0][B0][C0][A1][C1][-][-][-]...

Block Tables:
A: [0, 3]     → Block 0 has tokens 0-15, Block 3 has tokens 16-29
B: [1]        → Block 1 has tokens 0-4
C: [2, 4]     → Block 2 has tokens 0-15, Block 4 has tokens 16-19
```

**Step 2**: Generation continues
```
Request A: Generates 20 more tokens (now 50 total, needs 4 blocks)
Request B: Generates 8 more tokens (now 13 total, still fits in 1 block)
Request C: Generates 30 more tokens (now 50 total, needs 4 blocks)

New Allocations:
Physical Blocks: [0][1][2][3][4][5][6][7][8][9]...
Allocated:       [A0][B0][C0][A1][C1][A2][C2][A3]...

Block Tables:
A: [0, 3, 5, 7]   → 4 blocks for 50 tokens
B: [1]            → Still 1 block (has room for 16)
C: [2, 4, 6, 8]   → 4 blocks for 50 tokens
```

**Step 3**: Request B completes
```
Free block 1, can be reused immediately!

Physical Blocks: [0][-][2][3][4][5][6][7][8][9]...
                  ↑        ↑
                  A0      C0

Block Table:
A: [0, 3, 5, 7]   → Unchanged
B: [deleted]      → Freed
C: [2, 4, 6, 8]   → Unchanged
```

---

## 📊 Memory Efficiency Analysis

### Comparison: Traditional vs PagedAttention

**Scenario**: Llama-2 7B, 8 concurrent requests on A100 (40GB)

```
Traditional Approach:
─────────────────────
- Max sequence length: 2048
- KV cache per token: 0.5 MB
- Per-request allocation: 2048 × 0.5 MB = 1024 MB = 1 GB
- 8 requests: 8 GB
- Model weights: 14 GB
- Total: 22 GB
- Actual average usage: ~400 tokens per request = 200 MB × 8 = 1.6 GB
- Memory efficiency: 1.6 / 8 = 20%

PagedAttention:
───────────────
- Block size: 16 tokens
- Average sequence length: 400 tokens → 25 blocks per request
- Per-request allocation: 25 × 16 × 0.5 MB = 200 MB
- 8 requests: 1.6 GB
- Model weights: 14 GB
- Total: 15.6 GB
- Memory efficiency: 1.6 / 1.6 = 100%

Savings: 8 GB → 1.6 GB (5x reduction!)
Can fit: 8 requests → 40 requests (5x more throughput!)
```

### Internal Fragmentation Analysis

**Internal Fragmentation**: Wasted space within allocated blocks

```
Block size: 16 tokens

Example sequences:
- 15 tokens → 1 block → waste 1 token (6.7%)
- 17 tokens → 2 blocks → waste 15 tokens (46.9%)
- 32 tokens → 2 blocks → waste 0 tokens (0%)
- 50 tokens → 4 blocks → waste 14 tokens (21.9%)

Average internal fragmentation: ~12.5% (vs 80-90% with pre-allocation)
```

**Block Size Trade-off**:
```
Smaller blocks (8 tokens):
+ Less fragmentation
- More block table entries
- More non-contiguous accesses

Larger blocks (32 tokens):
+ Fewer block table entries
+ More contiguous accesses
- More fragmentation

Optimal: 16-32 tokens based on profiling
```

---

## 🔬 Mathematical Formulation

### Attention Computation with Paging

**Standard Attention**:
```
For token i, compute attention with all previous tokens:
Q_i: [1, hidden_dim]
K: [i, hidden_dim]  ← Contiguous
V: [i, hidden_dim]  ← Contiguous

scores = Q_i @ K.T / √d_k
output = softmax(scores) @ V
```

**Paged Attention**:
```
For token i, with block table [b₀, b₁, ..., b_n]:

For each block b_j:
  - Fetch K_j from physical block: [block_size, hidden_dim]
  - Fetch V_j from physical block: [block_size, hidden_dim]
  - Compute partial scores: s_j = Q_i @ K_j.T / √d_k

Concatenate all scores: s = [s₀, s₁, ..., s_n]
Apply softmax: attn_weights = softmax(s)
Compute output: output = Σⱼ (attn_weights_j @ V_j)

Key difference: K and V are NOT contiguous in memory!
```

### Memory Access Pattern

**Traditional (Contiguous)**:
```
Memory addresses for sequence of length 100:
K: [0x1000, 0x1064, 0x10C8, ..., 0x2800]  ← Sequential
V: [0x2900, 0x2964, 0x29C8, ..., 0x4100]  ← Sequential

Access pattern: Sequential (cache-friendly)
```

**PagedAttention (Non-Contiguous)**:
```
Block table: [23, 87, 5, 142]

K blocks:
- Block 23: 0x5C00 - 0x5FFF
- Block 87: 0x15C00 - 0x15FFF
- Block 5: 0x1400 - 0x17FF
- Block 142: 0x23800 - 0x23BFF

Access pattern: Random (need to optimize!)
```

**Challenge**: How to maintain performance with non-contiguous access?
**Answer**: Custom CUDA kernels (covered in Part 2!)

---

## 📈 Performance Implications

### Throughput Improvement

```
GPU: A100 (40GB)
Model: Llama-2 7B
Workload: Average 500 tokens/request

Traditional:
- Max concurrent requests: 8 (limited by memory)
- Throughput: 8 requests × 20 tokens/sec = 160 tokens/sec

PagedAttention:
- Max concurrent requests: 40 (memory efficient)
- Throughput: 40 requests × 20 tokens/sec = 800 tokens/sec

Speedup: 5x higher throughput! 🚀
```

### Latency Considerations

**Potential slowdowns**:
1. Block table lookups
2. Non-contiguous memory access
3. Kernel overhead for block iteration

**Optimizations** (Part 2):
1. Fused kernels that iterate over blocks efficiently
2. Coalesced memory access within blocks
3. Shared memory for block metadata

**Result**: < 5% latency overhead with 5x throughput gain!

---

## 🧩 Key Concepts Summary

### Vocabulary

| Term | Definition |
|------|------------|
| **KV Cache** | Cached key and value tensors from previous tokens |
| **Block** | Fixed-size chunk of memory (e.g., 16 tokens) |
| **Block Table** | Mapping from logical to physical blocks |
| **Block Manager** | Allocates and frees blocks |
| **Logical Blocks** | Sequential block IDs (0, 1, 2, ...) |
| **Physical Blocks** | Actual GPU memory locations |
| **Internal Fragmentation** | Wasted space within allocated blocks |
| **External Fragmentation** | Unusable space between allocated regions |

### Core Principles

1. **Page-based Memory Management**
   - Break KV cache into fixed-size blocks
   - Non-contiguous physical memory OK

2. **Dynamic Allocation**
   - Allocate blocks on demand
   - No over-provisioning

3. **Efficient Reuse**
   - Freed blocks immediately available
   - No fragmentation issues

4. **Virtual-to-Physical Mapping**
   - Block table provides indirection
   - Enables flexible memory management

---

## 🎯 Self-Assessment Quiz

### Question 1: Memory Calculation
```
Given:
- Model: 32 layers, 32 heads, 128 head_dim, FP16
- Sequence: 1024 tokens
- Traditional approach: Pre-allocate 4096 tokens

Calculate:
a) KV cache size for actual sequence (1024 tokens)
b) KV cache size with traditional approach (4096 pre-allocated)
c) Percentage of waste

Answers:
a) 2 × 32 × 32 × 1024 × 128 × 2 = 536,870,912 bytes = 512 MB
b) 2 × 32 × 32 × 4096 × 128 × 2 = 2,147,483,648 bytes = 2048 MB = 2 GB
c) (2048 - 512) / 2048 = 75% waste
```

### Question 2: Block Allocation
```
Block size: 16 tokens
Request generates: 50 tokens

How many blocks needed?
What's the internal fragmentation?

Answers:
Blocks needed: ⌈50 / 16⌉ = 4 blocks
Allocated space: 4 × 16 = 64 tokens
Wasted: 64 - 50 = 14 tokens
Fragmentation: 14 / 64 = 21.9%
```

### Question 3: Throughput Calculation
```
GPU Memory: 80 GB (H100)
Model size: 20 GB
KV cache per token: 1 MB
Average sequence length: 800 tokens

Traditional (pre-allocate 2048):
- Per request: 2048 MB
- Available: 60 GB
- Concurrent: 60,000 / 2048 = 29 requests

PagedAttention (block size 16):
- Per request: 50 blocks × 16 MB = 800 MB
- Available: 60 GB
- Concurrent: 60,000 / 800 = 75 requests

Throughput gain: 75 / 29 = 2.59x
```

---

## 🚀 What's Next?

In **Part 2: Implementation**, you'll learn:
- CUDA kernel implementation of paged attention
- Block table management in C++
- Memory coalescing strategies for non-contiguous access
- Performance optimization techniques
- Actual vLLM code walkthrough

In **Part 3: Optimization**, you'll learn:
- Kernel fusion opportunities
- Multi-GPU paged attention
- Quantization with paging
- Advanced scheduling with memory awareness

---

## 📚 Additional Reading

1. **vLLM Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **Blog Post**: [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
3. **Virtual Memory**: "Operating Systems: Three Easy Pieces" - Chapter on Paging
4. **Flash Attention**: Compare with another memory-efficient attention approach

---

**Next**: `paged_attention_part2_implementation.md` 🔧
