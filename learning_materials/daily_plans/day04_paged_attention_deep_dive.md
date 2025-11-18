# Day 4: PagedAttention Deep Dive - Algorithm & Implementation

> **Goal**: Master PagedAttention algorithm, understand memory benefits, trace CUDA implementation
> **Time**: 6-8 hours
> **Prerequisites**: Day 1-3 completed, solid understanding of attention mechanism, basic CUDA knowledge
> **Deliverables**: PagedAttention explainer document, simplified PyTorch implementation, memory comparison analysis

---

## 📅 Daily Schedule

### Morning Session (3-4 hours): Theory & Algorithm

**9:00-9:45** - Attention Mechanism Review (Q, K, V)
**9:45-10:45** - Traditional KV Cache Problems
**10:45-11:00** - Break
**11:00-12:30** - PagedAttention Algorithm Deep Dive

### Afternoon Session (3-4 hours): Implementation

**14:00-15:00** - Python Implementation Walkthrough
**15:00-16:00** - CUDA Kernel Analysis
**16:00-16:30** - Break
**16:30-18:00** - Hands-On: Implement Simplified Version

### Evening (Optional, 1-2 hours): Experiments

**19:00-21:00** - Memory profiling, performance comparison, optimization ideas

---

## 🎯 Learning Objectives

By end of day, you should be able to:
- [ ] Explain attention mechanism and KV cache clearly
- [ ] Describe traditional KV cache memory problems
- [ ] Explain PagedAttention algorithm in detail
- [ ] Calculate memory savings with PagedAttention
- [ ] Read and understand PagedAttention CUDA kernels
- [ ] Implement simplified PagedAttention in PyTorch
- [ ] Analyze performance trade-offs

---

## 📚 Morning: PagedAttention Theory (9:00-12:30)

### Task 1: Attention Mechanism Review (45 min)

**Standard Self-Attention**:

```python
# Simplified attention computation

def attention(Q, K, V):
    """
    Args:
        Q: Query matrix [batch, seq_len, head_dim]
        K: Key matrix [batch, seq_len, head_dim]
        V: Value matrix [batch, seq_len, head_dim]

    Returns:
        Output [batch, seq_len, head_dim]
    """
    # 1. Compute attention scores
    scores = Q @ K.transpose(-2, -1)  # [batch, seq_len, seq_len]

    # 2. Scale
    scores = scores / sqrt(head_dim)

    # 3. Softmax (attention weights)
    attn_weights = softmax(scores, dim=-1)  # [batch, seq_len, seq_len]

    # 4. Weighted sum of values
    output = attn_weights @ V  # [batch, seq_len, head_dim]

    return output
```

**Key Insight**: For each query position, we need access to ALL keys and values.

**In Autoregressive Generation**:

```
Step 1 (Prefill): Process "Hello world"
  Q: [2, head_dim]  (2 tokens)
  K: [2, head_dim]
  V: [2, head_dim]
  → Generate token "is"

Step 2 (Decode): Process "is" (new token)
  Q: [1, head_dim]  (1 new token)
  K: [3, head_dim]  (2 old + 1 new) ← Need to recompute K for old tokens? NO!
  V: [3, head_dim]  (2 old + 1 new)
  → Generate token "great"
```

**💡 Solution: KV Cache**

Cache the K and V vectors from previous tokens:

```
Step 1:
  Compute K, V for "Hello world"
  Store: KV_cache = {K: [2, head_dim], V: [2, head_dim]}

Step 2:
  Compute K, V ONLY for "is"
  Append to cache: KV_cache = {K: [3, head_dim], V: [3, head_dim]}
  Use full cache for attention with new query

Benefit: Don't recompute K, V for already-processed tokens!
```

### Task 2: Traditional KV Cache Problems (60 min)

**Memory Layout - Naive Approach**:

```python
# Allocate max possible size upfront
max_seq_len = 2048
batch_size = 32
num_layers = 32
num_heads = 32
head_dim = 128

# Per-sequence cache size
cache_size_per_seq = (
    2                    # K and V
    * num_layers         # Each layer has its own cache
    * num_heads
    * max_seq_len        # Maximum sequence length
    * head_dim
    * 2                  # bytes per fp16
)

cache_size_per_seq = 2 * 32 * 32 * 2048 * 128 * 2
                   = 1.07 GB per sequence!

# For batch of 32:
total_cache = 32 * 1.07 GB = 34.4 GB
```

**❌ Problem 1: Memory Waste**

Most sequences don't reach max_seq_len:

```
Allocated: [████████████████████████████████] 2048 tokens (1.07 GB)
Actually used: [████]                         100 tokens  (0.05 GB)
Wasted: 95% of allocated memory!
```

**❌ Problem 2: Memory Fragmentation**

```
Sequence A: [████----] 512 tokens (finished, freed)
Sequence B: [████████] 1024 tokens
Sequence C: [██------] 256 tokens
Sequence D: needs 600 tokens

Problem: Total free space = 512 + gaps, but not contiguous!
Cannot fit Sequence D even though total space available.
```

**❌ Problem 3: Inflexible Allocation**

```
User request: "Generate 100 tokens"
Actually generates: 50 tokens (stopped early)

Waste: Allocated for 100, used 50 → 50% waste
```

**📊 Real-World Impact**:

```
Without optimization:
  - A100 40GB GPU
  - OPT-13B model (26 GB for weights)
  - Remaining for KV cache: 14 GB
  - With naive allocation: Can only fit ~13 sequences (batch_size=13)

With PagedAttention:
  - Same 14 GB for KV cache
  - Can fit ~100+ sequences (batch_size=100+)
  - 7-8x improvement in throughput!
```

### Task 3: PagedAttention Algorithm (90 min)

**🔑 Core Idea**: Borrow virtual memory paging from operating systems!

**Virtual Memory Analogy**:

```
Virtual Memory (OS):
  - Program sees contiguous virtual address space
  - OS maps to non-contiguous physical pages
  - Pages allocated on-demand

PagedAttention:
  - Sequence sees contiguous logical KV cache
  - System maps to non-contiguous physical blocks
  - Blocks allocated on-demand
```

**PagedAttention Design**:

```
1. Divide KV cache into BLOCKS (like OS pages)
   Block size: 16 tokens (configurable)

2. Maintain BLOCK TABLE for each sequence
   Maps logical block index → physical block index

3. Allocate blocks ON-DEMAND as sequence grows

4. Attention kernel uses block table to find KV vectors
```

**Example**:

```
Sequence A: 42 tokens
  Logical blocks: 3 blocks (42 / 16 = 2.625 → round up to 3)
  Physical allocation:

  Block Table for Seq A:
  Logical  | Physical
  ---------|----------
     0     |    5      ← KV for tokens 0-15 stored in physical block 5
     1     |    2      ← KV for tokens 16-31 stored in physical block 2
     2     |    8      ← KV for tokens 32-42 stored in physical block 8

Physical Memory (GPU):
  Block 0: [Seq B data...]
  Block 1: [Seq C data...]
  Block 2: [Seq A tokens 16-31] ← Non-contiguous!
  Block 3: [Free]
  Block 4: [Seq D data...]
  Block 5: [Seq A tokens 0-15]
  Block 6: [Free]
  Block 7: [Seq B data...]
  Block 8: [Seq A tokens 32-42]
  ...
```

**Attention Computation with Paging**:

```python
# Pseudocode for paged attention

def paged_attention(
    query,           # [num_heads, head_dim]
    key_cache,       # [num_blocks, block_size, num_heads, head_dim]
    value_cache,     # [num_blocks, block_size, num_heads, head_dim]
    block_table,     # [num_logical_blocks] - mapping
    context_len,     # Actual sequence length
):
    """
    Compute attention using paged KV cache.
    """
    num_logical_blocks = len(block_table)
    output = zeros([num_heads, head_dim])

    # For each logical block
    for logical_idx in range(num_logical_blocks):
        # Get physical block index
        physical_idx = block_table[logical_idx]

        # Get keys and values from physical block
        keys = key_cache[physical_idx]    # [block_size, num_heads, head_dim]
        values = value_cache[physical_idx]

        # Determine how many tokens in this block
        if logical_idx < num_logical_blocks - 1:
            num_tokens = block_size  # Full block
        else:
            num_tokens = context_len % block_size  # Partial last block

        # Compute attention for this block
        # Q @ K^T
        scores = query @ keys[:num_tokens].transpose(-2, -1)

        # Softmax and apply to values
        attn_weights = softmax(scores)
        output += attn_weights @ values[:num_tokens]

    return output
```

**🎯 Key Benefits**:

1. **No Fragmentation**:
   - Free blocks can be allocated to any sequence
   - No need for contiguous memory

2. **On-Demand Allocation**:
   - Only allocate blocks as sequence grows
   - No waste from over-allocation

3. **Memory Sharing** (for beam search, prefix sharing):
   - Multiple sequences can share same physical blocks
   - Example: Same prompt, different continuations

**Memory Calculation Example**:

```
Configuration:
  - Block size: 16 tokens
  - Block memory: 16 * num_layers * num_heads * head_dim * 2 (K+V) * 2 bytes
  - For OPT-13B: 16 * 40 * 40 * 128 * 2 * 2 = 13.1 MB per block

GPU Memory: 40 GB
  - Model weights: 26 GB
  - Activations: 2 GB
  - Available for KV cache: 12 GB

Number of blocks: 12 GB / 13.1 MB = ~915 blocks

Without paging (naive):
  - Each sequence needs max_len / block_size blocks
  - Max_len = 2048 → 128 blocks per sequence
  - Max batch size = 915 / 128 = 7 sequences

With paging (actual usage):
  - Average sequence length: 256 tokens
  - Average blocks needed: 256 / 16 = 16 blocks per sequence
  - Max batch size = 915 / 16 = 57 sequences
  - 8x improvement!
```

---

## 💻 Afternoon: Implementation Deep Dive (14:00-18:00)

### Task 4: Python Implementation Walkthrough (60 min)

**File**: `vllm/attention/ops/paged_attn.py`

```python
# vllm/attention/ops/paged_attn.py (simplified)

import torch

def paged_attention_v1(
    output: torch.Tensor,           # [num_seqs, num_heads, head_dim] - output
    query: torch.Tensor,            # [num_seqs, num_heads, head_dim] - queries
    key_cache: torch.Tensor,        # [num_blocks, block_size, num_heads, head_dim]
    value_cache: torch.Tensor,      # [num_blocks, block_size, num_heads, head_dim]
    num_kv_heads: int,              # Number of KV heads (for GQA)
    scale: float,                   # Attention scale (1/sqrt(head_dim))
    block_tables: torch.Tensor,     # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,     # [num_seqs] - actual sequence lengths
    block_size: int,
    max_context_len: int,
) -> None:
    """
    Paged attention kernel launcher.

    This is a wrapper around CUDA kernel.
    """
    # Determine kernel version based on head_dim and dtype
    if query.dtype == torch.float32:
        # Call FP32 kernel
        from vllm._C import paged_attention_v1_f32
        paged_attention_v1_f32(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
        )
    elif query.dtype == torch.float16:
        # Call FP16 kernel
        from vllm._C import paged_attention_v1_f16
        paged_attention_v1_f16(...)
    else:
        raise ValueError(f"Unsupported dtype: {query.dtype}")
```

**File**: `vllm/attention/backends/flash_attn.py`

```python
# vllm/attention/backends/flash_attn.py

class FlashAttentionBackend:
    """Attention backend using FlashAttention and PagedAttention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with mixed prefill/decode.
        """
        # Separate prefill and decode
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        # === PREFILL ===
        if num_prefill_tokens > 0:
            # Use FlashAttention for prefill (efficient for long sequences)
            prefill_output = self._run_flash_attention(
                query[:num_prefill_tokens],
                key[:num_prefill_tokens],
                value[:num_prefill_tokens],
            )

            # Store K, V in cache
            self._store_kv_cache(
                kv_cache,
                key[:num_prefill_tokens],
                value[:num_prefill_tokens],
                attn_metadata.slot_mapping[:num_prefill_tokens],
            )

        # === DECODE ===
        if num_decode_tokens > 0:
            # Use PagedAttention for decode (efficient with KV cache)
            decode_output = paged_attention_v1(
                query=query[num_prefill_tokens:],
                key_cache=kv_cache[0],    # Paged K cache
                value_cache=kv_cache[1],  # Paged V cache
                block_tables=attn_metadata.block_tables,
                context_lens=attn_metadata.context_lens,
                block_size=attn_metadata.block_size,
                scale=self.scale,
            )

        # Combine outputs
        if num_prefill_tokens > 0 and num_decode_tokens > 0:
            output = torch.cat([prefill_output, decode_output])
        elif num_prefill_tokens > 0:
            output = prefill_output
        else:
            output = decode_output

        return output
```

### Task 5: CUDA Kernel Analysis (60 min)

**File**: `csrc/attention/attention_kernels.cu`

**Kernel Signature**:

```cuda
// csrc/attention/attention_kernels.cu

template<
    typename scalar_t,      // Data type (float, half, etc.)
    int HEAD_SIZE,          // Head dimension (64, 128, etc.)
    int BLOCK_SIZE,         // Block size (16 tokens)
    int NUM_THREADS         // Threads per block
>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,              // Output [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,          // Query [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,    // Key cache [num_blocks, block_size, num_heads, head_size]
    const scalar_t* __restrict__ v_cache,    // Value cache [num_blocks, block_size, num_heads, head_size]
    const int num_kv_heads,                  // Number of KV heads
    const float scale,                       // Attention scale
    const int* __restrict__ block_tables,    // Block tables [num_seqs, max_blocks]
    const int* __restrict__ context_lens,    // Context lengths [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes   // Optional ALiBi slopes
) {
    // Each thread block handles one sequence and one head
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // Load query for this sequence and head
    const int q_offset = seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    __shared__ scalar_t q_shared[HEAD_SIZE];

    // Load query into shared memory (parallel load)
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
        q_shared[i] = q[q_offset + i];
    }
    __syncthreads();

    // Attention computation
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for attention scores (one per thread)
    __shared__ float scores[NUM_THREADS];
    __shared__ float max_score;
    __shared__ float sum_exp;

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    // === Phase 1: Compute attention scores ===
    // Each thread processes some tokens
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block_idx = block_table[block_idx];

        // Number of tokens in this block
        const int block_token_count = min(BLOCK_SIZE,
                                          context_len - block_idx * BLOCK_SIZE);

        // Each thread handles some tokens in the block
        for (int i = threadIdx.x; i < block_token_count; i += blockDim.x) {
            // Global token index
            const int token_idx = block_idx * BLOCK_SIZE + i;

            // Get key vector from cache
            const int k_offset = physical_block_idx * BLOCK_SIZE * num_heads * HEAD_SIZE
                               + i * num_heads * HEAD_SIZE
                               + head_idx * HEAD_SIZE;

            // Compute Q · K^T
            float qk = 0.0f;
            for (int d = 0; d < HEAD_SIZE; ++d) {
                qk += float(q_shared[d]) * float(k_cache[k_offset + d]);
            }

            // Apply scale
            qk *= scale;

            // Track max for numerical stability
            thread_max = fmaxf(thread_max, qk);
            scores[token_idx] = qk;
        }
    }

    // === Phase 2: Softmax ===
    // Find global max (reduction across threads)
    __syncthreads();
    if (threadIdx.x == 0) {
        max_score = thread_max;
        for (int i = 1; i < blockDim.x; ++i) {
            max_score = fmaxf(max_score, scores[i]);
        }
    }
    __syncthreads();

    // Compute exp(score - max) and sum
    thread_sum = 0.0f;
    for (int token_idx = threadIdx.x; token_idx < context_len; token_idx += blockDim.x) {
        float exp_score = expf(scores[token_idx] - max_score);
        scores[token_idx] = exp_score;
        thread_sum += exp_score;
    }

    // Global sum (reduction)
    __syncthreads();
    if (threadIdx.x == 0) {
        sum_exp = thread_sum;
        for (int i = 1; i < blockDim.x; ++i) {
            sum_exp += scores[i];
        }
    }
    __syncthreads();

    // === Phase 3: Weighted sum of values ===
    // Each thread computes part of the output
    for (int d = threadIdx.x; d < HEAD_SIZE; d += blockDim.x) {
        float acc = 0.0f;

        for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const int physical_block_idx = block_table[block_idx];
            const int block_token_count = min(BLOCK_SIZE,
                                              context_len - block_idx * BLOCK_SIZE);

            for (int i = 0; i < block_token_count; ++i) {
                const int token_idx = block_idx * BLOCK_SIZE + i;

                // Get value vector
                const int v_offset = physical_block_idx * BLOCK_SIZE * num_heads * HEAD_SIZE
                                   + i * num_heads * HEAD_SIZE
                                   + head_idx * HEAD_SIZE
                                   + d;

                // Weighted sum: attention_weight * value
                acc += (scores[token_idx] / sum_exp) * float(v_cache[v_offset]);
            }
        }

        // Write output
        out[q_offset + d] = scalar_t(acc);
    }
}
```

**🔍 Understanding the Kernel**:

1. **Thread Organization**:
   - Each thread block handles: (sequence, head) pair
   - Threads within block process tokens in parallel

2. **Shared Memory Usage**:
   - `q_shared`: Query vector (reused for all K·Q computations)
   - `scores`: Attention scores (intermediate results)

3. **Three Phases**:
   - Compute Q·K^T for all tokens (using block table)
   - Softmax normalization
   - Weighted sum with V

4. **Block Table Indirection**:
   - `physical_block_idx = block_table[logical_idx]`
   - Enables non-contiguous KV cache storage

### Task 6: Implement Simplified Version (90 min)

**Exercise: PyTorch Implementation**

Create `simplified_paged_attention.py`:

```python
#!/usr/bin/env python3
"""
Day 4: Simplified PagedAttention implementation in PyTorch
"""

import torch
import torch.nn.functional as F
import math

class SimplePagedAttention:
    """Simplified PagedAttention for educational purposes."""

    def __init__(self, block_size=16):
        self.block_size = block_size

    def allocate_kv_cache(self, num_blocks, num_layers, num_heads, head_dim):
        """
        Allocate paged KV cache.

        Returns:
            key_cache: [num_blocks, block_size, num_heads, head_dim]
            value_cache: [num_blocks, block_size, num_heads, head_dim]
        """
        key_cache = torch.zeros(
            num_blocks, self.block_size, num_heads, head_dim,
            dtype=torch.float32, device='cuda'
        )
        value_cache = torch.zeros(
            num_blocks, self.block_size, num_heads, head_dim,
            dtype=torch.float32, device='cuda'
        )
        return key_cache, value_cache

    def write_kv_cache(
        self,
        key_cache,
        value_cache,
        keys,          # [num_tokens, num_heads, head_dim]
        values,        # [num_tokens, num_heads, head_dim]
        slot_mapping,  # [num_tokens] - which slot to write
    ):
        """Write keys and values to paged cache."""
        num_tokens = keys.shape[0]

        for i in range(num_tokens):
            slot_idx = slot_mapping[i]

            # Decode slot index to (block_idx, offset)
            block_idx = slot_idx // self.block_size
            block_offset = slot_idx % self.block_size

            # Write to cache
            key_cache[block_idx, block_offset] = keys[i]
            value_cache[block_idx, block_offset] = values[i]

    def paged_attention(
        self,
        query,        # [num_heads, head_dim]
        key_cache,    # [num_blocks, block_size, num_heads, head_dim]
        value_cache,  # [num_blocks, block_size, num_heads, head_dim]
        block_table,  # [num_logical_blocks] - logical to physical mapping
        context_len,  # Actual sequence length
    ):
        """
        Compute attention using paged KV cache.

        Returns:
            output: [num_heads, head_dim]
        """
        num_heads, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        # Collect all keys and values using block table
        all_keys = []
        all_values = []

        num_logical_blocks = len(block_table)

        for logical_idx in range(num_logical_blocks):
            physical_idx = block_table[logical_idx]

            # Determine tokens in this block
            if logical_idx < num_logical_blocks - 1:
                num_tokens = self.block_size
            else:
                num_tokens = context_len - logical_idx * self.block_size

            # Get keys and values
            keys_block = key_cache[physical_idx, :num_tokens]  # [num_tokens, num_heads, head_dim]
            values_block = value_cache[physical_idx, :num_tokens]

            all_keys.append(keys_block)
            all_values.append(values_block)

        # Concatenate
        all_keys = torch.cat(all_keys, dim=0)      # [context_len, num_heads, head_dim]
        all_values = torch.cat(all_values, dim=0)  # [context_len, num_heads, head_dim]

        # Transpose for attention
        # query: [num_heads, head_dim]
        # all_keys: [context_len, num_heads, head_dim]

        # Compute attention
        output = torch.zeros_like(query)

        for h in range(num_heads):
            q_h = query[h]  # [head_dim]
            k_h = all_keys[:, h, :]  # [context_len, head_dim]
            v_h = all_values[:, h, :]  # [context_len, head_dim]

            # Attention scores
            scores = torch.matmul(k_h, q_h) * scale  # [context_len]

            # Softmax
            attn_weights = F.softmax(scores, dim=0)  # [context_len]

            # Weighted sum
            output[h] = torch.matmul(attn_weights, v_h)  # [head_dim]

        return output


# === Test the implementation ===

def test_paged_attention():
    """Test simplified PagedAttention."""
    print("Testing Simplified PagedAttention\n")

    # Configuration
    block_size = 16
    num_blocks = 10
    num_heads = 8
    head_dim = 64
    context_len = 42  # Requires 3 blocks (42 / 16 = 2.625 → 3)

    # Initialize
    pa = SimplePagedAttention(block_size=block_size)

    # Allocate cache
    key_cache, value_cache = pa.allocate_kv_cache(
        num_blocks=num_blocks,
        num_layers=1,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    # Simulate writing to cache
    keys = torch.randn(context_len, num_heads, head_dim, device='cuda')
    values = torch.randn(context_len, num_heads, head_dim, device='cuda')

    # Block table for this sequence (using physical blocks 2, 5, 7)
    block_table = torch.tensor([2, 5, 7], dtype=torch.long)

    # Create slot mapping
    slot_mapping = []
    for i in range(context_len):
        logical_block = i // block_size
        offset = i % block_size
        physical_block = block_table[logical_block]
        slot_idx = physical_block * block_size + offset
        slot_mapping.append(slot_idx)

    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long)

    # Write to cache
    pa.write_kv_cache(key_cache, value_cache, keys, values, slot_mapping)

    # Query for attention
    query = torch.randn(num_heads, head_dim, device='cuda')

    # Run paged attention
    output = pa.paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        context_len=context_len,
    )

    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected: [{num_heads}, {head_dim}]")

    # Compare with standard attention
    all_keys = torch.cat([
        key_cache[2, :16],
        key_cache[5, :16],
        key_cache[7, :10],  # Last block partial
    ], dim=0)
    all_values = torch.cat([
        value_cache[2, :16],
        value_cache[5, :16],
        value_cache[7, :10],
    ], dim=0)

    # Standard attention
    standard_output = torch.zeros_like(query)
    scale = 1.0 / math.sqrt(head_dim)

    for h in range(num_heads):
        scores = torch.matmul(all_keys[:, h, :], query[h]) * scale
        attn_weights = F.softmax(scores, dim=0)
        standard_output[h] = torch.matmul(attn_weights, all_values[:, h, :])

    # Compare
    diff = torch.abs(output - standard_output).max()
    print(f"\n📊 Max difference from standard attention: {diff:.6f}")

    if diff < 1e-5:
        print("✅ Implementation correct!")
    else:
        print("❌ Implementation has errors")

if __name__ == "__main__":
    test_paged_attention()
```

Run the test:
```bash
python simplified_paged_attention.py
```

**Exercise 2: Memory Comparison**

Create `memory_comparison.py`:

```python
#!/usr/bin/env python3
"""Compare memory usage: Naive vs PagedAttention"""

def calculate_memory(config, use_paging=False):
    """Calculate KV cache memory requirements."""

    num_layers = config['num_layers']
    num_heads = config['num_heads']
    head_dim = config['head_dim']
    block_size = config.get('block_size', 16)
    bytes_per_element = 2  # FP16

    if use_paging:
        # PagedAttention: allocate blocks on-demand
        avg_seq_len = config['avg_seq_len']
        num_seqs = config['num_seqs']

        # Blocks per sequence
        blocks_per_seq = (avg_seq_len + block_size - 1) // block_size

        # Total blocks needed
        total_blocks = blocks_per_seq * num_seqs

        # Memory per block
        mem_per_block = (
            2  # K and V
            * block_size
            * num_layers
            * num_heads
            * head_dim
            * bytes_per_element
        )

        total_memory = total_blocks * mem_per_block

    else:
        # Naive: allocate max_seq_len for each sequence
        max_seq_len = config['max_seq_len']
        num_seqs = config['num_seqs']

        mem_per_seq = (
            2  # K and V
            * max_seq_len
            * num_layers
            * num_heads
            * head_dim
            * bytes_per_element
        )

        total_memory = num_seqs * mem_per_seq

    return total_memory

# Example: OPT-13B
config = {
    'num_layers': 40,
    'num_heads': 40,
    'head_dim': 128,
    'block_size': 16,
    'max_seq_len': 2048,
    'avg_seq_len': 256,  # Average actual usage
    'num_seqs': 32,
}

naive_memory = calculate_memory(config, use_paging=False)
paged_memory = calculate_memory(config, use_paging=True)

print("Memory Comparison: Naive vs PagedAttention")
print("=" * 50)
print(f"Configuration: OPT-13B")
print(f"  Layers: {config['num_layers']}")
print(f"  Heads: {config['num_heads']}")
print(f"  Head dim: {config['head_dim']}")
print(f"  Max seq len: {config['max_seq_len']}")
print(f"  Avg seq len: {config['avg_seq_len']}")
print(f"  Batch size: {config['num_seqs']}")
print()
print(f"Naive allocation: {naive_memory / 1e9:.2f} GB")
print(f"PagedAttention: {paged_memory / 1e9:.2f} GB")
print(f"Savings: {(naive_memory - paged_memory) / 1e9:.2f} GB ({100 * (1 - paged_memory/naive_memory):.1f}%)")
print(f"Efficiency: {paged_memory / naive_memory:.2f}x memory")
```

---

## 📝 End of Day Summary

### What You Learned Today

✅ **Attention Mechanism**
- Q, K, V computation
- KV cache concept and benefits
- Autoregressive generation process

✅ **Traditional KV Cache Problems**
- Memory waste from pre-allocation
- Fragmentation issues
- Inflexibility

✅ **PagedAttention Algorithm**
- Virtual memory analogy
- Block-based storage
- Block table mapping
- On-demand allocation

✅ **Implementation Details**
- Python wrapper code
- CUDA kernel structure
- Three-phase computation (scores, softmax, weighted sum)
- Simplified PyTorch implementation

✅ **Performance Benefits**
- 7-8x memory efficiency
- Higher batch sizes
- Better GPU utilization

### Knowledge Check (Quiz)

**Question 1**: Why do we need KV cache in autoregressive generation?
<details>
<summary>Answer</summary>
To avoid recomputing Key and Value vectors for already-processed tokens. Each decode step only computes K, V for the new token and reuses cached values, saving ~2N FLOPs per token where N is sequence length.
</details>

**Question 2**: What is the main problem with naive KV cache allocation?
<details>
<summary>Answer</summary>
Memory waste: Must pre-allocate for max possible length, but most sequences are shorter. Example: allocate 2048 tokens but use 100 → 95% waste. Also causes fragmentation preventing efficient packing.
</details>

**Question 3**: How does PagedAttention solve the fragmentation problem?
<details>
<summary>Answer</summary>
By using non-contiguous blocks with a block table mapping. Blocks can be anywhere in physical memory - the mapping table handles indirection. Similar to OS virtual memory paging.
</details>

**Question 4**: What is a typical block size and why?
<details>
<summary>Answer</summary>
16 tokens. Trade-off:
- Too small (e.g., 1): Too much overhead from block table lookups
- Too large (e.g., 256): Waste in partial last blocks
16 balances memory efficiency and performance.
</details>

**Question 5**: Can PagedAttention be used for prefill?
<details>
<summary>Answer</summary>
Not optimal - prefill benefits from FlashAttention (processes all tokens in parallel). PagedAttention is best for decode where we need to access scattered KV cache. vLLM uses FlashAttention for prefill, PagedAttention for decode.
</details>

### Daily Reflection

**What went well?**
- [ ] Understood PagedAttention algorithm
- [ ] Implemented simplified version
- [ ] Calculated memory savings

**What was challenging?**
- [ ] CUDA kernel complexity
- [ ] Block table indirection concept
- [ ] Three-phase attention computation

**Questions for tomorrow**:
1. _______________________________________
2. _______________________________________
3. _______________________________________

---

## 🚀 Preview: Day 5

Tomorrow's focus:
- **Continuous Batching**: How vLLM batches requests dynamically
- **Scheduling Policies**: FCFS, priority, preemption
- **Performance Analysis**: Throughput vs latency trade-offs
- **Hands-On**: Analyze batching behavior with different workloads

**Preparation**:
- Review scheduler code from Day 3
- Think about how batching affects performance
- Read about batching in traditional serving systems

---

## 📚 Additional Resources

**Papers**:
- [ ] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM paper)
- [ ] [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer paper)

**Code Reading**:
- [ ] `vllm/attention/ops/paged_attn.py`
- [ ] `csrc/attention/attention_kernels.cu`
- [ ] `vllm/attention/backends/flash_attn.py`

**Visualizations**:
- [ ] Draw your own PagedAttention diagrams
- [ ] Create animations of block allocation
- [ ] Visualize memory savings for different scenarios

---

**Congratulations on mastering PagedAttention! 🎉**

**This is the KEY innovation that makes vLLM so efficient!**

---

*Completed: ___/___/___*
*Time spent: _____ hours*
*Confidence level (1-10): _____*
