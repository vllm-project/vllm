# Tutorial 04: Attention Layer Internals

## Learning Objectives

1. Understand the attention mechanism implementation in vLLM
2. Learn how PagedAttention integrates with KV cache blocks
3. Explore attention backends (FlashAttention, xformers, SDPA)
4. Master performance optimizations for attention computation
5. Debug attention-related issues and memory problems

## Overview

The Attention Layer is where vLLM's innovations truly shine. This tutorial explores how vLLM implements efficient attention computation using PagedAttention, enabling non-contiguous KV cache storage and dramatically improving memory utilization.

## Attention Fundamentals

### Standard Attention Mechanism

```
Given:
  Q (Query):  [batch, seq_len_q, d_k]
  K (Keys):   [batch, seq_len_k, d_k]
  V (Values): [batch, seq_len_v, d_v]

Compute:
  scores = Q @ K.T / sqrt(d_k)          # [batch, seq_len_q, seq_len_k]
  attn_weights = softmax(scores, dim=-1)
  output = attn_weights @ V              # [batch, seq_len_q, d_v]
```

### The Memory Challenge

For a 2048-token sequence in a 7B model:
- Each token's KV cache: 2 × 32 layers × 4096 hidden × 2 bytes (fp16) = **1MB per token**
- Full sequence: 2048 tokens × 1MB = **2GB for one request**

With traditional contiguous allocation, this doesn't scale!

## PagedAttention: The Key Innovation

### Problem with Contiguous KV Cache

```
Traditional Approach (Wasteful):
┌──────────────────────────────────────────┐
│ Request A: [KV KV KV KV ░░░░░░░░░░░░░░] │ ← Pre-allocated, wasted
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Request B: [KV KV ░░░░░░░░░░░░░░░░░░░░] │ ← More waste
└──────────────────────────────────────────┘

Problem: Must allocate max_seq_len upfront!
```

### PagedAttention Solution

```
Block-Based Storage:
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ Blk │ Blk │ Blk │ Blk │ Blk │ Blk │
│  0  │  1  │  2  │  3  │  4  │  5  │
└─────┴─────┴─────┴─────┴─────┴─────┘
  ▲     ▲           ▲     ▲
  │     │           │     │
  │     │           └─────┴─ Request B (blocks 3,4)
  │     │
  └─────┴─ Request A (blocks 0,1)

Benefits:
✓ No pre-allocation waste
✓ Dynamic growth
✓ Memory sharing via block pointers
```

### Block Table Indirection

Each request has a block table mapping logical positions to physical blocks:

```
Request A Block Table:
┌──────────┬──────────┬──────────┐
│ Logical  │ Physical │ Physical │
│ Position │ Block ID │ Address  │
├──────────┼──────────┼──────────┤
│  0-15    │    0     │ 0x1000   │
│  16-31   │    1     │ 0x2000   │
│  32-47   │    2     │ 0x3000   │
└──────────┴──────────┴──────────┘

During Attention:
  1. Map token position → block ID
  2. Fetch KV from physical block
  3. Compute attention
```

## Core Components

### 1. Attention Layer Base

**File**: `/vllm/model_executor/layers/attention_layer_base.py`

```python
class AttentionLayerBase(nn.Module):
    """
    Base class for all attention layers in vLLM.
    Provides interface for PagedAttention.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

        # For Grouped Query Attention (GQA)
        self.num_kv_heads = num_kv_heads or num_heads

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with PagedAttention.

        Args:
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
            value: [num_tokens, num_kv_heads * head_size]
            kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata including block tables
        """
        pass
```

### 2. Attention Metadata

Contains information needed for PagedAttention:

```python
@dataclass
class AttentionMetadata:
    """
    Metadata for attention computation.
    Critical for PagedAttention to map logical to physical positions.
    """

    # Prefill phase (processing prompt)
    prefill_metadata: PrefillMetadata | None = None

    # Decode phase (generating tokens)
    decode_metadata: DecodeMetadata | None = None

    # Block tables: [num_seqs, max_num_blocks_per_seq]
    # Maps logical block index → physical block ID
    block_tables: torch.Tensor | None = None

    # Sequence lengths
    seq_lens: torch.Tensor | None = None

    # Context lengths (for continuing generation)
    context_lens: torch.Tensor | None = None

    # Maximum sequence length in batch
    max_seq_len: int = 0

@dataclass
class DecodeMetadata:
    """Metadata for decode phase"""

    # Block tables for mapping
    block_tables: torch.Tensor

    # Sequence lengths
    seq_lens: torch.Tensor

    # Maximum context length
    max_decode_seq_len: int

    # Slot mapping: token position → cache slot
    slot_mapping: torch.Tensor
```

### 3. Attention Backends

**File**: `/vllm/attention/layer.py` (lines 1-150)

vLLM supports multiple attention backends:

```python
def get_attn_backend(
    num_heads: int,
    head_size: int,
    num_kv_heads: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    block_size: int,
) -> type[AttentionBackend]:
    """
    Select appropriate attention backend based on hardware and config.
    """

    backend = AttentionBackendEnum.FLASH_ATTN

    # Check for FlashAttention availability
    if current_platform.is_cuda():
        if not check_flash_attn_availability():
            # Fall back to xformers
            backend = AttentionBackendEnum.XFORMERS

    elif current_platform.is_rocm():
        backend = AttentionBackendEnum.ROCM_FLASH

    elif current_platform.is_tpu():
        backend = AttentionBackendEnum.PALLAS

    else:
        # CPU or other: use PyTorch SDPA
        backend = AttentionBackendEnum.TORCH_SDPA

    return backend.get_backend_class()
```

Backend hierarchy:

```
AttentionBackend (Abstract)
    │
    ├── FlashAttentionBackend (CUDA, optimized)
    ├── XFormersBackend (CUDA, fallback)
    ├── TorchSDPABackend (CPU/fallback)
    ├── ROCmFlashAttentionBackend (AMD GPUs)
    └── PallasAttentionBackend (TPUs)
```

## PagedAttention Implementation

### Kernel Overview

The PagedAttention kernel performs attention with non-contiguous KV cache:

```python
def paged_attention_v1(
    query: torch.Tensor,                    # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,                # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,              # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,             # [num_seqs, max_num_blocks_per_seq]
    context_lens: torch.Tensor,             # [num_seqs]
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    PagedAttention kernel V1.

    For each query token:
      1. Look up which blocks contain its KV cache
      2. Gather K, V from those blocks
      3. Compute attention
      4. Return output
    """

    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]

    # Output buffer
    output = torch.empty_like(query)

    # Launch kernel
    _paged_attention_kernel(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )

    return output
```

### Block Table Lookup Logic

```python
def get_kv_for_token(
    token_position: int,
    block_table: torch.Tensor,  # [max_num_blocks]
    key_cache: torch.Tensor,    # [num_blocks, block_size, ...]
    value_cache: torch.Tensor,  # [num_blocks, block_size, ...]
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get K, V for a specific token position using block table.
    """

    # Calculate which block and offset
    block_idx = token_position // block_size
    block_offset = token_position % block_size

    # Look up physical block ID
    physical_block_id = block_table[block_idx]

    # Fetch K, V from physical block
    k = key_cache[physical_block_id, block_offset]
    v = value_cache[physical_block_id, block_offset]

    return k, v
```

### Attention Computation with Paging

```
Logical Sequence (Request A):
[Token 0][Token 1][Token 2]...[Token 47]

Block Table:
┌────────┬────────┬────────┐
│ Blk 0  │ Blk 1  │ Blk 2  │ ← Logical blocks
│   ↓    │   ↓    │   ↓    │
│ Phy 5  │ Phy 2  │ Phy 7  │ ← Physical blocks (non-contiguous!)
└────────┴────────┴────────┘

Computing Attention for Token 20:
  1. Token 20 is in logical block 1 (20 // 16 = 1)
  2. Offset within block: 20 % 16 = 4
  3. Logical block 1 → Physical block 2
  4. Fetch K, V from: key_cache[2, 4, :, :]
  5. Compute attention score with query[20]
```

## Attention Backends Deep Dive

### FlashAttention Backend

Optimized CUDA kernel for efficient attention:

```python
class FlashAttentionBackend(AttentionBackend):
    """
    Backend using FlashAttention for optimized computation.
    Fuses attention operations and reduces memory bandwidth.
    """

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward with FlashAttention"""

        if attn_metadata.prefill_metadata is not None:
            # Prefill: standard FlashAttention
            return flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.prefill_metadata.seq_start_loc,
                cu_seqlens_k=attn_metadata.prefill_metadata.seq_start_loc,
                max_seqlen_q=attn_metadata.max_seq_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=attn_metadata.scale,
                causal=True,
            )
        else:
            # Decode: use PagedAttention
            return paged_attention_v1(
                query=query,
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                block_tables=attn_metadata.block_tables,
                context_lens=attn_metadata.context_lens,
                block_size=attn_metadata.block_size,
                max_context_len=attn_metadata.max_decode_seq_len,
            )
```

**Benefits**:
- Fused softmax and dropout
- Reduced HBM accesses
- Better GPU utilization
- 2-4x faster than naive implementation

### XFormers Backend

Fallback for systems without FlashAttention:

```python
class XFormersBackend(AttentionBackend):
    """Backend using xformers memory-efficient attention"""

    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_scale: float = 1.0,
    ) -> torch.Tensor:
        from xformers import ops as xops

        if attn_metadata.prefill_metadata is not None:
            # Use xformers memory-efficient attention
            return xops.memory_efficient_attention(
                query=query,
                key=key,
                value=value,
                attn_bias=xops.LowerTriangularMask(),
                scale=attn_metadata.scale,
            )
        else:
            # Decode: PagedAttention
            return paged_attention_v1(...)
```

## Optimizations

### 1. Multi-Query Attention (MQA)

Reduces KV cache size by sharing K, V across attention heads:

```
Standard Multi-Head Attention:
  Q: [batch, seq_len, num_heads, head_size]
  K: [batch, seq_len, num_heads, head_size]  ← Full K for each head
  V: [batch, seq_len, num_heads, head_size]  ← Full V for each head

Multi-Query Attention (MQA):
  Q: [batch, seq_len, num_heads, head_size]
  K: [batch, seq_len, 1, head_size]          ← Single K shared
  V: [batch, seq_len, 1, head_size]          ← Single V shared

KV Cache Reduction: num_heads × smaller!
```

Implementation:

```python
def multi_query_attention(
    query: torch.Tensor,  # [batch, seq_len, num_heads, head_size]
    key: torch.Tensor,    # [batch, seq_len, 1, head_size]
    value: torch.Tensor,  # [batch, seq_len, 1, head_size]
    scale: float,
) -> torch.Tensor:
    """Attention with shared K, V"""

    # Expand K, V to match Q heads (broadcasting)
    # No actual memory copy due to broadcasting
    key = key.expand(-1, -1, query.shape[2], -1)
    value = value.expand(-1, -1, query.shape[2], -1)

    # Standard attention
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output
```

### 2. Grouped Query Attention (GQA)

Middle ground between MHA and MQA:

```
Grouped Query Attention (GQA):
  Q: [batch, seq_len, num_heads, head_size]     (e.g., 32 heads)
  K: [batch, seq_len, num_kv_heads, head_size]  (e.g., 8 heads)
  V: [batch, seq_len, num_kv_heads, head_size]  (e.g., 8 heads)

Each KV head is shared by num_heads / num_kv_heads query heads.
Example: 32 query heads, 8 KV heads → 4 Q heads per KV head
```

Implementation:

```python
def grouped_query_attention(
    query: torch.Tensor,  # [batch, seq_len, num_heads, head_size]
    key: torch.Tensor,    # [batch, seq_len, num_kv_heads, head_size]
    value: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_size]
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention with grouped K, V"""

    batch_size, seq_len, _, head_size = query.shape

    # Reshape query: [batch, seq_len, num_kv_heads, group_size, head_size]
    group_size = num_heads // num_kv_heads
    query = query.reshape(batch_size, seq_len, num_kv_heads, group_size, head_size)

    # Add group dimension to K, V
    key = key.unsqueeze(3)  # [batch, seq_len, num_kv_heads, 1, head_size]
    value = value.unsqueeze(3)

    # Compute attention per group
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    # Reshape back
    output = output.reshape(batch_size, seq_len, num_heads, head_size)

    return output
```

### 3. Sliding Window Attention

Limits attention to recent context for long sequences:

```
Full Attention (O(n²)):
      0   1   2   3   4   5
  0 [ ✓   ✗   ✗   ✗   ✗   ✗ ]
  1 [ ✓   ✓   ✗   ✗   ✗   ✗ ]
  2 [ ✓   ✓   ✓   ✗   ✗   ✗ ]
  3 [ ✓   ✓   ✓   ✓   ✗   ✗ ]
  4 [ ✓   ✓   ✓   ✓   ✓   ✗ ]
  5 [ ✓   ✓   ✓   ✓   ✓   ✓ ]

Sliding Window (window=3, O(n*w)):
      0   1   2   3   4   5
  0 [ ✓   ✗   ✗   ✗   ✗   ✗ ]
  1 [ ✓   ✓   ✗   ✗   ✗   ✗ ]
  2 [ ✓   ✓   ✓   ✗   ✗   ✗ ]
  3 [ ✗   ✓   ✓   ✓   ✗   ✗ ]  ← Only last 3 tokens
  4 [ ✗   ✗   ✓   ✓   ✓   ✗ ]
  5 [ ✗   ✗   ✗   ✓   ✓   ✓ ]
```

Implementation:

```python
def sliding_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Attention with sliding window"""

    seq_len = query.shape[1]

    # Create sliding window mask
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device)

    for i in range(seq_len):
        # Attend to current token and previous window_size tokens
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = False  # False = attend

    # Compute attention with mask
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output
```

## Hands-On Exercises

### Exercise 1: Trace PagedAttention Lookup

**Objective**: Understand block table mapping

```python
def trace_paged_attention_lookup():
    """Trace how PagedAttention maps tokens to blocks"""

    # Setup
    block_size = 16
    block_table = torch.tensor([5, 2, 7, 1])  # Physical block IDs

    # Token positions to trace
    token_positions = [0, 15, 16, 20, 47]

    for pos in token_positions:
        # Calculate block and offset
        logical_block = pos // block_size
        offset = pos % block_size

        # Look up physical block
        if logical_block < len(block_table):
            physical_block = block_table[logical_block].item()

            print(f"Token {pos}:")
            print(f"  Logical block: {logical_block}")
            print(f"  Offset in block: {offset}")
            print(f"  Physical block: {physical_block}")
            print(f"  Cache location: key_cache[{physical_block}, {offset}, :, :]")
        else:
            print(f"Token {pos}: Out of range")
        print()
```

**Task**: Run this and verify your understanding of the mapping.

### Exercise 2: Compare Attention Backends

**Objective**: Benchmark different attention implementations

```python
import time
import torch

def benchmark_attention_backends():
    """Compare performance of different attention backends"""

    batch_size = 32
    seq_len = 512
    num_heads = 32
    head_size = 128

    # Create test inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_size, device='cuda')
    k = torch.randn(batch_size, seq_len, num_heads, head_size, device='cuda')
    v = torch.randn(batch_size, seq_len, num_heads, head_size, device='cuda')

    backends = {
        'PyTorch SDPA': lambda: F.scaled_dot_product_attention(q, k, v),
        'FlashAttention': lambda: flash_attn_func(q, k, v),
        'XFormers': lambda: xops.memory_efficient_attention(q, k, v),
    }

    for name, backend_fn in backends.items():
        # Warmup
        for _ in range(10):
            _ = backend_fn()
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(100):
            _ = backend_fn()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"{name}: {elapsed/100*1000:.2f} ms/iter")

        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        _ = backend_fn()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_mem:.1f} MB")
```

**Task**: Run on your GPU and compare results.

### Exercise 3: Visualize Attention Patterns

**Objective**: Understand what the model attends to

```python
import matplotlib.pyplot as plt

def visualize_attention_pattern(
    query: torch.Tensor,
    key: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
):
    """Visualize attention weights for a specific head"""

    # Compute attention scores
    scores = torch.matmul(query[layer_idx, head_idx], key[layer_idx, head_idx].T)
    scores = scores / (query.shape[-1] ** 0.5)

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Convert to numpy
    attn_weights_np = attn_weights.cpu().detach().numpy()

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_weights_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(f'Attention Pattern (Layer {layer_idx}, Head {head_idx})')
    plt.tight_layout()
    plt.show()
```

**Task**: Visualize attention for different heads and analyze patterns.

## Common Pitfalls and Solutions

### Pitfall 1: Block Table Out of Bounds

**Problem**: Accessing blocks beyond allocated range.

```python
# BAD: No bounds checking
physical_block = block_table[logical_block]  # ❌ May be out of bounds
```

**Solution**: Always validate block indices:

```python
# GOOD: Bounds checking
def safe_block_lookup(block_table, logical_block):
    if logical_block >= len(block_table):
        raise IndexError(f"Logical block {logical_block} out of range")

    physical_block = block_table[logical_block]

    if physical_block < 0 or physical_block >= num_physical_blocks:
        raise ValueError(f"Invalid physical block {physical_block}")

    return physical_block
```

### Pitfall 2: Incorrect KV Cache Shapes

**Problem**: Mismatched tensor shapes cause cryptic errors.

```python
# BAD: Wrong cache shape
kv_cache = torch.zeros(num_blocks, num_heads, block_size, head_size)  # ❌ Wrong order!
```

**Solution**: Follow vLLM's convention:

```python
# GOOD: Correct shape
kv_cache = torch.zeros(
    num_blocks,
    2,  # K and V
    block_size,
    num_kv_heads,
    head_size,
    dtype=torch.float16,
    device='cuda'
)

# Access as:
key_cache = kv_cache[:, 0]    # [num_blocks, block_size, num_kv_heads, head_size]
value_cache = kv_cache[:, 1]  # [num_blocks, block_size, num_kv_heads, head_size]
```

### Pitfall 3: Numerical Instability in Attention

**Problem**: Softmax overflow/underflow for long sequences.

```python
# BAD: Direct softmax on large values
scores = query @ key.T  # Can be very large
attn_weights = torch.exp(scores) / torch.sum(torch.exp(scores))  # ❌ Overflow!
```

**Solution**: Use stable softmax implementation:

```python
# GOOD: Numerically stable softmax
def safe_softmax(scores, dim=-1):
    # Subtract max for numerical stability
    scores_max = scores.max(dim=dim, keepdim=True)[0]
    scores_exp = torch.exp(scores - scores_max)
    return scores_exp / scores_exp.sum(dim=dim, keepdim=True)

# Or use PyTorch's built-in
attn_weights = F.softmax(scores, dim=-1)  # Already stable
```

## Debugging Attention Issues

### Enable Attention Logging

```python
class DebugAttentionLayer(nn.Module):
    """Attention layer with debugging"""

    def forward(self, query, key, value, kv_cache, attn_metadata):
        print(f"\n=== Attention Layer Debug ===")
        print(f"Query shape: {query.shape}")
        print(f"Key shape: {key.shape}")
        print(f"Value shape: {value.shape}")
        print(f"KV cache shape: {kv_cache.shape}")
        print(f"Block tables shape: {attn_metadata.block_tables.shape}")

        # Check for NaN/Inf
        if torch.isnan(query).any():
            print("⚠️  NaN detected in query!")
        if torch.isinf(query).any():
            print("⚠️  Inf detected in query!")

        # Continue with normal forward
        output = super().forward(query, key, value, kv_cache, attn_metadata)

        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        return output
```

### Validate Block Tables

```python
def validate_block_table(block_table, num_physical_blocks):
    """Validate that block table is correct"""

    # Check shape
    assert block_table.dim() == 2, "Block table should be 2D"

    # Check values
    assert (block_table >= 0).all(), "Negative block IDs found"
    assert (block_table < num_physical_blocks).all(), "Block ID out of range"

    print(f"✓ Block table valid")
    print(f"  Shape: {block_table.shape}")
    print(f"  Range: [0, {num_physical_blocks})")
```

## Performance Optimization

### 1. Choose Optimal Block Size

```python
def find_optimal_block_size(avg_seq_len, head_size):
    """Find block size that minimizes waste and fragmentation"""

    candidates = [8, 16, 32, 64, 128]

    best_size = 16  # Default
    min_waste = float('inf')

    for block_size in candidates:
        # Calculate waste
        num_blocks = (avg_seq_len + block_size - 1) // block_size
        allocated = num_blocks * block_size
        waste = allocated - avg_seq_len

        # Calculate fragmentation (prefer power of 2 for alignment)
        fragmentation_penalty = 0 if (block_size & (block_size - 1)) == 0 else 1000

        total_cost = waste + fragmentation_penalty

        if total_cost < min_waste:
            min_waste = total_cost
            best_size = block_size

    return best_size
```

### 2. Use Appropriate Attention Backend

```python
def select_attention_backend(seq_len, batch_size, device):
    """Select backend based on workload characteristics"""

    if device.type == 'cuda':
        if seq_len > 512 and batch_size > 1:
            # Long sequences: FlashAttention excels
            return 'flash_attn'
        else:
            # Short sequences: xformers may be faster
            return 'xformers'
    else:
        # CPU: use SDPA
        return 'torch_sdpa'
```

### 3. Fuse Operations

```python
def fused_attention_update(
    query, key, value,
    kv_cache, block_table,
    new_tokens,
):
    """
    Fuse attention computation with KV cache update.
    Reduces memory bandwidth.
    """

    # Instead of:
    #   1. Update KV cache
    #   2. Run attention
    # Do both in single kernel:

    return fused_paged_attention_with_update_kernel(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        block_table=block_table,
        new_tokens=new_tokens,
    )
```

## Advanced Topics

### Prefix-Aware Attention

Optimize for common prefixes:

```python
def prefix_aware_attention(
    query, key, value,
    kv_cache, block_table,
    prefix_blocks,  # Shared prefix blocks
):
    """
    Attention that leverages shared prefix blocks.
    Compute attention over prefix once and reuse.
    """

    # Compute attention over shared prefix (cache this!)
    prefix_attn = compute_prefix_attention(query, prefix_blocks)

    # Compute attention over unique suffix
    suffix_attn = compute_suffix_attention(query, key, value, block_table)

    # Combine
    output = prefix_attn + suffix_attn

    return output
```

### Multi-Modal Attention

Extend PagedAttention to multi-modal inputs:

```python
def multimodal_paged_attention(
    query,
    text_kv_cache,
    image_kv_cache,
    text_block_table,
    image_block_table,
):
    """
    PagedAttention for multi-modal (text + image) inputs.
    """

    # Attention over text blocks
    text_output = paged_attention_v1(
        query,
        text_kv_cache[0],
        text_kv_cache[1],
        text_block_table,
        ...
    )

    # Attention over image blocks
    image_output = paged_attention_v1(
        query,
        image_kv_cache[0],
        image_kv_cache[1],
        image_block_table,
        ...
    )

    # Combine outputs
    output = text_output + image_output

    return output
```

## References

### Source Code Files

- **Attention Layer**: `/vllm/attention/layer.py`
- **Attention Backends**: `/vllm/attention/backends/`
- **PagedAttention Kernels**: `/vllm/attention/ops/`
- **Attention Metadata**: `/vllm/attention/backends/abstract.py`

### Key Papers

- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM paper)
- "GQA: Training Generalized Multi-Query Transformer" (Ainslie et al., 2023)

### Configuration

```python
@dataclass
class AttentionConfig:
    num_heads: int
    head_size: int
    num_kv_heads: int | None = None  # For GQA
    sliding_window: int | None = None
    attention_backend: str = "flash_attn"
```

## Summary

In this tutorial, you learned:

- How attention mechanisms work in vLLM
- PagedAttention's block-based approach to KV cache management
- Different attention backends and their trade-offs
- Optimizations like MQA, GQA, and sliding window attention
- Debugging techniques and common pitfalls
- Performance optimization strategies

Attention is the computational bottleneck in LLM serving. vLLM's PagedAttention innovation enables efficient memory usage while maintaining high performance.

## Next Steps

- **Tutorial 05**: Sampler Implementation - Token generation strategies
- **Tutorial 06**: KV Cache Management - Complete cache lifecycle
- **Module 3**: Understanding PagedAttention fundamentals
