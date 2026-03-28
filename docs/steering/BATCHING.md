# Batching in vLLM

This document explains how vLLM handles batching of multiple requests, with a focus on the V1 engine. Understanding these concepts is essential for implementing per-request features like steering vectors, custom samplers, or other request-specific behavior.

## Overview

vLLM uses **continuous batching** where sequences at different stages (prefill vs decode) are processed together. The key challenge is tracking which tokens belong to which request when everything is flattened into contiguous tensors for GPU efficiency.

## Core Data Structures

### `query_start_loc` - Token Boundaries

The most important tensor for per-request operations. It's a cumulative sum that defines where each request's tokens start and end in the flattened batch.

```python
# Shape: (num_reqs + 1,)
# Tokens for request i: hidden_states[query_start_loc[i]:query_start_loc[i+1]]

# Example with 3 requests having 4, 2, and 3 tokens:
query_start_loc = [0, 4, 6, 9]
#                  ^  ^  ^  ^
#                  |  |  |  └─ end of request 2 (total tokens)
#                  |  |  └─ start of request 2 / end of request 1
#                  |  └─ start of request 1 / end of request 0
#                  └─ start of request 0
```

**Location:** Built in `vllm/v1/worker/gpu/model_runner.py:645-654`

### `idx_mapping` - Request State Lookup

Maps batch index to request state index for accessing per-request persistent state.

```python
# Shape: (num_reqs,)
# Usage: req_state_idx = idx_mapping[batch_idx]
```

**Location:** Built in `vllm/v1/worker/gpu/model_runner.py:607-609`

### `CommonAttentionMetadata`

Container for all attention-related metadata, including the above tensors.

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor      # (num_reqs + 1,) cumulative token positions
    query_start_loc_cpu: torch.Tensor  # CPU copy for sync operations
    seq_lens: torch.Tensor             # (num_reqs,) total tokens per request
    num_reqs: int                      # Number of active requests
    num_actual_tokens: int             # Total tokens in batch
    max_query_len: int                 # Longest query in batch
    max_seq_len: int                   # Longest context length
    block_table_tensor: torch.Tensor   # KV cache block mappings
    slot_mapping: torch.Tensor         # Token to cache slot mappings
```

**Location:** `vllm/v1/attention/backend.py:286-418`

### `InputBatch`

Manages the batch assembly process, mapping request IDs to indices and unpacking per-request configuration into arrays.

```python
@dataclass
class InputBatch:
    req_ids: list[str]              # batch_idx -> request ID string
    num_reqs: int                   # Active request count
    idx_mapping: torch.Tensor       # batch_idx -> req_state_idx (GPU)
    idx_mapping_np: np.ndarray      # batch_idx -> req_state_idx (CPU)
    num_scheduled_tokens: np.ndarray  # Tokens per request this step
    query_start_loc: torch.Tensor   # Cumulative token positions
    seq_lens: torch.Tensor          # Total sequence length per request
```

**Location:** `vllm/v1/worker/gpu/input_batch.py:35-145`

## Token-to-Request Mapping

### In Python

```python
def process_per_request(hidden_states, attn_metadata):
    query_start_loc = attn_metadata.query_start_loc

    for req_idx in range(attn_metadata.num_reqs):
        start = query_start_loc[req_idx].item()
        end = query_start_loc[req_idx + 1].item()

        # This request's tokens
        request_hidden = hidden_states[start:end]

        # Do something per-request
        process(request_hidden)
```

### In Triton Kernels

```python
@triton.jit
def per_request_kernel(
    data_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one request
    req_idx = tl.program_id(0)
    if req_idx >= num_reqs:
        return

    # Get token boundaries
    start = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_tokens = end - start

    # Get request state index if needed
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    # Process tokens for this request
    for i in tl.range(0, num_tokens, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_tokens
        token_idx = start + offs

        vals = tl.load(data_ptr + token_idx, mask=mask)
        # ... process ...
        tl.store(data_ptr + token_idx, vals, mask=mask)
```

**Reference:** `vllm/v1/worker/gpu/input_batch.py:207-262`

## GPU Optimization Patterns

### Pattern 1: Cumulative Index Arrays

Use cumulative sums (like `query_start_loc`) instead of per-request iteration. This allows O(1) lookup of any request's token range.

```python
# Bad: Store lengths and compute offsets
lengths = [4, 2, 3]
offset = sum(lengths[:i])  # O(n) per lookup

# Good: Store cumulative sums
start_loc = [0, 4, 6, 9]
start, end = start_loc[i], start_loc[i+1]  # O(1) lookup
```

### Pattern 2: Sorted Index Grouping (LoRA Pattern)

When applying different operations to different requests (e.g., different steering vectors), sort tokens by operation type for coalesced processing.

```python
@dataclass
class OperationKernelMeta:
    # Which operation each token uses (-1 = none)
    token_op_mapping: torch.Tensor           # [num_tokens]

    # Tokens sorted by operation ID
    token_indices_sorted: torch.Tensor       # [num_affected_tokens]

    # Cumulative count per operation (like query_start_loc)
    op_token_start_loc: torch.Tensor         # [num_ops + 1]

    # Active operation IDs
    active_op_ids: torch.Tensor              # [num_active_ops]
```

**Reference:** `vllm/lora/ops/triton_ops/lora_kernel_metadata.py`

### Pattern 3: Early Exit for Inactive Requests

```python
@triton.jit
def kernel_with_early_exit(active_flags_ptr, ...):
    req_idx = tl.program_id(0)

    # Check if this request needs processing
    is_active = tl.load(active_flags_ptr + req_idx)
    if not is_active:
        return  # Skip entirely

    # ... rest of kernel ...
```

### Pattern 4: Vectorized Hidden Dimension

Process the hidden dimension in blocks for memory coalescing:

```python
@triton.jit
def process_hidden(
    hidden_ptr,
    token_idx,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_base = token_idx * HIDDEN_DIM

    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < HIDDEN_DIM

        vals = tl.load(hidden_ptr + hidden_base + offs, mask=mask)
        # ... process ...
        tl.store(hidden_ptr + hidden_base + offs, vals, mask=mask)
```

## Per-Request Metadata Flow

### Request Lifecycle

```
API Request (sampling_params, lora_request, etc.)
    │
    ▼
EngineCoreRequest (vllm/v1/engine/__init__.py:67)
    │
    ▼
InputBatch.add_request() - unpacks to per-request arrays
    │
    ▼
InputBatch.make_*_metadata() - builds GPU tensors
    │
    ▼
Model Forward Pass - accesses via attn_metadata
```

### Adding Custom Per-Request Data

1. **Use `SamplingParams.extra_args`** for simple cases:
```python
SamplingParams(
    temperature=0.7,
    extra_args={"my_feature_id": "abc123", "my_scale": 0.5}
)
```

2. **Create a dedicated request type** for complex cases (like LoRA):
```python
@msgspec.Struct
class MyFeatureRequest:
    feature_id: str
    feature_config: dict
```

3. **Unpack in `InputBatch`** to per-request arrays:
```python
# In InputBatch.add_request()
if my_feature := request.sampling_params.extra_args.get("my_feature_id"):
    self.my_feature_mapping[req_index] = my_feature
```

4. **Build metadata struct** for GPU access:
```python
def make_my_feature_metadata(self) -> MyFeatureMetadata:
    return MyFeatureMetadata(
        feature_ids=self.my_feature_mapping[:self.num_reqs].clone(),
        # ... other tensors
    )
```

## Memory Layout Patterns

### Contiguous Batch Layout

For per-request tensors that don't vary in size:

```
Memory: [Req0_Data | Req1_Data | Req2_Data | ...]
Access: ptr + req_idx * stride + local_offset
Shape:  [num_reqs, feature_dim]
```

### Flattened Token Layout

For per-token tensors with variable tokens per request:

```
Memory: [Req0_Tok0, Req0_Tok1, ..., Req1_Tok0, Req1_Tok1, ...]
Access: ptr + query_start_loc[req_idx] + token_offset
Shape:  [total_tokens, hidden_dim]
```

### Sorted/Grouped Layout

For efficient processing of token subsets:

```
Memory: [tokens_for_op0..., tokens_for_op1..., tokens_for_op2...]
Index:  token_indices_sorted[op_start_loc[op_id] + local_idx]
```

## Example: Per-Request Steering (Actual Implementation)

The per-request steering implementation uses a **request-indexed gather** pattern rather than a Triton kernel. This approach is simpler, CUDA graph compatible, and leverages the torch.compile splitting point mechanism.

### Per-Layer Buffers

```python
# On each Gemma3DecoderLayer:
steering_table: torch.Tensor   # (max_configs + 2, hidden_size)
steering_index: torch.Tensor   # (max_tokens,) — shared across all layers
```

### Table Layout

```
Row 0:  [0, 0, 0, ..., 0]              ← prefill / no steering
Row 1:  [g₁, g₂, ..., gₕ]             ← global vector only
Row 2:  [g₁+a₁, g₂+a₂, ..., gₕ+aₕ]   ← global + per-request config A
Row 3:  [g₁+b₁, g₂+b₂, ..., gₕ+bₕ]   ← global + per-request config B
```

### Index Mapping

```python
# Model runner builds this before each forward pass:
# Token:  [decode₁, decode₂, prefill₁, prefill₂, decode₃]
# Index:  [   2,       1,        0,        0,       3    ]
```

### Custom Op (splitting point)

```python
# In Gemma3DecoderLayer.forward():
residual = torch.ops.vllm.apply_steering(
    residual,              # (num_tokens, hidden_size)
    self.steering_table,   # (max_configs + 2, hidden_size)
    self.steering_index,   # (max_tokens,)
)
# = residual + steering_table[steering_index[:N]]
```

The custom op is registered as a torch.compile splitting point, so the Python implementation runs at runtime between compiled graph segments. In-place buffer updates are visible across CUDA graph replays.

### How Requests Map to Table Rows

The `SteeringManager` assigns table rows to distinct steering configs (identified by hash). Multiple requests sharing identical vectors share a row (reference counted). The `InputBatch` tracks each request's `steering_config_hash`, and the model runner builds the `steering_index` by looking up each request's hash → row mapping.

See [STEERING.md](STEERING.md) for the full architecture.

## Key Files Reference

| File | Purpose |
|------|---------|
| `vllm/v1/attention/backend.py` | `CommonAttentionMetadata` definition |
| `vllm/v1/worker/gpu/input_batch.py` | Batch assembly, per-request array management |
| `vllm/v1/worker/gpu/model_runner.py` | `query_start_loc` and `idx_mapping` construction |
| `vllm/v1/sample/metadata.py` | `SamplingMetadata` - example of per-request GPU tensors |
| `vllm/lora/ops/triton_ops/lora_kernel_metadata.py` | Sorted index grouping pattern |
| `vllm/lora/ops/triton_ops/lora_expand_op.py` | LoRA kernel with per-request adapters |
| `vllm/forward_context.py` | `ForwardContext` for passing data to layers |

## See Also

- [STEERING.md](STEERING.md) - Implementing activation steering
- [EXTRACTION.md](EXTRACTION.md) - Extracting activations from the model
