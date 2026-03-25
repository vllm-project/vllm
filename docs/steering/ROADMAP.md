# Steering Implementation Roadmap

This document describes the phased implementation plan for activation steering in vLLM, starting with Gemma 3 and pre-computed steering vectors (e.g., from SAE decoder columns).

## Design Decisions

### Registered Buffer Pattern (Unconditional Addition)

All approaches in this roadmap use **pre-allocated registered buffers** with unconditional addition. This is driven by two hard constraints:

1. **`@support_torch_compile`**: Model classes like `Gemma3Model` are decorated with `@support_torch_compile`, which traces the entire forward pass into a computation graph. Data-dependent branching (`if steering is not None`) causes graph breaks or dead code elimination. The forward path must be identical whether steering is active or not.

2. **CUDA graph compatibility**: CUDA graphs capture kernel launches with fixed tensor addresses. Registered buffers have stable addresses that persist across graph replays. Values can be updated between replays via `.copy_()`.

The pattern:

```python
# Buffer initialized to zeros - addition is a no-op when not steering
self.register_buffer('steering_post_mlp', torch.zeros(...), persistent=False)

# In forward - always executes, zero buffer = no effect
hidden_states = hidden_states + self.steering_post_mlp
```

This is the same pattern vLLM uses for LoRA weight buffers with CUDA graphs.

### Residual Stream Targeting

Steering vectors are added to the raw residual stream, not before normalization layers. In Gemma 3, the decoder layer flow is:

```
hidden_states, residual = pre_feedforward_layernorm(hidden_states, residual)
hidden_states = mlp(hidden_states)
hidden_states = post_feedforward_layernorm(hidden_states)  # Gemma 3-specific
return hidden_states, residual
```

The next layer's `input_layernorm(hidden_states, residual)` fuses the residual add: `residual = hidden_states + residual`. Adding the steering vector to `hidden_states` after `post_feedforward_layernorm` places it directly into the residual stream without passing through any additional normalization. This matches where Gemma 3 SAEs are trained (on the residual stream).

### Steering Vectors, Not SAE Encode/Decode

The implementation passes pre-computed steering vectors (e.g., SAE decoder columns scaled by desired feature activation) rather than running full SAE encode/decode in the inference loop. Vectors are computed offline and provided at request time.

---

## Phase 1: MVP

**Goal**: Validate that steering works end-to-end with Gemma 3 using broadcast steering vectors.

### Scope

- **Model**: Gemma 3 only
- **Intervention point**: Post-MLP only
- **Phase**: Decode only (no prefix caching interaction)
- **Granularity**: Broadcast — same steering vector applied to all requests in batch
- **Buffer shape**: `[1, hidden_size]` per steered layer (~7 KB each, negligible memory)

### Implementation

**`Gemma3DecoderLayer` changes** (`vllm/model_executor/models/gemma3.py`):

```python
class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        # ... existing init ...

        self.register_buffer(
            'steering_post_mlp',
            torch.zeros(1, config.hidden_size),
            persistent=False,
      )

    def forward(self, positions, hidden_states, residual, **kwargs):
        # ... existing attention + MLP code ...
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)

        hidden_states = hidden_states + self.steering_post_mlp

        return hidden_states, residual
```

**Model runner steering interface** (`vllm/v1/worker/gpu/model_runner.py`):

```python
# Update steering buffers before forward pass
def _update_steering(self, steering_config: dict[int, tuple[torch.Tensor, float]]):
    for module in self.model.modules():
        if hasattr(module, 'steering_post_mlp'):
            if module.layer_idx in steering_config:
                vec, scale = steering_config[module.layer_idx]
                module.steering_post_mlp.copy_((scale * vec).unsqueeze(0))
            else:
                module.steering_post_mlp.zero_()
```

### Validation

- Load Gemma 3 pretrained SAE decoder columns as steering vectors
- Apply at known-effective layers and verify behavioral shift in generation
- Confirm no regression in throughput with zero-vector (no-op) steering
- Confirm CUDA graph capture and replay works with steering buffers

### Not In Scope

- API exposure (use direct model runner access for testing)
- Per-request steering
- Prefill steering
- Pre/post-attention intervention points
- Multi-model support

---

## Phase 2: Per-Request Steering

**Goal**: Different requests in a batch receive different steering vectors and scaling factors.

### Scope Changes

- **Buffer shape**: `[1, hidden_size]` → `[max_num_tokens, hidden_size]` per steered layer
- **Memory**: ~56 MB per steered layer (Gemma 3 27B, `max_num_batched_tokens=8192`, bf16)
- **New bookkeeping**: Per-request steering config tracked through `InputBatch`

### Key Implementation Points

**Decode batch layout**: During decode, `hidden_states[i]` = request `i` (one token per request). The steering buffer row `i` holds request `i`'s vector (or zeros). No `query_start_loc` indexing needed.

**Mixed prefill+decode batches**: In continuous batching, a step can contain both prefill tokens (new request) and decode tokens (ongoing requests). For decode-only steering, prefill tokens get zero rows in the buffer. This works naturally — just zero the buffer and scatter only decode requests' vectors.

**Continuous batching lifecycle**:
- Request joins → scatter its steering vector into its batch index row
- Request finishes → row gets overwritten by next request (or zeroed)
- No interference with continuous batching; steering is per-token independent

**CUDA graph padding**: Buffer is `[max_num_tokens, hidden_size]`. Padded entries beyond `num_tokens_padded` are zeros. The CUDA graph captures the full addition at the padded size.

### Data Flow

Per-request steering configs need to flow through the full request pipeline:

```
API Request (steering_vectors, steering_scales)
  → SamplingParams or SteeringParams
    → EngineCoreRequest
      → SchedulerOutput / NewRequestData
        → InputBatch (per-request state arrays)
          → Model Runner (scatter into buffers before forward)
```

This follows the same pattern as LoRA adapter tracking in `InputBatch`.

### API Exposure

Expose steering via the OpenAI-compatible API as vLLM extension fields:

```python
# Client usage
response = client.chat.completions.create(
    model="google/gemma-3-27b-it",
    messages=[...],
    extra_body={
        "steering_vectors": {20: vector.tolist()},
        "steering_scales": {20: 1.5},
    }
)
```

---

## Phase 3: Additional Intervention Points

**Goal**: Support pre-attention and post-attention steering in addition to post-MLP.

### Scope Changes

- Add `steering_pre_attn` and `steering_post_attn` registered buffers
- Same unconditional addition pattern at each intervention point
- Per-request capable (reuses Phase 2 scatter infrastructure)

### Gemma 3 Intervention Points

```
input_layernorm(hidden_states, residual)
  ↓
[steering_pre_attn]          ← before attention
  ↓
self_attn(hidden_states)
  ↓
post_attention_layernorm(hidden_states)
  ↓
[steering_post_attn]         ← after attention, before MLP
  ↓
pre_feedforward_layernorm(hidden_states, residual)
  ↓
mlp(hidden_states)
  ↓
post_feedforward_layernorm(hidden_states)
  ↓
[steering_post_mlp]          ← after MLP (Phase 1)
```

### API Extension

```python
extra_body={
    "steering_vectors": {20: vector.tolist()},
    "steering_scales": {20: 1.5},
    "steering_position": "post_mlp",  # or "pre_attn", "post_attn"
}
```

---

## Phase 4: Prefill Steering

**Goal**: Apply steering during the prefill phase, not just decode.

### Key Challenge: Prefix Caching

Steering modifies the residual stream, which changes all downstream KV cache entries. A prefix cached without steering is invalid when steering is applied (and vice versa). Options:

1. **Disable prefix caching when steering is active** — simplest, some performance cost
2. **Include steering config in cache key** — correct but expensive (different steering = different cache entries, low hit rate)
3. **Only steer non-cached tokens** — inconsistent behavior across the sequence, not recommended

Option 1 is the likely initial approach. Option 2 may be viable if steering configs are commonly reused across requests.

### Per-Request Prefill Complexity

During prefill, requests have variable token counts. The steering buffer must expand per-request vectors to per-token using `query_start_loc`:

```python
# For each request i:
#   start = query_start_loc[i]
#   end = query_start_loc[i + 1]
#   steering_buffer[start:end] = request_i_steering_vector
```

This scatter is more complex than decode (where it's 1:1) but follows established patterns in vLLM (see LoRA kernel metadata in `vllm/lora/ops/triton_ops/`).

---

## Phase 5: Multi-Model Support

**Goal**: Extend steering to all major model families.

### Change Per Model

The same ~4-line pattern applied to each model's `DecoderLayer`:

1. `self.layer_idx = extract_layer_index(prefix)` in `__init__`
2. `self.register_buffer('steering_post_mlp', zeros(1, hidden_size), persistent=False)` in `__init__`
3. `hidden_states = hidden_states + self.steering_post_mlp` in `forward`

Most models have their own `DecoderLayer` (60+ implementations). The change is nearly identical across all of them — a tractable repetitive refactor.

### Model-Specific Considerations

- **Llama-style models**: No post-feedforward layernorm. Steering goes after `mlp()` directly.
- **Gemma 2/3**: Extra pre/post feedforward layernorms. Steering after `post_feedforward_layernorm()`.
- **Mixture-of-experts** (Mixtral, DeepSeek, Qwen-MoE): MLP is a MoE layer. Steering applies to the MoE output, same pattern.
- **Mamba/SSM models**: Different architecture, may need separate analysis for where steering makes sense.

---

## Memory Budget Summary

| Phase | Buffer Shape | Per Layer (Gemma 3 27B, bf16) | 5 Steered Layers |
|-------|-------------|-------------------------------|-------------------|
| 1 (broadcast) | `[1, 3584]` | 7 KB | 35 KB |
| 2 (per-request) | `[max_tokens, 3584]` | ~56 MB (`max=8192`) | ~280 MB |
| 3 (3 intervention points) | 3x Phase 2 | ~168 MB | ~840 MB |

Phase 3 memory is significant. Consider allocating buffers only for intervention points actually in use, gated by a startup configuration flag (not runtime branching in the forward pass).
