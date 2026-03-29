# Steering Implementation Roadmap

This document describes the phased implementation plan for activation steering in vLLM, starting with Gemma 3 and pre-computed steering vectors (e.g., from SAE decoder columns).

## Design Decisions

### Registered Buffer Pattern (Unconditional Addition)

All phases use **pre-allocated registered buffers** with unconditional addition. This is driven by two hard constraints:

1. **`@support_torch_compile`**: Model classes like `Gemma3Model` are decorated with `@support_torch_compile`, which traces the entire forward pass into a computation graph. Data-dependent branching (`if steering is not None`) causes graph breaks or dead code elimination. The forward path must be identical whether steering is active or not.

2. **CUDA graph compatibility**: CUDA graphs capture kernel launches with fixed tensor addresses. Registered buffers have stable addresses that persist across graph replays. Values can be updated between replays via in-place operations.

### Residual Stream Targeting

Steering vectors are added to the raw residual stream, not before normalization layers. The next layer's `input_layernorm(hidden_states, residual)` fuses the residual add, placing the steering vector directly into the residual stream. This matches where Gemma 3 SAEs are trained.

### Steering Vectors, Not SAE Encode/Decode

The implementation passes pre-computed steering vectors (e.g., SAE decoder columns scaled by desired feature activation) rather than running full SAE encode/decode in the inference loop. Vectors are computed offline and provided at request time.

---

## Phase 1: MVP — DONE

**Goal**: Validate that steering works end-to-end with Gemma 3 using broadcast steering vectors.

### What was built

- Global steering via `steering_vector` buffer `(1, hidden_size)` per layer
- Custom op `apply_steering` registered as torch.compile splitting point
- REST API (`POST /v1/steering/set`, `/clear`, `GET /v1/steering`) with two-phase validate-then-apply
- Decode-only mask buffer to prevent steering during prefill

See [MVP_IMPLEMENTATION.md](MVP_IMPLEMENTATION.md) for detailed implementation notes.

---

## Phase 2: Per-Request Steering — DONE

**Goal**: Different requests in a batch receive different steering vectors.

### What was built

Replaced the broadcast pattern with a **request-indexed gather** approach:

- **Steering table** `(max_configs + 2, hidden_size)` per layer — row 0=zeros, row 1=global, rows 2+=global+per_request
- **Steering index** `(max_tokens,)` shared across all layers — maps each token to its table row
- **SteeringManager** — config registration with reference counting, table population, global vector caching
- **InputBatch tracking** — `steering_config_hash` per request for efficient dedup
- **Scheduler admission control** — `max_steering_configs` capacity check (LoRA pattern)
- **SamplingParams.steering_vectors** — per-request field, also passable via OpenAI `extra_body`
- **Additive combination** — global + per_request vectors summed in table rows
- **`--enable-steering` / `--max-steering-configs`** CLI flags

### Memory

| Config | Per Layer (Gemma 3 4B, bf16) | 26 Layers |
|--------|------------------------------|-----------|
| `max_steering_configs=4` (default) | 6 rows × 3072 × 2B = ~36 KB | ~940 KB |
| `max_steering_configs=16` | 18 rows × 3072 × 2B = ~108 KB | ~2.8 MB |

The request-indexed gather approach is dramatically cheaper than the per-token buffer approach originally considered (~56 MB/layer for max_tokens=8192).

---

## Phase 3: Additional Intervention Points — DONE

**Goal**: Support pre-attention, post-attention, and post-MLP-post-layernorm steering in addition to the default post-MLP-pre-layernorm.

### What was built

Four hook points on the residual stream, all unconditionally allocated:

- `pre_attn` — after `input_layernorm`, before `self_attn`
- `post_attn` — after `post_attention_layernorm`, before `pre_feedforward_layernorm`
- `post_mlp_pre_ln` — after `mlp`, before `post_feedforward_layernorm`
- `post_mlp_post_ln` — after `post_feedforward_layernorm`

Key design decisions:

- **Always-allocated buffers**: All 4 hook point buffers are registered on every decoder layer. Memory cost is trivial (~3.6 MB for 26 layers at `max_steering_configs=4`). Zero rows make unused hook points a no-op.
- **No graph partitions**: The `apply_steering` custom op is opaque to the torch.compile tracer (preventing constant-folding) but is NOT a splitting op. Steering adds zero graph partitions regardless of hook point count.
- **Per-hook-point buffers**: Each hook point gets its own `steering_table_<hook>` and `steering_vector_<hook>` buffer. The shared `steering_index` is reused across all hook points (token→row mapping is the same).
- **API**: `SamplingParams.steering_vectors: dict[str, dict[int, list[float]]]` — hook point name → layer → vector. Same format for `SetSteeringRequest.vectors` (global HTTP API).

### Gemma 3 Intervention Points

```
input_layernorm(hidden_states, residual)     # fused add + norm → updates residual
  ↓
[pre_attn]                                   ← steer residual before attention
  ↓
self_attn(hidden_states)
post_attention_layernorm(hidden_states)       # hidden_states only
  ↓
[post_attn]                                  ← steer residual after attention
  ↓
pre_feedforward_layernorm(hidden_states, residual)  # fused add + norm → updates residual
mlp(hidden_states)
  ↓
[post_mlp_pre_ln]                            ← default (backward compat)
  ↓
post_feedforward_layernorm(hidden_states)     # hidden_states only
  ↓
[post_mlp_post_ln]                           ← steer residual after post_ff_ln
```

### Memory

All 4 hook points are always allocated:

| Config | Per Layer (Gemma 3 4B, bf16) | 26 Layers |
|--------|------------------------------|-----------|
| `max_steering_configs=4` (default) | 4 × 6 rows × 3072 × 2B = ~144 KB | ~3.6 MB |
| `max_steering_configs=16` | 4 × 18 rows × 3072 × 2B = ~432 KB | ~11.2 MB |

---

## Phase 4: Prefill Steering

**Goal**: Apply steering during the prefill phase, not just decode.

**Status**: Not started.

### Key Challenge: Prefix Caching

Steering modifies the residual stream, which changes all downstream KV cache entries. A prefix cached without steering is invalid when steering is applied (and vice versa). Options:

1. **Disable prefix caching when steering is active** — simplest, some performance cost
2. **Include steering config in cache key** — correct but expensive (different steering = different cache entries, low hit rate)
3. **Only steer non-cached tokens** — inconsistent behavior across the sequence, not recommended

### Per-Request Prefill Complexity

During prefill, requests have variable token counts. The `steering_index` already handles this — prefill tokens map to row 0 (zeros). To enable prefill steering, prefill tokens would instead map to the request's assigned row, same as their decode tokens.

---

## Phase 5: Multi-Model Support

**Goal**: Extend steering to all major model families.

**Status**: Not started.

### Change Per Model

Each model's `DecoderLayer` needs:

1. `self.layer_idx = extract_layer_index(prefix)` in `__init__`
2. Register `steering_vector`, `steering_table`, and `steering_index` buffers
3. Call `torch.ops.vllm.apply_steering(residual, self.steering_table, self.steering_index)` in `forward`
4. Share `steering_index` across layers in the model class

### Model-Specific Considerations

- **Llama-style models**: No post-feedforward layernorm. Steering goes after `mlp()` directly.
- **Gemma 2/3**: Extra pre/post feedforward layernorms. Steering after `post_feedforward_layernorm()`.
- **Mixture-of-experts** (Mixtral, DeepSeek, Qwen-MoE): MLP is a MoE layer. Steering applies to the MoE output, same pattern.
- **Mamba/SSM models**: Different architecture, may need separate analysis for where steering makes sense.

---

## Phase 6: Named Steering Configs

**Goal**: Pre-register steering configs with names (like LoRA adapters) for efficient reuse.

**Status**: Not started. The SteeringManager's hash-based deduplication and row assignment already lays the groundwork for this — adding a `name → hash` registry would be straightforward.
