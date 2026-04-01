# Live Activation Steering

This document describes the activation steering implementation in this vLLM fork.

## Overview

Activation steering modifies model hidden states at runtime by adding pre-computed vectors to the residual stream during decode. This enables:
- **Representation engineering**: Adding "concept vectors" to shift model behavior
- **SAE feature steering**: Amplifying or suppressing specific SAE features via decoder columns
- **Safety interventions**: Modifying activations to reduce harmful outputs
- **Behavioral control**: Adjusting style, tone, or content properties

Steering supports a three-tier additive composition model:
- **Base vectors** (`steering_vectors`): Applied to both prefill and decode phases
- **Prefill-specific** (`prefill_steering_vectors`): Added to base during prefill only
- **Decode-specific** (`decode_steering_vectors`): Added to base during decode only

Both global (server-wide, via HTTP API) and per-request (via `SamplingParams`) steering use this three-tier model. All tiers are additive:
```
effective_prefill = global_base + global_prefill + request_base + request_prefill
effective_decode  = global_base + global_decode  + request_base + request_decode
```

Each vector entry supports co-located scales: either a bare `list[float]` (scale=1.0) or `{"vector": [...], "scale": float}`.

## Architecture

The implementation uses a **request-indexed gather** pattern with five components.
Each decoder layer can have multiple **hook points** (e.g. `pre_attn`, `post_attn`,
`post_mlp_pre_ln`, `post_mlp_post_ln`), each with its own steering table and global
vector buffer.

```
    API Layer                    Config                    Scheduler
┌──────────────────┐    ┌───────────────────┐    ┌──────────────────────────┐
│ POST /v1/steering│    │ SteeringConfig     │    │ Admission control        │
│ set / clear / GET│    │ --enable-steering  │    │ scheduled_steering_      │
│                  │    │ --max-steering-    │    │   configs: set[int]      │
│ SamplingParams.  │    │   configs 4        │    │ skipped_waiting queue    │
│ steering_vectors │    └───────────────────┘    └──────────────────────────┘
└──────────────────┘              │                          │
         │                        ▼                          ▼
         │              ┌───────────────────┐    ┌──────────────────────────┐
         └─────────────>│ SteeringManager    │    │ InputBatch               │
                        │ register/release   │    │ request_steering_config_ │
                        │ populate_tables()  │    │   hash[req_idx]          │
                        │ global_vectors     │    └──────────────────────────┘
                        │ (per hook point)   │              │
                        └───────────────────┘              ▼
                                 │
                        ┌───────────────────────────────────────────────────┐
                        │ Gemma3DecoderLayer (per layer, per hook point)    │
                        │   steering_vector_<hp>:  (1, H)  ← global API    │
                        │   steering_table_<hp>:   (max_configs+2, H)      │
                        │   steering_index:   (max_tokens,)  ← shared      │
                        │                                                   │
                        │   apply_steering(residual, table_<hp>, index)    │
                        │   = residual + table[index[:N]]                  │
                        └───────────────────────────────────────────────────┘
```

### 1. Steering Table (Per-Layer, Per-Hook-Point Buffer)

Each steerable decoder layer has a per-hook-point `steering_table_<hp>` buffer
(e.g. `steering_table_post_mlp_pre_ln`) of shape `(max_steering_configs + 3, hidden_size)`.

| Row | Contents | When used |
|-----|----------|-----------|
| 0 | Zeros | Tokens with no steering at all |
| 1 | Global prefill effective (base + prefill) | Prefill tokens without per-request steering |
| 2 | Global decode effective (base + decode) | Decode tokens without per-request steering |
| 3+ | Phase-appropriate global + per-request | Tokens with per-request steering |

The table is populated by `SteeringManager.populate_steering_tables()` before each forward pass.

### 2. Steering Index (Shared Buffer)

A single `steering_index` tensor of shape `(max_tokens,)` is shared across all decoder layers (same backing tensor). Each position maps a token to its steering table row:

```
Token positions: [decode₁, decode₂, prefill₁, prefill₂, decode₃]
steering_index:  [   4,       2,        1,        3,       4    ]
```

Updated in-place by the model runner before each forward pass.

### 3. Custom Op — Indexed Gather

The `apply_steering` custom op is opaque to torch.compile (prevents constant-folding) but is NOT a splitting op — it does not partition the compiled graph:

```python
# vllm/model_executor/layers/steering.py
def apply_steering(hidden_states, steering_table, steering_index):
    return hidden_states + steering_table[steering_index[:N]].to(hidden_states.dtype)
```

Buffer values updated in-place between forward passes are visible to CUDA graph replays. Steering adds zero graph partitions regardless of hook point count.

### 4. SteeringManager — Vector Lifecycle

`SteeringManager` (`vllm/v1/worker/steering_manager.py`) tracks per-request steering configs
with phase awareness:

- **Register**: Assigns a table row to a config hash with a phase ("prefill" or "decode"), stores vectors, increments refcount. Vectors format: `{hook_point: {layer_idx: [floats]}}`.
- **Release**: Decrements refcount, frees row when it hits 0
- **Deduplication**: Requests with identical effective vectors share a table row (same hash = same row)
- **Phase tracking**: `config_phase[hash]` maps each config to its phase, used during table population to combine with the right global effective vector
- **Three-tier globals**: Caches `global_base_vectors`, `global_prefill_vectors`, `global_decode_vectors` separately
- **populate_steering_tables**: Row 1 = base+prefill, Row 2 = base+decode, Rows 3+ = phase-appropriate global + per-request

### 5. Worker Layer — Global API

`WorkerBase` provides three RPC-callable methods that discover steerable layers by walking the model's module tree (looking for modules with `layer_idx` and any `steering_vector_*` attribute):

- **`set_steering_vectors(vectors_data, validate_only)`** — Accepts `{hook_point: {layer_idx: [floats]}}`. Validates hook point validity, vector dimensions, and finiteness, then copies vectors into per-hook `steering_vector_*` buffers. Notifies SteeringManager of global changes.
- **`clear_steering_vectors()`** — Zeros all `steering_vector_*` buffers across all hook points. Notifies SteeringManager.
- **`get_steering_status()`** — Returns `{layer_idx: {hook_point: {"norm": float}}}` for non-zero layers/hook-points.

### 6. API Layer — REST Endpoints

Three endpoints under `/v1/steering/`, gated behind `VLLM_SERVER_DEV_MODE`. See [API.md](API.md) for details.

## Intervention Points (Hook Points)

Steering can be applied at multiple positions within `Gemma3DecoderLayer.forward()`.
Each position is identified by a `SteeringHookPoint` enum value:

```
input_layernorm(hidden_states, residual)
    ↓
apply_steering(residual, table_pre_attn, index)    ← PRE_ATTN
    ↓
self_attn(hidden_states)
    ↓
post_attention_layernorm(hidden_states)
    ↓
apply_steering(residual, table_post_attn, index)   ← POST_ATTN
    ↓
pre_feedforward_layernorm(hidden_states, residual)
    ↓
mlp(hidden_states)
    ↓
apply_steering(residual, table_post_mlp_pre_ln, index)  ← POST_MLP_PRE_LN (default)
    ↓
post_feedforward_layernorm(hidden_states)
    ↓
apply_steering(residual, table_post_mlp_post_ln, index) ← POST_MLP_POST_LN
    ↓
return hidden_states, residual
```

Only hook points that are enabled (have buffers registered) are active.
The default hook point is `post_mlp_pre_ln`, which matches where Gemma 3
SAEs are trained.

## Key Files

| File | Purpose |
|------|---------|
| `vllm/config/steering.py` | `SteeringConfig` with `max_steering_configs` |
| `vllm/config/steering_types.py` | `SteeringVectorSpec`, `SteeringLayerEntry`, resolve/hash helpers |
| `vllm/model_executor/layers/steering.py` | Custom op, hook point enum, buffer attr mappings |
| `vllm/model_executor/models/gemma3.py` | Buffer registration and op calls |
| `vllm/v1/worker/steering_manager.py` | Phase-aware config management, refcounting, table population |
| `vllm/v1/worker/worker_base.py` | Three-tier global set/clear/status |
| `vllm/v1/worker/gpu_model_runner.py` | Phase-aware index building, prefill→decode transition |
| `vllm/v1/worker/gpu_input_batch.py` | Dual hash tracking (`prefill_steering_hash`, `decode_steering_hash`) |
| `vllm/v1/core/sched/scheduler.py` | Transition-aware dual-hash admission control |
| `vllm/v1/core/kv_cache_utils.py` | Prefill steering hash in block hash extra keys |
| `vllm/sampling_params.py` | Three-tier fields + co-located scales |
| `vllm/v1/request.py` | `prefill_steering_config_hash`, `decode_steering_config_hash` |
| `vllm/engine/arg_utils.py` | `--enable-steering` and `--max-steering-configs` CLI args |
| `vllm/entrypoints/serve/steering/api_router.py` | Three-tier REST endpoints with cache invalidation |
| `vllm/entrypoints/serve/steering/protocol.py` | `SetSteeringRequest` schema |

## Current Limitations

- **Gemma 3 only** — no other model has the steering buffers wired in.
- **Hook point support** — supports `pre_attn`, `post_attn`, `post_mlp_pre_ln`, `post_mlp_post_ln`; which are active depends on model configuration.
- **Dev mode only** — global HTTP API endpoints require `VLLM_SERVER_DEV_MODE=1`.

## Adding Steering to a New Model

To add steering to another model family (e.g., Llama, Qwen):

1. **Import the custom op module** at the top of the model file:
   ```python
   import vllm.model_executor.layers.steering  # noqa: F401  # registers custom op
   ```

2. **Add per-hook-point buffers** in the decoder layer `__init__`:
   ```python
   from vllm.model_executor.layers.steering import (
       HOOK_POINT_TABLE_ATTR, HOOK_POINT_VECTOR_ATTR, SteeringHookPoint,
   )

   self.layer_idx = extract_layer_index(prefix)

   # Register buffers for each active hook point
   for hp in active_hook_points:
       table_attr = HOOK_POINT_TABLE_ATTR[hp]
       vec_attr = HOOK_POINT_VECTOR_ATTR[hp]
       self.register_buffer(vec_attr, torch.zeros(1, config.hidden_size), persistent=False)
       self.register_buffer(table_attr, torch.zeros(max_steering_configs + 2, config.hidden_size), persistent=False)

   self.register_buffer("steering_index", torch.zeros(max_steering_tokens, dtype=torch.long), persistent=False)
   ```

3. **Call the op** in `forward()`, at each active hook point:
   ```python
   # At POST_MLP_PRE_LN position:
   residual = torch.ops.vllm.apply_steering(
       residual, self.steering_table_post_mlp_pre_ln, self.steering_index
   )
   ```

4. **Share the steering_index** across all layers in the model `__init__`:
   ```python
   if self.layers:
       shared_idx = self.layers[0].steering_index
       for layer in self.layers[1:]:
           layer.steering_index = shared_idx
   ```

The model runner and worker code is model-agnostic — it discovers steerable layers by walking the module tree for modules that have `layer_idx` and any `steering_table_*` or `steering_vector_*` attribute.

## Related Docs

- [API.md](API.md) — REST endpoint details and usage examples
- [BATCHING.md](BATCHING.md) — vLLM batching internals
- [ROADMAP.md](ROADMAP.md) — Phased plan for multi-model support, additional intervention points
- [EXTRACTION.md](EXTRACTION.md) — Extracting activations for SAE training
