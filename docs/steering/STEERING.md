# Live Activation Steering

This document describes the activation steering implementation in this vLLM fork.

## Overview

Activation steering modifies model hidden states at runtime by adding pre-computed vectors to the residual stream during decode. This enables:
- **Representation engineering**: Adding "concept vectors" to shift model behavior
- **SAE feature steering**: Amplifying or suppressing specific SAE features via decoder columns
- **Safety interventions**: Modifying activations to reduce harmful outputs
- **Behavioral control**: Adjusting style, tone, or content properties

Steering supports two modes:
- **Global steering**: Server-wide vectors applied to all requests (via HTTP API)
- **Per-request steering**: Different vectors per request (via `SamplingParams.steering_vectors`)

Both modes are additive — when a request has per-request steering and global steering is also active, the effective vector is `global + per_request`.

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
(e.g. `steering_table_post_mlp_pre_ln`) of shape `(max_steering_configs + 2, hidden_size)`.
Only hook points that are active for the model get buffers.

| Row | Contents | When used |
|-----|----------|-----------|
| 0 | Zeros | Prefill tokens, padding |
| 1 | Global steering vector | Decode tokens without per-request steering |
| 2+ | Global + per-request combined | Decode tokens with per-request steering |

The table is populated by `SteeringManager.populate_steering_tables()` before each forward pass.

### 2. Steering Index (Shared Buffer)

A single `steering_index` tensor of shape `(max_tokens,)` is shared across all decoder layers (same backing tensor). Each position maps a token to its steering table row:

```
Token positions: [decode₁, decode₂, prefill₁, prefill₂, decode₃]
steering_index:  [   2,       1,        0,        0,       3    ]
```

Updated in-place by the model runner before each forward pass.

### 3. Custom Op — Indexed Gather

The `apply_steering` custom op is registered as a splitting point for `torch.compile`:

```python
# vllm/model_executor/layers/steering.py
def apply_steering(hidden_states, steering_table, steering_index):
    return hidden_states + steering_table[steering_index[:N]].to(hidden_states.dtype)
```

The splitting point ensures the Python implementation runs at runtime between compiled graph segments, reading live buffer values rather than baked-in constants. This is the same mechanism used by attention ops.

### 4. SteeringManager — Vector Lifecycle

`SteeringManager` (`vllm/v1/worker/steering_manager.py`) tracks per-request steering configs
keyed by hook point:

- **Register**: Assigns a table row to a config hash, stores vectors keyed by hook point, increments refcount. Vectors format: `{hook_point: {layer_idx: [floats]}}`.
- **Release**: Decrements refcount, frees row when it hits 0
- **Deduplication**: Requests with identical vectors share a table row (same hash = same row)
- **Global sync**: Caches global vectors per hook point set via the HTTP API, includes them in combined rows
- **populate_steering_tables**: Iterates over all hook points, writing combined vectors into each layer's per-hook table buffer

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
| `vllm/model_executor/layers/steering.py` | Custom op definition (`apply_steering` indexed gather) |
| `vllm/model_executor/layers/steering.py` | Hook point enum, buffer attr mappings (`HOOK_POINT_TABLE_ATTR`, `HOOK_POINT_VECTOR_ATTR`) |
| `vllm/model_executor/models/gemma3.py` | Buffer registration (per-hook `steering_table_*`, `steering_index`, `steering_vector_*`) and op calls |
| `vllm/v1/worker/steering_manager.py` | `SteeringManager` — config registration, refcounting, table population |
| `vllm/v1/worker/worker_base.py` | Worker methods for global set/clear/status |
| `vllm/v1/worker/gpu_model_runner.py` | `_update_steering_buffers()` — per-step table and index updates |
| `vllm/v1/worker/gpu_input_batch.py` | Per-request `steering_config_hash` tracking in InputBatch |
| `vllm/v1/core/sched/scheduler.py` | Admission control for `max_steering_configs` capacity |
| `vllm/sampling_params.py` | `steering_vectors` field on `SamplingParams` |
| `vllm/v1/request.py` | `steering_config_hash` cached property |
| `vllm/engine/arg_utils.py` | `--enable-steering` and `--max-steering-configs` CLI args |
| `vllm/entrypoints/serve/steering/api_router.py` | REST endpoints |
| `vllm/entrypoints/serve/steering/protocol.py` | `SetSteeringRequest` schema |
| `vllm/config/compilation.py` | `vllm::apply_steering` in `splitting_ops` |

## Current Limitations

- **Gemma 3 only** — no other model has the steering buffers wired in.
- **Decode-only** — prefill tokens are not steered.
- **Hook point support** — supports `pre_attn`, `post_attn`, `post_mlp_pre_ln`, `post_mlp_post_ln`; which are active depends on model configuration.
- **Dev mode only** — global HTTP API endpoints require `VLLM_SERVER_DEV_MODE=1`.
- **Spec decode** — the decode detection heuristic (`n_tokens == 1`) doesn't work with speculative decoding.

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
