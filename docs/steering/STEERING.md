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

The implementation uses a **request-indexed gather** pattern with five components:

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
                        └───────────────────┘              │
                                 │                          ▼
                        ┌───────────────────────────────────────────────────┐
                        │ Gemma3DecoderLayer (per layer)                    │
                        │   steering_vector:  (1, H)  ← global API writes  │
                        │   steering_table:   (max_configs+2, H)           │
                        │   steering_index:   (max_tokens,)  ← shared      │
                        │                                                   │
                        │   apply_steering(residual, table, index)         │
                        │   = residual + table[index[:N]]                  │
                        └───────────────────────────────────────────────────┘
```

### 1. Steering Table (Per-Layer Buffer)

Each steerable decoder layer has a `steering_table` buffer of shape `(max_steering_configs + 2, hidden_size)`:

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

`SteeringManager` (`vllm/v1/worker/steering_manager.py`) tracks per-request steering configs:

- **Register**: Assigns a table row to a config hash, stores vectors, increments refcount
- **Release**: Decrements refcount, frees row when it hits 0
- **Deduplication**: Requests with identical vectors share a table row (same hash = same row)
- **Global sync**: Caches global vectors set via the HTTP API, includes them in combined rows

### 5. Worker Layer — Global API

`WorkerBase` provides three RPC-callable methods that discover steerable layers by walking the model's module tree (looking for modules with both `steering_vector` and `layer_idx` attributes):

- **`set_steering_vectors(vectors_data, validate_only)`** — Validates vector dimensions and finiteness, then copies vectors into `steering_vector` buffers. Notifies SteeringManager of global changes.
- **`clear_steering_vectors()`** — Zeros all `steering_vector` buffers. Notifies SteeringManager.
- **`get_steering_status()`** — Returns `{layer_idx: {"norm": float}}` for non-zero layers.

### 6. API Layer — REST Endpoints

Three endpoints under `/v1/steering/`, gated behind `VLLM_SERVER_DEV_MODE`. See [API.md](API.md) for details.

## Intervention Point

Steering is applied in `Gemma3DecoderLayer.forward()` to the raw residual stream, after the MLP and `post_feedforward_layernorm`:

```
input_layernorm(hidden_states, residual)
    ↓
self_attn(hidden_states)
    ↓
post_attention_layernorm(hidden_states)
    ↓
pre_feedforward_layernorm(hidden_states, residual)
    ↓
mlp(hidden_states)
    ↓
post_feedforward_layernorm(hidden_states)
    ↓
apply_steering(residual, steering_table, steering_index)    ← HERE
    ↓
return hidden_states, residual
```

The next layer's `input_layernorm(hidden_states, residual)` fuses the residual add: `residual_new = hidden_states + residual`. So the steering vector enters the residual stream directly, which matches where Gemma 3 SAEs are trained.

## Key Files

| File | Purpose |
|------|---------|
| `vllm/config/steering.py` | `SteeringConfig` with `max_steering_configs` |
| `vllm/model_executor/layers/steering.py` | Custom op definition (`apply_steering` indexed gather) |
| `vllm/model_executor/models/gemma3.py` | Buffer registration (`steering_table`, `steering_index`, `steering_vector`) and op call |
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
- **Post-MLP only** — no pre-attention or post-attention intervention points.
- **Dev mode only** — global HTTP API endpoints require `VLLM_SERVER_DEV_MODE=1`.
- **Spec decode** — the decode detection heuristic (`n_tokens == 1`) doesn't work with speculative decoding.

## Adding Steering to a New Model

To add steering to another model family (e.g., Llama, Qwen):

1. **Import the custom op module** at the top of the model file:
   ```python
   import vllm.model_executor.layers.steering  # noqa: F401  # registers custom op
   ```

2. **Add buffers** in the decoder layer `__init__`:
   ```python
   self.layer_idx = extract_layer_index(prefix)
   self.register_buffer("steering_vector", torch.zeros(1, config.hidden_size), persistent=False)
   self.register_buffer("steering_table", torch.zeros(max_steering_configs + 2, config.hidden_size), persistent=False)
   self.register_buffer("steering_index", torch.zeros(max_steering_tokens, dtype=torch.long), persistent=False)
   ```

3. **Call the op** in `forward()`, on the residual stream after the MLP:
   ```python
   residual = torch.ops.vllm.apply_steering(residual, self.steering_table, self.steering_index)
   ```

4. **Share the steering_index** across all layers in the model `__init__`:
   ```python
   if self.layers:
       shared_idx = self.layers[0].steering_index
       for layer in self.layers[1:]:
           layer.steering_index = shared_idx
   ```

The model runner and worker code is model-agnostic — it discovers steerable layers by walking the module tree for modules that have both `steering_table` and `layer_idx` attributes.

## Related Docs

- [API.md](API.md) — REST endpoint details and usage examples
- [BATCHING.md](BATCHING.md) — vLLM batching internals
- [ROADMAP.md](ROADMAP.md) — Phased plan for multi-model support, additional intervention points
- [EXTRACTION.md](EXTRACTION.md) — Extracting activations for SAE training
