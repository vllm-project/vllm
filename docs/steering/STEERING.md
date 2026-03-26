# Live Activation Steering

This document describes the current activation steering implementation in this vLLM fork.

## Overview

Activation steering modifies model hidden states at runtime by adding pre-computed vectors to the residual stream during decode. This enables:
- **Representation engineering**: Adding "concept vectors" to shift model behavior
- **SAE feature steering**: Amplifying or suppressing specific SAE features via decoder columns
- **Safety interventions**: Modifying activations to reduce harmful outputs
- **Behavioral control**: Adjusting style, tone, or content properties

## Architecture

The implementation uses a **stateful buffer pattern** with four components:

```
                  API Layer                    Worker Layer                 Model Layer
          ┌─────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
          │  POST /v1/steering/ │    │  WorkerBase              │    │  Gemma3DecoderLayer   │
          │  set / clear / GET  │───>│  set_steering_vectors()  │───>│  steering_vector buf  │
          │                     │    │  clear_steering_vectors() │    │  [1, hidden_size]     │
          │  Two-phase validate │    │  get_steering_status()   │    │                       │
          │  + apply via lock   │    │  _steerable_layers()     │    │  apply_steering op    │
          └─────────────────────┘    └──────────────────────────┘    └──────────────────────┘
                                              │                               │
                                     collective_rpc()                  ForwardContext
                                     (all workers)               (reads num_decode_tokens)
```

### 1. Model Layer — Buffer + Custom Op

Each steerable decoder layer has a non-persistent `steering_vector` buffer of shape `[1, hidden_size]`, initialized to zeros. During forward, the buffer is unconditionally added to hidden states via a custom op:

```python
# In Gemma3DecoderLayer.__init__()
self.register_buffer("steering_vector", torch.zeros(1, config.hidden_size), persistent=False)

# In Gemma3DecoderLayer.forward(), after MLP + post_feedforward_layernorm
hidden_states = torch.ops.vllm.apply_steering(hidden_states, self.steering_vector)
```

The unconditional addition is required by two constraints:
- **`@support_torch_compile`**: Data-dependent branching (`if steering is not None`) causes graph breaks. The forward path must be identical whether steering is active or not.
- **CUDA graphs**: Registered buffers have stable addresses that persist across graph replays. Values are updated between replays via `.copy_()`.

When steering is inactive, the zero buffer makes the addition a no-op.

### 2. Custom Op — Decode-Only Masking

The `apply_steering` op is registered via `direct_register_custom_op` so `torch.compile` treats it as opaque. At runtime it reads `num_decode_tokens` from the `ForwardContext` to apply steering only to decode tokens (the first N rows of the hidden states tensor):

```python
# vllm/model_executor/layers/steering.py
def apply_steering(hidden_states, steering_vector):
    num_decode_tokens = get_num_decode_tokens(default=0)
    decode_mask = (torch.arange(hidden_states.shape[0], device=hidden_states.device)
                   < num_decode_tokens).unsqueeze(1)
    return hidden_states + decode_mask.to(hidden_states.dtype) * steering_vector
```

Reading from forward context at runtime (rather than accepting `num_decode_tokens` as a parameter) avoids the value being baked into the compiled graph as a constant.

### 3. Worker Layer — Buffer Management

`WorkerBase` provides three RPC-callable methods that discover steerable layers by walking the model's module tree (looking for modules with both `steering_vector` and `layer_idx` attributes):

- **`set_steering_vectors(vectors_data, validate_only)`** — Validates vector dimensions and finiteness, then copies scaled vectors into buffers. The `validate_only` flag supports two-phase commit.
- **`clear_steering_vectors()`** — Zeros all steering buffers.
- **`get_steering_status()`** — Returns `{layer_idx: {"norm": float}}` for non-zero layers.

### 4. API Layer — REST Endpoints

Three endpoints under `/v1/steering/`, gated behind `VLLM_SERVER_DEV_MODE`. See [API.md](API.md) for details.

## Intervention Point

Steering is applied in `Gemma3DecoderLayer.forward()` after the MLP and `post_feedforward_layernorm`, before the return:

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
apply_steering(hidden_states, steering_vector)    ← HERE
    ↓
return hidden_states, residual
```

The next layer's `input_layernorm(hidden_states, residual)` fuses the residual add: `residual_new = hidden_states + residual`. So the steering vector enters the residual stream directly, which matches where Gemma 3 SAEs are trained.

## Key Files

| File | Purpose |
|------|---------|
| `vllm/model_executor/layers/steering.py` | Custom op definition (`apply_steering`) |
| `vllm/model_executor/models/gemma3.py` | Buffer registration and op call in forward |
| `vllm/v1/worker/worker_base.py` | Worker methods for set/clear/status |
| `vllm/forward_context.py` | `get_num_decode_tokens()` helper |
| `vllm/entrypoints/serve/steering/api_router.py` | REST endpoints |
| `vllm/entrypoints/serve/steering/protocol.py` | `SetSteeringRequest` schema |

## Current Limitations

- **Gemma 3 only** — no other model has the steering buffer or op import.
- **Decode-only** — prefill tokens are not steered.
- **Global steering** — all requests in a batch see the same vectors. No per-request steering.
- **Post-MLP only** — no pre-attention or post-attention intervention points.
- **Dev mode only** — API endpoints require `VLLM_SERVER_DEV_MODE=1`.

## Adding Steering to a New Model

To add steering to another model family (e.g., Llama, Qwen):

1. **Import the custom op module** at the top of the model file:
   ```python
   import vllm.model_executor.layers.steering  # noqa: F401  # registers custom op
   ```

2. **Add `layer_idx` and buffer** in the decoder layer `__init__`:
   ```python
   self.layer_idx = extract_layer_index(prefix)
   self.register_buffer("steering_vector", torch.zeros(1, config.hidden_size), persistent=False)
   ```

3. **Call the op** in `forward()`, after the MLP (and post-MLP layernorm if the model has one):
   ```python
   hidden_states = torch.ops.vllm.apply_steering(hidden_states, self.steering_vector)
   ```

No changes to the worker or API layers are needed — `_steerable_layers()` discovers any module with both `steering_vector` and `layer_idx` automatically.

## Related Docs

- [API.md](API.md) — REST endpoint details and usage examples
- [BATCHING.md](BATCHING.md) — vLLM batching internals (context for per-request steering)
- [ROADMAP.md](ROADMAP.md) — Phased plan for per-request steering, multi-model support, etc.
- [TODO.md](TODO.md) — Known bugs and open issues
