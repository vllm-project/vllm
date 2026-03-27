# Steering MVP: Implementation Notes

This document captures everything learned while making decode-only activation steering work with vLLM's torch.compile + CUDA graph pipeline. It's meant as a reference for anyone extending steering to other models or debugging issues in this area.

## The Core Problem

vLLM compiles model forward passes with `torch.compile` (Inductor backend) and captures them as CUDA graphs for replay. Any value that is read dynamically at runtime — rather than passed as a tensor argument — gets **baked as a constant** during compilation or graph capture. This is the central constraint that shaped the entire implementation.

Activation steering needs to know which tokens in the batch are **decode tokens** (should be steered) vs **prefill tokens** (should not be steered, to avoid polluting the prefix cache with steered KV entries). The number of decode tokens changes every forward pass.

## What Doesn't Work

### Reading from ForwardContext inside the compiled graph

The first attempt used `get_num_decode_tokens()` which reads from the global `ForwardContext`:

```python
# BROKEN: value is baked as constant (0) during compilation
def apply_steering(hidden_states, steering_vector):
    num_decode_tokens = get_num_decode_tokens(default=0)  # <-- constant-folded
    mask = (torch.arange(hidden_states.shape[0]) < num_decode_tokens).unsqueeze(1)
    return hidden_states + mask * steering_vector
```

During torch.compile tracing (profiling/warmup), `ForwardContext` is either `None` or has `num_decode_tokens=0`. Inductor bakes this as a constant. The mask becomes all-false, the multiply becomes zero, and the addition gets dead-code-eliminated. On all subsequent forward passes, the compiled graph skips steering entirely — even when the mask buffer has been updated.

This fails regardless of whether the op is registered via `direct_register_custom_op` or inlined as regular Python.

### Inline tensor math with module buffers

```python
# BROKEN: torch.compile constant-folds buffer values seen at trace time
hidden_states = hidden_states + self.steering_decode_mask[:N] * self.steering_vector
```

`register_buffer` tensors are captured by torch.compile as part of the module state. If the buffer is all-zeros at trace time, Inductor may specialize the graph for that case. Even if it doesn't fully constant-fold, CUDA graph capture records the kernel launches that occurred with zero buffers, and replay won't re-evaluate the buffer contents if the kernels were optimized away.

### Custom op WITHOUT splitting_ops registration

```python
# INCOMPLETE: op is traced through by Inductor, not treated as opaque
direct_register_custom_op(op_name="apply_steering", op_func=fn, fake_impl=fake_fn)
```

`direct_register_custom_op` registers the op to the CUDA dispatch key. During FX tracing, the fake impl runs (returning `torch.empty_like`). During CUDA graph capture, the real impl runs — but with whatever buffer values exist at capture time (typically zeros). The captured graph then replays those zero-valued operations.

With `custom_ops: ['none']` (the default for `VLLM_COMPILE` mode), the `CustomOp` framework disables custom ops in favor of native implementations. But `direct_register_custom_op` bypasses the `CustomOp` framework entirely, so the `custom_ops` setting doesn't affect it directly. The issue is that without being a **splitting op**, the op is still traced through by Inductor.

### Wrong attention backend

The first fix attempt (option-a) added `num_decode_tokens` as a field on `FlashAttentionMetadata`. But on the test GPU, vLLM selected `TRITON_ATTN` as the attention backend, not `FLASH_ATTN`. The `TritonAttentionMetadata` class didn't have the field, so `get_num_decode_tokens()` fell back to `default=0` and no tokens were steered.

When adding fields to attention metadata, you must add them to **all** backends that might be selected, not just FlashAttention. The backend selection depends on GPU architecture, driver version, and model configuration.

### Insufficient steering magnitude with dummy weights

With `load_format="dummy"` (random weights), the model's logit distribution is dominated by noise. A steering vector of magnitude 10.0 per dimension is not enough to change the argmax token through 17+ layers of random transformations. The test needs a magnitude of ~500.0 to reliably flip the output. This isn't a bug — it's a property of random-weight models where one token (e.g., 66052 for Gemma 3) becomes a strong attractor regardless of hidden state perturbations.

## What Works

### Custom op as a splitting op with persistent mask buffers

The working approach combines three mechanisms:

1. **Per-layer `steering_decode_mask` registered buffers** — `register_buffer(..., persistent=False)` tensors that move to GPU with the model via `.to(device)`.

2. **A custom op registered as a splitting op** — `vllm::apply_steering` in `splitting_ops` makes torch.compile treat it as an opaque graph break point. The real Python implementation runs between compiled graph segments at runtime, reading whatever values the buffers currently hold.

3. **Model runner updates masks in-place before each forward** — `_update_steering_decode_mask()` sets mask positions to 1.0 for decode tokens and 0.0 for prefill/padding using `.zero_()` and `.fill_()`. These in-place operations are visible to CUDA graph replays because they modify the same GPU memory the captured kernels reference.

### Architecture

```
Request arrives
    │
    ▼
gpu_model_runner.execute_model()
    │
    ├── _build_attention_metadata()
    │
    ├── _update_steering_decode_mask(scheduler_output)
    │       │
    │       ├── Count decode tokens: sum(1 for n in num_scheduled_tokens.values() if n == 1)
    │       │   (decodes are ordered first in batch, each with 1 scheduled token)
    │       │
    │       └── For each steerable layer's mask buffer:
    │               mask.zero_()
    │               mask[:num_decode_tokens].fill_(1.0)
    │
    ├── set_forward_context(...)
    │
    └── model.forward(...)
            │
            └── Per Gemma3DecoderLayer:
                    │
                    ├── attention, MLP, layernorms ...
                    │
                    └── torch.ops.vllm.apply_steering(
                            hidden_states,        # (num_tokens, hidden_size)
                            self.steering_vector,  # (1, hidden_size) - zero = no-op
                            self.steering_decode_mask  # (max_tokens, 1) - updated by runner
                        )
                        │
                        └── hidden + mask[:N] * vec  (runs as real Python, not compiled)
```

### Files

| File | Role |
|------|------|
| `vllm/model_executor/layers/steering.py` | Custom op definition: `apply_steering(hidden, vec, mask)` with fake impl for FX tracing |
| `vllm/model_executor/models/gemma3.py` | `Gemma3DecoderLayer`: registers `steering_vector` and `steering_decode_mask` buffers, calls `torch.ops.vllm.apply_steering` in forward. `Gemma3Model`: passes `max_steering_tokens` from scheduler config to each layer. |
| `vllm/v1/worker/gpu_model_runner.py` | `_update_steering_decode_mask()`: discovers steerable layers (cached), counts decode tokens from scheduler output, updates masks in-place before forward |
| `vllm/v1/worker/worker_base.py` | `set_steering_vectors()` / `clear_steering_vectors()` / `get_steering_status()`: API for modifying steering vector buffers via `collective_rpc` |
| `vllm/entrypoints/serve/steering/api_router.py` | REST API: `POST /v1/steering/set`, `POST /v1/steering/clear`, `GET /v1/steering` |
| `vllm/config/compilation.py` | `vllm::apply_steering` added to `splitting_ops` so Inductor treats it as a graph break |
| `vllm/v1/attention/backends/triton_attn.py` | Added `num_decode_tokens` field (good hygiene, not strictly required for buffer approach) |
| `vllm/v1/attention/backends/flash_attn.py` | Added `num_decode_tokens` field |

### Key Invariants

1. **Steering vectors are always present on every steerable layer** — zero vector is a numerical no-op. No conditional branches based on "is steering active" in the forward path.

2. **Mask buffers are always present and always updated** — even when no steering vectors are set. The mask computation runs every forward pass (~34 `.zero_()` + `.fill_()` calls). This is negligible cost compared to the model forward.

3. **Steering only affects decode tokens** — prefill tokens pass through with mask=0.0, so the KV cache entries written during prefill are unsteered. This is important for prefix caching: steered and unsteered requests can share prefix cache entries for the prompt.

4. **Buffers are registered with `persistent=False`** — they don't appear in state dicts or checkpoints. They're transient runtime state.

5. **The custom op is a splitting op** — it creates a graph break in the torch.compile piecewise compilation. The real Python `apply_steering` function executes between compiled segments. This is the same mechanism used by attention ops (`vllm::unified_attention`, etc.).

## Testing

### What the tests cover

| Test | What it verifies |
|------|-----------------|
| `test_steering_op.py` (7 tests) | Buffer math: decode-only masking, zero vector no-op, dtype preservation, oversized mask slicing, in-place update visibility |
| `test_worker_steering.py` (25 tests) | WorkerBase methods: set/clear/status, validation (wrong size, NaN, Inf), validate-only mode, no-model-runner edge cases |
| `test_api_router.py` (22 tests) | REST API: set/clear/get, scales, replace mode, error responses (400/500), dev mode gating |
| `test_forward_context.py` (10 tests) | `get_num_decode_tokens()` utility: all ForwardContext layouts, fallback paths |
| `test_protocol.py` (9 tests) | Pydantic request validation: key coercion, defaults, required fields |
| `test_steering.py` (1 test) | E2E with real model (dummy weights): baseline → steer → verify different → clear → verify restored. Runs with torch.compile + CUDA graphs (default VllmRunner settings). |

### Running tests

```bash
# Fast unit tests (~20s)
.venv/bin/python -m pytest tests/entrypoints/serve/steering/ tests/model_executor/layers/test_steering_op.py -v

# E2E test with real model (~50s, needs GPU)
.venv/bin/python -m pytest tests/models/language/generation/test_steering.py -v --timeout=300
```

## Extending to Other Models

To add steering to a new model:

1. In the decoder layer `__init__`, add `max_steering_tokens: int = 1` parameter and register both buffers:
   ```python
   self.register_buffer("steering_vector", torch.zeros(1, hidden_size), persistent=False)
   self.register_buffer("steering_decode_mask", torch.zeros(max_steering_tokens, 1), persistent=False)
   ```

2. In the decoder layer `forward`, add after the MLP + post-layernorm:
   ```python
   import vllm.model_executor.layers.steering  # noqa: F401  (at module top)
   # ...
   hidden_states = torch.ops.vllm.apply_steering(
       hidden_states, self.steering_vector, self.steering_decode_mask
   )
   ```

3. In the model class that creates layers, pass `max_steering_tokens=vllm_config.scheduler_config.max_num_batched_tokens` to each layer constructor.

4. Also set `self.layer_idx = extract_layer_index(prefix)` on the decoder layer so `WorkerBase._steerable_layers()` can discover it.

The model runner and worker code is model-agnostic — it discovers steerable layers by walking the module tree for modules that have both `steering_vector` and `steering_decode_mask` attributes.
