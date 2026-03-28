# Steering MVP: Implementation Notes

This document captures everything learned while making decode-only activation steering work with vLLM's torch.compile + CUDA graph pipeline. It's meant as a reference for anyone extending steering to other models or debugging issues in this area.

> **Note**: The MVP used a broadcast `steering_vector` + `steering_decode_mask` approach. This has since been replaced by the request-indexed gather pattern (`steering_table` + `steering_index`) in Phase 2. See [STEERING.md](STEERING.md) for the current architecture. The lessons below about torch.compile and CUDA graph constraints remain relevant.

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

During torch.compile tracing (profiling/warmup), `ForwardContext` is either `None` or has `num_decode_tokens=0`. Inductor bakes this as a constant. The mask becomes all-false, the multiply becomes zero, and the addition gets dead-code-eliminated.

### Inline tensor math with module buffers

```python
# BROKEN: torch.compile constant-folds buffer values seen at trace time
hidden_states = hidden_states + self.steering_decode_mask[:N] * self.steering_vector
```

`register_buffer` tensors are captured by torch.compile as part of the module state. If the buffer is all-zeros at trace time, Inductor may specialize the graph for that case.

### Custom op WITHOUT splitting_ops registration

```python
# INCOMPLETE: op is traced through by Inductor, not treated as opaque
direct_register_custom_op(op_name="apply_steering", op_func=fn, fake_impl=fake_fn)
```

Without being a **splitting op**, the op is still traced through by Inductor. The captured graph replays zero-valued operations.

### Insufficient steering magnitude with dummy weights

With `load_format="dummy"` (random weights), a steering vector of magnitude 10.0 per dimension is not enough to change the argmax token through 17+ layers of random transformations. Tests need ~500.0 magnitude to reliably flip the output.

## What Works

### Custom op as a splitting op with persistent buffers

The working approach combines three mechanisms:

1. **Persistent registered buffers** — `register_buffer(..., persistent=False)` tensors that move to GPU with the model. In-place updates (`.zero_()`, `.fill_()`, `.copy_()`) are visible to CUDA graph replays because they modify the same GPU memory.

2. **Custom op registered as a splitting op** — `vllm::apply_steering` in `splitting_ops` makes torch.compile treat it as an opaque graph break point. The real Python implementation runs between compiled graph segments at runtime, reading whatever values the buffers currently hold.

3. **Model runner updates buffers in-place before each forward** — The model runner writes fresh data into the buffers before each forward pass. These in-place operations are visible to CUDA graph replays.

### Key Invariants

1. **Buffers are always present** — zero values make the addition a no-op. No conditional branches in the forward path.
2. **Buffers are updated every step** — even when no steering is active. The cost is negligible.
3. **Steering only affects decode tokens** — prefill tokens see zeros, preserving prefix cache correctness.
4. **Buffers are non-persistent** — they don't appear in state dicts or checkpoints. They're transient runtime state.
5. **The custom op is a splitting op** — same mechanism as attention ops (`vllm::unified_attention`, etc.).

## Testing

### What the tests cover

| Test | What it verifies |
|------|-----------------|
| `test_steering_op.py` (8 tests) | Indexed gather math: row selection, mixed indices, dtype preservation, oversized buffer slicing, in-place update visibility |
| `test_steering_manager.py` (18 tests) | SteeringManager: register/release lifecycle, refcounting, dedup, table population, additive combination, capacity exhaustion |
| `test_worker_steering.py` (25 tests) | WorkerBase methods: set/clear/status, validation (wrong size, NaN, Inf), validate-only mode, no-model-runner edge cases |
| `test_steering_scheduler.py` (8 tests) | Scheduler admission control logic: capacity checks, hash dedup, freed capacity |
| `test_steering.py` (2 tests) | E2E with real model (dummy weights): global steering changes output, per-request steering via SamplingParams changes output without contamination |

### Running tests

```bash
# All unit tests (~20s)
.venv/bin/python -m pytest tests/entrypoints/serve/steering/ tests/model_executor/layers/test_steering_op.py tests/v1/worker/test_steering_manager.py tests/v1/core/test_steering_scheduler.py -v

# E2E test with real model (~70s, needs GPU)
.venv/bin/python -m pytest tests/models/language/generation/test_steering.py -v --timeout=300
```
