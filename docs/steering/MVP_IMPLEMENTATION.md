# Steering Implementation Notes

This document captures everything learned while making activation steering work with vLLM's torch.compile + CUDA graph pipeline. It covers both the original MVP (broadcast steering) and the subsequent per-request steering implementation. It's meant as a reference for anyone extending steering to other models or debugging issues in this area.

## Phase 1: MVP (Broadcast Steering)

### The Core Problem

vLLM compiles model forward passes with `torch.compile` (Inductor backend) and captures them as CUDA graphs for replay. Any value that is read dynamically at runtime — rather than passed as a tensor argument — gets **baked as a constant** during compilation or graph capture. This is the central constraint that shaped the entire implementation.

Activation steering needs to know which tokens in the batch are **decode tokens** (should be steered) vs **prefill tokens** (should not be steered, to avoid polluting the prefix cache with steered KV entries). The number of decode tokens changes every forward pass.

### What Doesn't Work

#### Reading from ForwardContext inside the compiled graph

The first attempt used `get_num_decode_tokens()` which reads from the global `ForwardContext`:

```python
# BROKEN: value is baked as constant (0) during compilation
def apply_steering(hidden_states, steering_vector):
    num_decode_tokens = get_num_decode_tokens(default=0)  # <-- constant-folded
    mask = (torch.arange(hidden_states.shape[0]) < num_decode_tokens).unsqueeze(1)
    return hidden_states + mask * steering_vector
```

During torch.compile tracing (profiling/warmup), `ForwardContext` is either `None` or has `num_decode_tokens=0`. Inductor bakes this as a constant. The mask becomes all-false, the multiply becomes zero, and the addition gets dead-code-eliminated.

#### Inline tensor math with module buffers

```python
# BROKEN: torch.compile constant-folds buffer values seen at trace time
hidden_states = hidden_states + self.steering_decode_mask[:N] * self.steering_vector
```

`register_buffer` tensors are captured by torch.compile as part of the module state. If the buffer is all-zeros at trace time, Inductor may specialize the graph for that case.

#### Custom op WITHOUT splitting_ops registration

```python
# INCOMPLETE: op is traced through by Inductor, not treated as opaque
direct_register_custom_op(op_name="apply_steering", op_func=fn, fake_impl=fake_fn)
```

Without being a **splitting op**, the op is still traced through by Inductor. The captured graph replays zero-valued operations.

#### Insufficient steering magnitude with dummy weights

With `load_format="dummy"` (random weights), a steering vector of magnitude 10.0 per dimension is not enough to change the argmax token through 17+ layers of random transformations. Tests need ~500.0 magnitude to reliably flip the output. This isn't a bug — it's a property of random-weight models where certain tokens become strong attractors regardless of hidden state perturbations.

### What Works: Splitting Op + Persistent Buffers

The working approach combines three mechanisms:

1. **Persistent registered buffers** — `register_buffer(..., persistent=False)` tensors that move to GPU with the model. In-place updates (`.zero_()`, `.fill_()`, `.copy_()`) are visible to CUDA graph replays because they modify the same GPU memory.

2. **Custom op registered as a splitting op** — `vllm::apply_steering` in `splitting_ops` makes torch.compile treat it as an opaque graph break point. The real Python implementation runs between compiled graph segments at runtime, reading whatever values the buffers currently hold.

3. **Model runner updates buffers in-place before each forward** — The model runner writes fresh data into the buffers before each forward pass. These in-place operations are visible to CUDA graph replays.

### MVP Key Invariants

1. **Buffers are always present** — zero values make the addition a no-op. No conditional branches in the forward path.
2. **Buffers are updated every step** — even when no steering is active. The cost is negligible.
3. **Steering only affects decode tokens** — prefill tokens see zeros, preserving prefix cache correctness.
4. **Buffers are non-persistent** — they don't appear in state dicts or checkpoints. They're transient runtime state.
5. **The custom op is a splitting op** — same mechanism as attention ops (`vllm::unified_attention`, etc.).

---

## Phase 2: Per-Request Steering

The MVP applied one global steering vector to all requests. Phase 2 added per-request steering where different requests in the same batch get different vectors. This section documents the design choices, what we tried, and what we learned.

### Design Space Exploration

We considered four approaches:

| Approach | Memory | CUDA Graph Safe | Flexibility |
|----------|--------|-----------------|-------------|
| Per-token matrix `(max_tokens, H)` per layer | ~2 GB | Yes | Full |
| Request-indexed gather `(max_configs+2, H)` + index | ~65 MB | Yes | Full |
| Named configs (LoRA-style pre-registration) | ~65 MB | Yes | Pre-registered only |
| Palette + per-request scale | ~1 MB | Yes | Limited directions |

**We chose request-indexed gather** because it balances memory efficiency with full flexibility. The per-token matrix was a non-starter at ~2 GB. The palette approach was too restrictive. Named configs are a natural future extension of the gather approach.

### The LoRA Analogy (and Where It Breaks Down)

vLLM's LoRA implementation was the primary reference for per-request customization:
- LoRA: `request_lora_mapping[req_idx]` stores an adapter ID; scheduler caps at `max_loras`
- Steering: `request_steering_config_hash[req_idx]` stores a config hash; scheduler caps at `max_steering_configs`

The pattern transferred cleanly to the scheduler (admission control, `skipped_waiting` queue, FCFS retry), InputBatch tracking (add/remove/condense), and the model runner (register on arrival, release on finish).

**Where it differs from LoRA**: LoRA adapters are heavyweight (loaded from files, cached in an LRU). Steering vectors are lightweight (specified inline with each request, ~4 KB per layer). This means:
- No adapter loading/caching infrastructure needed
- Deduplication is by content hash, not by file path
- The `SteeringManager` is ~150 lines vs LoRA's multi-file manager

### Shared steering_index: One Tensor, All Layers

Each decoder layer needs to know which table row each token should use. Naively, each layer would have its own `steering_index` buffer, but they'd all contain identical data. Instead, we register a placeholder buffer on each layer during `__init__`, then in `Gemma3Model.__init__()` replace all layers' buffers with one shared tensor:

```python
if self.layers:
    shared_idx = self.layers[0].steering_index
    for layer in self.layers[1:]:
        layer.steering_index = shared_idx
```

This works because:
- `register_buffer` stores a reference, not a copy
- Reassigning `layer.steering_index = shared_tensor` updates the reference
- The custom op receives `self.steering_index` which resolves to the shared tensor
- In-place updates from the model runner are visible to all layers
- CUDA graph capture sees the same memory address for all layers

**Lesson**: PyTorch `register_buffer` with `persistent=False` creates module attributes that can be freely reassigned to share backing storage. This is safe for CUDA graphs as long as the tensor object identity doesn't change after graph capture.

### Hash Overflow Bug

The `steering_config_hash` property used `int(hashlib.sha256(...).hexdigest()[:16], 16)` which produces values up to 2^64 - 1. But `InputBatch.request_steering_config_hash` is a `np.int64` array (signed, max 2^63 - 1). Values in the upper half of the uint64 range caused `OverflowError: Python int too large to convert to C long` when assigned.

**Fix**: Mask the hash to 63 bits: `& 0x7FFFFFFFFFFFFFFF`. This halves the hash space but 2^63 distinct values is more than sufficient for collision avoidance.

**Lesson**: When storing Python ints in numpy arrays, check the dtype's signedness. `np.int64` is signed; `np.uint64` would work but has its own interop quirks. Masking to fit is simpler.

### Additive Combination: Global + Per-Request

We chose additive combination (`effective = global + per_request`) over replacement. The `SteeringManager.populate_steering_tables()` method pre-computes the sum for each active config and writes it into the table row. This means:

- Row 1 always has the current global vector (or zeros)
- Rows 2+ have `global + per_request_i` for each active config
- When global steering changes (via the HTTP API), all combined rows are recomputed on the next `populate_steering_tables()` call — which happens every step

**Lesson**: Pre-computing the sum in the table avoids doing two additions in the hot path (the custom op just does one indexed gather + add). The tradeoff is that global vector changes require rewriting all active rows, but this is cheap (~microseconds for a few rows) and global changes are rare.

### Worker → Manager Notification

When `WorkerBase.set_steering_vectors()` updates the global `steering_vector` buffer, it also calls `SteeringManager.update_global_vectors()`. This notification is necessary because the manager caches global vectors to compute combined rows. Without it, the table would contain stale global vectors.

The notification uses `hasattr` guards:
```python
if hasattr(self, "model_runner") and self.model_runner is not None:
    mgr = getattr(self.model_runner, "_steering_manager", None)
    if mgr is not None:
        for idx in valid_indices:
            mgr.update_global_vectors(idx, steerable[idx].steering_vector)
```

**Lesson**: The defensive `hasattr`/`getattr` chain is necessary because `set_steering_vectors` can be called before the model runner has initialized (during server startup). The manager is lazily created on first forward pass, so it may not exist yet.

### Lazy Initialization vs Pre-Allocation

The `SteeringManager` is lazily initialized on the first call to `_update_steering_buffers()`, not during model runner `__init__`. This is because:
- Steerable layers must be discovered by walking the model's module tree
- The model may not be loaded yet during `__init__`
- The steering config may be `None` (steering disabled)

However, the GPU buffers themselves (`steering_table`, `steering_index`) are **pre-allocated at model init time** in `Gemma3DecoderLayer.__init__()`. This is critical for CUDA graphs — buffer shapes must be fixed before graph capture.

**Lesson**: Lazy init is fine for CPU-side management objects. GPU buffers that participate in compiled/captured forward passes must be pre-allocated with their final shapes.

### Why Not Lazy Buffer Allocation?

We considered only allocating `steering_table` buffers on layers that actually receive steering. This would save memory when only a few layers are steered. We rejected this because:

1. The custom op call `torch.ops.vllm.apply_steering(residual, self.steering_table, self.steering_index)` must be present in every layer's forward at trace time
2. If the buffer doesn't exist, the op can't be called, and adding it later requires re-tracing/re-capturing
3. The memory cost is trivial (~36 KB per layer with `max_steering_configs=4`)

**Lesson**: With CUDA graphs, the shape of every buffer in the forward path is fixed at capture time. Conditional buffer existence is not compatible with this constraint. Pre-allocate everything, rely on zeros for no-op behavior.

### Scheduler Admission Control

The scheduler tracks distinct `steering_config_hash` values in the current batch, exactly mirroring the LoRA `scheduled_loras` pattern. When `max_steering_configs` slots are occupied:

1. New requests with a **new** hash are moved to `skipped_waiting`
2. They retry next step (FCFS priority)
3. Requests with a hash already in the batch pass through (dedup means no extra slot needed)
4. Requests without steering (`hash == 0`) are never blocked

**Lesson**: The LoRA admission control pattern is cleanly reusable. The key insight is that deduplication (same hash = same slot) means the effective capacity is higher than `max_steering_configs` when requests share configs.

### Dummy Weight Testing Pitfalls

With `load_format="dummy"`, certain tokens become strong attractors regardless of steering. In our tests:
- Token 66052 dominates for Gemma 3 with random weights
- Steering magnitudes of 500.0 per dimension are needed to change output
- But different magnitudes (e.g., 250.0 vs 500.0) can produce the **same** output because both push past the noise floor into the same attractor basin
- Opposite directions (+500 vs -500) reliably produce different output

**Lesson**: Don't assert that different steering magnitudes produce different tokens with dummy weights. Assert on opposite directions or on steered-vs-unsteered. The magnitude test only works with real weights where the logit landscape is meaningful.

---

## Phase 3: Multiple Hook Points

Phase 3 extended steering from a single post-MLP position to four hook points on the residual stream (`pre_attn`, `post_attn`, `post_mlp_pre_ln`, `post_mlp_post_ln`). This section documents what we learned about torch.compile, graph partitions, and buffer allocation strategy.

### Splitting Ops Are Not Required for Buffer Correctness

The Phase 1/2 implementation registered `apply_steering` as a splitting op based on the belief that torch.compile would constant-fold zero-valued buffers, eliminating the steering addition. During Phase 3, we tested removing the splitting op registration and found that **the custom op alone is sufficient**.

A custom op registered via `direct_register_custom_op` is opaque to the torch.compile tracer — Inductor cannot see through it to constant-fold the buffer values. The splitting op registration controls whether the op **partitions the compiled graph**, which is a separate concern from opacity.

Without the splitting op:
- The op is still opaque (Inductor generates a call to it, not inlined code)
- The compiled graph is NOT partitioned at the op call
- Buffer values updated in-place between forward passes are visible to CUDA graph replays
- The op runs as part of the compiled graph, not between graph segments

**Lesson**: `direct_register_custom_op` provides opacity. `splitting_ops` provides graph partitioning. You need opacity for correctness, but you don't necessarily need partitioning. We were conflating the two in Phase 1.

**Lesson**: The "inline tensor math with module buffers" approach documented as broken in Phase 1 fails for a different reason than we thought — it's not that buffers are constant-folded, it's that inline math gets traced through by Inductor and the buffer values at trace time are baked into the generated code. A custom op prevents tracing through the operation entirely.

### Inline Tensor Math Causes AOT Autograd Cache Pickle Failures

We attempted to remove the custom op entirely and use inline tensor math:

```python
# Fails with AOT autograd cache pickle error
residual = residual + self.steering_table_pre_attn[
    self.steering_index[:residual.shape[0]]
].to(residual.dtype)
```

This caused `AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'` during PyTorch's AOT autograd cache save step. The error comes from Python's `Enum` class (which uses `WeakValueDictionary` internally for member tracking) being captured in the traced graph's metadata.

The issue is NOT stale caches — clearing `/tmp/torchinductor_nymph/aotautograd/` does not help. The error is reproducible with a fresh cache.

**Lesson**: Inline tensor math that references module buffers can trigger AOT autograd serialization issues when the module's class hierarchy includes Python Enums or other unpicklable objects. Custom ops sidestep this by creating an opaque boundary that AOT autograd doesn't try to trace through.

### Always Allocate All Hook Point Buffers

We initially designed Phase 3 with an opt-in `--steering-hook-points` CLI flag that controlled which hook points allocated buffers. This was motivated by the belief that each hook point added a graph partition (splitting op). After discovering that splitting ops are unnecessary, the performance argument evaporated.

All four hook point buffers are now allocated unconditionally on every decoder layer:

| Config | Per Layer (Gemma 3 4B, bf16) | 26 Layers |
|--------|------------------------------|-----------|
| `max_steering_configs=4` (default) | 4 × 6 rows × 3072 × 2B = ~144 KB | ~3.6 MB |
| `max_steering_configs=16` | 4 × 18 rows × 3072 × 2B = ~432 KB | ~11.2 MB |

Benefits:
- No `hasattr` guards in the forward path — unconditional code is easier to reason about
- No `--steering-hook-points` CLI flag to configure
- No `SteeringConfig.steering_hook_points` field to manage
- No trace-time branching concerns for torch.compile
- Zero rows make unused hook points a no-op with negligible compute cost

**Lesson**: When the memory cost of "always allocate" is trivial relative to model size, prefer unconditional allocation over opt-in configuration. The configuration surface area (CLI flags, config fields, validation, documentation) costs more in complexity than the memory saves.

### Hook Points at the Same Residual State

In Gemma 3's architecture, some hook points see identical residual state:
- `pre_attn` and `post_attn` — attention only modifies `hidden_states`, not `residual`
- `post_mlp_pre_ln` and `post_mlp_post_ln` — `post_feedforward_layernorm` only modifies `hidden_states`

The steering effect is therefore identical whether applied at `pre_attn` or `post_attn` (and likewise for the post-MLP pair). Both hook points exist for semantic matching with SAE training positions, not because they produce different results.

**Lesson**: Hook point naming should match the conventions of the tools that produce steering vectors (e.g., TransformerLens SAE positions), even when the underlying residual state is identical at adjacent positions.

### Unified API: Hook Points Required

Phase 3 initially maintained backward compatibility with Phase 2's flat `steering_vectors: dict[int, list[float]]` format (mapped to `post_mlp_pre_ln` by default) alongside a new `steering_hook_vectors` field. This created:
- Two fields for the same concept
- Merging logic when both were specified
- Ambiguity about which format to use

We removed backward compatibility and unified on a single format:

```python
steering_vectors: dict[str, dict[int, list[float]]]
# {hook_point: {layer_idx: [floats]}}
```

**Lesson**: When adding a dimension to an API (hook points), don't maintain backward compatibility with the old format. The merging logic is a source of bugs and the old format will confuse users. Make the breaking change cleanly.

---

## Phase 4: Prefill Steering

Phase 4 extended steering from decode-only to both prefill and decode phases, with a three-tier additive composition model and prefix cache correctness.

### The Prefix Cache Problem

Steering modifies the residual stream during prefill, which changes all downstream KV cache entries. Two requests with the same prompt but different prefill steering produce different KV entries. The prefix cache must not share blocks between them.

We chose **steering-aware cache keys**: include the prefill steering config hash in `generate_block_hash_extra_keys()`. This adds zero overhead when no prefill steering is active (empty list contributes nothing to the hash). Global prefill steering changes are handled separately by calling `reset_prefix_cache(reset_running_requests=True)` rather than encoding mutable global state into every block hash.

**Lesson**: The `extra_keys` pattern in `hash_block_tokens()` is designed for exactly this kind of extension — LoRA, multimodal, and now steering all add keys through the same mechanism. Adding a new dimension to the cache key is a one-function change.

### Three-Tier Additive Composition

We debated three API options for specifying prefill vs decode steering:
1. **Override**: phase-specific replaces base at (hook, layer) granularity
2. **Mutual exclusion**: error if both base and phase-specific specified for same (hook, layer)
3. **Additive**: phase-specific adds to base after independent scaling

We chose **additive** because it's consistent with the existing global + per-request model (also additive) and opens up research possibilities (base direction + phase-specific adjustments).

The composition stack is: `global_base + global_phase + request_base + request_phase`, where each component is independently pre-scaled by its co-located scale factor. Resolution happens once at request creation via `resolve_effective_vectors()` and is cached.

**Lesson**: When the existing system already uses additive composition (global + per-request), extending to a new dimension (phase) with additive semantics is the path of least conceptual surprise.

### Co-Located Scales

The original API had a separate `scales: dict[int, float]` field that applied per-layer. With three tiers and per-hook-point granularity, a separate scales dict becomes ambiguous (which tier? which hook?). We moved scales to be co-located with each vector entry:

```python
# Bare list (scale=1.0)
{15: [0.1, 0.2, ...]}
# Explicit scale
{15: {"vector": [0.1, 0.2, ...], "scale": 1.5}}
```

This eliminates ambiguity and makes each entry self-describing. The `SamplingParams` validation accepts both forms. Resolution pre-multiplies scales into vectors so downstream code never sees the dict form.

**Lesson**: When adding dimensions to an API (phases, hook points), co-locate metadata (scales) with the data it applies to rather than using a separate parallel structure. The parallel structure doesn't scale.

### Two Hashes, One Active at a Time

Each request has two steering config hashes (`prefill_steering_config_hash` and `decode_steering_config_hash`), but only one is registered with the `SteeringManager` at any time. The lifecycle:

1. Request arrives → register prefill config (if any)
2. Prefill completes → release prefill config, register decode config
3. Request finishes → release whichever config is active

This matters for the `max_steering_configs` row budget. If we registered both upfront, a request with different prefill and decode configs would consume two rows for its entire lifetime, even though only one is active at any moment. With one-active-at-a-time, the row is freed for reuse during the other phase.

The transition detection happens in `_update_steering_buffers()`: `num_computed_after = num_computed + n_tokens; if num_computed_after >= num_prompt: _handle_steering_transition(...)`. This runs before the forward pass, so the decode config is ready for the next step's table population.

**Lesson**: For phased resources with a fixed budget, allocate-on-demand and release-on-transition is worth the complexity over allocate-all-upfront when the budget is tight (`max_steering_configs` is typically small, 4-16).

### Lazy Init First-Step Race Condition

The `SteeringManager` is lazily created in `_update_steering_buffers()`, but per-request config registration happens in `_update_states()` which runs BEFORE. On the very first request:

1. `_update_states()` tries to register the config: `getattr(self, "_steering_manager", None)` → `None` → silently skipped
2. `_update_steering_buffers()` creates the manager
3. The config was never registered → request gets no steering

This bug only affected the first request in the engine's lifetime. Subsequent requests worked because the manager already existed.

**Fix**: After creating the manager in the lazy init, scan the current batch and register any configs that were skipped:

```python
for i in range(self.input_batch.num_reqs):
    ph = int(self.input_batch.request_prefill_steering_hash[ri])
    if ph != 0 and ph not in self._steering_manager.config_to_row:
        self._steering_manager.register_config(ph, eff, phase="prefill")
```

**Lesson**: When using lazy initialization, audit all code paths that depend on the lazily-initialized object. If any run BEFORE the lazy init, they silently fail. The fix is either (a) move init earlier, or (b) backfill state after init. We chose (b) because moving the init earlier would require the model to be loaded, which it may not be during `__init__`.

### Phase Detection: Replacing the n_tokens == 1 Heuristic

The old decode detection heuristic `is_decode = (n_tokens == 1)` was fragile:
- Chunked prefill can schedule exactly 1 token in the final chunk
- Speculative decoding can schedule >1 tokens for decode

We replaced it with `is_prefilling = num_computed_tokens < num_prompt_tokens`, which correctly handles both cases. Both arrays are already available in `InputBatch` as numpy arrays, so the check is zero-cost.

**Lesson**: Avoid heuristics that can be computed exactly. The token-count comparison is always correct; the n_tokens heuristic was a shortcut that became a liability.

### Scheduler Admission with Two Hashes

The scheduler must ensure a request's steering configs won't exceed `max_steering_configs` at any point during its lifetime. Since a request uses its prefill hash during prefill and decode hash during decode, both must fit.

The approach: count the union of all non-zero hashes. For running requests, include whichever hash is currently active. For a new request being admitted, count how many genuinely new unique hashes it would add. If `len(existing) + len(new_unique) > max`, skip the request.

This is conservative — a request's prefill hash might be released before its decode hash is needed, freeing a slot. But tracking temporal overlaps would add significant complexity for marginal gain when `max_steering_configs` is small.

**Lesson**: For admission control with phased resources, pessimistic union counting is simpler and safer than optimistic temporal analysis. The capacity loss is bounded by the number of phases (2).

### Global Steering Cache Invalidation

When global `vectors` or `prefill_vectors` change via the HTTP API, all prefix-cached blocks are potentially stale (they were computed with old global steering). Rather than encoding global state into every block hash (which would be expensive and complex), we call `reset_prefix_cache(reset_running_requests=True)` to:

1. Clear all cached blocks
2. Preempt all running requests (including mid-multi-chunk prefills)

This is conservative but correct. Global steering changes are rare (manual API calls), so the cost of recomputing is acceptable.

The `EngineClient` already exposes `reset_prefix_cache` as an async method, so the API router calls it directly — no new RPC mechanism needed.

**Lesson**: For rare events (global config changes), conservative invalidation (clear everything) is better than clever incremental invalidation (track what changed). The code complexity difference is enormous and the performance impact is negligible for infrequent operations.

### Additive Composition Test Subtlety

The additive composition test initially asserted: `base(250) + decode(250)` should equal `base(500)`. This fails because `steering_vectors` affects BOTH phases. So:
- `base(250) + decode(250)`: prefill=250, decode=500
- `base(500)`: prefill=500, decode=500

Different prefill steering → different KV cache → different decode output, even though the decode-phase steering is the same.

The correct test: use explicit `prefill_steering_vectors` and `decode_steering_vectors` to ensure both phases have the same effective vectors:
- Approach A: `prefill=200 + base=300`, `decode=base(300) + decode(200)` → (500, 500)
- Approach B: `prefill=500`, `decode=500` → (500, 500)

**Lesson**: When testing additive composition with a multi-phase system, you must account for ALL phases, not just the one you're testing. A steering field that says "both" really means both — changing it for the decode test also changes the prefill input.

---

## Testing

### What the tests cover

| Test | What it verifies |
|------|-----------------|
| `test_steering_op.py` (8 tests) | Indexed gather math: row selection, mixed indices, dtype preservation, oversized buffer slicing, in-place update visibility |
| `test_steering_types.py` (26 tests) | `SteeringVectorSpec` normalization, `resolve_effective_vectors` additive merge, `hash_steering_config` determinism, co-located scale handling |
| `test_steering_manager.py` (37 tests) | Phase-aware SteeringManager: 3-row reserved layout, phase-aware `get_row_for_config`, three-tier global population, phase registration/transition |
| `test_worker_steering.py` (36 tests) | Three-tier set/clear/status, validation, co-located scales, replace mode |
| `test_steering_scheduler.py` (28 tests) | Dual-hash admission control: union counting, phase-aware tracking, both hashes checked at admission |
| `test_steering_cache_keys.py` (6 tests) | Prefix cache key integration: prefill hash in extra keys, zero-overhead when unused, decode-only doesn't affect cache |
| `test_protocol.py` (10 tests) | Three-tier protocol, co-located scale format |
| `test_api_router.py` (32 tests) | HTTP API three-tier handling, validation, cache invalidation |
| `test_steering.py` (10 tests) | E2E GPU: global, per-request, CUDA graphs, prefill steering, decode steering, split phases, additive composition, prefix cache correctness, co-located scale, global prefill via worker API |

### Running tests

```bash
# All unit tests (~50s)
.venv/bin/python -m pytest tests/v1/test_steering_types.py tests/v1/worker/test_steering_manager.py tests/v1/core/test_steering_scheduler.py tests/v1/core/test_steering_cache_keys.py tests/entrypoints/serve/steering/ -v

# E2E integration tests (~4min, needs GPU)
.venv/bin/python -m pytest tests/models/language/generation/test_steering.py -v --timeout=300
```
