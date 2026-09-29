# Response to the Modulewise Reload Design

Thank you for the detailed modulewise reload proposal. We agree with the
problem statement and with the main architectural direction: the reload path
should preserve runtime tensor addresses, reuse the model's existing
`weight_loader` mapping logic, avoid retaining non-local expert inputs, and
separate checkpoint-to-kernel transforms from state derived from live runtime
weights.

Our current `reload_mode="trace"` implementation already follows most of this
direction:

- `ModelReloadTracer` owns the reload round and routes arrivals to
  per-module `ReloadState` objects.
- `ReloadState` records expected and arrived slots, including fused and expert
  shard identities.
- The original `weight_loader` remains responsible for TP/EP slicing, fused
  shard offsets, padding, and expert mapping.
- A module prepares its checkpoint-format destination on first accepted input.
- Runtime targets are bound after cold load and are updated through `copy_`
  without replacing the `Parameter`, buffer, or kernel-visible storage.
- Backend-specific policies perform FP8, MoE, MLA, and other conversions.
- Missing, duplicate, unknown, and invalid arrivals fail the reload round
  instead of silently completing with partial input.

There is one terminology difference from the proposal. We have refactored
post-load processing into re-entrant and non-re-entrant parts rather than
introducing a single method named `refresh()`.

The intended correspondence is:

```text
modulewise reload:    PWAL + refresh()
trace reload:         re-entrant PWAL + non-re-entrant PWAL
```

The re-entrant part recomputes derived state from the newly landed runtime
weights and must update existing tensors in place. The non-re-entrant part
performs one-time construction or destructive layout changes and must not be
repeated against graph-visible runtime objects. In other words, the semantic
contract is the same even where the API name differs.

## Points of Agreement

### 1. Preserve runtime identity

The runtime `Parameter` objects, storage addresses, and kernel-visible tensors
must survive a reload. A reload may write new values, but it must not rebuild
the runtime object graph or silently move storage.

### 2. Keep loader semantics in the loader

The tracer should not reimplement model-specific TP, EP, padding, fused-QKV,
or expert mapping rules. Those rules already live in `weight_loader`; the
reload layer should observe and validate them.

### 3. Validate arrivals explicitly

Element counts alone are insufficient for fused parameters, padding, and MoE
experts. Expected slots should be represented by logical shard identity,
expert identity, and role. A missing shard and a duplicate shard must be
reported as different errors.

### 4. Reuse storage when it is safe

Checkpoint-format input may use existing runtime storage when dtype, footprint,
layout, and aliasing rules permit it. Conversion-required inputs should use
staging and release it after the module policy completes.

### 5. Treat failure after an in-place write as terminal

The current design intentionally does not promise rollback after runtime
storage has been modified. A failed round must poison the reload session and
prevent serving until a complete recovery or cold load has occurred.

## Differences and Follow-up Work

### Re-entrant versus `refresh()`

We should document the re-entrant PWAL split as the implementation of the
proposal's `refresh()` semantics. The important requirements are:

1. re-entrant code only updates existing derived storage;
2. it does not replace graph-visible parameters or kernel objects;
3. non-re-entrant transforms run only at the appropriate cold-load or
   module-completion point;
4. derived tensors are refreshed after their source runtime values are valid;
5. backend tests compare repeated reload against a fresh cold load.

An explicit `refresh()` method can still be added as a naming and discovery
convention, but it is not required if the same contract is enforced by the
split PWAL implementation.

### Direct-storage validation

The trace path already supports storage reuse, but the generic safety checks
should continue to be strengthened around byte footprint, storage offset,
alignment, and shared-storage exclusivity. Backend policies may conservatively
fall back to staging when those facts cannot be proven.

### Cross-rank failure handling

The tracer validates local state and poisons a failed round. A complete
deployment contract still needs an explicit worker-level dirty/serve guard and
rank-wide agreement before serving resumes. This is separate from backend
conversion and should be owned by the weight-transfer lifecycle.

### RDT integration

Sharded RDT should use the same public trace concepts for materialization,
completion, and abort rather than reaching into legacy reload bookkeeping.
That migration is needed to ensure that RDT and NCCL/IPC share the same slot,
landing, derived-state, and failure semantics.

## Suggested Integration Order

1. Keep the current trace state machine and slot validation as the base.
2. Document re-entrant/non-re-entrant PWAL as the `refresh()` equivalent.
3. Complete backend coverage and refit-versus-fresh tests for derived state.
4. Strengthen generic direct-storage validation and staging fallback.
5. Add worker-level dirty and rank-wide recovery semantics.
6. Move sharded RDT to the same public trace lifecycle.
7. Add per-expert online-quantization completion only where the backend can
   prove independent expert conversion and collective-safe ordering.

Overall, we view the proposal and the current trace implementation as
compatible in architecture. The main remaining work is to make the lifecycle
contract uniform across all transports and backends, rather than to introduce
another independent reload implementation.
