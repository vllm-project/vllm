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
introducing a single method named `refresh()`. This is not just a naming
difference: the split is the mechanism that makes the reload path safe for
runtime objects already captured by CUDA Graphs.

The intended correspondence is:

```text
modulewise reload:    PWAL + refresh()
trace reload:         re-entrant PWAL + non-re-entrant PWAL
```

The re-entrant part recomputes derived state from the newly loaded canonical
inputs or newly landed runtime weights and updates existing outputs in place.
The non-re-entrant part performs one-time construction or destructive layout
changes and must not be repeated against graph-visible runtime objects. In
other words, the semantic contract is the same even where the API name differs.

The implementation details are documented in
[Reload Flow Walkthrough](reload-flow-walkthrough.en.md) and
[Reload Loading Layout](reload-loading-layout.md).

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

## PWAL Refactoring in Trace Reload

### Cold PWAL: build the runtime representation

During cold loading, the ordinary
`process_weights_after_loading()` path is allowed to perform operations that
cannot be repeated safely:

- allocate or install the final runtime `Parameter` and buffer objects;
- choose the backend layout and create the processing plan;
- construct the kernel and its workspace/configuration;
- transpose, repack, shuffle, pad, or requantize checkpoint tensors;
- create derived tensors whose addresses will later be observed by inference;
- replace checkpoint-format parameters with their runtime-format versions.

After this phase, trace reload calls `bind_runtime()`. The tracer records the
final runtime objects, their storage addresses, shape/stride/dtype/device, and
the backend objects or processing plans needed for later reloads. The reload
round never tries to reverse the cold PWAL. It starts from fresh
checkpoint-format inputs and applies the forward conversion again.

This distinction is important for quantized backends. A runtime tensor may have
a different layout from its checkpoint input, and a scale or auxiliary tensor
may not have a checkpoint role at all. The runtime target is therefore bound
after cold PWAL, while the reload input is recreated from the cold-load
metadata. See the target and storage rules in
[Reload Loading Layout](reload-loading-layout.md).

### Re-entrant PWAL: repeat conversion without rebuilding runtime objects

For reload, each supported policy retains the cold-load processing plan or
kernel-specific conversion metadata. `policy.finish(state)` receives the
canonical checkpoint-format inputs accumulated for the current state and runs
the re-entrant conversion path. The result is written through
`ReloadTarget.copy_()` into the already-bound runtime target.

The important sequence is:

```text
checkpoint-format loader inputs
    -> state.work(role)
    -> re-entrant conversion / processing plan
    -> converted output tensors
    -> ReloadTarget.copy_()
    -> existing runtime storage
```

The reload path does not:

- call the cold installation helper again;
- create a new live kernel;
- replace a graph-visible `Parameter` or buffer;
- replace a derived tensor object that a kernel already references;
- infer a new expert placement from the converted tensor layout.

For example, the MoE processing plan can be reused to convert new canonical
weights, while the cold-only installation step remains disabled. The processing
plan may still allocate temporary conversion outputs, so “re-entrant” does not
mean “allocation-free” or “pure”. It means that the operation can be repeated
without rebuilding or replacing the runtime representation.

### Derived state and the `refresh()` correspondence

The proposal's `refresh()` is represented by the re-entrant portion of PWAL plus
the explicit derived-target writes performed by the policy. Derived outputs are
bound as `ReloadTarget`s even when they have no checkpoint role. They are
updated with `copy_()` after their source values are valid.

Examples include:

- per-tensor MoE alpha and reciprocal-scale values;
- block-scale clamping and backend scale metadata;
- MLA outputs such as `W_UV` and `W_UK`;
- packed or fused runtime targets;
- kernel/config values computed from newly landed weights.

This gives the same invariant as a named `refresh()` method:

```text
cold PWAL:
    allocate and initialize derived objects

reload:
    recompute their values
    copy_ into the existing derived objects
```

The distinction between a derived target and a conversion output is explicit:
the former must preserve its runtime identity across rounds; the latter may be
a temporary tensor that is copied into a stable target and then released.
`ReloadTarget.validate()` checks the identity and layout of the stable target
before and after the operation.

### Why this is different from rerunning the original PWAL

Rerunning the original cold-load method would be unsafe because many existing
PWAL implementations combine three different actions:

```text
read checkpoint tensors
    -> transform layout or quantize
    -> replace parameters / build kernels / cache derived values
```

The trace implementation separates those actions. The reload policy reuses the
forward transform but does not repeat object installation. This avoids the
double-permute and stale-off-registry-state failures described in the design
discussion. It also lets the same processing plan be exercised by cold-load
and reload tests without maintaining an independent inverse-transform path.

### Ordering guarantees

The re-entrant conversion runs only after all required slots for a state and
all explicit dependencies are complete. The high-level order is:

```text
validate arrival
    -> original loader writes canonical input
    -> mark slot arrived
    -> policy.finish(state)
    -> convert from state.work()
    -> copy converted and derived outputs into stable targets
    -> mark state complete
```

The model-level `trace.finish()` does not rerun these conversions. It checks
missing slots, target identity, backend invariants, placement, and completion,
then unwraps the loaders. The detailed state transition and model example are
in [Reload Flow Walkthrough](reload-flow-walkthrough.en.md).

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
split PWAL implementation. For this codebase, the stronger requirement is
behavioral: every derived tensor used after reload must be either a declared
stable target or a declared conversion output, and the appropriate one must be
updated in place or copied into place.

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
