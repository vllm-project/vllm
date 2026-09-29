# Weight Reload Abstraction

Status: design evolution record, including early proposals and later
implementation notes.

> For the current implementation, see
> [Reload Flow Walkthrough](reload-flow-walkthrough.md). It describes the
> current `reload_mode="trace"` path, including the call flow, object
> relationships, chunked examples, and source-reading order.
>
> Earlier sections that describe a whole-model commit are historical proposals.
> The current trace implementation completes states independently, permits
> in-place writes, and does not roll back a failed in-place update.

This document covers the runtime weight-update path under
`vllm/model_executor/model_loader/reload/` and its interaction with
quantization methods such as dense FP8 and CUTLASS FP8 MoE.

## 1. Design Goals

The following constraints take priority over implementation convenience:

1. **Complete weight loading.** A reload may take effect only after every
   planned target tensor has been fully and exactly written. Missing shards,
   duplicate shards, and shape or precision mismatches must be rejected.
2. **Stable runtime pointers.** Addresses captured by CUDA Graphs must remain
   byte-for-byte stable after reload. Do not replace `Parameter` objects,
   reallocate their storage, or rebuild graph-visible kernels. Writes must use
   `copy_` into existing storage. Parameter and kernel installation is a
   cold-load operation.
3. **Minimize staging.** If checkpoint and runtime representations are
   compatible, write in place. Allocate conversion buffers only for requant,
   block-scale rearrangement, expert fusion, or other real transforms.
4. **Use one verifiable state machine.** The reload abstraction must make
   arrivals, completion, dependencies, and errors explicit. The current trace
   implementation completes states independently; it does not promise
   whole-model rollback.

## 2. Core Abstraction

A reload round has five logical phases:

```text
DECLARE -> SOURCE -> VALIDATE -> COMMIT -> FINISH
```

| Phase | Responsibility | Allowed allocation |
| --- | --- | --- |
| DECLARE | Declare checkpoint names, expected shape/dtype, loader metadata, and runtime targets | Metadata only |
| SOURCE | Receive tensors from NCCL, IPC, RDT, or another transport and choose in-place or conversion storage | Conversion storage only |
| VALIDATE | Check coverage, duplicates, bounds, alignment, dtype, and shape | None |
| COMMIT | Convert when required and copy results into existing runtime storage | Temporary conversion buffers |
| FINISH | Refresh derived state and release per-round staging | None |

The key distinction is between errors detectable before a runtime write and
errors detectable only after the stream ends. Unknown, duplicate, malformed, or
misaligned arrivals should be rejected before writing. Missing shards discovered
at finish may leave in-place storage partially updated; that condition must
poison the engine and prevent further serving.

## 3. Runtime Targets and Storage

Every reloadable parameter, buffer, or derived tensor is represented by a
runtime target bound after cold load. Commit writes into that target without
replacing the object or its storage.

Storage policy is backend-specific but should prefer:

- **IN_PLACE** when the checkpoint-format input can safely use existing runtime
  storage;
- **CONVERT** when the input requires requantization, repacking, padding,
  transposition, or expert fusion.

A policy must be conservative when byte footprint, stride, storage offset,
alignment, or shared-storage exclusivity cannot be proven.

## 4. Arrival Integrity

Expected arrivals should be identified by logical role and shard identity, not
only by element count. For MoE, the identity normally includes:

- logical expert id;
- physical expert slot;
- shard or half, such as `w1`, `w2`, or `w3`;
- target role, such as `w13_weight` or `w2_scale`.

The original model `weight_loader` remains responsible for TP/EP slicing,
offsets, padding, fused mappings, and expert mappings. The reload tracer
records and validates the result; it must not duplicate those layout rules.

## 5. Re-entrant and Non-re-entrant PWAL

The current implementation expresses the usual `PWAL + refresh()` split by
separating post-load processing into two categories:

### Re-entrant processing

This processing may run during reload after new runtime values are available.
It must:

- read the newly loaded runtime weights;
- update derived tensors in place;
- preserve tensor and kernel object identity;
- be safe to run repeatedly;
- avoid destructive layout changes and replacement of graph-visible objects.

Examples include derived scales, reciprocal scales, alpha values, and other
small tensors cached by a kernel or quantization configuration.

### Non-re-entrant processing

This processing performs one-time or destructive work, such as:

- constructing a kernel object;
- changing a parameter layout;
- replacing a parameter;
- applying a transform that consumes checkpoint-format inputs;
- creating a runtime object whose address is captured by a CUDA Graph.

It must run only during cold load or in a controlled module-completion phase
where its outputs are explicitly copied into existing runtime targets.

This split is semantically equivalent to a separate `refresh()` API. The name
is less important than enforcing the lifecycle and identity guarantees.

## 6. ModelReloadTracer

`ModelReloadTracer` is the model-level coordinator. It owns the reload round,
routes arrivals, schedules state completion, reports errors, and manages
dependencies.

`ReloadState` represents one module or logical weight group. It contains:

```text
targets
slot tables
backend policy
dependencies
completion state
```

`ReloadPolicy` owns backend-specific behavior. It may implement FP8, Marlin,
DeepGEMM, CUTLASS, MoE, MLA, or other conversion logic, but it should not own
the global reload round.

The expected lifecycle is:

```text
cold load observation
    -> bind runtime targets and policies
    -> begin reload round
    -> route and validate arrivals
    -> write to checkpoint or conversion destinations
    -> finish ready states
    -> refresh derived state
    -> validate the complete round
```

## 7. Quantized Linear and Routed Experts

A quantized linear normally has one state with weight and scale roles. Its
policy decides whether to copy directly, stage and convert, or refresh derived
kernel metadata.

A routed-expert module also has one state, not one tracer per expert. Its slot
tables distinguish expert and shard identities. The policy builds the expected
slots from the current expert placement and must not require non-local experts
that this rank does not own.

For fused `w13`, `w1` and `w3` are separate logical arrivals even though they
share one runtime tensor. An expert is complete only after every required half
and scale has arrived.

## 8. Finish Scheduling

A state becomes ready only when its own required slots and all dependency states
are complete. The scheduler then calls its policy exactly once for the round.
After a successful finish, the state releases staging, records completion, and
notifies dependents.

The scheduler must ensure:

- a state finishes at most once per round;
- repeated completion notifications are harmless;
- failed states do not notify dependents as complete;
- dependency cycles are rejected at registration;
- reports include the missing dependency path, not only a parameter name.

## 9. Error Semantics

- Unknown, duplicate, malformed, or misaligned arrivals fail before the
  corresponding runtime write.
- Missing arrivals are reported at finish with state, role, expert, and shard
  identity.
- If in-place writes have already occurred when a missing arrival or transport
  failure is discovered, the runtime state is not considered recoverable.
  Serving must stop until a successful recovery or cold load.
- A staging conversion failure can preserve the old runtime target if no
  in-place write has occurred.
- An empty round is a no-op.
- Repeated successful finish calls must be idempotent.

## 10. Migration and Validation

The migration should proceed incrementally:

1. Keep `ModelReloadTracer`, `ReloadState`, `ReloadTarget`, slot tables, arrival
   validation, and missing reporting as the common infrastructure.
2. Add builders and policies for ordinary parameters, fused QKV, quantized
   linear layers, routed experts, FP8, Marlin, DeepGEMM, and MLA.
3. Compare tracer observations with the existing loader observation path.
4. Validate ordinary weights, fused weights, MoE experts, EPLB mappings,
   weight/scale completion, and derived-state refresh.
5. Move transport integrations to the common lifecycle.
6. Add worker-level dirty and rank-wide recovery semantics.
7. Remove duplicate state sources only after all required transport and backend
   coverage is validated.

The final division of responsibility should remain:

```text
ModelReloadTracer -> round lifecycle, routing, scheduling, error reporting
ReloadState       -> targets, slots, completion state
ReloadPolicy      -> backend-specific arrival and finish behavior
loader adapter    -> normalized arrival information
weight_loader     -> model-specific mapping and physical writes
```

The full historical validation log and implementation notes remain in the
Chinese source document:
[`weight-reload-abstraction.md`](weight-reload-abstraction.md).
