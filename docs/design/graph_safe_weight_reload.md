# Graph-safe weight reload

This document describes the sealed reload plan used by Model Runner V2 to
update model weights without invalidating an existing CUDA or HIP graph.

## Problem

A weight update is not always a direct copy into an `nn.Parameter`. Backends
often derive additional runtime tensors after loading a checkpoint:

- transposed or split projections;
- packed or shuffled kernel weights;
- converted attention sinks;
- quantization scales;
- strides, workspaces, and backend descriptors.

A graph records the addresses used during capture. Rebinding one of these
tensors updates the Python model but leaves the graph reading its old storage.
Keeping the address stable is also insufficient when a derived tensor or a
runtime cache still contains values from the previous model version.

Weight reload therefore has four independent correctness requirements:

1. **Storage identity:** every graph-visible stable slot keeps its original
   storage and layout.
2. **Derived freshness:** every affected derived slot is refreshed in the
   current reload epoch.
3. **Loader lifecycle:** load, post-load processing, validation, and commit run
   exactly once and in order.
4. **Runtime state:** caches and backend state are refreshed, invalidated, or
   explicitly proven safe to preserve.

## Scope and assumptions

The current implementation is intentionally limited to Model Runner V2.
Model Runner V1 does not participate in this contract.

Reload starts only after inference has been paused and in-flight requests have
been drained. A successful update reuses the existing graph. The implementation
does not automatically recapture a graph, keep a shadow model, or roll back a
partially written model.

An unsupported backend fails before the first model mutation. After a failure
during loading or validation, the reload plan remains failed and the worker
must be restarted or fully reloaded before serving resumes.

## Architecture

The implementation has three layers:

```text
LLM / AsyncLLM
  prepare every worker
  commit only after every prepare succeeds
                |
                v
Worker weight-transfer lifecycle
  begin -> receive -> finalize -> prepare -> commit
                |
                v
Model Runner V2 ReloadCoordinator
  ReloadPlan + one active ReloadTransaction
```

The transport is not the owner of reload correctness. Disk, NCCL, IPC, and
future transports provide inputs to the same transaction. `ReloadCoordinator`
owns the plan, active epoch, state policy, and commit gate.

## Reload plan

`ReloadPlan` is compiled after initial model loading and before compilation or
graph capture. It contains the following objects.

### Direct inputs

Parameters and buffers whose checkpoint and runtime layouts are identical are
direct inputs. A transaction updates them with `copy_`; rebinding the parameter
or changing its view is rejected during validation.

### Stable tensor slots

Every direct or derived graph-visible tensor has a stable slot. A slot retains
a strong reference to its initial tensor and records:

- storage identity;
- data pointer and storage size;
- shape, stride, and storage offset;
- dtype and device.

Comparing only `data_ptr` is insufficient because an allocator can reuse an
address. Comparing only storage identity is also insufficient because a view
can change its offset or layout.

### Derived nodes

A reload participant declares an explicit dependency and output set:

```python
def build_reload_plan(self, builder, prefix):
    builder.derived(
        f"{prefix}.runtime_projection",
        owner=self,
        owner_key="runtime_projection",
        outputs=("runtime_weight",),
        depends_on=(f"{prefix}.weight",),
    )
```

Post-load processing publishes a newly computed value with
`refresh_reload_derived`. During initial loading this binds the attribute.
During reload it copies into the original stable slot and marks the node with
the active epoch.

```python
refresh_reload_derived(
    self,
    "runtime_projection",
    {"runtime_weight": transformed_weight},
)
```

Commit rejects a required node whose `last_executed_epoch` is stale. This
catches a missing post-load refresh even when the old and new tensors have
identical shapes, strides, and values in a same-weight test.

### Runtime state

Non-checkpoint state uses one of three policies:

| Policy | Meaning |
| --- | --- |
| `REFRESH` | Recompute state for the new model version. |
| `INVALIDATE` | Clear state before publishing the new epoch. |
| `PRESERVE` | Keep state only when its validator proves it version-independent. |

Model Runner V2 currently invalidates encoder and multimodal caches as part of
the transaction.

### Capabilities

A plan has one of three capabilities:

| Capability | Behavior |
| --- | --- |
| `GRAPH_SAFE_V1` | Reload is allowed with a live graph. |
| `EAGER_ONLY` | Reload is numerically complete but graph storage is not certified. |
| `UNSUPPORTED` | Reload is rejected in every execution mode. |

Quantization backends that have not declared a complete reload plan are
currently `UNSUPPORTED`. This is deliberate: derived-value staleness can
silently corrupt eager execution as well as graph replay.

## Transaction lifecycle

A transaction has the following state machine:

```text
LOADING -> FINALIZING -> VALIDATING -> PREPARED -> COMMITTED
    \           \             \            \
     +-----------+-------------+-------------> FAILED
```

- `begin` verifies capability before mutation and assigns a new epoch.
- `write` validates a direct input and copies it into its stable slot.
- `FINALIZING` permits post-load participants to refresh derived slots.
- `prepare` checks input completeness, derived epochs, state policies, and all
  storage fingerprints without publishing the epoch.
- `commit` publishes the epoch only after every worker prepared successfully.

There is no transition from `FAILED` back to an active serving state.

## Distributed commit

`LLM.finish_weight_update` and `AsyncLLM.finish_weight_update` perform two
collective calls:

1. `prepare_weight_update` finalizes and validates every worker.
2. `commit_weight_update` runs only if the first collective succeeded on every
   worker.

If any prepare fails, the coordinator broadcasts `abort_weight_update` and no
worker publishes the new epoch. Commit itself only publishes already validated
local state; model mutation and backend finalization happen during prepare.

## Initial participants

The first participants cover known high-risk attention paths:

- MLA projections derived from `kv_b_proj`, including `W_UK_T`, `W_UV`, and
  AITER FP4/FP8 projection tensors;
- FlashInfer attention sinks converted to FP32 after loading.

FlashInfer retains the original checkpoint source separately from the runtime
FP32 slot. Every reload derives the runtime value from the updated source and
copies it into the storage captured by the graph.

Additional quantization, MoE, and opaque C++ backends must implement the same
participant contract before their capability can be upgraded from
`UNSUPPORTED`.

## Failure examples

### Storage replacement

```python
layer.runtime_weight = transform(layer.weight)
```

If this bypasses `refresh_reload_derived`, storage validation reports the slot
name and its expected and actual fingerprint before inference resumes.

### Missing derived refresh

If the source parameter was written but its required derived node did not run,
commit raises `StaleDerivedSlotError` with the node and epoch.

### Incomplete input set

For a reload with a declared input manifest, commit reports the missing logical
inputs rather than publishing a partially updated model.

### Stale runtime cache

A state node without its required action fails plan compilation. A preserve
validator that returns false fails transaction preparation.

## Testing strategy

CPU contract tests cover:

- direct in-place writes;
- parameter and derived-slot rebinding;
- missing and duplicate inputs;
- stale derived epochs;
- transaction ordering and permanent failure;
- state invalidation and preserve validation;
- eager-only and unsupported capability gates;
- prepare without premature epoch publication.

V2 integration tests cover coordinator reload, cache invalidation, rejection
before mutation, Worker lifecycle, and sync/async two-phase commit. A focused
FlashInfer test verifies that updated BF16 source values reach the original
FP32 runtime storage.

Hardware CI should add differential output checks for every backend promoted to
`GRAPH_SAFE_V1`: capture once, reload different weights, replay the existing
graph, and compare logits with a fresh eager model.
