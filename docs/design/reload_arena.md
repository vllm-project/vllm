# ReloadArena

## Overview

`ReloadArena` preserves the storage identity of runtime tensors that remain
visible to an accelerator graph across an in-place weight reload.

An in-place reload changes model values without rebuilding the serving
process. During the reload,
`process_weights_after_loading` (PWAL) may rebuild quantization methods,
kernel objects, expert objects, workspaces, or derived tensors. Replacing one
of those tensors is unsafe when a previously captured graph still contains
its device address:

```text
graph capture                       reload
-------------                       ------
kernel records address A            PWAL creates tensor at address B
graph keeps address A               Python state now points to B
```

The old graph may then read stale values from address A, reference freed
storage, or fail with an illegal memory access.

The arena changes the storage owner. Storage belongs to the persistent layer,
not to a transient kernel or quantization object:

```text
layer
  └── ReloadArena
        └── named stable storage
              ↑
       transient objects borrow it
```

PWAL may rebuild transient objects, but acquiring the same arena slot returns
the same storage. Newly derived values are copied into that storage rather
than published by rebinding a tensor.

## Scope

The arena protects tensors that satisfy all of the following:

1. their device address can outlive the Python object that created them;
2. they are owned logically by one persistent `nn.Module`;
3. their shape, dtype, device, and layout remain invariant across an in-place
   reload;
4. they are not checkpoint state and should not participate in parameter or
   buffer loading.

Examples include:

- kernel workspaces and scratch buffers;
- activation permutations derived during PWAL;
- quantization scales or transformed values derived from loaded weights;
- tensors stored in transient kernel, expert, or quantization objects;
- buffers allocated lazily on the first forward and reused by later forwards.

The arena does not make an arbitrary tensor safe merely by retaining it. The
consumer must use the tensor returned by the arena, and every rebuild path
must acquire the same slot.

## Lifecycle

### Initial model load

During initial PWAL, graph-visible runtime tensors are published through the
layer's arena:

```text
load checkpoint weights
  → process_weights_after_loading
  → acquire or publish arena slots
  → warm up model
  → capture accelerator graphs
```

The graph therefore captures addresses owned by the layer's arena.

For lazily allocated storage, a constructor must retain the arena selected
during PWAL. The first forward may create the slot, after which the slot is
owned by the layer for the remainder of its lifetime.

### Reload

`GPUModelRunner.reload_weights` takes a model-wide arena snapshot before
mutation:

```python
arena_snaps = snapshot_model_arenas(model)
```

For checkpoint-format layerwise reload, each layer also snapshots its arena
before the layer is restored to meta:

```text
save kernel tensors
snapshot layer arena
restore checkpoint state to meta
buffer and load checkpoint tensors
rerun PWAL
copy processed state back
verify layer arena
```

Arena tensors are deliberately not restored to meta. They are runtime state,
not checkpoint state, and their storage must remain alive while PWAL is
rebuilt.

PWAL reacquires the existing slots:

- `get_or_alloc` returns the original storage;
- `put` copies a recomputed value into the original storage.

After reload, the model-wide commit gate verifies all slots captured before
mutation:

```python
problems = verify_model_arenas(model, arena_snaps)
```

A moved, missing, or respecified slot fails closed by default. This check is
necessary even for an identity reload: output equality cannot detect a graph
that still reads an old allocation containing the same values.

### New slots

A slot created after a snapshot is allowed. This is required for legitimate
first-forward lazy allocation. Once that slot exists, a subsequent reload
snapshot includes it and requires it to remain stable.

## API

### Obtaining an arena

Use the layer that logically owns the runtime tensor:

```python
from vllm.model_executor.reload_arena import get_reload_arena

arena = get_reload_arena(layer)
```

The arena is stored as the plain attribute `_reload_arena`. It is
intentionally not an `nn.Module`, and its tensors are intentionally not
registered as parameters or buffers. This prevents checkpoint loading,
state-dict handling, meta restoration, and copy-back logic from treating
runtime storage as model state.

Use `peek_reload_arena(layer)` when inspection must not create an arena.

### Publishing a derived value

Use `put` when PWAL recomputes a value:

```python
perm = torch.argsort(layer.g_idx).to(torch.int)
layer.act_perm = get_reload_arena(layer).put("act_perm", perm)
```

On first use, `put` stores a detached, contiguous clone. On later calls, it
copies the new value into the existing slot and returns the existing tensor.

Always bind the consumer to the return value:

```python
# Correct
layer.act_perm = arena.put("act_perm", recomputed)

# Incorrect: the arena is populated, but the consumer uses transient storage.
arena.put("act_perm", recomputed)
layer.act_perm = recomputed
```

`put` provides both:

- stable storage identity;
- refreshed derived values after every PWAL run.

### Allocating workspace or scratch

Use `get_or_alloc` when only stable storage is required:

```python
from vllm.model_executor.reload_arena import InitPolicy

layer.workspace = get_reload_arena(layer).get_or_alloc(
    "decode_workspace",
    shape=(max_tokens, hidden_size),
    dtype=layer.weight.dtype,
    device=layer.weight.device,
    init=InitPolicy.PRESERVE,
)
```

Initialization policies are:

| Policy | First acquisition | Later acquisition |
|---|---|---|
| `EMPTY` | `torch.empty` | preserve existing contents |
| `PRESERVE` | `torch.empty` | preserve existing contents |
| `ZERO` | `torch.zeros` | zero the existing tensor |

`EMPTY` and `PRESERVE` currently have the same implementation. Use
`PRESERVE` when retained contents are intentional or the caller overwrites
the storage before reading it. Use `ZERO` only when every acquisition must
clear the workspace.

The arena does not enforce use-before-initialization. A caller using
`EMPTY` or `PRESERVE` must fully initialize the relevant region before a
kernel reads it.

### Deep construction chains

Some kernel or expert constructors cannot receive the layer directly. Open an
ambient scope at the PWAL site:

```python
from vllm.model_executor.reload_arena import (
    arena_scope,
    get_reload_arena,
)

with arena_scope(get_reload_arena(layer)):
    kernel = make_moe_kernel(...)
```

The constructor may resolve and retain the current arena:

```python
from vllm.model_executor.reload_arena import current_arena

class Experts:
    def __init__(self):
        self._reload_arena = current_arena()
```

For a lazy first-forward allocation, use the retained arena:

```python
if self._scratch is None:
    if self._reload_arena is None:
        self._scratch = torch.empty(spec)
    else:
        self._scratch = self._reload_arena.get_or_alloc(
            "experts_scratch", shape, dtype, device
        )
```

Do not call `current_arena()` for the first time during forward. The scope is
opened during construction/PWAL and is normally absent during inference.

`arena_scope` uses a `ContextVar`, so nested construction restores the prior
scope and concurrent contexts do not share a process-global mutable pointer.

## Slot contract

A slot name identifies one stable storage specification for the lifetime of
its owning layer.

Later acquisitions must use the same:

- shape;
- dtype;
- device.

Verification additionally protects:

- `data_ptr`;
- stride/layout.

An incompatible acquisition raises instead of reallocating:

```text
A shape-changing reload is not an in-place update; it requires a cold restart.
```

Unindexed accelerator devices such as `"cuda"` are canonicalized to the
device on which PyTorch would allocate, such as `cuda:0`. CPU remains
unindexed. This avoids false mismatches while preserving device-specific
ownership.

Slot names must be:

- stable across initial load and reload;
- unique within the owning layer;
- semantic rather than based on object identity or allocation order.

Good names:

```text
machete_act_perm
rdna3_w1_buf
cutlass_ab_strides1
```

Unsafe names:

```text
slot_{id(kernel)}
workspace_{allocation_counter}
```

## Verification and failure policy

`snapshot()` records each slot's pointer and layout. `verify()` reports:

| Violation | Meaning |
|---|---|
| `moved` | the slot now has a different `data_ptr` |
| `gone` | a previously snapshotted slot disappeared |
| `respecified` | shape, stride, or dtype changed |

Model-wide verification is authoritative. Per-layer verification runs closer
to PWAL and is compared against the model-wide result to detect coverage
gaps.

`VLLM_RELOAD_GATE` controls the model-wide gate:

| Value | Behavior |
|---|---|
| `strict` or unset | raise and refuse to serve the reload |
| `warn` | log the violation and continue, which is unsafe |
| `off` | suppress the gate, which is unsafe |

Production use should retain the default strict behavior.

## Relationship to the global storage manifest

Not every graph-visible tensor has a layer owner. Module-level caches and
registries are outside the model walk:

```python
_workspace_by_device: dict[torch.device, torch.Tensor]
```

These are covered separately by
`vllm.model_executor.reload_manifest.GlobalStorageManifest`.

Use:

- `ReloadArena` when a persistent layer can own and reuse the storage;
- the global manifest to detect movement of module-level state that cannot be
  placed in a layer arena.

The global manifest detects and reports movement but does not stabilize
storage. It defaults to warning through
`VLLM_RELOAD_GLOBAL_MANIFEST=warn`; it is not a replacement for arena
ownership.

## Boundaries and non-goals

### Shape-changing updates

The arena supports in-place value updates, not architecture changes. Changes
to hidden size, expert layout, quantization layout, tensor shape, dtype,
device, or stride require a cold model rebuild and graph recapture.

### Parameter and buffer identity

The arena is for non-checkpoint runtime storage. Parameters and registered
buffers remain the responsibility of the model loader and layerwise copy-back
flow. Moving ordinary model state into the arena to bypass those mechanisms
is unsupported.

### Arbitrary Python aliases and closures

The arena cannot repair a consumer that retains the wrong tensor object. For
example, a callable that closes over a transient PWAL tensor continues using
that object even if an arena slot with the same value exists.

Resolve arena-backed tensors from the persistent layer at call time, or bind
the returned arena tensor into the rebuilt object.

### Opaque external allocations

Storage allocated internally by a third-party extension cannot be stabilized
unless its allocation path accepts or reuses arena-owned tensors. Reflective
walking can detect only some such state and cannot transfer ownership.

### Value correctness

Stable identity does not prove that contents were refreshed correctly:

- `put` refreshes values because it performs `copy_`;
- `get_or_alloc` only controls allocation and initialization policy;
- scratch users must overwrite or clear the required region;
- semantic correctness still requires kernel/model tests.

### Transaction rollback

The arena gate detects identity violations after mutation. It does not provide
rollback or a shadow model. A strict failure prevents the reload from being
reported as successful, but callers must treat the worker as unsafe for
continued serving and recover according to the surrounding reload protocol.

### Graph visibility discovery

The arena is an explicit declaration mechanism, not an automatic proof that
every graph-visible tensor has been registered. CI discovery and backend
audits are still required to find new workspaces, lazy caches, closure-held
tensors, and third-party allocations.

## Integration checklist

When adding an arena-backed tensor:

1. Confirm that a graph or long-lived kernel can retain its device address.
2. Select the persistent layer that logically owns it.
3. Choose `put` for recomputed values or `get_or_alloc` for workspace.
4. Define a stable, layer-local slot name.
5. Rebind every consumer to the tensor returned by the arena.
6. Ensure every PWAL/backend branch reacquires the same slot.
7. For lazy allocation, capture the arena during construction rather than
   resolving the ambient scope during forward.
8. Verify that shape, dtype, device, and layout cannot change during an
   in-place reload.
9. Test at least two PWAL/reload passes and assert stable `data_ptr`.
10. Test that values are refreshed or scratch is initialized as required.
11. Exercise the real graph-capture backend when possible; CPU identity tests
    alone do not establish graph visibility.

## Testing

Focused tests live under:

```text
tests/model_executor/model_loader/test_reload_arena.py
tests/model_executor/model_loader/test_reload_arena_perlayer.py
tests/model_executor/model_loader/test_reload_lazy_storage.py
tests/model_executor/model_loader/test_reload_manifest.py
tests/model_executor/model_loader/test_reload_rdna3_buffers.py
tests/model_executor/model_loader/test_post_load_storage_stability.py
tests/model_executor/model_loader/test_moe_experts_storage_stability.py
```

Tests should distinguish:

- pointer stability from value equality;
- initial allocation from reacquisition;
- direct layer access from scoped deep construction;
- PWAL allocation from first-forward lazy allocation;
- layer-owned arena state from module-level manifest state;
- expected cold-start slot growth from illegal movement of existing slots.
