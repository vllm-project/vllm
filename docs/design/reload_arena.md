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
- runtime wrappers that own private graph-visible storage internally.

The arena does not make an arbitrary tensor safe merely by retaining it. The
consumer must use the tensor returned by the arena, and every rebuild path
must acquire the same slot.

## Weight updates and arena updates are different

Layerwise weight reload and arena refresh use intentionally different update
paths.

### Checkpoint weights use deferred staging and copy-back

Parameters and registered buffers participate in the layerwise reload
transaction:

```text
incoming checkpoint tensors
  → buffer weight-loader arguments for one layer
  → materialize a temporary checkpoint-format layer
  → replay the buffered weight loaders
  → run PWAL on the temporary state
  → copy processed parameters and buffers back to original kernel storage
```

The original parameter and buffer storage is retained in
`LayerReloadingInfo.kernel_tensors`. Incoming checkpoint values and PWAL
results can therefore be staged on temporary storage. Only after the layer
has loaded and processed its required weights does
`_copy_and_restore_kernel_tensors` publish the result into the storage already
used by serving kernels and captured graphs.

This delayed copy-back is possible because parameters and registered buffers
have an enumerable target set, and their processed values exist by the end of
the layer's PWAL call.

### Arena slots update their stable destination directly

Arena slots do not have a second staging/copy-back phase:

```text
PWAL or first forward
  → acquire the layer-owned slot
  → initialize or copy directly into that stable slot
  → rebuilt kernel/expert object uses the slot immediately
```

For a derived tensor, `arena.put()` performs `copy_` into the stable slot
during PWAL. For workspace, `arena.get_or_alloc()` returns the stable slot
itself. The arena is therefore the destination, not a temporary source whose
contents are committed later.

The ordering inside `_layerwise_process` is:

```text
replay buffered checkpoint loaders
  → process_weights_after_loading
      → arena writes happen here
  → copy processed parameters/buffers back to kernel_tensors
  → verify arena identities
```

Arena updates happen before parameter/buffer copy-back because PWAL both
computes arena values and constructs objects that must immediately bind to
the stable tensors returned by the arena.

### Why arena updates cannot use the weight staging mechanism

Arena-backed storage has more than one initialization lifetime:

1. **Eager derived state.** MLA and similar PWAL paths compute the complete
   runtime value during PWAL. `put()` must immediately refresh the stable
   slot and return it to the rebuilt attention or kernel object.
2. **Eager allocation with later writes.** Some PWAL paths allocate scratch
   storage, but its contents are not meaningful until a kernel invocation.
3. **Lazy allocation and initialization.** MoE implementations may construct
   the expert object during PWAL but allocate permute scratch only on the
   first real forward. The runtime call, not PWAL or reload finalization,
   determines when and how the memory is initialized.

Consequently, reload does not have a complete set of arena values that could
be held in a temporary buffer and copied back at one final commit point.
Some slots do not exist yet, some contain scratch with no persistent value,
and some must be returned to a newly constructed object before reload
finalization. Attempting to stage them like checkpoint weights would either:

- allocate a second address and let the rebuilt object retain the wrong one;
- require backend-specific knowledge of when scratch becomes initialized;
- copy uninitialized or semantically meaningless workspace contents;
- miss slots created lazily after PWAL.

The arena instead preserves the destination address continuously. Eager
values are copied into it when computed, and lazy users retain the arena so
their eventual allocation resolves to that same layer-owned destination.

This also means arena refresh is not an atomic value transaction. If a later
reload step fails, an arena slot may already contain a newly computed value.
The arena guarantees storage identity and provides a place to refresh values;
it does not provide shadow storage or rollback.

### Proposed staged publication for value-bearing slots

The direct-update rule above is the current implementation. A future version
can provide a stronger publication boundary by separating arena slots into two
categories, using the acquisition API as the declaration:

- `put()` declares value-bearing derived state whose contents must be
  refreshed from the newly loaded layer;
- `get_or_alloc()` declares workspace or scratch whose address matters but
  whose contents have no checkpoint-like value to commit.

Workspace and scratch should remain direct. Clearing, preserving, or lazily
initializing them is part of the consumer's runtime protocol, and copying a
shadow workspace back would add memory traffic without publishing meaningful
state.

Value-bearing `put()` slots can instead stage their newly derived contents in
a per-layer shadow:

```text
replay buffered checkpoint loaders for layer L
  → run PWAL for layer L
      → put(slot, value) records value in L's shadow
      → rebuilt consumers bind the existing live slot
  → publish processed parameters and buffers for L
  → copy L's value-bearing shadows into their live arena slots
  → verify L's arena identities
  → release L's shadows
```

The publication point should be the existing layer copy-back boundary, not a
single model-wide commit. Keeping shadows only for the layer currently being
processed bounds peak memory to that layer's value-bearing arena state. A
model-wide shadow would retain derived tensors for every layer until reload
completion and could require gigabytes for large models.

Staged `put()` has an important return-value constraint. It must still return
the **live** arena tensor so a rebuilt kernel, expert, or attention object
binds the address already captured by the accelerator graph:

```python
live = arena.put("W_UV", newly_derived)  # newly_derived is staged
kernel.W_UV = live                       # captured address remains stable
```

Until layer copy-back publishes the shadow, `live` contains the old value.
PWAL must therefore not use the tensor returned by a staged `put()` as the
source for another value derived in the same pass. Follow-on values must be
computed from the fresh local source:

```python
# Correct under staged put.
fresh_alpha = derive_alpha(loaded_scale)
live_alpha = arena.put("alpha", fresh_alpha)
live_c = arena.put("scale_c", derive_scale_c(fresh_alpha))

# Incorrect: live_alpha remains stale until layer publication.
live_alpha = arena.put("alpha", derive_alpha(loaded_scale))
live_c = arena.put("scale_c", derive_scale_c(live_alpha))
```

The layer transaction would need explicit arena operations analogous to:

```text
begin_layer_update()
publish_layer_update()
discard_layer_update()
```

`publish_layer_update()` copies staged values into existing live slots; it
must never rebind those slots. `discard_layer_update()` drops shadows when
PWAL or parameter copy-back fails before publication. This prevents a failed
layer from partially publishing value-bearing arena contents, but it is not a
model-wide rollback: layers published earlier in the reload may already hold
new parameters and arena values. Full model-level atomicity still requires a
shadow model or a worker recovery protocol.

Before enabling staged `put()`, every current caller must be audited for
same-pass data dependencies on the returned tensor. Tests must cover both the
pre-publication behavior (live value remains old) and the publication behavior
(pointer unchanged, new value visible), as well as failure before publication.
Until that audit and transaction API exist, `put()` continues to update the
live slot immediately as documented above.

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
- `put` immediately copies a recomputed value into the original storage.

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

All framework-owned PWAL entry points open this scope, including initial
loading, layerwise reload, attention finalization, attention-scale reload,
and dummy loading:

```python
with arena_scope(get_reload_arena(layer)):
    quant_method.process_weights_after_loading(layer)
```

This is an intentional PWAL boundary, rather than a statement that every
quantization method needs arena storage. A quantization method can construct
several layers of transient objects:

```text
quant_method.process_weights_after_loading(layer)
  -> select a kernel backend
  -> construct an expert or kernel object
  -> allocate graph-visible constants or scratch
```

The deepest constructor may know the runtime tensor but not the persistent
layer that owns it. Passing the layer or arena through every quantization,
kernel, and expert interface would couple those interfaces to reload. The
ambient scope instead makes the layer's arena available only while that
layer's PWAL call is running. Deep code can opt in with `current_arena()`.

Using the same boundary for every PWAL call is important for three reasons:

1. Initial load and reload execute backend selection through the same arena
   context. Initial PWAL creates the slot before graph capture; later PWAL
   reacquires it.
2. Correctness does not depend on a central list of quantization methods that
   happen to allocate graph-visible runtime tensors today. A newly introduced
   backend can use the existing boundary without changing every caller.
3. The scope has a precise lifetime. It is absent during normal inference, so
   one layer cannot accidentally acquire storage from another layer's PWAL.

Opening a scope does **not** automatically register tensors created during
PWAL. Registration remains explicit: deep code must call `current_arena()`
and publish through `put()` or `get_or_alloc()`. Methods that do not opt in
simply enter and leave the context.

The current boundary eagerly calls `get_reload_arena(layer)`, so a module
whose PWAL never uses the arena may retain an empty arena. This is a small
bookkeeping cost, not a device allocation: no tensor storage is allocated
until a slot is acquired. If empty arenas become significant, the boundary
can be changed to retain only an owner and lazily create its arena on the
first deep acquisition. Such an optimization must preserve the uniform PWAL
boundary and the initial-load/reload symmetry above.

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

### Third-party wrappers with private graph storage

Some runtime objects allocate graph-visible workspaces internally and do not
expose those tensors for `put()` or `get_or_alloc()`. For these narrowly scoped
runtime owners, the arena provides an object slot:

```python
rebuilt_backend._wrapper = arena.get_or_create_object(
    "backend.wrapper", build_wrapper
)
```

The first acquisition calls the factory and stores the result. Later
acquisitions return the same object without calling the factory again. Because
the arena belongs to the persistent layer, transient kernels or experts can be
rebuilt while continuing to borrow the original wrapper. Any private workspace
whose lifetime matches the wrapper's lifetime remains stable.

`FlashInferB12xExperts` uses this pattern for `B12xMoEWrapper`. The flow follows
the ownership fix demonstrated and validated on SM120 in
[vLLM PR #50538](https://github.com/vllm-project/vllm/pull/50538): create the
wrapper during the first PWAL, retain it on the persistent layer side, and bind
every rebuilt experts object to the retained wrapper. This implementation
stores it in the routed-experts arena as `flashinfer_b12x.wrapper`. Later PWAL
passes reacquire the same wrapper, preserving its private routing workspaces
and output buffer. The wrapper's tensor arguments are protected separately by
parameter copy-back or arena tensor slots.

Object slots deliberately do not accept a device or construction spec. They
follow the same in-place reload boundary as the owning layer: architecture,
backend, dtype, and device cannot change during a legal reload. For B12x, the
wrapper factory follows FlashInfer's default `device="cuda"`, which resolves to
the current device already selected for the worker/rank running PWAL.

This pattern is valid only when:

- the wrapper construction spec cannot change during a legal in-place reload;
- the wrapper does not retain the transient experts object;
- weight and scale addresses retained by the wrapper are independently stable;
- the third-party wrapper does not reallocate captured internal storage while
  the object remains alive.

Object slots participate in the same snapshot and verification gate as tensor
slots. A snapshot retains the original Python object, and verification reports:

| Violation | Object-slot meaning |
|---|---|
| `moved` | the slot now refers to a different Python object |
| `gone` | the object slot disappeared |

Holding the original object in the snapshot prevents Python identity reuse
from hiding a replacement during the reload window. New object slots created
after a snapshot remain legal, matching the existing rule for lazy tensor
slots. Verification proves the outer wrapper identity, not that third-party
code avoided reallocating storage internally while keeping the same wrapper.

Object slots are not a general cache for arbitrary modules or PWAL objects. If
any construction input can change, the backend must use a separate explicit
check and require a cold rebuild rather than silently reusing an incompatible
object.

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
unless its allocation path accepts arena-owned tensors or the object owning
that storage can itself persist on the layer. Reflective walking can detect
only some such state and cannot transfer ownership.

### Value correctness

Stable identity does not prove that contents were refreshed correctly:

- `put` refreshes values because it performs `copy_`;
- `get_or_alloc` only controls allocation and initialization policy;
- scratch users must overwrite or clear the required region;
- semantic correctness still requires kernel/model tests.

### Transaction rollback

The arena gate detects identity violations after mutation. Arena values may
already have been refreshed during PWAL, before parameter/buffer copy-back or
model-wide verification completes. The arena does not provide rollback,
shadow slots, or a shadow model. A strict failure prevents the reload from
being reported as successful, but callers must treat the worker as unsafe for
continued serving and recover according to the surrounding reload protocol.

`reload_storage_guard` also runs arena and global-manifest verification when
the reload body itself raises. Findings on that path are diagnostic: they are
logged without replacing the original load exception. The original exception
is annotated, when supported by the Python runtime, to state that mutation may
have started and the worker must be restarted. A clean identity check does not
make a failed reload safe, because parameters or value-bearing slots may
already have been partially updated without changing addresses.

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
