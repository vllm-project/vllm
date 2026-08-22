# Sleep Mode Tensor Ownership and Recovery

## Status

Proposed.

This RFC describes a correctness issue in the current sleep-mode allocation
boundaries and proposes ownership and recovery rules for tensors that outlive
model and KV-cache initialization. It is motivated by a similar issue found in
vLLM Ascend ([vllm-ascend PR #13352](https://github.com/vllm-project/vllm-ascend/pull/13352)).

## Summary

vLLM sleep mode places model initialization and KV-cache initialization inside
tagged device memory pools. The allocator records a tag for each underlying
allocation, not the semantic role of every tensor backed by that allocation.
Consequently, every device tensor created while a tagged pool is active inherits
the sleep policy for that tag, even if the tensor is actually persistent runtime
metadata, a kernel constant, a synchronization buffer, or derived model state.

This is mostly benign for level-1 sleep and model weights because all allocations
tagged `weights` are copied to host memory. It is unsafe for allocations tagged
`kv_cache`, which are intentionally discarded. On wake-up, the same virtual
addresses are mapped to new physical memory, but discarded contents are not
restored. A CUDA graph can therefore replay successfully using stable pointers
while reading corrupted constants or indices.

Level-2 sleep has an additional variant of the problem. Parameters are expected
to be reloaded and registered model buffers are explicitly saved and restored,
but plain tensor attributes are neither reloaded nor restored. A level-2
sleep/wake/reload cycle can therefore leave model constants, kernel metadata, or
synchronization workspaces corrupted.

This RFC proposes:

1. Restrict the `kv_cache` memory-pool scope to actual discardable cache backing
   allocations.
2. Define explicit ownership and recovery requirements for every persistent
   device tensor.
3. Represent static model tensors as registered buffers or host/Python values,
   rebuild derived weights after reload, and explicitly reset stateful kernel
   workspaces.
4. Add deterministic remap poisoning tests so fresh zero-filled pages cannot
   hide missing recovery logic.

## Motivation

### Current allocation scopes

Model loading currently runs as one `weights` allocation scope in
[`GPUWorker.load_model`](../../vllm/v1/worker/gpu_worker.py). KV-cache
initialization similarly runs as one `kv_cache` allocation scope:

```python
with self._maybe_get_memory_pool_context(tag="kv_cache"):
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

`initialize_kv_cache` performs substantially more work than allocating KV-cache
storage. Depending on the model runner and backend, it can also construct:

- attention metadata builders;
- persistent block-table and scheduler metadata;
- speculative-decoding and adaptive-verification state;
- Mamba or GDN state-cache metadata;
- connector state;
- CUDA-graph-safe constants and index buffers.

Those objects survive initialization and are read on later model executions.
Their tensors nevertheless receive the `kv_cache` tag solely because of when
they were allocated.

### Allocator behavior

[`CuMemAllocator`](../../vllm/device_allocator/cumem.py) records each allocation
handle, its virtual address, and the allocator's current tag. Sleep behavior is
selected per allocation:

| Sleep mode | `weights` allocations | `kv_cache` allocations |
| --- | --- | --- |
| Level 1 | Copy to host, then unmap and release | Unmap and release without backup |
| Level 2 | Unmap and release without backup | Unmap and release without backup |

Wake-up recreates physical memory at the original virtual address. If an
allocation has a host backup, vLLM copies it back. Otherwise, the allocation has
valid storage and a stable address but no preserved contents.

The distinction between virtual-address stability and value preservation is
important for CUDA graphs. Stable addresses keep captured pointer arguments
valid; they do not restore a scale, index sequence, sentinel, or lock value at
that address.

### Observable failure

A minimal example is a persistent attention scale created in the KV-cache
scope:

```python
with allocator.use_memory_pool(tag="kv_cache"):
    kv_cache = torch.empty(cache_shape, device="cuda")
    scale = torch.tensor([3.0], device="cuda")
```

After level-1 sleep and wake-up:

- `kv_cache.data_ptr()` is unchanged and its old contents are intentionally
  gone;
- `scale.data_ptr()` is also unchanged;
- the value `3.0` is gone even though `scale` is not KV cache;
- a captured graph can replay without an invalid-pointer error but calculate a
  different result.

The likely manifestations depend on the tensor:

- zero or incorrect attention output from a corrupted scale;
- incorrect block lookup from a corrupted `arange` or block-table prefix;
- incorrect sparse-attention bounds or top-k selection;
- a hang or intermittent failure from a corrupted lock or synchronization
  sentinel;
- model-specific accuracy loss from a corrupted embedding or preprocessing
  constant.

## Why This Has Not Caused Widespread Accuracy Failures

Several existing properties limit exposure without making the ownership model
safe:

1. Level-1 sleep preserves every allocation tagged `weights`, including tensors
   that were accidentally included in the scope.
2. Actual KV-cache contents are intentionally disposable. Sleep aborts or drains
   requests and clears request-dependent caches before new work is accepted.
3. Most persistent attention metadata is a workspace that is fully overwritten
   for each batch before it is read.
4. Common configurations use backends that overwrite their metadata, while many
   confirmed nonzero constants occur in optional RSWA, sparse MLA, ROCm Aiter,
   XPU, or model-specific paths.
5. Fresh physical memory is commonly observed as zero-filled. This turns the
   problem into feature-specific deterministic corruption rather than arbitrary
   process-wide corruption, and can make zero-initialized workspaces appear
   correct by accident.
6. Existing wake-up hooks already repair FP8 KV-cache scales and MRV2 block-table
   layout tensors.
7. Current end-to-end sleep tests primarily use small dense-attention models and
   therefore do not exercise the affected metadata.

These factors explain the limited blast radius but are not correctness
guarantees.

## Confirmed Risk Inventory

The following inventory is based on a static scan of vLLM commit
`41f179b57aa8ab6f634f508128ce1f1efadd0eb1`. A tensor is listed as confirmed when
it is retained after its allocation scope and a consumer can read it without a
complete preceding rewrite.

### KV-cache scope

| Component | Persistent tensor | Failure mode |
| --- | --- | --- |
| Static Sink Attention | fixed sink block IDs in `block_table_with_sink` | wrong cache blocks selected |
| FlashAttention R-SWA | `persistent_rswa_window_tensor` | wrong recurrent sliding-window bound |
| FlashMLA Sparse | `topk_tokens_tensor`, `max_model_len_tensor` | wrong sparse selection bounds |
| DSA indexer | `offsets_buffer`, `arange_buffer` | wrong decode expansion and padding indices |
| ROCm Aiter FlashAttention | `scale` | wrong K/V scaling |
| ROCm Aiter MLA | `paged_kv_last_page_len` initialized to ones | invalid page metadata |
| ROCm Aiter Sparse MLA | `max_model_len_tensor`, `qo_indptr`, `paged_kv_last_page_len` | invalid sparse decode metadata |
| XPU Sparse MLA | `topk_tokens_tensor`, `max_model_len_tensor` | wrong sparse selection bounds |

Examples can be found in:

- [`static_sink_attention.py`](../../vllm/model_executor/layers/attention/static_sink_attention.py)
- [`flash_attn.py`](../../vllm/v1/attention/backends/flash_attn.py)
- [`flashmla_sparse.py`](../../vllm/v1/attention/backends/mla/flashmla_sparse.py)
- [`indexer.py`](../../vllm/v1/attention/backends/mla/indexer.py)
- [`rocm_aiter_fa.py`](../../vllm/v1/attention/backends/rocm_aiter_fa.py)
- [`rocm_aiter_mla.py`](../../vllm/v1/attention/backends/mla/rocm_aiter_mla.py)
- [`rocm_aiter_mla_sparse.py`](../../vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py)
- [`xpu_mla_sparse.py`](../../vllm/v1/attention/backends/mla/xpu_mla_sparse.py)

Other attention, Mamba, input-batch, connector, and speculative-decoding
workspaces are conditionally risky. Most are currently overwritten before use,
but they depend on an implicit invariant that is not expressed by the allocator
or component interface. Out-of-tree backends and connectors can introduce the
same issue during registration or initialization.

### Weights scope and level-2 reload

The following classes retain plain device tensor attributes that are not
parameters or registered buffers:

- Gemma3n router and embedding scales;
- Conformer relative positional encodings;
- Voxtral mel filter banks;
- Ernie VL visual token ID tensors;
- OpenPangu's zero-valued sink tensor;
- TRT-LLM and FlashInfer MoE activation constants and fake scales;
- Cutlass MoE stride tensors;
- Humming kernel lock buffers;
- MiniMax Lamport all-reduce flags, layout metadata, negative-zero sentinels,
  and device pointer arrays.

Examples can be found in:

- [`gemma3n.py`](../../vllm/model_executor/models/gemma3n.py)
- [`conformer_encoder.py`](../../vllm/model_executor/models/conformer_encoder.py)
- [`voxtral.py`](../../vllm/model_executor/models/voxtral.py)
- [`ernie45_vl.py`](../../vllm/model_executor/models/ernie45_vl.py)
- [`openpangu.py`](../../vllm/model_executor/models/openpangu.py)
- [`fused_moe/experts`](../../vllm/model_executor/layers/fused_moe/experts)
- [`humming.py`](../../vllm/model_executor/layers/quantization/humming.py)
- [`lamport_workspace.py`](../../vllm/model_executor/layers/minimax_rms_norm/lamport_workspace.py)

Some derived DFlash weights are plain attributes but are already rebuilt by
model-specific `load_weights` hooks. They are not currently confirmed failures,
although their recovery contract should be made explicit and tested.

## Goals

1. Ensure that no persistent non-KV tensor is silently discarded by level-1
   sleep.
2. Make level-2 recovery requirements explicit for parameters, buffers, derived
   weights, kernel constants, and stateful workspaces.
3. Preserve CUDA-graph pointer stability where required.
4. Keep actual KV-cache and recurrent-state backing allocations discardable.
5. Provide a deterministic test mechanism that fails before missing recovery
   logic reaches an end-to-end model test.
6. Define a contract that also applies to out-of-tree attention backends,
   connectors, and hardware plugins.

## Non-Goals

1. Preserve active request KV-cache contents across sleep.
2. Make level-2 wake-up automatically reload model parameters. The caller still
   controls weight reload or replacement.
3. Persist CUDA state across process termination.
4. Change scheduler pause, abort, or drain semantics.
5. Guarantee that every scratch allocation is zeroed. Scratch buffers may remain
   uninitialized if their owner fully writes them before every read.

## Proposed Tensor Ownership Model

Every device tensor that survives initialization must belong to one of the
following classes.

| Ownership class | Sleep behavior | Recovery requirement |
| --- | --- | --- |
| Discardable cache backing | Unmap without backup | New requests must not read old contents |
| Model parameter | Level 1 host backup; level 2 discard | Reload before execution after level 2 |
| Static model buffer | Restore after both levels | Register as a model buffer or rematerialize from host/Python data |
| Derived weight | Rebuild after parameter reload | Explicit post-reload hook |
| Persistent runtime metadata | Keep outside discardable pool or restore in place | Explicit owner and wake hook |
| Scratch workspace | May be discarded | Complete write/reset before first read |
| Stateful synchronization workspace | Discard only with explicit reset | Restore sentinels, counters, layouts, and pointer arrays in place |

The classification is semantic and belongs to the component that owns the
tensor. Allocation time must not be used as a proxy for ownership.

### Required invariants

1. A tensor in a discardable allocation must either be discardable itself or be
   fully rewritten before its first post-wake read.
2. Recovery must be in place for tensors referenced by CUDA graphs. Replacing a
   Python tensor object after capture is not sufficient unless all captured
   graphs are rebuilt.
3. Level-2 parameter reload must be followed by rebuilding every derived tensor
   that snapshots or repacks those parameters.
4. Synchronization buffers must restore protocol-specific initial state, not
   merely zero memory.
5. Plugins that allocate persistent device tensors during KV registration must
   obey the same ownership contract.

## Proposed Design

### 1. Narrow the KV-cache allocation scope

Split KV-cache initialization into preparation, allocation, and binding phases:

```python
prepared = model_runner.prepare_kv_cache(kv_cache_config)

with worker._maybe_get_memory_pool_context(tag="kv_cache"):
    kv_cache_tensors = model_runner.allocate_kv_cache(prepared)

model_runner.bind_kv_cache(prepared, kv_cache_tensors)
```

Only actual cache backing allocations should occur in
`allocate_kv_cache`. Attention builders, block tables, connector objects,
speculator state, and fixed metadata should be constructed outside the pool.

The exact method split can differ between MRV1 and MRV2, but both must enforce
the same boundary. The existing KV-zero metadata initialization already follows
this principle by running outside the pool.

Small persistent metadata allocated outside the custom pool remains resident
during sleep. This is preferable for correctness and usually negligible beside
model and KV-cache storage. A future recoverable metadata tag can be introduced
if measurements show that this memory is material.

### 2. Add explicit component recovery hooks

Components that own state which must be reset after discarded storage is
remapped should implement a narrow lifecycle contract, for example:

```python
class SleepAwareTensorOwner(Protocol):
    def post_wake_up(self, discarded_tags: set[str]) -> None: ...

    def post_weights_reload(self) -> None: ...
```

The model runner should maintain an explicit registry of such owners rather
than discover them through arbitrary Python object traversal. Recovery order
must be deterministic:

1. remap selected allocator tags;
2. restore host-backed allocations and registered model buffers;
3. reload model parameters when requested by the caller;
4. rebuild derived weights;
5. reset persistent runtime and synchronization state;
6. permit scheduling and CUDA-graph replay.

This hook is a fallback for state that cannot be moved outside a discardable
pool. Narrowing the allocation scope remains the primary fix.

### 3. Use model buffers and host values for static model tensors

Static tensors owned by an `nn.Module` should use
`register_buffer(..., persistent=False)`. The current level-2 worker path saves
and restores `named_buffers`, including non-persistent state-dict buffers.

Scalar constants should remain Python values when kernels do not require a
device pointer. Small ID lists should remain host tuples or tensors and be
materialized on the destination device when needed.

Non-module kernel implementations should register persistent tensors on their
owning layer. Keeping the only reference on a quantization or kernel helper
object bypasses the existing model-buffer recovery path.

### 4. Rebuild derived weights after reload

Models and quantization methods that cache concatenated, stacked, repacked, or
otherwise derived tensors must expose an idempotent post-reload operation. This
operation must update captured storage in place or explicitly invalidate and
recapture CUDA graphs.

Existing model-specific behavior, such as DFlash fused KV-buffer rebuilding,
should be routed through the common lifecycle and covered by level-2 tests.

### 5. Reset synchronization state explicitly

Lock buffers and distributed workspaces require protocol-aware recovery.
Examples include:

- zeroing Humming locks;
- restoring Lamport negative-zero sentinels;
- resetting Lamport rotation counters;
- restoring layout constants;
- recreating device pointer arrays from current stable virtual addresses.

These operations must complete before any kernel or captured graph can consume
the workspace.

## Alternatives Considered

### Back up the entire KV-cache tag

This preserves accidentally tagged tensors but also copies the full KV cache to
host memory, defeating the memory and latency goals of level-1 sleep.

### Zero every discarded allocation on wake-up

Zeroing makes behavior deterministic but does not restore nonzero scales,
`arange` indices, top-k values, sentinels, or pointer arrays. It can also hide
missing ownership declarations for zero-initialized state.

### Add only backend-specific wake hooks

Hooks can repair the currently known tensors, but every new backend or connector
can reintroduce the issue. Hooks are useful for protocol state; they are not a
substitute for a correct allocation boundary.

### Introduce a third recoverable metadata tag

A `persistent_metadata` tag backed up at both sleep levels is viable for larger
metadata that should be reclaimed. It adds policy and allocation complexity,
and allocation-level tagging still requires clean scope boundaries. This RFC
therefore proposes pool exclusion first and leaves the additional tag as a
future optimization.

### Recapture every CUDA graph after wake-up

Recapture avoids stale pointer and value assumptions but significantly increases
wake latency and still does not define how eager execution state is restored.
It should be required only when a component cannot restore captured storage in
place.

## Test Strategy

### Deterministic remap poisoning

Tests must not assume that freshly mapped memory is nonzero. A test helper should
wrap `create_and_map`, fill every new mapping with a known byte such as `0xA5`,
and then allow the normal CPU-backup restoration to run:

```python
original = cumem.create_and_map

def create_map_and_poison(handle):
    original(handle)
    _, size, ptr, _ = handle
    cumem.libcudart.cudaMemset(ptr, 0xA5, size)
```

Host-backed weights are overwritten with their saved bytes after poisoning.
Discarded KV allocations retain the poison. A persistent tensor passes only if
it was outside the discarded allocation or its owner restored it.

The helper must be isolated in a fresh subprocess because `CuMemAllocator` is a
singleton and keeps allocator pools alive.

### Test 1: allocator ownership reproducer

Extend [`tests/basic_correctness/test_mem.py`](../../tests/basic_correctness/test_mem.py)
with a minimal allocation-level test:

1. allocate a known byte tensor under `weights`;
2. allocate a fake KV tensor and unrelated metadata tensor under `kv_cache`;
3. record all virtual addresses;
4. run level-1 sleep and poisoned wake-up;
5. assert that all addresses are stable;
6. assert that the weight is restored;
7. assert that both KV and unrelated metadata contain poison.

This test documents the allocator contract. It does not assert that the
allocator should infer tensor ownership.

### Test 2: CUDA-graph precision reproducer

Capture a graph that multiplies an input by a nonzero scale allocated in the
`kv_cache` pool. After poisoned sleep/wake, replay the graph and verify that:

- graph replay succeeds;
- the scale pointer is unchanged;
- the output is wrong before recovery;
- the output matches the reference after the owner's wake hook or after moving
  the scale outside the pool.

This is the smallest test demonstrating that pointer stability can turn a
memory-lifecycle bug into silent accuracy corruption.

### Test 3: level-2 recovery classification

Construct a toy module containing:

- one parameter;
- one `persistent=False` registered buffer;
- one plain tensor attribute.

Run poisoned level-2 sleep/wake, simulate parameter reload, and restore
`named_buffers`. Verify that the parameter and buffer recover while the plain
attribute remains poisoned. The permanent regression version should convert the
plain attribute to a buffer and require all three values to recover.

### Test 4: real backend regression

At least one CUDA backend test must instantiate an actual metadata builder in
the real initialization scope and verify values after sleep/wake. FlashMLA
Sparse is a suitable first target because
[`test_sparse_mla_backends.py`](../../tests/v1/attention/test_sparse_mla_backends.py)
already has compact builder fixtures.

Platform-specific tests should then cover:

- FlashAttention R-SWA and Static Sink Attention on CUDA;
- ROCm Aiter FlashAttention and MLA metadata;
- XPU Sparse MLA metadata;
- DSA indexer offsets and padding indices.

Each test should check tensor contents or kernel output, not only shape and
allocation validity.

### Test 5: targeted end-to-end generation

The existing dense tiny-model sleep test remains useful but is insufficient.
Add end-to-end tests only where a small model exercises a confirmed affected
path. Compare deterministic tokens or logits before and after sleep, and include
multiple cycles to expose incomplete reset logic.

Model-affecting changes must also run the relevant evaluation suite described in
the contribution guidelines.

## Migration Plan

### Phase 1: establish deterministic failures

1. Add the poison helper and allocator contract test.
2. Add CUDA-graph scale and level-2 classification tests.
3. Add one real affected backend test that fails before the fix.

### Phase 2: correct allocation boundaries

1. Split MRV1 KV preparation, allocation, and binding.
2. Split MRV2 KV preparation, allocation, and binding.
3. Move attention builders, block tables, connectors, and fixed metadata outside
   the `kv_cache` pool.
4. Verify that only cache backing allocations carry the `kv_cache` tag.

### Phase 3: migrate persistent owners

1. Repair the confirmed KV-scope inventory.
2. Convert static model constants to buffers or host/Python values.
3. Register non-module kernel constants on owning modules.
4. standardize derived-weight rebuild hooks.
5. add Humming and Lamport protocol resets.

### Phase 4: enforce the contract

Add a debug-only allocation audit using `TorchDispatchMode` or equivalent
instrumentation during tagged scopes. Record tensor creation stacks with weak
references and report surviving tensors at scope exit. The report should include
the tag, size, allocation range, owner type, and creation location.

This audit should be available to backend and plugin authors without enabling it
in production by default.

## Compatibility and Performance

### CUDA graphs

Moving persistent metadata outside the discardable pool preserves its virtual
and physical storage across sleep, so existing captured pointers remain valid.
In-place recovery hooks preserve addresses for the remaining state. Components
that replace storage must explicitly invalidate captured graphs.

### Memory reclamation

Some small metadata allocations will remain resident during sleep. Their size
should be measured and reported separately from weights and cache backing. If
the retained memory is significant for a backend, that backend can use a future
recoverable metadata tag rather than the discardable KV tag.

### Fragmentation

Narrower scopes can change allocation grouping and fragmentation. Validation
must record peak model-load memory, available KV-cache memory, memory freed by
sleep, and wake latency before and after the change.

### Platform behavior

CUDA, ROCm, and XPU allocators differ in mapping and initialization behavior.
The ownership contract is platform-independent: discarded contents must never
be consumed without recovery. Poisoning tests should use the platform's memory
set operation and should not rely on observed zero-fill behavior.

### Out-of-tree components

Attention backends, connectors, and hardware plugins must not allocate
persistent metadata inside a callback documented as a discardable cache
allocation phase. If unavoidable, they must register a sleep-aware owner and
restore state in place.

## Open Questions

1. Should recoverable runtime metadata receive a dedicated allocator tag, or is
   keeping it resident sufficient for all supported configurations?
2. Should the sleep-aware lifecycle be a new protocol or an extension of the
   existing model-runner/backend interfaces?
3. Should level-2 wake-up reject execution until parameter reload and all
   post-reload hooks have completed?
4. How should plugins declare and validate persistent tensor ownership without
   exposing allocator internals?
5. Should the debug allocation audit become a CI mode for selected backend
   initialization tests?

## Acceptance Criteria

The proposal is complete when:

1. only discardable cache backing allocations receive the `kv_cache` tag;
2. all confirmed persistent constants survive or are restored after poisoned
   level-1 sleep/wake;
3. all registered buffers, derived weights, locks, and synchronization state
   satisfy their level-2 recovery contract;
4. CUDA graphs produce the same result before and after sleep/wake;
5. targeted CUDA, ROCm, and XPU tests cover their affected backends;
6. sleep memory reclamation and wake latency do not regress materially;
7. backend and plugin documentation states the persistent tensor ownership
   requirements.
