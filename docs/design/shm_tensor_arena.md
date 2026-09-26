# Zero-Copy Shared-Memory Tensor Arena for `MessageQueue`

This document describes an optimization to vLLM's engine→worker IPC path for
multimodal serving under tensor parallelism, implemented in
`vllm/distributed/device_communicators/shm_tensor_arena.py` and wired into
`shm_broadcast.py`'s `MessageQueue`. It **builds on the existing out-of-band
tensor channel** (`_reduce_tensor` / `_rebuild_tensor`, still in
`shm_broadcast.py`), which already keeps CPU-tensor bytes out of the pickle
stream.

> **Status: experimental, opt-in.** This lands as an increment over the
> existing out-of-band channel, not a replacement for it. vLLM also has an
> existing multimodal-tensor shm caching mechanism; reconciling the two is
> intentionally left as follow-up work rather than blocking this PR.

## TL;DR

`MessageQueue.enqueue` already avoids copying CPU tensor bytes into the pickle
stream: `_reduce_tensor` emits each tensor as a protocol-5 out-of-band
`PickleBuffer`, and large payloads are published to the local readers over a
ZMQ socket. That removed the dominant cost of the older *in-band* path — the
per-tensor `pickle` serialize (see [History](#history-the-in-band-pickle-era)).

Two costs remain for a large multimodal `pixel_values` tensor on a TP=N worker:

1. **N transport copies.** On the overflow (large-payload) path the out-of-band
   buffer is published to each of the N node-local readers via `send_multipart`,
   so the tensor bytes are moved once **per reader** by the transport — ~N×200 MB
   of host data motion for a payload the model needs exactly once.
2. **A pageable H2D.** Each reader zero-copies a view over its received ZMQ
   frame, but that frame is ordinary pageable memory, so the reader's subsequent
   `cudaMemcpyAsync` to the GPU pays pageable-copy staging plus first-touch
   faults.

This change adds a slotted shared-memory **arena**: the writer does **one**
memcpy of the tensor into a free slot; every reader takes a **zero-copy view**
of the same slot (no per-reader transport copy); and the mapping is
`cudaHostRegister`-pinned so the H2D is a true DMA. Net for a 200 MB image on
TP=4: **1 copy + a pinned DMA** instead of **4 transport copies + a pageable
H2D**. Small tensors, non-contiguous tensors, and arena exhaustion all fall
through to the existing `_reduce_tensor` path unchanged.

## Scope of the win (and what still needs measuring)

The arena is an **incremental** improvement layered on the out-of-band channel,
not a from-scratch fix — the out-of-band channel already eliminated the single
largest cost (the serialize copy). The benefit that remains is:

- **N→1 copies**, which grows with TP degree (no gain at TP=1), and
- the **pinned vs. pageable H2D**, which shows up as reader-side H2D latency.

> **The benchmarks in [§4](#4-validation) were measured against the *in-band
> pickle* baseline that predates `_reduce_tensor`**, where the dominant cost was
> the serialize copy. They are retained as motivation and as an upper bound. The
> incremental improvement of the arena *over the current out-of-band path on
> `main`* is smaller and has **not yet been re-measured**; that A/B (arena vs.
> oob, same seeded workload) is the outstanding validation item before this is
> merge-ready.

## 1. Where it bites

Any multimodal model served with `--tensor-parallel-size > 1` through the
multiproc executor. The `rpc_broadcast_mq` `MessageQueue`
(`v1/executor/multiproc_executor.py`) carries scheduler output — including the
raw preprocessed `pixel_values` — from the EngineCore to every TP worker each
step. Images at native resolution reach tens to hundreds of MB (a ~4k×4k image
is ~200 MB as bf16 patches). The out-of-band channel keeps those bytes out of
the pickle stream, but on the overflow path they are still published **once per
node-local reader**, and each reader's H2D is from **pageable** memory. The
per-reader copy cost scales with TP degree; the pinning cost is a fixed per-image
tax on the reader H2D. Latency-sensitive (interactive) serving makes the residual
visible in TTFT and, via head-of-line blocking, in the end-to-end latency of
*other* requests.

### History: the in-band-pickle era

Before `_reduce_tensor`, `MessageQueue.enqueue` pickled the whole payload —
including tensor bytes — because `torch.Tensor.__reduce_ex__` does not emit
protocol-5 out-of-band buffers. A ~200 MB tensor was byte-copied into the pickle
stream on the engine (`THPStorage_writeFileRaw`, ~6.5 ms/MB → ~1.3 s), *inside*
the EngineCore step loop, so all GPUs on the worker went idle for **0.9–1.3 s per
large image** (the vision-encoder compute for the same image is 50–90 ms →
transport dominated compute ~15:1); each reader then paid a full `pickle.loads`
byte-copy. Per 200 MB image on TP=4 that was ~1.8 GB of host data motion.

The `_reduce_tensor` out-of-band channel removed the serialize copy (and the
per-reader `pickle.loads` copy on the zero-copy ZMQ-frame path). Note that
merely raising `VLLM_MQ_MAX_CHUNK_BYTES_MB` (ring vs. socket path) did **not**
help in the in-band era: the serialize ran before the ring-vs-socket branch, so
it was common to both — a negative result that pinned the real cost on the
byte-copies, not the transport choice.

## 2. The arena

The arena itself (`ShmTensorArena`, `_ArenaPickler`, `_rebuild_arena_tensor`)
lives in `shm_tensor_arena.py`; `shm_broadcast.py`'s `MessageQueue` creates
and wires it in but otherwise treats it as an internal implementation detail.

### 2.1 `ShmTensorArena`

A second shared-memory region created alongside the existing `ShmRingBuffer`:

- **N slots × slot_bytes** (default 8 × 256 MB, internal constants -- see
  §3), plus per-slot metadata `[written_flag, reader0_done … readerN_done]`.
- Concurrency uses the **same lock-free single-writer/N-reader protocol as
  `ShmRingBuffer`** (memory fences, per-reader done flags), so the model is one
  the codebase already trusts.
- Created by the queue **writer** in `MessageQueue.__init__` when all readers are
  node-local; readers attach via a new `tensor_arena_handle` field on the queue
  `Handle`. Queues with remote readers get no arena and keep today's behavior.
- Pages are allocated lazily by the kernel on first write, so arenas on queues
  that never carry big tensors (e.g. worker→engine response queues) cost
  approximately nothing beyond the (virtual) reservation.

### 2.2 Writer path — arena diversion, composing with `_reduce_tensor`

`enqueue` builds one `dispatch_table` that routes CPU tensors through
`_reduce_tensor` (out-of-band `PickleBuffer`). When a node-local arena exists,
the pickler is an `_ArenaPickler` whose `reducer_override` *additionally* diverts
**large contiguous** CPU tensors into the arena:

```python
class _ArenaPickler(pickle.Pickler):
    def reducer_override(self, obj):
        rebuild_fn = _ARENA_REBUILD_FNS.get(type(obj))   # {Tensor: ..., Parameter: ...}
        if (rebuild_fn is not None and obj.device.type == "cpu"
                and obj.layout is torch.strided and obj.is_contiguous()
                and not obj.requires_grad
                and obj.numel() * obj.element_size() >= MIN_BYTES):
            idx = self.arena.write_tensor(obj)        # ONE memcpy into a free slot
            if idx is not None:
                return (rebuild_fn,
                        (self.arena.shared_memory.name, idx, nbytes, dtype_str, shape))
        return NotImplemented   # fall through to dispatch_table → _reduce_tensor
```

Note the exact-type dict lookup (not `isinstance`), matching how
`dispatch_table` itself dispatches: an unrecognized `Tensor` subclass must
decline here too, or it would come back as a plain `Tensor`, silently
losing its subclass identity. `torch.nn.Parameter` is the one subclass this
*does* divert safely, via its own rebuild function — see §2.3 for why that
needs more than just handling the type-identity case.

`reducer_override` is consulted before an object's normal reduction, and
returning `NotImplemented` falls through to the `dispatch_table` — so a diverted
tensor becomes a ~100-byte rebuild stub, and **everything the arena declines**
(too small, non-contiguous, or arena full) is handled by `_reduce_tensor`
exactly as on `main`. The single arena copy —
`torch.frombuffer(slot).copy_(t.view(torch.uint8))` — is a multithreaded memcpy
at memory bandwidth (~50–70 ms for 200 MB), and, unlike the oob path, the *same*
slot is then read zero-copy by all N readers rather than transported to each.

**The writer never blocks.** If no slot is free or the tensor exceeds the slot
size, `write_tensor` returns `None`; the pickler falls through to `_reduce_tensor`
— the worst case is exactly the current out-of-band behavior, and deadlock is
structurally impossible.

### 2.3 Reader path — zero copies

The stub unpickles through a module-level rebuild function:

```python
def _rebuild_arena_tensor(arena_name, slot_idx, nbytes, dtype_str, shape):
    arena = _TENSOR_ARENAS[arena_name]             # this process's registry of attached arenas
    t = arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    # get_tensor: torch.frombuffer over the mapped slot — zero bytes copied
    arena.schedule_release(t, slot_idx)            # see below — must be `t` itself here
    return t
```

**The `torch.nn.Parameter` trap.** `get_tensor` and release scheduling are
separate calls because they aren't always scheduled on the same object. For
a plain tensor, `t` is exactly what the caller ends up holding. But
`torch.nn.Parameter(data_tensor, requires_grad)` shares `data_tensor`'s
storage at the C++ level **without keeping a Python reference to
`data_tensor` itself** — enough for PyTorch, not enough for a
`weakref.finalize` scheme that watches one specific Python object.
Scheduling release on `data_tensor` would queue the slot the instant it's
constructed (nothing references `data_tensor` anymore), while the
`Parameter` — still a live zero-copy view into that slot — is what the
caller actually holds; the writer could then overwrite the slot underneath
it with no error raised anywhere.

The arena closes this with a dedicated rebuild function for
`torch.nn.Parameter` that schedules release on the constructed `Parameter`
instead:

```python
def _rebuild_arena_parameter(arena_name, slot_idx, nbytes, dtype_str, shape):
    arena = _TENSOR_ARENAS[arena_name]
    t = arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    param = torch.nn.Parameter(t, requires_grad=False)
    arena.schedule_release(param, slot_idx)        # schedule on `param`, not `t`
    return param
```

`_ARENA_REBUILD_FNS = {torch.Tensor: _rebuild_arena_tensor, torch.nn.
Parameter: _rebuild_arena_parameter}` is the full set of types the arena
knows how to divert correctly today; anything else (any other `Tensor`
subclass) declines in `reducer_override` and takes the out-of-band path,
which needs no such scheme — `_reduce_tensor` never constructs a new
wrapper object around its payload.

The registry is keyed by arena shm name (`_TENSOR_ARENAS:
weakref.WeakValueDictionary[str, ShmTensorArena]`) rather than a single
per-process slot, since nothing guarantees a process only ever attaches to
one arena-bearing queue. It holds only a *weak* reference — a strong one
would keep the arena (and its pinned mapping) alive for the process
lifetime — and needs no explicit removal on `MessageQueue.shutdown()`
(unsafe anyway, since shutdown can run on a different thread than an
in-flight `dequeue()`); an entry just disappears once nothing else
references that arena.

The rebuilt tensor *is* the shared memory — no transport copy and no deserialize
on any rank.

**Slot lifecycle.** The rebuilt tensor is the *source* of an async H2D while
the worker executes that step, so the reader must not release the slot at
unpickle time. Release is tied to **garbage collection of the
caller-retained object** (`weakref.finalize` in `schedule_release`), not a
fixed "next dequeue" schedule — needed for callers like `prompt_embeds` that
retain the tensor across many `dequeue` calls during chunked prefill, where
a fixed schedule would let the writer reclaim (and corrupt) the slot while
the worker was still reading it. Once queued, the pinned fast path (§2.4)
adds a second gate: the H2D is a true async DMA that can outlive
`execute_model`, so "the tensor was collected" alone isn't sufficient.
`flush_releases` records a CUDA event on the compute stream after each
step's H2D and only marks a slot done once that event completes
(non-blocking `event.query()`; not-yet-done just waits one more dequeue).
Unpinned, `cudaMemcpyAsync` from pageable memory already stages
synchronously, so the slot releases immediately. Either way, the writer
requires every reader's done flag before reusing a slot.

> **Known residual limitation.** `weakref.finalize` tracks the garbage
> collection of the *specific* object each rebuild function passes to
> `schedule_release` (see §2.3). A caller that takes a view/slice of that
> object and drops the original — instead of keeping it alive, as vLLM's
> current callers do — would not delay the release, since PyTorch views
> keep the underlying storage alive via the C++ refcount independent of
> this (Python-object-level) finalizer.
>
> Assumes the multimodal H2D is issued on the worker's current/default compute
> stream (true today: mm inputs are copied eagerly, outside the decode CUDA
> graph). If a future vLLM issues that copy on a dedicated side stream, the event
> must be recorded at the copy site rather than at `flush_releases`.

### 2.4 Pinning — the zero-copy trap

A zero-copy view over shared memory the reader process has never touched makes
the subsequent `cudaMemcpyAsync` pay **first-touch page faults on ~50k pages plus
pageable-copy staging** — the same tax the current oob path pays on its pageable
ZMQ frame. Fix: each reader lazily `cudaHostRegister`s the whole arena mapping
once (~1 s, first use), after which every H2D from the arena is a **pinned-memory
DMA** (~10 ms for 192 MB). Processes without a CUDA context skip registration
silently. (Because the arena mapping is stable and reused, this one-time pin
amortizes across all future images — an option the transient per-message ZMQ
frame does not have.)

## 3. Configuration

A single CLI flag toggles the arena; the slot count/size and divert threshold
are internal constants (no env vars).

| Flag (`ParallelConfig` field) | Default | Meaning |
| --- | --- | --- |
| `--enable-shm-tensor-arena` / `--no-enable-shm-tensor-arena` (`enable_shm_tensor_arena`) | **off** | Opt in to route large CPU tensors through the arena (reserves slots in `/dev/shm`). Off = the out-of-band `_reduce_tensor` path only, identical to stock behavior. |

Internal constants in `shm_tensor_arena.py`: **8 slots × 256 MB**; tensors larger
than a slot, or smaller than the **8 MB** divert threshold, take the out-of-band
`_reduce_tensor` path.

## 4. Validation

> **Baseline caveat.** The A/B numbers below were collected against the *in-band
> pickle* path that predates `_reduce_tensor`. They quantify the arena vs. that
> older baseline (an upper bound), **not** the arena vs. the current out-of-band
> path. The current-`main` oob baseline would land between the "stock" and
> "arena + pinning" rows; measuring exactly where is the open item.

1. **Unit tests** (`tests/distributed/test_shm_broadcast.py`): byte-exact
   zero-copy round-trips (incl. bf16/fp8), slot lifecycle (no reuse until every
   reader releases; exhaustion / oversize / non-contiguous fall back to the
   out-of-band path), event-gated release on the pinned path, `_ArenaPickler` ⊕
   `_reduce_tensor` composition, and an end-to-end `MessageQueue` broadcast
   through forked processes. (Historical microbenchmark: enqueue of a 199 MB
   payload took **66.7 ms** vs ~1275 ms for the old in-band serialize.)
2. **Same-seed A/B** (baseline = in-band pickle, pre-`_reduce_tensor`), interactive
   multimodal workload, uncapped input images, TP=4 ×2 workers on one 8-GPU node,
   low qps so individual images decompose cleanly (~1.2k aligned requests):

   | TTFT (ms) | p50 | p90 | p99 | max | >1 s | >1.5 s |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | in-band pickle (old baseline) | 93 | 401 | 1321 | 2139 | 30 | 9 |
   | + bigger MQ chunk (config lever) | 91 | 368 | 1287 | 2027 | 22 | 5 |
   | **arena + pinning** | **89** | **241** | **862** | **1375** | **7** | **0** |

   p50 unchanged → `reducer_override` adds no measurable overhead when no large
   tensor is present.
3. **Per-image stall** (CUDA-level all-GPU idle gap around the pixel H2D), vs the
   in-band baseline: 192 MB image 1318 ms → 681 ms (arena only) → **351 ms class**
   (arena + pinning); vision-encoder compute for the same image is ~80 ms.
4. **Outstanding — arena vs. out-of-band on `main`.** Re-run the same seeded A/B
   with the baseline set to current `main` (i.e. `_reduce_tensor` on, arena off
   via `--no-enable-shm-tensor-arena`) vs. arena on, to quantify the incremental N→1
   copy + pinning gain at TP=4 (and ideally TP=2/8 to show the copy-count scaling).

## 5. Limitations and future work

- **Value is TP- and size-dependent.** The arena helps only when a queue has
  ≥2 node-local readers *and* carries tensors ≥ `MIN_MB`; at TP=1, or for small
  payloads, it adds nothing and everything takes the out-of-band path.
- **Shared-memory reservation.** The arena reserves `slots × slot_bytes` of
  `/dev/shm` (default 2 GB, lazily paged). Like `ShmRingBuffer`, creation is
  guarded by a free-space check (`check_shm_free_space`), so an undersized
  `/dev/shm` fails fast with a clear error instead of a `SIGBUS` the first
  time a tensor is copied into pages beyond tmpfs capacity.
- **Relationship to `_reduce_tensor`.** This is strictly additive: the arena is
  an opt-in fast path for the large-multimodal-tensor case; declining it (or
  disabling via env) reverts to the merged out-of-band behavior.
- **Relationship to existing mm tensor shm caching.** vLLM already has a
  separate shared-memory caching path for multimodal tensors; this arena is
  not yet unified with it (see the experimental note in §1). Reconciling the
  two is follow-up work.
- **Slot release granularity**: releases are tied to the returned tensor's
  garbage collection and, on the pinned path, further gated on a per-slot
  CUDA event recorded after the consuming H2D (§2.3), which closes both the
  multi-step-retention and the async-DMA reuse windows. Bursts deeper than
  the slot count safely fall back to the out-of-band path when the arena is
  exhausted. The one case this does *not* cover is a caller that drops the
  base tensor while still holding a view into it (see the residual-
  limitation callout in §2.3) — that remains a real, if currently
  theoretical, hazard.
- **Fallback observability**: arena exhaustion (no free slot) is a
  rate-limited log line today; a counter metric would be better. An
  oversize tensor (bigger than a slot) falls back silently, with no log at
  all -- worth adding if that path turns out to matter in practice.
- **Scope**: the arena activates only when every queue reader is node-local.
  Remote readers (multi-node PP/TP) keep the existing socket path.
