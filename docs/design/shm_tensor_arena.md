# Zero-Copy Shared-Memory Tensor Arena for `MessageQueue`

This document describes an optimization to vLLM's engine→worker IPC path for
multimodal serving under tensor parallelism, implemented in
`vllm/distributed/device_communicators/shm_broadcast.py`. It **builds on the
existing out-of-band tensor channel** (`_reduce_tensor` / `_rebuild_tensor`),
which already keeps CPU-tensor bytes out of the pickle stream.

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

All changes are contained in `shm_broadcast.py`.

### 2.1 `ShmTensorArena`

A second shared-memory region created alongside the existing `ShmRingBuffer`:

- **N slots × slot_bytes** (default 8 × 256 MB, env-tunable), plus per-slot
  metadata `[written_flag, reader0_done … readerN_done]`.
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
        if (isinstance(obj, torch.Tensor) and obj.device.type == "cpu"
                and obj.layout is torch.strided and obj.is_contiguous()
                and obj.numel() * obj.element_size() >= MIN_BYTES):
            idx = self.arena.write_tensor(obj)        # ONE memcpy into a free slot
            if idx is not None:
                return (_rebuild_arena_tensor,
                        (arena_name, idx, nbytes, dtype_str, shape))
        return NotImplemented   # fall through to dispatch_table → _reduce_tensor
```

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
    arena = _TENSOR_ARENAS[arena_name]            # this process's attached arena
    return arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    # get_tensor: torch.frombuffer over the mapped slot — zero bytes copied
```

The rebuilt tensor *is* the shared memory — no transport copy and no deserialize
on any rank.

**Slot lifecycle.** The rebuilt tensor is the *source* of an async H2D
(`x.to(device, non_blocking=True)`) while the worker executes that step, so the
reader must not release the slot at unpickle time. Releases are deferred to the
reader's **next `dequeue`** and gated on H2D completion. On the pinned fast path
(§2.4) the H2D is a true async DMA that can outlive `execute_model` — under
`--async-scheduling` no device sync covers it — so a step-count deferral alone is
*not* sufficient: the writer could reclaim the slot while the DMA is still reading
it. At `flush_releases` the reader records a CUDA event on the compute stream
(ordered after the previous step's H2D) and only sets its done flag for a slot
once that event has completed (non-blocking `event.query()`; a slot not yet done
simply waits one more dequeue). When the mapping is *not* pinned,
`cudaMemcpyAsync` from pageable host memory stages the copy synchronously before
returning, so the slot is released immediately. The writer requires all readers'
done flags before reusing a slot, so a slot is never overwritten while any
reader's DMA is in flight.

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
|---|---|---|
| `--enable-shm-tensor-arena` / `--no-enable-shm-tensor-arena` (`enable_shm_tensor_arena`) | **off** | Opt in to route large CPU tensors through the arena (reserves slots in `/dev/shm`). Off = the out-of-band `_reduce_tensor` path only, identical to stock behavior. |

Internal constants in `shm_broadcast.py`: **8 slots × 256 MB**; tensors larger
than a slot, or smaller than the **8 MB** divert threshold, take the out-of-band
`_reduce_tensor` path.

## 4. Validation

> **Baseline caveat.** The A/B numbers below were collected against the *in-band
> pickle* path that predates `_reduce_tensor`. They quantify the arena vs. that
> older baseline (an upper bound), **not** the arena vs. the current out-of-band
> path. The current-`main` oob baseline would land between the "stock" and
> "arena + pinning" rows; measuring exactly where is the open item.

1. **Unit test** (to be added to the branch): a 199 MB bf16 tensor pushed through
   a real `MessageQueue` to two forked reader processes — byte-exact checksums,
   arena path confirmed, deferred release, slot reuse, and exhaustion/oversize
   fallbacks. Enqueue of the 199 MB payload: **66.7 ms** (vs ~1275 ms for the
   in-band serialize of the same size).
2. **Same-seed A/B** (baseline = in-band pickle, pre-`_reduce_tensor`), interactive
   multimodal workload, uncapped input images, TP=4 ×2 workers on one 8-GPU node,
   low qps so individual images decompose cleanly (~1.2k aligned requests):

   | TTFT (ms) | p50 | p90 | p99 | max | >1 s | >1.5 s |
   |---|---:|---:|---:|---:|---:|---:|
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
  `/dev/shm` (default 2 GB, lazily paged). Like `ShmRingBuffer`, it should guard
  creation with a free-space check (`check_shm_free_space`) and surface a clear
  error rather than failing late.
- **Relationship to `_reduce_tensor`.** This is strictly additive: the arena is
  an opt-in fast path for the large-multimodal-tensor case; declining it (or
  disabling via env) reverts to the merged out-of-band behavior.
- **Slot release granularity**: releases are gated on a per-slot CUDA event
  recorded after the consuming H2D (§2.3), which closes the async-DMA reuse
  window on the pinned path. Bursts deeper than the slot count safely fall back
  to the out-of-band path when the arena is exhausted.
- **Fallback observability**: arena exhaustion / oversize fallbacks are
  rate-limited log lines today; a counter metric would be better.
- **Scope**: the arena activates only when every queue reader is node-local.
  Remote readers (multi-node PP/TP) keep the existing socket path.
