# Zero-Copy Shared-Memory Tensor Arena for `MessageQueue`

This document describes a performance problem in vLLM's engine→worker IPC path that
severely penalizes multimodal serving under tensor parallelism, and the fix implemented
on this branch: a zero-copy shared-memory tensor arena inside
`vllm/distributed/device_communicators/shm_broadcast.py`.

## TL;DR

With TP > 1, every `execute_model` call is broadcast from the EngineCore process to the
TP worker processes through `MessageQueue.enqueue`, which serializes the whole payload
with `pickle` — **including raw multimodal `pixel_values` tensors**. Torch tensors do
not participate in pickle's out-of-band buffer protocol, so a large image is byte-copied
into the pickle stream on the engine side and byte-copied back out **once per TP rank**
on the reader side. For a ~200 MB pixel tensor on a TP=4 worker that is ~1.8 GB of host
byte-copying, most of it serial and **inside the engine's step loop** — the scheduler
stops, and all GPUs on the worker go idle for **0.9–1.3 s per large image**, including
decode for unrelated in-flight requests (head-of-line blocking). The vision-encoder
compute for the same images is 50–90 ms; transport dominates compute ~15:1.

The fix routes large CPU tensors around the pickle stream entirely, through a slotted
shared-memory arena: one memcpy into a slot on the writer, a `torch.frombuffer` view
(zero bytes copied) on each reader, with the mapping `cudaHostRegister`-pinned so the
subsequent H2D copy is a true DMA. Measured end to end on an interactive multimodal
workload (uncapped input images, identical seeded request sequence): TTFT p99
1321 ms → **862 ms**, TTFT max 2139 ms → **1375 ms**, requests over 1.5 s: 9 → **0**,
TTFT p50 unchanged (no overhead on the small-request path).

## 1. The problem

### Where it bites

Any multimodal model served with `--tensor-parallel-size > 1` through the multiproc
executor. The `rpc_broadcast_mq` `MessageQueue` (`v1/executor/multiproc_executor.py`)
carries scheduler output — with the multimodal kwargs, i.e. the raw preprocessed pixel
tensors — from the EngineCore to every TP worker on each step. Image inputs at native
resolution easily reach tens to hundreds of MB of `pixel_values` (a ~4k×4k image is
~200 MB as bf16 patches). Latency-sensitive (interactive) serving makes the resulting
tail visible in TTFT and, through head-of-line blocking, in the end-to-end latency of
*other* requests.

### The mechanism, at source level

`MessageQueue.enqueue` (before this branch):

```python
serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL,
                              buffer_callback=oob_callback)   # ← the cost lives here
if total_bytes >= self.buffer.max_chunk_bytes:    # 16 MB default
    # write a 1-byte overflow marker into the shm ring...
    self.local_socket.send_multipart(all_buffers)  # ...and ZMQ to every local reader
else:
    # copy the bytes into a shm ring chunk
```

Three compounding costs:

1. **Serialize.** Torch tensors do not implement pickle protocol-5 out-of-band
   buffers, so `pickle.dumps` drives `THPStorage_writeFileRaw`, which byte-copies the
   entire tensor into the pickle stream through Python-driven small writes. Measured at
   ~6.5 ms/MB (~155 MB/s) on the critical path — ~1.3 s for a 200 MB tensor.
2. **It blocks the engine.** The serialize runs inside the EngineCore step loop
   (`collective_rpc → enqueue`), so no new GPU work is scheduled anywhere on the worker
   until it finishes. CUDA-level profiling shows all GPUs of the TP group completely
   silent for the duration — decode graph launches for other requests included.
3. **Deserialize, per rank.** Payloads above `max_chunk_bytes` bypass the ring and go
   through a local ZMQ socket to each reader; every rank then runs `pickle.loads` →
   `THPStorage_readFileRaw` — another full byte-copy per rank — before the (cheap,
   ~20 ms) H2D copy.

Per 200 MB image on TP=4: ~200 MB × (1 serialize + 4 socket transfers + 4 deserializes)
≈ **1.8 GB of host data motion**, where the model itself only needs one 200 MB H2D.

### What does *not* fix it

Raising `VLLM_MQ_MAX_CHUNK_BYTES_MB` so the payload takes the shm-ring path instead of
the ZMQ socket was tested first (same seeded workload, per-image A/B): **no
improvement**. The path switch happens (zmq frames disappear from the stall
callchains), but `pickle.dumps` runs *before* the ring-vs-socket branch — the serialize
and the ×N deserialize copies are common to both paths and are the actual cost. This
negative result pins the fix target: the pickle byte-copies themselves must go.

## 2. The fix

All changes are contained in `shm_broadcast.py`.

### 2.1 `ShmTensorArena`

A second shared-memory region created alongside the existing `ShmRingBuffer`:

- **N slots × slot_bytes** (default 8 × 256 MB, env-tunable), plus per-slot metadata
  `[written_flag, reader0_done … readerN_done]`.
- Concurrency uses the **same lock-free single-writer/N-reader protocol as
  `ShmRingBuffer`** (memory fences, per-reader done flags), so the model is one the
  codebase already trusts.
- Created by the queue **writer** in `MessageQueue.__init__` when all readers are
  node-local; readers attach via a new `tensor_arena_handle` field on the queue
  `Handle`. Queues with remote readers get no arena and keep today's behavior.
- Pages are allocated lazily by the kernel on first write, so arenas on queues that
  never carry big tensors (e.g. worker→engine response queues) cost approximately
  nothing.

### 2.2 Writer path — `_ArenaPickler.reducer_override`

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
        return NotImplemented          # everything else: default pickling
```

`reducer_override` is consulted for every object before its `__reduce_ex__`, so a
diverted tensor never reaches `THPStorage_writeFileRaw`: the pickle stream carries a
~100-byte rebuild stub instead of the tensor bytes. The one remaining copy —
`torch.frombuffer(slot).copy_(t.view(torch.uint8))` — is a multithreaded memcpy at
memory bandwidth (~50–70 ms for 200 MB, vs ~1.3 s for the pickle serialize).

**The writer never blocks.** If no slot is free or the tensor exceeds the slot size,
`write_tensor` returns `None` and the pickler falls through to the default path — the
worst case is exactly today's behavior, and deadlock is structurally impossible.

### 2.3 Reader path — zero copies

The stub unpickles through a module-level rebuild function:

```python
def _rebuild_arena_tensor(arena_name, slot_idx, nbytes, dtype_str, shape):
    arena = _TENSOR_ARENAS[arena_name]            # this process's attached arena
    return arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    # get_tensor: torch.frombuffer over the mapped slot — zero bytes copied
```

The rebuilt tensor *is* the shared memory — no deserialize on any rank.

**Slot lifecycle.** The rebuilt tensor is the *source* of an async H2D (`x.to(device,
non_blocking=True)`) while the worker executes that step, so the reader must not release
the slot at unpickle time. Releases are deferred to the reader's **next `dequeue`** and
gated on H2D completion. On the pinned fast path (§2.4) the H2D is a true async DMA that
can outlive `execute_model` — under `--async-scheduling` no device sync covers it — so a
step-count deferral alone is *not* sufficient: the writer could reclaim the slot while
the DMA is still reading it. At `flush_releases` the reader records a CUDA event on the
compute stream (ordered after the previous step's H2D) and only sets its done flag for a
slot once that event has completed (non-blocking `event.query()`; a slot not yet done
simply waits one more dequeue). When the mapping is *not* pinned, `cudaMemcpyAsync` from
pageable host memory stages the copy synchronously before returning, so the slot is
released immediately. The writer requires all readers' done flags before reusing a slot,
so a slot is never overwritten while any reader's DMA is in flight.

> Assumes the multimodal H2D is issued on the worker's current/default compute stream
> (true today: mm inputs are copied eagerly, outside the decode CUDA graph). If a future
> vLLM issues that copy on a dedicated side stream, the event must be recorded at the
> copy site rather than at `flush_releases`.

### 2.4 Pinning — the zero-copy trap

The first version of the patch only halved the stall. Profiling showed the residual
entirely inside `cudaMemcpyAsync`: the zero-copy view references tmpfs pages the reader
process has never touched, so the H2D pays **first-touch page faults on ~50k pages plus
pageable-copy staging**. (The old deserialize, for all its waste, incidentally left the
bytes in hot process-local heap pages — which is why the baseline H2D was fast.)

Fix: each reader lazily `cudaHostRegister`s the whole arena mapping once (~1 s, first
use), after which every H2D from the arena is a **pinned-memory DMA** (~10 ms for
192 MB). Processes without a CUDA context skip registration silently.

## 3. Configuration

| Env var | Default | Meaning |
|---|---|---|
| `VLLM_SHM_TENSOR_ARENA` | `1` | Enable the arena (`0` disables; behavior reverts to stock). |
| `VLLM_SHM_TENSOR_ARENA_SLOTS` | `8` | Number of slots. |
| `VLLM_SHM_TENSOR_ARENA_SLOT_MB` | `256` | Slot size; tensors larger than this fall back to pickle. |
| `VLLM_SHM_TENSOR_ARENA_MIN_MB` | `8` | Minimum tensor size to divert; smaller tensors pickle as before. |

## 4. Validation

1. **Unit test**: a 199 MB bf16 tensor pushed through a real `MessageQueue` to two
   forked reader processes — byte-exact checksums, arena path confirmed, deferred
   release, slot reuse, and exhaustion/oversize fallbacks exercised. Enqueue of the
   199 MB payload: **66.7 ms** (vs ~1275 ms observed in production for the same size).
2. **Same-seed A/B**, interactive multimodal workload, uncapped input images, TP=4 ×2
   workers on one 8-GPU node, low qps so individual images decompose cleanly
   (~1.2k aligned requests):

   | TTFT (ms) | p50 | p90 | p99 | max | >1 s | >1.5 s |
   |---|---:|---:|---:|---:|---:|---:|
   | stock | 93 | 401 | 1321 | 2139 | 30 | 9 |
   | bigger MQ chunk (config lever) | 91 | 368 | 1287 | 2027 | 22 | 5 |
   | **arena + pinning** | **89** | **241** | **862** | **1375** | **7** | **0** |

   p50 unchanged → `reducer_override` adds no measurable overhead when no large tensor
   is present.
3. **Per-image stall** (worst offenders, CUDA-level all-GPU idle gap around the pixel
   H2D): 192 MB image 1318 ms → 681 ms (arena only) → **351 ms class** (arena +
   pinning); vision-encoder compute for the same image is ~80 ms.
4. **Load scaling**: on the same workload the stock build failed a 1.5 s e2e-p99 target
   even at very low arrival rates (the stall is a fixed per-image cost); with the patch
   the passing arrival rate is bounded by prefill/decode compute interference instead —
   transport no longer appears in the profile at the tail.

## 5. Limitations and future work

- **Slot release granularity**: releases are gated on a per-slot CUDA event recorded
  after the consuming H2D (§2.3), which closes the async-DMA reuse window on the pinned
  path. Bursts deeper than the slot count still safely fall back to pickle when the arena
  is exhausted.
- **Fallback observability**: arena exhaustion / oversize fallbacks are rate-limited
  log lines today; a counter metric would be better.
- **Scope**: the arena activates only when every queue reader is node-local. Remote
  readers (multi-node PP/TP) keep the existing socket path.
- **Generality**: the problem is generic to any multimodal model on TP > 1 with the
  multiproc executor; upstreaming the approach (or an equivalent out-of-band tensor
  channel) is worth discussing.
