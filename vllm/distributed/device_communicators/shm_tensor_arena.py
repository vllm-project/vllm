# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy shared-memory arena for large CPU tensors (multimodal fast path).

Problem: `MessageQueue.enqueue` (see `shm_broadcast.py`) pickles the whole
payload, and torch's default tensor pickling copies bytes into the pickle
stream on the writer and back out on every local reader. For a large
`pixel_values` tensor (up to hundreds of MB) this blocks the EngineCore step
loop for ~1s with all GPUs idle.

Fast path: `_ArenaPickler.reducer_override` intercepts qualifying CPU
tensors during pickling, memcpys them ONCE into a free arena slot, and
pickles a tiny `(arena_name, slot, nbytes, dtype, shape)` stub. Readers
rebuild a zero-copy view of the mapped slot (`torch.frombuffer`).

Slot lifecycle: single writer, n_reader readers, per-slot metadata
[written_flag, reader0_done, ..., readerN_done] (same protocol as
ShmRingBuffer). A slot is not released when rebuilt: the rebuilt tensor is
the SOURCE of an async H2D copy, and callers like chunked-prefill
`prompt_embeds` retain it across many steps. Release is tied via
`weakref.finalize` to the caller-retained object's GC, not a fixed
"next dequeue" schedule — see `ShmTensorArena.schedule_release` for the
mechanism (and its sharp edge: the tracked object must be exactly what the
caller ends up holding). Once queued, release is further gated on a CUDA
event on the pinned fast path, since the H2D can outlive the step that
issued it — see `flush_releases`.

The writer NEVER blocks: no free slot (or an oversized tensor) falls back to
the default pickle path, so worst case matches today's behavior.

Physical memory: slots are paged in lazily, so an arena on a queue that
never carries big tensors costs ~0. Creation is guarded by
`check_shm_free_space` (like `ShmRingBuffer`) so an undersized `/dev/shm`
fails fast instead of a `SIGBUS` mid-copy.
"""

import pickle
import weakref
from contextlib import suppress
from multiprocessing import shared_memory
from unittest.mock import patch

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Arena sizing — internal constants (kept off the env/CLI surface).
_ARENA_SLOTS = 8
_ARENA_SLOT_BYTES = 256 * 1024 * 1024
# Minimum contiguous CPU-tensor size to divert into the arena; smaller tensors
# take the out-of-band pickle-buffer path.
_ARENA_MIN_BYTES = 8 * 1024 * 1024

# Reader-side registry: arena shm name -> attached ShmTensorArena, keyed by
# name rather than a single slot since a process isn't guaranteed to attach
# to only one. Populated by `MessageQueue.create_from_handle`; consumed by
# the `_rebuild_arena_*` unpickle hooks. WEAK-value: a strong reference here
# would keep the arena (and its pinned mapping) alive for the process
# lifetime, and would need `MessageQueue.shutdown` to remove it explicitly --
# unsafe since shutdown() can run on a different thread than an in-flight
# `dequeue`. Instead an entry just disappears once nothing else references
# that arena.
_TENSOR_ARENAS: weakref.WeakValueDictionary[str, "ShmTensorArena"] = (
    weakref.WeakValueDictionary()
)


class ShmTensorArena:
    """Slotted shared-memory arena: one writer, n_reader zero-copy readers.

    Memory layout: [slot0 | slot1 | ... | slotN-1 | meta0 | meta1 | ... ]
    where each meta is (1 + n_reader) bytes: [written_flag, reader_done...].
    Slot states mirror ShmRingBuffer:
      written=0                      -> free (never written / being written)
      written=1, some reader_done=0 -> in use, cannot reuse
      written=1, all reader_done=1  -> consumed, can reuse
    """

    def __init__(
        self,
        n_reader: int,
        slot_bytes: int,
        n_slots: int,
        name: str | None = None,
        reader_rank: int = -1,
    ):
        self.n_reader = n_reader
        self.slot_bytes = slot_bytes
        self.n_slots = n_slots
        self.reader_rank = reader_rank  # -1 for the writer
        self.metadata_size = 1 + n_reader
        self.metadata_offset = slot_bytes * n_slots
        self.total_bytes = (slot_bytes + self.metadata_size) * n_slots
        self._next_slot = 0
        self._fallbacks = 0
        self._pin_attempted = False
        self._pinned = False
        self._pinned_ptr = 0
        # slots queued for release once the tensor `get_tensor` handed out is
        # garbage-collected (see `get_tensor`); flushed at the next dequeue.
        self._pending_release: list[int] = []
        # slots whose release is gated on an H2D-completion CUDA event: each
        # entry is (event, [slot_idx, ...]). Only populated on the pinned fast
        # path, where the source-side DMA is asynchronous (see flush_releases).
        self._deferred_releases: list[tuple[torch.cuda.Event, list[int]]] = []

        if name is None:
            self.is_creator = True
            # Same guard as ShmRingBuffer: SharedMemory/ftruncate can succeed
            # on an undersized /dev/shm, so check up front instead of SIGBUS
            # on first write past tmpfs capacity.
            from vllm.distributed.device_communicators.shm_broadcast import (
                check_shm_free_space,
            )

            check_shm_free_space(self.total_bytes, allocation_name="ShmTensorArena")
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes
            )
            assert self.shared_memory.buf is not None
            with self.shared_memory.buf[self.metadata_offset :] as meta:
                torch.frombuffer(meta, dtype=torch.uint8).fill_(0)
        else:
            self.is_creator = False
            # same resource_tracker workaround as ShmRingBuffer
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert self.shared_memory.size >= self.total_bytes
                except FileNotFoundError:
                    # deserialized on a different node; arena unused there
                    pass

    def handle(self):
        return (self.n_reader, self.slot_bytes, self.n_slots, self.shared_memory.name)

    def _meta(self, idx: int) -> memoryview:
        start = self.metadata_offset + idx * self.metadata_size
        assert self.shared_memory.buf is not None
        return self.shared_memory.buf[start : start + self.metadata_size]

    def _slot(self, idx: int, nbytes: int) -> memoryview:
        start = idx * self.slot_bytes
        assert self.shared_memory.buf is not None
        return self.shared_memory.buf[start : start + nbytes]

    # ---- writer side ----

    def write_tensor(self, t: torch.Tensor) -> int | None:
        """Copy tensor bytes into a free slot; return slot idx or None
        (caller must then fall back to the default pickle path)."""
        from vllm.distributed.device_communicators.shm_broadcast import memory_fence

        nbytes = t.numel() * t.element_size()
        if nbytes > self.slot_bytes:
            return None
        try:
            # Computed before claiming a slot: exotic tensors (e.g. conjugate
            # bit set) raise RuntimeError here, same as `_reduce_tensor`'s
            # identical view -- fail before claiming, not after.
            src = t.detach().reshape(-1).view(torch.uint8)
        except RuntimeError:
            return None
        memory_fence()
        for probe in range(self.n_slots):
            idx = (self._next_slot + probe) % self.n_slots
            with self._meta(idx) as meta:
                free = meta[0] == 0 or sum(meta[1:]) == self.n_reader
                if not free:
                    continue
                meta[0] = 0  # claim
            slot_mv = self._slot(idx, nbytes)
            try:
                dst = torch.frombuffer(slot_mv, dtype=torch.uint8, count=nbytes)
                dst.copy_(src)
            finally:
                slot_mv.release()
            with self._meta(idx) as meta:
                for i in range(1, self.n_reader + 1):
                    meta[i] = 0
                memory_fence()
                meta[0] = 1
                memory_fence()
            self._next_slot = (idx + 1) % self.n_slots
            return idx
        self._fallbacks += 1
        if self._fallbacks == 1 or self._fallbacks % 100 == 0:
            logger.info(
                "ShmTensorArena: no free slot (%d bytes, %d fallbacks so far); "
                "falling back to the out-of-band pickle path.",
                nbytes,
                self._fallbacks,
            )
        return None

    # ---- reader side ----

    def _ensure_pinned(self):
        """Pin the whole arena mapping via cudaHostRegister in THIS process
        (lazy, once). Without it the HtoD of a zero-copy tensor pays
        first-touch page faults on the tmpfs mapping plus pageable staging
        (~hundreds of ms for 200MB); registration allocates+pins the pages
        once, making every later HtoD a true DMA. Failure (no CUDA in this
        process, etc.) is fine — the copy still works, just slower."""
        if self._pin_attempted:
            return
        self._pin_attempted = True
        try:
            if not current_platform.is_cuda_alike():
                return
            import ctypes

            buf = self.shared_memory.buf
            assert buf is not None
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            ret = current_platform.cudart().cudaHostRegister(ptr, self.total_bytes, 0)
            self._pinned = int(ret) == 0
            if self._pinned:
                self._pinned_ptr = ptr
            logger.info(
                "ShmTensorArena: cudaHostRegister(%d MB) -> %s",
                self.total_bytes >> 20,
                "pinned" if self._pinned else f"error {int(ret)}",
            )
        except Exception as e:
            logger.info("ShmTensorArena: host-register skipped: %s", e)

    def _unpin(self):
        """Undo the pin from _ensure_pinned via cudaHostUnregister; must run
        before the mapping is closed. Failures are ignored — at interpreter
        shutdown the CUDA context may already be gone, and the registration
        dies with the process anyway."""
        if not self._pinned:
            return
        self._pinned = False
        with suppress(Exception):
            current_platform.cudart().cudaHostUnregister(self._pinned_ptr)

    def get_tensor(
        self, idx: int, nbytes: int, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Zero-copy view of a slot as a tensor. Does NOT itself schedule
        the slot's release -- see `schedule_release`, which every unpickle
        hook using this must call on whichever object it actually hands
        back to the caller."""
        self._ensure_pinned()
        # NOT a context-manager view: the tensor must keep the mapping alive.
        slot_mv = self._slot(idx, nbytes)
        t8 = torch.frombuffer(slot_mv, dtype=torch.uint8, count=nbytes)
        return t8.view(dtype).view(shape)

    def schedule_release(self, obj, idx: int) -> None:
        """Queue slot `idx` for release once `obj` is garbage-collected.

        Tied to GC (`weakref.finalize`) rather than the reader's next
        `dequeue`, since callers like chunked-prefill `prompt_embeds` retain
        the tensor across many steps -- a fixed next-dequeue release would
        let the writer reuse (and mutate) the slot out from under them. The
        CUDA-event gate in `flush_releases` still applies once queued.

        `obj` MUST be the *exact* object the caller ends up holding, not an
        intermediate value another type's rebuild wraps without keeping a
        Python reference to. E.g. `torch.nn.Parameter(t, requires_grad=False)`
        shares `t`'s storage at the C++ level but keeps no Python reference
        to `t`, so `_rebuild_arena_parameter` schedules release on the
        constructed `Parameter`, not on `t`. See `_ArenaPickler.
        reducer_override` for the fixed set of types this is known safe for.

        Residual limitation: a caller that drops `obj` itself but keeps a
        view/slice of it would not delay release -- torch views keep the
        underlying storage alive at the C++ level, independent of this
        Python-object-level finalizer.
        """
        # Binds to the current list OBJECT. `flush_releases` must mutate it
        # in place (`.clear()`), never rebind `self._pending_release` --
        # doing so would orphan any finalizer already registered here.
        weakref.finalize(obj, self._pending_release.append, idx)

    def _mark_released(self, idxs: list[int]):
        """Set THIS reader's done flag on the given slots; a slot becomes
        reusable once every reader has done so."""
        from vllm.distributed.device_communicators.shm_broadcast import memory_fence

        for idx in idxs:
            with self._meta(idx) as meta:
                meta[1 + self.reader_rank] = 1
        memory_fence()

    def _record_release_event(self):
        """Record a CUDA event on the current (compute) stream, or return None
        if this process has no usable CUDA context. Called from flush_releases,
        which runs after the previous step's H2D was enqueued on that same
        stream, so the event is ordered strictly after that H2D."""
        try:
            event = current_platform.Event()
            event.record()
            return event
        except Exception:
            return None

    def flush_releases(self):
        """Retire this reader's arena slots that are ready for writer reuse.

        Called on every `MessageQueue.dequeue`. `_pending_release` holds
        slots whose tensor was just garbage-collected (see `get_tensor`). On
        the pinned fast path that tensor was the source of an async
        `non_blocking=True` H2D that can outlive the collection, so instead
        of marking released immediately we record a CUDA event (ordered
        after that H2D on the same stream) and move the slot to
        `_deferred_releases` until `event.query()` says it's done -- a
        not-yet-complete slot just waits one more `dequeue`. Unpinned (or no
        CUDA context), the H2D already staged synchronously, so the slot is
        safe to release right away.

        A slot only becomes free for the writer (`write_tensor`) once every
        reader has marked it released (`_mark_released`).
        """
        if self._deferred_releases:
            still_pending = []
            for deferred_event, idxs in self._deferred_releases:
                if deferred_event.query():
                    self._mark_released(idxs)
                else:
                    still_pending.append((deferred_event, idxs))
            self._deferred_releases = still_pending

        if not self._pending_release:
            return
        # Snapshot-and-clear in place, never rebind (see schedule_release).
        idxs = list(self._pending_release)
        self._pending_release.clear()

        event = self._record_release_event() if self._pinned else None
        if event is None:
            self._mark_released(idxs)
        else:
            self._deferred_releases.append((event, idxs))

    def __del__(self):
        if hasattr(self, "shared_memory"):
            self._unpin()
            try:
                self.shared_memory.close()
                if self.is_creator:
                    self.shared_memory.unlink()
            except BufferError:
                # zero-copy tensor views may still hold exported pointers at
                # interpreter shutdown; the mapping dies with the process.
                pass


def _rebuild_arena_tensor(arena_name, slot_idx, nbytes, dtype_str, shape):
    """Unpickle hook: rebuild a tensor as a zero-copy view of an arena slot."""
    arena = _TENSOR_ARENAS[arena_name]
    t = arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    arena.schedule_release(t, slot_idx)
    return t


def _rebuild_arena_parameter(arena_name, slot_idx, nbytes, dtype_str, shape):
    """Unpickle hook for `torch.nn.Parameter`: schedules release on the
    constructed `Parameter`, not the intermediate tensor it wraps (see
    `ShmTensorArena.schedule_release`)."""
    arena = _TENSOR_ARENAS[arena_name]
    t = arena.get_tensor(slot_idx, nbytes, getattr(torch, dtype_str), shape)
    param = torch.nn.Parameter(t, requires_grad=False)
    arena.schedule_release(param, slot_idx)
    return param


# Types `_ArenaPickler.reducer_override` knows how to divert safely, mapped
# to the rebuild hook that schedules release on the right object (see
# `ShmTensorArena.schedule_release`). Any other type -- including
# `torch.Tensor` subclasses -- falls through to the normal pickle path,
# since we can't assume its `__reduce_ex__` keeps a Python reference to the
# tensor it's built from.
_ARENA_REBUILD_FNS = {
    torch.Tensor: _rebuild_arena_tensor,
    torch.nn.Parameter: _rebuild_arena_parameter,
}


class _ArenaPickler(pickle.Pickler):
    """Pickler that diverts large contiguous CPU tensors into the arena.

    `reducer_override` is consulted before an object's normal reduction. For
    a qualifying tensor (exact type in `_ARENA_REBUILD_FNS`, CPU, strided,
    contiguous, not `requires_grad`, above the divert-size threshold -- the
    same criteria `_reduce_tensor` uses, plus size) it does one memcpy into a
    free slot and returns a tiny rebuild stub. Anything that doesn't qualify,
    or finds the arena full, returns `NotImplemented` and falls through to
    the normal out-of-band pickle path, exactly as if no arena were attached.
    """

    def __init__(self, file, arena: ShmTensorArena, buffer_callback=None):
        super().__init__(
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
            buffer_callback=buffer_callback,
        )
        self.arena = arena

    def reducer_override(self, obj):
        rebuild_fn = _ARENA_REBUILD_FNS.get(type(obj))
        if (
            rebuild_fn is not None
            and obj.device.type == "cpu"
            and obj.layout == torch.strided
            and obj.is_contiguous()
            and not obj.requires_grad
            and obj.numel() * obj.element_size() >= _ARENA_MIN_BYTES
        ):
            idx = self.arena.write_tensor(obj)
            if idx is not None:
                return (
                    rebuild_fn,
                    (
                        self.arena.shared_memory.name,
                        idx,
                        obj.numel() * obj.element_size(),
                        str(obj.dtype).removeprefix("torch."),
                        tuple(obj.shape),
                    ),
                )
        return NotImplemented
