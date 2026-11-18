# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer side of the sharded-RDT (pull-based NIXL) backend.

Every trainer rank owns a producer server: an internal Ray actor that shares
the rank's GPU over CUDA IPC and serves the workers' pulls, each packed into
a block of an arena registered with NIXL. Each sync, the rank gathers its
owned names unit by unit and publishes each unit to its server; the sender
also drives the inference-side start/update/finish calls. The protocol is in
`sharded_rdt_common`.
"""

import contextlib
import gc
import itertools
import os
import threading
import time
import uuid
import weakref
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

import ray
import torch
from torch.multiprocessing.reductions import (
    StorageWeakRef,
    rebuild_cuda_tensor,
    reduce_tensor,
)

from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    DEFAULT_BUFFER_BYTES,
    DEFAULT_NUM_BUFFERS,
    PIECE_ALIGNMENT,
    FlatPiece,
    RdtRouter,
    WireBucket,
    arena_bytes,
    box_slices,
    check_ray_rdt_version,
    deregister_nixl_memory,
    flat_layout,
    gather_units,
    meta_nbytes,
    piece_nbytes,
    register_nixl_memory,
    validate_int,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Units a rank gathers ahead of the ones its consumers still read.
DEFAULT_GATHER_LOOKAHEAD = 2
# Seconds without progress before a producer fails the sync: a consumer that
# dies mid-sync never frees its buckets, which would wedge the gather loop.
DEFAULT_STALL_TIMEOUT_S = 300.0
_STALL_POLL_S = 5.0
# Storages at most this large share one export slot and CUDA-IPC handle.
_EXPORT_RING_MAX_BYTES = 64 << 20


@dataclass
class ShardedRDTTrainerInitInfo(TrainerInitInfo):
    """Trainer init info for the sharded-RDT backend; the same on every rank
    except `rank` (rank 0 is the sender)."""

    backend: ClassVar[str] = "sharded_rdt"

    num_consumers: int
    """Inference workers across the whole fleet."""
    workers_per_replica: int = 0
    """Inference workers per deployment, for the M:N routing; 0 means one
    deployment."""
    trainer_actor_namespace: str | None = None
    """Ray namespace of the producer actors; the workers resolve them there."""
    buffer_bytes: int = DEFAULT_BUFFER_BYTES
    """Most bytes one pull moves, and the size of a gather unit."""
    num_buffers: int = DEFAULT_NUM_BUFFERS
    """Pulls each worker keeps in flight."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Units the gather loop runs ahead of the consumers."""
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    """Seconds without progress before a producer fails the sync."""

    def __post_init__(self) -> None:
        validate_int("buffer_bytes", self.buffer_bytes, minimum=1)
        validate_int("num_buffers", self.num_buffers, minimum=1)
        validate_int("num_consumers", self.num_consumers, minimum=1)


@dataclass(frozen=True, slots=True)
class _ServedBucket:
    unit: int
    pieces: tuple[tuple[str, tuple[slice, ...]], ...]
    nbytes: int
    layout: tuple[FlatPiece, ...]


@dataclass(frozen=True, slots=True)
class _Sources:
    """One bucket's source boxes for one generation of its unit: the
    contiguous ones as 8-byte words where aligned, else as bytes, each with
    the split sizes (in words or bytes) whose odd chunks of the bucket are
    the pieces they fill; and the strided ones as runs `(first piece,
    count)`, each one strided view."""

    gen: int
    word_splits: tuple[int, ...]
    word_srcs: tuple[torch.Tensor, ...]
    byte_splits: tuple[int, ...]
    byte_srcs: tuple[torch.Tensor, ...]
    typed_idx: tuple[tuple[int, int], ...]
    typed_srcs: tuple[torch.Tensor, ...]


def _splits(
    idx: list[int], layout: Sequence[FlatPiece], nbytes: int, unit: int
) -> tuple[int, ...]:
    """Split sizes, in `unit` bytes, cutting a bucket of `nbytes` into gaps
    and pieces `idx`, so that its odd chunks are those pieces."""
    splits, end = [], 0
    for i in idx:
        start, size, *_ = layout[i]
        splits += [(start - end) // unit, size // unit]
        end = start + size
    return (*splits, (nbytes - end) // unit)


def _sources(
    gen: int, views: list[torch.Tensor], layout: Sequence[FlatPiece], nbytes: int
) -> _Sources:
    """Split a bucket's source boxes into contiguous ones and runs of strided
    ones. A run is consecutive strided boxes of one shape and stride from one
    storage, at evenly spaced offsets, landing back to back in the bucket:
    e.g. the TP column slices of consecutive experts' weights. It is copied
    as one view with a leading run dimension, one kernel instead of one per
    box. Contiguous boxes copy as 8-byte words when their address and size
    allow (pieces start on 16 B in the bucket), which moves far more bytes
    per element than a byte copy."""
    word_idx: list[int] = []
    byte_idx: list[int] = []
    word_srcs, byte_srcs = [], []
    runs: list[tuple[int, int, int | None]] = []
    for i, view in enumerate(views):
        if view.is_contiguous():
            raw = view.view(-1).view(torch.uint8)
            if raw.data_ptr() % 8 == 0 and raw.numel() % 8 == 0:
                word_idx.append(i)
                word_srcs.append(raw.view(torch.int64))
            else:
                byte_idx.append(i)
                byte_srcs.append(raw)
            continue
        if runs:
            first, count, step = runs[-1]
            last, base = first + count - 1, views[first]
            spacing = view.storage_offset() - views[last].storage_offset()
            if (
                i == last + 1
                and layout[i][0] == layout[last][0] + layout[last][1]
                and view.dtype == base.dtype
                and view.shape == base.shape
                and view.stride() == base.stride()
                and view.untyped_storage().data_ptr()
                == base.untyped_storage().data_ptr()
                and spacing > 0
                and step in (None, spacing)
            ):
                runs[-1] = (first, count + 1, spacing)
                continue
        runs.append((i, 1, None))
    typed_srcs = []
    for first, count, step in runs:
        base = views[first]
        if count > 1:
            base = torch.as_strided(
                base,
                (count, *base.shape),
                (step, *base.stride()),
                base.storage_offset(),
            )
        typed_srcs.append(base)
    return _Sources(
        gen,
        _splits(word_idx, layout, nbytes, 8),
        tuple(word_srcs),
        _splits(byte_idx, layout, nbytes, 1),
        tuple(byte_srcs),
        tuple((first, count) for first, count, _ in runs),
        tuple(typed_srcs),
    )


class _RDTProducerServer:
    """One trainer rank's serve surface: an internal Ray actor on the rank's
    GPU. The engine publishes gathered units into it over CUDA IPC; workers
    register their buckets and pull them.

    A plain class; the engine wraps it with `ray.remote` at spawn.
    """

    def __init__(
        self,
        *,
        served: dict[str, tuple[str, tuple[int, ...]]],
        units: dict[str, int],
        arena_bytes: int,
        bucket_bytes: int,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
    ) -> None:
        self._device_index = torch.accelerator.current_device_index()
        self._served = {
            name: (getattr(torch, dtype), tuple(shape))
            for name, (dtype, shape) in served.items()
        }
        self._unit_of = dict(units)
        self._arena_bytes = int(arena_bytes)
        # the largest bucket a consumer may register (its `buffer_bytes`)
        self._bucket_bytes = int(bucket_bytes)
        self._stall_timeout = float(stall_timeout_s)

        # One lock; a pull waits on its unit's condition, then on arena room,
        # and the engine-facing calls on releases, so each event wakes only
        # the waits it can unblock.
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)  # releases
        self._room = threading.Condition(self._lock)  # freed arena blocks
        self._unit_conds: dict[int, threading.Condition] = {}
        self._error: BaseException | None = None
        self._failed_sync: int | None = None
        self._last_progress = time.monotonic()

        # consumer -> its buckets in pull order, and their count per unit
        self._buckets: dict[int, list[_ServedBucket]] = {}
        self._needs: dict[int, Counter[int]] = {}

        # per sync
        self._sync_idx = -1
        self._live: frozenset[int] = frozenset()
        self._freed: dict[int, Counter[int]] = {}
        self._resident: list[int] = []  # published, not released
        self._freed_pending: list[int] = []  # released, not yet collected
        # the registered arena, and its served, unfreed blocks by offset
        self._arena: torch.Tensor | None = None
        self._blocks: dict[int, int] = {}
        self._serve_wait, self._served_count = 0.0, 0
        self._base_cache: dict[tuple, torch.Tensor] = {}
        # Per unit, its last publish: the signature (where its tensors are),
        # the rebuilt tensors, and their generation, a new one whenever the
        # tensors move. Per consumer and seq, the
        # bucket's source boxes, built once per generation of its unit.
        self._published: dict[int, tuple[Any, dict[str, torch.Tensor], int]] = {}
        self._gens = itertools.count(1)
        self._sources: dict[int, dict[int, _Sources]] = {}
        # per unit, the trainer's event its published tensors are ready at
        self._ready: dict[int, torch.cuda.Event | None] = {}
        # Packs run one at a time: the GIL serializes their Python anyway, and
        # concurrent packs only convoy on it, which multiplies the cost of
        # every tensor op. The device sync after a pack is outside it.
        self._pack_lock = threading.Lock()

        # A frozen post-init object graph keeps gen-2 GC from stopping the
        # world mid-serve.
        gc.collect()
        gc.freeze()

    def warmup_nixl(self) -> None:
        """Create the NIXL agent and register the arena now, while the rank's
        GPU is idle and before any rank enters a collective: creating the
        agent lazily deadlocks on EFA-class fabrics."""
        device = torch.device("cuda", self._device_index)
        arena = torch.empty(self._arena_bytes, dtype=torch.uint8, device=device)
        register_nixl_memory(arena)
        self._arena = arena

    # ---------------- liveness ----------------

    def _unit_cond(self, unit: int) -> threading.Condition:
        """The condition a pull for `unit` waits on. Caller holds `_lock`."""
        cond = self._unit_conds.get(unit)
        if cond is None:
            cond = self._unit_conds[unit] = threading.Condition(self._lock)
        return cond

    def _notify_all(self) -> None:
        """Wake every wait (an error or a new sync). Caller holds `_lock`."""
        for cond in (self._cond, self._room, *self._unit_conds.values()):
            cond.notify_all()

    def _wait_for(
        self, cond: threading.Condition, blocked: Callable[[], bool], what: str
    ) -> None:
        """Wait on `cond` while `blocked()`, returning early on an error;
        fail the sync when nothing on this producer progressed for the stall
        timeout. Caller holds `_lock`."""
        while blocked() and self._error is None:
            cond.wait(_STALL_POLL_S)
            stalled = time.monotonic() - self._last_progress
            if blocked() and self._error is None and stalled >= self._stall_timeout:
                msg = (
                    f"RDT stall: no progress for {stalled:.0f}s waiting for "
                    f"{what}. A consumer most likely died mid-sync: "
                    f"{len(self._resident)} unit(s) published and unreleased."
                )
                logger.error("[rdt-stall] %s", msg)
                self._error = RuntimeError(msg)
                self._notify_all()
        if self._error is not None:
            raise RuntimeError(f"sharded RDT sync failed: {self._error!r}")

    # ---------------- engine-facing ----------------

    def begin_sync(self, live_consumer_ids: list[int], sync_idx: int) -> None:
        """Open sync `sync_idx` for the given live consumers."""
        with self._cond:
            if self._failed_sync != sync_idx:  # failed before this rank began it
                self._error = None
            self._sync_idx = int(sync_idx)
            self._live = frozenset(int(c) for c in live_consumer_ids)
            self._freed = {c: Counter() for c in self._live}
            self._serve_wait, self._served_count = 0.0, 0
            self._resident.clear()
            self._freed_pending.clear()
            self._base_cache.clear()
            self._last_progress = time.monotonic()
            self._notify_all()

    def publish_unit(self, unit: int, entries: tuple) -> None:
        """Rebuild one gathered unit's CUDA-IPC tensors and make them
        servable; never blocks. `entries` is `(storages, views, event)`: one
        export per storage, per name `(sid, dtype_name, shape, stride,
        storage_offset)`, and the IPC handle of the event the unit's tensors
        are ready at (or None), which this unit's packs wait for on the
        device.

        Raises:
            ValueError: a tensor does not match its metadata.

        """
        storages, views, event = entries
        self._ready[unit] = (
            torch.cuda.Event.from_ipc_handle(
                torch.device("cuda", self._device_index), event
            )
            if event is not None
            else None
        )
        # Where the tensors are: each storage's CUDA allocation, size and
        # offset in it (not its per-export ref counter and event), and the
        # views. Rebuilding every publish keeps CUDA IPC's ref counting
        # exact; the same signature keeps the unit's cached sources.
        signature = (
            tuple((sid, args[7], args[8], args[9]) for sid, args in storages.items()),
            views,
        )
        bases: dict[int, torch.Tensor] = {}
        for sid, reduce_args in storages.items():
            cached = self._base_cache.get(reduce_args)
            if cached is None:
                args = list(reduce_args)
                args[6] = self._device_index  # the exporter's device index
                cached = self._base_cache[reduce_args] = rebuild_cuda_tensor(*args)
            bases[sid] = cached
        rebuilt: dict[str, torch.Tensor] = {}
        for name, (sid, dtype_name, shape, stride, offset) in views.items():
            dtype = getattr(torch, dtype_name)
            rebuilt[name] = torch.as_strided(
                bases[sid].view(dtype), shape, stride, offset
            )
        self._publish(unit, rebuilt, signature)

    def _publish(
        self, unit: int, tensors: dict[str, torch.Tensor], signature: Any = None
    ) -> None:
        """Make one unit's tensors servable.

        Raises:
            ValueError: a tensor does not match its metadata.

        """
        published = self._published.get(unit)
        if (
            published is not None
            and signature is not None
            and published[0] == signature
        ):
            # the same memory as last sync: the cached sources still read it
            published = (signature, tensors, published[2])
        else:
            for name, t in tensors.items():
                if self._served.get(name) != (t.dtype, tuple(t.shape)):
                    raise ValueError(
                        f"{name!r} is {t.dtype} {tuple(t.shape)}, but its "
                        f"metadata says {self._served.get(name)}"
                    )
            published = (signature, tensors, next(self._gens))
        with self._lock:
            self._published[unit] = published
            self._resident.append(unit)
            self._last_progress = time.monotonic()
            self._unit_cond(unit).notify_all()
            self._release_if_done(unit)

    def wait_freed(self) -> list[int]:
        """Block until a published unit is released; return the released ones."""
        with self._cond:
            self._wait_for(
                self._cond, lambda: not self._freed_pending, "a released unit"
            )
            freed, self._freed_pending = self._freed_pending, []
            return freed

    def end_sync(self) -> list[int]:
        """Block until every published unit is released; return the rest."""
        with self._cond:
            self._wait_for(
                self._cond, lambda: bool(self._resident), "every unit's release"
            )
            freed, self._freed_pending = self._freed_pending, []
            self._base_cache.clear()
            logger.info(
                "Sharded RDT producer sync %d: served %d buckets, which waited "
                "%.2f s in total for their unit or arena room",
                self._sync_idx,
                self._served_count,
                self._serve_wait,
            )
            return freed

    def set_gather_error(self, message: str, sync_idx: int | None = None) -> None:
        """Fail sync `sync_idx` (the current one by default), even if this
        rank has not begun it yet: every blocked wait here raises."""
        with self._lock:
            self._error = RuntimeError(message)
            self._failed_sync = self._sync_idx if sync_idx is None else sync_idx
            self._notify_all()

    # ---------------- consumer-facing ----------------

    def register_consumer(self, consumer_id: int, buckets: list[WireBucket]) -> None:
        """Record the buckets `consumer_id` pulls from here, in pull order.
        Replaces an earlier registration (a re-planned consumer).

        Raises:
            ValueError: a bucket names a tensor this rank does not serve, a
                box outside its tensor, or tensors of two units, or is larger
                than the arena.

        """
        served: list[_ServedBucket] = []
        for i, wire in enumerate(buckets):
            pieces, units, nbytes, boxes, dtypes = [], set(), 0, [], []
            for name, box in wire:
                if name not in self._served:
                    raise ValueError(
                        f"consumer {consumer_id} bucket {i}: {name!r} is not "
                        "served by this trainer rank"
                    )
                dtype, shape = self._served[name]
                if len(box) != len(shape) or any(
                    not 0 <= a < b <= n for (a, b), n in zip(box, shape)
                ):
                    raise ValueError(f"{name!r}: box {box} is outside {shape}")
                pieces.append((name, box_slices(box)))
                boxes.append(box)
                dtypes.append(dtype)
                units.add(self._unit_of[name])
                nbytes += piece_nbytes(box, dtype)
            if len(units) != 1:
                raise ValueError(
                    f"consumer {consumer_id} bucket {i} spans units {sorted(units)}"
                )
            if nbytes > self._bucket_bytes:
                raise ValueError(
                    f"consumer {consumer_id} bucket {i} is {nbytes} B, over "
                    f"the {self._bucket_bytes} B bucket: the consumer cut its "
                    "buckets with a larger buffer_bytes"
                )
            layout = flat_layout(boxes, dtypes)
            served.append(
                _ServedBucket(units.pop(), tuple(pieces), nbytes, tuple(layout))
            )
        with self._lock:
            self._buckets[consumer_id] = served
            self._sources[consumer_id] = {}
            self._needs[consumer_id] = Counter(b.unit for b in served)
            for unit in list(self._resident):
                self._release_if_done(unit)
            self._last_progress = time.monotonic()

    @ray.method(tensor_transport="nixl")
    def bucket(self, consumer_id: int, seq: int, sync_idx: int) -> list[torch.Tensor]:
        """Serve one registered bucket of sync `sync_idx` once its unit is
        gathered and the arena has room: its boxes packed, with one fused
        copy, into an arena block at `flat_layout`.

        Raises:
            ValueError: an unregistered bucket, or a sync this producer left.

        """
        try:
            spec = self._buckets[consumer_id][seq]
        except (KeyError, IndexError):
            raise ValueError(
                f"consumer {consumer_id} pulled bucket {seq}, which it never "
                "registered here"
            ) from None
        t0 = time.perf_counter()
        with self._lock:
            self._wait_for(
                self._unit_cond(spec.unit),
                lambda: (
                    self._sync_idx < sync_idx
                    or (self._sync_idx == sync_idx and spec.unit not in self._resident)
                ),
                f"unit {spec.unit} of sync {sync_idx}",
            )
            # the unit stays resident until this bucket is served and freed
            self._wait_for(
                self._room,
                lambda: (
                    self._sync_idx == sync_idx
                    and self._carve(spec.nbytes, dry_run=True) < 0
                ),
                f"arena room for unit {spec.unit} of sync {sync_idx}",
            )
            if self._sync_idx != sync_idx:
                raise ValueError(f"pull for sync {sync_idx}, now {self._sync_idx}")
            offset = self._carve(spec.nbytes)
            self._serve_wait += time.perf_counter() - t0
            self._served_count += 1
            _, tensors, gen = self._published[spec.unit]
            ready = self._ready.get(spec.unit)
            sources = self._sources[consumer_id]
            self._last_progress = time.monotonic()
        try:
            flat = self._pack_bucket(sources, seq, spec, offset, tensors, gen, ready)
        except BaseException:
            self._freed_bucket(-1, spec.unit, sync_idx, offset)  # only the block
            raise
        del tensors
        self._track_free(flat, consumer_id, spec.unit, sync_idx, offset)
        return [flat]

    def _carve(self, nbytes: int, *, dry_run: bool = False) -> int:
        """First fit for `nbytes` in the arena: its offset, or -1 (no room).
        Caller holds `_lock`."""
        start = 0
        for offset, size in sorted(self._blocks.items()):
            if offset - start >= nbytes:
                break
            start = offset + size
        if start + nbytes > self._arena_bytes:
            return -1
        if not dry_run:
            self._blocks[start] = nbytes
        return start

    def _pack_bucket(
        self,
        sources: dict[int, _Sources],
        seq: int,
        spec: _ServedBucket,
        offset: int,
        tensors: dict[str, torch.Tensor],
        gen: int,
        ready: torch.cuda.Event | None,
    ) -> torch.Tensor:
        """Pack bucket `seq` into its arena block, one pack at a time, and
        wait for its copies: the contiguous boxes as words or bytes with one
        fused copy each, each kind cut from the block with one split, and the
        strided ones (e.g. TP column slices) typed, in runs. One strided box
        or a second dtype would otherwise put every box on the per-tensor
        path, a kernel each."""
        assert self._arena is not None
        flat = self._arena[offset : offset + spec.nbytes]
        with self._pack_lock:
            if ready is not None:
                torch.cuda.current_stream(self._device_index).wait_event(ready)
            src = sources.get(seq)
            if src is None or src.gen != gen:
                src = sources[seq] = _sources(
                    gen,
                    [tensors[name][sl] for name, sl in spec.pieces],
                    spec.layout,
                    spec.nbytes,
                )
            if src.word_srcs:
                words = flat.view(torch.int64).split(src.word_splits)[1::2]
                torch._foreach_copy_(words, src.word_srcs)
            if src.byte_srcs:
                torch._foreach_copy_(flat.split(src.byte_splits)[1::2], src.byte_srcs)
            if src.typed_srcs:
                dsts = []
                for first, count in src.typed_idx:
                    start, nbytes, dtype, shape = spec.layout[first]
                    run = flat[start : start + count * nbytes].view(dtype)
                    dsts.append(
                        run.view(shape) if count == 1 else run.view(count, *shape)
                    )
                torch._foreach_copy_(dsts, src.typed_srcs)
        if flat.is_cuda:
            # Ray synchronizes the device under its cache lock; do it here so
            # its wait is empty.
            torch.cuda.current_stream(flat.device).synchronize()
        return flat

    def _track_free(self, flat: torch.Tensor, *key: int) -> None:
        """Account the bucket once Ray drops its last reference to `flat`,
        i.e. frees the object after the consumer's read completed (or the
        consumer died). Nothing here may reference `flat` past the return."""
        weakref.finalize(flat, self._freed_bucket, *key)

    def _freed_bucket(
        self, consumer_id: int, unit: int, sync_idx: int, offset: int
    ) -> None:
        """A served bucket's arena block at `offset` is free, and the bucket
        no longer holds its unit."""
        with self._lock:
            self._blocks.pop(offset, None)
            self._room.notify_all()
            if sync_idx == self._sync_idx and consumer_id in self._freed:
                self._freed[consumer_id][unit] += 1
                self._release_if_done(unit)
            self._last_progress = time.monotonic()

    def _release_if_done(self, unit: int) -> None:
        """Release `unit` once every live consumer registered and freed all
        its buckets of it. Caller holds `_lock`."""
        if unit not in self._resident:
            return
        for c in self._live:
            needs = self._needs.get(c)
            if needs is None or self._freed[c][unit] < needs[unit]:
                return
        self._resident.remove(unit)
        self._freed_pending.append(unit)
        self._cond.notify_all()

    def shutdown(self) -> None:
        if self._arena is not None and self._arena.is_cuda:
            with contextlib.suppress(Exception):
                deregister_nixl_memory(self._arena)
            self._arena = None


class ShardedRDTTrainerWeightTransferEngine(
    TrainerWeightTransferEngine[ShardedRDTTrainerInitInfo]
):
    """Trainer-side engine for the sharded-RDT backend, on every trainer rank.

    `send_weights` gathers this rank's names unit by unit from the source,
    publishes each to the rank's producer server, and on the sender drives
    the inference-side calls. Ownership comes from `ParamMeta.owners`, the
    same on every rank, so no exchange is needed to plan who serves what.
    """

    init_info_cls = ShardedRDTTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        init_info: ShardedRDTTrainerInitInfo,
    ) -> None:
        super().__init__(
            client=client, source=source, is_sender=is_sender, rank=init_info.rank
        )
        self._init_info = init_info
        self._server: Any = None
        self._server_name: str | None = None
        self._server_names: list[str] | None = None
        self._world = 1
        self._meta: list[ParamMeta] = []
        self._units: list[int] = []  # per metadata name
        self._owner_sets: list[list[int]] = []
        self._name_owner_class: list[int] = []
        self._owned_names: set[str] = set()
        # this rank's units in gather order, each with its owned names
        self._owned_units: dict[int, list[str]] = {}
        # Gathered tensors shared with the server, by unit, and the event
        # that orders the server's reads after them: a CUDA-IPC export must
        # outlive its import, so they stay until released.
        self._inflight: dict[int, tuple[dict[str, torch.Tensor], Any]] = {}
        # Reusable export slots for small tensors, one per resident unit; kept
        # across syncs, since reallocating them each sync fragments the
        # caching allocator.
        self._export_ring: list[torch.Tensor | None] = []
        self._export_ring_args: list[tuple | None] = []
        self._slot_of_unit: dict[int, int] = {}
        self._sync_idx = -1

    def _remote(self, method: str, *args: Any) -> Any:
        """Call a server method; return its ref. The seam tests replace."""
        return getattr(self._server, method).remote(*args)

    def _rpc(self, method: str, *args: Any) -> Any:
        return ray.get(self._remote(method, *args))

    # ---------------- construction ----------------

    @classmethod
    def trainer_init(
        cls,
        init_info: ShardedRDTTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError(
                "Sharded RDT trainer weight transfer requires a WeightSource."
            )
        if not source.implements_iter_full():
            raise TypeError(
                "Sharded RDT serves full tensors, but "
                f"{type(source).__name__} does not implement iter_full()."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            init_info=init_info,
        )
        if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            logger.warning(
                "Sharded-RDT trainer: PYTORCH_CUDA_ALLOC_CONF enables "
                "expandable_segments, which makes CUDA-IPC publishing several "
                "times slower; disable it around weight sync."
            )
        world = engine._world_size()
        engine._plan(world, engine.rank)
        engine._spawn_server()
        # Every server must exist before the sender's init reaches the
        # workers, which register with all of them; the all-gather of server
        # names is that barrier.
        engine._server_names = engine._all_gather_server_names(world, engine.rank)
        if engine.is_sender:
            engine.client.init_weight_transfer_engine(engine.get_worker_init_payload())
        return engine

    def get_worker_init_payload(self) -> dict:
        """The worker-side init payload, from retained state; no collective,
        so a restarted inference engine can rejoin between syncs.

        Raises:
            RuntimeError: before `trainer_init` gathered the server names.

        """
        if self._server_names is None:
            raise RuntimeError("get_worker_init_payload needs trainer_init first.")
        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferInitInfo,
        )

        ii = self._init_info
        return asdict(
            ShardedRDTWeightTransferInitInfo(
                trainer_actor_names=self._server_names,
                trainer_actor_namespace=ii.trainer_actor_namespace,
                names=[m.name for m in self._meta],
                dtype_names=[_dtype_name(m.dtype) for m in self._meta],
                shapes=[list(m.shape) for m in self._meta],
                owner_sets=self._owner_sets,
                name_owner_class=self._name_owner_class,
                num_consumers=ii.num_consumers,
                workers_per_replica=ii.workers_per_replica,
                buffer_bytes=ii.buffer_bytes,
                num_buffers=ii.num_buffers,
            )
        )

    def _world_size(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != self.rank:
                raise ValueError(
                    f"Sharded RDT needs TrainerInitInfo.rank ({self.rank}) to be "
                    f"the default process group rank "
                    f"({torch.distributed.get_rank()})."
                )
            return torch.distributed.get_world_size()
        return 1

    def _plan(self, world: int, rank: int) -> None:
        """Owner table, gather units, and this rank's share of both."""
        assert self.source is not None
        self._world = world
        self._meta = list(self.source.metadata)
        everyone = tuple(range(world))
        class_of: dict[tuple[int, ...], int] = {}
        self._owner_sets, self._name_owner_class = [], []
        for m in self._meta:
            owners = everyone if m.owners is None else tuple(m.owners)
            if max(owners) >= world:
                raise ValueError(
                    f"parameter {m.name!r} is owned by trainer ranks {owners}, "
                    f"but there are only {world} trainer ranks."
                )
            if owners not in class_of:
                class_of[owners] = len(self._owner_sets)
                self._owner_sets.append(list(owners))
            self._name_owner_class.append(class_of[owners])
        self._units = gather_units(
            [meta_nbytes(m) for m in self._meta], self._init_info.buffer_bytes
        )
        self._owned_names = set(self.source.owned_names(rank))
        self._owned_units = {}
        for m, u in zip(self._meta, self._units):
            if m.name in self._owned_names:
                self._owned_units.setdefault(u, []).append(m.name)

    def _meta_digest(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(f"{len(self._meta)} {self._init_info.buffer_bytes}\n".encode())
        for m in self._meta:
            h.update(f"{m.name}\t{m.dtype}\t{tuple(m.shape)}\t{m.owners}\n".encode())
        return h.hexdigest()[:16]

    def _all_gather_server_names(self, world: int, rank: int) -> list[str]:
        """Server names by rank; a metadata digest rides along, since every
        rank plans units and ownership from its own metadata."""
        assert self._server_name is not None
        if world <= 1 or not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return [self._server_name]
        digest = self._meta_digest()
        gathered: list[tuple[str, str] | None] = [None] * world
        torch.distributed.all_gather_object(gathered, (self._server_name, digest))
        mismatched = [r for r, g in enumerate(gathered) if g is None or g[1] != digest]
        if mismatched:
            raise ValueError(
                f"WeightSource.metadata or buffer_bytes disagree across trainer "
                f"ranks (rank {rank} digest {digest}, differing ranks "
                f"{mismatched[:4]}). Every rank must describe the whole model, "
                "with the same owners."
            )
        return [g[0] for g in gathered if g is not None]

    def _arena_bytes(self) -> int:
        """`arena_bytes` for this rank, over the consumers that may pull from it."""
        ii = self._init_info
        router = RdtRouter(
            self._world,
            ii.num_consumers,
            self._owner_sets,
            self._name_owner_class,
            [m.name for m in self._meta],
            workers_per_replica=ii.workers_per_replica,
        )
        return arena_bytes(
            buffer_bytes=ii.buffer_bytes,
            num_buffers=ii.num_buffers,
            num_consumers=len(router.consumers_of(self.rank)),
        )

    def _spawn_server(self) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ii = self._init_info
        # The trainer is often a separate install from the workers.
        check_ray_rdt_version()
        self._server_name = f"vllm_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        # The server is the rank's twin process: same library env, pinned to
        # the rank's physical GPU (num_gpus=0 so Ray does not hand it a
        # second one) for CUDA IPC.
        env_vars = {
            k: os.environ[k]
            for k in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "NCCL_CUMEM_ENABLE",
                "VLLM_NCCL_SO_PATH",
                "PATH",
            )
            if k in os.environ
        }
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        served = {
            m.name: (_dtype_name(m.dtype), tuple(m.shape))
            for m in self._meta
            if m.name in self._owned_names
        }
        units = {
            m.name: u
            for m, u in zip(self._meta, self._units)
            if m.name in self._owned_names
        }
        server_cls = ray.remote(_RDTProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            # every in-flight pull may wait in `bucket` for its unit, plus
            # control calls; a smaller pool would queue a publish behind them
            max_concurrency=ii.num_consumers * ii.num_buffers + 8,
            enable_tensor_transport=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            runtime_env={"env_vars": env_vars} if env_vars else {},
        )
        self._server = server_cls.remote(
            served=served,
            units=units,
            arena_bytes=self._arena_bytes(),
            bucket_bytes=ii.buffer_bytes,
            stall_timeout_s=ii.stall_timeout_s,
        )
        self._rpc("warmup_nixl")

    # ---------------- per sync ----------------

    def send_weights(self, live_consumer_ids: Collection[int] | None = None) -> None:
        """Gather this rank's weights and serve them to the consumers.

        `live_consumer_ids` restricts the sync to the consumers still alive
        (every rank must get the same set); `None` serves all of them.
        """
        self._sync_idx += 1
        if live_consumer_ids is None:
            live = list(range(self._init_info.num_consumers))
        else:
            live = sorted(set(int(c) for c in live_consumer_ids))
            logger.warning(
                "[rdt-degraded] serving %d/%d live consumers",
                len(live),
                self._init_info.num_consumers,
            )
        if not self.is_sender:
            self._run_gather_loop(None, live)
            return
        from vllm.distributed.weight_transfer.sharded_rdt_engine import (
            ShardedRDTWeightTransferUpdateInfo,
        )

        try:
            self.client.start_weight_update()
        except BaseException as e:
            # the other ranks are gathering this sync: fail it on their
            # producers too (a collective gather still waits on this rank)
            self._fail_producers(f"start_weight_update failed: {e!r}")
            raise
        update = asdict(ShardedRDTWeightTransferUpdateInfo(sync_idx=self._sync_idx))
        with ThreadPoolExecutor(max_workers=1) as exe:
            # the workers pull inside update_weights, alongside the gathers
            future = exe.submit(self.client.update_weights, update)
            future.add_done_callback(self._fail_producers_on_error)
            self._run_gather_loop(future, live)
            future.result()
        self.client.finish_weight_update()

    def _fail_producers_on_error(self, update_future) -> None:
        """A failed `update_weights` (a worker's pull failed) fails the sync
        on every producer at once, so no rank waits out the stall watchdog
        for releases that will not come."""
        error = update_future.exception()
        if error is not None:
            self._fail_producers(f"update_weights failed: {error!r}")

    def _fail_producers(self, message: str) -> None:
        """Fail this sync on every producer."""
        for name in self._server_names or []:
            with contextlib.suppress(Exception):
                actor = ray.get_actor(
                    name, namespace=self._init_info.trainer_actor_namespace
                )
                actor.set_gather_error.remote(message, self._sync_idx)

    def _run_gather_loop(self, update_future, live: list[int]) -> None:
        """Gather and publish this rank's units in order, holding at most
        `gather_lookahead + 1` unreleased ones. Runs on every rank."""
        assert self.source is not None
        self._rpc("begin_sync", live, self._sync_idx)
        bound = max(0, self._init_info.gather_lookahead)
        nslots = bound + 1
        if len(self._export_ring) != nslots:
            self._export_ring = [None] * nslots
            self._export_ring_args = [None] * nslots
        self._slot_of_unit = {}
        # data_ptr -> (storage weakref, reduce args), this sync only; the
        # weakref catches an address reused by a new allocation.
        handle_cache: dict[int, tuple] = {}
        pending: list = []  # publishes not yet confirmed, at most 2 deep
        units = self._iter_owned_units()
        # seconds waiting for released units, gathering, exporting
        spent = [0.0, 0.0, 0.0]
        t_start = time.perf_counter()
        try:
            for unit in self._owned_units:
                t0 = time.perf_counter()
                while len(self._inflight) > bound:
                    while pending:
                        ray.get(pending.pop(0))
                    self._drop_inflight(self._rpc("wait_freed"))
                t1 = time.perf_counter()
                names, tensors = next(units)
                t2 = time.perf_counter()
                entries, refs, ready = self._export(unit, names, tensors, handle_cache)
                del tensors
                self._inflight[unit] = (refs, ready)
                pending.append(self._remote("publish_unit", unit, entries))
                while len(pending) >= 2:
                    ray.get(pending.pop(0))
                spent[0] += t1 - t0
                spent[1] += t2 - t1
                spent[2] += time.perf_counter() - t2
                if update_future is not None and update_future.done():
                    update_future.result()  # a worker failed: stop gathering
            while pending:
                ray.get(pending.pop(0))
            t0 = time.perf_counter()
            self._drop_inflight(self._rpc("end_sync"))
            logger.info(
                "Sharded RDT rank %d sync %d: %d units in %.2f s (waiting for "
                "releases %.2f s, gathering %.2f s, publishing %.2f s, end %.2f s)",
                self.rank,
                self._sync_idx,
                len(self._owned_units),
                time.perf_counter() - t_start,
                *spent,
                time.perf_counter() - t0,
            )
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e), self._sync_idx)
            self._inflight.clear()
            raise
        finally:
            # the server's IPC base cache is per sync, so the handles are too
            self._export_ring_args = [None] * len(self._export_ring)

    def _export(
        self,
        unit: int,
        names: list[str],
        tensors: list[torch.Tensor],
        handle_cache: dict[int, tuple],
    ) -> tuple[tuple, dict[str, torch.Tensor], Any]:
        """Share one unit's tensors over CUDA IPC: each large storage once,
        the small ones packed into an export slot. Returns the entries for
        `publish_unit`, the tensors to keep alive, and the event that marks
        them written."""
        storages: dict[int, tuple] = {}
        views: dict[str, tuple] = {}
        refs: dict[str, torch.Tensor] = {}
        small: list = []
        for name, tensor in zip(names, tensors):
            tensor = tensor.detach()
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            # by storage: many small views of one large buffer are not small
            if tensor.untyped_storage().nbytes() <= _EXPORT_RING_MAX_BYTES:
                small.append((name, tensor))
                continue
            tensor = tensor.contiguous()
            refs[name] = tensor
            ust = tensor.untyped_storage()
            sid = ust.data_ptr()
            if sid not in storages:
                cached = handle_cache.get(sid)
                if cached is not None and not cached[0].expired():
                    reduce_args = cached[1]
                else:
                    base = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                    base.set_(ust, 0, (ust.nbytes(),))
                    _, reduce_args = reduce_tensor(base)
                    handle_cache[sid] = (StorageWeakRef(ust), reduce_args)
                storages[sid] = reduce_args
            views[name] = (
                sid,
                _dtype_name(tensor.dtype),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.storage_offset(),
            )
        if small:
            # the credit gate leaves at most `gather_lookahead` units holding one
            taken = set(self._slot_of_unit.values())
            free = [i for i in range(len(self._export_ring)) if i not in taken]
            assert free, "no free export slot past the credit gate"
            slot = self._slot_of_unit[unit] = free[0]
            st, vw, rf = self._pack_for_export(small, slot)
            storages.update(st)
            views.update(vw)
            refs.update(rf)
        # The server reads these from another process, which nothing orders
        # after the gather's and the packing's kernels: an interprocess event
        # it waits for on its own stream, instead of a device sync here.
        ready = handle = None
        device = next((t.device for t in refs.values() if t.is_cuda), None)
        if device is not None:
            ready = torch.cuda.Event(interprocess=True)
            ready.record(torch.cuda.current_stream(device))
            handle = ready.ipc_handle()
        return (storages, views, handle), refs, ready

    def _pack_for_export(self, held: list, slot_idx: int) -> tuple:
        """Copy `held` into export slot `slot_idx` and describe them as views
        of its one storage, the way `publish_unit` rebuilds them. A slot is
        handed back only when its unit is released (`_drop_inflight`)."""
        offsets: list[int] = []
        cur = 0
        for _, t in held:
            cur = round_up(cur, PIECE_ALIGNMENT)
            offsets.append(cur)
            cur += t.numel() * t.element_size()
        need = max(PIECE_ALIGNMENT, round_up(cur, PIECE_ALIGNMENT))
        slot = self._export_ring[slot_idx]
        device = held[0][1].device
        if slot is None or slot.numel() < need or slot.device != device:
            self._export_ring[slot_idx] = self._export_ring_args[slot_idx] = None
            slot = torch.empty(need, dtype=torch.uint8, device=device)
            self._export_ring[slot_idx] = slot
        ust = slot.untyped_storage()
        sid = ust.data_ptr()
        reduce_args = self._export_ring_args[slot_idx]
        if reduce_args is None:
            base = torch.empty(0, dtype=torch.uint8, device=device)
            base.set_(ust, 0, (ust.nbytes(),))
            _, reduce_args = reduce_tensor(base)
            self._export_ring_args[slot_idx] = reduce_args
        views: dict[str, tuple] = {}
        refs: dict[str, torch.Tensor] = {}
        for (name, t), off in zip(held, offsets):
            ref = slot[off : off + t.nbytes].view(t.dtype).view(t.shape)
            refs[name] = ref
            views[name] = (
                sid,
                _dtype_name(t.dtype),
                list(ref.shape),
                list(ref.stride()),
                ref.storage_offset(),
            )
        # ordered before the server's reads by the unit's event (`_export`)
        torch._foreach_copy_(list(refs.values()), [t for _, t in held])
        return {sid: reduce_args}, views, refs

    def _iter_owned_units(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        """This rank's units in order, as `(names, tensors)`."""
        assert self.source is not None
        pairs = self.source.iter_full()
        for unit, expected in self._owned_units.items():
            got = list(itertools.islice(pairs, len(expected)))
            names = [name for name, _ in got]
            if names != expected:
                raise RuntimeError(
                    f"WeightSource yielded {names[:2]!r} for unit {unit}, but this "
                    f"rank owns {expected[:2]!r} there; iter_full() must yield "
                    "the owned names in metadata order."
                )
            yield names, [tensor for _, tensor in got]

    def _drop_inflight(self, released: list) -> None:
        """Drop released units' tensors and return their export slots."""
        for unit in released:
            self._inflight.pop(int(unit), None)
            self._slot_of_unit.pop(int(unit), None)

    def shutdown(self) -> None:
        if self._server is None:
            return
        with contextlib.suppress(Exception):
            ray.get(self._server.shutdown.remote())
            ray.kill(self._server)
        self._server = None
        self._inflight.clear()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")
