# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer side of the sharded-RDT (pull-based NIXL) backend.

Every trainer rank owns a producer server: an internal Ray actor that shares
the rank's GPU over CUDA IPC and serves the workers' pulls out of a NIXL
memory pool. Each sync, the rank gathers its owned names unit by unit and
publishes each unit to its server; the sender also drives the inference-side
start/update/finish calls. The protocol is in `sharded_rdt_common`.
"""

import contextlib
import gc
import itertools
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Collection, Iterator
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
    Box,
    RdtRouter,
    WireBucket,
    box_slices,
    check_ray_rdt_version,
    from_wire_box,
    gather_units,
    piece_nbytes,
    pool_bytes,
    register_nixl_memory_pool,
    validate_sharded_rdt_settings,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Gathered units a rank may hold beyond the ones its consumers still read:
# at most `gather_lookahead + 1` units are resident on the trainer.
DEFAULT_GATHER_LOOKAHEAD = 1

# Seconds without progress (publish, serve, free) before a producer fails the
# sync. A consumer that dies mid-sync never frees its buckets, so its units
# are never released and the gather loop, a collective, would wedge every
# rank. A liveness backstop, not a latency target.
DEFAULT_STALL_TIMEOUT_S = 300.0
_STALL_POLL_S = 5.0

# Storages at most this large are packed into one export slot and shared over
# one CUDA-IPC handle; a handle per small tensor dominated the publish time.
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
        validate_sharded_rdt_settings(
            buffer_bytes=self.buffer_bytes,
            num_buffers=self.num_buffers,
            num_consumers=self.num_consumers,
            allow_zero_consumers=False,
        )


@dataclass(frozen=True, slots=True)
class _ServedBucket:
    unit: int
    pieces: tuple[tuple[str, tuple[slice, ...]], ...]
    nbytes: int


def _rdt_store() -> Any:
    from ray._private.worker import global_worker

    return global_worker.rdt_manager.rdt_store


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
        pool_bytes: int,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        max_outstanding: int = 64,
    ) -> None:
        self._device_index = torch.accelerator.current_device_index()
        self._served = {
            name: (getattr(torch, dtype), tuple(shape))
            for name, (dtype, shape) in served.items()
        }
        self._unit_of = dict(units)
        self._pool_bytes = int(pool_bytes)
        self._stall_timeout = float(stall_timeout_s)

        self._cond = threading.Condition()
        self._error: BaseException | None = None
        self._last_progress = time.monotonic()
        self._shutdown = False

        # name -> rebuilt CUDA-IPC view; units whose names are in it
        self._cache: dict[str, torch.Tensor] = {}
        self._unit_names: dict[int, list[str]] = {}
        # consumer -> its buckets in pull order, and their count per unit
        self._buckets: dict[int, list[_ServedBucket]] = {}
        self._needs: dict[int, Counter[int]] = {}

        # per sync
        self._sync_idx = -1
        self._live: frozenset[int] = frozenset()
        self._freed: dict[int, Counter[int]] = {}
        self._inflight_units: list[int] = []  # published, not released
        self._freed_pending: list[int] = []  # released, not yet collected
        # served buckets whose pool blocks Ray has not freed yet
        self._pool_used = 0
        self._waiters = ThreadPoolExecutor(
            max_workers=max(1, max_outstanding), thread_name_prefix="rdt-free"
        )
        self._base_cache: dict[tuple, torch.Tensor] = {}

        # A frozen post-init object graph keeps gen-2 GC from stopping the
        # world mid-serve.
        gc.collect()
        gc.freeze()

    def ping(self) -> int:
        return self._device_index

    def warmup_nixl(self) -> None:
        """Create the NIXL agent and register the pool now, while the rank's
        GPU is idle and before any rank enters a collective: creating the
        agent lazily deadlocks on EFA-class fabrics."""
        register_nixl_memory_pool(
            self._pool_bytes, torch.device("cuda", self._device_index)
        )

    # ---------------- liveness ----------------

    def _progress(self) -> None:
        """Caller holds `_cond`."""
        self._last_progress = time.monotonic()
        self._cond.notify_all()

    def _wait_for(self, blocked: Callable[[], bool], what: str) -> None:
        """Wait while `blocked()`, returning early on an error; fail the sync
        when nothing on this producer progressed for the stall timeout.
        Caller holds `_cond`."""
        while blocked() and self._error is None:
            self._cond.wait(_STALL_POLL_S)
            stalled = time.monotonic() - self._last_progress
            if blocked() and self._error is None and stalled >= self._stall_timeout:
                msg = (
                    f"RDT stall: no progress for {stalled:.0f}s waiting for "
                    f"{what}. A consumer most likely died mid-sync: "
                    f"{len(self._inflight_units)} unit(s) published and unreleased."
                )
                logger.error("[rdt-stall] %s", msg)
                self._error = RuntimeError(msg)
                self._cond.notify_all()
        if self._error is not None:
            raise RuntimeError(f"sharded RDT sync failed: {self._error!r}")

    # ---------------- engine-facing ----------------

    def begin_sync(self, live_consumer_ids: list[int], sync_idx: int) -> None:
        """Open sync `sync_idx` for the given live consumers."""
        with self._cond:
            self._error = None
            self._sync_idx = int(sync_idx)
            self._live = frozenset(int(c) for c in live_consumer_ids)
            self._freed = {c: Counter() for c in self._live}
            self._inflight_units.clear()
            self._freed_pending.clear()
            self._base_cache.clear()
            self._progress()

    def publish_unit(self, unit: int, entries: tuple) -> None:
        """Rebuild one gathered unit's CUDA-IPC tensors into the cache; never
        blocks. `entries` is `(storages, views)`: one export per storage, and
        per name `(sid, dtype_name, shape, stride, storage_offset)`.

        Raises:
            ValueError: a tensor does not match its metadata.

        """
        storages, views = entries
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
        self._publish(unit, rebuilt)

    def _publish(self, unit: int, tensors: dict[str, torch.Tensor]) -> None:
        """Make one unit's tensors servable.

        Raises:
            ValueError: a tensor does not match its metadata.

        """
        for name, t in tensors.items():
            if self._served.get(name) != (t.dtype, tuple(t.shape)):
                raise ValueError(
                    f"{name!r} is {t.dtype} {tuple(t.shape)}, but its metadata "
                    f"says {self._served.get(name)}"
                )
        with self._cond:
            self._cache.update(tensors)
            self._unit_names[unit] = list(tensors)
            self._inflight_units.append(unit)
            self._release_if_done(unit)
            self._progress()

    def wait_freed(self) -> list[int]:
        """Block until a published unit is released; return the released ones."""
        with self._cond:
            self._wait_for(lambda: not self._freed_pending, "a released unit")
            freed, self._freed_pending = self._freed_pending, []
            return freed

    def end_sync(self) -> list[int]:
        """Block until every published unit is released; return the rest."""
        with self._cond:
            self._wait_for(lambda: bool(self._inflight_units), "every unit's release")
            freed, self._freed_pending = self._freed_pending, []
            self._base_cache.clear()
            return freed

    def set_gather_error(self, message: str) -> None:
        """Fail the sync: every blocked wait here raises."""
        with self._cond:
            self._error = RuntimeError(message)
            self._cond.notify_all()

    # ---------------- consumer-facing ----------------

    def register_consumer(self, consumer_id: int, buckets: list[WireBucket]) -> None:
        """Record the buckets `consumer_id` pulls from here, in pull order.
        Replaces an earlier registration (a re-planned consumer).

        Raises:
            ValueError: a bucket names a tensor this rank does not serve, a
                box outside its tensor, or tensors of two units.

        """
        served: list[_ServedBucket] = []
        for i, wire in enumerate(buckets):
            pieces, units, nbytes = [], set(), 0
            for name, wire_box in wire:
                if name not in self._served:
                    raise ValueError(
                        f"consumer {consumer_id} bucket {i}: {name!r} is not "
                        "served by this trainer rank"
                    )
                dtype, shape = self._served[name]
                box: Box = from_wire_box(wire_box)
                if len(box) != len(shape) or any(
                    not 0 <= a < b <= n for (a, b), n in zip(box, shape)
                ):
                    raise ValueError(f"{name!r}: box {box} is outside {shape}")
                pieces.append((name, box_slices(box)))
                units.add(self._unit_of[name])
                nbytes += piece_nbytes(box, dtype)
            if len(units) != 1:
                raise ValueError(
                    f"consumer {consumer_id} bucket {i} spans units {sorted(units)}"
                )
            served.append(_ServedBucket(units.pop(), tuple(pieces), nbytes))
        with self._cond:
            self._buckets[consumer_id] = served
            self._needs[consumer_id] = Counter(b.unit for b in served)
            for unit in list(self._inflight_units):
                self._release_if_done(unit)
            self._progress()

    @ray.method(tensor_transport="nixl")
    def bucket(self, consumer_id: int, seq: int, sync_idx: int) -> list[torch.Tensor]:
        """Serve one registered bucket of sync `sync_idx`: its boxes, once
        the unit is gathered and the pool has room. Ray copies them into the
        pool when this returns.

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
        with self._cond:
            if sync_idx < self._sync_idx:
                raise ValueError(f"pull for sync {sync_idx}, now {self._sync_idx}")
            self._wait_for(
                lambda: (
                    self._sync_idx != sync_idx
                    or spec.unit not in self._unit_names
                    # an empty pool admits any bucket
                    or (
                        self._pool_used > 0
                        and self._pool_used + spec.nbytes > self._pool_bytes
                    )
                ),
                f"unit {spec.unit} of sync {sync_idx} and pool room",
            )
            self._pool_used += spec.nbytes
            tensors = [
                self._cache[name][slices].contiguous() for name, slices in spec.pieces
            ]
            self._progress()
        self._track_free(tensors, consumer_id, spec.unit, spec.nbytes, sync_idx)
        return tensors

    def _track_free(self, tensors: list[torch.Tensor], *key: int) -> None:
        """Account the bucket once Ray frees it (`_await_free`)."""
        self._waiters.submit(self._await_free, tensors, *key)

    def _await_free(
        self,
        tensors: list[torch.Tensor],
        consumer_id: int,
        unit: int,
        nbytes: int,
        sync_idx: int,
    ) -> None:
        """Account a served bucket once Ray frees it: the consumer has read
        it, so its pool block and the unit's tensors are no longer read."""
        from ray.experimental import wait_tensor_freed

        probe = tensors[0]
        store = _rdt_store()
        # Ray stores the returned tensors after `bucket` returns; before
        # that, the free wait would return at once. One never stored (Ray
        # failed to serialize it) is not in the pool, but its unit stays
        # unreleased and the stall watchdog fails the sync.
        deadline = time.monotonic() + self._stall_timeout
        stored = False
        while not self._shutdown and time.monotonic() < deadline:
            stored = store.has_tensor(probe)
            if stored:
                break
            time.sleep(0.001)
        while stored and not self._shutdown:
            try:
                wait_tensor_freed(probe, timeout=_STALL_POLL_S)
                break
            except TimeoutError:
                continue
        del tensors, probe
        self._freed_bucket(consumer_id, unit, nbytes, sync_idx, read=stored)

    def _freed_bucket(
        self, consumer_id: int, unit: int, nbytes: int, sync_idx: int, *, read: bool
    ) -> None:
        """A served bucket's pool block is free; if it was `read`, it no
        longer holds its unit."""
        with self._cond:
            self._pool_used -= nbytes
            if read and sync_idx == self._sync_idx and consumer_id in self._freed:
                self._freed[consumer_id][unit] += 1
                self._release_if_done(unit)
            self._progress()

    def _release_if_done(self, unit: int) -> None:
        """Release `unit` once every live consumer registered and freed all
        its buckets of it. Caller holds `_cond`."""
        if unit not in self._unit_names:
            return
        for c in self._live:
            needs = self._needs.get(c)
            if needs is None or self._freed[c][unit] < needs[unit]:
                return
        for name in self._unit_names.pop(unit):
            self._cache.pop(name, None)
        if unit in self._inflight_units:
            self._inflight_units.remove(unit)
            self._freed_pending.append(unit)

    def shutdown(self) -> None:
        self._shutdown = True
        with self._cond:
            self._cache.clear()
            self._unit_names.clear()
        self._waiters.shutdown(wait=False, cancel_futures=True)


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
        self._owned_units: list[int] = []
        # Gathered tensors shared with the server, by unit: a CUDA-IPC
        # export must outlive its import, so they stay until released.
        self._inflight: dict[int, dict[str, torch.Tensor]] = {}
        # Reusable export slots for small tensors, one per resident unit; kept
        # across syncs, since reallocating them each sync fragments the
        # caching allocator.
        self._export_ring: list[torch.Tensor | None] = []
        self._export_ring_args: list[tuple | None] = []
        self._slot_of_unit: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._sync_idx = -1

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call a server method and wait; the seam tests replace."""
        return ray.get(getattr(self._server, method).remote(*args))

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
        world, rank = engine._world_and_rank()
        engine._plan(world, rank)
        engine._spawn_server()
        # Every server must exist before the sender's init reaches the
        # workers, which register with all of them; the all-gather of server
        # names is that barrier.
        engine._server_names = engine._all_gather_server_names(world, rank)
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
                buffer_bytes=ii.buffer_bytes,
                num_buffers=ii.num_buffers,
            )
        )

    def _world_and_rank(self) -> tuple[int, int]:
        rank = self.rank
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != rank:
                raise ValueError(
                    f"Sharded RDT needs TrainerInitInfo.rank ({rank}) to be the "
                    f"default process group rank ({torch.distributed.get_rank()})."
                )
            return torch.distributed.get_world_size(), rank
        return 1, rank

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
            [_nbytes(m) for m in self._meta], self._init_info.buffer_bytes
        )
        self._owned_names = set(self.source.owned_names(rank))
        self._owned_units = sorted(
            {u for m, u in zip(self._meta, self._units) if m.name in self._owned_names}
        )

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

    def _pool_bytes(self) -> int:
        """`pool_bytes` for this rank, over the consumers that may pull from it."""
        ii = self._init_info
        router = RdtRouter(
            self._world,
            ii.num_consumers,
            self._owner_sets,
            self._name_owner_class,
            [m.name for m in self._meta],
            workers_per_replica=ii.workers_per_replica,
        )
        largest = max(
            (_nbytes(m) for m in self._meta if m.name in self._owned_names),
            default=0,
        )
        return pool_bytes(
            buffer_bytes=ii.buffer_bytes,
            num_buffers=ii.num_buffers,
            num_consumers=len(router.consumers_of(self.rank)),
            largest_tensor_bytes=largest,
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
        # Every in-flight pull may wait in `bucket` for its unit, plus
        # control calls; a smaller pool would queue a publish behind them.
        max_outstanding = ii.num_consumers * ii.num_buffers + 4
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
            max_concurrency=max_outstanding + 4,
            enable_tensor_transport=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            runtime_env={"env_vars": env_vars} if env_vars else {},
        )
        self._server = server_cls.remote(
            served=served,
            units=units,
            pool_bytes=self._pool_bytes(),
            stall_timeout_s=ii.stall_timeout_s,
            max_outstanding=max_outstanding,
        )
        ray.get(self._server.ping.remote())
        ray.get(self._server.warmup_nixl.remote())

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

        self.client.start_weight_update()
        update = asdict(ShardedRDTWeightTransferUpdateInfo(sync_idx=self._sync_idx))
        with ThreadPoolExecutor(max_workers=1) as exe:
            # the workers pull inside update_weights, alongside the gathers
            future = exe.submit(self.client.update_weights, update)
            self._run_gather_loop(future, live)
            future.result()
        self.client.finish_weight_update()

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
        self._free_slots = list(range(nslots))
        # data_ptr -> (storage weakref, reduce args), this sync only; the
        # weakref catches an address reused by a new allocation.
        handle_cache: dict[int, tuple] = {}
        pending: list = []  # publishes not yet confirmed, at most 2 deep
        units = self._iter_owned_units()
        try:
            for unit in self._owned_units:
                while len(self._inflight) > bound:
                    while pending:
                        self._await_publish(pending.pop(0))
                    self._drop_inflight(self._rpc("wait_freed"))
                names, tensors = next(units)
                entries, refs = self._export(
                    unit, names, tensors, handle_cache, pending
                )
                del tensors
                self._inflight[unit] = refs
                pending.append(self._publish_async(unit, entries))
                while len(pending) >= 2:
                    self._await_publish(pending.pop(0))
                if update_future is not None and update_future.done():
                    update_future.result()  # a worker failed: stop gathering
            while pending:
                self._await_publish(pending.pop(0))
            self._drop_inflight(self._rpc("end_sync"))
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e))
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
        pending: list,
    ) -> tuple[tuple, dict[str, torch.Tensor]]:
        """Share one unit's tensors over CUDA IPC: each large storage once,
        the small ones packed into an export slot."""
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
            # The credit gate leaves a free slot; this is a backstop.
            while not self._free_slots:
                while pending:
                    self._await_publish(pending.pop(0))
                self._drop_inflight(self._rpc("wait_freed"))
            slot = self._free_slots.pop(0)
            self._slot_of_unit[unit] = slot
            st, vw, rf = self._pack_for_export(small, slot)
            storages.update(st)
            views.update(vw)
            refs.update(rf)
        return (storages, views), refs

    @staticmethod
    def _contig_stride(shape) -> tuple:
        stride: list[int] = []
        acc = 1
        for s in reversed(list(shape)):
            stride.append(acc)
            acc *= int(s)
        return tuple(reversed(stride))

    def _pack_for_export(self, held: list, slot_idx: int) -> tuple:
        """Copy `held` into export slot `slot_idx` and describe them as views
        of its one storage, the way `publish_unit` rebuilds them. A slot is
        handed back only when its unit is released (`_drop_inflight`)."""
        offsets: list[int] = []
        cur = 0
        for _, t in held:
            cur = (cur + 15) & ~15
            offsets.append(cur)
            cur += t.numel() * t.element_size()
        need = max(16, (cur + 15) & ~15)
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
            shape, esz = tuple(t.shape), t.element_size()
            stride = self._contig_stride(shape)
            dst = torch.as_strided(slot.view(t.dtype), shape, stride, off // esz)
            dst.copy_(t)
            refs[name] = dst
            views[name] = (
                sid,
                _dtype_name(t.dtype),
                list(shape),
                list(stride),
                off // esz,
            )
        # The server reads the slot from another process, which nothing
        # orders after these copies.
        if slot.is_cuda:
            torch.accelerator.current_stream(slot.device).synchronize()
        return {sid: reduce_args}, views, refs

    def _publish_async(self, unit: int, entries):
        """Fire `publish_unit` without waiting (a local fake server runs it
        inline); `_await_publish` surfaces its error."""
        method = self._server.publish_unit
        remote = getattr(method, "remote", None)
        if remote is not None:
            return remote(unit, entries)
        return method(unit, entries)

    def _await_publish(self, ref) -> None:
        if isinstance(ref, ray.ObjectRef):
            ray.get(ref)

    def _iter_owned_units(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        """This rank's units in order, as `(names, tensors)`."""
        assert self.source is not None
        by_unit: dict[int, list[str]] = {}
        for m, u in zip(self._meta, self._units):
            if m.name in self._owned_names:
                by_unit.setdefault(u, []).append(m.name)
        pairs = self.source.iter_full()
        for unit in self._owned_units:
            expected = by_unit[unit]
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
            slot = self._slot_of_unit.pop(int(unit), None)
            if slot is not None and slot not in self._free_slots:
                self._free_slots.append(slot)

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


def _nbytes(meta: ParamMeta) -> int:
    n = meta.dtype.itemsize
    for d in meta.shape:
        n *= d
    return n
