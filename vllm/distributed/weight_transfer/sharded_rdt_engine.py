# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine (consumer side).

Each worker pulls only the boxes of each checkpoint tensor that its load plan
names, from the trainer ranks that own them, and writes them where the plan
says. The protocol is described in `sharded_rdt_common`.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sharded_rdt_common import (
    DEFAULT_BUFFER_BYTES,
    DEFAULT_NUM_BUFFERS,
    Box,
    RdtRouter,
    WireBucket,
    box_shape,
    check_ray_rdt_version,
    deregister_nixl_memory,
    gather_units,
    piece_nbytes,
    register_nixl_memory,
    validate_int,
    validate_sharded_rdt_settings,
    wire_box,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.model_loader.reload import LoadPlan

logger = init_logger(__name__)


@dataclass
class ShardedRDTWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the sharded RDT backend, built by the trainer
    (`ShardedRDTTrainerWeightTransferEngine`)."""

    trainer_actor_names: list[str] = field(default_factory=list)
    """Producer actor names, ordered by trainer rank."""

    trainer_actor_namespace: str | None = None
    """Ray namespace the producer actors live in."""

    names: list[str] = field(default_factory=list)
    """Every checkpoint tensor name, in the trainer's gather order."""

    dtype_names: list[str] = field(default_factory=list)
    """Dtype name (e.g. 'bfloat16') for each entry of `names`."""

    shapes: list[list[int]] = field(default_factory=list)
    """Full shape for each entry of `names`."""

    owner_sets: list[list[int]] = field(default_factory=list)
    """The distinct sets of trainer ranks that hold a name. Empty means every
    rank holds every name."""

    name_owner_class: list[int] = field(default_factory=list)
    """Per name, its index into `owner_sets`."""

    num_consumers: int = 0
    """Workers across the fleet; 0 infers it from the parallel config."""

    buffer_bytes: int = DEFAULT_BUFFER_BYTES
    """Most bytes one pull moves, and the size of a gather unit."""

    num_buffers: int = DEFAULT_NUM_BUFFERS
    """Pulls each worker keeps in flight."""

    replica_rank: int = 0
    """This inference deployment's index in a fleet of `num_replicas`
    independent ones, whose worker indices each restart at 0."""

    num_replicas: int = 1
    """Independent inference deployments in the fleet (uniform)."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend."""

    sync_idx: int = 0
    """The trainer's count of this sync; producers serve a pull only in the
    sync it names."""


@dataclass(slots=True)
class _Bucket:
    """One pull: the pieces one producer serves to this worker, and where
    they go.

    A direct bucket's `targets` are model tensors, one per piece, written by
    NIXL. A staged bucket lands in a buffer Ray allocates: per piece, the
    tensors to copy it into, or the name to load it as (a fallback)."""

    producer: int
    seq: int
    unit: int
    pieces: list[tuple[str, Box]]
    nbytes: int
    targets: list[torch.Tensor] | None = None
    copies: list[list[torch.Tensor]] = field(default_factory=list)
    loads: list[str | None] = field(default_factory=list)

    def wire(self) -> WireBucket:
        return [(name, wire_box(box)) for name, box in self.pieces]


@dataclass(slots=True)
class _PullHooks:
    """The transport, injected so tests can run the engine without Ray."""

    # Issue a pull of `bucket` in sync `sync_idx`; return its handle.
    issue: "Callable[[_Bucket, int], Any]"
    # Block until at least one handle is ready; return every ready one.
    wait: "Callable[[list[Any]], list[Any]]"
    # Fetch ready handles; return each one's tensors.
    get: "Callable[[list[Any]], list[list[torch.Tensor]]]"
    # Send each producer (by index) this worker's buckets; wait for all.
    register: "Callable[[list[list[WireBucket]]], None]"


class ShardedRDTWeightTransferEngine(
    WeightTransferEngine[
        ShardedRDTWeightTransferInitInfo,
        ShardedRDTWeightTransferUpdateInfo,
    ]
):
    """Pull-based RDT/NIXL backend that moves only the boxes each worker's
    load plan needs. Requires `VLLM_RELOAD_MODE=direct`,
    `distributed_executor_backend="ray"` and `nixl`.

    Every inference deployment must call `start_weight_update` before any
    calls `update_weights`: the first one plans and registers with the
    trainer, which frees a gathered unit only once every live worker has.
    """

    init_info_cls = ShardedRDTWeightTransferInitInfo
    update_info_cls = ShardedRDTWeightTransferUpdateInfo
    # The plan belongs to one model's tensors.
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        backend = vllm_config.parallel_config.distributed_executor_backend
        # A custom Executor subclass is a deliberate override we cannot judge.
        if isinstance(backend, str) and backend != "ray":
            raise ValueError(
                "The 'sharded_rdt' weight transfer backend requires "
                f"distributed_executor_backend='ray', got {backend!r}: the "
                "workers must be Ray actors with tensor transport."
            )
        self._metas: list[ParamMeta] | None = None
        self._units: dict[str, int] = {}
        self._router: RdtRouter | None = None
        self._consumer_id = 0
        self._buffer_bytes = DEFAULT_BUFFER_BYTES
        self._num_buffers = DEFAULT_NUM_BUFFERS
        self._producers: list[Any] = []
        self._hooks: _PullHooks | None = None
        # Built at the first start_weight_update after an init.
        self._plan: LoadPlan | None = None
        self._buckets: list[_Bucket] = []
        # One tensor per model storage registered with NIXL.
        self._registered: list[torch.Tensor] = []

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Check the configuration and bind the producers. Touches no model
        state; the plan is built at the next `start_weight_update`.

        Raises:
            ValueError: invalid settings or metadata.
            RuntimeError: the model cannot be planned (`check_plannable`), or
                a producer cannot be found.

        """
        from vllm.model_executor.model_loader.reload import check_plannable

        validate_sharded_rdt_settings(
            buffer_bytes=init_info.buffer_bytes,
            num_buffers=init_info.num_buffers,
            num_consumers=init_info.num_consumers,
            allow_zero_consumers=True,
        )
        check_plannable(self.model)
        check_ray_rdt_version()
        metas = _metas(init_info)
        num_consumers = init_info.num_consumers or (
            self.parallel_config.world_size_across_dp
        )
        replicas = max(1, init_info.num_replicas)
        per_replica = max(1, num_consumers // replicas)
        if len(init_info.name_owner_class) not in (0, len(metas)):
            raise ValueError(
                f"{len(init_info.name_owner_class)} owner classes for "
                f"{len(metas)} names"
            )
        router = RdtRouter(
            len(init_info.trainer_actor_names),
            num_consumers,
            init_info.owner_sets or None,
            init_info.name_owner_class or None,
            [m.name for m in metas],
            workers_per_replica=per_replica,
        )
        router.validate()
        producers = _resolve_producers(init_info)

        self._drop_plan()
        self._metas = metas
        nbytes = [_nbytes(m) for m in metas]
        self._units = {
            m.name: u
            for m, u in zip(metas, gather_units(nbytes, init_info.buffer_bytes))
        }
        self._router = router
        self._consumer_id = (
            max(0, init_info.replica_rank) * per_replica + self._global_worker_index()
        )
        self._buffer_bytes = init_info.buffer_bytes
        self._num_buffers = init_info.num_buffers
        self._producers = producers
        self._hooks = self._ray_hooks()
        logger.info(
            "Sharded RDT engine: consumer %d of %d, %d producers, %d names",
            self._consumer_id,
            num_consumers,
            len(producers),
            len(metas),
        )

    def _global_worker_index(self) -> int:
        """`data_parallel_index * world_size + rank` over the TP*PP world:
        distinct per worker of one deployment."""
        pc = self.parallel_config
        return pc.data_parallel_index * pc.world_size + pc.rank

    def start_weight_update(self) -> None:
        """Begin a direct reload; the first one after an init also plans,
        registers the destinations with NIXL and sends each producer its
        buckets.

        Raises:
            RuntimeError: not initialized.

        """
        from vllm.model_executor.model_loader.reload import plan_loads, start_reload

        if self._metas is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized; call init_transfer_engine()."
            )
        start_reload(self.model)
        if self._plan is not None:
            return
        # Trace the real tensors `start_reload` just viewed back, so every
        # destination is the one direct reload restores.
        try:
            t0 = time.perf_counter()
            plan = plan_loads(self.model, self._metas)
            buckets = self._build_buckets(plan)
            registered = _register_storages(
                t for b in buckets for t in (b.targets or ())
            )
            try:
                self._send_buckets(buckets)
            except BaseException:
                _deregister(registered)
                raise
        except Exception as e:
            e.add_note(
                "Sharded RDT: planning failed after start_reload, so the model "
                "is mid-reload; the next start_weight_update plans again"
            )
            raise
        self._plan, self._buckets, self._registered = plan, buckets, registered
        direct = sum(b.targets is not None for b in buckets)
        logger.info(
            "Sharded RDT plan: %d buckets (%d direct, %d staged), %.1f MiB, "
            "%d storages registered, in %.2f s",
            len(buckets),
            direct,
            len(buckets) - direct,
            sum(b.nbytes for b in buckets) / (1 << 20),
            len(registered),
            time.perf_counter() - t0,
        )

    def _build_buckets(self, plan: "LoadPlan") -> list[_Bucket]:
        """Route, split and order this worker's pieces into pulls."""
        from vllm.model_executor.model_loader.reload import LoadPlan

        assert isinstance(plan, LoadPlan) and self._metas is not None
        router = self._router
        assert router is not None
        # (producer, unit, direct) -> pieces as (name, box, dtype, target|copies|load)
        groups: dict[tuple[int, int, bool], list[tuple]] = defaultdict(list)
        for meta in self._metas:
            boxes = plan.get_boxes(meta.name)
            if boxes is None:
                continue
            unit = self._units[meta.name]
            producer = router.producer_for(self._consumer_id, meta.name, unit)
            dest = plan.get_dest(meta.name)
            if dest is None:
                (box,) = boxes
                groups[(producer, unit, False)].append(
                    (meta.name, box, meta.dtype, meta.name)
                )
                continue
            by_box: dict[Box, list[torch.Tensor]] = defaultdict(list)
            for box, tensor in dest:
                if box not in boxes:
                    raise RuntimeError(
                        f"Sharded RDT: {meta.name!r} writes box {box}, which is "
                        f"not one of its plan boxes {boxes}"
                    )
                by_box[box].append(tensor)
            for box in boxes:
                staged = []
                for tensor in by_box[box]:
                    if _receivable(tensor, box, meta.dtype):
                        groups[(producer, unit, True)].append(
                            (meta.name, box, meta.dtype, tensor)
                        )
                    else:
                        staged.append(tensor)
                if staged:
                    groups[(producer, unit, False)].append(
                        (meta.name, box, meta.dtype, staged)
                    )

        components = router.components()
        unit_rank = _component_unit_rank(self._metas, self._units, router)
        buckets: list[_Bucket] = []
        for (producer, unit, direct), pieces in groups.items():
            for chunk in _cut(pieces, self._buffer_bytes):
                bucket = _Bucket(
                    producer=producer,
                    seq=-1,
                    unit=unit,
                    pieces=[(name, box) for name, box, _, _ in chunk],
                    nbytes=sum(piece_nbytes(box, dt) for _, box, dt, _ in chunk),
                )
                if direct:
                    bucket.targets = [target for *_, target in chunk]
                else:
                    for *_, sink in chunk:
                        is_load = isinstance(sink, str)
                        bucket.copies.append([] if is_load else sink)
                        bucket.loads.append(sink if is_load else None)
                buckets.append(bucket)

        def order(b: _Bucket) -> tuple:
            comp = components[b.producer]
            return (unit_rank[comp][b.unit], comp, b.unit, b.producer, not b.targets)

        buckets.sort(key=order)
        seqs: dict[int, int] = defaultdict(int)
        for b in buckets:
            b.seq = seqs[b.producer]
            seqs[b.producer] += 1
        return buckets

    def _send_buckets(self, buckets: list[_Bucket]) -> None:
        """Tell every producer, including those this worker pulls nothing
        from, which buckets it will pull: a producer releases a unit only
        once every live worker has registered."""
        assert self._hooks is not None
        per_producer: list[list[WireBucket]] = [[] for _ in self._producers]
        for b in buckets:
            per_producer[b.producer].append(b.wire())
        self._hooks.register(per_producer)

    def receive_weights(self, update_info: ShardedRDTWeightTransferUpdateInfo) -> None:
        """Pull every bucket of the plan, `num_buffers` in flight, and apply
        each as it lands.

        Raises:
            RuntimeError: no plan (call `start_weight_update` first).

        """
        validate_int("sync_idx", update_info.sync_idx, minimum=0)
        if self._plan is None or self._hooks is None:
            raise RuntimeError(
                "Sharded RDT: no load plan; call start_weight_update() first."
            )
        hooks, sync_idx = self._hooks, update_info.sync_idx
        todo = iter(self._buckets)
        inflight: dict[Any, _Bucket] = {}

        def refill() -> None:
            while len(inflight) < self._num_buffers:
                bucket = next(todo, None)
                if bucket is None:
                    return
                inflight[hooks.issue(bucket, sync_idx)] = bucket

        # seconds spent waiting for producers, fetching, and applying
        spent = [0.0, 0.0, 0.0]

        def land(ready: list[Any]) -> None:
            # Returning drops the handles, which frees the producers' pool
            # blocks before the next pulls are issued.
            t0 = time.perf_counter()
            fetched = hooks.get(ready)
            t1 = time.perf_counter()
            for handle, tensors in zip(ready, fetched):
                self._apply(inflight.pop(handle), tensors)
            spent[1] += t1 - t0
            spent[2] += time.perf_counter() - t1

        t_start = time.perf_counter()
        refill()
        while inflight:
            t0 = time.perf_counter()
            ready = hooks.wait(list(inflight))
            spent[0] += time.perf_counter() - t0
            land(ready)
            del ready
            refill()
        logger.info(
            "Sharded RDT sync %d: %d pulls in %.2f s (waiting %.2f s, fetching "
            "%.2f s, applying %.2f s)",
            sync_idx,
            len(self._buckets),
            time.perf_counter() - t_start,
            *spent,
        )

    def _apply(self, bucket: _Bucket, tensors: list[torch.Tensor]) -> None:
        """Land a staged bucket; a direct one is already in place."""
        if bucket.targets is not None:
            return
        loads = []
        for received, dests, name in zip(tensors, bucket.copies, bucket.loads):
            if received.is_cuda:
                # Ray allocated it, possibly on another stream
                received.record_stream(torch.cuda.current_stream(received.device))
            for dest in dests:
                dest.copy_(received.reshape(dest.shape))
            if name is not None:
                loads.append((name, received))
        if loads:
            self.model.load_weights(loads)

    def finish_weight_update(self) -> None:
        """Rerun post-load processing on the received weights."""
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def _ray_hooks(self) -> _PullHooks:
        import ray
        from ray.experimental import set_target_for_ref

        consumer_id, producers = self._consumer_id, self._producers

        def issue(bucket: _Bucket, sync_idx: int) -> Any:
            ref = producers[bucket.producer].bucket.remote(
                consumer_id, bucket.seq, sync_idx
            )
            if bucket.targets is not None:
                set_target_for_ref(ref, bucket.targets)
            return ref

        def wait(refs: list[Any]) -> list[Any]:
            ray.wait(refs, num_returns=1)
            ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
            return ready

        def register(per_producer: list[list[WireBucket]]) -> None:
            ray.get(
                [
                    actor.register_consumer.remote(consumer_id, buckets)
                    for actor, buckets in zip(producers, per_producer)
                ]
            )

        return _PullHooks(issue=issue, wait=wait, get=ray.get, register=register)

    def _drop_plan(self) -> None:
        _deregister(self._registered)
        self._registered = []
        self._plan = None
        self._buckets = []

    def shutdown(self) -> None:
        self._drop_plan()
        self._metas = None
        self._producers = []
        self._hooks = None
        self._router = None


def _metas(init_info: ShardedRDTWeightTransferInitInfo) -> list[ParamMeta]:
    names, dtypes, shapes = init_info.names, init_info.dtype_names, init_info.shapes
    if not len(names) == len(dtypes) == len(shapes):
        raise ValueError(
            f"Sharded RDT init info: {len(names)} names, {len(dtypes)} dtypes, "
            f"{len(shapes)} shapes"
        )
    metas = []
    for name, dtype_name, shape in zip(names, dtypes, shapes):
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{name!r}: unknown dtype {dtype_name!r}")
        metas.append(ParamMeta(name, dtype, tuple(shape)))
    return metas


def _nbytes(meta: ParamMeta) -> int:
    n = meta.dtype.itemsize
    for d in meta.shape:
        n *= d
    return n


def _resolve_producers(init_info: ShardedRDTWeightTransferInitInfo) -> list[Any]:
    """Producer actor handles, by trainer rank."""
    try:
        import ray
    except ImportError as e:
        raise RuntimeError(
            "The 'sharded_rdt' weight transfer backend requires Ray."
        ) from e
    if not init_info.trainer_actor_names:
        raise RuntimeError("Sharded RDT: init_info.trainer_actor_names is empty.")
    actors = []
    for name in init_info.trainer_actor_names:
        try:
            actor = ray.get_actor(name, namespace=init_info.trainer_actor_namespace)
        except ValueError as e:
            raise RuntimeError(
                f"Sharded RDT: no producer actor {name!r} in namespace "
                f"{init_info.trainer_actor_namespace!r}."
            ) from e
        # A handle resolved by name does not carry the class metadata that
        # enables tensor transport, so Ray would refuse the pull; the producer
        # is always created with it (see the trainer's `_spawn_server`).
        actor._ray_enable_tensor_transport = True
        actors.append(actor)
    return actors


def _receivable(tensor: torch.Tensor, box: Box, dtype: torch.dtype) -> bool:
    """Whether NIXL can write the box straight into `tensor`: Ray receives
    into a target only with the sent shape and dtype, contiguously."""
    return (
        tensor.is_contiguous()
        and tuple(tensor.shape) == box_shape(box)
        and tensor.dtype == dtype
    )


def _cut(pieces: list[tuple], budget: int) -> list[list[tuple]]:
    """Consecutive runs of at most `budget` bytes; a larger piece goes alone."""
    chunks: list[list[tuple]] = []
    used = budget
    for piece in pieces:
        n = piece_nbytes(piece[1], piece[2])
        if used + n > budget or not chunks:
            chunks.append([])
            used = 0
        chunks[-1].append(piece)
        used += n
    return chunks


def _component_unit_rank(
    metas: list[ParamMeta], units: dict[str, int], router: RdtRouter
) -> dict[int, dict[int, int]]:
    """Per coupling component, each unit's index among the units its ranks
    gather. Ordering pulls by it keeps every component in gather order (so
    no rank waits on a unit its consumers pull later) while independent
    components, pipeline stages, pull side by side."""
    components = router.components()
    rank: dict[int, dict[int, int]] = defaultdict(dict)
    for meta in metas:
        comp = components[router.owners(meta.name)[0]]
        seen = rank[comp]
        seen.setdefault(units[meta.name], len(seen))
    return rank


def _register_storages(tensors) -> list[torch.Tensor]:
    """Register each distinct storage once; return one tensor per storage."""
    registered: dict[int, torch.Tensor] = {}
    try:
        for tensor in tensors:
            key = tensor.untyped_storage().data_ptr()
            if key not in registered:
                register_nixl_memory(tensor)
                registered[key] = tensor
    except BaseException:
        _deregister(list(registered.values()))
        raise
    return list(registered.values())


def _deregister(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        try:
            deregister_nixl_memory(tensor)
        except Exception:
            logger.exception("Sharded RDT: NIXL deregistration failed")
