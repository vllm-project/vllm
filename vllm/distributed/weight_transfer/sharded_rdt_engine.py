# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded Ray Direct Transport (RDT) weight transfer engine (consumer side).

Each worker pulls only the boxes of each checkpoint tensor that its load plan
names, from the trainer ranks that own them, and writes them where the plan
says. The protocol is described in `sharded_rdt_common`.
"""

import itertools
import math
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
    PIECE_ALIGNMENT,
    Box,
    FlatPiece,
    RdtRouter,
    WireBucket,
    box_shape,
    check_ray_rdt_version,
    deregister_nixl_memory,
    flat_layout,
    flat_view,
    gather_units,
    meta_nbytes,
    piece_nbytes,
    register_nixl_memory,
    validate_int,
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

    workers_per_replica: int = 0
    """Workers per inference deployment, from the trainer; 0 means one."""

    replica_rank: int = 0
    """This inference deployment's index in a fleet of independent ones,
    whose worker indices each restart at 0; set per deployment."""

    num_replicas: int = 1
    """Independent inference deployments in the fleet (uniform); must match
    the trainer's `workers_per_replica`."""


@dataclass
class ShardedRDTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sharded RDT backend."""

    sync_idx: int = 0
    """The trainer's count of this sync; producers serve a pull only in the
    sync it names."""


@dataclass(eq=False, slots=True)
class _Staging:
    """A fallback's whole tensor `name`, allocated when its first piece lands
    and loaded through `load_weights` (then freed) once all `count` have."""

    name: str
    box: Box
    count: int = 0
    landed: int = 0
    tensor: torch.Tensor | None = None


# A piece: (name, box, sent dtype, destinations), a destination being a
# model tensor shaped like the box or the staging of a fallback.
_Piece = tuple[str, Box, torch.dtype, list[Any]]


@dataclass(slots=True)
class _Bucket:
    """One pull: the pieces one producer serves to this worker, and where
    they go. It lands as one flat byte tensor (pieces at `flat_layout`) in a
    slot of the receive arena."""

    producer: int
    seq: int
    unit: int
    pieces: list[tuple[str, Box]]
    nbytes: int
    # Built once by `prepare`, used every sync: where each piece sits in the
    # bucket; contiguous destinations of the sent dtype as bytes, runs of
    # pieces adjacent in both the bucket and memory merged into one, with
    # the (offset, nbytes) of the bucket each copies; the other
    # destinations; the stagings, with the slices of each the piece fills.
    layout: list[FlatPiece] = field(default_factory=list)
    byte_dsts: list[torch.Tensor] = field(default_factory=list)
    byte_srcs: list[tuple[int, int]] = field(default_factory=list)
    typed: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    typed_dsts: list[torch.Tensor] = field(default_factory=list)
    staged: list[tuple[int, _Staging, tuple[slice, ...]]] = field(default_factory=list)
    # Per receive slot, built when the arena is: where the bucket lands, and
    # the source views of its copy-out.
    targets: list[torch.Tensor] = field(default_factory=list)
    srcs: list[tuple] = field(default_factory=list)

    def _add_byte_copy(self, src: tuple[int, int], dest: torch.Tensor) -> None:
        """Copy bucket bytes `src` (offset, nbytes) into `dest`, extending
        the last copy when both continue where it ends."""
        offset, nbytes = src
        if self.byte_dsts:
            last, (last_offset, last_nbytes) = self.byte_dsts[-1], self.byte_srcs[-1]
            if (
                last_offset + last_nbytes == offset
                and last.untyped_storage().data_ptr()
                == dest.untyped_storage().data_ptr()
                and last.data_ptr() + last_nbytes == dest.data_ptr()
            ):
                self.byte_dsts[-1] = last.as_strided((last_nbytes + nbytes,), (1,))
                self.byte_srcs[-1] = (last_offset, last_nbytes + nbytes)
                return
        self.byte_dsts.append(dest.reshape(-1).view(torch.uint8))
        self.byte_srcs.append(src)

    def wire(self) -> WireBucket:
        return list(self.pieces)

    def prepare(self, dtypes: list[torch.dtype], copies: list[list[Any]]) -> None:
        """Build the copy-out from each piece's sent dtype and destinations."""
        self.layout = flat_layout([box for _, box in self.pieces], dtypes)
        for i, ((_, box), dests, dtype) in enumerate(
            zip(self.pieces, copies, dtypes, strict=True)
        ):
            for dest in dests:
                if isinstance(dest, _Staging):
                    rows = tuple(
                        slice(a - o, b - o) for (a, b), (o, _) in zip(box, dest.box)
                    )
                    self.staged.append((i, dest, rows))
                elif dest.is_contiguous() and dest.dtype == dtype:
                    self._add_byte_copy(self.layout[i][:2], dest)
                else:
                    self.typed.append((i, dest))
                    self.typed_dsts.append(dest)


@dataclass(slots=True)
class _PullHooks:
    """The transport, injected so tests can run the engine without Ray."""

    # Issue a pull of `bucket` in sync `sync_idx`, to land in `target`;
    # return its handle.
    issue: "Callable[[_Bucket, int, torch.Tensor], Any]"
    # Block until at least one handle is ready; return every ready one.
    wait: "Callable[[list[Any]], list[Any]]"
    # Fetch ready handles into their targets (the same order); raise if one
    # did not land there.
    get: "Callable[[list[Any], list[torch.Tensor]], None]"
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
        self._planned = False
        self._buckets: list[_Bucket] = []
        # Receive arena: `num_buffers` slots of the largest bucket of at most
        # `buffer_bytes`, pinned with NIXL from planning until the plan is
        # dropped.
        self._arena: torch.Tensor | None = None
        self._slot_bytes = 0

    def init_transfer_engine(self, init_info: ShardedRDTWeightTransferInitInfo) -> None:
        """Check the configuration and bind the producers. Touches no model
        state; the plan is built at the next `start_weight_update`.

        Raises:
            ValueError: invalid settings or metadata.
            RuntimeError: the model cannot be planned (`check_plannable`:
                e.g. `VLLM_RELOAD_MODE` is not `direct`), or a producer
                cannot be found.

        """
        from vllm.model_executor.model_loader.reload import check_plannable

        validate_int("buffer_bytes", init_info.buffer_bytes, minimum=1)
        validate_int("num_buffers", init_info.num_buffers, minimum=1)
        validate_int("num_consumers", init_info.num_consumers, minimum=0)
        try:
            check_plannable(self.model)
        except RuntimeError as e:
            raise RuntimeError(
                f"The 'sharded_rdt' weight transfer backend cannot plan this model: {e}"
            ) from e
        check_ray_rdt_version()
        metas = _metas(init_info)
        num_consumers = init_info.num_consumers or (
            self.parallel_config.world_size_across_dp
        )
        per_replica = init_info.workers_per_replica or num_consumers
        if init_info.num_replicas > 1 and (
            per_replica * init_info.num_replicas != num_consumers
        ):
            raise ValueError(
                f"Sharded RDT: num_replicas={init_info.num_replicas}, but the "
                f"trainer splits {num_consumers} workers {per_replica} per "
                "deployment; set the trainer's workers_per_replica"
            )
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
        # distinct per worker of the fleet: deployment, DP index, rank in TP*PP
        pc = self.parallel_config
        local = pc.data_parallel_index * pc.world_size + pc.rank
        consumer_id = init_info.replica_rank * per_replica + local
        if (
            init_info.replica_rank < 0
            or local >= per_replica
            or consumer_id >= num_consumers
        ):
            raise ValueError(
                f"Sharded RDT: this worker is worker {local} of deployment "
                f"{init_info.replica_rank}, but the trainer expects "
                f"{num_consumers} workers, {per_replica} per deployment; check "
                "num_consumers, workers_per_replica and replica_rank"
            )
        producers = _resolve_producers(init_info)

        self._drop_plan()
        self._metas = metas
        nbytes = [meta_nbytes(m) for m in metas]
        self._units = {
            m.name: u
            for m, u in zip(metas, gather_units(nbytes, init_info.buffer_bytes))
        }
        self._router = router
        self._consumer_id = consumer_id
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

    def start_weight_update(self) -> None:
        """Begin a direct reload; the first one after an init also plans and
        sends each producer its buckets.

        Raises:
            RuntimeError: not initialized.

        """
        from vllm.model_executor.model_loader.reload import plan_loads, start_reload

        if self._metas is None:
            raise RuntimeError(
                "Sharded RDT engine not initialized; call init_transfer_engine()."
            )
        start_reload(self.model)
        if self._planned:
            return
        # Trace the real tensors `start_reload` just viewed back, so every
        # destination is the one direct reload restores.
        try:
            t0 = time.perf_counter()
            plan = plan_loads(self.model, self._metas)
            buckets = self._build_buckets(plan)
            self._send_buckets(buckets)
            self._pin_arena(buckets)
        except Exception as e:
            e.add_note(
                "Sharded RDT: planning failed after start_reload, so the model "
                "is mid-reload; the next start_weight_update plans again"
            )
            raise
        self._planned, self._buckets = True, buckets
        logger.info(
            "Sharded RDT plan: %d buckets, %d pieces (%d loaded in full), "
            "%.1f MiB, receive arena %.1f MiB, in %.2f s",
            len(buckets),
            sum(len(b.pieces) for b in buckets),
            sum(len(b.staged) for b in buckets),
            sum(b.nbytes for b in buckets) / (1 << 20),
            self._arena.numel() / (1 << 20) if self._arena is not None else 0.0,
            time.perf_counter() - t0,
        )

    def _build_buckets(self, plan: "LoadPlan") -> list[_Bucket]:
        """Route, split and order this worker's pieces into pulls."""
        from vllm.model_executor.model_loader.reload import LoadPlan

        assert isinstance(plan, LoadPlan) and self._metas is not None
        router = self._router
        assert router is not None
        groups: dict[tuple[int, int], list[_Piece]] = defaultdict(list)
        for meta in self._metas:
            boxes = plan.get_boxes(meta.name)
            if boxes is None:
                continue
            unit = self._units[meta.name]
            key = (router.producer_for(self._consumer_id, meta.name, unit), unit)
            dest = plan.get_dest(meta.name)
            if dest is None:
                (box,) = boxes
                staging = _Staging(meta.name, box)
                groups[key].append((meta.name, box, meta.dtype, [staging]))
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
                groups[key].append((meta.name, box, meta.dtype, by_box[box]))

        components = router.components()
        unit_rank = _component_unit_rank(self._metas, self._units, router, components)
        # Pieces adjacent in one storage (a fused tensor's slices, e.g.
        # per-expert weights) go next to each other in the bucket and land
        # with one copy. Storages rank by first use, not address, so every
        # replica builds the same buckets.
        storages: dict[int, int] = {}
        for pieces in groups.values():
            pieces.sort(key=lambda piece: _by_destination(piece, storages))
        cap = self._bucket_cap(groups)
        buckets: list[_Bucket] = []
        for (producer, unit), pieces in groups.items():
            pieces = [seg for piece in pieces for seg in _rows(piece, cap)]
            for chunk in _cut(pieces, cap):
                bucket = _Bucket(
                    producer=producer,
                    seq=-1,
                    unit=unit,
                    pieces=[(name, box) for name, box, *_ in chunk],
                    nbytes=sum(piece_nbytes(box, dt) for _, box, dt, _ in chunk),
                )
                bucket.prepare([dt for _, _, dt, _ in chunk], [d for *_, d in chunk])
                buckets.append(bucket)

        def order(b: _Bucket) -> tuple:
            comp = components[b.producer]
            return (unit_rank[comp][b.unit], comp, b.unit, b.producer)

        buckets.sort(key=order)
        seqs: dict[int, int] = defaultdict(int)
        for b in buckets:
            b.seq = seqs[b.producer]
            seqs[b.producer] += 1
            for _, staging, _ in b.staged:
                staging.count += 1
        return buckets

    def _bucket_cap(self, groups: dict[tuple[int, int], list[_Piece]]) -> int:
        """The size every bucket is cut to: the largest bucket the pieces
        within `buffer_bytes` make, raised to hold one row of every larger one.

        Raises:
            RuntimeError: one row of a piece is over `buffer_bytes`.

        """
        limit = self._buffer_bytes
        regular = [
            [p for p in ps if piece_nbytes(p[1], p[2]) <= limit]
            for ps in groups.values()
        ]
        cap = max(
            (
                sum(piece_nbytes(box, dt) for _, box, dt, _ in chunk)
                for pieces in regular
                for chunk in _cut(pieces, limit)
            ),
            default=limit,
        )
        for pieces in groups.values():
            for name, box, dtype, _ in pieces:
                if piece_nbytes(box, dtype) <= limit:
                    continue
                row = piece_nbytes(((0, 1), *box[1:]), dtype)
                if row > limit:
                    raise RuntimeError(
                        f"Sharded RDT: one row of {name!r} (box {box}) is "
                        f"{row} B, over buffer_bytes={limit}; raise buffer_bytes"
                    )
                cap = max(cap, row)
        return cap

    def _send_buckets(self, buckets: list[_Bucket]) -> None:
        """Tell every producer, including those this worker pulls nothing
        from, which buckets it will pull: a producer releases a unit only
        once every live worker has registered."""
        assert self._hooks is not None
        per_producer: list[list[WireBucket]] = [[] for _ in self._producers]
        for b in buckets:
            per_producer[b.producer].append(b.wire())
        self._hooks.register(per_producer)

    def _pin_arena(self, buckets: list[_Bucket]) -> None:
        """Allocate and register the receive arena: per pull in flight, one
        slot of the largest bucket (at most `buffer_bytes`). Each bucket
        lands at the start of a slot, so its targets and copy-out sources
        are views built here, once."""
        self._slot_bytes = slot = max((b.nbytes for b in buckets), default=0)
        if not slot:
            return
        arena = torch.empty(
            self._num_buffers * slot, dtype=torch.uint8, device=self.device
        )
        register_nixl_memory(arena)
        self._arena = arena
        for b in buckets:
            b.targets = [
                arena[i * slot : i * slot + b.nbytes] for i in range(self._num_buffers)
            ]
            b.srcs = [self._source_views(b, target) for target in b.targets]

    def receive_weights(self, update_info: ShardedRDTWeightTransferUpdateInfo) -> None:
        """Pull every bucket of the plan, `num_buffers` in flight, and apply
        each as it lands.

        Raises:
            RuntimeError: no plan (call `start_weight_update` first).

        """
        validate_int("sync_idx", update_info.sync_idx, minimum=0)
        if not self._planned or self._hooks is None:
            raise RuntimeError(
                "Sharded RDT: no load plan; call start_weight_update() first."
            )
        hooks, sync_idx = self._hooks, update_info.sync_idx
        for bucket in self._buckets:  # a failed sync may have left some partial
            for _, staging, _ in bucket.staged:
                staging.landed, staging.tensor = 0, None
        todo = iter(self._buckets)
        # handle -> (bucket, slot); slots free to land a pull in; per slot,
        # an event recorded after its copy-out, and whether it is pending
        inflight: dict[Any, tuple[_Bucket, int]] = {}
        free = list(range(self._num_buffers))
        cuda = self.device.type == "cuda"
        copied = [torch.cuda.Event() for _ in free] if cuda else []
        pending = [False] * self._num_buffers

        def refill() -> None:
            while free:
                bucket = next(todo, None)
                if bucket is None:
                    return
                slot = free.pop()
                handle = hooks.issue(bucket, sync_idx, bucket.targets[slot])
                inflight[handle] = (bucket, slot)

        # seconds spent waiting for producers, fetching, and applying
        spent = [0.0, 0.0, 0.0]

        def land(ready: list[Any]) -> None:
            # NIXL writes a slot during `get`, so the copy-out that last read
            # it must be done by then; issuing a pull into it may go ahead.
            for _, slot in (inflight[h] for h in ready):
                if pending[slot]:
                    copied[slot].synchronize()
                    pending[slot] = False
            t0 = time.perf_counter()
            landed = [inflight.pop(handle) for handle in ready]
            hooks.get(ready, [bucket.targets[slot] for bucket, slot in landed])
            t1 = time.perf_counter()
            # drop the handles (frees the producers' blocks) and issue the next
            # pulls before copying out
            ready.clear()
            free.extend(slot for _, slot in landed)
            refill()
            t2 = time.perf_counter()
            for bucket, slot in landed:
                self._apply(bucket, slot)
                if cuda:
                    copied[slot].record()
                    pending[slot] = True
            spent[1] += t1 - t0
            spent[2] += time.perf_counter() - t2

        t_start = time.perf_counter()
        refill()
        while inflight:
            t0 = time.perf_counter()
            ready = hooks.wait(list(inflight))
            spent[0] += time.perf_counter() - t0
            land(ready)
        logger.info(
            "Sharded RDT sync %d: %d pulls in %.2f s (waiting %.2f s, fetching "
            "%.2f s, applying %.2f s)",
            sync_idx,
            len(self._buckets),
            time.perf_counter() - t_start,
            *spent,
        )

    def _apply(self, bucket: _Bucket, slot: int) -> None:
        """Copy a bucket landed in receive slot `slot` into its destinations,
        the contiguous ones with one fused byte copy, and load its
        fallbacks."""
        byte_srcs, typed_srcs, staged_srcs = bucket.srcs[slot]
        if byte_srcs:
            torch._foreach_copy_(bucket.byte_dsts, byte_srcs)
        if typed_srcs:
            torch._foreach_copy_(bucket.typed_dsts, typed_srcs)
        loads = []
        for (_, staging, rows), src in zip(bucket.staged, staged_srcs):
            if staging.count == 1:
                loads.append((staging.name, src))
                continue
            if staging.tensor is None:
                staging.tensor = torch.empty(
                    box_shape(staging.box), dtype=src.dtype, device=src.device
                )
            staging.tensor[rows].copy_(src)
            staging.landed += 1
            if staging.landed == staging.count:
                loads.append((staging.name, staging.tensor))
                staging.tensor, staging.landed = None, 0
        if loads:
            self.model.load_weights(loads)

    @staticmethod
    def _source_views(bucket: _Bucket, flat: torch.Tensor) -> tuple:
        """Where each copy of `bucket` reads in `flat`."""
        layout = bucket.layout
        return (
            [flat[offset : offset + nbytes] for offset, nbytes in bucket.byte_srcs],
            [
                flat_view(flat, layout[i]).reshape(dest.shape)
                for i, dest in bucket.typed
            ],
            [flat_view(flat, layout[i]) for i, *_ in bucket.staged],
        )

    def finish_weight_update(self) -> None:
        """Rerun post-load processing on the received weights."""
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)

    def _ray_hooks(self) -> _PullHooks:
        import ray
        from ray.experimental import set_target_for_ref

        consumer_id, producers = self._consumer_id, self._producers

        def issue(bucket: _Bucket, sync_idx: int, target: torch.Tensor) -> Any:
            ref = producers[bucket.producer].bucket.remote(
                consumer_id, bucket.seq, sync_idx
            )
            # A slice of the pinned arena, so Ray registers nothing per pull.
            # Ray limitation: it holds a target only weakly; the plan keeps
            # every target alive.
            set_target_for_ref(ref, [target])
            return ref

        def wait(refs: list[Any]) -> list[Any]:
            return ray.wait(refs, num_returns=1)[0]

        def get(refs: list[Any], targets: list[torch.Tensor]) -> None:
            for tensors, target in zip(ray.get(refs), targets, strict=True):
                # Ray limitation: it receives into a buffer of its own,
                # silently, when it cannot use the target.
                if len(tensors) != 1 or tensors[0].data_ptr() != target.data_ptr():
                    raise RuntimeError(
                        "Sharded RDT: a pull did not land in its receive slot"
                    )

        def register(per_producer: list[list[WireBucket]]) -> None:
            ray.get(
                [
                    actor.register_consumer.remote(consumer_id, buckets)
                    for actor, buckets in zip(producers, per_producer)
                ]
            )

        return _PullHooks(issue=issue, wait=wait, get=get, register=register)

    def _drop_plan(self) -> None:
        self._planned = False
        self._buckets = []
        if self._arena is not None:
            deregister_nixl_memory(self._arena)
            self._arena = None

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


def _by_destination(piece: _Piece, storages: dict[int, int]) -> tuple[int, int, int]:
    """Sort key: a piece with one contiguous destination of its dtype by its
    storage's first-use rank and byte offset; every other piece after them,
    in plan order."""
    _, _, dtype, dests = piece
    if len(dests) == 1:
        (dest,) = dests
        if (
            isinstance(dest, torch.Tensor)
            and dest.is_contiguous()
            and dest.dtype == dtype
        ):
            rank = storages.setdefault(dest.untyped_storage().data_ptr(), len(storages))
            return (0, rank, dest.storage_offset() * dest.element_size())
    return (1, 0, 0)


def _rows(piece: _Piece, cap: int) -> list[_Piece]:
    """A piece over `cap` cut along dim 0 of its box into segments of the
    most whole rows that fit `cap` (with their alignment): ordinary pieces,
    whose destinations are the same rows of the piece's, so every segment is
    a view on both sides (a staging is indexed by the segment's box).

    Raises:
        RuntimeError: a destination's rows are not rows of the box.

    """
    name, box, dtype, dests = piece
    if piece_nbytes(box, dtype) <= cap:
        return [piece]
    shape = box_shape(box)
    rows = (cap - cap % PIECE_ALIGNMENT) // (math.prod(shape[1:]) * dtype.itemsize)
    assert rows, f"{name!r}: the cap {cap} B holds no row"
    by_rows = []
    for dest in dests:
        if isinstance(dest, torch.Tensor) and tuple(dest.shape) != shape:
            if not dest.is_contiguous():
                raise RuntimeError(
                    f"Sharded RDT: {name!r} (box {box}) is over the {cap} B "
                    "bucket cap and its destination is not laid out by rows "
                    "of it; raise buffer_bytes"
                )
            dest = dest.view(shape)
        by_rows.append(dest)
    (start, stop), rest = box[0], box[1:]
    return [
        (
            name,
            ((start + r, min(start + r + rows, stop)), *rest),
            dtype,
            [d[r : r + rows] if isinstance(d, torch.Tensor) else d for d in by_rows],
        )
        for r in range(0, stop - start, rows)
    ]


def _cut(pieces: list[_Piece], budget: int) -> list[list[_Piece]]:
    """Consecutive runs of at most `budget` bytes; a larger piece goes alone."""
    units = gather_units([piece_nbytes(p[1], p[2]) for p in pieces], budget)
    return [
        [piece for _, piece in run]
        for _, run in itertools.groupby(zip(units, pieces), key=lambda x: x[0])
    ]


def _component_unit_rank(
    metas: list[ParamMeta],
    units: dict[str, int],
    router: RdtRouter,
    components: list[int],
) -> dict[int, dict[int, int]]:
    """Per coupling component, each unit's index among the units its ranks
    gather. Ordering pulls by it keeps every component in gather order (so
    no rank waits on a unit its consumers pull later) while independent
    components, pipeline stages, pull side by side."""
    rank: dict[int, dict[int, int]] = defaultdict(dict)
    for meta in metas:
        comp = components[router.owners(meta.name)[0]]
        seen = rank[comp]
        seen.setdefault(units[meta.name], len(seen))
    return rank
