# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight transfer engine for the NCCL M2N backend.

Symmetric to `M2NWeightTransferEngine` but in the training process. Every
trainer rank joins the shared communicator and runs every reshard, sending its
own local shard; only rank 0 touches the inference control plane.

Workers join the communicator during init and run reshard from inside
`update_weights`, so those two RPCs overlap with work on this side. Both run on
a helper thread and are joined afterwards, which keeps that overlap inside the
engine -- where the `TrainerWeightTransferEngine` contract puts it -- rather than
in every caller.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from typing_extensions import Self

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATED,
    M2NMesh,
    M2NParamMeta,
    Placements,
    check_transferable,
    comm_ptr,
    import_m2n,
    publish_destination_placements,
    resolve_layout,
    to_mesh,
    to_placements,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_source import M2NWeightSource
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    trainer_init,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = init_logger(__name__)

__all__ = ["M2NTrainerInitInfo", "M2NTrainerWeightTransferEngine"]


def _dtype_name(dtype: torch.dtype) -> str:
    """Short name for a dtype, e.g. `bfloat16`, for the worker-side init dict."""
    return str(dtype).split(".")[-1]


@dataclass
class M2NTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for nccl_m2n.

    `rank` (from `TrainerInitInfo`) is this trainer process's rank; it is also
    its rank in the shared communicator, since the trainer occupies
    `[0, num_trainer_ranks)`. Rank 0 drives the control plane.
    """

    backend: ClassVar[str] = "nccl_m2n"

    master_address: str
    master_port: int
    world_size: int
    """Trainer ranks + all inference workers."""
    num_trainer_ranks: int = 1
    dst_mesh_dims: tuple[int, int] | None = None
    """How the inference ranks are laid out: axis 0 replicates and axis 1
    shards. The trainer declares it so both sides describe the destination
    identically; defaults to a flat `(num_workers, 1)`, which is all a
    replicated destination needs."""
    max_cta: int | None = None

    @property
    def destination_mesh_dims(self) -> tuple[int, int]:
        """`dst_mesh_dims`, or a flat mesh over every inference worker."""
        if self.dst_mesh_dims is not None:
            return tuple(self.dst_mesh_dims)
        return (self.world_size - self.num_trainer_ranks, 1)

    def __post_init__(self) -> None:
        """Reject rank counts or destination meshes that cannot form a group."""
        if not 0 < self.num_trainer_ranks < self.world_size:
            raise ValueError(
                f"`num_trainer_ranks` ({self.num_trainer_ranks}) must leave at "
                f"least one worker in world_size {self.world_size}"
            )
        num_workers = self.world_size - self.num_trainer_ranks
        if self.dst_mesh_dims is not None:
            dims = tuple(self.dst_mesh_dims)
            if len(dims) != 2 or dims[0] * dims[1] != num_workers:
                raise ValueError(
                    f"`dst_mesh_dims` {dims} must be two dims covering the "
                    f"{num_workers} inference workers"
                )
        if not 0 <= self.rank < self.num_trainer_ranks:
            raise ValueError(
                f"trainer `rank` ({self.rank}) must be below "
                f"`num_trainer_ranks` ({self.num_trainer_ranks})"
            )


class M2NTrainerWeightTransferEngine(TrainerWeightTransferEngine[M2NTrainerInitInfo]):
    """Trainer-side engine: sends every rank's local shard, once per parameter.

    Called on every trainer rank. All ranks join the communicator and run every
    reshard; only rank 0 touches the control plane.
    """

    init_info_cls = M2NTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: M2NWeightSource,
        is_controller: bool,
        num_trainer_ranks: int,
    ) -> None:
        """Hold the client and source; `trainer_init` fills in the group."""
        super().__init__(client=client, source=source, is_sender=is_controller)
        self.is_controller = is_controller
        self.num_trainer_ranks = num_trainer_ranks
        self.group: PyNcclCommunicator | None = None
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._src_mesh: M2NMesh | None = None
        self._dst_mesh: M2NMesh | None = None
        self._dst_placements: list[Placements | None] = []
        self._executor: ThreadPoolExecutor | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: M2NTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
    ) -> Self:
        """Build the engine and rendezvous with the inference side.

        Runs on *every* trainer rank. Rank 0 additionally ships the transfer
        plan to the workers and drives the control plane; the other ranks only
        build local state and join the communicator. Every rank participates
        in every reshard and sends its local shard.

        The ordering here is not arbitrary -- see the comments inline. In
        short: validate before anything blocks; start the init RPC without
        waiting (workers join the communicator inside that RPC, so it cannot
        return until this side has joined too); then join the NCCL
        communicator; then create the M2N handle; and only then wait for the RPC
        to finish.
        """
        # The base WeightSource carries name/dtype/shape but not layout, which
        # is not enough to plan a reshard. Fail here rather than at the first
        # missing attribute deep inside the send loop.
        if not isinstance(source, M2NWeightSource):
            raise TypeError(
                "nccl_m2n needs per-parameter layouts, so its source must be an "
                f"M2NWeightSource (e.g. DTensorModuleSource); got "
                f"{type(source).__name__}"
            )

        # Trainer-side weight transfer engine for this rank. `__init__` only
        # holds the client and source; the communicator, meshes, and M2N handle
        # are filled in below.
        engine = cls(
            client=client,
            source=source,
            is_controller=init_info.rank == 0,
            num_trainer_ranks=init_info.num_trainer_ranks,
        )
        engine._m2n = import_m2n()

        engine._src_mesh = source.mesh()
        if engine._src_mesh.size != init_info.num_trainer_ranks:
            raise ValueError(
                f"source mesh covers {engine._src_mesh.size} ranks, but there "
                f"are {init_info.num_trainer_ranks} trainer ranks"
            )

        # Validate the whole plan before any rendezvous. A layout m2n cannot
        # express is a mismatched collective at send time, which hangs rather
        # than raises -- so every precondition is checked while an exception can
        # still propagate to the caller.
        engine._metas = source.metadata()
        for meta in engine._metas:
            check_transferable(meta.name, meta.dtype, meta.shape)
            mesh, placements = resolve_layout(engine._src_mesh, meta.placements)
            validate_layout(mesh, placements, meta.shape, "source")

        rendezvous: Future | None = None
        if engine.is_controller:
            engine._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            # Only trainer rank 0 starts the control-plane RPC. Workers join
            # the communicator inside it, so submit it first, join below, and
            # wait for it at the end.
            rendezvous = engine._executor.submit(
                client.init_weight_transfer_engine,
                engine._worker_init_info(init_info),
            )

        # Every trainer rank joins -- reshard is collective over the whole
        # communicator. Trainer ranks take
        # [0, num_trainer_ranks) and the workers follow, which is the
        # contiguous-interval layout m2n requires of both meshes.
        engine.group = trainer_init(
            NCCLWeightTransferInitInfo(
                master_address=init_info.master_address,
                master_port=init_info.master_port,
                rank_offset=init_info.num_trainer_ranks,
                world_size=init_info.world_size,
            ),
            rank=init_info.rank,
        )

        # The trainer declares the inference topology; the worker uses exactly
        # this, so neither side has to infer the other's factorization.
        engine._dst_mesh = M2NMesh(
            init_info.destination_mesh_dims, init_info.num_trainer_ranks
        )
        engine._dst_placements = engine._receive_destination_plan(
            init_info.num_trainer_ranks
        )
        logger.info(
            "nccl_m2n trainer ready: %d parameters, trainer ranks [0, %d), "
            "inference ranks [%d, %d)",
            len(engine._metas),
            init_info.num_trainer_ranks,
            init_info.num_trainer_ranks,
            init_info.world_size,
        )

        # Created after the group so a failed rendezvous does not leave a live
        # handle behind; shutdown() releases it once caller streams are synced.
        engine._handle = engine._m2n.Handle.create(
            engine._m2n.Config(max_cta=init_info.max_cta)
        )

        # Now safe to collect: both sides have joined, so the RPC has returned
        # or raised. Doing this last means a worker-side init failure surfaces
        # here, as an exception from trainer_init, rather than at the first send.
        if rendezvous is not None:
            rendezvous.result()
        return engine

    def _receive_destination_plan(
        self, first_worker_rank: int
    ) -> list[Placements | None]:
        """Receive and validate the inference-side per-parameter placements."""
        if self.group is None or self._dst_mesh is None:
            raise RuntimeError(
                "destination plan requires the communicator and destination mesh"
            )

        # The first inference worker publishes one placement per parameter.
        # Every trainer rank receives the complete plan in this single,
        # initialization-time broadcast.
        placements = publish_destination_placements(self.group, first_worker_rank, None)
        # The send loop matches placements to metadata by position, so reject
        # an incomplete or oversized plan before entering any M2N collective.
        if len(placements) != len(self._metas):
            raise RuntimeError(
                f"inference side planned {len(placements)} parameters, but "
                f"this trainer announced {len(self._metas)}"
            )
        # Check every worker-selected destination against the corresponding
        # global parameter shape before the first transfer.
        for meta, placement in zip(self._metas, placements):
            mesh, resolved = resolve_layout(self._dst_mesh, placement)
            validate_layout(mesh, resolved, meta.shape, "destination")
        return placements

    def _worker_init_info(self, init_info: M2NTrainerInitInfo) -> dict[str, Any]:
        """Handshake payload: rendezvous, both meshes, and the transfer plan."""
        return {
            "master_address": init_info.master_address,
            "master_port": init_info.master_port,
            "rank_offset": init_info.num_trainer_ranks,
            "world_size": init_info.world_size,
            "src_mesh_dims": list(self._src_mesh.dims),
            "dst_mesh_dims": list(init_info.destination_mesh_dims),
            "names": [m.name for m in self._metas],
            "dtype_names": [_dtype_name(m.dtype) for m in self._metas],
            "shapes": [list(m.shape) for m in self._metas],
            "src_placements": [
                None if m.placements is REPLICATED else list(m.placements)
                for m in self._metas
            ],
            "max_cta": init_info.max_cta,
        }

    def send_weights(self) -> None:
        """Drive one update round: start, reshard concurrently, then finish."""
        if self._handle is None or self.group is None:
            raise RuntimeError("nccl_m2n trainer engine is not initialized")

        update: Future | None = None
        if self.is_controller:
            self.client.start_weight_update()
            # The workers reshard from inside `update_weights`, concurrently
            # with the sends below.
            update = self._executor.submit(
                self.client.update_weights,
                {"names": [m.name for m in self._metas]},
            )

        try:
            self._send()
        finally:
            if update is not None:
                update.result()

        if self.is_controller:
            self.client.finish_weight_update()

    def _send(self) -> None:
        """Reshard each local shard into a replicated full tensor on the workers."""
        m2n = self._m2n
        comm = comm_ptr(self.group)
        stream = torch.cuda.current_stream()

        for index, (name, tensor) in enumerate(self.source):
            meta = self._metas[index]
            if name != meta.name:
                raise RuntimeError(
                    f"weight source yielded '{name}' at position {index}, but "
                    f"its metadata declared '{meta.name}'; the source must "
                    "iterate in a stable order"
                )
            shard = tensor.detach().contiguous()
            src_mesh, src_placements = resolve_layout(self._src_mesh, meta.placements)
            dst_mesh, dst_placements = resolve_layout(
                self._dst_mesh, self._dst_placements[index]
            )
            m2n.reshard(
                shard,
                None,
                comm,
                stream,
                src_mesh=to_mesh(m2n, src_mesh),
                src_placements=to_placements(m2n, src_placements),
                dst_mesh=to_mesh(m2n, dst_mesh),
                dst_placements=to_placements(m2n, dst_placements),
                handle=self._handle,
            )
            # A custom source may yield temporary storage allocated on another
            # stream. M2N retains only its device pointer for asynchronous work,
            # so tell the allocator not to reuse that storage until this stream
            # has finished with it. For storage allocated on this stream,
            # record_stream is effectively a no-op.
            shard.record_stream(stream)

        # As in the other weight-transfer backends, wait before returning
        # rather than relying on same-stream ordering. The caller may then
        # safely mutate parameters or continue work on another stream.
        stream.synchronize()

    def shutdown(self) -> None:
        """Destroy the m2n handle, join the helper thread, and drop the group."""
        if self._handle is not None:
            torch.accelerator.synchronize()
            self._handle.destroy()
            self._handle = None
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        self.group = None
