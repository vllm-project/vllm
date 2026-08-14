# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-side weight transfer engine built on NCCL M2N (`nccl_m2n`).

The trainer and the inference workers share one NCCL communicator: trainer ranks
occupy `[0, T)`, workers `[T, T + N)`. Each parameter is moved with a single
`nccl.m2n.reshard`, which redistributes it from the trainer's layout (FSDP / EP
/ arbitrary DTensor sharding) to the inference layout — so the trainer sends its
local shards and never all-gathers a full tensor, which is what the broadcast
NCCL backend forces it to do.

Each worker currently receives the whole tensor and loads it through
`load_weights`, exactly as the broadcast backend does, so the two are directly
comparable. Resharding into each worker's own shard is a follow-up.

Both meshes participate in every reshard, in the same order, so this backend has
the same concurrency shape as the broadcast NCCL backend: the worker must be
inside `receive_weights` while the trainer is sending. Driving that from the
trainer is the trainer engine's job.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_common import (
    MESH_NDIMS,
    REPLICATED,
    M2NMesh,
    M2NParamMeta,
    check_plan_limits,
    check_transferable,
    comm_ptr,
    import_m2n,
    resolve_layout,
    to_mesh,
    to_placements,
    validate_layout,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    worker_init_process_group,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

__all__ = [
    "M2NWeightTransferInitInfo",
    "M2NWeightTransferUpdateInfo",
    "M2NWeightTransferEngine",
]


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


@dataclass
class M2NWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info: the rendezvous plus the full transfer plan.

    Layouts are static for the whole run, so they ride the one-time init
    handshake and the per-round update info stays a list of names. Everything is
    plain JSON so the HTTP control plane carries it unchanged.
    """

    master_address: str
    master_port: int
    rank_offset: int
    """First worker rank, i.e. the number of trainer ranks."""
    world_size: int
    """Trainer ranks + all inference workers."""
    src_mesh_dims: list[int]
    """The trainer's mesh, shared by every parameter (`start_rank` is 0: the
    trainer occupies the front of the communicator)."""
    dst_mesh_dims: list[int]
    """The inference mesh, starting at `rank_offset`. Declared rather than
    derived so both sides describe the destination identically."""
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    src_placements: list[list[int] | None]
    """Per parameter, relative to `src_mesh_dims`; `None` means replicated."""
    max_cta: int | None = None

    def __post_init__(self) -> None:
        num_params = len(self.names)
        for label, values in (
            ("dtype_names", self.dtype_names),
            ("shapes", self.shapes),
            ("src_placements", self.src_placements),
        ):
            if len(values) != num_params:
                raise ValueError(
                    f"`{label}` should be of the same size as `names`: "
                    f"got {len(values)} and {num_params}"
                )
        if self.rank_offset < 1 or self.rank_offset >= self.world_size:
            raise ValueError(
                f"`rank_offset` ({self.rank_offset}) must leave at least one "
                f"trainer rank and one worker in world_size {self.world_size}"
            )
        num_workers = self.world_size - self.rank_offset
        dst = tuple(self.dst_mesh_dims)
        if len(dst) != MESH_NDIMS or dst[0] * dst[1] != num_workers:
            raise ValueError(
                f"`dst_mesh_dims` {dst} must be {MESH_NDIMS} dims covering the "
                f"{num_workers} inference workers"
            )


@dataclass
class M2NWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info: which parameters this chunk carries, in order.

    Shapes, dtypes and layouts were fixed at init, so a round only needs to say
    what is coming and in what order — both sides must issue their reshards in
    exactly that order.
    """

    names: list[str]


# ---------------------------------------------------------------------------
# Worker engine
# ---------------------------------------------------------------------------


class M2NWeightTransferEngine(
    WeightTransferEngine[M2NWeightTransferInitInfo, M2NWeightTransferUpdateInfo]
):
    """Inference-side engine: receives each parameter with one reshard.

    Every reshard gathers the parameter from whatever layout the trainer holds
    it in and delivers the **whole** tensor to each worker, which then hands it
    to `load_weights` — the same thing the broadcast NCCL backend does, and
    therefore the same behavior to compare against. The win over broadcast is on
    the trainer side: it sends its local shards and never materializes a full
    tensor. Resharding straight into each worker's own shard is a follow-up.
    """

    init_info_cls = M2NWeightTransferInitInfo
    update_info_cls = M2NWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyNcclCommunicator | None = None
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._src_mesh: M2NMesh | None = None
        self._dst_mesh: M2NMesh | None = None
        self._index: dict[str, int] = {}

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Join the trainer's communicator and rebuild the transfer plan.

        Every precondition (dtype, tensor rank, divisibility) is checked here so
        a bad plan fails the init RPC instead of hanging the first collective.
        """
        self._m2n = import_m2n()
        self._metas = []

        self._src_mesh = M2NMesh(tuple(init_info.src_mesh_dims), 0)
        if self._src_mesh.size != init_info.rank_offset:
            raise ValueError(
                f"source mesh covers {self._src_mesh.size} ranks, but there are "
                f"{init_info.rank_offset} trainer ranks"
            )
        # The inference topology, as declared by the trainer. This is the mesh
        # a sharded destination is placed over; a replicated one is described
        # by the alternate descriptor `resolve_layout` derives from it, which
        # covers the same ranks but is not this mesh.
        self._dst_mesh = M2NMesh(tuple(init_info.dst_mesh_dims), init_info.rank_offset)

        for name, name_dtype, shape, placements in zip(
            init_info.names,
            init_info.dtype_names,
            init_info.shapes,
            init_info.src_placements,
        ):
            dtype = getattr(torch, name_dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"parameter '{name}' has invalid dtype name "
                    f"'{name_dtype}'; expected the name of a torch.dtype"
                )
            check_transferable(name, dtype, shape)
            codes = REPLICATED if placements is None else tuple(placements)
            src_mesh, src_placements = resolve_layout(
                self._src_mesh, codes, f"parameter '{name}' source placements"
            )
            validate_layout(src_mesh, src_placements, shape, "source")
            dst_mesh, dst_placements = resolve_layout(self._dst_mesh, REPLICATED)
            validate_layout(dst_mesh, dst_placements, shape, "destination")
            check_plan_limits(
                (src_mesh, src_placements), (dst_mesh, dst_placements), name
            )
            self._metas.append(M2NParamMeta(name, dtype, tuple(shape), codes))

        self._index = {meta.name: i for i, meta in enumerate(self._metas)}

        self.model_update_group = worker_init_process_group(
            NCCLWeightTransferInitInfo(
                master_address=init_info.master_address,
                master_port=init_info.master_port,
                rank_offset=init_info.rank_offset,
                world_size=init_info.world_size,
            ),
            self.parallel_config,
        )
        self._handle = self._m2n.Handle.create(
            self._m2n.Config(max_cta=init_info.max_cta)
        )

    def start_weight_update(self) -> None:
        """Initialize layerwise reloading for the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        if self._handle is None or self.model_update_group is None:
            raise RuntimeError(
                "nccl_m2n weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

        requested = []
        for name in update_info.names:
            index = self._index.get(name)
            if index is None:
                raise ValueError(
                    f"parameter '{name}' was not declared at init; the "
                    "trainer must send the same parameter set it announced"
                )
            requested.append((name, index))

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        comm = comm_ptr(self.model_update_group)
        stream = torch.cuda.current_stream()
        with disable_mtp_completeness_check():
            for name, index in requested:
                meta = self._metas[index]
                buffer = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
                self._reshard(comm, stream, meta, buffer)
                # `load_weights` reads on the host stream, so the transfer has
                # to have landed before it runs.
                stream.synchronize()
                self.model.load_weights([(name, buffer)])
                del buffer

    def _reshard(
        self,
        comm: int,
        stream: torch.cuda.Stream,
        meta: M2NParamMeta,
        buffer: torch.Tensor,
    ) -> None:
        m2n = self._m2n
        src_mesh, src_placements = resolve_layout(self._src_mesh, meta.placements)
        dst_mesh, dst_placements = resolve_layout(self._dst_mesh, REPLICATED)
        m2n.reshard(
            None,
            buffer,
            comm,
            stream,
            src_mesh=to_mesh(m2n, src_mesh),
            src_placements=to_placements(m2n, src_placements),
            dst_mesh=to_mesh(m2n, dst_mesh),
            dst_placements=to_placements(m2n, dst_placements),
            handle=self._handle,
        )

    def shutdown(self) -> None:
        if self._handle is not None:
            # M2N does not synchronize caller streams on finalize.
            torch.accelerator.synchronize()
            self._handle.destroy()
            self._handle = None
        self.model_update_group = None
