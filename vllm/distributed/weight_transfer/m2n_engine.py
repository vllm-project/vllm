# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-side weight transfer engine built on NCCL M2N (`nccl_m2n`).

The trainer and the inference workers share one NCCL communicator: trainer ranks
occupy `[0, T)`, workers `[T, T + N)`. Each parameter is moved with a single
`nccl.m2n.reshard`, which redistributes it from the trainer's layout (FSDP / EP
/ arbitrary DTensor sharding) to the inference layout — so the trainer sends its
local shards and never all-gathers a full tensor, which is what the broadcast
NCCL backend forces it to do.

When an incoming checkpoint parameter maps directly to a live vLLM parameter,
M2N reshards into that worker's local model storage. Parameters whose names or
layouts cannot be resolved receive a full tensor and fall back to
`load_weights`, preserving the existing backend's loading behavior.

Both meshes participate in every reshard, in the same order, so this backend has
the same concurrency shape as the broadcast NCCL backend: the worker must be
inside `receive_weights` while the trainer is sending. Driving that from the
trainer is the trainer engine's job.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_common import (
    DESTINATION_SHARD_AXIS,
    MESH_NDIMS,
    REPLICATED,
    M2NMesh,
    M2NParamMeta,
    Placements,
    check_plan_limits,
    check_transferable,
    comm_ptr,
    import_m2n,
    publish_destination_placements,
    resolve_layout,
    to_mesh,
    to_placements,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_layout import (
    M2NDestination,
    resolve_parameter_destinations,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    decode_nccl_unique_id,
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


@dataclass(kw_only=True)
class M2NWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info: the rendezvous plus the full transfer plan.

    Layouts are static for the whole run, so they ride the one-time init
    handshake and the per-round update info stays a list of names. Everything is
    plain JSON so the HTTP control plane carries it unchanged.
    """

    master_address: str | None = None
    master_port: int | None = None
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
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    max_cta: int | None = None

    def __post_init__(self) -> None:
        _ = self.nccl_unique_id_bytes
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

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        return decode_nccl_unique_id(
            master_address=self.master_address,
            master_port=self.master_port,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="M2NWeightTransferInitInfo",
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

    Resolvable parameters are resharded from the trainer layout directly into
    each worker's live local model storage. Unresolvable parameters receive a
    full tensor and use `load_weights`. In both cases, the trainer sends its
    local shards without materializing a full tensor.
    """

    init_info_cls = M2NWeightTransferInitInfo
    update_info_cls = M2NWeightTransferUpdateInfo
    # The destination plan is resolved against the model this engine was built
    # for, so it cannot be retargeted at a draft model mid-flight.
    supports_draft_weight_update = False

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
        self._parameter_destinations: list[M2NDestination] = []
        self._index: dict[str, int] = {}
        self._uses_load_weights = False

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Join the trainer's communicator and rebuild the transfer plan.

        Every precondition (dtype, tensor rank, divisibility) is checked here so
        a bad plan fails the init RPC instead of hanging the first collective.
        """
        self._m2n = import_m2n()
        self._metas = []

        self._src_mesh = M2NMesh(
            cast(tuple[int, int], tuple(init_info.src_mesh_dims)), 0
        )
        if self._src_mesh.size != init_info.rank_offset:
            raise ValueError(
                f"source mesh covers {self._src_mesh.size} ranks, but there are "
                f"{init_info.rank_offset} trainer ranks"
            )
        # The inference topology, as declared by the trainer. This is the mesh
        # a sharded destination is placed over; a replicated one is described
        # by the alternate descriptor `resolve_layout` derives from it, which
        # covers the same ranks but is not this mesh.
        self._dst_mesh = M2NMesh(
            cast(tuple[int, int], tuple(init_info.dst_mesh_dims)),
            init_info.rank_offset,
        )

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
            codes = (
                REPLICATED
                if placements is None
                else cast(Placements, tuple(placements))
            )
            src_mesh, src_placements = resolve_layout(
                self._src_mesh, codes, f"parameter '{name}' source placements"
            )
            validate_layout(src_mesh, src_placements, shape, "source")
            # Destination checks are deferred until loader-aware resolution;
            # assuming replication here can reject a valid sharded plan.
            self._metas.append(M2NParamMeta(name, dtype, tuple(shape), codes))

        parallel_config = self.parallel_config
        shard_axis_size = self._dst_mesh.dims[DESTINATION_SHARD_AXIS]
        quantization = getattr(self.model_config, "quantization", None)
        # A quantized weight loader may dequantize / transpose / requantize, and
        # under PP a rank's local shape is not the checkpoint shape split over
        # the shard axis. Both break shape-derived resolution, so take the
        # fallback.
        allow_direct = (
            quantization is None and parallel_config.pipeline_parallel_size == 1
        )
        # Match every incoming checkpoint parameter to its rank-local target.
        # Resolvable parameters can be resharded directly into live model
        # storage; the rest receive a full tensor for `load_weights`.
        self._parameter_destinations = resolve_parameter_destinations(
            self.model,
            [m.name for m in self._metas],
            [m.dtype for m in self._metas],
            [m.shape for m in self._metas],
            num_workers=self._dst_mesh.size,
            shard_axis_size=shard_axis_size,
            allow_direct=allow_direct,
        )
        # Check the resolved layout rather than a provisional replicated one.
        # For example, 32 source shards feeding 4 destination shards may need
        # only 8 sources per destination, while a replicated plan needs all 32.
        for meta, destination in zip(self._metas, self._parameter_destinations):
            src_mesh, src_placements = resolve_layout(
                self._src_mesh,
                meta.placements,
                f"parameter '{meta.name}' source placements",
            )
            dst_mesh, dst_placements = resolve_layout(
                self._dst_mesh,
                destination.placements,
                f"parameter '{meta.name}' destination placements",
            )
            validate_layout(dst_mesh, dst_placements, meta.shape, "destination")
            check_plan_limits(
                (src_mesh, src_placements),
                (dst_mesh, dst_placements),
                meta.name,
            )

        # Update requests carry names only, so cache their plan indices. Track
        # whether any fallback entry requires the `load_weights` lifecycle.
        self._index = {meta.name: i for i, meta in enumerate(self._metas)}
        self._uses_load_weights = any(
            not destination.direct for destination in self._parameter_destinations
        )

        self.model_update_group = worker_init_process_group(
            NCCLWeightTransferInitInfo(
                master_address=init_info.master_address,
                master_port=init_info.master_port,
                nccl_unique_id_b64=init_info.nccl_unique_id_b64,
                rank_offset=init_info.rank_offset,
                world_size=init_info.world_size,
            ),
            parallel_config,
        )
        # Destinations are per-parameter and depend on the inference model, so
        # the trainer cannot derive them. The first worker publishes the plan
        # and every other rank checks its own against it — a silent
        # disagreement between workers would mean mismatched collectives.
        mine = [destination.placements for destination in self._parameter_destinations]
        agreed = publish_destination_placements(
            self.model_update_group, init_info.rank_offset, mine, len(self._metas)
        )
        if agreed != mine:
            mismatched = next(
                meta.name for meta, a, b in zip(self._metas, mine, agreed) if a != b
            )
            raise RuntimeError(
                "inference workers resolved different nccl_m2n destinations "
                f"(first mismatch: '{mismatched}'); all workers must run the "
                "same model and parallel config"
            )

        self._handle = self._m2n.Handle.create(
            self._m2n.Config(max_cta=init_info.max_cta)
        )

    def start_weight_update(self) -> None:
        """Set up layerwise reloading, but only if some parameter needs it.

        Directly-resharded parameters are written in place and never go through
        `load_weights`, so a plan with no fallback entries has nothing to
        reload.
        """
        if not self._uses_load_weights:
            return
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading when the plan uses fallback entries."""
        if not self._uses_load_weights:
            return
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(self.model, self.model_config)

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        """Receive each requested parameter using its initialization-time plan."""
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

        # Reshard lazily so one `load_weights` invocation sees the complete
        # fallback sequence without staging every full tensor at once. Direct
        # destinations still execute in request order as the iterator advances.
        def reshard_requested_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, index in requested:
                meta = self._metas[index]
                destination = self._parameter_destinations[index]
                buffer = destination.tensor
                if buffer is None:
                    buffer = torch.empty(
                        meta.shape, dtype=meta.dtype, device=self.device
                    )

                self._reshard(comm, stream, meta, destination.placements, buffer)

                if destination.tensor is None:
                    # `load_weights` reads on the host stream, so the transfer
                    # has to have landed before it runs.
                    stream.synchronize()
                    yield name, buffer

        has_fallback = any(
            self._parameter_destinations[index].tensor is None for _, index in requested
        )
        with disable_mtp_completeness_check():
            received_weights = reshard_requested_weights()
            if has_fallback:
                # Some model loaders keep invocation-local state to combine
                # related checkpoint tensors, so preserve one iterable scope.
                self.model.load_weights(received_weights)
            else:
                # The iterator performs the reshards. With nothing to yield to
                # `load_weights`, exhaust it here to execute direct transfers.
                for _ in received_weights:
                    pass

    def _reshard(
        self,
        comm: int,
        stream: torch.cuda.Stream,
        src_meta: M2NParamMeta,
        dst_placements: Placements | None,
        dst_buffer: torch.Tensor,
    ) -> None:
        """Reshard one parameter from its trainer layout into a worker buffer.

        `dst_placements` describes the buffer's logical placement over the
        worker mesh; `dst_buffer` is this rank's physical storage.
        """
        assert self._src_mesh is not None
        assert self._dst_mesh is not None
        m2n = self._m2n
        src_mesh, resolved_src_placements = resolve_layout(
            self._src_mesh, src_meta.placements
        )
        dst_mesh, resolved_dst_placements = resolve_layout(
            self._dst_mesh, dst_placements
        )
        m2n.reshard(
            None,
            dst_buffer,
            comm,
            stream,
            src_mesh=to_mesh(m2n, src_mesh),
            src_placements=to_placements(m2n, resolved_src_placements),
            dst_mesh=to_mesh(m2n, dst_mesh),
            dst_placements=to_placements(m2n, resolved_dst_placements),
            handle=self._handle,
        )

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        if self._handle is not None:
            # M2N does not synchronize caller streams on finalize.
            torch.accelerator.synchronize()
            self._handle.destroy()
            self._handle = None
        self.model_update_group = None
        self._parameter_destinations = []
