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

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
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
    M2N_WIRE_SCHEMA_VERSION,
    MESH_NDIMS,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    Placements,
    check_data_plane_agreement,
    check_plan_limits,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    comm_ptr,
    import_m2n,
    prepare_m2n_local_runtime,
    publish_destination_placements,
    resolve_layout,
    source_plan_digest,
    to_mesh,
    to_placements,
    validate_layout,
    validate_m2n_nccl_communicator,
)
from vllm.distributed.weight_transfer.m2n_layout import (
    M2NDestination,
    resolve_parameter_destinations,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    decode_nccl_unique_id,
    pynccl_from_metadata_group,
    uid_init_process_group,
    worker_init_metadata_group,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

logger = init_logger(__name__)

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

    schema_version: int
    master_address: str
    master_port: int
    rank_offset: int
    """First worker rank, i.e. the number of trainer ranks."""
    world_size: int
    """Trainer ranks + all inference workers."""
    dst_mesh_dims: list[int]
    """The inference mesh, starting at `rank_offset`. Declared rather than
    derived so both sides describe the destination identically."""
    source_digest: str
    """SHA-256 of the ordered, versioned source parameter plan."""
    params: list[dict[str, Any]]
    """Ordered per-parameter source metadata and layouts."""
    worker_rank_offset: int | None = None
    """First global rank assigned to this inference deployment.

    `rank_offset` remains the start of the complete destination mesh. A
    multi-deployment client sets this field per deployment so deployment-local
    DP ranks remain globally unique. Single deployments may leave it unset.
    """
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    max_cta: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.master_address, str) or not self.master_address:
            raise ValueError("`master_address` must name the metadata rendezvous")
        if (
            not isinstance(self.master_port, int)
            or isinstance(self.master_port, bool)
            or not 0 < self.master_port < 65536
        ):
            raise ValueError("`master_port` must be a valid metadata port")
        if (
            not isinstance(self.rank_offset, int)
            or isinstance(self.rank_offset, bool)
            or not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.rank_offset < 1
            or self.rank_offset >= self.world_size
        ):
            raise ValueError(
                f"`rank_offset` ({self.rank_offset}) must leave at least one "
                f"trainer rank and one worker in world_size {self.world_size}"
            )
        if self.worker_rank_offset is not None and (
            not isinstance(self.worker_rank_offset, int)
            or isinstance(self.worker_rank_offset, bool)
            or not self.rank_offset <= self.worker_rank_offset < self.world_size
        ):
            raise ValueError(
                "`worker_rank_offset` must lie inside the destination rank "
                f"interval [{self.rank_offset}, {self.world_size}); got "
                f"{self.worker_rank_offset}"
            )
        num_workers = self.world_size - self.rank_offset
        dst = tuple(self.dst_mesh_dims)
        try:
            dst_mesh = M2NMesh(cast(tuple[int, int], dst), self.rank_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid `dst_mesh_dims` {dst}: {exc}") from exc
        if dst_mesh.size != num_workers:
            raise ValueError(
                f"`dst_mesh_dims` {dst} must be {MESH_NDIMS} dims covering the "
                f"{num_workers} inference workers"
            )

    def parse_wire_params(self) -> tuple[M2NWireParam, ...]:
        """Parse and authenticate the ordered source plan."""
        if self.schema_version != M2N_WIRE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported nccl_m2n init schema version "
                f"{self.schema_version}; expected {M2N_WIRE_SCHEMA_VERSION}"
            )
        if not isinstance(self.params, list):
            raise TypeError("`params` must be a list of wire parameter mappings")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise ValueError("`source_digest` must be a non-empty string")
        wire_params: list[M2NWireParam] = []
        names: set[str] = set()
        for index, value in enumerate(self.params):
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"`params[{index}]` must be a mapping, got {type(value).__name__}"
                )
            param = M2NWireParam.from_dict(value)
            if param.name in names:
                raise ValueError(
                    f"nccl_m2n init params contain duplicate parameter '{param.name}'"
                )
            names.add(param.name)
            wire_params.append(param)
        parsed = tuple(wire_params)
        computed_digest = source_plan_digest(parsed)
        if self.source_digest != computed_digest:
            raise ValueError(
                "nccl_m2n init source digest does not match params: "
                f"computed {computed_digest}, declared {self.source_digest}"
            )
        return parsed

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        if self.nccl_unique_id_b64 is None:
            return None
        return decode_nccl_unique_id(
            master_address=None,
            master_port=None,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="M2NWeightTransferInitInfo data plane",
        )


@dataclass
class M2NWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info: which parameters this chunk carries, in order.

    Shapes, dtypes and layouts were fixed at init, so a round only needs to say
    what is coming and in what order — both sides must issue their reshards in
    exactly that order.
    """

    names: list[str]


class _M2NEngineState(str, Enum):
    PREFLIGHT = "preflight"
    READY = "ready"
    UPDATING = "updating"
    POISONED = "poisoned"
    CLOSED = "closed"


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
        self._dst_mesh: M2NMesh | None = None
        self._parameter_destinations: list[M2NDestination] = []
        self._index: dict[str, int] = {}
        self._uses_load_weights = False
        self._state = _M2NEngineState.PREFLIGHT
        self._failure: BaseException | None = None

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        try:
            self._require_state(_M2NEngineState.PREFLIGHT, "initialize nccl_m2n")
            self._init_transfer_engine(init_info)
            self._state = _M2NEngineState.READY
        except BaseException as exc:
            self._poison(exc)
            raise

    def _init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Agree on source and local readiness before constructing NCCL."""
        rendezvous_info = NCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=(
                init_info.rank_offset
                if init_info.worker_rank_offset is None
                else init_info.worker_rank_offset
            ),
            world_size=init_info.world_size,
        )
        metadata_group = worker_init_metadata_group(
            rendezvous_info, self.parallel_config
        )

        wire_params: tuple[M2NWireParam, ...] = ()
        source_error: str | None = None
        try:
            self._m2n = import_m2n()
            wire_params = init_info.parse_wire_params()
            self._prepare_source_plan(wire_params, init_info.rank_offset)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
            self._metas = []
        check_source_plan_agreement(
            metadata_group,
            wire_params,
            expected_digest=init_info.source_digest,
            local_error=source_error,
        )

        destination_error: str | None = None
        try:
            self._prepare_destination_plan(init_info)
        except Exception as exc:
            destination_error = f"{type(exc).__name__}: {exc}"
            self._parameter_destinations = []

        unique_id_bytes: bytes | None = None
        data_plane_error: str | None = None
        try:
            unique_id_bytes = init_info.nccl_unique_id_bytes
        except Exception as exc:
            data_plane_error = f"{type(exc).__name__}: {exc}"
        check_data_plane_agreement(
            metadata_group,
            unique_id_bytes,
            init_info.max_cta,
            data_plane_error,
        )

        runtime_error = destination_error
        nccl_runtime = None
        device = None
        if runtime_error is None:
            try:
                nccl_runtime, device, self._handle = prepare_m2n_local_runtime(
                    self._m2n, init_info.max_cta
                )
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
        check_runtime_ready_agreement(metadata_group, runtime_error)
        assert nccl_runtime is not None and device is not None
        if unique_id_bytes is None:
            self.model_update_group = pynccl_from_metadata_group(
                metadata_group,
                device,
                library_path=nccl_runtime.library_path,
            )
        else:
            self.model_update_group = uid_init_process_group(
                unique_id_bytes,
                rank=metadata_group.rank,
                world_size=metadata_group.world_size,
                device=device,
                library_path=nccl_runtime.library_path,
            )
        validate_m2n_nccl_communicator(self.model_update_group, nccl_runtime)

        # Keep the merged destination-placement publication unchanged: source
        # agreement is independent of worker-side destination selection.
        mine = [destination.placements for destination in self._parameter_destinations]
        agreed = publish_destination_placements(
            self.model_update_group, init_info.rank_offset, mine, len(self._metas)
        )
        if agreed != mine:
            mismatched = next(
                meta.name
                for meta, ours, theirs in zip(self._metas, mine, agreed)
                if ours != theirs
            )
            raise RuntimeError(
                "inference workers resolved different nccl_m2n destinations "
                f"(first mismatch: '{mismatched}'); all workers must run the "
                "same model and parallel config"
            )

    def _prepare_source_plan(
        self,
        wire_params: tuple[M2NWireParam, ...],
        num_trainer_ranks: int,
    ) -> None:
        """Validate worker-visible source metadata without constructing NCCL."""
        self._metas = []
        for param in wire_params:
            dtype = getattr(torch, param.dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"parameter '{param.name}' has invalid dtype name "
                    f"'{param.dtype_name}'; expected the name of a torch.dtype"
                )
            check_transferable(param.name, dtype, param.shape)
            layout = M2NLayout(M2NMesh(param.src_mesh_dims, 0), param.src_placements)
            if layout.mesh.size != num_trainer_ranks:
                raise ValueError(
                    f"parameter '{param.name}' source mesh covers "
                    f"{layout.mesh.size} ranks, but there are "
                    f"{num_trainer_ranks} trainer ranks"
                )
            src_mesh, src_placements = resolve_layout(
                layout.mesh,
                layout.placements,
                f"parameter '{param.name}' source placements",
            )
            validate_layout(src_mesh, src_placements, param.shape, "source")
            self._metas.append(M2NParamMeta(param.name, dtype, param.shape, layout))

    def _prepare_destination_plan(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Resolve the merged-#51520 in-place/full-fallback destination plan."""
        # The inference topology, as declared by the trainer. This is the mesh
        # a sharded destination is placed over; a replicated one is described
        # by the alternate descriptor `resolve_layout` derives from it, which
        # covers the same ranks but is not this mesh.
        self._dst_mesh = M2NMesh(
            cast(tuple[int, int], tuple(init_info.dst_mesh_dims)),
            init_info.rank_offset,
        )
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
                meta.source_layout.mesh,
                meta.source_layout.placements,
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

    def start_weight_update(self) -> None:
        """Set up layerwise reloading, but only if some parameter needs it.

        Directly-resharded parameters are written in place and never go through
        `load_weights`, so a plan with no fallback entries has nothing to
        reload.
        """
        try:
            self._require_state(_M2NEngineState.READY, "start a weight update")
            self._state = _M2NEngineState.UPDATING
            if self._uses_load_weights:
                from vllm.model_executor.model_loader.reload import (
                    initialize_layerwise_reload,
                )

                initialize_layerwise_reload(self.model)
        except BaseException as exc:
            self._poison(exc)
            raise

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading when the plan uses fallback entries."""
        try:
            self._require_state(_M2NEngineState.UPDATING, "finish a weight update")
            if self._uses_load_weights:
                from vllm.model_executor.model_loader.reload import (
                    finalize_layerwise_reload,
                )

                finalize_layerwise_reload(self.model, self.model_config)
            self._state = _M2NEngineState.READY
        except BaseException as exc:
            self._poison(exc)
            raise

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Poison the engine if parsing, transfer, or synchronization fails."""
        try:
            super().update_weights(update_info)
        except BaseException as exc:
            self._poison(exc)
            raise

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        """Receive each requested parameter using its initialization-time plan."""
        try:
            self._require_state(_M2NEngineState.UPDATING, "receive weights")
            self._receive_weights(update_info)
        except BaseException as exc:
            self._poison(exc)
            raise

    def _receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        assert self._handle is not None and self.model_update_group is not None
        if not isinstance(update_info.names, list) or any(
            not isinstance(name, str) for name in update_info.names
        ):
            raise TypeError("nccl_m2n update names must be a list of strings")

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
        assert self._dst_mesh is not None
        m2n = self._m2n
        src_mesh, resolved_src_placements = resolve_layout(
            src_meta.source_layout.mesh, src_meta.source_layout.placements
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

    def _require_state(self, expected: _M2NEngineState, action: str) -> None:
        if self._state is _M2NEngineState.POISONED:
            raise RuntimeError(
                "nccl_m2n engine failed earlier and cannot be reused"
            ) from self._failure
        if self._state is _M2NEngineState.CLOSED:
            raise RuntimeError("nccl_m2n engine is shut down")
        if self._state is not expected:
            raise RuntimeError(
                f"cannot {action} while nccl_m2n engine is "
                f"{self._state.value}; expected {expected.value}"
            )

    def _poison(self, failure: BaseException) -> None:
        """Mark unusable before aborting NCCL; never synchronize this path."""
        if self._state is _M2NEngineState.CLOSED:
            return
        if self._failure is None:
            self._failure = failure
        self._state = _M2NEngineState.POISONED

        group = self.model_update_group
        self.model_update_group = None
        if group is not None:
            try:
                group.destroy()
            except BaseException:
                logger.exception("failed to abort poisoned nccl_m2n communicator")

        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                handle.destroy()
            except BaseException:
                logger.exception("failed to destroy poisoned nccl_m2n handle")

    def shutdown(self) -> None:
        """Idempotently release resources, without syncing a poisoned engine."""
        if self._state is _M2NEngineState.CLOSED:
            return
        try:
            if self._state is _M2NEngineState.UPDATING:
                self._poison(RuntimeError("nccl_m2n shut down during an active update"))
            elif self._state is not _M2NEngineState.POISONED:
                if self._state is _M2NEngineState.READY:
                    torch.accelerator.synchronize()
                handle = self._handle
                self._handle = None
                if handle is not None:
                    handle.destroy()
                group = self.model_update_group
                self.model_update_group = None
                if group is not None:
                    group.destroy()
        except BaseException as exc:
            self._poison(exc)
            raise
        finally:
            self._state = _M2NEngineState.CLOSED
            self._drop_plan()

    def _drop_plan(self) -> None:
        self._m2n = None
        self._metas = []
        self._dst_mesh = None
        self._parameter_destinations = []
        self._index = {}
        self._uses_load_weights = False
