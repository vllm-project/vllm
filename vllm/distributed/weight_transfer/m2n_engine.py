# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-side weight transfer engine built on NCCL M2N (`nccl_m2n`).

The trainer and the inference workers share one NCCL communicator: trainer ranks
occupy `[0, T)`, workers `[T, T + N)`. Each parameter is moved with a single
`nccl.m2n.reshard`, which redistributes it from the trainer's layout (FSDP / EP
/ arbitrary DTensor sharding) to the inference layout — so the trainer sends its
local shards and never all-gathers a full tensor, which is what the broadcast
NCCL backend forces it to do.

The destination is planned per parameter. A plan is either entirely direct
(M2N writes live model storage), or it uses the model's layerwise reload
lifecycle, with bounded EP-local staging for semantic fused-MoE consumers and
full checkpoint-format fallback for everything else.

Both meshes participate in every reshard, in the same order, so this backend has
the same concurrency shape as the broadcast NCCL backend: the worker must be
inside `receive_weights` while the trainer is sending. Driving that from the
trainer is the trainer engine's job.
"""

from collections.abc import Iterator, Mapping
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
    M2N_WIRE_SCHEMA_VERSION,
    MESH_NDIMS,
    M2NDestinationMode,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    Placements,
    check_data_plane_agreement,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    create_m2n_communicator,
    expected_local_shape,
    import_m2n,
    prepare_m2n_local_runtime,
    resolve_layout,
    source_plan_digest,
    to_mesh,
    to_placements,
)
from vllm.distributed.weight_transfer.m2n_layout import (
    M2NDestination,
    resolve_parameter_destinations,
)
from vllm.distributed.weight_transfer.m2n_plan import (
    M2NWireDestination,
    agree_destination_plan,
)
from vllm.distributed.weight_transfer.m2n_staging import (
    M2NStagingPool,
    M2NStagingRequirement,
)
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    decode_nccl_unique_id,
    worker_init_metadata_group,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.sharded_weight import (
    ShardedWeightRequest,
    ShardedWeightTarget,
    resolve_sharded_weight_target,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

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
    """Deployment-local worker-rank offset.

    ``rank_offset`` remains the common start rank of the complete destination
    mesh. Multi-deployment inference sets this field to the start rank of one
    deployment so vLLM's deployment-local DP ranks remain globally unique. A
    single deployment may leave it unset.
    """
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    """Optional UID for the NCCL data plane; TCP remains the metadata plane."""
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
                "`worker_rank_offset` must lie inside the common destination "
                f"rank interval [{self.rank_offset}, {self.world_size}); got "
                f"{self.worker_rank_offset}"
            )
        dst = tuple(self.dst_mesh_dims)
        try:
            dst_mesh = M2NMesh(cast(tuple[int, int], dst), self.rank_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid `dst_mesh_dims` {dst}: {exc}") from exc
        num_workers = self.world_size - self.rank_offset
        if dst_mesh.size != num_workers:
            raise ValueError(
                f"`dst_mesh_dims` {dst} must be {MESH_NDIMS} dims covering the "
                f"{num_workers} inference workers"
            )

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

    def parse_wire_params(self) -> tuple[M2NWireParam, ...]:
        """Parse and authenticate the ordered source plan for preflight."""
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
    """Receiver with immutable per-parameter M2N destinations."""

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
        self.model_update_group: Any = None
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._dst_mesh: M2NMesh | None = None
        self._parameter_destinations: list[M2NDestination] = []
        self._uses_load_weights = False
        self._destination_shard_index: int | None = None
        self._staging_pool: M2NStagingPool | None = None
        self._staged_targets: dict[int, ShardedWeightTarget] = {}
        self._next_parameter_index = 0

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Agree on source and local readiness before constructing NCCL."""
        if self._handle is not None or self.model_update_group is not None:
            raise RuntimeError(
                "nccl_m2n weight transfer is already initialized; call "
                "shutdown_weight_transfer_engine() before initializing again"
            )
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
        plan_digest = check_source_plan_agreement(
            metadata_group,
            wire_params,
            expected_digest=init_info.source_digest,
            local_error=source_error,
        )

        num_workers = init_info.world_size - init_info.rank_offset
        agreement_mesh = M2NMesh(
            (num_workers, 1),
            init_info.rank_offset,
        )

        local_wire_plan: list[M2NWireDestination] | None = None
        destination_error: str | None = None
        try:
            self._dst_mesh = M2NMesh(
                cast(tuple[int, int], tuple(init_info.dst_mesh_dims)),
                init_info.rank_offset,
            )
            self._prepare_destination_plan(
                metadata_rank=metadata_group.rank,
                num_workers=num_workers,
            )
            agreement_mesh = self._dst_mesh
            assert self._destination_shard_index is not None
            shard_axis_size = self._dst_mesh.dims[DESTINATION_SHARD_AXIS]
            local_wire_plan = [
                M2NWireDestination.from_destination(
                    destination,
                    dtype_name=wire_param.dtype_name,
                    destination_shard_index=self._destination_shard_index,
                    shard_axis_size=shard_axis_size,
                )
                for wire_param, destination in zip(
                    wire_params, self._parameter_destinations
                )
            ]
        except Exception as exc:
            destination_error = f"{type(exc).__name__}: {exc}"
            self._clear_destination_plan()

        try:
            agree_destination_plan(
                metadata_group,
                source_digest=plan_digest,
                params=wire_params,
                metas=self._metas,
                dst_mesh=agreement_mesh,
                local_plan=local_wire_plan,
                local_error=destination_error,
            )
            unique_id_bytes: bytes | None = None
            data_plane_error: str | None = None
            try:
                unique_id_bytes = init_info.nccl_unique_id_bytes
            except Exception as exc:
                data_plane_error = f"{type(exc).__name__}: {exc}"
            check_data_plane_agreement(
                metadata_group,
                unique_id_bytes,
                data_plane_error,
            )
        except BaseException:
            self._clear_destination_plan()
            raise
        handle = None
        group = None
        try:
            runtime_error = destination_error
            nccl = None
            device = None
            if runtime_error is None:
                try:
                    nccl, device, handle = prepare_m2n_local_runtime(
                        self._m2n, init_info.max_cta
                    )
                except Exception as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
            check_runtime_ready_agreement(metadata_group, runtime_error)
            assert nccl is not None and device is not None and handle is not None
            group = create_m2n_communicator(nccl, metadata_group, unique_id_bytes)
        except BaseException:
            if group is not None:
                try:
                    group.abort()
                except BaseException:
                    logger.exception(
                        "failed to abort partially initialized nccl_m2n communicator"
                    )
            if handle is not None:
                try:
                    handle.destroy()
                except BaseException:
                    logger.exception(
                        "failed to destroy partially initialized nccl_m2n handle"
                    )
            self._clear_destination_plan()
            raise

        self._handle = handle
        self.model_update_group = group

    def _clear_destination_plan(self) -> None:
        self._clear_update_bindings()
        self._dst_mesh = None
        self._parameter_destinations = []
        self._uses_load_weights = False
        self._destination_shard_index = None
        self._staging_pool = None

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
            meta = M2NParamMeta(
                param.name,
                dtype,
                param.shape,
                layout,
                param.allow_full_fallback,
            )
            expected_local_shape(meta)
            self._metas.append(meta)

    def _prepare_destination_plan(
        self,
        *,
        metadata_rank: int,
        num_workers: int,
    ) -> None:
        """Resolve and validate this worker's destination plan."""
        if self._dst_mesh is None:
            raise RuntimeError("destination planning requires a destination mesh")
        if self._dst_mesh.size != num_workers:
            raise ValueError(
                f"`dst_mesh_dims` {self._dst_mesh.dims} must cover the "
                f"{num_workers} inference workers"
            )
        local_worker_rank = metadata_rank - self._dst_mesh.start_rank
        if not 0 <= local_worker_rank < num_workers:
            raise ValueError(
                f"metadata rank {metadata_rank} is outside destination ranks "
                f"[{self._dst_mesh.start_rank}, "
                f"{self._dst_mesh.start_rank + num_workers})"
            )
        shard_axis_size = self._dst_mesh.dims[DESTINATION_SHARD_AXIS]
        self._destination_shard_index = local_worker_rank % shard_axis_size
        # A quantized weight loader may dequantize / transpose / requantize, and
        # under PP a rank's local shape is not the checkpoint shape split over
        # the shard axis. Both disable raw and semantic sharded destinations, so
        # take the full-tensor fallback.
        allow_direct = (
            getattr(self.model_config, "quantization", None) is None
            and self.parallel_config.pipeline_parallel_size == 1
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
            destination_shard_index=self._destination_shard_index,
            allow_full_fallback=[meta.allow_full_fallback for meta in self._metas],
        )
        self._uses_load_weights = any(
            destination.mode is not M2NDestinationMode.IN_PLACE
            for destination in self._parameter_destinations
        )
        requirements = [
            M2NStagingRequirement(
                group=cast(int, destination.staging_group),
                slot=cast(int, destination.staging_slot),
                dtype=meta.dtype,
                shape=destination.local_shape,
            )
            for meta, destination in zip(self._metas, self._parameter_destinations)
            if destination.mode is M2NDestinationMode.SHARDED_STAGING
        ]
        self._staging_pool = (
            M2NStagingPool(requirements, self.device) if requirements else None
        )

    def start_weight_update(self) -> None:
        """Enter one update and bind semantic callbacks to this generation."""
        try:
            if self._staging_pool is not None:
                self._staging_pool.start_update()
            if self._uses_load_weights:
                from vllm.model_executor.model_loader.reload import (
                    initialize_layerwise_reload,
                )

                initialize_layerwise_reload(self.model)
            self._bind_staged_targets()
        except BaseException:
            self._discard_update()
            raise

    def finish_weight_update(self) -> None:
        """Require a complete update, finalize consumers, then release staging."""
        try:
            if self._next_parameter_index != len(self._metas):
                missing = self._metas[self._next_parameter_index].name
                raise RuntimeError(
                    "cannot finish an incomplete nccl_m2n update: received "
                    f"{self._next_parameter_index}/{len(self._metas)} "
                    f"parameters (next expected: {missing!r})"
                )

            if self._uses_load_weights:
                from vllm.model_executor.model_loader.reload import (
                    finalize_layerwise_reload,
                )

                finalize_layerwise_reload(self.model, self.model_config)
                # Complete copies from retained staging inputs before reuse.
                torch.accelerator.synchronize()
            if self._staging_pool is not None:
                self._staging_pool.release_retained_after_finalize()
                self._staging_pool.finish_update()

            self._clear_update_bindings()
        except BaseException:
            self._discard_update()
            raise

    def _bind_staged_targets(self) -> None:
        """Re-resolve update-scoped consumers after layerwise initialization."""
        group_to_key: dict[int, object] = {}
        key_to_group: dict[object, int] = {}
        for index, (meta, destination) in enumerate(
            zip(self._metas, self._parameter_destinations)
        ):
            if destination.mode is not M2NDestinationMode.SHARDED_STAGING:
                continue
            request = ShardedWeightRequest(meta.name, meta.dtype, meta.shape)
            target = resolve_sharded_weight_target(self.model, request)
            if target is None:
                raise RuntimeError(
                    f"staged destination {meta.name!r} disappeared after "
                    "initialize_layerwise_reload"
                )
            if target.spec != destination.sharded_spec:
                raise RuntimeError(
                    f"staged destination {meta.name!r} changed geometry after "
                    "initialize_layerwise_reload"
                )
            group = cast(int, destination.staging_group)
            group_size = cast(int, destination.staging_group_size)
            if target.retention_group_size != group_size:
                raise RuntimeError(
                    f"staged destination {meta.name!r} changed retention group "
                    f"size from {group_size} to {target.retention_group_size}"
                )
            key = target.retention_key
            if group in group_to_key and group_to_key[group] != key:
                raise RuntimeError(
                    f"staged destination group {group} changed retention owner"
                )
            if key in key_to_group and key_to_group[key] != group:
                raise RuntimeError(
                    f"retention owner for {meta.name!r} spans planned groups "
                    f"{key_to_group[key]} and {group}"
                )
            group_to_key[group] = key
            key_to_group[key] = group
            self._staged_targets[index] = target

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Discard update-local staging after parse or receive failures."""
        try:
            super().update_weights(update_info)
        except BaseException:
            self._discard_update()
            raise

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        """Receive one exact contiguous slice of the immutable transfer plan."""
        if self._handle is None or self.model_update_group is None:
            raise RuntimeError(
                "nccl_m2n weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )
        try:
            self._receive_weights(update_info)
        except BaseException:
            self._discard_update()
            raise

    def _receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        assert self.model_update_group is not None
        comm = self.model_update_group
        if not isinstance(update_info.names, list) or any(
            not isinstance(name, str) for name in update_info.names
        ):
            raise TypeError("nccl_m2n update names must be a list of strings")
        start = self._next_parameter_index
        end = start + len(update_info.names)
        expected = tuple(meta.name for meta in self._metas[start:end])
        received = tuple(update_info.names)
        if received != expected:
            raise ValueError(
                "nccl_m2n update order mismatch: expected the contiguous "
                f"slice {expected!r} at offset {start}, got {received!r}"
            )

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        stream = torch.cuda.current_stream()
        processed = 0

        def receive_requested_weights() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal processed
            for index in range(start, end):
                meta = self._metas[index]
                destination = self._parameter_destinations[index]
                if destination.mode is M2NDestinationMode.IN_PLACE:
                    buffer = self._resolve_direct_buffer(meta, destination)
                    self._reshard(comm, stream, meta, destination.placements, buffer)
                    processed += 1
                    continue

                if destination.mode is M2NDestinationMode.FULL_FALLBACK:
                    buffer = torch.empty(
                        meta.shape, dtype=meta.dtype, device=self.device
                    )
                    self._reshard(comm, stream, meta, destination.placements, buffer)
                    stream.synchronize()
                    processed += 1
                    yield meta.name, buffer
                    continue

                assert self._staging_pool is not None
                target = self._staged_targets[index]
                group = cast(int, destination.staging_group)
                slot = cast(int, destination.staging_slot)
                group_size = cast(int, destination.staging_group_size)
                buffer = self._staging_pool.acquire(
                    group=group,
                    slot=slot,
                    dtype=meta.dtype,
                    shape=destination.local_shape,
                )
                self._reshard(comm, stream, meta, destination.placements, buffer)
                stream.synchronize()
                if target.load(buffer):
                    if slot != group_size - 1:
                        raise RuntimeError(
                            f"staged consumer for {meta.name!r} released group "
                            f"{group} at slot {slot}, before planned final slot "
                            f"{group_size - 1}"
                        )
                    self._staging_pool.release_group(group)
                processed += 1

        has_fallback = any(
            self._parameter_destinations[index].mode is M2NDestinationMode.FULL_FALLBACK
            for index in range(start, end)
        )
        with disable_mtp_completeness_check():
            received_weights = receive_requested_weights()
            if has_fallback:
                loaded = self.model.load_weights(received_weights)
                if loaded is not None:
                    for _ in loaded:
                        pass
            else:
                for _ in received_weights:
                    pass
        if processed != end - start:
            raise RuntimeError(
                "model.load_weights did not consume the complete nccl_m2n "
                f"fallback sequence: processed {processed}/{end - start}"
            )
        self._next_parameter_index = end

    def _resolve_direct_buffer(
        self, meta: M2NParamMeta, destination: M2NDestination
    ) -> torch.Tensor:
        """Validate the live destination captured before layerwise reload."""
        parameter_name = destination.name
        buffer = destination.tensor
        if buffer is None:
            raise RuntimeError(f"in-place destination {parameter_name!r} is missing")
        if buffer.dtype != meta.dtype:
            raise RuntimeError(
                f"in-place destination {parameter_name!r} changed dtype from "
                f"{meta.dtype} to {buffer.dtype}"
            )
        if tuple(buffer.shape) != destination.local_shape:
            raise RuntimeError(
                f"in-place destination {parameter_name!r} changed shape from "
                f"{destination.local_shape} to {tuple(buffer.shape)}"
            )
        if not buffer.is_contiguous():
            raise RuntimeError(
                f"in-place destination {parameter_name!r} is not contiguous"
            )
        if buffer.device != self.device:
            raise RuntimeError(
                f"in-place destination {parameter_name!r} is on "
                f"{buffer.device}, expected {self.device}"
            )
        return buffer

    def _reshard(
        self,
        comm: Any,
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

    def _clear_update_bindings(self) -> None:
        self._staged_targets.clear()
        self._next_parameter_index = 0

    def _discard_update(self) -> None:
        """Release only update-local staging after a failed update."""
        self._clear_update_bindings()
        if self._staging_pool is not None:
            self._staging_pool.discard_update()

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        handle, self._handle = self._handle, None
        group, self.model_update_group = self.model_update_group, None
        self._m2n = None
        self._metas = []
        self._clear_destination_plan()

        first_error: BaseException | None = None

        def attempt(action: Any, message: str) -> None:
            nonlocal first_error
            try:
                action()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(message)

        if handle is not None or group is not None:
            # M2N does not synchronize caller streams on finalize.
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        if group is not None:
            attempt(group.abort, "failed to abort nccl_m2n communicator")
        if first_error is not None:
            raise first_error
