# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-side weight transfer engine built on NCCL M2N (`nccl_m2n`).

Each ordered trainer source group shares one NCCL communicator with the
inference workers. A parameter is routed to exactly one of those routes and
moved with one `nccl.m2n.reshard`, which redistributes it from the trainer's
layout (FSDP / TP / EP / arbitrary DTensor sharding) to the inference layout.
The trainer sends local shards and never all-gathers a full tensor, which is
what the broadcast NCCL backend forces it to do.

The destination is planned per parameter. A plan is either entirely direct
(M2N writes live model storage), or it uses the model's layerwise reload
lifecycle, with bounded EP-local staging for semantic fused-MoE consumers and
full checkpoint-format fallback for everything else.

Members of each route participate in its reshards in the same authenticated
order. A bounded wave schedule keeps launch order identical across routes
while separate CUDA streams permit independent routes to overlap. The
worker must be inside `receive_weights` while the trainer is sending; driving
that from the trainer is the trainer engine's job.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    RetryableWeightUpdateError,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    M2NDestinationMode,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NSourceTopology,
    M2NTransferRoute,
    M2NWireParam,
    build_m2n_param_routes,
    build_m2n_transfer_routes,
    build_m2n_transfer_waves,
    check_communicator_init_agreement,
    check_data_plane_agreement,
    check_multi_communicator_launch_order,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    create_m2n_communicator,
    expected_local_shape,
    import_m2n,
    m2n_load_group,
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
    dst_mesh_dims: list[int] | None = None
    """Optional fixed inference mesh; ``None`` selects per-parameter meshes."""
    source_digest: str
    """SHA-256 of the ordered, versioned source parameter plan."""
    params: list[dict[str, Any]]
    """Ordered per-parameter source metadata and layouts."""
    source_topology: dict[str, Any] | None = None
    """Ordered source rank groups and one owning group per parameter.

    Omitted payloads retain the legacy single-source-group topology: all
    trainer ranks own the complete ordered manifest.
    """
    worker_rank_offset: int | None = None
    """Deployment-local worker-rank offset.

    ``rank_offset`` remains the common start rank of the complete destination
    mesh. Multi-deployment inference sets this field to the start rank of one
    deployment so vLLM's deployment-local DP ranks remain globally unique. A
    single deployment may leave it unset.
    """
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    """Optional single-communicator UID retained for compatibility."""
    nccl_unique_ids_b64: list[str] | None = field(default=None, repr=False)
    """Optional UIDs in canonical transfer-route order."""
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
        if self.dst_mesh_dims is not None:
            dst = tuple(self.dst_mesh_dims)
            try:
                dst_mesh = M2NMesh(cast(tuple[int, int], dst), self.rank_offset)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid `dst_mesh_dims` {dst}: {exc}") from exc
            num_workers = self.world_size - self.rank_offset
            if dst_mesh.size != num_workers:
                raise ValueError(
                    f"`dst_mesh_dims` {dst} must cover the {num_workers} "
                    "inference workers"
                )

    @property
    def destination_mesh_dims(self) -> tuple[int, int]:
        """Concrete mesh used to describe the destination rank interval."""
        if self.dst_mesh_dims is not None:
            return cast(tuple[int, int], tuple(self.dst_mesh_dims))
        return (self.world_size - self.rank_offset, 1)

    @property
    def destination_mesh_constraint(self) -> tuple[int, int] | None:
        """Fixed mesh constraint, or ``None`` for per-parameter planning."""
        if self.dst_mesh_dims is None:
            return None
        return cast(tuple[int, int], tuple(self.dst_mesh_dims))

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

    def nccl_unique_id_bytes_by_route(
        self, num_routes: int
    ) -> tuple[bytes | None, ...]:
        """Decode the optional pre-shared UID table for canonical routes."""
        if (
            not isinstance(num_routes, int)
            or isinstance(num_routes, bool)
            or num_routes <= 0
        ):
            raise ValueError("num_routes must be a positive integer")
        if self.nccl_unique_id_b64 is not None and self.nccl_unique_ids_b64 is not None:
            raise ValueError(
                "set only one of `nccl_unique_id_b64` and `nccl_unique_ids_b64`"
            )
        if self.nccl_unique_ids_b64 is not None:
            if (
                not isinstance(self.nccl_unique_ids_b64, list)
                or len(self.nccl_unique_ids_b64) != num_routes
            ):
                raise ValueError(
                    "`nccl_unique_ids_b64` must contain exactly one UID per "
                    f"transfer route ({num_routes})"
                )
            return tuple(
                decode_nccl_unique_id(
                    master_address=None,
                    master_port=None,
                    nccl_unique_id_b64=value,
                    ctx=f"M2NWeightTransferInitInfo route {route_index} data plane",
                )
                for route_index, value in enumerate(self.nccl_unique_ids_b64)
            )
        if self.nccl_unique_id_b64 is not None:
            if num_routes != 1:
                raise ValueError(
                    "`nccl_unique_id_b64` is valid only for a single transfer route; "
                    "use `nccl_unique_ids_b64` for pipeline groups"
                )
            return (self.nccl_unique_id_bytes,)
        return (None,) * num_routes

    def parse_source_plan(
        self,
    ) -> tuple[tuple[M2NWireParam, ...], M2NSourceTopology]:
        """Parse and authenticate parameters together with source routing."""
        params = self._parse_wire_params_unchecked()
        if self.source_topology is None:
            topology = M2NSourceTopology.single_group(self.rank_offset, len(params))
        else:
            if not isinstance(self.source_topology, Mapping):
                raise TypeError("`source_topology` must be a mapping")
            topology = M2NSourceTopology.from_dict(self.source_topology)
        topology.validate(
            num_trainer_ranks=self.rank_offset,
            num_params=len(params),
        )
        computed_digest = source_plan_digest(params, topology)
        if self.source_digest != computed_digest:
            raise ValueError(
                "nccl_m2n init source digest does not match params/topology: "
                f"computed {computed_digest}, declared {self.source_digest}"
            )
        return params, topology

    def parse_wire_params(self) -> tuple[M2NWireParam, ...]:
        """Compatibility view of :meth:`parse_source_plan`."""
        return self.parse_source_plan()[0]

    def _parse_wire_params_unchecked(self) -> tuple[M2NWireParam, ...]:
        """Parse the ordered parameter records before topology authentication."""
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
        self._model_update_groups: dict[int, Any] = {}
        self._route_streams: dict[int, torch.cuda.Stream] = {}
        self._routes: tuple[M2NTransferRoute, ...] = ()
        self._param_routes: tuple[int, ...] = ()
        self._source_topology = M2NSourceTopology((), ())
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._parameter_destinations: list[M2NDestination] = []
        self._uses_load_weights = False
        self._staging_pool: M2NStagingPool | None = None
        self._staged_targets: dict[int, ShardedWeightTarget] = {}
        self._next_parameter_index = 0

    def init_transfer_engine(self, init_info: M2NWeightTransferInitInfo) -> None:
        """Agree on source and local readiness before constructing NCCL."""
        if self._handle is not None or self._model_update_groups:
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
        source_topology = M2NSourceTopology((), ())
        source_error: str | None = None
        try:
            self._m2n = import_m2n()
            wire_params, source_topology = init_info.parse_source_plan()
            self._source_topology = source_topology
            self._prepare_source_plan(wire_params, source_topology)
            self._routes = build_m2n_transfer_routes(
                source_topology,
                num_trainer_ranks=init_info.rank_offset,
                world_size=init_info.world_size,
                dst_mesh_dims=init_info.destination_mesh_dims,
            )
            self._param_routes = build_m2n_param_routes(self._routes, len(wire_params))
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
            self._metas = []
            self._routes = ()
            self._param_routes = ()
        plan_digest = check_source_plan_agreement(
            metadata_group,
            wire_params,
            expected_digest=init_info.source_digest,
            local_error=source_error,
            topology=source_topology,
        )

        num_workers = init_info.world_size - init_info.rank_offset
        agreement_mesh = M2NMesh(init_info.destination_mesh_dims, init_info.rank_offset)

        local_wire_plan: list[M2NWireDestination] | None = None
        destination_error: str | None = None
        try:
            local_worker_rank = metadata_group.rank - init_info.rank_offset
            if not 0 <= local_worker_rank < num_workers:
                raise ValueError(
                    f"metadata rank {metadata_group.rank} is outside destination "
                    f"ranks [{init_info.rank_offset}, {init_info.world_size})"
                )
            self._prepare_destination_plan(
                local_worker_rank=local_worker_rank,
                num_workers=num_workers,
                dst_mesh_dims=init_info.destination_mesh_constraint,
            )
            local_wire_plan = [
                M2NWireDestination.from_destination(
                    destination,
                    dtype_name=wire_param.dtype_name,
                    local_worker_rank=local_worker_rank,
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
                source_topology=source_topology,
                metas=self._metas,
                dst_mesh=agreement_mesh,
                dst_mesh_constraint=init_info.destination_mesh_constraint,
                local_plan=local_wire_plan,
                local_error=destination_error,
            )
            unique_id_bytes: tuple[bytes | None, ...] = ()
            data_plane_error: str | None = None
            try:
                unique_id_bytes = init_info.nccl_unique_id_bytes_by_route(
                    len(self._routes)
                )
                check_multi_communicator_launch_order(len(self._routes))
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
        groups: dict[int, Any] = {}
        route_streams: dict[int, torch.cuda.Stream] = {}
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
            communicator_errors: list[str | None] = []
            for route_index, (route, route_uid) in enumerate(
                zip(self._routes, unique_id_bytes, strict=True)
            ):
                route_error: str | None = None
                try:
                    group = create_m2n_communicator(
                        nccl,
                        metadata_group,
                        route_uid,
                        members=route.members,
                    )
                    if group is None:
                        raise RuntimeError(
                            "inference worker is missing from transfer route "
                            f"{route_index}"
                        )
                    groups[route_index] = group
                    route_streams[route_index] = torch.cuda.Stream(device=device)
                except Exception as exc:
                    route_error = f"{type(exc).__name__}: {exc}"
                communicator_errors.append(route_error)
            check_communicator_init_agreement(
                metadata_group,
                communicator_errors,
            )
        except BaseException:
            for route_group in groups.values():
                try:
                    route_group.abort()
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
            self._routes = ()
            self._param_routes = ()
            self._source_topology = M2NSourceTopology((), ())
            raise

        self._handle = handle
        self._model_update_groups = groups
        self._route_streams = route_streams
        # Retain the scalar alias for single-communicator callers and diagnostics.
        self.model_update_group = groups.get(0)

    def _clear_destination_plan(self) -> None:
        self._clear_update_bindings()
        self._parameter_destinations = []
        self._uses_load_weights = False
        self._staging_pool = None

    def _prepare_source_plan(
        self,
        wire_params: tuple[M2NWireParam, ...],
        source_topology: M2NSourceTopology,
    ) -> None:
        """Validate worker-visible source metadata without constructing NCCL."""
        self._metas = []
        for param_index, param in enumerate(wire_params):
            dtype = getattr(torch, param.dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"parameter '{param.name}' has invalid dtype name "
                    f"'{param.dtype_name}'; expected the name of a torch.dtype"
                )
            check_transferable(param.name, dtype, param.shape)
            layout = M2NLayout(M2NMesh(param.src_mesh_dims, 0), param.src_placements)
            group_index = source_topology.param_groups[param_index]
            source_group = source_topology.groups[group_index]
            if layout.mesh.size != len(source_group):
                raise ValueError(
                    f"parameter '{param.name}' source mesh covers "
                    f"{layout.mesh.size} ranks, but source group {group_index} "
                    f"contains {len(source_group)} ranks"
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
        local_worker_rank: int,
        num_workers: int,
        dst_mesh_dims: tuple[int, int] | None,
    ) -> None:
        """Resolve and validate this worker's destination plan."""
        if not 0 <= local_worker_rank < num_workers:
            raise ValueError(
                f"local worker rank {local_worker_rank} must be in [0, {num_workers})"
            )
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
            num_workers=num_workers,
            local_worker_rank=local_worker_rank,
            dst_mesh_dims=dst_mesh_dims,
            allow_direct=allow_direct,
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
        self._require_initialized()
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
            try:
                self._reset_failed_update()
            except BaseException:
                logger.exception("failed to reset nccl_m2n update state")
            raise

    def finish_weight_update(self) -> None:
        """Finalize one update, or reset it so the trainer can replay it."""
        self._require_initialized()
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
        except BaseException as exc:
            try:
                self._reset_failed_update()
            except BaseException as reset_exc:
                raise RuntimeError(
                    "nccl_m2n failed to reset vLLM after a finish failure"
                ) from reset_exc
            raise RetryableWeightUpdateError(
                "vLLM failed to finalize the weight update; worker state was "
                "reset so the complete update can be retried"
            ) from exc

    def _reset_failed_update(self) -> None:
        """Reset loader and staging state without touching M2N transport."""
        if self._uses_load_weights:
            from vllm.model_executor.model_loader.reload import (
                abort_layerwise_reload,
            )

            # Loader callbacks can enqueue reads from retained staging buffers.
            # Finish them before restoring tensors and releasing those buffers.
            torch.accelerator.synchronize()
            abort_layerwise_reload(self.model)
        self._discard_update()

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
        """Preserve pre-transfer failures; abort failures after transfer starts."""
        self._require_initialized()
        try:
            typed_update_info = self.parse_update_info(update_info)
            self.receive_weights(typed_update_info)
        except BaseException:
            if self._handle is not None:
                self._reset_failed_update()
            raise
        try:
            torch.accelerator.synchronize()
        except BaseException:
            self._abort_data_plane()
            raise

    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        """Receive one exact contiguous slice of the immutable transfer plan."""
        self._require_initialized()
        self._receive_weights(update_info)

    def _receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
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

        processed = 0
        transfer_started = False
        pending_buffers: dict[int, torch.Tensor] = {}
        waves = build_m2n_transfer_waves(
            self._routes,
            [
                m2n_load_group(destination.mode, destination.staging_group)
                for destination in self._parameter_destinations
            ],
            active_indices=range(start, end),
        )

        def receive_requested_weights() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal processed, transfer_started
            for wave in waves:
                # One host dispatcher replays the agreed bounded wave. Distinct
                # streams may overlap, while every participant observes the
                # same cross-communicator launch order.
                for index in wave.issue_indices:
                    route_index = self._param_routes[index]
                    comm = self._model_update_groups[route_index]
                    stream = self._route_streams[route_index]
                    meta = self._metas[index]
                    destination = self._parameter_destinations[index]

                    if destination.mode is M2NDestinationMode.IN_PLACE:
                        buffer = self._resolve_direct_buffer(meta, destination)
                        transfer_started = True
                        self._reshard(
                            comm,
                            stream,
                            meta,
                            destination,
                            buffer,
                        )
                        processed += 1
                        continue

                    if destination.mode is M2NDestinationMode.FULL_FALLBACK:
                        buffer = torch.empty(
                            meta.shape, dtype=meta.dtype, device=self.device
                        )
                    else:
                        assert self._staging_pool is not None
                        buffer = self._staging_pool.acquire(
                            group=cast(int, destination.staging_group),
                            slot=cast(int, destination.staging_slot),
                            dtype=meta.dtype,
                            shape=destination.local_shape,
                        )
                    # Retain ownership before launching. If reshard raises,
                    # abort must complete before this buffer can be released.
                    pending_buffers[index] = buffer
                    transfer_started = True
                    self._reshard(
                        comm,
                        stream,
                        meta,
                        destination,
                        buffer,
                    )

                if wave.consume_index is None:
                    continue
                index = wave.consume_index
                meta = self._metas[index]
                destination = self._parameter_destinations[index]
                route_index = self._param_routes[index]
                stream = self._route_streams[route_index]
                stream.synchronize()
                try:
                    buffer = pending_buffers.pop(index)
                except KeyError as exc:
                    raise RuntimeError(
                        "nccl_m2n transfer schedule consumed parameter "
                        f"{meta.name!r} before issuing it"
                    ) from exc
                if destination.mode is M2NDestinationMode.FULL_FALLBACK:
                    processed += 1
                    yield meta.name, buffer
                    continue

                assert self._staging_pool is not None
                target = self._staged_targets[index]
                group = cast(int, destination.staging_group)
                slot = cast(int, destination.staging_slot)
                group_size = cast(int, destination.staging_group_size)
                if target.load(buffer):
                    if slot != group_size - 1:
                        raise RuntimeError(
                            f"staged consumer for {meta.name!r} released group "
                            f"{group} at slot {slot}, before planned final slot "
                            f"{group_size - 1}"
                        )
                    # Model loaders may enqueue reads of the staging inputs.
                    # Order every possible reuse stream after those reads.
                    consumer_stream = torch.cuda.current_stream(self.device)
                    for transfer_stream in self._route_streams.values():
                        transfer_stream.wait_stream(consumer_stream)
                    self._staging_pool.release_group(group)
                processed += 1

            if pending_buffers:
                raise RuntimeError(
                    "nccl_m2n route scheduler ended with pending buffers"
                )
            self._synchronize_route_streams()

        has_fallback = any(
            self._parameter_destinations[index].mode is M2NDestinationMode.FULL_FALLBACK
            for index in range(start, end)
        )
        try:
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
        except BaseException:
            if transfer_started:
                # pending buffers and a suspended fallback iterator remain alive
                # until every local communicator has been aborted.
                self._abort_data_plane()
            raise

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
        destination: M2NDestination,
        dst_buffer: torch.Tensor,
    ) -> None:
        """Reshard one parameter from its trainer layout into a worker buffer.

        ``destination`` carries this parameter's agreed mesh and placement;
        ``dst_buffer`` is this rank's physical storage.
        """
        m2n = self._m2n
        src_mesh, resolved_src_placements = resolve_layout(
            src_meta.source_layout.mesh, src_meta.source_layout.placements
        )
        source_group_size = len(self._source_topology.groups[0])
        dst_mesh, resolved_dst_placements = resolve_layout(
            M2NMesh(destination.mesh_dims, source_group_size),
            destination.placements,
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

    def _synchronize_route_streams(self) -> None:
        """Wait for every joined route without assuming a default stream."""
        for route_index in range(len(self._routes)):
            stream = self._route_streams.get(route_index)
            if stream is not None:
                stream.synchronize()

    def _clear_update_bindings(self) -> None:
        self._staged_targets.clear()
        self._next_parameter_index = 0

    def _discard_update(self) -> None:
        """Release only update-local staging after a failed update."""
        self._clear_update_bindings()
        if self._staging_pool is not None:
            self._staging_pool.discard_update()

    def _require_initialized(self) -> None:
        if self._handle is None or not self._model_update_groups:
            raise RuntimeError(
                "nccl_m2n weight transfer not initialized. "
                "Call init_transfer_engine() first."
            )

    def _abort_data_plane(self) -> None:
        """Abort every communicator without waiting on CUDA streams."""
        groups, self._model_update_groups = self._model_update_groups, {}
        handle, self._handle = self._handle, None
        streams, self._route_streams = self._route_streams, {}
        self.model_update_group = None

        for group_index, group in groups.items():
            try:
                group.abort()
            except BaseException:
                logger.exception(
                    "failed to abort nccl_m2n transfer communicator %d",
                    group_index,
                )
        if handle is not None:
            try:
                handle.destroy()
            except BaseException:
                logger.exception("failed to destroy nccl_m2n handle")
        try:
            self._discard_update()
        except BaseException:
            logger.exception("failed to discard nccl_m2n update state")

        self._routes = ()
        self._param_routes = ()
        self._source_topology = M2NSourceTopology((), ())
        self._m2n = None
        self._metas = []
        self._clear_destination_plan()
        # Keep the stream wrappers alive until communicator abort and handle
        # destruction have completed.
        del streams

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        handle, self._handle = self._handle, None
        groups, self._model_update_groups = self._model_update_groups, {}
        self.model_update_group = None
        streams, self._route_streams = self._route_streams, {}
        self._routes = ()
        self._param_routes = ()
        self._source_topology = M2NSourceTopology((), ())
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

        if handle is not None or groups:
            # M2N does not synchronize caller streams on finalize.
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        for group_index, group in groups.items():
            attempt(
                group.abort,
                f"failed to abort nccl_m2n route {group_index} communicator",
            )
        # Keep stream wrappers alive through communicator cleanup.
        del streams
        if first_error is not None:
            raise first_error
