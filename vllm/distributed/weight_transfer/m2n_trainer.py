# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight transfer engine for the NCCL M2N backend.

Symmetric to `M2NWeightTransferEngine` but in the training process. Each
trainer rank joins only the source-group communicators that own its parameters
and sends those rank-local shards; only rank 0 touches the inference control
plane.

Workers join the communicator during init and run reshard from inside
`update_weights`, so those two RPCs overlap with work on this side. Both run on
a helper thread and are joined afterwards, which keeps that overlap inside the
engine -- where the `TrainerWeightTransferEngine` contract puts it -- rather than
in every caller.
"""

from __future__ import annotations

import math
import socket
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import torch

from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.m2n_common import (
    M2N_WIRE_SCHEMA_VERSION,
    M2NMesh,
    M2NParamMeta,
    M2NSourceTopology,
    M2NTransferRoute,
    M2NTransferWave,
    M2NWireParam,
    Placements,
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
    validate_local_buffer,
)
from vllm.distributed.weight_transfer.m2n_plan import (
    M2NWireDestination,
    agree_destination_plan,
)
from vllm.distributed.weight_transfer.m2n_source import M2NWeightSource
from vllm.distributed.weight_transfer.nccl_common import (
    decode_nccl_unique_id,
    stateless_init_metadata_group,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

logger = init_logger(__name__)

_FINISH_RETRY_LIMIT = 1

__all__ = ["M2NTrainerInitInfo", "M2NTrainerWeightTransferEngine"]


@dataclass(kw_only=True)
class M2NTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for nccl_m2n.

    `rank` (from `TrainerInitInfo`) is this trainer process's global metadata
    rank. Each transfer route maps participating trainer ranks into its own
    communicator-local source order. Rank 0 drives the control plane.
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
    replicated destination needs.

    TODO: Carry a destination mesh/layout per parameter before supporting mixed
    generator TP/DP layouts whose dense and expert weights need different meshes."""
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    """Optional UID for the NCCL data plane; TCP remains the metadata plane."""
    nccl_unique_ids_b64: list[str] | None = field(default=None, repr=False)
    """Optional UIDs in canonical transfer-route order."""
    max_cta: int | None = None
    listen_socket: socket.socket | None = field(default=None, repr=False, compare=False)
    """Optional rank-zero TCPStore reservation; never sent to workers."""

    @property
    def destination_mesh_dims(self) -> tuple[int, int]:
        """`dst_mesh_dims`, or a flat mesh over every inference worker."""
        if self.dst_mesh_dims is not None:
            return cast(tuple[int, int], tuple(self.dst_mesh_dims))
        return (self.world_size - self.num_trainer_ranks, 1)

    def __post_init__(self) -> None:
        """Reject rank counts or destination meshes that cannot form a group."""
        if not isinstance(self.master_address, str) or not self.master_address:
            raise ValueError("`master_address` must name the metadata rendezvous")
        if (
            not isinstance(self.master_port, int)
            or isinstance(self.master_port, bool)
            or not 0 < self.master_port < 65536
        ):
            raise ValueError("`master_port` must be a valid metadata port")
        if not 0 < self.num_trainer_ranks < self.world_size:
            raise ValueError(
                f"`num_trainer_ranks` ({self.num_trainer_ranks}) must leave at "
                f"least one worker in world_size {self.world_size}"
            )
        num_workers = self.world_size - self.num_trainer_ranks
        if self.dst_mesh_dims is not None:
            dims = tuple(self.dst_mesh_dims)
            try:
                mesh = M2NMesh(cast(tuple[int, int], dims), self.num_trainer_ranks)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid `dst_mesh_dims` {dims}: {exc}") from exc
            if mesh.size != num_workers:
                raise ValueError(
                    f"`dst_mesh_dims` {dims} must be two dims covering the "
                    f"{num_workers} inference workers"
                )
        if not 0 <= self.rank < self.num_trainer_ranks:
            raise ValueError(
                f"trainer `rank` ({self.rank}) must be below "
                f"`num_trainer_ranks` ({self.num_trainer_ranks})"
            )
        if self.listen_socket is not None:
            if self.rank != 0:
                raise ValueError("only trainer rank zero may reserve the TCPStore")
            if not isinstance(self.listen_socket, socket.socket):
                raise TypeError("listen_socket must be a socket.socket")
            if self.listen_socket.fileno() < 0:
                raise ValueError("listen_socket is already closed")
        if self.nccl_unique_id_b64 is not None and self.nccl_unique_ids_b64 is not None:
            raise ValueError(
                "pass `nccl_unique_id_b64` or `nccl_unique_ids_b64`, not both"
            )
        if self.nccl_unique_ids_b64 is not None:
            if (
                not isinstance(self.nccl_unique_ids_b64, list)
                or not self.nccl_unique_ids_b64
            ):
                raise ValueError("`nccl_unique_ids_b64` must be a non-empty list")
            if any(
                not isinstance(value, str) or not value
                for value in self.nccl_unique_ids_b64
            ):
                raise ValueError(
                    "every entry in `nccl_unique_ids_b64` must be a non-empty string"
                )

    @property
    def nccl_unique_id_bytes(self) -> bytes | None:
        if self.nccl_unique_id_b64 is None:
            return None
        return decode_nccl_unique_id(
            master_address=None,
            master_port=None,
            nccl_unique_id_b64=self.nccl_unique_id_b64,
            ctx="M2NTrainerInitInfo data plane",
        )

    def data_plane_unique_ids(self, num_routes: int) -> tuple[bytes | None, ...]:
        """Decode one optional pre-shared NCCL UID per canonical route."""
        if (
            not isinstance(num_routes, int)
            or isinstance(num_routes, bool)
            or num_routes <= 0
        ):
            raise ValueError("nccl_m2n needs at least one transfer route")
        if self.nccl_unique_ids_b64 is not None:
            if len(self.nccl_unique_ids_b64) != num_routes:
                raise ValueError(
                    f"`nccl_unique_ids_b64` has {len(self.nccl_unique_ids_b64)} "
                    f"entries, expected {num_routes} transfer routes"
                )
            return tuple(
                decode_nccl_unique_id(
                    master_address=None,
                    master_port=None,
                    nccl_unique_id_b64=value,
                    ctx=f"M2NTrainerInitInfo data plane route {route_index}",
                )
                for route_index, value in enumerate(self.nccl_unique_ids_b64)
            )
        unique_id = self.nccl_unique_id_bytes
        if unique_id is not None:
            if num_routes != 1:
                raise ValueError(
                    "singular `nccl_unique_id_b64` is valid only for one transfer "
                    "route; pass `nccl_unique_ids_b64` for pipeline groups"
                )
            return (unique_id,)
        return (None,) * num_routes


class M2NTrainerWeightTransferEngine(TrainerWeightTransferEngine[M2NTrainerInitInfo]):
    """Send source-group-local shards over their canonical transfer routes.

    Called on every trainer rank. Ranks join only routes whose source group
    contains them; only rank 0 touches the inference control plane.
    """

    init_info_cls = M2NTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: M2NWeightSource,
        is_sender: bool,
    ) -> None:
        """Hold the client and source; `trainer_init` fills in the group."""
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.group: Any = None
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._wire_params: list[M2NWireParam] = []
        self._source_digest = ""
        self._destination_commit_token = ""
        self._dst_mesh: M2NMesh | None = None
        self._dst_placements: list[Placements | None] = []
        self._metadata_group: StatelessProcessGroup | None = None
        self._num_trainer_ranks = 0
        self._trainer_rank = -1
        self._destination_plan: list[M2NWireDestination] = []
        self._source_topology: M2NSourceTopology | None = None
        self._routes: tuple[M2NTransferRoute, ...] = ()
        self._param_routes: tuple[int, ...] = ()
        self._transfer_waves: tuple[M2NTransferWave, ...] = ()
        self._route_communicators: dict[int, Any] = {}
        self._route_streams: dict[int, torch.cuda.Stream] = {}
        self._round_id = 0

    @classmethod
    def trainer_init(
        cls,
        init_info: M2NTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        """Build the engine and rendezvous with the inference side.

        Runs on *every* trainer rank. Rank 0 additionally ships the transfer
        plan to the workers and drives the control plane; the other ranks only
        build local state and join their declared source-group communicators.

        The ordering here is not arbitrary -- see the comments inline. Build a
        local validation envelope, start the worker RPC without waiting, join
        the metadata rendezvous, and require unanimous source and destination
        plan commits before constructing NCCL or the M2N handle.
        """
        # Trainer-side weight transfer engine for this rank. `__init__` only
        # holds the client and source; the communicator, meshes, and M2N handle
        # are filled in below.
        engine = cls(
            client=client,
            source=cast(M2NWeightSource, source),
            is_sender=init_info.is_sender,
        )
        engine._initialize(init_info, source)
        return engine

    def _initialize(
        self,
        init_info: M2NTrainerInitInfo,
        source: WeightSource | None,
    ) -> None:
        """Run source validation, metadata agreement, and NCCL rendezvous."""
        self._num_trainer_ranks = init_info.num_trainer_ranks
        self._trainer_rank = init_info.rank
        local_source_error: str | None = None
        try:
            if not isinstance(source, M2NWeightSource):
                raise TypeError(
                    "nccl_m2n needs per-parameter layouts, so its source must "
                    f"be an M2NWeightSource; got {type(source).__name__}"
                )
            self._m2n = import_m2n()
            self._prepare_source_plan(source, init_info.num_trainer_ranks)
        except Exception as exc:
            local_source_error = f"{type(exc).__name__}: {exc}"
            self._metas = []
            self._wire_params = []
            self._source_topology = M2NSourceTopology((), ())
            self._source_digest = source_plan_digest([], self._source_topology)

        executor: ThreadPoolExecutor | None = None
        init_future: Future[None] | None = None
        if self.is_sender:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            init_future = executor.submit(
                self.client.init_weight_transfer_engine,
                self._worker_init_info(init_info),
            )
        metadata_group: StatelessProcessGroup | None = None
        handle = None
        route_communicators: dict[int, Any] = {}
        route_streams: dict[int, torch.cuda.Stream] = {}
        try:
            reservation = init_info.listen_socket
            try:
                metadata_group = stateless_init_metadata_group(
                    init_info.master_address,
                    init_info.master_port,
                    init_info.rank,
                    init_info.world_size,
                    listen_socket=reservation,
                )
            finally:
                init_info.listen_socket = None
                if reservation is not None:
                    reservation.close()
            plan_digest = check_source_plan_agreement(
                metadata_group,
                self._wire_params,
                expected_digest=self._source_digest,
                local_error=local_source_error,
                topology=self._source_topology,
            )

            # The worker resolves its rank-local destinations, and every rank
            # commits the same ordered plan before constructing NCCL.
            self._dst_mesh = M2NMesh(
                init_info.destination_mesh_dims, init_info.num_trainer_ranks
            )
            self._destination_plan, commit_token = agree_destination_plan(
                metadata_group,
                source_digest=plan_digest,
                params=self._wire_params,
                metas=self._metas,
                dst_mesh=self._dst_mesh,
                local_plan=None,
                local_error=None,
                source_topology=self._source_topology,
            )
            self._dst_placements = [
                destination.placements for destination in self._destination_plan
            ]
            self._destination_commit_token = commit_token
            assert self._source_topology is not None
            self._routes = build_m2n_transfer_routes(
                self._source_topology,
                num_trainer_ranks=init_info.num_trainer_ranks,
                world_size=init_info.world_size,
                dst_mesh_dims=init_info.destination_mesh_dims,
            )
            self._param_routes = build_m2n_param_routes(self._routes, len(self._metas))
            self._transfer_waves = build_m2n_transfer_waves(
                self._routes,
                tuple(
                    m2n_load_group(destination.mode, destination.staging_group)
                    for destination in self._destination_plan
                ),
            )
            unique_ids: tuple[bytes | None, ...] = (None,) * len(self._routes)
            data_plane_error: str | None = None
            try:
                unique_ids = init_info.data_plane_unique_ids(len(self._routes))
                check_multi_communicator_launch_order(len(self._routes))
            except Exception as exc:
                data_plane_error = f"{type(exc).__name__}: {exc}"
            check_data_plane_agreement(
                metadata_group,
                unique_ids,
                data_plane_error,
            )

            participates = any(
                init_info.rank in route.source_ranks for route in self._routes
            )
            runtime_error: str | None = None
            nccl = None
            device = None
            if participates:
                try:
                    nccl, device, handle = prepare_m2n_local_runtime(
                        self._m2n, init_info.max_cta
                    )
                except Exception as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
            check_runtime_ready_agreement(metadata_group, runtime_error)
            if participates:
                assert nccl is not None and device is not None and handle is not None

            # Every metadata rank bootstraps the fixed communicator list in
            # canonical order. Members initialize each communicator, then all
            # ranks report the complete ordered result once below.
            communicator_errors: list[str | None] = []
            for route_index, (route, unique_id) in enumerate(
                zip(self._routes, unique_ids, strict=True)
            ):
                route_error: str | None = None
                communicator = None
                try:
                    communicator = create_m2n_communicator(
                        nccl,
                        metadata_group,
                        unique_id,
                        members=route.members,
                    )
                    participates = metadata_group.rank in route.members
                    if participates != (communicator is not None):
                        raise RuntimeError(
                            f"transfer route {route_index} communicator membership "
                            "does not match its agreed rank list"
                        )
                    if communicator is not None:
                        assert device is not None
                        route_communicators[route_index] = communicator
                        route_streams[route_index] = torch.cuda.Stream(device=device)
                except Exception as exc:
                    route_error = f"{type(exc).__name__}: {exc}"
                communicator_errors.append(route_error)
            check_communicator_init_agreement(
                metadata_group,
                communicator_errors,
            )

            self._sync_trainer_init_outcome(metadata_group, init_future)
            logger.info(
                "nccl_m2n trainer ready: %d parameters, %d source groups, "
                "%d transfer routes, "
                "trainer ranks [0, %d), inference ranks [%d, %d)",
                len(self._metas),
                len(self._source_topology.groups),
                len(self._routes),
                init_info.num_trainer_ranks,
                init_info.num_trainer_ranks,
                init_info.world_size,
            )
            logger.info("nccl_m2n source plan digest: %s", plan_digest)
        except BaseException:
            self._destination_plan = []
            self._destination_commit_token = ""
            self._dst_placements = []
            self._dst_mesh = None
            self._routes = ()
            self._param_routes = ()
            self._transfer_waves = ()
            self._route_streams = {}
            for communicator in reversed(tuple(route_communicators.values())):
                try:
                    communicator.abort()
                except BaseException:
                    logger.exception(
                        "failed to abort partially initialized nccl_m2n communicator"
                    )
            route_communicators.clear()
            if handle is not None:
                try:
                    handle.destroy()
                except BaseException:
                    logger.exception(
                        "failed to destroy partially initialized nccl_m2n handle"
                    )
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

        self._metadata_group = metadata_group
        self._handle = handle
        self._route_communicators = route_communicators
        self._route_streams = route_streams
        # Compatibility alias for the legacy single-communicator surface.
        self.group = next(iter(route_communicators.values()), None)

    def _sync_trainer_init_outcome(
        self,
        metadata_group: StatelessProcessGroup,
        init_future: Future[None] | None,
    ) -> None:
        """Require every trainer to observe rank zero's worker-init result."""
        if self.is_sender:
            if init_future is None:
                raise RuntimeError("trainer rank zero is missing its worker-init RPC")
            failure: BaseException | None = None
            try:
                init_future.result()
            except BaseException as exc:
                failure = exc
            message = {
                "phase": "trainer_init_outcome",
                "plan_token": self._destination_commit_token,
                "ok": failure is None,
                "error": (
                    None if failure is None else f"{type(failure).__name__}: {failure}"
                ),
            }
            for rank in range(1, self._num_trainer_ranks):
                metadata_group.send_obj(message, rank)
            if failure is not None:
                raise failure
            return

        message = metadata_group.recv_obj(0)
        expected_fields = {"phase", "plan_token", "ok", "error"}
        if not isinstance(message, Mapping) or set(message) != expected_fields:
            raise RuntimeError("trainer rank zero sent an invalid init outcome")
        if (
            message["phase"] != "trainer_init_outcome"
            or message["plan_token"] != self._destination_commit_token
        ):
            raise RuntimeError("trainer rank zero sent an invalid init outcome")
        ok = message["ok"]
        error = message["error"]
        if not isinstance(ok, bool) or ok != (error is None):
            raise RuntimeError("trainer rank zero sent an invalid init outcome")
        if not ok:
            if not isinstance(error, str) or not error:
                raise RuntimeError("trainer rank zero sent an invalid init outcome")
            raise RuntimeError(
                "nccl_m2n worker initialization failed on trainer rank zero: " + error
            )

    def _prepare_source_plan(
        self, source: M2NWeightSource, num_trainer_ranks: int
    ) -> None:
        """Build and validate this rank's immutable source wire plan."""
        self._metas = source.metadata()
        topology = source.source_topology()
        if topology is None:
            topology = M2NSourceTopology.single_group(
                num_trainer_ranks, len(self._metas)
            )
        elif not isinstance(topology, M2NSourceTopology):
            raise TypeError(
                "M2NWeightSource.source_topology() must return an "
                f"M2NSourceTopology or None, got {type(topology).__name__}"
            )
        topology.validate(
            num_trainer_ranks=num_trainer_ranks,
            num_params=len(self._metas),
        )
        self._source_topology = topology
        names: set[str] = set()
        for param_index, meta in enumerate(self._metas):
            if not isinstance(meta, M2NParamMeta):
                raise TypeError(
                    "nccl_m2n source metadata must contain M2NParamMeta "
                    f"entries, got {type(meta).__name__}"
                )
            if meta.name in names:
                raise ValueError(
                    "nccl_m2n source metadata contains duplicate parameter "
                    f"'{meta.name}'"
                )
            names.add(meta.name)
            check_transferable(meta.name, meta.dtype, meta.shape)
            layout = meta.source_layout
            if layout.mesh.start_rank != 0:
                raise ValueError(
                    f"parameter '{meta.name}' source mesh must start at rank 0, "
                    f"got {layout.mesh.start_rank}"
                )
            group_index = topology.param_groups[param_index]
            source_group_size = len(topology.groups[group_index])
            if layout.mesh.size != source_group_size:
                raise ValueError(
                    f"parameter '{meta.name}' source mesh covers "
                    f"{layout.mesh.size} ranks, but source group {group_index} "
                    f"contains {source_group_size} ranks"
                )
            expected_local_shape(meta)
        held_names = source.held_names()
        if held_names is not None:
            held_values = tuple(held_names)
            if any(not isinstance(name, str) for name in held_values):
                raise TypeError("M2NWeightSource.held_names() must contain strings")
            held = set(held_values)
            required = {
                meta.name
                for param_index, meta in enumerate(self._metas)
                if self._trainer_rank
                in topology.groups[topology.param_groups[param_index]]
            }
            missing = sorted(required - held)
            if missing:
                raise ValueError(
                    "M2NWeightSource.held_names() omits parameters assigned to "
                    f"trainer rank {self._trainer_rank}, including {missing[0]!r}"
                )
        self._wire_params = [
            M2NWireParam(
                name=meta.name,
                dtype_name=str(meta.dtype).split(".")[-1],
                shape=meta.shape,
                src_mesh_dims=meta.source_layout.mesh.dims,
                src_placements=meta.source_layout.placements,
                allow_full_fallback=meta.allow_full_fallback,
            )
            for meta in self._metas
        ]
        self._source_digest = source_plan_digest(self._wire_params, topology)

    def plan_summary(self) -> dict[str, Any]:
        """Return the authenticated static plan as JSON-safe run metadata."""
        if not self._destination_commit_token or not self._destination_plan:
            raise RuntimeError("nccl_m2n transfer plan is not initialized")
        assert self._dst_mesh is not None
        assert self._source_topology is not None
        mode_counts: dict[str, int] = {}
        logical_bytes_by_mode: dict[str, int] = {}
        receiver_bytes_by_mode: dict[str, int] = {}
        for meta, destination in zip(self._metas, self._destination_plan, strict=True):
            mode = destination.mode
            element_size = torch.empty((), dtype=meta.dtype).element_size()
            logical_bytes = math.prod(meta.shape) * element_size
            receiver_bytes = (
                math.prod(destination.local_shape) * element_size * self._dst_mesh.size
            )
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            logical_bytes_by_mode[mode] = (
                logical_bytes_by_mode.get(mode, 0) + logical_bytes
            )
            receiver_bytes_by_mode[mode] = (
                receiver_bytes_by_mode.get(mode, 0) + receiver_bytes
            )
        logical_global_bytes = sum(logical_bytes_by_mode.values())
        planned_receiver_payload_bytes = sum(receiver_bytes_by_mode.values())
        return {
            "wire_schema_version": M2N_WIRE_SCHEMA_VERSION,
            "source_digest": self._source_digest,
            "destination_commit_token": self._destination_commit_token,
            "destination_mesh_dims": list(self._dst_mesh.dims),
            "source_group_count": len(self._source_topology.groups),
            "parameter_count": len(self._metas),
            "transfer_communicators": [
                {
                    "source_group_index": route.source_group_index,
                    "source_ranks": list(route.source_ranks),
                    "destination_ranks": list(route.worker_ranks),
                    "param_indices": list(route.param_indices),
                }
                for route in self._routes
            ],
            "transfer_waves": [
                {
                    "issue_indices": list(wave.issue_indices),
                    "consume_index": wave.consume_index,
                }
                for wave in self._transfer_waves
            ],
            # Each complete logical parameter is counted once.
            "logical_global_bytes": logical_global_bytes,
            # Parameter counts, not byte counts.
            "destination_mode_counts": mode_counts,
            "logical_global_bytes_by_destination_mode": logical_bytes_by_mode,
            # Aggregate destination tensor bytes implied by the agreed plan;
            # this is not measured NCCL, InfiniBand, or protocol traffic.
            "planned_receiver_payload_bytes": planned_receiver_payload_bytes,
            "planned_receiver_payload_bytes_by_destination_mode": (
                receiver_bytes_by_mode
            ),
        }

    def _worker_init_info(self, init_info: M2NTrainerInitInfo) -> dict[str, Any]:
        """Handshake payload: rendezvous, both meshes, and the transfer plan."""
        payload: dict[str, Any] = {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "master_address": init_info.master_address,
            "master_port": init_info.master_port,
            "nccl_unique_id_b64": init_info.nccl_unique_id_b64,
            "nccl_unique_ids_b64": init_info.nccl_unique_ids_b64,
            "rank_offset": init_info.num_trainer_ranks,
            "world_size": init_info.world_size,
            "dst_mesh_dims": list(init_info.destination_mesh_dims),
            "source_digest": self._source_digest,
            "source_topology": (
                None
                if self._source_topology is None
                else self._source_topology.to_dict()
            ),
            "params": [param.to_dict() for param in self._wire_params],
            "max_cta": init_info.max_cta,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def send_weights(self) -> None:
        """Drive one update, replaying it once after a reset finish failure."""
        participates = any(
            self._trainer_rank in route.source_ranks for route in self._routes
        )
        if not self._routes or (participates and self._handle is None):
            raise RuntimeError("nccl_m2n trainer engine is not initialized")

        for attempt in range(_FINISH_RETRY_LIMIT + 1):
            if not self._send_weight_round(retry_finish=attempt < _FINISH_RETRY_LIMIT):
                return
        raise AssertionError("unreachable nccl_m2n finish retry state")

    def _send_weight_round(self, *, retry_finish: bool) -> bool:
        """Run one complete update attempt; return whether to replay it."""
        if not self.is_sender:
            failure: BaseException | None = None
            retry_update = False
            try:
                error, requires_reinit = self._receive_round_dispatch()
            except BaseException as exc:
                failure = exc
                self._abort_data_plane()
            else:
                if requires_reinit:
                    self._abort_data_plane()
                if error is not None:
                    failure = RuntimeError(
                        "nccl_m2n worker update was not dispatched: " + error
                    )
            if failure is None:
                try:
                    self._send()
                except BaseException as exc:
                    failure = exc
            try:
                error, requires_reinit, retry_update = self._receive_round_outcome()
            except BaseException as exc:
                self._abort_data_plane()
                if failure is None:
                    failure = exc
                else:
                    logger.exception(
                        "failed to consume nccl_m2n round outcome after local failure"
                    )
            else:
                if requires_reinit:
                    self._abort_data_plane()
                if error is not None and failure is None and not retry_update:
                    failure = RuntimeError(
                        "nccl_m2n worker update failed on trainer rank zero: " + error
                    )
            finally:
                self._round_id += 1
            if failure is not None:
                raise failure
            return retry_update

        executor: ThreadPoolExecutor | None = None
        dispatch_attempted = False
        transfer_completed = False
        try:
            self.client.start_weight_update()
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            update = executor.submit(
                self.client.update_weights,
                {"names": [m.name for m in self._metas]},
            )
            # Surface an immediate worker-side rejection before trainers
            # enter the collective sequence.
            if update.done():
                update.result()
            dispatch_attempted = True
            self._publish_round_dispatch(None)
            self._send()
            update.result()
            transfer_completed = True
            self.client.finish_weight_update()
        except BaseException as exc:
            requires_reinit = dispatch_attempted and not transfer_completed
            retry_update = transfer_completed and retry_finish
            if requires_reinit:
                self._abort_data_plane()
            if not dispatch_attempted:
                try:
                    dispatch_attempted = True
                    self._publish_round_dispatch(exc, requires_reinit=False)
                except BaseException:
                    self._abort_data_plane()
                    requires_reinit = True
                    logger.exception(
                        "failed to publish nccl_m2n round dispatch failure"
                    )
            try:
                self._publish_round_outcome(
                    exc,
                    requires_reinit=requires_reinit,
                    retry_update=retry_update,
                )
            except BaseException:
                self._abort_data_plane()
                logger.exception("failed to publish nccl_m2n round failure")
            if retry_update:
                return True
            raise
        else:
            try:
                self._publish_round_outcome(None)
            except BaseException:
                self._abort_data_plane()
                raise
            return False
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            self._round_id += 1

    def _publish_round_dispatch(
        self,
        failure: BaseException | None,
        *,
        requires_reinit: bool = False,
    ) -> None:
        """Release non-senders only after rank zero dispatched worker receive."""
        self._publish_round_control(
            "round_dispatch",
            failure,
            requires_reinit=requires_reinit,
            retry_update=False,
        )

    def _receive_round_dispatch(self) -> tuple[str | None, bool]:
        """Wait for rank zero before entering any source-side M2N collective."""
        error, requires_reinit, retry_update = self._receive_round_control(
            "round_dispatch"
        )
        assert not retry_update
        return error, requires_reinit

    def _publish_round_outcome(
        self,
        failure: BaseException | None,
        *,
        requires_reinit: bool = False,
        retry_update: bool = False,
    ) -> None:
        """Tell non-sender trainer ranks whether worker finalization passed."""
        self._publish_round_control(
            "round_outcome",
            failure,
            requires_reinit=requires_reinit,
            retry_update=retry_update,
        )

    def _publish_round_control(
        self,
        phase: str,
        failure: BaseException | None,
        *,
        requires_reinit: bool,
        retry_update: bool,
    ) -> None:
        """Publish one fixed-count trainer-only control message and await ACKs."""
        if self._num_trainer_ranks <= 1:
            return
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        envelope = {
            "phase": phase,
            "round_id": self._round_id,
            "ok": failure is None,
            "error": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
            "requires_reinit": requires_reinit,
            "retry_update": retry_update,
        }
        for rank in range(1, self._num_trainer_ranks):
            self._metadata_group.send_obj(envelope, rank)
        for rank in range(1, self._num_trainer_ranks):
            ack = self._metadata_group.recv_obj(rank)
            expected_ack = {
                "phase": f"{phase}_consumed",
                "round_id": self._round_id,
                "rank": rank,
                "error": None,
            }
            if ack == expected_ack:
                continue
            if (
                isinstance(ack, Mapping)
                and set(ack) == set(expected_ack)
                and ack.get("phase") == expected_ack["phase"]
                and ack.get("round_id") == self._round_id
                and ack.get("rank") == rank
                and isinstance(ack.get("error"), str)
                and ack["error"]
            ):
                raise RuntimeError(
                    f"trainer rank {rank} rejected {phase}: {ack['error']}"
                )
            else:
                raise RuntimeError(
                    f"trainer rank {rank} sent an invalid {phase} acknowledgement"
                )

    def _receive_round_outcome(self) -> tuple[str | None, bool, bool]:
        """Wait until trainer rank zero observed every worker's finish RPC."""
        return self._receive_round_control("round_outcome")

    def _receive_round_control(self, phase: str) -> tuple[str | None, bool, bool]:
        """Receive and acknowledge one rank-zero control message for this round."""
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        message = self._metadata_group.recv_obj(0)
        validation_error: str | None = None
        remote_error: str | None = None
        requires_reinit = False
        retry_update = False
        try:
            expected_fields = {
                "phase",
                "round_id",
                "ok",
                "error",
                "requires_reinit",
                "retry_update",
            }
            if not isinstance(message, Mapping) or set(message) != expected_fields:
                raise RuntimeError(f"trainer rank zero sent an invalid {phase}")
            if message["phase"] != phase or message["round_id"] != self._round_id:
                raise RuntimeError(f"trainer rank zero sent an out-of-order {phase}")
            ok = message["ok"]
            remote_error = message["error"]
            requires_reinit = message["requires_reinit"]
            retry_update = message["retry_update"]
            if not isinstance(ok, bool) or ok != (remote_error is None):
                raise RuntimeError(f"trainer rank zero sent an invalid {phase}")
            if not isinstance(requires_reinit, bool) or (ok and requires_reinit):
                raise RuntimeError(f"trainer rank zero sent an invalid {phase}")
            if (
                not isinstance(retry_update, bool)
                or (ok and retry_update)
                or (requires_reinit and retry_update)
                or (phase != "round_outcome" and retry_update)
            ):
                raise RuntimeError(f"trainer rank zero sent an invalid {phase}")
            if not ok and (not isinstance(remote_error, str) or not remote_error):
                raise RuntimeError(f"trainer rank zero sent an invalid {phase} error")
        except RuntimeError as exc:
            validation_error = str(exc)
        self._metadata_group.send_obj(
            {
                "phase": f"{phase}_consumed",
                "round_id": self._round_id,
                "rank": self._trainer_rank,
                "error": validation_error,
            },
            0,
        )
        if validation_error is not None:
            raise RuntimeError(validation_error)
        return remote_error, requires_reinit, retry_update

    def _send(self) -> None:
        """Reshard each local shard into its worker-planned destination."""
        assert self.source is not None
        assert self._source_topology is not None
        source = cast(M2NWeightSource, self.source)
        m2n = self._m2n

        joined_routes = {
            route_index
            for route_index, route in enumerate(self._routes)
            if self._trainer_rank in route.source_ranks
        }
        if not joined_routes:
            return

        iterators: dict[int, Any] = {}
        inflight_buffers: list[Any] = []
        try:
            # Optimizer work may have used another stream. Complete it before
            # each provider materializes onto its persistent transfer stream.
            if joined_routes:
                device_index = torch.accelerator.current_device_index()
                torch.accelerator.synchronize(device_index)
            # Replay the same bounded schedule as the workers. Keeping the
            # buffers in this frame protects them until completion or abort.
            for wave in self._transfer_waves:
                for param_index in wave.issue_indices:
                    group_index = self._source_topology.param_groups[param_index]
                    route_index = self._param_routes[param_index]
                    if route_index in joined_routes:
                        route = self._routes[route_index]
                        meta = self._metas[param_index]
                        communicator = self._route_communicators.get(route_index)
                        stream = self._route_streams.get(route_index)
                        if communicator is None or stream is None:
                            raise RuntimeError(
                                f"trainer rank {self._trainer_rank} is missing "
                                f"transfer route {route_index} for source group "
                                f"{group_index}"
                            )

                        with torch.cuda.stream(stream):
                            iterator = iterators.get(group_index)
                            if iterator is None:
                                iterator = iter(source.iter_group(group_index))
                                iterators[group_index] = iterator
                            try:
                                name, buffer = next(iterator)
                            except StopIteration as exc:
                                raise RuntimeError(
                                    f"source group {group_index} ended before "
                                    f"parameter {meta.name!r} at manifest position "
                                    f"{param_index}"
                                ) from exc
                            if name != meta.name:
                                raise RuntimeError(
                                    f"source group {group_index} yielded {name!r} "
                                    f"for manifest position {param_index}, expected "
                                    f"{meta.name!r}"
                                )
                            if buffer is None:
                                raise RuntimeError(
                                    f"source group {group_index} yielded no local "
                                    f"buffer for owned parameter {meta.name!r}"
                                )
                            validate_local_buffer(m2n, meta, buffer)
                            src_mesh, src_placements = resolve_layout(
                                meta.source_layout.mesh,
                                meta.source_layout.placements,
                            )
                            dst_mesh, dst_placements = resolve_layout(
                                route.dst_mesh,
                                self._dst_placements[param_index],
                            )
                            raw_pointer_args = {}
                            if isinstance(buffer, int):
                                raw_pointer_args = {
                                    "src_local_shape": expected_local_shape(meta),
                                    "src_dtype": meta.dtype,
                                }
                            m2n.reshard(
                                buffer,
                                None,
                                communicator,
                                stream,
                                src_mesh=to_mesh(m2n, src_mesh),
                                src_placements=to_placements(m2n, src_placements),
                                dst_mesh=to_mesh(m2n, dst_mesh),
                                dst_placements=to_placements(m2n, dst_placements),
                                handle=self._handle,
                                **raw_pointer_args,
                            )
                            record_stream = getattr(buffer, "record_stream", None)
                            if callable(record_stream):
                                record_stream(stream)
                            elif not isinstance(buffer, int):
                                inflight_buffers.append(buffer)

            for group_index, iterator in iterators.items():
                route_index = next(
                    index
                    for index, route in enumerate(self._routes)
                    if route.source_group_index == group_index
                    and index in joined_routes
                )
                stream = self._route_streams[route_index]
                with torch.cuda.stream(stream):
                    try:
                        name, _ = next(iterator)
                    except StopIteration:
                        continue
                    raise RuntimeError(
                        f"source group {group_index} yielded extra parameter {name!r}"
                    )

            # Completion protects source-buffer lifetime and prevents optimizer
            # mutation before worker finalization.
            for route_index in range(len(self._routes)):
                stream = self._route_streams.get(route_index)
                if stream is not None:
                    stream.synchronize()
        except BaseException:
            self._abort_data_plane()
            raise

    def _abort_data_plane(self) -> None:
        """Abort every local communicator without waiting on CUDA streams."""
        handle, self._handle = self._handle, None
        legacy_group, self.group = self.group, None
        communicators = list(self._route_communicators.values())
        if legacy_group is not None and all(
            legacy_group is not communicator for communicator in communicators
        ):
            communicators.append(legacy_group)
        self._route_communicators = {}
        streams, self._route_streams = self._route_streams, {}
        self._routes = ()
        self._param_routes = ()
        self._transfer_waves = ()

        for communicator in reversed(communicators):
            try:
                communicator.abort()
            except BaseException:
                logger.exception("failed to abort nccl_m2n communicator")
        if handle is not None:
            try:
                handle.destroy()
            except BaseException:
                logger.exception("failed to destroy nccl_m2n handle")
        self._source_topology = None
        self._destination_plan = []
        self._destination_commit_token = ""
        self._dst_placements = []
        self._dst_mesh = None
        self._m2n = None
        # Keep stream wrappers alive until communicator abort and handle
        # destruction have completed.
        del streams

    def shutdown(self) -> None:
        """Finish pending GPU work and release M2N communication state."""
        handle, self._handle = self._handle, None
        legacy_group, self.group = self.group, None
        communicators = list(self._route_communicators.values())
        if legacy_group is not None and all(
            legacy_group is not communicator for communicator in communicators
        ):
            communicators.append(legacy_group)
        self._route_communicators = {}
        streams, self._route_streams = self._route_streams, {}
        self._routes = ()
        self._param_routes = ()
        self._transfer_waves = ()
        self._source_topology = None
        self._metadata_group = None
        self._destination_plan = []
        self._destination_commit_token = ""
        self._dst_placements = []
        self._dst_mesh = None
        self._m2n = None

        first_error: BaseException | None = None

        def attempt(action: Any, message: str) -> None:
            nonlocal first_error
            try:
                action()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                logger.exception(message)

        if handle is not None or communicators:
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        for communicator in reversed(communicators):
            attempt(
                communicator.abort,
                "failed to abort nccl_m2n communicator",
            )
        # Keep stream wrappers alive through communicator cleanup.
        del streams
        if first_error is not None:
            raise first_error
