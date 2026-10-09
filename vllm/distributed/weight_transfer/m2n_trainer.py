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

from __future__ import annotations

import math
import socket
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
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
    M2NWireParam,
    Placements,
    check_data_plane_agreement,
    check_runtime_ready_agreement,
    check_source_plan_agreement,
    check_transferable,
    comm_ptr,
    import_m2n,
    prepare_m2n_local_runtime,
    resolve_layout,
    source_plan_digest,
    to_mesh,
    to_placements,
    validate_layout,
    validate_local_tensor,
    validate_m2n_nccl_communicator,
)
from vllm.distributed.weight_transfer.m2n_plan import (
    M2NWireDestination,
    agree_destination_plan,
)
from vllm.distributed.weight_transfer.m2n_source import M2NWeightSource
from vllm.distributed.weight_transfer.nccl_common import (
    decode_nccl_unique_id,
    pynccl_from_metadata_group,
    stateless_init_metadata_group,
    uid_init_process_group,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

logger = init_logger(__name__)

__all__ = ["M2NTrainerInitInfo", "M2NTrainerWeightTransferEngine"]


@dataclass(kw_only=True)
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
    replicated destination needs.

    TODO: Carry a destination mesh/layout per parameter before supporting mixed
    generator TP/DP layouts whose dense and expert weights need different meshes."""
    nccl_unique_id_b64: str | None = field(default=None, repr=False)
    """Optional UID for the NCCL data plane; TCP remains the metadata plane."""
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
        is_sender: bool,
    ) -> None:
        """Hold the client and source; `trainer_init` fills in the group."""
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.group: PyNcclCommunicator | None = None
        self._m2n: Any = None
        self._handle: Any = None
        self._metas: list[M2NParamMeta] = []
        self._wire_params: list[M2NWireParam] = []
        self._source_digest = ""
        self._destination_commit_token = ""
        self._dst_mesh: M2NMesh | None = None
        self._dst_placements: list[Placements | None] = []
        self._destination_plan: list[M2NWireDestination] = []
        self._executor: ThreadPoolExecutor | None = None
        self._init_future: Future[None] | None = None
        self._update_future: Future[None] | None = None
        self._metadata_group: StatelessProcessGroup | None = None
        self._num_trainer_ranks = 0
        self._failure: BaseException | None = None
        self._closed = False
        self._lifecycle_lock = Lock()

    @classmethod
    def trainer_init(
        cls,
        init_info: M2NTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,  # type: ignore[override]
    ) -> Self:
        """Build the engine and rendezvous with the inference side.

        Runs on *every* trainer rank. Rank 0 additionally ships the transfer
        plan to the workers and drives the control plane; the other ranks only
        build local state and join the communicator. Every rank participates
        in every reshard and sends its local shard.

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
        try:
            engine._initialize(init_info, source)
        except BaseException as exc:
            engine._abort(exc)
            raise
        return engine

    def _initialize(
        self,
        init_info: M2NTrainerInitInfo,
        source: WeightSource,
    ) -> None:
        """Run the source, control-plane, metadata, and NCCL rendezvous."""
        self._num_trainer_ranks = init_info.num_trainer_ranks
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
            self._source_digest = source_plan_digest([])

        if self.is_sender:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="m2n-weight-sync"
            )
            # Only trainer rank 0 starts the control-plane RPC. Workers join
            # the communicator inside it, so submit it first, join below, and
            # wait for it at the end.
            self._init_future = self._executor.submit(
                self.client.init_weight_transfer_engine,
                self._worker_init_info(init_info),
            )

        # Every trainer rank joins -- reshard is collective over the whole
        # communicator. Trainer ranks take
        # [0, num_trainer_ranks) and the workers follow, which is the
        # contiguous-interval layout m2n requires of both meshes.
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
        self._metadata_group = metadata_group
        plan_digest = check_source_plan_agreement(
            metadata_group,
            self._wire_params,
            expected_digest=self._source_digest,
            local_error=local_source_error,
        )

        # The trainer declares the inference topology; the worker uses exactly
        # this, so neither side has to infer the other's factorization.
        self._dst_mesh = M2NMesh(
            init_info.destination_mesh_dims, init_info.num_trainer_ranks
        )
        self._destination_plan, commit_token = agree_destination_plan(
            metadata_group,
            source_digest=plan_digest,
            params=self._wire_params,
            dst_mesh=self._dst_mesh,
            local_plan=None,
            local_error=None,
        )
        self._dst_placements = [
            destination.placements for destination in self._destination_plan
        ]
        self._destination_commit_token = commit_token
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
        runtime_error: str | None = None
        nccl_runtime = None
        device = None
        try:
            nccl_runtime, device, self._handle = prepare_m2n_local_runtime(
                self._m2n, init_info.max_cta
            )
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"
        check_runtime_ready_agreement(metadata_group, runtime_error)
        assert nccl_runtime is not None and device is not None
        if unique_id_bytes is None:
            self.group = pynccl_from_metadata_group(
                metadata_group,
                device,
                library_path=nccl_runtime.library_path,
            )
        else:
            self.group = uid_init_process_group(
                unique_id_bytes,
                rank=metadata_group.rank,
                world_size=metadata_group.world_size,
                device=device,
                library_path=nccl_runtime.library_path,
            )
        validate_m2n_nccl_communicator(self.group, nccl_runtime)
        logger.info(
            "nccl_m2n trainer ready: %d parameters, trainer ranks [0, %d), "
            "inference ranks [%d, %d)",
            len(self._metas),
            init_info.num_trainer_ranks,
            init_info.num_trainer_ranks,
            init_info.world_size,
        )
        logger.info("nccl_m2n source plan digest: %s", plan_digest)

        # Now safe to collect: both sides have joined, so the RPC has returned
        # or raised. Doing this last means a worker-side init failure surfaces
        # here, as an exception from trainer_init, rather than at the first send.
        if self._init_future is not None:
            self._init_future.result()
            self._init_future = None

    def _prepare_source_plan(
        self, source: M2NWeightSource, num_trainer_ranks: int
    ) -> None:
        """Build and validate this rank's immutable source wire plan."""
        self._metas = source.metadata()
        names: set[str] = set()
        for meta in self._metas:
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
            if layout.mesh.size != num_trainer_ranks:
                raise ValueError(
                    f"parameter '{meta.name}' source mesh covers "
                    f"{layout.mesh.size} ranks, but there are "
                    f"{num_trainer_ranks} trainer ranks"
                )
            mesh, placements = resolve_layout(
                layout.mesh,
                layout.placements,
                f"parameter '{meta.name}' source placements",
            )
            validate_layout(mesh, placements, meta.shape, "source")
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
        self._source_digest = source_plan_digest(self._wire_params)

    def plan_summary(self) -> dict[str, Any]:
        """Return the authenticated static plan as JSON-safe run metadata."""
        if not self._destination_commit_token or not self._destination_plan:
            raise RuntimeError("nccl_m2n transfer plan is not initialized")
        assert self._dst_mesh is not None
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
            "parameter_count": len(self._metas),
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
            "rank_offset": init_info.num_trainer_ranks,
            "world_size": init_info.world_size,
            "dst_mesh_dims": list(init_info.destination_mesh_dims),
            "source_digest": self._source_digest,
            "params": [param.to_dict() for param in self._wire_params],
            "max_cta": init_info.max_cta,
        }
        return {key: value for key, value in payload.items() if value is not None}

    def send_weights(self) -> None:
        """Drive one update round: start, reshard concurrently, then finish."""
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("nccl_m2n trainer lifecycle operation is in progress")
        try:
            self._send_weights_locked()
        finally:
            self._lifecycle_lock.release()

    def _send_weights_locked(self) -> None:
        """Run one update while holding the lifecycle lock."""
        if self._failure is not None:
            raise RuntimeError(
                "nccl_m2n trainer engine failed during an earlier update and "
                "cannot be reused"
            ) from self._failure
        if self._closed:
            raise RuntimeError("nccl_m2n trainer engine is shut down")
        if self._handle is None or self.group is None:
            raise RuntimeError("nccl_m2n trainer engine is not initialized")

        try:
            if self.is_sender:
                if self._executor is None:
                    raise RuntimeError("nccl_m2n controller executor is missing")
                self.client.start_weight_update()
                # The workers reshard from inside `update_weights`, concurrently
                # with the sends below.
                self._update_future = self._executor.submit(
                    self.client.update_weights,
                    {"names": [m.name for m in self._metas]},
                )

            self._send()
            if self._update_future is not None:
                self._update_future.result()
                self._update_future = None
            if self.is_sender:
                self.client.finish_weight_update()
        except BaseException as exc:
            self._abort(exc)
            if self.is_sender:
                try:
                    self._publish_round_outcome(exc)
                except BaseException:
                    logger.exception("failed to publish nccl_m2n round failure")
            raise
        if self.is_sender:
            try:
                self._publish_round_outcome(None)
            except BaseException as exc:
                self._abort(exc)
                raise
        else:
            try:
                self._receive_round_outcome()
            except BaseException as exc:
                self._abort(exc)
                raise

    def _publish_round_outcome(self, failure: BaseException | None) -> None:
        """Tell non-controller trainer ranks whether worker finalization passed."""
        if self._num_trainer_ranks <= 1:
            return
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        envelope = {
            "ok": failure is None,
            "error": (
                None if failure is None else f"{type(failure).__name__}: {failure}"
            ),
        }
        for rank in range(1, self._num_trainer_ranks):
            self._metadata_group.send_obj(envelope, rank)

    def _receive_round_outcome(self) -> None:
        """Wait until trainer rank zero has observed every worker's finish RPC."""
        if self._metadata_group is None:
            raise RuntimeError("nccl_m2n metadata group is unavailable")
        outcome = self._metadata_group.recv_obj(0)
        if not isinstance(outcome, Mapping) or set(outcome) != {"ok", "error"}:
            raise RuntimeError("trainer rank zero sent an invalid round outcome")
        ok = outcome["ok"]
        error = outcome["error"]
        if not isinstance(ok, bool) or ok != (error is None):
            raise RuntimeError("trainer rank zero sent an invalid round outcome")
        if not ok:
            if not isinstance(error, str) or not error:
                raise RuntimeError("trainer rank zero sent an invalid round error")
            raise RuntimeError(
                "nccl_m2n worker update failed on trainer rank zero: " + error
            )

    def _abort(self, failure: BaseException) -> None:
        """Poison the engine and best-effort abort every owned resource."""
        if self._failure is None:
            self._failure = failure
        self._closed = True

        # Detach first so cleanup is idempotent and a second exception cannot
        # operate on a half-destroyed communicator or helper thread.
        group, self.group = self.group, None
        handle, self._handle = self._handle, None
        init_future, self._init_future = self._init_future, None
        update_future, self._update_future = self._update_future, None
        executor, self._executor = self._executor, None
        for future in (init_future, update_future):
            if future is not None:
                future.cancel()

        # PyNcclCommunicator.destroy uses ncclCommAbort. Do it before destroying
        # the M2N handle so peers blocked in a collective can make progress.
        if group is not None:
            try:
                group.destroy()
            except BaseException:
                logger.exception("failed to abort nccl_m2n communicator")
        if handle is not None:
            try:
                handle.destroy()
            except BaseException:
                logger.exception("failed to destroy nccl_m2n handle after failure")
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except BaseException:
                logger.exception("failed to stop nccl_m2n executor after failure")

        self._destination_plan = []
        self._destination_commit_token = ""
        self._dst_placements = []
        self._dst_mesh = None
        self._m2n = None

    def _send(self) -> None:
        """Reshard every local shard using its own source layout."""
        try:
            m2n = self._m2n
            assert self.group is not None
            assert self.source is not None
            assert self._dst_mesh is not None
            comm = comm_ptr(self.group)
            stream = torch.cuda.current_stream()
            expected_device = torch.device(
                "cuda", torch.accelerator.current_device_index()
            )
            # Optimizer work may have used another CUDA stream. Make all prior
            # writes visible before providers materialize current-stream shards.
            torch.accelerator.synchronize(expected_device.index)
            sent = 0

            for index, (name, tensor) in enumerate(self.source):
                if index >= len(self._metas):
                    raise RuntimeError(
                        "weight source yielded more parameters than metadata() "
                        f"declared (first extra parameter: '{name}')"
                    )
                meta = self._metas[index]
                if name != meta.name:
                    raise RuntimeError(
                        f"weight source yielded '{name}' at position {index}, but "
                        f"its metadata declared '{meta.name}'; the source must "
                        "iterate in a stable order"
                    )
                validate_local_tensor(meta, tensor, expected_device)
                shard = tensor.detach()
                layout = meta.source_layout
                src_mesh, src_placements = resolve_layout(
                    layout.mesh, layout.placements
                )
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
                # M2N retains only a device pointer for asynchronous work. This
                # protects temporary storage lifetime; source providers are
                # responsible for materializing on the caller's current stream.
                shard.record_stream(stream)
                sent += 1

            if sent != len(self._metas):
                missing = self._metas[sent].name
                raise RuntimeError(
                    f"weight source yielded {sent} parameters, but metadata() "
                    f"declared {len(self._metas)} (first missing: '{missing}')"
                )

            # As in the other weight-transfer backends, wait before returning
            # rather than relying on same-stream ordering. The caller may then
            # safely mutate parameters or continue work on another stream.
            stream.synchronize()
        except BaseException as exc:
            # A rank-local materializer or validation failure would otherwise
            # leave peers blocked in their next reshard. PyNcclCommunicator's
            # destroy path uses ncclCommAbort, which unblocks the communicator.
            self._abort(exc)
            raise

    def shutdown(self) -> None:
        """Synchronize and explicitly release every successful-run resource."""
        if not self._lifecycle_lock.acquire(blocking=False):
            raise RuntimeError("cannot shut down nccl_m2n during a weight update")
        try:
            self._shutdown_locked()
        finally:
            self._lifecycle_lock.release()

    def _shutdown_locked(self) -> None:
        """Release resources while holding the lifecycle lock."""
        if self._failure is not None:
            self._abort(self._failure)
            return

        self._closed = True
        handle, self._handle = self._handle, None
        group, self.group = self.group, None
        executor, self._executor = self._executor, None
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
            attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
        if handle is not None:
            attempt(handle.destroy, "failed to destroy nccl_m2n handle")
        if group is not None:
            attempt(group.destroy, "failed to destroy nccl_m2n communicator")
        if executor is not None:
            attempt(executor.shutdown, "failed to stop nccl_m2n executor")

        self._destination_plan = []
        self._destination_commit_token = ""
        self._dst_placements = []
        self._dst_mesh = None
        self._m2n = None
        if first_error is not None:
            raise first_error
