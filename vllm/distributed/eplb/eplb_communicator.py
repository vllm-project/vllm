# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EPLB communicator implementations and factory.
"""

import contextlib
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import timedelta

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
)

import vllm.distributed.nixl_utils as nixl_utils
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pp_group,
    is_local_first_rank,
)
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def has_nixl() -> bool:
    """Whether the optional NIXL / RIXL package is available."""
    return nixl_utils.NixlWrapper is not None


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers."""

    @abstractmethod
    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    def set_transfer_context(  # noqa: B027
        self, old_indices: np.ndarray, layer_idx: int
    ) -> None:
        """Pre-set layer context before add_recv calls.

        Default is a no-op; overridden by backends (e.g. NIXL) that need
        layer-level context to issue transfers inside add_recv.
        """

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        """Whether the profile path must run a dummy collective operation to reserve
        communication buffers."""
        return True

    def set_stream(self, cuda_stream: torch.cuda.Stream | None) -> None:
        self._cuda_stream = cuda_stream

    def _log_initialized(self) -> None:
        if is_local_first_rank():
            logger.info("Initialized EPLB communicator: %s.", self.__class__.__name__)


class TorchDistNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by torch.distributed isend/irecv."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._cuda_stream = cuda_stream
        self._p2p_ops: list[P2POp] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.isend,
                    tensor,
                    dst_rank,
                    self._ep_group,
                )
            )

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._p2p_ops.append(
                P2POp(
                    torch.distributed.irecv,
                    tensor,
                    src_rank,
                    self._ep_group,
                )
            )

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        try:
            with torch.cuda.stream(self._cuda_stream):
                reqs = batch_isend_irecv(self._p2p_ops)
                for req in reqs:
                    req.wait()
        finally:
            self._p2p_ops.clear()


class TorchDistGlooStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using gloo P2P with CPU staging."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._log_initialized()

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("send", tensor, dst_rank))

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        for tensor in tensors:
            self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []

        def build_ops() -> None:
            for op, tensor, peer_rank in self._ops:
                if op == "send":
                    cpu_tensor = tensor.to(device="cpu", non_blocking=True)
                    p2p_ops.append(
                        P2POp(
                            torch.distributed.isend,
                            cpu_tensor,
                            peer_rank,
                            self._cpu_group,
                        )
                    )
                    continue
                cpu_tensor = torch.empty_like(tensor, device="cpu")
                p2p_ops.append(
                    P2POp(
                        torch.distributed.irecv,
                        cpu_tensor,
                        peer_rank,
                        self._cpu_group,
                    )
                )
                recv_staging.append((tensor, cpu_tensor))

        try:
            with torch.cuda.stream(self._cuda_stream):
                build_ops()
        finally:
            self._ops.clear()

        # Wait for all D2H copies to finish
        # before issuing gloo batch_isend_irecv operations.
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

        if not recv_staging:
            return
        with torch.cuda.stream(self._cuda_stream):
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)


class NixlEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by NIXL READ transfers."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        all_expert_weights: Sequence[Sequence[torch.Tensor]],
        expert_buffer: Sequence[torch.Tensor],
    ) -> None:
        assert all_expert_weights, (
            "NixlEplbCommunicator requires non-empty all_expert_weights."
        )
        assert expert_buffer, "NixlEplbCommunicator requires non-empty expert_buffer."
        nixl_wrapper_cls = nixl_utils.NixlWrapper
        if nixl_wrapper_cls is None:
            raise RuntimeError("NIXL/ RIXL is unavailable.")

        self._cpu_group = cpu_group
        self._world_size = cpu_group.size()
        self._rank = cpu_group.rank()

        self._all_expert_weights = all_expert_weights
        self._expert_buffer = expert_buffer
        self._num_local_experts: int = all_expert_weights[0][0].shape[0]
        self._device = all_expert_weights[0][0].device

        for layer_tensors in all_expert_weights:
            for tensor in layer_tensors:
                assert tensor.is_contiguous(), (
                    "Expert weight tensors must be contiguous"
                )
                assert tensor.device == self._device, (
                    "All local EPLB tensors are expected to be on the same "
                    f"device: expected={self._device}, got={tensor.device}"
                )
        for tensor in expert_buffer:
            assert tensor.is_contiguous(), "expert_buffer tensors must be contiguous"

        # (local_dlist, remote_dlist, xfer_handle) for in-flight READs;
        # accumulated by add_recv, drained by execute.
        self._xfer_entries: list[tuple[int, int, int]] = []
        # Per-rank expert_id -> physical row; set by set_transfer_context.
        self._expert_to_src_row: list[dict[int, int]] | None = None
        self._layer_idx: int | None = None

        nixl_agent_config = nixl_utils.nixl_agent_config
        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)
        self._nixl_memory_type = "VRAM"
        # NIXL registration handles; deregistered in __del__.
        self._registered_descs: list[object] = []
        self._remote_agents: dict[int, str] = {}
        # peer -> (layer, tensor) -> (base_ptr, bytes_per_expert, dev_id).
        self._remote_send_meta: dict[
            int, dict[tuple[int, int], tuple[int, int, int]]
        ] = {}

        self._cuda_device_id = int(self._device.index or 0)
        self._init_step("buffers", self._init_registered_buffers)
        self._init_step("agents", self._init_remote_agents)
        self._init_step("send meta", self._exchange_remote_send_meta)
        self._log_initialized()

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False

    @staticmethod
    def _init_step(name: str, fn: object, *args: object, **kwargs: object) -> None:
        try:
            fn(*args, **kwargs)  # type: ignore[operator]
        except Exception as exc:
            raise RuntimeError(f"NIXL EPLB init failed: {name}") from exc

    def _make_agent_name(self) -> str:
        """Build a deployment-unique nixl agent name."""
        pp_size = get_pp_group().world_size
        pp_suffix = f"-pp{get_pp_group().rank_in_group}" if pp_size > 1 else ""
        uid = uuid.uuid4().hex[:8]
        return f"eplb-{self._rank}{pp_suffix}-{uid}"

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        # No-op: NIXL READ is receiver-initiated. The sender's expert
        # weights are pre-registered and always readable in-place.
        pass

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int) -> None:
        # Pre-compute expert_id -> src_row mapping for every rank so that
        # add_recv can immediately issue NIXL READs.
        assert not self._xfer_entries, (
            f"set_transfer_context() called with {len(self._xfer_entries)} "
            f"pending transfers from layer {self._layer_idx}; "
            f"execute() was not called after previous add_recv() calls"
        )
        self._layer_idx = layer_idx
        n = self._num_local_experts
        rank_experts = old_indices[: self._world_size * n].reshape(self._world_size, n)
        self._expert_to_src_row = [
            {int(eid): i for i, eid in enumerate(row) if eid != -1}
            for row in rank_experts
        ]

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        # Build NIXL descriptors and issue the RDMA READ immediately,
        # overlapping the transfer with the remaining Python loop in
        # move_to_buffer.
        assert self._expert_to_src_row is not None and self._layer_idx is not None, (
            "set_transfer_context() must be called before add_recv()"
        )
        src_row = self._expert_to_src_row[src_rank][expert_id]
        layer_idx = self._layer_idx

        local_descs: list[tuple[int, int, int]] = []
        remote_descs: list[tuple[int, int, int]] = []
        for t_idx, t in enumerate(tensors):
            send_base, send_stride, remote_dev = self._remote_send_meta[src_rank][
                (layer_idx, t_idx)
            ]
            assert t.nbytes == send_stride, (
                f"tensor {t_idx} size {t.nbytes} != remote stride {send_stride}"
            )
            local_descs.append(
                (
                    t.data_ptr(),
                    t.nbytes,
                    self._cuda_device_id,
                )
            )
            remote_descs.append(
                (
                    send_base + src_row * send_stride,
                    send_stride,
                    remote_dev,
                )
            )

        local_h, remote_h, xfer_h = self._create_peer_xfer(
            src_rank, local_descs, remote_descs
        )
        self._nixl_wrapper.transfer(xfer_h)
        self._xfer_entries.append((local_h, remote_h, xfer_h))

    def _init_remote_agents(self) -> None:
        local_metadata = self._nixl_wrapper.get_agent_metadata()
        gathered_metadata: list[bytes | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered_metadata, local_metadata, group=self._cpu_group
        )
        for peer in range(self._world_size):
            if peer == self._rank:
                continue
            peer_metadata = gathered_metadata[peer]
            assert peer_metadata is not None
            self._remote_agents[peer] = self._nixl_wrapper.add_remote_agent(
                peer_metadata
            )

    def _init_registered_buffers(self) -> None:
        all_tensors: list[torch.Tensor] = []
        for layer_tensors in self._all_expert_weights:
            all_tensors.extend(layer_tensors)
        all_tensors.extend(self._expert_buffer)

        descs = self._nixl_wrapper.get_reg_descs(all_tensors)
        self._nixl_wrapper.register_memory(descs)
        self._registered_descs.append(descs)

    def _exchange_remote_send_meta(self) -> None:
        """Exchange per-layer per-tensor metadata so receivers can compute
        remote RDMA addresses at transfer time."""
        local_meta: dict[tuple[int, int], tuple[int, int, int]] = {}
        for layer_idx, layer_tensors in enumerate(self._all_expert_weights):
            for t_idx, t in enumerate(layer_tensors):
                nbytes_per_expert = t.nbytes // self._num_local_experts
                local_meta[(layer_idx, t_idx)] = (
                    t.data_ptr(),
                    nbytes_per_expert,
                    self._cuda_device_id,
                )

        # Per-rank map: (layer_idx, tensor_idx) -> (base_ptr, bytes_per_expert, dev_id).
        # add_recv uses base_ptr + src_row * bytes_per_expert to compute
        # the remote RDMA address for each expert.
        gathered_meta: list[dict[tuple[int, int], tuple[int, int, int]] | None] = [
            None
        ] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        local_keys = set(local_meta.keys())
        for peer in self._remote_agents:
            peer_meta = gathered_meta[peer]
            assert peer_meta is not None
            peer_keys = set(peer_meta.keys())
            if peer_keys != local_keys:
                raise RuntimeError(
                    f"NIXL EPLB metadata key mismatch with rank {peer}: "
                    f"local={sorted(local_keys)}, peer={sorted(peer_keys)}"
                )
            for key in local_keys:
                _, local_stride, _ = local_meta[key]
                _, peer_stride, _ = peer_meta[key]
                if local_stride != peer_stride:
                    raise RuntimeError(
                        f"NIXL EPLB nbytes_per_expert mismatch for {key} "
                        f"with rank {peer}: "
                        f"local={local_stride}, peer={peer_stride}"
                    )
            self._remote_send_meta[peer] = peer_meta

    def _wait_for_all_transfers(self, handles: list[int]) -> None:
        pending = set(handles)
        while pending:
            completed: list[int] = []
            for handle in pending:
                state = self._nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    completed.append(handle)
                    continue
                if state != "PROC":
                    raise RuntimeError(f"NIXL transfer failed with state={state}")
            for handle in completed:
                pending.remove(handle)
            if pending:
                time.sleep(0.0005)

    def _create_peer_xfer(
        self,
        src: int,
        local_descs: list[tuple[int, int, int]],
        remote_descs: list[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        """Create a batched xfer for multiple descriptors from one peer.

        Each element in *local_descs* / *remote_descs* is an
        ``(address, size, device_id)`` tuple.

        Returns ``(local_dlist, remote_dlist, xfer_handle)``.
        """
        local_desc = self._nixl_wrapper.get_xfer_descs(
            local_descs, self._nixl_memory_type
        )
        local_handle = self._nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_desc,
        )

        remote_desc = self._nixl_wrapper.get_xfer_descs(
            remote_descs, self._nixl_memory_type
        )
        remote_handle = self._nixl_wrapper.prep_xfer_dlist(
            self._remote_agents[src],
            remote_desc,
        )

        indices = list(range(len(local_descs)))
        xfer_handle = self._nixl_wrapper.make_prepped_xfer(
            "READ",
            local_handle,
            indices,
            remote_handle,
            indices,
        )
        return (local_handle, remote_handle, xfer_handle)

    def execute(self) -> None:
        assert self._layer_idx is not None or not self._xfer_entries, (
            "set_transfer_context() must be called before execute() "
            "if any add_recv() calls were made"
        )
        try:
            self._wait_for_all_transfers([x[2] for x in self._xfer_entries])

            # Post-READ barrier.
            # Correctness fence for zero-copy: prevents overwrite-while-
            # remote-read race.
            torch.distributed.monitored_barrier(
                group=self._cpu_group,
                timeout=timedelta(minutes=5),
            )
        finally:
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
            self._xfer_entries.clear()
            self._expert_to_src_row = None
            self._layer_idx = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            for local_h, remote_h, xfer_h in self._xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
        with contextlib.suppress(Exception):
            for descs in self._registered_descs:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.deregister_memory(descs)
            self._registered_descs.clear()
        with contextlib.suppress(Exception):
            for agent_name in self._remote_agents.values():
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.remove_remote_agent(agent_name)
            self._remote_agents.clear()


class PyNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyNcclCommunicator using ncclSend/ncclRecv."""

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._pynccl_comm = pynccl_comm
        self._cuda_stream = cuda_stream
        self._group_started = False
        self._log_initialized()

    def _ensure_group_started(self) -> None:
        if not self._group_started:
            self._pynccl_comm.group_start()
            self._group_started = True

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,  # unused by this backend
    ) -> None:
        self._ensure_group_started()
        for tensor in tensors:
            self._pynccl_comm.recv(tensor, src_rank, stream=self._cuda_stream)

    def execute(self) -> None:
        if self._group_started:
            self._pynccl_comm.group_end()
            self._group_started = False


def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str | None,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_buffer: Sequence[torch.Tensor],
) -> EplbCommunicator:
    """Create an EPLB communicator for the given backend.

    Args:
        group_coordinator: Process-group coordinator that provides the
            device and CPU communication groups.
        backend: Communicator backend name (``"torch_nccl"``,
            ``"torch_gloo"``, ``"pynccl"``, or ``"nixl"``).
            Falls back to ``"torch_nccl"`` when *None*.
            Stateless (elastic EP) groups only support ``"torch_nccl"``
            and ``"pynccl"``; ``"torch_nccl"`` is silently promoted to
            ``"pynccl"`` in that case.  When tensors reside on CPU,
            ``"torch_gloo"`` or ``"torch_nccl"`` are used via the CPU
            process group.
        expert_weights: Expert weight tensors for *all* MoE layers.
            Shape ``(num_layers)(num_tensors_per_layer)``.
            NixlEplbCommunicator registers all layers with NIXL for
            zero-copy RDMA reads.
        expert_buffer: Pre-allocated receive buffer tensors (one per
            weight tensor in a single layer).
    """
    if backend is None:
        backend = "torch_nccl"

    first_layer = expert_weights[0] if expert_weights else []
    tensor_device_type = first_layer[0].device.type if first_layer else "cpu"
    torch_group = (
        group_coordinator.cpu_group
        if tensor_device_type == "cpu"
        else group_coordinator.device_group
    )

    def _create_pynccl() -> EplbCommunicator:
        if tensor_device_type == "cpu":
            raise RuntimeError(
                "EPLB communicator 'pynccl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        unsupported_dtypes = sorted(
            {
                tensor.dtype
                for tensor in first_layer
                if not ncclDataTypeEnum.supports_torch_dtype(tensor.dtype)
            },
            key=str,
        )
        if unsupported_dtypes:
            raise RuntimeError(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtypes: "
                f"({', '.join(str(dtype) for dtype in unsupported_dtypes)})."
            )

        device_comm = group_coordinator.device_communicator
        pynccl_comm = (
            getattr(device_comm, "pynccl_comm", None)
            if device_comm is not None
            else None
        )
        if pynccl_comm is None or pynccl_comm.disabled or not pynccl_comm.available:
            raise RuntimeError("EPLB communicator 'pynccl' requested but unavailable.")
        try:
            return PyNcclEplbCommunicator(pynccl_comm=pynccl_comm)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PyNcclEplbCommunicator ({exc})."
            ) from exc

    is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
    if is_stateless:
        if backend not in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"Elastic EP requires 'torch_nccl' or 'pynccl' EPLB communicator "
                f"(got '{backend}')."
            )
        if backend == "torch_nccl":
            logger.warning(
                "Stateless elastic EP requires PyNCCL backend. "
                "Forcing EPLB communicator to 'pynccl'."
            )
            backend = "pynccl"
        return _create_pynccl()

    if backend == "nixl":
        if not has_nixl():
            raise RuntimeError(
                "EPLB communicator 'nixl' requested but NIXL is unavailable."
            )
        if not (current_platform.is_cuda_alike() and tensor_device_type != "cpu"):
            raise RuntimeError(
                "EPLB communicator 'nixl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        try:
            return NixlEplbCommunicator(
                cpu_group=group_coordinator.cpu_group,
                all_expert_weights=expert_weights,
                expert_buffer=expert_buffer,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize NixlEplbCommunicator ({exc})."
            ) from exc
    elif backend == "torch_gloo":
        return TorchDistGlooStagedEplbCommunicator(
            cpu_group=group_coordinator.cpu_group,
        )
    elif backend == "torch_nccl":
        return TorchDistNcclEplbCommunicator(ep_group=torch_group)
    elif backend == "pynccl":
        return _create_pynccl()
    raise ValueError(f"Unknown EPLB communicator backend: {backend}")
