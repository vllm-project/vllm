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
    def execute(self, old_indices: np.ndarray | None = None) -> None:
        pass

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

    def execute(self, old_indices: np.ndarray | None = None) -> None:
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

    def execute(self, old_indices: np.ndarray | None = None) -> None:
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
        expert_weights: Sequence[torch.Tensor],
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert expert_weights, "NixlEplbCommunicator requires non-empty expert_weights."
        nixl_wrapper_cls = nixl_utils.NixlWrapper
        if nixl_wrapper_cls is None:
            raise RuntimeError("NIXL/ RIXL is unavailable.")
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._world_size = cpu_group.size()
        self._rank = cpu_group.rank()
        # expert_id -> weight tensors to pack into the send buffer.
        self._expert_send_map: dict[int, list[torch.Tensor]] = {}
        # src_rank -> expert_id -> weight tensors to unpack after transfer.
        self._recv_map: dict[int, dict[int, list[torch.Tensor]]] = {}
        self._num_local_experts: int = expert_weights[0].shape[0]
        self._device = expert_weights[0].device
        for tensor in expert_weights:
            assert tensor.device == self._device, (
                "All local EPLB tensors are expected to be on the same device: "
                f"expected={self._device}, got={tensor.device}"
            )

        nixl_agent_config = nixl_utils.nixl_agent_config
        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._nixl_wrapper = nixl_wrapper_cls(self._make_agent_name(), config)
        self._nixl_memory_type = "VRAM"
        self._registered_desc: object | None = None
        self._remote_agents: dict[int, str] = {}
        self._remote_send_meta: dict[int, tuple[int, int]] = {}
        self._send_buffer: torch.Tensor = torch.empty(0)
        self._recv_buffer: torch.Tensor = torch.empty(0)
        self._expert_bytes: int = 0

        self._cuda_device_id = int(self._device.index or 0)
        self._init_step("buffers", self._init_registered_buffers, expert_weights)
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
        assert dst_rank != self._rank, (
            "EPLB communicator should not enqueue same-rank sends: "
            f"rank={self._rank}, dst_rank={dst_rank}"
        )
        # An expert sent to multiple peers is packed only once; skip duplicates.
        if expert_id not in self._expert_send_map:
            self._expert_send_map[expert_id] = tensors

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        assert src_rank != self._rank, (
            "EPLB communicator should not enqueue same-rank recvs: "
            f"rank={self._rank}, src_rank={src_rank}"
        )
        recv_experts = self._recv_map.setdefault(src_rank, {})
        if expert_id not in recv_experts:
            recv_experts[expert_id] = tensors

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

    def _init_registered_buffers(self, expert_weights: Sequence[torch.Tensor]) -> None:
        total_bytes = max(sum(t.nbytes for t in expert_weights), 1)
        assert total_bytes % self._num_local_experts == 0, (
            f"Number of bytes in moe layer {total_bytes} is not divisible "
            f"by number of local experts {self._num_local_experts}"
        )
        self._expert_bytes = total_bytes // self._num_local_experts

        self._send_buffer = torch.empty(
            total_bytes, device=self._device, dtype=torch.uint8
        )
        self._recv_buffer = torch.empty(
            total_bytes, device=self._device, dtype=torch.uint8
        )

        descs = self._nixl_wrapper.get_reg_descs([self._send_buffer, self._recv_buffer])
        self._nixl_wrapper.register_memory(descs)
        self._registered_desc = descs

    def _exchange_remote_send_meta(self) -> None:
        """Exchange send-buffer metadata so each rank can build dynamic
        descriptors at execute time."""
        local_meta: tuple[int, int] = (
            self._send_buffer.data_ptr(),
            self._cuda_device_id,
        )
        gathered_meta: list[tuple[int, int] | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        for peer in self._remote_agents:
            peer_meta = gathered_meta[peer]
            assert peer_meta is not None
            self._remote_send_meta[peer] = peer_meta

    @staticmethod
    def _pack_send_buffer(
        in_tensors: list[torch.Tensor],
        send_buffer: torch.Tensor,
        byte_offset: int,
    ) -> None:
        for tensor in in_tensors:
            raw = tensor.reshape(-1).view(torch.uint8)
            if raw.numel() == 0:
                continue
            send_buffer[byte_offset : byte_offset + raw.numel()].copy_(
                raw, non_blocking=True
            )
            byte_offset += raw.numel()

    @staticmethod
    def _unpack_recv_buffer(
        recv_buffer: torch.Tensor,
        out_tensors: list[torch.Tensor],
        byte_offset: int,
    ) -> None:
        for tensor in out_tensors:
            num_bytes = tensor.numel() * tensor.element_size()
            if num_bytes == 0:
                continue
            tensor.reshape(-1).view(torch.uint8).copy_(
                recv_buffer[byte_offset : byte_offset + num_bytes],
                non_blocking=True,
            )
            byte_offset += num_bytes

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

    def execute(self, old_indices: np.ndarray | None = None) -> None:
        assert old_indices is not None, (
            "NixlEplbCommunicator.execute requires old_indices"
        )

        xfer_entries: list[tuple[int, int, int]] = []
        try:
            n = self._num_local_experts
            rank_experts = old_indices[: self._world_size * n].reshape(
                self._world_size, n
            )
            # Build expert_id -> send slot mapping per rank.
            expert_to_send_slot: list[dict[int, int]] = [
                {int(eid): i for i, eid in enumerate(row) if eid != -1}
                for row in rank_experts
            ]

            # Phase 1: pack each expert at its slot offset in the send buffer.
            with torch.cuda.stream(self._cuda_stream):
                for expert_id, tensors in self._expert_send_map.items():
                    slot = expert_to_send_slot[self._rank][expert_id]
                    byte_offset = slot * self._expert_bytes
                    self._pack_send_buffer(tensors, self._send_buffer, byte_offset)

            # Ensure all packed data is visible in device memory before pulls.
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()
            else:
                torch.cuda.current_stream().synchronize()
            # READ is receiver-initiated; synchronize all ranks before transfer.
            # We use monitored_barrier so a rank that crashes or exits early
            # produces a diagnostic timeout instead of a silent hang.
            torch.distributed.monitored_barrier(
                group=self._cpu_group,
                timeout=timedelta(minutes=5),
            )

            # Phase 2: issue one batched READ per peer.
            recv_offsets: dict[tuple[int, int], int] = {}
            recv_offset = 0
            recv_base = self._recv_buffer.data_ptr()
            for src in range(self._world_size):
                if src == self._rank:
                    continue
                recv_experts = self._recv_map.get(src)
                if not recv_experts:
                    continue
                expert_ids = list(recv_experts.keys())
                remote_base, remote_dev = self._remote_send_meta[src]
                local_descs: list[tuple[int, int, int]] = []
                remote_descs: list[tuple[int, int, int]] = []
                for expert_id in expert_ids:
                    slot = expert_to_send_slot[src][expert_id]
                    remote_off = slot * self._expert_bytes
                    recv_offsets[(src, expert_id)] = recv_offset
                    local_descs.append(
                        (
                            recv_base + recv_offset,
                            self._expert_bytes,
                            self._cuda_device_id,
                        )
                    )
                    remote_descs.append(
                        (remote_base + remote_off, self._expert_bytes, remote_dev)
                    )
                    recv_offset += self._expert_bytes
                    assert recv_offset <= self._recv_buffer.nbytes
                local_h, remote_h, xfer_h = self._create_peer_xfer(
                    src, local_descs, remote_descs
                )
                self._nixl_wrapper.transfer(xfer_h)
                xfer_entries.append((local_h, remote_h, xfer_h))

            # Phase 3: wait for all in-flight transfers, then unpack.
            self._wait_for_all_transfers([x[2] for x in xfer_entries])

            with torch.cuda.stream(self._cuda_stream):
                for (src, expert_id), offset in recv_offsets.items():
                    self._unpack_recv_buffer(
                        self._recv_buffer,
                        self._recv_map[src][expert_id],
                        offset,
                    )
        finally:
            for local_h, remote_h, xfer_h in xfer_entries:
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_xfer_handle(xfer_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(local_h)
                with contextlib.suppress(Exception):
                    self._nixl_wrapper.release_dlist_handle(remote_h)
            self._expert_send_map.clear()
            self._recv_map.clear()

    def __del__(self) -> None:
        try:
            if self._registered_desc is not None:
                self._nixl_wrapper.deregister_memory(self._registered_desc)
                self._registered_desc = None
            for agent_name in self._remote_agents.values():
                self._nixl_wrapper.remove_remote_agent(agent_name)
            self._remote_agents.clear()
        except Exception as e:
            logger.warning("Error during NixlEplbCommunicator cleanup: %s", e)


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

    def execute(self, old_indices: np.ndarray | None = None) -> None:
        if self._group_started:
            self._pynccl_comm.group_end()
            self._group_started = False


def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str | None,
    expert_weights: Sequence[torch.Tensor],
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
        expert_weights: Expert weight tensors from *one* MoE layer.
            NixlEplbCommunicator pre-allocates send/recv buffers sized
            to this layer, so all other MoE layers must have the same
            tensor count, shapes, and dtypes.
    """
    # Keep a safe default for callers that have not resolved communicator yet.
    if backend is None:
        backend = "torch_nccl"

    tensor_device_type = expert_weights[0].device.type if expert_weights else "cpu"
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
                for tensor in expert_weights
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
                expert_weights=expert_weights,
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
