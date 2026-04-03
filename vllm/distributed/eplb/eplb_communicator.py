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

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
)

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

try:
    if not current_platform.is_rocm():
        from nixl._api import (
            nixl_agent as NixlWrapper,  # type: ignore[reportMissingImports]
        )
        from nixl._api import nixl_agent_config  # type: ignore[reportMissingImports]
    else:
        from rixl._api import (
            nixl_agent as NixlWrapper,  # type: ignore[reportMissingImports]
        )
        from rixl._api import nixl_agent_config  # type: ignore[reportMissingImports]

    nixl_available = True
except ImportError:
    NixlWrapper = None
    nixl_agent_config = None
    nixl_available = False


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers."""

    @abstractmethod
    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        pass

    @abstractmethod
    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

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

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._p2p_ops.append(
            P2POp(
                torch.distributed.isend,
                tensor,
                dst_rank,
                self._ep_group,
            )
        )

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
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

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ops.append(("send", tensor, dst_rank))

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
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
        expert_weights: Sequence[torch.Tensor],
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert expert_weights, "NixlEplbCommunicator requires non-empty expert_weights."
        if NixlWrapper is None:
            raise RuntimeError("NIXL/ RIXL is unavailable.")
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._world_size = cpu_group.size()
        self._rank = cpu_group.rank()
        self._send_tensors: dict[torch.dtype, list[list[torch.Tensor]]] = {}
        self._recv_tensors: dict[torch.dtype, list[list[torch.Tensor]]] = {}
        self._dtypes: list[torch.dtype] = []
        self._device = expert_weights[0].device
        for tensor in expert_weights:
            assert tensor.device == self._device, (
                "All local EPLB tensors are expected to be on the same device: "
                f"expected={self._device}, got={tensor.device}"
            )
            if tensor.dtype not in self._dtypes:
                self._dtypes.append(tensor.dtype)

        config = (
            nixl_agent_config(capture_telemetry=False)
            if nixl_agent_config is not None
            else None
        )
        self._nixl_wrapper = NixlWrapper(self._make_agent_name(), config)
        self._nixl_memory_type = "VRAM"
        self._registered_desc: object | None = None
        self._remote_agents: dict[int, str] = {}
        self._remote_send_meta: dict[torch.dtype, dict[int, tuple[int, int, int]]] = {}
        self._send_buffers: dict[torch.dtype, torch.Tensor] = {}
        self._recv_buffers: dict[torch.dtype, torch.Tensor] = {}
        self._peer_partition_numels: dict[torch.dtype, int] = {}
        self._cuda_device_id = int(self._device.index or 0)
        try:
            self._init_registered_buffers(expert_weights)
        except Exception as exc:
            raise RuntimeError("NIXL EPLB init failed: buffers") from exc
        try:
            self._init_remote_agents()
        except Exception as exc:
            raise RuntimeError("NIXL EPLB init failed: agents") from exc
        try:
            self._exchange_remote_send_meta()
        except Exception as exc:
            raise RuntimeError("NIXL EPLB init failed: send meta") from exc
        self._log_initialized()

    def _make_agent_name(self) -> str:
        """Build a deployment-unique nixl agent name."""
        pp_size = get_pp_group().world_size
        pp_suffix = f"-pp{get_pp_group().rank_in_group}" if pp_size > 1 else ""
        uid = uuid.uuid4().hex[:8]
        return f"eplb-{self._rank}{pp_suffix}-{uid}"

    def _get_peer_buckets(
        self,
        bucket_map: dict[torch.dtype, list[list[torch.Tensor]]],
        dtype: torch.dtype,
    ) -> list[list[torch.Tensor]]:
        peer_buckets = bucket_map.get(dtype)
        if peer_buckets is None:
            peer_buckets = [[] for _ in range(self._world_size)]
            bucket_map[dtype] = peer_buckets
        return peer_buckets

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        assert dst_rank != self._rank, (
            "EPLB communicator should not enqueue same-rank sends: "
            f"rank={self._rank}, dst_rank={dst_rank}"
        )
        self._get_peer_buckets(self._send_tensors, tensor.dtype)[dst_rank].append(
            tensor
        )

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        assert src_rank != self._rank, (
            "EPLB communicator should not enqueue same-rank recvs: "
            f"rank={self._rank}, src_rank={src_rank}"
        )
        self._get_peer_buckets(self._recv_tensors, tensor.dtype)[src_rank].append(
            tensor
        )

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
        buffers_to_register: list[torch.Tensor] = []
        for dtype in self._dtypes:
            max_peer_partition_numel = max(
                sum(t.numel() for t in expert_weights if t.dtype == dtype), 1
            )
            total_numel = max_peer_partition_numel * self._world_size
            send_buffer = torch.empty(total_numel, device=self._device, dtype=dtype)
            recv_buffer = torch.empty(total_numel, device=self._device, dtype=dtype)
            self._send_buffers[dtype] = send_buffer
            self._recv_buffers[dtype] = recv_buffer
            self._peer_partition_numels[dtype] = max_peer_partition_numel
            buffers_to_register.extend([send_buffer, recv_buffer])

        descs = self._nixl_wrapper.get_reg_descs(buffers_to_register)
        self._nixl_wrapper.register_memory(descs)
        self._registered_desc = descs

    def _exchange_remote_send_meta(self) -> None:
        """Exchange send-buffer metadata so each rank can build dynamic
        descriptors at execute time."""
        local_meta: dict[torch.dtype, tuple[int, int, int]] = {}
        for dtype in self._dtypes:
            send_buffer = self._send_buffers[dtype]
            peer_partition_numel = self._peer_partition_numels[dtype]
            local_meta[dtype] = (
                send_buffer.data_ptr(),
                peer_partition_numel * send_buffer.element_size(),
                self._cuda_device_id,
            )
        gathered_meta: list[dict[torch.dtype, tuple[int, int, int]] | None] = [
            None
        ] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        for dtype in self._dtypes:
            self._remote_send_meta[dtype] = {}
            for peer in self._remote_agents:
                peer_meta = gathered_meta[peer]
                assert peer_meta is not None
                self._remote_send_meta[dtype][peer] = peer_meta[dtype]

    @staticmethod
    def _pack_send_buffer(
        peer_tensors: list[torch.Tensor],
        send_buffer: torch.Tensor,
        peer_partition_start: int,
    ) -> None:
        offset = peer_partition_start
        for tensor in peer_tensors:
            flat = tensor.reshape(-1)
            if flat.numel() == 0:
                continue
            send_buffer[offset : offset + flat.numel()].copy_(flat, non_blocking=True)
            offset += flat.numel()

    @staticmethod
    def _unpack_recv_buffer(
        recv_buffer: torch.Tensor,
        peer_tensors: list[torch.Tensor],
        peer_partition_start: int,
    ) -> None:
        offset = peer_partition_start
        for tensor in peer_tensors:
            flat = tensor.reshape(-1)
            if flat.numel() == 0:
                continue
            flat.copy_(
                recv_buffer[offset : offset + flat.numel()],
                non_blocking=True,
            )
            offset += flat.numel()

    def _release_nixl_handles(
        self,
        xfer_handles: list[int],
        dlist_handles: list[int],
    ) -> None:
        """Best-effort cleanup of NIXL handles on exception paths."""
        for h in xfer_handles:
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_xfer_handle(h)
        for h in dlist_handles:
            with contextlib.suppress(Exception):
                self._nixl_wrapper.release_dlist_handle(h)

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
                self._nixl_wrapper.release_xfer_handle(handle)
                pending.remove(handle)
            if pending:
                time.sleep(0.0005)

    def execute(self) -> None:
        xfer_handles: list[int] = []
        dlist_handles: list[int] = []
        try:
            # Phase 1: pack send buffers.
            with torch.cuda.stream(self._cuda_stream):
                for dtype in self._dtypes:
                    send_per_peer = self._send_tensors.get(
                        dtype, [[] for _ in range(self._world_size)]
                    )
                    peer_partition_numel = self._peer_partition_numels[dtype]
                    send_buffer = self._send_buffers[dtype]

                    for dst, peer_tensors in enumerate(send_per_peer):
                        count = sum(t.numel() for t in peer_tensors)
                        if count == 0:
                            continue
                        if count > peer_partition_numel:
                            raise RuntimeError(
                                "NIXL EPLB send overflow for dtype "
                                f"{dtype}: peer={dst}, required={count}, "
                                f"capacity={peer_partition_numel}"
                            )
                        self._pack_send_buffer(
                            peer_tensors, send_buffer, dst * peer_partition_numel
                        )

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

            # Phase 2: create exact-size descriptors, transfer, unpack.
            for dtype in self._dtypes:
                recv_per_peer = self._recv_tensors.get(
                    dtype, [[] for _ in range(self._world_size)]
                )
                peer_partition_numel = self._peer_partition_numels[dtype]
                recv_buffer = self._recv_buffers[dtype]
                element_size = recv_buffer.element_size()
                partition_bytes = peer_partition_numel * element_size
                recv_base = recv_buffer.data_ptr()

                for src, peer_tensors in enumerate(recv_per_peer):
                    if not peer_tensors:
                        continue
                    recv_count = sum(t.numel() for t in peer_tensors)
                    actual_bytes = recv_count * element_size

                    # Local recv descriptor covering the
                    # received payload in this rank's recv buffer.
                    local_desc = self._nixl_wrapper.get_xfer_descs(
                        [
                            (
                                recv_base + src * partition_bytes,
                                actual_bytes,
                                self._cuda_device_id,
                            )
                        ],
                        self._nixl_memory_type,
                    )
                    local_handle = self._nixl_wrapper.prep_xfer_dlist(
                        "NIXL_INIT_AGENT",
                        local_desc,
                    )
                    dlist_handles.append(local_handle)

                    # Remote send descriptor pointing at the slice of
                    # the source rank's send buffer packed for us.
                    remote_base, remote_part_bytes, remote_dev = self._remote_send_meta[
                        dtype
                    ][src]
                    agent_name = self._remote_agents[src]
                    remote_desc = self._nixl_wrapper.get_xfer_descs(
                        [
                            (
                                remote_base + self._rank * remote_part_bytes,
                                actual_bytes,
                                remote_dev,
                            )
                        ],
                        self._nixl_memory_type,
                    )
                    remote_handle = self._nixl_wrapper.prep_xfer_dlist(
                        agent_name,
                        remote_desc,
                    )
                    dlist_handles.append(remote_handle)

                    # Initiate READ from the remote send buffer
                    # into the local recv buffer.
                    xfer_handle = self._nixl_wrapper.make_prepped_xfer(
                        "READ",
                        local_handle,
                        [0],
                        remote_handle,
                        [0],
                    )
                    self._nixl_wrapper.transfer(xfer_handle)
                    xfer_handles.append(xfer_handle)

                self._wait_for_all_transfers(xfer_handles)
                xfer_handles.clear()

                for h in dlist_handles:
                    self._nixl_wrapper.release_dlist_handle(h)
                dlist_handles.clear()

                with torch.cuda.stream(self._cuda_stream):
                    for src, peer_tensors in enumerate(recv_per_peer):
                        if not peer_tensors:
                            continue
                        self._unpack_recv_buffer(
                            recv_buffer,
                            peer_tensors,
                            src * peer_partition_numel,
                        )
        finally:
            self._release_nixl_handles(xfer_handles, dlist_handles)
            self._send_tensors.clear()
            self._recv_tensors.clear()

    def __del__(self) -> None:
        try:
            if self._registered_desc is not None:
                self._nixl_wrapper.deregister_memory(self._registered_desc)
            for agent_name in self._remote_agents.values():
                self._nixl_wrapper.remove_remote_agent(agent_name)
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

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ensure_group_started()
        self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._ensure_group_started()
        self._pynccl_comm.recv(tensor, src_rank, stream=self._cuda_stream)

    def execute(self) -> None:
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
                f"(got '{backend}'). torch_gloo is not supported with stateless groups."
            )
        if backend == "torch_nccl":
            logger.warning(
                "Stateless elastic EP requires PyNCCL backend. "
                "Forcing EPLB communicator to 'pynccl'."
            )
            backend = "pynccl"
        return _create_pynccl()

    if backend == "nixl":
        if not nixl_available:
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
