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
from vllm.distributed.nixl_utils import (
    NixlWrapper,
    nixl_agent_config,
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
    return NixlWrapper is not None


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
        self._remote_send_meta: dict[int, tuple[int, int, int]] = {}
        self._send_buffer: torch.Tensor = torch.empty(0)
        self._recv_buffer: torch.Tensor = torch.empty(0)
        self._peer_partition_bytes: int = 0
        self._dtype_max_bytes: dict[torch.dtype, int] = {}
        self._cuda_device_id = int(self._device.index or 0)
        self._xfer_cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
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
        total_max_bytes = 0
        for dtype in self._dtypes:
            max_numel = max(
                sum(t.numel() for t in expert_weights if t.dtype == dtype), 1
            )
            max_bytes = max_numel * dtype.itemsize
            self._dtype_max_bytes[dtype] = max_bytes
            total_max_bytes += max_bytes

        self._peer_partition_bytes = total_max_bytes

        # The send buffer needs world_size partitions because remote peers
        # READ from fixed offsets (rank * partition_bytes).
        # This allocates world_size * partition_bytes
        # which can cause OOM on large models.
        # TODO(ilmarkov): shrink to const * partition_bytes and execute
        # communication in multiple steps dealing with the worst case.
        send_total_bytes = self._peer_partition_bytes * self._world_size

        self._send_buffer = torch.empty(
            send_total_bytes, device=self._device, dtype=torch.uint8
        )
        self._recv_buffer = torch.empty(
            self._peer_partition_bytes, device=self._device, dtype=torch.uint8
        )

        descs = self._nixl_wrapper.get_reg_descs([self._send_buffer, self._recv_buffer])
        self._nixl_wrapper.register_memory(descs)
        self._registered_desc = descs

    def _exchange_remote_send_meta(self) -> None:
        """Exchange send-buffer metadata so each rank can build dynamic
        descriptors at execute time."""
        local_meta: tuple[int, int, int] = (
            self._send_buffer.data_ptr(),
            self._peer_partition_bytes,
            self._cuda_device_id,
        )
        gathered_meta: list[tuple[int, int, int] | None] = [None] * self._world_size
        torch.distributed.all_gather_object(
            gathered_meta, local_meta, group=self._cpu_group
        )

        for peer in self._remote_agents:
            peer_meta = gathered_meta[peer]
            assert peer_meta is not None
            self._remote_send_meta[peer] = peer_meta

    @staticmethod
    def _pack_send_buffer(
        peer_tensors: list[torch.Tensor],
        send_buffer: torch.Tensor,
        byte_offset: int,
    ) -> int:
        """
        Returns the byte offset after the last written byte.
        """
        for tensor in peer_tensors:
            raw = tensor.reshape(-1).view(torch.uint8)
            if raw.numel() == 0:
                continue
            send_buffer[byte_offset : byte_offset + raw.numel()].copy_(
                raw, non_blocking=True
            )
            byte_offset += raw.numel()
        return byte_offset

    @staticmethod
    def _unpack_recv_buffer(
        recv_buffer: torch.Tensor,
        peer_tensors: list[torch.Tensor],
        byte_offset: int,
    ) -> int:
        """
        Returns the byte offset after the last read byte.
        """
        for tensor in peer_tensors:
            num_bytes = tensor.numel() * tensor.element_size()
            if num_bytes == 0:
                continue
            tensor.reshape(-1).view(torch.uint8).copy_(
                recv_buffer[byte_offset : byte_offset + num_bytes],
                non_blocking=True,
            )
            byte_offset += num_bytes
        return byte_offset

    def _release_all_cached_handles(self) -> None:
        """Best-effort release of every cached dlist and xfer handle."""
        for local_dlist, remote_dlist, xfer in self._xfer_cache.values():
            for release_fn, handle in (
                (self._nixl_wrapper.release_xfer_handle, xfer),
                (self._nixl_wrapper.release_dlist_handle, local_dlist),
                (self._nixl_wrapper.release_dlist_handle, remote_dlist),
            ):
                with contextlib.suppress(Exception):
                    release_fn(handle)
        self._xfer_cache.clear()

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

    def _get_or_create_xfer(self, src: int, total_bytes: int, recv_offset: int) -> int:
        """Return a cached xfer handle or create and cache a new one."""
        key = (src, total_bytes, recv_offset)
        cached = self._xfer_cache.get(key)
        if cached is not None:
            return cached[2]

        recv_base = self._recv_buffer.data_ptr()
        local_desc = self._nixl_wrapper.get_xfer_descs(
            [
                (
                    recv_base + recv_offset,
                    total_bytes,
                    self._cuda_device_id,
                )
            ],
            self._nixl_memory_type,
        )
        local_handle = self._nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT",
            local_desc,
        )

        remote_base, remote_part_bytes, remote_dev = self._remote_send_meta[src]
        agent_name = self._remote_agents[src]
        remote_desc = self._nixl_wrapper.get_xfer_descs(
            [
                (
                    remote_base + self._rank * remote_part_bytes,
                    total_bytes,
                    remote_dev,
                )
            ],
            self._nixl_memory_type,
        )
        remote_handle = self._nixl_wrapper.prep_xfer_dlist(
            agent_name,
            remote_desc,
        )

        xfer_handle = self._nixl_wrapper.make_prepped_xfer(
            "READ",
            local_handle,
            [0],
            remote_handle,
            [0],
        )
        self._xfer_cache[key] = (local_handle, remote_handle, xfer_handle)
        return xfer_handle

    def execute(self) -> None:
        xfer_handles: list[int] = []
        try:
            # Phase 1: pack send buffers.
            with torch.cuda.stream(self._cuda_stream):
                for dst in range(self._world_size):
                    byte_offset = dst * self._peer_partition_bytes
                    for dtype in self._dtypes:
                        peer_tensors = self._send_tensors.get(
                            dtype, [[] for _ in range(self._world_size)]
                        )[dst]
                        actual_bytes = sum(
                            t.numel() * t.element_size() for t in peer_tensors
                        )
                        if actual_bytes > self._dtype_max_bytes[dtype]:
                            raise RuntimeError(
                                "NIXL EPLB send overflow for dtype "
                                f"{dtype}: peer={dst}, "
                                f"required={actual_bytes}, "
                                f"capacity={self._dtype_max_bytes[dtype]}"
                            )
                        byte_offset = self._pack_send_buffer(
                            peer_tensors,
                            self._send_buffer,
                            byte_offset,
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

            # Phase 2: look up or create descriptors and issue all READs.
            # Data from all peers is packed sequentially into the single
            # partition-sized recv buffer at running offsets.
            recv_offsets: dict[int, int] = {}
            recv_offset = 0
            for src in range(self._world_size):
                if src == self._rank:
                    continue
                actual_total_bytes = 0
                for dtype in self._dtypes:
                    peer_tensors = self._recv_tensors.get(
                        dtype, [[] for _ in range(self._world_size)]
                    )[src]
                    actual_total_bytes += sum(
                        t.numel() * t.element_size() for t in peer_tensors
                    )
                if actual_total_bytes == 0:
                    continue

                recv_offsets[src] = recv_offset
                xfer_handle = self._get_or_create_xfer(
                    src, actual_total_bytes, recv_offset
                )
                self._nixl_wrapper.transfer(xfer_handle)
                xfer_handles.append(xfer_handle)
                recv_offset += actual_total_bytes

            # Phase 3: single wait for all in-flight transfers, then unpack.
            self._wait_for_all_transfers(xfer_handles)

            with torch.cuda.stream(self._cuda_stream):
                for src, offset in recv_offsets.items():
                    byte_offset = offset
                    for dtype in self._dtypes:
                        peer_tensors = self._recv_tensors.get(
                            dtype, [[] for _ in range(self._world_size)]
                        )[src]
                        byte_offset = self._unpack_recv_buffer(
                            self._recv_buffer,
                            peer_tensors,
                            byte_offset,
                        )
        except Exception:
            self._release_all_cached_handles()
            raise
        finally:
            self._send_tensors.clear()
            self._recv_tensors.clear()

    def __del__(self) -> None:
        try:
            self._release_all_cached_handles()
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
