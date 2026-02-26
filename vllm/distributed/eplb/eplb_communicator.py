# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EPLB communicator implementations and factory.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
    get_global_rank,
)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False


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


class TorchDistributedEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by torch.distributed isend/irecv."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._cuda_stream = cuda_stream
        self._p2p_ops: list[P2POp] = []
        self._rank_to_global = {
            rank: get_global_rank(ep_group, rank) for rank in range(ep_group.size())
        }

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._p2p_ops.append(
            P2POp(
                torch.distributed.isend,
                tensor,
                self._rank_to_global[dst_rank],
                self._ep_group,
            )
        )

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._p2p_ops.append(
            P2POp(
                torch.distributed.irecv,
                tensor,
                self._rank_to_global[src_rank],
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


class GlooCpuStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using gloo P2P with CPU staging."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._rank_to_global = {
            rank: get_global_rank(cpu_group, rank) for rank in range(cpu_group.size())
        }

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ops.append(("send", tensor, dst_rank))

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        try:
            p2p_ops: list[P2POp] = []
            recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []

            def build_ops() -> None:
                for op, tensor, peer_rank in self._ops:
                    peer_global_rank = self._rank_to_global[peer_rank]
                    if op == "send":
                        cpu_tensor = tensor.to(device="cpu", non_blocking=True)
                        p2p_ops.append(
                            P2POp(
                                torch.distributed.isend,
                                cpu_tensor,
                                peer_global_rank,
                                self._cpu_group,
                            )
                        )
                        continue
                    cpu_tensor = torch.empty_like(tensor, device="cpu")
                    p2p_ops.append(
                        P2POp(
                            torch.distributed.irecv,
                            cpu_tensor,
                            peer_global_rank,
                            self._cpu_group,
                        )
                    )
                    recv_staging.append((tensor, cpu_tensor))

            with torch.cuda.stream(self._cuda_stream):
                build_ops()
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
                    dst_tensor.copy_(
                        cpu_tensor, non_blocking=self._cuda_stream is not None
                    )
        finally:
            self._ops.clear()


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


class SymmMemEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by symmetric-memory all_to_all_vdev."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        expert_weights: Sequence[torch.Tensor],
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._cuda_stream = cuda_stream
        self._world_size = ep_group.size()
        self._send_tensors: dict[torch.dtype, list[list[torch.Tensor]]] = {}
        self._recv_tensors: dict[torch.dtype, list[list[torch.Tensor]]] = {}
        self._ordered_dtypes: list[torch.dtype] = []
        self._dtype_to_device: dict[torch.dtype, torch.device] = {}
        self._symm_send_buffers: dict[torch.dtype, torch.Tensor] = {}
        self._symm_recv_buffers: dict[torch.dtype, torch.Tensor] = {}
        self._symm_in_splits: torch.Tensor | None = None
        self._symm_out_splits_offsets: torch.Tensor | None = None
        # Keep rendezvous handles alive with their tensors.
        self._symm_send_handles: dict[torch.dtype, object] = {}
        self._symm_recv_handles: dict[torch.dtype, object] = {}
        self._symm_in_splits_handle: object | None = None
        self._symm_out_splits_offsets_handle: object | None = None
        for tensor in expert_weights:
            if tensor.dtype not in self._dtype_to_device:
                self._ordered_dtypes.append(tensor.dtype)
            self._dtype_to_device[tensor.dtype] = tensor.device
        self._set_nvshmem_backend()
        self._maybe_enable_symm_mem_groups_for_nvshmem(
            device=self._dtype_to_device[self._ordered_dtypes[0]]
        )
        for dtype in self._ordered_dtypes:
            local_dtype_numel = sum(
                tensor.numel() for tensor in expert_weights if tensor.dtype == dtype
            )
            # Worst-case send: all local experts sent to each peer.
            send_buffer_numel = max(local_dtype_numel * self._world_size, 1)
            # Worst-case recv: local-expert payload for this rank.
            recv_buffer_numel = max(local_dtype_numel, 1)
            self._ensure_symmetric_buffer(
                dtype=dtype,
                device=self._dtype_to_device[dtype],
                min_numel=send_buffer_numel,
                is_send=True,
            )
            self._ensure_symmetric_buffer(
                dtype=dtype,
                device=self._dtype_to_device[dtype],
                min_numel=recv_buffer_numel,
                is_send=False,
            )
        # Split metadata tensors are dtype-agnostic and can be shared across
        # all payload dtypes because we process dtype transfers sequentially.
        self._ensure_split_buffers(
            device=self._dtype_to_device[self._ordered_dtypes[0]]
        )

    @staticmethod
    def _set_nvshmem_backend() -> None:
        try:
            torch_symm_mem.set_backend("NVSHMEM")
        except Exception as exc:
            raise RuntimeError(
                "Failed to set symmetric-memory backend to NVSHMEM."
            ) from exc

    def _maybe_enable_symm_mem_groups_for_nvshmem(
        self, *, device: torch.device
    ) -> None:
        backend = torch_symm_mem.get_backend(device)
        if backend is None or str(backend).upper() != "NVSHMEM":
            return
        torch_symm_mem.enable_symm_mem_for_group(
            torch.distributed.group.WORLD.group_name
        )
        torch_symm_mem.enable_symm_mem_for_group(self._ep_group.group_name)

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
        self._get_peer_buckets(self._send_tensors, tensor.dtype)[dst_rank].append(
            tensor
        )

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._get_peer_buckets(self._recv_tensors, tensor.dtype)[src_rank].append(
            tensor
        )

    def _all_to_all_vdev(
        self,
        *,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        in_splits_list: list[int],
        out_splits_list: list[int],
    ) -> None:
        del out_splits_list
        assert self._symm_in_splits is not None
        assert self._symm_out_splits_offsets is not None
        in_splits = self._symm_in_splits
        out_splits_offsets = self._symm_out_splits_offsets

        in_splits.copy_(
            torch.tensor(in_splits_list, device=input_tensor.device, dtype=torch.int64),
            non_blocking=True,
        )

        torch.ops.symm_mem.all_to_all_vdev(
            input_tensor,
            output_tensor,
            in_splits,
            out_splits_offsets,
            self._ep_group.group_name,
        )

    @staticmethod
    def _pack_send_buffer(
        send_per_peer: list[list[torch.Tensor]],
        input_buffer: torch.Tensor,
        in_splits: list[int],
    ) -> None:
        assert len(in_splits) == len(send_per_peer)
        offset = 0
        for peer_rank, peer_tensors in enumerate(send_per_peer):
            split = in_splits[peer_rank]
            if split == 0:
                continue
            tensor_idx, elem_offset = 0, 0
            written = 0
            while written < split:
                flat = peer_tensors[tensor_idx].reshape(-1)
                available = flat.numel() - elem_offset
                to_copy = min(available, split - written)
                input_buffer[offset + written : offset + written + to_copy].copy_(
                    flat[elem_offset : elem_offset + to_copy], non_blocking=True
                )
                written += to_copy
                elem_offset += to_copy
                if elem_offset == flat.numel():
                    tensor_idx += 1
                    elem_offset = 0
            offset += split

    @staticmethod
    def _unpack_recv_buffer(
        output_buffer: torch.Tensor,
        recv_per_peer: list[list[torch.Tensor]],
        out_splits: list[int],
    ) -> None:
        assert len(out_splits) == len(recv_per_peer)
        offset = 0
        for peer_rank, peer_tensors in enumerate(recv_per_peer):
            split = out_splits[peer_rank]
            if split == 0:
                continue
            tensor_idx, elem_offset = 0, 0
            consumed = 0
            while consumed < split:
                flat = peer_tensors[tensor_idx].reshape(-1)
                available = flat.numel() - elem_offset
                to_copy = min(available, split - consumed)
                flat[elem_offset : elem_offset + to_copy].copy_(
                    output_buffer[offset + consumed : offset + consumed + to_copy],
                    non_blocking=True,
                )
                consumed += to_copy
                elem_offset += to_copy
                if elem_offset == flat.numel():
                    tensor_idx += 1
                    elem_offset = 0
            offset += split

    def _ensure_symmetric_buffer(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
        min_numel: int,
        is_send: bool,
    ) -> torch.Tensor:
        if is_send:
            buffers = self._symm_send_buffers
            handles = self._symm_send_handles
        else:
            buffers = self._symm_recv_buffers
            handles = self._symm_recv_handles
        buf = buffers.get(dtype)
        if buf is not None and buf.numel() >= min_numel:
            return buf
        # all_to_all_vdev requires symmetric tensors, so allocate through
        # torch symmetric-memory allocator and rendezvous collectively.
        buf = torch_symm_mem.empty(min_numel, device=device, dtype=dtype)
        handle = torch_symm_mem.rendezvous(buf, self._ep_group.group_name)
        buffers[dtype] = buf
        handles[dtype] = handle
        return buf

    def _ensure_split_buffers(self, *, device: torch.device) -> None:
        if self._symm_in_splits is None:
            in_splits = torch_symm_mem.empty(
                self._world_size, device=device, dtype=torch.int64
            )
            self._symm_in_splits = in_splits
            self._symm_in_splits_handle = torch_symm_mem.rendezvous(
                in_splits, self._ep_group.group_name
            )
        if self._symm_out_splits_offsets is None:
            out_splits_offsets = torch_symm_mem.empty(
                (2, self._world_size), device=device, dtype=torch.int64
            )
            self._symm_out_splits_offsets = out_splits_offsets
            self._symm_out_splits_offsets_handle = torch_symm_mem.rendezvous(
                out_splits_offsets, self._ep_group.group_name
            )

    def execute(self) -> None:
        try:
            for dtype in self._ordered_dtypes:
                send_per_peer = self._send_tensors.get(
                    dtype, [[] for _ in range(self._world_size)]
                )
                recv_per_peer = self._recv_tensors.get(
                    dtype, [[] for _ in range(self._world_size)]
                )
                input_buffer = self._symm_send_buffers[dtype]
                output_buffer = self._symm_recv_buffers[dtype]
                in_splits = [sum(t.numel() for t in peer) for peer in send_per_peer]
                out_splits = [sum(t.numel() for t in peer) for peer in recv_per_peer]
                total_send = sum(in_splits)
                total_recv = sum(out_splits)
                assert total_send <= input_buffer.numel(), (
                    "EPLB symm_mem input buffer overflow: "
                    f"required={total_send}, capacity={input_buffer.numel()}. "
                    "This violates the local-expert send bound assumption."
                )
                assert total_recv <= output_buffer.numel(), (
                    "EPLB symm_mem output buffer overflow: "
                    f"required={total_recv}, capacity={output_buffer.numel()}. "
                    "This violates the local-expert recv bound assumption."
                )
                if total_send == 0 and total_recv == 0:
                    continue
                with torch.cuda.stream(self._cuda_stream):
                    self._pack_send_buffer(send_per_peer, input_buffer, in_splits)
                    self._all_to_all_vdev(
                        input_tensor=input_buffer,
                        output_tensor=output_buffer,
                        in_splits_list=in_splits,
                        out_splits_list=out_splits,
                    )
                    self._unpack_recv_buffer(output_buffer, recv_per_peer, out_splits)
        finally:
            self._send_tensors.clear()
            self._recv_tensors.clear()


def create_eplb_communicator(
    ep_group: ProcessGroup,
    backend: str,
    expert_weights: Sequence[torch.Tensor],
) -> EplbCommunicator:
    if backend == "torch_gloo":
        cpu_group = get_ep_group().cpu_group
        return GlooCpuStagedEplbCommunicator(cpu_group=cpu_group)
    if backend == "torch_nccl":
        return TorchDistributedEplbCommunicator(ep_group=ep_group)
    if backend == "pynccl":
        unsupported_dtypes = sorted(
            {
                tensor.dtype
                for tensor in expert_weights
                if not ncclDataTypeEnum.supports_torch_dtype(tensor.dtype)
            },
            key=str,
        )
        if unsupported_dtypes:
            logger.warning_once(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtypes (%s); falling back to torch.distributed.",
                ", ".join(str(dtype) for dtype in unsupported_dtypes),
            )
            return TorchDistributedEplbCommunicator(ep_group=ep_group)

        group_coordinator = get_ep_group()
        device_comm = group_coordinator.device_communicator
        pynccl_comm = (
            getattr(device_comm, "pynccl_comm", None)
            if device_comm is not None
            else None
        )
        if pynccl_comm is None or pynccl_comm.disabled or not pynccl_comm.available:
            logger.warning(
                "EPLB communicator 'pynccl' requested but unavailable; "
                "falling back to torch.distributed."
            )
        else:
            try:
                return PyNcclEplbCommunicator(pynccl_comm=pynccl_comm)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize PyNcclEplbCommunicator (%s); "
                    "falling back to torch.distributed.",
                    exc,
                )
    elif backend == "symm_mem":
        if not symm_mem_available:
            logger.warning(
                "EPLB communicator 'symm_mem' requested but torch symmetric memory "
                "is unavailable; falling back to torch.distributed."
            )
            return TorchDistributedEplbCommunicator(ep_group=ep_group)
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(90)
        ):
            logger.warning(
                "EPLB communicator 'symm_mem' requested but device capability "
                " is below 90; falling back to torch.distributed."
            )
            return TorchDistributedEplbCommunicator(ep_group=ep_group)
        if not hasattr(torch.ops.symm_mem, "all_to_all_vdev"):
            logger.warning(
                "EPLB communicator 'symm_mem' requested but symm_mem all_to_all_vdev "
                "is unavailable; falling back to torch.distributed."
            )
            return TorchDistributedEplbCommunicator(ep_group=ep_group)
        try:
            return SymmMemEplbCommunicator(
                ep_group=ep_group,
                expert_weights=expert_weights,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize NVSHMEM EPLB symm_mem communicator (%s); "
                "falling back to torch.distributed.",
                exc,
            )
    return TorchDistributedEplbCommunicator(ep_group=ep_group)
