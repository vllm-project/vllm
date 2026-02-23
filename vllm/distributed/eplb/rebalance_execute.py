# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False


@dataclass
class RecvMetadata:
    """Metadata describing remote receives during EPLB rebalancing."""

    recv_primary_mask: np.ndarray
    """Mask of (num_local_experts,) indicating primary experts received."""
    recv_count: int
    """Number of received experts for the layer."""
    recv_expert_ids: np.ndarray
    """Expert ids (num_local_experts,) of remote primary experts."""
    recv_dst_rows: np.ndarray
    """Target expert indices (num_local_experts,) in local tensors to send."""


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers.

    Implementations provide P2P send/recv and execution semantics.
    """

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
            if self._cuda_stream is not None:
                with torch.cuda.stream(self._cuda_stream):
                    reqs = batch_isend_irecv(self._p2p_ops)
                    for req in reqs:
                        req.wait()
            else:
                reqs = batch_isend_irecv(self._p2p_ops)
                for req in reqs:
                    req.wait()
        finally:
            self._p2p_ops.clear()


class PyNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyNcclCommunicator using ncclSend/ncclRecv."""

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._pynccl_comm = pynccl_comm
        self._cuda_stream = cuda_stream
        # Track if group has been started
        self._group_started = False

    def _ensure_group_started(self) -> None:
        """Start NCCL group on first add_send/add_recv call."""
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
    if backend == "pynccl":
        # Check if expert weights have dtypes supported by PyNccl
        try:
            # Try converting each dtype to verify PyNccl support
            for tensor in expert_weights:
                ncclDataTypeEnum.from_torch(tensor.dtype)
        except ValueError as e:
            logger.warning_once(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtype; falling back to torch.distributed. Error: %s",
                str(e),
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
        if not hasattr(torch.ops.symm_mem, "all_to_all_vdev"):
            logger.warning(
                "EPLB communicator 'symm_mem' requested but symm_mem all_to_all_vdev "
                "is unavailable; falling back to torch.distributed."
            )
            return TorchDistributedEplbCommunicator(ep_group=ep_group)
        symm_device = expert_weights[0].device
        try:
            if not torch_symm_mem.is_nvshmem_available():
                logger.warning(
                    "EPLB communicator 'symm_mem' requested but NVSHMEM is "
                    "unavailable; falling back to torch.distributed."
                )
                return TorchDistributedEplbCommunicator(ep_group=ep_group)
            current_backend = torch_symm_mem.get_backend(symm_device)
            if current_backend is None or str(current_backend).upper() != "NVSHMEM":
                logger.warning(
                    "EPLB communicator 'symm_mem' requested but active backend is %s "
                    "(NVSHMEM required); falling back to torch.distributed.",
                    current_backend,
                )
                return TorchDistributedEplbCommunicator(ep_group=ep_group)

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


# Type alias for the result of move_to_buffer or transfer_layer
MoveToBufferResult = tuple[np.ndarray, np.ndarray, RecvMetadata]


def get_ep_ranks_with_experts_batch(
    expert_ids: np.ndarray,
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        expert_ids: 1D array of expert indices to query.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two dictionaries mapping expert_id to:
        - ranks_to_send: The ranks that have this expert and need to send.
        - ranks_to_recv: The ranks that need to receive this expert.
    """
    ranks_to_send_map: dict[int, list[int]] = {}
    ranks_to_recv_map: dict[int, list[int]] = {}

    # Fast path: if no experts, return empty dicts
    if expert_ids.size == 0:
        return ranks_to_send_map, ranks_to_recv_map

    unique_experts = np.unique(expert_ids)
    num_positions = len(old_indices)
    position_indices = np.arange(num_positions, dtype=np.int32)

    # Vectorized approach: find all positions matching any query expert in one pass
    # Use np.isin to get boolean masks for all relevant positions at once
    old_relevant_mask = np.isin(old_indices, unique_experts)
    new_relevant_mask = np.isin(new_indices, unique_experts)

    # Process old_indices (send ranks)
    if np.any(old_relevant_mask):
        old_relevant_positions = position_indices[old_relevant_mask]
        old_relevant_experts = old_indices[old_relevant_mask]
        old_relevant_ranks = old_relevant_positions // num_local_experts

        # Sort by expert first, then by position (to maintain first-appearance order)
        sort_order = np.lexsort((old_relevant_positions, old_relevant_experts))
        sorted_experts = old_relevant_experts[sort_order]
        sorted_ranks = old_relevant_ranks[sort_order]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        )

        # For each expert, extract unique ranks in order of first appearance
        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]
            ranks_to_send_map[expert] = unique_ranks.tolist()

    # Process new_indices (recv ranks)
    if np.any(new_relevant_mask):
        new_relevant_positions = position_indices[new_relevant_mask]
        new_relevant_experts = new_indices[new_relevant_mask]
        new_relevant_ranks = new_relevant_positions // num_local_experts

        # Sort by expert first, then by position
        sort_order = np.lexsort((new_relevant_positions, new_relevant_experts))
        sorted_experts = new_relevant_experts[sort_order]
        sorted_ranks = new_relevant_ranks[sort_order]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        )

        # For each expert, extract unique ranks and exclude local copies
        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]

            # Remove ranks that have local copies (in send map)
            send_ranks_set = set(ranks_to_send_map.get(expert, []))
            recv_ranks_actual = [
                int(r) for r in unique_ranks if r not in send_ranks_set
            ]
            ranks_to_recv_map[expert] = recv_ranks_actual

    # Handle experts that only appear in old (send only) or new (recv only)
    for expert in unique_experts:
        expert = int(expert)
        if expert not in ranks_to_send_map:
            ranks_to_send_map[expert] = []
        if expert not in ranks_to_recv_map:
            ranks_to_recv_map[expert] = []

    return ranks_to_send_map, ranks_to_recv_map


def move_to_buffer(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: Sequence[torch.Tensor],
    cuda_stream: torch.cuda.Stream | None,
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
) -> MoveToBufferResult:
    """
    Rearranges expert weights during EPLB rebalancing.

    Args:
        num_local_experts: Number of local experts.
        old_indices: (num_experts_total,) ndarray of current (old)
            global-to-local expert assignments.
        new_indices: (num_experts_total,) ndarray of desired (new)
            global-to-local assignments after rebalance.
        expert_weights: Original expert weights for the layer.
        expert_weights_buffers: Intermediate buffers (one per tensor).
        cuda_stream: CUDA stream for async copies (can be None for sync mode).
        ep_group: Distributed process group for expert parallel comms.
        communicator: EplbCommunicator instance for P2P communication.

    Returns:
        is_unchanged (np.ndarray): (num_local_experts,), True where an expert row
            is unchanged after rebalance.
        is_received_locally (np.ndarray): (num_local_experts,), True where a row
            can be updated from local data.
        RecvMetadata: Metadata needed for completing remote weight transfers.
    """
    assert old_indices.shape == new_indices.shape
    ep_rank = ep_group.rank()

    recv_primary_mask = np.zeros((num_local_experts,), dtype=np.bool_)
    send_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    send_src_rows = np.full((num_local_experts,), -1, dtype=np.int32)
    recv_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    recv_dst_rows = np.full((num_local_experts,), -1, dtype=np.int32)

    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows

    old_local_expert_ids = old_indices[local_global]
    new_local_expert_ids = new_indices[local_global]

    # Unchanged mask
    is_unchanged = old_local_expert_ids == new_local_expert_ids

    # Local receive eligibility
    new_valid = new_local_expert_ids != -1
    can_recv_local = np.isin(
        new_local_expert_ids, old_local_expert_ids, assume_unique=False
    )
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local)
    )

    # Send map: first src row per unique expert present locally in old mapping
    send_count = 0
    valid_old = old_local_expert_ids != -1
    if np.any(valid_old):
        uniq_experts, first_idx = np.unique(
            old_local_expert_ids[valid_old], return_index=True
        )
        filtered_rows = local_rows[valid_old]
        src_rows = filtered_rows[first_idx]
        send_count = int(uniq_experts.shape[0])
        send_expert_ids[:send_count] = uniq_experts
        send_src_rows[:send_count] = src_rows

    # Recv map: primary dst per unique expert needed remotely
    recv_count = 0
    need_recv_mask = np.logical_and(~is_received_locally, new_valid)
    if np.any(need_recv_mask):
        desired_experts = new_local_expert_ids[need_recv_mask]
        desired_dsts = local_rows[need_recv_mask]
        uniq_recv_experts, uniq_indices = np.unique(desired_experts, return_index=True)
        dst_rows = desired_dsts[uniq_indices]
        recv_count = int(uniq_recv_experts.shape[0])
        recv_expert_ids[:recv_count] = uniq_recv_experts
        recv_dst_rows[:recv_count] = dst_rows
        recv_primary_mask[dst_rows] = True

    eligible_local_buffer_mask = np.logical_and(~is_unchanged, is_received_locally)

    # 1. Local moves into tmp buffers
    if bool(eligible_local_buffer_mask.any()) and send_count > 0:
        dest_indices = np.nonzero(eligible_local_buffer_mask)[0].tolist()
        expert_to_src_map = dict(
            zip(send_expert_ids[:send_count], send_src_rows[:send_count])
        )
        for dst in dest_indices:
            expert = new_local_expert_ids[dst]
            src_local = expert_to_src_map.get(expert, -1)
            if src_local != -1:
                for w, b in zip(expert_weights, expert_weights_buffers):
                    b[dst].copy_(w[src_local], non_blocking=True)

    # 2. Post sends
    if send_count > 0:
        experts = send_expert_ids[:send_count]
        srcs = send_src_rows[:send_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, src in zip(experts.tolist(), srcs.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv):
                recv_ranks.append(ranks_to_recv[recver_pos])
            for dst in recv_ranks:
                for w in expert_weights:
                    communicator.add_send(w[src], dst)

    # 3. Post recvs
    if recv_count > 0:
        experts = recv_expert_ids[:recv_count]
        dsts = recv_dst_rows[:recv_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, dst in zip(experts.tolist(), dsts.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank)
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src = ranks_to_send[recver_pos - remainder_start]
            for b in expert_weights_buffers:
                communicator.add_recv(b[dst], src)

    # 4. Execute the P2P operations. The real communication happens here.
    communicator.execute()
    # wait for the communication to finish
    return (
        is_unchanged,
        is_received_locally,
        RecvMetadata(
            recv_primary_mask=recv_primary_mask,
            recv_count=recv_count,
            recv_expert_ids=recv_expert_ids,
            recv_dst_rows=recv_dst_rows,
        ),
    )


def move_from_buffer(
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: list[torch.Tensor],
    is_unchanged: np.ndarray,
    is_received_locally: np.ndarray,
    recv_metadata: RecvMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """
    Copies expert weights from communication buffers back to the target weight tensors
    after EPLB rebalancing.

    Args:
        expert_weights: List of the actual MoE layer weights used in the execution.
        expert_weights_buffers: Intermediate buffers containing the experts weights
            after the transfer is completed.
        is_unchanged: (num_local_experts,), True where an expert row is unchanged.
        is_received_locally: (num_local_experts,), True where a row is updated locally.
        recv_metadata: RecvMetadata containing remote receive metadata.
        new_indices: (num_experts_total,) mapping from local rows to desired
            (possibly global) expert id, after rebalance.
        ep_rank: Rank of the process in the expert parallel group.
    """
    recv_primary_mask = recv_metadata.recv_primary_mask
    recv_count = recv_metadata.recv_count
    recv_expert_ids = recv_metadata.recv_expert_ids
    recv_dst_rows = recv_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    # Mask for rows to copy back from buffers:
    # copy if locally received OR remote primary recv
    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask_np = np.logical_and(~is_unchanged, copy_mask)
    if bool(dest_mask_np.any()):
        dest_indices = np.nonzero(dest_mask_np)[0].tolist()
        for dst in dest_indices:
            for w, b in zip(expert_weights, expert_weights_buffers):
                w[dst].copy_(b[dst], non_blocking=True)

    if recv_count == 0:
        return

    # Duplicate remote received rows to non-primary duplicate dsts
    base = ep_rank * num_local_experts
    local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1),
    )
    # All received experts are unique in the destination, so no need to copy duplicates
    if not bool(duplicate_mask.any()):
        return

    dup_dst_rows = np.nonzero(duplicate_mask)[0]
    dup_experts = local_experts[dup_dst_rows]

    prim_experts = recv_expert_ids[:recv_count]
    prim_dsts = recv_dst_rows[:recv_count]
    order = np.argsort(prim_experts, kind="stable")
    prim_experts_sorted = prim_experts[order]
    prim_dsts_sorted = prim_dsts[order]
    pos = np.searchsorted(prim_experts_sorted, dup_experts)
    valid = np.logical_and(
        pos < prim_experts_sorted.shape[0],
        prim_experts_sorted[np.minimum(pos, prim_experts_sorted.shape[0] - 1)]
        == dup_experts,
    )
    if not bool(valid.any()):
        return

    matched_dst_rows = dup_dst_rows[valid]
    matched_src_rows = prim_dsts_sorted[pos[valid]]

    for dst, src in zip(matched_dst_rows.tolist(), matched_src_rows.tolist()):
        for w in expert_weights:
            w[dst].copy_(w[src], non_blocking=True)


async def transfer_layer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
) -> MoveToBufferResult:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_layer_indices: Shape (num_physical_experts,).
        new_layer_indices: Shape (num_physical_experts,).
        expert_weights: Iterable of weight tensors for this layer, each with shape
            (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection.
        expert_weights_buffer: Intermediate buffers (one per weight tensor).
        ep_group: The device process group for expert parallelism.
        communicator: EplbCommunicator instance for P2P communication.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        cuda_stream: CUDA stream for async copies (can be None for sync mode).
        rank_mapping: Optional rank mapping for elastic expert parallelism.

    Returns:
        is_unchanged (np.ndarray): (num_local_experts,), True where expert
            is left unchanged.
        is_received_locally (np.ndarray): (num_local_experts,), True where expert
            can be received locally.
        RecvMetadata: Metadata needed for completing remote weight transfers.
    """
    ep_size = ep_group.size()
    if rank_mapping is not None:
        # Add a layer dimension for compatibility with mapping functions
        old_layer_indices_2d = old_layer_indices.unsqueeze(0)
        new_layer_indices_2d = new_layer_indices.unsqueeze(0)

        if len(rank_mapping) == ep_group.size():
            # scale down
            new_layer_indices_2d = _map_new_expert_indices_with_rank_mapping(
                new_layer_indices_2d,
                rank_mapping,
            )
        else:
            # scale up
            old_layer_indices_2d = _map_old_expert_indices_with_rank_mapping(
                old_layer_indices_2d,
                rank_mapping,
                ep_group.size(),
            )

        # Remove the layer dimension
        old_layer_indices = old_layer_indices_2d.squeeze(0)
        new_layer_indices = new_layer_indices_2d.squeeze(0)

    assert old_layer_indices.shape == new_layer_indices.shape
    num_physical_experts = old_layer_indices.shape[0]
    assert len(expert_weights[0]) >= 1
    num_local_physical_experts = expert_weights[0].shape[0]
    assert num_physical_experts == ep_size * num_local_physical_experts

    old_layer_indices_np = old_layer_indices.cpu().numpy()
    new_layer_indices_np = new_layer_indices.cpu().numpy()

    is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices=old_layer_indices_np,
        new_indices=new_layer_indices_np,
        expert_weights=expert_weights,
        expert_weights_buffers=expert_weights_buffer,
        cuda_stream=cuda_stream,
        ep_group=ep_group,
        communicator=communicator,
    )
    return is_unchanged, is_received_locally, recv_metadata


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    ep_group: ProcessGroup,
    communicator: EplbCommunicator,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        communicator: EplbCommunicator instance for P2P communication.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    assert len(expert_weights[0]) >= 1

    num_local_physical_experts = expert_weights[0][0].shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    first_layer_weights = list(expert_weights[0])
    # Buffers to hold the expert weights during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    weights_buffer: list[torch.Tensor] = [
        torch.empty_like(w) for w in first_layer_weights
    ]
    if is_profile:
        # Reserve communication buffers via a minimal dummy all_gather on first layer
        for weight, buffer in zip(expert_weights[0], weights_buffer):
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            torch.distributed.barrier()
            all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        return

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()

    old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    for layer_idx in range(num_moe_layers):
        is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices=old_global_expert_indices_cpu[layer_idx],
            new_indices=new_global_expert_indices_cpu[layer_idx],
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            cuda_stream=None,
            ep_group=ep_group,
            communicator=communicator,
        )

        move_from_buffer(
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            is_unchanged=is_unchanged,
            is_received_locally=is_received_locally,
            recv_metadata=recv_metadata,
            new_indices=new_global_expert_indices_cpu[layer_idx],
            ep_rank=ep_group.rank(),
        )


def _map_old_expert_indices_with_rank_mapping(
    old_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
    new_ep_size: int,
) -> torch.Tensor:
    """
    Map the old global expert indices to the new global expert indices.

    Args:
        old_global_expert_indices:
            Shape (num_layers, old_ep_size * num_local_physical_experts).
        rank_mapping: Mapping from old rank to new rank.
        new_ep_size: New expert parallelism size.

    Returns:
        Mapped expert indices with shape
        (num_layers, new_ep_size * num_local_physical_experts).
    """
    num_layers, old_num_physical_experts = old_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    num_local_physical_experts = old_num_physical_experts // old_ep_size
    new_num_physical_experts = new_ep_size * num_local_physical_experts

    # Create mapped tensor with new shape, initialized to -1
    mapped_expert_indices = torch.full(
        (num_layers, new_num_physical_experts),
        fill_value=-1,
        dtype=old_global_expert_indices.dtype,
        device=old_global_expert_indices.device,
    )

    # Handle rank mapping (scale up/down with rank changes)
    for old_rank in range(old_ep_size):
        new_rank = rank_mapping.get(old_rank)
        if new_rank is not None and new_rank >= 0 and new_rank < new_ep_size:
            # This old rank exists in the new configuration
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, new_start_idx:new_end_idx] = (
                old_global_expert_indices[:, old_start_idx:old_end_idx]
            )
        # If new_rank is None or >= new_ep_size, the experts remain -1
        # (scale down case)

    return mapped_expert_indices


def _map_new_expert_indices_with_rank_mapping(
    new_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
) -> torch.Tensor:
    num_layers, new_num_physical_experts = new_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    new_ep_size = sum(new_rank != -1 for new_rank in rank_mapping.values())
    num_local_physical_experts = new_num_physical_experts // new_ep_size
    old_num_physical_experts = old_ep_size * num_local_physical_experts

    mapped_expert_indices = torch.full(
        (num_layers, old_num_physical_experts),
        fill_value=-1,
        dtype=new_global_expert_indices.dtype,
        device=new_global_expert_indices.device,
    )

    for old_rank in range(old_ep_size):
        new_rank = rank_mapping[old_rank]
        if new_rank >= 0 and new_rank < new_ep_size:
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, old_start_idx:old_end_idx] = (
                new_global_expert_indices[:, new_start_idx:new_end_idx]
            )

    return mapped_expert_indices


__all__ = ["transfer_layer", "move_from_buffer", "RecvMetadata"]
