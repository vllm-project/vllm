# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP (Prefill Context Parallelism) Virtual Request Manager.


This module implements PCP using a "virtual request" approach with DualChunkSwap
for load balancing. Each physical request is split into 2 virtual requests
(head and tail), allowing _prepare_inputs to be PCP-agnostic.

For PCP=2, a physical request with N tokens becomes:
  Rank 0:
    - Virtual req 0 (head): tokens [0, N/4), num_computed=0
    - Virtual req 1 (tail): tokens [3N/4, N), num_computed=3N/4
  Rank 1:
    - Virtual req 0 (head): tokens [N/4, N/2), num_computed=N/4
    - Virtual req 1 (tail): tokens [N/2, 3N/4), num_computed=N/2

Example with 8 tokens and PCP=2:
  Rank 0: vreq0 (head, 2 tokens, computed=0), vreq1 (tail, 2 tokens, computed=6)
  Rank 1: vreq0 (head, 2 tokens, computed=2), vreq1 (tail, 2 tokens, computed=4)

The standard position formula (num_computed + arange) produces correct positions.
After KV all-gather, all ranks have full KV [0, N).
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


class PCPVirtualRequestManager:
    """Manages PCP virtual request partitioning and output restoration."""

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_num_reqs: int,
        max_num_batched_tokens: int,
        device: torch.device,
    ):
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        # State for restore_hidden_states
        self._allgather_restore_idx: torch.Tensor | None = None
        self._local_num_tokens: int = 0
        self._global_num_tokens: int = 0

        # State for KV restore (reordering after all-gather)
        self._kv_restore_idx: torch.Tensor | None = None

        # State from partition()
        self._num_physical_reqs: int = 0
        self._per_rank_tokens: np.ndarray | None = None
        self._physical_num_scheduled: np.ndarray | None = None

        # Pre-allocated buffers for virtual requests (2 per physical request)
        max_virtual_reqs = 2 * max_num_reqs
        self._virtual_num_scheduled = np.zeros(max_virtual_reqs, dtype=np.int32)
        self._virtual_num_computed = np.zeros(max_virtual_reqs, dtype=np.int64)
        self._virtual_to_physical = np.zeros(max_virtual_reqs, dtype=np.int64)

    def partition(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Partition physical requests into virtual requests using DualChunkSwap.

        Each physical request becomes 2 virtual requests (head + tail).
        This allows _prepare_inputs to be PCP-agnostic - it just sees
        virtual requests with their own num_scheduled and num_computed.

        Example N=8, PCP=2, physical request with num_computed=100:
          Rank 0:
            - vreq 0 (head): num_scheduled=2, num_computed=100
            - vreq 1 (tail): num_scheduled=2, num_computed=106
          Rank 1:
            - vreq 0 (head): num_scheduled=2, num_computed=102
            - vreq 1 (tail): num_scheduled=2, num_computed=104

        Args:
            num_scheduled_tokens: Token counts per physical request [num_reqs]
            num_computed_tokens: Already computed tokens per request [num_reqs]

        Returns:
            virtual_num_scheduled: Token counts per virtual request [2*num_reqs]
            virtual_num_computed: Starting positions per virtual request [2*num_reqs]
            virtual_to_physical: Maps virtual req idx to physical req idx [2*num_reqs]
        """
        num_reqs = len(num_scheduled_tokens)
        num_virtual_reqs = 2 * num_reqs
        pcp_size = self.pcp_world_size
        pcp_rank = self.pcp_rank

        # Output arrays for virtual requests
        virtual_num_scheduled = self._virtual_num_scheduled[:num_virtual_reqs]
        virtual_num_computed = self._virtual_num_computed[:num_virtual_reqs]
        virtual_to_physical = self._virtual_to_physical[:num_virtual_reqs]

        # Pre-compute per-rank token assignments for all ranks (needed for restore)
        per_rank_tokens = np.zeros((pcp_size, num_reqs), dtype=np.int32)

        for req_idx in range(num_reqs):
            n_tokens = num_scheduled_tokens[req_idx]
            base_computed = num_computed_tokens[req_idx]

            if n_tokens == 0:
                # Empty request -> empty virtual requests
                head_vreq = 2 * req_idx
                tail_vreq = 2 * req_idx + 1
                virtual_num_scheduled[head_vreq] = 0
                virtual_num_scheduled[tail_vreq] = 0
                virtual_num_computed[head_vreq] = base_computed
                virtual_num_computed[tail_vreq] = base_computed
                virtual_to_physical[head_vreq] = req_idx
                virtual_to_physical[tail_vreq] = req_idx
                continue

            # Divide into 2*pcp_size chunks, distribute remainder to first chunks
            num_chunks = 2 * pcp_size
            chunk_size = n_tokens // num_chunks
            remainder = n_tokens % num_chunks

            # Compute size of each chunk (first 'remainder' chunks get +1)
            chunk_sizes = np.array(
                [chunk_size + (1 if i < remainder else 0) for i in range(num_chunks)],
                dtype=np.int32,
            )

            # Compute cumulative start positions for each chunk
            chunk_starts = np.zeros(num_chunks + 1, dtype=np.int32)
            for i in range(num_chunks):
                chunk_starts[i + 1] = chunk_starts[i] + chunk_sizes[i]

            # Store per-rank tokens for all ranks (needed for restore indices)
            for r in range(pcp_size):
                head_idx = r
                tail_idx = num_chunks - 1 - r
                per_rank_tokens[r, req_idx] = (
                    chunk_sizes[head_idx] + chunk_sizes[tail_idx]
                )

            # This rank's head and tail chunks
            head_chunk_idx = pcp_rank
            tail_chunk_idx = num_chunks - 1 - pcp_rank

            # Virtual request indices
            head_vreq = 2 * req_idx
            tail_vreq = 2 * req_idx + 1

            # Head virtual request
            virtual_num_scheduled[head_vreq] = chunk_sizes[head_chunk_idx]
            virtual_num_computed[head_vreq] = (
                base_computed + chunk_starts[head_chunk_idx]
            )
            virtual_to_physical[head_vreq] = req_idx

            # Tail virtual request
            virtual_num_scheduled[tail_vreq] = chunk_sizes[tail_chunk_idx]
            virtual_num_computed[tail_vreq] = (
                base_computed + chunk_starts[tail_chunk_idx]
            )
            virtual_to_physical[tail_vreq] = req_idx

        total_virtual_tokens = int(virtual_num_scheduled.sum())

        # Store state for restore_hidden_states and KV restore
        self._local_num_tokens = total_virtual_tokens
        self._num_physical_reqs = num_reqs
        self._per_rank_tokens = per_rank_tokens
        self._physical_num_scheduled = num_scheduled_tokens.copy()
        self._setup_restore_indices(num_reqs, per_rank_tokens)
        self._setup_kv_restore_indices(num_reqs, num_scheduled_tokens, per_rank_tokens)

        logger.info(
            "PCP partition (DualChunkSwap): rank=%d/%d, physical_reqs=%d, "
            "virtual_reqs=%d, total_tokens=%d -> local_tokens=%d",
            pcp_rank,
            pcp_size,
            num_reqs,
            num_virtual_reqs,
            int(num_scheduled_tokens.sum()),
            total_virtual_tokens,
        )

        return (
            virtual_num_scheduled[:num_virtual_reqs].copy(),
            virtual_num_computed[:num_virtual_reqs].copy(),
            virtual_to_physical[:num_virtual_reqs].copy(),
        )

    def _setup_restore_indices(
        self,
        num_reqs: int,
        per_rank_tokens: np.ndarray,
    ) -> None:
        """Set up indices for restoring hidden states after all-gather.

        After all-gather, we need to interleave the tokens from each rank
        back into the correct order (rank0_req0, rank1_req0, rank0_req1, ...).
        """
        pcp_size = self.pcp_world_size

        # Total tokens across all ranks
        total_global_tokens = int(per_rank_tokens.sum())
        self._global_num_tokens = total_global_tokens

        # Build restore indices: map from gathered order to interleaved order
        restore_idx = np.zeros(total_global_tokens, dtype=np.int64)

        # After all-gather, tokens are arranged as:
        # [rank0_all_tokens, rank1_all_tokens, ...]
        # We want to restore to:
        # [rank0_req0, rank1_req0, rank0_req1, rank1_req1, ...]

        out_idx = 0
        rank_offsets = np.zeros(pcp_size, dtype=np.int64)
        for req_idx in range(num_reqs):
            for rank in range(pcp_size):
                n_tokens = per_rank_tokens[rank, req_idx]
                for _ in range(n_tokens):
                    # Index into the all-gathered buffer
                    gathered_idx = (
                        rank * int(per_rank_tokens[rank].sum()) + rank_offsets[rank]
                    )
                    restore_idx[out_idx] = gathered_idx
                    out_idx += 1
                    rank_offsets[rank] += 1

        self._allgather_restore_idx = torch.from_numpy(restore_idx).to(self.device)

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """All-gather and restore hidden states to original token order.

        Args:
            hidden_states: Local hidden states [local_num_tokens, hidden_dim]

        Returns:
            Restored hidden states [global_num_tokens, hidden_dim]
        """
        pcp_group = get_pcp_group()
        hidden_dim = hidden_states.shape[1]

        # All-gather with variable sizes using all_gather_into_tensor
        # First, gather the sizes from all ranks
        local_size = torch.tensor(
            [hidden_states.shape[0]], dtype=torch.int64, device=self.device
        )
        all_sizes = torch.empty(
            self.pcp_world_size, dtype=torch.int64, device=self.device
        )
        torch.distributed.all_gather_into_tensor(
            all_sizes, local_size, group=pcp_group.device_group
        )

        # Compute total and max size
        total_size = int(all_sizes.sum().item())
        max_size = int(all_sizes.max().item())

        # Pad local hidden states to max_size
        if hidden_states.shape[0] < max_size:
            pad_size = max_size - hidden_states.shape[0]
            hidden_states = torch.cat(
                [
                    hidden_states,
                    torch.zeros(
                        pad_size,
                        hidden_dim,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ),
                ],
                dim=0,
            )

        # All-gather padded tensors
        gathered = torch.empty(
            self.pcp_world_size * max_size,
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.distributed.all_gather_into_tensor(
            gathered, hidden_states, group=pcp_group.device_group
        )

        # Remove padding and restore order
        # Build indices accounting for padding
        unpadded_indices = []
        for rank in range(self.pcp_world_size):
            rank_offset = rank * max_size
            rank_size = int(all_sizes[rank].item())
            for i in range(rank_size):
                unpadded_indices.append(rank_offset + i)

        unpadded_indices = torch.tensor(
            unpadded_indices, dtype=torch.int64, device=self.device
        )
        unpadded = gathered[unpadded_indices]

        # Restore order using pre-computed indices
        assert self._allgather_restore_idx is not None
        restored = unpadded[self._allgather_restore_idx]

        logger.info(
            "PCP restore: local_size=%d, total_size=%d, restored_size=%d",
            int(local_size.item()),
            total_size,
            restored.shape[0],
        )

        return restored

    def _setup_kv_restore_indices(
        self,
        num_reqs: int,
        num_scheduled_tokens: np.ndarray,
        per_rank_tokens: np.ndarray,
    ) -> None:
        """Set up indices for restoring KV order after all-gather.

        After all-gather, KV tokens are arranged as:
        [rank0_all_tokens, rank1_all_tokens, ...]

        We need to reorder to global position order:
        [req0_pos0, req0_pos1, ..., req1_pos0, req1_pos1, ...]

        The key insight is that each rank's tokens have specific positions
        (head + tail positions). We need to map from gathered order to
        sorted position order.
        """
        pcp_size = self.pcp_world_size

        # Build a list of (global_position, gathered_idx) pairs
        # Then sort by position to get restore indices
        position_to_gathered_idx = []

        # Track offsets into each rank's gathered segment
        rank_base_offsets = np.zeros(pcp_size, dtype=np.int64)
        for r in range(1, pcp_size):
            rank_base_offsets[r] = rank_base_offsets[r - 1] + int(
                per_rank_tokens[r - 1].sum()
            )

        rank_local_offsets = np.zeros(pcp_size, dtype=np.int64)

        # For each request, compute positions assigned to each rank
        req_base_pos = 0
        for req_idx in range(num_reqs):
            n_tokens = num_scheduled_tokens[req_idx]
            if n_tokens == 0:
                continue

            # Recompute chunk sizes for this request
            num_chunks = 2 * pcp_size
            chunk_size = n_tokens // num_chunks
            remainder = n_tokens % num_chunks
            chunk_sizes = np.array(
                [chunk_size + (1 if i < remainder else 0) for i in range(num_chunks)],
                dtype=np.int32,
            )

            # Compute cumulative start positions for each chunk
            chunk_starts = np.zeros(num_chunks + 1, dtype=np.int32)
            for i in range(num_chunks):
                chunk_starts[i + 1] = chunk_starts[i] + chunk_sizes[i]

            for rank in range(pcp_size):
                n_rank_tokens = per_rank_tokens[rank, req_idx]
                if n_rank_tokens == 0:
                    continue

                # This rank's head chunk and tail chunk
                head_chunk_idx = rank
                tail_chunk_idx = num_chunks - 1 - rank

                # Head positions
                head_start = chunk_starts[head_chunk_idx]
                head_count = chunk_sizes[head_chunk_idx]
                for i in range(head_count):
                    global_pos = req_base_pos + head_start + i
                    gathered_idx = int(
                        rank_base_offsets[rank] + rank_local_offsets[rank]
                    )
                    position_to_gathered_idx.append((global_pos, gathered_idx))
                    rank_local_offsets[rank] += 1

                # Tail positions
                tail_start = chunk_starts[tail_chunk_idx]
                tail_count = chunk_sizes[tail_chunk_idx]
                for i in range(tail_count):
                    global_pos = req_base_pos + tail_start + i
                    gathered_idx = int(
                        rank_base_offsets[rank] + rank_local_offsets[rank]
                    )
                    position_to_gathered_idx.append((global_pos, gathered_idx))
                    rank_local_offsets[rank] += 1

            req_base_pos += n_tokens

        # Sort by global position and extract gathered indices
        position_to_gathered_idx.sort(key=lambda x: x[0])
        kv_restore_idx = np.array(
            [idx for _, idx in position_to_gathered_idx], dtype=np.int64
        )

        self._kv_restore_idx = torch.from_numpy(kv_restore_idx).to(self.device)

    def get_kv_restore_idx(self) -> torch.Tensor | None:
        """Get indices for restoring KV order after all-gather.

        Returns:
            Tensor of indices to reorder all-gathered KV to global position order,
            or None if not in PCP mode.
        """
        return self._kv_restore_idx
