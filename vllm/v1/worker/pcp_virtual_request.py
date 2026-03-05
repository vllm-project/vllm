# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP (Prefill Context Parallelism) Virtual Request Manager.


This module implements PCP using a "virtual request" approach with DualChunkSwap
for load balancing. Each physical request is partitioned such that each PCP rank
handles both head and tail tokens. This keeps PCP-specific logic isolated from
the main input preparation code.

For PCP=2, a physical request with N tokens becomes:
  - Rank 0: head tokens [0, N/4) AND tail tokens [3N/4, N)
  - Rank 1: head tokens [N/4, N/2) AND tail tokens [N/2, 3N/4)

Example with 8 tokens [0,1,2,3,4,5,6,7] and PCP=2:
  - Rank 0: positions [0,1,6,7] (head=[0,1], tail=[6,7])
  - Rank 1: positions [2,3,4,5] (head=[2,3], tail=[4,5])

This balances attention workload since head tokens attend to fewer KV and
tail tokens attend to more. After KV all-gather, all ranks have full KV [0, N).
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
        self._padded_local_num_tokens: int = 0

        # State for KV restore (reordering after all-gather)
        self._kv_restore_idx: torch.Tensor | None = None

        # State from partition()
        self._num_reqs: int = 0
        self._per_rank_tokens: np.ndarray | None = None
        self._num_scheduled_tokens: np.ndarray | None = None

        # Pre-allocated buffers for partitioning
        self._virtual_num_scheduled = np.zeros(max_num_reqs, dtype=np.int32)
        self._virtual_positions = np.zeros(max_num_batched_tokens, dtype=np.int64)
        self._virtual_req_indices = np.zeros(max_num_batched_tokens, dtype=np.int64)

    def partition(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition physical requests using DualChunkSwap for load balancing.

        Each rank gets both head and tail tokens:
        - Rank r head tokens: [r*chunk, (r+1)*chunk)
        - Rank r tail tokens: [N - (r+1)*chunk, N - r*chunk)

        Example N=8, PCP=2:
          Rank 0: head=[0,1], tail=[6,7]  -> positions [0,1,6,7]
          Rank 1: head=[2,3], tail=[4,5]  -> positions [2,3,4,5]

        Args:
            num_scheduled_tokens: Token counts per physical request [num_reqs]
            num_computed_tokens: Already computed tokens per request [num_reqs]

        Returns:
            virtual_num_scheduled: Token counts for this rank's virtual requests
            virtual_req_indices: Maps virtual tokens to physical request indices
            virtual_positions: Position values for this rank's tokens
            cu_num_tokens: Cumulative token counts for query_start_loc
        """
        num_reqs = len(num_scheduled_tokens)
        pcp_size = self.pcp_world_size
        pcp_rank = self.pcp_rank

        # For DualChunkSwap, we divide tokens into 2*pcp_size chunks
        # Each rank gets one head chunk and one tail chunk
        virtual_num_scheduled = np.zeros(num_reqs, dtype=np.int32)

        # Pre-compute per-rank token assignments for all ranks (needed for restore)
        # per_rank_tokens[rank, req] = number of tokens for that rank on that request
        per_rank_tokens = np.zeros((pcp_size, num_reqs), dtype=np.int32)

        for req_idx in range(num_reqs):
            n_tokens = num_scheduled_tokens[req_idx]
            if n_tokens == 0:
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

            # Each rank r gets:
            #   - head chunk: chunk r (from start)
            #   - tail chunk: chunk (num_chunks - 1 - r) (from end)
            for r in range(pcp_size):
                head_chunk_idx = r
                tail_chunk_idx = num_chunks - 1 - r
                per_rank_tokens[r, req_idx] = (
                    chunk_sizes[head_chunk_idx] + chunk_sizes[tail_chunk_idx]
                )

        virtual_num_scheduled = per_rank_tokens[pcp_rank].copy()
        total_virtual_tokens = int(virtual_num_scheduled.sum())

        # Build virtual positions and req_indices with head+tail interleaving
        virtual_positions = self._virtual_positions[:total_virtual_tokens]
        virtual_req_indices = self._virtual_req_indices[:total_virtual_tokens]

        out_idx = 0
        for req_idx in range(num_reqs):
            n_tokens = num_scheduled_tokens[req_idx]
            n_local = virtual_num_scheduled[req_idx]
            if n_local == 0:
                continue

            base_pos = num_computed_tokens[req_idx]

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

            # This rank's head chunk and tail chunk
            head_chunk_idx = pcp_rank
            tail_chunk_idx = num_chunks - 1 - pcp_rank

            # Head positions
            head_start = chunk_starts[head_chunk_idx]
            head_count = chunk_sizes[head_chunk_idx]
            for i in range(head_count):
                virtual_positions[out_idx] = base_pos + head_start + i
                virtual_req_indices[out_idx] = req_idx
                out_idx += 1

            # Tail positions
            tail_start = chunk_starts[tail_chunk_idx]
            tail_count = chunk_sizes[tail_chunk_idx]
            for i in range(tail_count):
                virtual_positions[out_idx] = base_pos + tail_start + i
                virtual_req_indices[out_idx] = req_idx
                out_idx += 1

        cu_num_tokens = np.cumsum(virtual_num_scheduled[:num_reqs])

        # Store state for restore_hidden_states and KV restore
        self._local_num_tokens = total_virtual_tokens
        self._num_reqs = num_reqs
        self._per_rank_tokens = per_rank_tokens
        self._num_scheduled_tokens = num_scheduled_tokens.copy()
        self._setup_restore_indices(num_reqs, per_rank_tokens, virtual_num_scheduled)
        self._setup_kv_restore_indices(num_reqs, num_scheduled_tokens, per_rank_tokens)

        logger.info(
            "PCP partition (DualChunkSwap): rank=%d/%d, num_reqs=%d, "
            "total_tokens=%d -> local_tokens=%d",
            pcp_rank,
            pcp_size,
            num_reqs,
            int(num_scheduled_tokens.sum()),
            total_virtual_tokens,
        )

        return (
            virtual_num_scheduled[:num_reqs].copy(),
            virtual_req_indices[:total_virtual_tokens].copy(),
            virtual_positions[:total_virtual_tokens].copy(),
            cu_num_tokens,
        )

    def _setup_restore_indices(
        self,
        num_reqs: int,
        per_rank_tokens: np.ndarray,
        _virtual_num_scheduled: np.ndarray,
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
