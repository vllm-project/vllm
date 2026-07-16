# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import torch
from torch.distributed import ProcessGroup

from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class DirectDCPA2AWorkspace:
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.num_ubatches = num_ubatches
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self._allocations: list[tuple[torch.Tensor, object, list[torch.Tensor]]] = []

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            symm_mem, output_shape, dtype, device
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(
            symm_mem, lse_shape, torch.float32, device
        )
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            symm_mem, signal_shape, torch.int32, device
        )
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=device)

    def _allocate(self, symm_mem, shape, dtype, device):
        storage = symm_mem.empty(shape, device=device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        output = partial_output.new_empty(
            (num_tokens, self.heads_per_rank, self.head_dim)
        )
        torch.ops._C.direct_dcp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            self.peer_output_ptrs[ubatch],
            self.peer_lse_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_output[ubatch],
            self.received_lse[ubatch],
            self.received_signal[ubatch],
            self.epoch[ubatch : ubatch + 1],
            output,
            self.world_size,
            self.rank,
            self.max_num_tokens,
            is_lse_base_on_e,
        )
        return output


@cache
def get_direct_dcp_a2a_workspace(
    group: ProcessGroup,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace:
    """Return the process-local workspace shared by all MLA layers."""
    return DirectDCPA2AWorkspace(
        group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )
