# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct symmetric-memory DCP chunked-context KV gather."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.v1.attention.ops.dcp_direct_utils import (
    _direct_dcp_enabled,
    _DirectDCPWorkspace,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


class DirectDCPKVGatherWorkspace(_DirectDCPWorkspace):
    """Persistent symmetric buffers for the DCP chunked-context KV all-gather.

    Each rank multimem-stores its contiguous [T, D] context-chunk KV slice
    into every rank's staging slot at offset ``rank * T`` and the kernel then
    materializes the gathered ``[world_size * T, D]`` result into the caller's
    ordinary workspace slice, so downstream consumers are unchanged.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_gathered_tokens: int,
        token_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Direct DCP kv-gather does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                "Direct DCP kv-gather requires at least one ubatch slot, "
                f"got {num_ubatches}"
            )
        if max_gathered_tokens < 1 or token_dim < 1:
            raise ValueError(
                "Direct DCP kv-gather dimensions must be positive, got "
                f"T={max_gathered_tokens}, D={token_dim}"
            )
        super().__init__(group, device, num_ubatches)
        if self.world_size <= 1:
            raise ValueError("Direct DCP kv-gather requires at least two ranks")
        if max_gathered_tokens % self.world_size != 0:
            raise ValueError(
                "Direct DCP kv-gather capacity must divide evenly across "
                f"ranks: {max_gathered_tokens} % {self.world_size} != 0"
            )
        self.max_gathered_tokens = max_gathered_tokens

        kv_shape = (num_ubatches, 2, max_gathered_tokens, token_dim)
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_kv, self.peer_kv_ptrs = self._allocate(kv_shape, dtype)
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            signal_shape, torch.int32
        )
        kv_mc_ptrs = self._multicast_ptrs(self.received_kv)
        signal_mc_ptrs = self._multicast_ptrs(self.received_signal)
        # Multicast is all-or-nothing per ubatch: the kernel pairs the data
        # multimem stream with a multicast completion signal.
        self.multicast_ptrs = [
            (kv_mc, signal_mc) if kv_mc and signal_mc else (0, 0)
            for kv_mc, signal_mc in zip(kv_mc_ptrs, signal_mc_ptrs)
        ]
        self.completion = self.received_signal.new_zeros((num_ubatches, 2))
        torch.accelerator.synchronize()

    def gather(self, gathered_kv: torch.Tensor, local_kv: torch.Tensor) -> None:
        """All-gather ``local_kv`` slices into ``gathered_kv`` (rank-major)."""
        ubatch = dbo_current_ubatch_id()
        if not 0 <= ubatch < self.num_ubatches:
            raise ValueError(
                f"DCP kv-gather ubatch {ubatch} exceeds {self.num_ubatches} slots"
            )
        kv_mc_ptr, signal_mc_ptr = self.multicast_ptrs[ubatch]
        torch.ops._C.direct_dcp_kv_gather(
            local_kv,
            self.peer_kv_ptrs[ubatch],
            self.peer_signal_ptrs[ubatch],
            self.received_kv[ubatch],
            self.received_signal[ubatch],
            self.completion[ubatch],
            self.epoch[ubatch : ubatch + 1],
            gathered_kv,
            self.world_size,
            self.rank,
            self.max_gathered_tokens,
            kv_mc_ptr,
            signal_mc_ptr,
        )


def _direct_dcp_kv_gather_enabled(group: GroupCoordinator, dtype: torch.dtype) -> bool:
    return _direct_dcp_enabled(
        group, dtype, envs.VLLM_USE_DIRECT_DCP_KV_GATHER, _SUPPORTED_DTYPES
    )


@cache
def get_direct_dcp_kv_gather_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    token_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPKVGatherWorkspace | None:
    """Return the shared direct kv-gather workspace, or None if disabled."""
    if not _direct_dcp_kv_gather_enabled(group, dtype):
        return None
    return DirectDCPKVGatherWorkspace(
        group.device_group,
        device,
        max_gathered_tokens,
        token_dim,
        dtype,
        num_ubatches,
    )
