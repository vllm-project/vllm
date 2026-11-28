# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer fused DCP A2A + LSE reduce for MLA decode.

FlashInfer's `decode_cp_a2a_lse_reduce` exchanges partial outputs over an
NCCL symmetric-memory window and merges them in one cooperative kernel. Its
workspace comes from PyTorch symmetric memory with the NCCL backend, which the
worker selects before vLLM's other symmetric-memory users allocate.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.device_communicators.nvlink_fabric import (
    SymmetricMemoryTopology,
    get_symmetric_memory_topology,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.flashinfer import has_flashinfer_dcp_a2a_lse_reduce
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator


def dcp_valid_token_mask(
    num_rows: int,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-row flag: the row belongs to a request with local KV on this rank.

    Rows past `query_start_loc[-1]` (CUDA graph padding) and rows of requests
    with no local KV (short sequences under DCP interleaving) are False.
    """
    if seq_lens is None or query_start_loc is None:
        return None
    if seq_lens.shape[0] == 0:
        return torch.zeros(num_rows, dtype=torch.bool, device=device)
    rows = torch.arange(num_rows, device=device, dtype=query_start_loc.dtype)
    seq_idx = torch.searchsorted(query_start_loc[1:], rows, right=True).clamp_max(
        seq_lens.shape[0] - 1
    )
    return (rows < query_start_loc[-1]) & (seq_lens[seq_idx] > 0)


class FlashInferDCPWorkspace:
    """FlashInfer DCP workspaces shared by all MLA layers of a model."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        if not has_flashinfer_dcp_a2a_lse_reduce():
            raise RuntimeError(
                "dcp_comm_backend='flashinfer_a2a_fused' requires FlashInfer >= 0.7.6."
            )
        from flashinfer.comm import dcp_lse_reduce

        self._fi = dcp_lse_reduce
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        # FlashInfer binds a workspace to the CUDA stream of its first call.
        # vLLM runs eager steps, CUDA graph capture, and DBO ubatches on
        # different streams, so keep one workspace per (stream, ubatch).
        self._workspaces: dict[tuple[int, int], torch.Tensor] = {}
        # Per ubatch: (key, mask, objects whose ids are in the key).
        self._masks: dict[int, tuple[tuple[Any, ...], torch.Tensor | None, Any]] = {}

    def _workspace(self) -> torch.Tensor:
        stream = torch.cuda.current_stream(self.device).cuda_stream
        key = (stream, dbo_current_ubatch_id())
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace
        # Creation is a collective and FlashInfer's first call sets up its NCCL
        # device communicator, so neither may run under capture. vLLM's eager
        # warmup on the capture stream reaches this point first.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer DCP workspace for this stream was not created before "
                "CUDA graph capture"
            )
        workspace = self._fi.decode_cp_a2a_lse_reduce_create_workspace(
            self.max_num_tokens,
            self.heads_per_rank,
            self.world_size,
            self.head_dim,
            self.dtype,
            self.group,
        )
        self._workspaces[key] = workspace
        return workspace

    def _token_mask(
        self,
        num_rows: int,
        seq_lens: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not is_forward_context_available():
            return dcp_valid_token_mask(
                num_rows, seq_lens, query_start_loc, self.device
            )
        # The MLA layers of one forward pass share its metadata, so build the
        # mask once per forward pass and ubatch. Metadata is only updated
        # between forward passes, and each pass has its own forward context.
        # The caller re-slices query_start_loc per layer, so key it by value.
        ctx = get_forward_context()
        qsl = (
            None
            if query_start_loc is None
            else (query_start_loc.data_ptr(), query_start_loc.shape[0])
        )
        key = (
            id(ctx),
            id(seq_lens),
            qsl,
            num_rows,
            torch.cuda.is_current_stream_capturing(),
        )
        ubatch = dbo_current_ubatch_id()
        cached = self._masks.get(ubatch)
        if cached is None or cached[0] != key:
            mask = dcp_valid_token_mask(
                num_rows, seq_lens, query_start_loc, self.device
            )
            # Holding the keyed objects keeps their ids from being reused.
            cached = (key, mask, (ctx, seq_lens))
            self._masks[ubatch] = cached
        return cached[1]

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_rows = partial_output.shape[0]
        # vLLM orders the gathered heads by destination rank, which is
        # FlashInfer's destination-major [T, cp, H, D] layout.
        partial_output = partial_output.unflatten(
            1, (self.world_size, self.heads_per_rank)
        )
        partial_lse = partial_lse.unflatten(1, (self.world_size, self.heads_per_rank))
        # FlashInfer requires contiguous inputs; FlashMLA's output and LSE
        # already are, so these copies only run for other backends.
        if not partial_output.is_contiguous():
            partial_output = partial_output.contiguous()
        if not partial_lse.is_contiguous():
            partial_lse = partial_lse.contiguous()
        mask = self._token_mask(num_rows, seq_lens, query_start_loc)
        return self._fi.decode_cp_a2a_lse_reduce(
            partial_output,
            partial_lse,
            self._workspace(),
            self.rank,
            self.world_size,
            "basee" if is_lse_base_on_e else "base2",
            mask=mask,
            layout="dst_major",
        )


@functools.cache
def get_flashinfer_dcp_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
) -> FlashInferDCPWorkspace:
    # FlashInfer reads peer buffers directly over NVLink.
    if (
        get_symmetric_memory_topology(group.cpu_group)
        is SymmetricMemoryTopology.UNSUPPORTED
    ):
        raise ValueError(
            "dcp_comm_backend='flashinfer_a2a_fused' requires the DCP ranks to "
            "share an NVLink domain."
        )
    return FlashInferDCPWorkspace(
        group.device_group, device, max_num_tokens, heads_per_rank, head_dim, dtype
    )
