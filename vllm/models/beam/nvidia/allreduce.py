# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MNNVL all-reduce with Beam's FP32 sandwich-norm epilogue."""

import torch
import torch.distributed as dist

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_up

_NUM_LAMPORT_BUFFERS = 3
# FlashInfer's MNNVL one-shot cutoff on the BF16 payload staged across ranks.
_ONESHOT_MAX_BYTES = 1024 * 1024


class BeamAllReduceWorkspace:
    """Multicast Lamport buffers and generation flags for the Beam kernels."""

    @classmethod
    def create(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        max_tokens: int,
        dtype: torch.dtype,
    ) -> "BeamAllReduceWorkspace | None":
        # The CUDA epilogue handles one packed load per thread.
        if (
            group.size() not in (2, 4, 8)
            or dtype != torch.bfloat16
            or hidden_size % 1024
            or not 1024 <= hidden_size <= 8192
            or not hasattr(torch.ops._C, "beam_allreduce_norm")
            or not has_flashinfer()
        ):
            return None
        from flashinfer.comm.mnnvl import (
            TorchDistBackend,
            all_ranks_agree,
            is_multicast_supported,
        )

        device = torch.accelerator.current_device_index()
        supported = torch.cuda.get_device_capability(device)[0] in (9, 10)
        supported = supported and is_multicast_supported(device)
        if not all_ranks_agree(TorchDistBackend(group=group), supported):
            return None
        return cls(group, hidden_size, max_tokens)

    def __init__(
        self, group: dist.ProcessGroup, hidden_size: int, max_tokens: int
    ) -> None:
        self.world_size = group.size()
        self.rank = dist.get_rank(group)
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        # Scale one-shot cutoffs by the BF16 payload staged across all ranks.
        token_bytes = hidden_size * 2
        rank_token_bytes = self.world_size * token_bytes
        self.oneshot_max_tokens = _ONESHOT_MAX_BYTES // rank_token_bytes
        self.small_batch_max_tokens = 3200 * 1024 // rank_token_bytes
        self.separate_norm_min_tokens = cdiv(
            10 * 1024**2 * self.world_size, token_bytes
        )

        # One-shot stages one row per rank; two-shot two rows per owned token.
        oneshot_tokens = min(max_tokens, self.small_batch_max_tokens)
        buffer_bytes = max(
            oneshot_tokens * rank_token_bytes,
            2 * cdiv(max_tokens, self.world_size) * rank_token_bytes,
        )
        # Both two-shot stages stay 16-byte aligned.
        buffer_bytes = round_up(buffer_bytes, 32)
        if buffer_bytes >= 2**32:
            raise ValueError(
                f"Beam all-reduce needs {buffer_bytes} bytes per Lamport "
                "buffer, which exceeds the 32-bit flag field"
            )
        # FlashInfer only maps the multicast memory; the Lamport protocol and
        # its flags are ours. Small batches reduce ~0.3us faster on its mapping
        # than on torch symmetric memory.
        from flashinfer.comm.mnnvl import McastGPUBuffer, TorchDistBackend

        device = torch.device("cuda", torch.accelerator.current_device_index())
        self.buffer = McastGPUBuffer(
            _NUM_LAMPORT_BUFFERS * buffer_bytes,
            self.world_size,
            self.rank,
            device,
            TorchDistBackend(group=group),
        )
        self.multicast_ptr = self.buffer.get_multicast_ptr()
        if not self.multicast_ptr:
            raise RuntimeError("Beam all-reduce requires NVLink multicast")
        self.local_ptr = self.buffer.get_unicast_ptr(self.rank)
        self.buffer_ptrs = self.buffer.get_buffer_ptrs_dev()
        self.buffer.lamport_initialize(self.rank, torch.float32)
        # [current, dirty, bytes per buffer, dirty stages, bytes to clear x4,
        # arrival count]
        self.flags = torch.tensor(
            [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=device,
        )
        torch.accelerator.synchronize()
        dist.barrier(group=group)


def beam_allreduce_norm_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float],
    workspace: BeamAllReduceWorkspace,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert x.shape == residual.shape
    for weight in (post_weight, next_weight):
        if weight is not None:
            assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
            assert weight.shape == (workspace.hidden_size,)
    assert x.dtype == residual.dtype == torch.bfloat16
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.shape[1] == workspace.hidden_size
    assert x.shape[0] <= workspace.max_tokens
    if x.shape[0] >= workspace.separate_norm_min_tokens:
        from ..common.layernorm import beam_norm_residual_norm

        reduced = torch.empty_like(x)
        torch.ops._C.beam_allreduce(
            x,
            reduced,
            workspace.buffer_ptrs,
            workspace.flags,
            workspace.multicast_ptr,
            workspace.world_size,
            workspace.rank,
        )
        return beam_norm_residual_norm(
            reduced,
            residual,
            post_weight,
            post_eps,
            next_weight,
            next_eps,
            scales,
            launch_pdl=True,
        )
    return _cuda_allreduce_norm_residual_norm(
        x, residual, post_weight, post_eps, next_weight, next_eps, scales, workspace
    )


def _cuda_allreduce_norm_residual_norm(
    x,
    residual,
    post_weight,
    post_eps,
    next_weight,
    next_eps,
    scales,
    workspace,
    unfinalized: UnfinalizedMoEOutput | None = None,
    routed_scale: float = 1.0,
):
    assert x.dtype == residual.dtype == torch.bfloat16
    assert x.is_contiguous() and residual.is_contiguous()
    m, n = x.shape
    assert n == workspace.hidden_size and m <= workspace.max_tokens
    hidden = torch.empty_like(x)
    normalized = torch.empty_like(x) if next_eps is not None else None
    if m:
        small_batch = unfinalized is None and m <= workspace.small_batch_max_tokens
        torch.ops._C.beam_allreduce_norm(
            x,
            residual,
            post_weight,
            next_weight,
            hidden,
            normalized,
            workspace.buffer_ptrs,
            workspace.flags,
            workspace.multicast_ptr,
            workspace.local_ptr,
            workspace.world_size,
            workspace.rank,
            post_eps or 0.0,
            next_eps or 0.0,
            *(scales if scales is not None else (1.0, 1.0)),
            post_eps is not None,
            next_eps is not None,
            scales is not None,
            small_batch or m <= workspace.oneshot_max_tokens,
            small_batch,
            unfinalized.gemm2_permuted if unfinalized is not None else None,
            unfinalized.expert_weights if unfinalized is not None else None,
            unfinalized.expanded_idx_to_permuted_idx
            if unfinalized is not None
            else None,
            routed_scale,
        )
    return hidden, normalized


def beam_finalize_allreduce_norm_residual_norm(
    x: UnfinalizedMoEOutput,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    post_eps: float,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float],
    workspace: BeamAllReduceWorkspace,
    routed_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize local experts before AR, retaining the local BF16 rounding."""
    m, n = residual.shape
    for weight in (post_weight, next_weight):
        if weight is not None:
            assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
            assert weight.shape == (n,)
    assert x.gemm2_permuted.dtype == torch.bfloat16
    assert x.gemm2_permuted.is_contiguous() and x.gemm2_permuted.shape[1] == n
    assert x.expert_weights.dtype in (torch.float32, torch.bfloat16)
    assert x.expanded_idx_to_permuted_idx.dtype == torch.int32
    assert x.expert_weights.is_contiguous()
    assert x.expanded_idx_to_permuted_idx.is_contiguous()
    assert x.expert_weights.shape == x.expanded_idx_to_permuted_idx.shape
    assert x.expert_weights.shape[0] == m
    return _cuda_allreduce_norm_residual_norm(
        residual,
        residual,
        post_weight,
        post_eps,
        next_weight,
        next_eps,
        scales,
        workspace,
        x,
        routed_scale,
    )
