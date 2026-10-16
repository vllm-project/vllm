# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA JIT MNNVL collective with Beam's FP32 sandwich-norm epilogue."""

import functools
import hashlib
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv


@functools.cache
def _module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    source = Path(__file__).with_suffix(".cu")
    communication_header = (
        jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"
    )
    digest = hashlib.sha256(
        source.read_bytes() + communication_header.read_bytes()
    ).hexdigest()[:16]
    flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10]
    )
    spec = gen_jit_spec(
        f"beam_allreduce_{digest}",
        [source],
        extra_cuda_cflags=[*flags, "--fmad=false"],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
    spec.extra_cuda_cflags.remove("-use_fast_math")
    return spec.build_and_load()


class BeamAllReduceWorkspace:
    """Reuse FlashInfer's multicast allocation and Lamport generation protocol."""

    @classmethod
    def create(
        cls,
        group: torch.distributed.ProcessGroup,
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
        self, group: torch.distributed.ProcessGroup, hidden_size: int, max_tokens: int
    ) -> None:
        from flashinfer.comm import create_allreduce_fusion_workspace
        from flashinfer.comm.mnnvl import TorchDistBackend

        self.world_size = group.size()
        self.rank = torch.distributed.get_rank(group)
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        # Scale one-shot cutoffs by the BF16 payload staged across all ranks.
        token_bytes = hidden_size * 2
        rank_token_bytes = self.world_size * token_bytes
        self.small_batch_max_tokens = 3200 * 1024 // rank_token_bytes
        self.ep_finalize_oneshot_max_tokens = 2560 * 1024 // rank_token_bytes
        # Two-shot reserves two rows per token; one-shot needs one per rank.
        workspace_tokens = max(
            max_tokens,
            min(max_tokens, self.small_batch_max_tokens) * self.world_size // 2,
        )
        self.workspace = create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=self.world_size,
            rank=self.rank,
            max_token_num=workspace_tokens,
            hidden_dim=hidden_size,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group=group),
            group=group,
        )
        self.module = _module()
        self.allocation_bytes = (
            self.workspace.handle.mcast_device_memory.get_allocation_size()
        )
        self.separate_norm_min_tokens = cdiv(
            10 * 1024**2 * self.world_size, token_bytes
        )
        self.ep_finalize_max_tokens = 640 * 1024**2 // token_bytes


def beam_allreduce_norm_residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float | None,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float] | None,
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
        from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

        from .layernorm import beam_norm_residual_norm

        reduced = allreduce_fusion(
            input=x,
            workspace=workspace.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
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
    oneshot_max_tokens: int = 0,
):
    from flashinfer.comm.trtllm_mnnvl_ar import MNNVLAllreduceFusionStrategy

    assert x.dtype == residual.dtype == torch.bfloat16
    assert x.is_contiguous() and residual.is_contiguous()
    m, n = x.shape
    assert n == workspace.hidden_size and m <= workspace.max_tokens
    hidden = torch.empty_like(x)
    normalized = torch.empty_like(x) if next_eps is not None else None
    if m:
        strategy = MNNVLAllreduceFusionStrategy.select_strategy(
            workspace.world_size, m, n, x.dtype
        )
        small_batch = unfinalized is None and m <= workspace.small_batch_max_tokens
        w = workspace.workspace
        workspace.module.beam_allreduce(
            x,
            residual,
            post_weight,
            next_weight,
            hidden,
            normalized,
            w.uc_ptrs_dev,
            w.buffer_flags,
            w.mc_ptr,
            w.uc_ptr_local,
            workspace.world_size,
            workspace.rank,
            post_eps or 0.0,
            next_eps or 0.0,
            *(scales if scales is not None else (1.0, 1.0)),
            post_eps is not None,
            next_eps is not None,
            scales is not None,
            small_batch
            or m <= oneshot_max_tokens
            or strategy == MNNVLAllreduceFusionStrategy.ONESHOT,
            unfinalized.gemm2_permuted if unfinalized is not None else None,
            unfinalized.expert_weights if unfinalized is not None else None,
            unfinalized.expanded_idx_to_permuted_idx
            if unfinalized is not None
            else None,
            routed_scale,
            small_batch,
        )
    return hidden, normalized


def beam_finalize_allreduce_norm_residual_norm(
    x: UnfinalizedMoEOutput,
    residual: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float | None,
    next_weight: torch.Tensor | None,
    next_eps: float | None,
    scales: tuple[float, float] | None,
    workspace: BeamAllReduceWorkspace,
    routed_scale: float,
    *,
    oneshot_max_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize local experts before AR, retaining the local BF16 rounding."""
    m, n = residual.shape
    assert 0 <= oneshot_max_tokens <= workspace.small_batch_max_tokens
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
        oneshot_max_tokens,
    )
