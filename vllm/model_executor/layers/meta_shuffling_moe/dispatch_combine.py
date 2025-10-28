# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_fbgemm_gpu_gen_ai

if current_platform.is_cuda_alike() and has_fbgemm_gpu_gen_ai():
    from fbgemm_gpu.experimental.gen_ai.moe import (
        gather_scale_dense_tokens,
        scatter_add_dense_tokens,
    )


@dataclass
class RouteInfo:
    expert_indices: torch.Tensor
    token_counts: torch.Tensor
    token_indices: torch.Tensor
    num_routed_tokens: torch.Tensor
    num_recv_tokens: torch.Tensor | None = None
    recv_sizes_across_ranks: torch.Tensor | None = None
    recv_sizes_across_ranks_cpu: torch.Tensor | None = None
    send_sizes_across_ranks: torch.Tensor | None = None
    send_sizes_across_ranks_cpu: torch.Tensor | None = None


# Skeleton code to prepare for enabling EP.
# In TP only case, dispatch/combine are almost no-ops.
class MetaShufflingDispatchAndCombine:
    """
    Dispatch/Combine using Meta Shuffling kernels.
    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
            cls.instance._initialized = False
        return cls.instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.world_size = 1
        assert current_platform.is_cuda_alike() and has_fbgemm_gpu_gen_ai()
        self._initialized: bool = True

    def dispatch(
        self,
        tokens: torch.Tensor,  # tokens
        route_info: RouteInfo,
        scores: torch.Tensor,  # scores,
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if apply_router_weight_on_input:
            tokens = gather_scale_dense_tokens(
                tokens,
                route_info.token_indices.flatten(),
                route_info.expert_indices.flatten(),
                scores,
                valid_token_count=route_info.num_routed_tokens,
            )
        assert self.world_size == 1
        return tokens, route_info.token_counts

    def combine(
        self,
        routed_out: torch.Tensor,
        route_info: RouteInfo,
        scores: torch.Tensor,
        shared_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.world_size == 1
        if envs.VLLM_META_SHUFFLING_GEMM_BACKEND == "cutlass":
            scatter_add_dense_tokens(
                out_tokens=shared_out,
                in_tokens=routed_out,
                token_indices=route_info.token_indices,
                valid_token_count=route_info.num_routed_tokens,
            )
            return shared_out
        # Assume in TP only case, we have already produced
        # fused output from routed and shared by calling
        # grouped_gemm with shared output when using triton grouped_gemm.
        else:
            return routed_out
