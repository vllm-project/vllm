# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

# TODO: add comments


class FusedMoEQuantizeDispatchCombine(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # TODO: figure this out
        # returns (quantized+dispatched a, quantized+dispatched a1_scales, dispatched topk_ids)
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,  # not reduced or weighted
        topk_weights: torch.Tensor,
    ) -> None:
        raise NotImplementedError


# store weights, etc. here
class FusedMoEPermuteExpertsUnpermute(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def workspace_shapes(self, M: int, N: int, K: int, topk: int,
                         num_experts: int) -> Tuple[int, int, torch.dtype]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        a1q: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


# Note: only intended for use with a single model layer (due to temp buffers, constants, etc.)
# TODO: permute/unpermute must be paired
class FusedMoEModularKernel(torch.nn.Module):  # should this be a module?

    def __init__(
        self,
        dispatch_combine: FusedMoEQuantizeDispatchCombine,
        fused_experts: FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__()
        self.dispatch_combine = dispatch_combine
        self.fused_experts = fused_experts

    def forward(
        self,
        a1: torch.Tensor,  # aka hidden states
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        M, _ = a1.shape
        E, K, N = w2.shape
        if global_num_experts == -1:
            global_num_experts = E
        top_k = topk_ids.shape[1]

        if inplace:
            output = a1
        else:
            output = torch.empty_like(a1)

        workspace13_shape, workspace2_shape, workspace_dtype = (
            self.fused_experts.workspace_shapes(M, N, K, top_k,
                                                global_num_experts))

        # We can reuse the memory between cache1 and cache3 because by the time
        # we need cache3, we're done with cache1
        workspace13 = torch.empty(workspace13_shape,
                                  device=a1.device,
                                  dtype=workspace_dtype)
        workspace2 = torch.empty(workspace2_shape,
                                 device=a1.device,
                                 dtype=workspace_dtype)

        a1q, a1q_scale, dispatched_topk_ids = self.dispatch_combine.dispatch(
            a1,
            a1_scale,
            a2_scale,
            topk_ids,
            global_num_experts,
            expert_map,
        )

        fused_out = self.fused_experts.apply(
            a1q,
            w1,
            w2,
            dispatched_topk_ids,
            activation,
            expert_map,
            w1_scale,
            w2_scale,
            a1q_scale,
            a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
        )

        self.dispatch_combine.combine(output, fused_out, topk_weights)

        return output
