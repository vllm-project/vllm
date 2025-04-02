from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import torch


class FusedMoEQuantizeDispatchCombine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def dispatch(
            self,
            a: torch.Tensor,
            a1_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            topk_ids: torch.Tensor,
            num_experts: int,
            expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # TODO: figure this out
        # returns (quantized+dispatched hidden_states, quantized+dispatched scales, dispatched topk_ids)
        raise NotImplementedError

    @abstractmethod
    def combine(
            self,
            out: torch.Tensor,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


# store weights, etc. here
class FusedMoEPermuteExpertsUnpermute(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def workspace_shapes(
            self,
            M: int,
            N: int,
            K: int,
            topk: int,
            num_experts: int
    ) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def apply(
            self,
            out: torch.Tensor,
            q_hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_ids: torch.Tensor,
            inplace: bool,
            activation: str,
            expert_map: Optional[torch.Tensor],
            w1_scale: Optional[torch.Tensor],
            w2_scale: Optional[torch.Tensor],
            a1_scale: Optional[torch.Tensor],
            a2_scale: Optional[torch.Tensor],
            workspace13: torch.Tensor,
            workspace2: torch.Tensor,
    ) -> torch.Tensor: # or None?  assume inplace?
        raise NotImplementedError


# Note: only intended for use with a single model layer (due to temp buffers, constants, etc.)
# TODO: permute/unpermute must be paired
class FusedMoEModularKernel(torch.nn.Module): # should this be a module?
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
            hidden_states: torch.Tensor,
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
        M, _ = hidden_states.shape
        E, K, N = w2.shape
        if global_num_experts == -1:
            global_num_experts = E
        top_k = topk_ids.shape[1]

        assert not inplace, "NYI"

        if inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        #print(f"TKN = {topk_ids.numel()} {M*top_k}")

        workspace13_shape, workspace2_shape = (
            self.fused_experts.workspace_shapes(
                M, N, K, top_k, global_num_experts)
        )

        # We can reuse the memory between cache1 and cache3 because by the time
        # we need cache3, we're done with cache1
        workspace13 = torch.empty(workspace13_shape,
                                  device=hidden_states.device,
                                  dtype=hidden_states.dtype)
        workspace2 = torch.empty(workspace2_shape,
                                 device=hidden_states.device,
                                 dtype=hidden_states.dtype)

        #print(f"\nbefore M = {hidden_states.shape[0]}")

        hidden_states, a1_scale, new_topk_ids = self.dispatch_combine.dispatch(
            a=hidden_states,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            topk_ids=topk_ids,
            num_experts=global_num_experts,
            expert_map=expert_map,
        )

        #print(f"after M = {hidden_states.shape[0]}")

        fused_out = self.fused_experts.apply(
            out=hidden_states,
            q_hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_ids=new_topk_ids,
            inplace=inplace,
            activation=activation,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
        )

        return self.dispatch_combine.combine(out_hidden_states, fused_out, topk_weights)
