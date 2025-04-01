from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch


class FusedMoEDispatchQuantize(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(
            self,
            hidden_states,
            hidden_states_scales,
            topk_ids,
            num_experts,
            expert_map,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # returns (hidden_states, scales, sorted_token_ids, expert_ids, inv_perm) # make more abstract?
        raise NotImplementedError


# store weights, etc. here
class FusedMoEExperts(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self):
        raise NotImplementedError


class FusedMoEUnpermuteCombine(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(
            self,
            out,
            hidden_states,
            topk_weights,
            topk,
            inv_perm,
    ) -> torch>Tensor:
        raise NotImplementedError


class ModularFusedMoEKernel(torch.nn.Module): # should this be a module?
    def __init__(
            self,
            dispatch: FusedMoEDispatchQuantize,
            fused_experts: FusedMoEExperts,
            combine: FusedMoEUnpermuteCombine,
    ):
        self.dispatch = dispatch
        self.fused_experts = fused_experts
        self.combine = combine

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
        self.dispatch()

        fused_out = self.fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace,
            activation,
            global_num_experts,
            expert_map,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
        )

        self.combine(hidden_states, fused_out)
        return hidden_states
