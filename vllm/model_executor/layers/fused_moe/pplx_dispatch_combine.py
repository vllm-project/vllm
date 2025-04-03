import torch
from typing import Optional, Tuple

import pplx_kernels as pplx
import vllm.model_executor.layers.fused_moe.modular_kernel as mk


class PplxDispatchCombine(mk.FusedMoEQuantizeDispatchCombine):
    def __init__(self, a2a: pplx.AllToAll):
        super().__init__()
        self.a2a = a2a

    def dispatch(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        self.a2a.dispatch(
            out_expert_num_tokens, # torch.Tensor,
            out_expert_x, # torch.Tensor,
            out_expert_x_scale, # torch.Tensor | None,
            dp_x, # torch.Tensor,
            dp_x_scale, # torch.Tensor | None,
            indices, # torch.Tensor,
            bound_m, # torch.Tensor | None,
            do_send, # bool = True,
            do_recv, # bool = True,
        )
        return 1q, a1q_scale, topk_ids

    def combine(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        self.a2a.combine(
            out_tokens, #: torch.Tensor,
            indices, #: torch.Tensor,
            weights, #: torch.Tensor,
            expert_y, #: torch.Tensor,
            bound_m, #: torch.Tensor | None,
            do_send, #: bool = True,
            do_recv, #: bool = True,
        )


# singleton-ish
def get_a2a(
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        dp_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
) -> pplx.AllToAll:
    pass
