import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

#@CustomOp.register("shared_fused_moe") ???

# TODO: add shared + fused combo function?
class SharedFusedMoE(FusedMoE):
    def __init__(self, shared_experts: torch.nn.Module, **kwargs):
        super().__init__(**kwargs, shared_experts=shared_experts)
        #self.shared_experts_fn = shared_experts # for now

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #shared_out = self.shared_experts_fn(hidden_states)
        shared_out, fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        return shared_out, fused_out
