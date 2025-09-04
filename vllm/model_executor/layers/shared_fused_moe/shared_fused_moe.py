# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

def post_process(shared: Optional[torch.Tensor], fused: torch.Tensor) -> torch.Tensor:
    return shared + fused if shared is not None else fused

class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        shared_fused_combine: Optional[Callable] = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts
        self._shared_fused_combine = shared_fused_combine if shared_fused_combine is not None else post_process
        self.use_overlapped = use_overlapped

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return self._shared_experts if self.use_overlapped else None

    @property
    def shared_fused_combine(self) -> Optional[Callable]:
        return self._shared_fused_combine if self.use_overlapped else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_overlapped:
            shared_out = self._shared_experts(hidden_states)

            # Reduce outputs if necessary, since the MLP should
            # have been created with reduce_results=False.
            if (self.reduce_results and self.tp_size > 1
                    and self.must_reduce_shared_expert_outputs()):
                # TODO: can this be async?
                shared_out = tensor_model_parallel_all_reduce(shared_out)

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            output = self._shared_fused_combine(shared_out, fused_out)

            if self.tp_size > 1 and self.must_reduce_shared_expert_outputs():
                tensor_model_parallel_all_reduce(output)

            return output
        else:
            return super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
