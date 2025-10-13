# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


# TODO(bnell): Add shared + fused combo function? e.g. +
class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts
        # Disable shared expert overlap if EP is disabled or we are not using
        # flashinfer + DP since there is nothing to be gained in this case.
        # Disabling the overlap optimization also prevents the shared experts
        # from being hidden from torch.compile.
        self.use_overlapped = (
            use_overlapped
            and not (self.use_ep or self.use_flashinfer_cutlass_kernels)
            and self._shared_experts is not None
        )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return self._shared_experts if self.use_overlapped else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # Reduce shared expert outputs if necessary, since the MLP
                # should have been created with reduce_results=False.
                if (
                    self.reduce_results
                    and self.tp_size > 1
                    and self.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return shared_out, fused_out
