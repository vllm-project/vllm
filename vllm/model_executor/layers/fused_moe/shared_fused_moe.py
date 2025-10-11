# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def post_process(
        self,
        shared_output: Optional[torch.Tensor],
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.fused_output_scaling_factor != 1.0:
            fused_output *= self.fused_output_scaling_factor

        if shared_output is not None:
            if self.shared_output_scaling_factor != 1.0:
                shared_output *= self.shared_output_scaling_factor

            fused_output += shared_output

        return fused_output

    def __init__(
        self,
        shared_experts: Optional[torch.nn.Module],
        use_overlapped: bool = True,
        fused_output_scaling_factor: float = 1.0,
        shared_output_scaling_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts
        self._shared_fused_combine = lambda a, b: self.post_process(a, b)
        # Disable shared expert overlap if EP is disabled or we are not using
        # flashinfer + DP since there is nothing to be gained in this case.
        # Disabling the overlap optimization also prevents the shared experts
        # from being hidden from torch.compile.
        self.use_overlapped = (
            use_overlapped
            and not (self.use_ep or self.use_flashinfer_cutlass_kernels)
            and self._shared_experts is not None
        )
        self.fused_output_scaling_factor = fused_output_scaling_factor
        self.shared_output_scaling_factor = shared_output_scaling_factor

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
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # Reduce shared expert outputs if necessary, since the MLP
                # should have been created with reduce_results=False.
                if self.tp_size > 1 and self.must_reduce_shared_expert_outputs():
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

            if self.shared_experts is not None:
                assert shared_out is not None
                output = self._shared_fused_combine(shared_out, fused_out)
            else:
                assert self.shared_output_scaling_factor == 1.0
                output = fused_out * self.fused_output_scaling_factor

            if self.tp_size > 1 or self.ep_size > 1:
                output = self.maybe_all_reduce_tensor_model_parallel(output)
        else:
            output = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        return output
