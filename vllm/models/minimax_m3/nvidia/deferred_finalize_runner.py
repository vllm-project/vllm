# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MoE runner that defers the CuTeDSL MoE's top-k reduce.

When ``MiniMaxM3CuteDSLMethod.defer_finalize`` is set (TP1), the experts
stop after GEMM2 and this runner returns a :class:`DeferredMoETail`. The next
layer's input norm consumes it with ``moe_finalize_add_norm_mxfp8`` (reduce +
shared add + residual add + Gemma RMSNorm + MXFP8 quant in one kernel); where no
such norm follows (PP boundary, final norm, composed forward), the model
materializes it with ``moe_finalize_add``. Follows Kimi-K3's
``LatentMoERunner`` deferred-finalize pattern.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, _unpack


@dataclass
class DeferredMoETail:
    """Unfinalized MoE output handed to the next layer's input norm.

    Only produced by ``MiniMaxM3DeferredFinalizeRunner``; the model materializes
    it with ``moe_finalize_add`` at pipeline-parallel boundaries, before the
    final norm, or whenever the next layer cannot consume it.
    """

    unfinalized: UnfinalizedMoEOutput
    shared_output: torch.Tensor | None

    def materialize(self) -> torch.Tensor:
        from vllm.models.minimax_m3.nvidia.ops.moe_finalize_add import (
            moe_finalize_add,
        )

        return moe_finalize_add(
            self.unfinalized.gemm2_permuted,
            self.unfinalized.expert_weights,
            self.unfinalized.expanded_idx_to_permuted_idx,
            self.shared_output,
        )


class MiniMaxM3DeferredFinalizeRunner(MoERunner):
    """MoERunner that hands the deferred CuTeDSL MoE output to the model."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._defer = getattr(self._quant_method, "defer_finalize", False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor | DeferredMoETail:
        if not self._defer:
            return super().forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        # No routed input transform and no hidden-dim padding on this model;
        # the gate lives in the model, so router_logits are already computed.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )
        shared_output, fused_output = _unpack(
            self._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        )
        assert isinstance(fused_output, UnfinalizedMoEOutput)
        return DeferredMoETail(fused_output, shared_output)
