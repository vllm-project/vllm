# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Intel XPU WNA16 (W4A16) fused-MoE method for AutoRound checkpoints.

AutoRound stores MoE expert weights with GPTQ suffixes (``qweight`` /
``scales`` / ``qzeros`` / ``g_idx``). :class:`MoeWNA16Method` already registers
those parameter names and, through its wrapped weight loader, repacks the int32
GPTQ tensors into the uint8 ``[E, 2N, K // 2]`` layout that
``xpu_fused_moe(is_int4=True)`` expects.

This subclass keeps that proven loading path and only swaps the compute backend
from the portable Triton ``fused_experts`` to the native XPU ``XpuFusedMoe``
kernel exposed through :class:`XPUExpertsWNA16`. It mirrors the previous working
``INCXPUMoEMethod`` design while staying compatible with the current modular
fused-MoE architecture.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsWNA16
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config


class INCXPUWNA16MoEMethod(MoeWNA16Method):
    """W4A16 INT4-symmetric group MoE executed by the native XPU kernel.

    Inherits weight creation/loading (GPTQ-named uint8 layout) from
    :class:`MoeWNA16Method` and overrides :meth:`apply` to dispatch to
    :class:`XPUExpertsWNA16`, which wraps ``xpu_fused_moe(is_int4=True)``.
    """

    def __init__(
        self, quant_config: "MoeWNA16Config", moe: "FusedMoEConfig"
    ) -> None:
        super().__init__(quant_config, moe)
        self._xpu_experts: XPUExpertsWNA16 | None = None

    def _get_xpu_experts(self) -> "XPUExpertsWNA16":
        if self._xpu_experts is None:
            from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
                XPUExpertsWNA16,
            )

            assert self.moe_quant_config is not None, (
                "moe_quant_config must be initialised before apply(); it is "
                "populated from get_fused_moe_quant_config() after weight load."
            )
            self._xpu_experts = XPUExpertsWNA16(self.moe, self.moe_quant_config)
        return self._xpu_experts

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if shared_experts is not None:
            raise NotImplementedError(
                "INCXPUWNA16MoEMethod does not support fused shared experts."
            )

        experts = self._get_xpu_experts()
        output = torch.empty_like(x)
        # XPUExpertsWNA16 runs the full fused MoE inside xpu_fused_moe, so the
        # modular workspaces are unused; pass empty placeholders.
        empty = x.new_empty(0)
        experts.apply(
            output=output,
            hidden_states=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=empty,
            workspace2=empty,
            expert_tokens_meta=None,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
        return output
