# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash adapter for the DeepGEMM MegaMoE kernel."""

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)


class Glm5NextMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts that also own GLM-5.3's shared experts.

    GLM builds no shared MLP under MegaMoE: the launch fuses
    `num_shared_experts` shared experts in the routed precision, and their
    checkpoint tensors load straight into the experts.
    """

    def __init__(
        self, vllm_config: VllmConfig, *, num_shared_experts: int = 0, **kwargs
    ):
        self._num_fused_shared_experts = num_shared_experts
        super().__init__(vllm_config, **kwargs)

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        n = self._num_fused_shared_experts
        return n, self.weight_format.routed if n else None
