# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.ir.ops
from vllm.config import VllmConfig

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)


class AddRMSNormPattern(VllmPatternReplacement):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    @property
    def pattern(self):
        def _pattern(
            branch: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual_out = residual + branch
            rms = vllm.ir.ops.rms_norm(residual_out, weight, self.epsilon)
            return rms, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            branch: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return vllm.ir.ops.fused_add_rms_norm(
                branch, residual, weight, self.epsilon
            )

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        return [
            self.empty_bf16(5, 16),  # branch
            self.empty_bf16(5, 16),  # residual
            self.empty_bf16(16),  # weight
        ]


class AddRMSNormFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "add_rmsnorm_fusion_pass")

        for epsilon in [1e-5, 1e-6]:
            self.register(AddRMSNormPattern(epsilon))

        self.dump_patterns(config, self.pm_pass)
