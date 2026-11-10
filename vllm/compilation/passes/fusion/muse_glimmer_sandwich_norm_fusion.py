# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fuse Muse's Gemma-style RMSNorm, residual add, and second RMSNorm."""

import torch

import vllm.ir.ops
from vllm.config import VllmConfig

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)


class MuseSandwichNormPattern(VllmPatternReplacement):
    def __init__(
        self,
        hidden_size: int,
        post_eps: float,
        pre_eps: float,
        fused_add: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.post_eps = post_eps
        self.pre_eps = pre_eps
        self.fused_add = fused_add

    @property
    def pattern(self):
        post_eps = self.post_eps
        pre_eps = self.pre_eps
        fused_add = self.fused_add

        def _pattern(x, residual, post_weight, pre_weight):
            post_norm = vllm.ir.ops.rms_norm(x, post_weight.float() + 1.0, post_eps)
            if fused_add:
                return vllm.ir.ops.fused_add_rms_norm(
                    post_norm, residual, pre_weight.float() + 1.0, pre_eps
                )
            residual_out = residual + post_norm
            pre_norm = vllm.ir.ops.rms_norm(
                residual_out, pre_weight.float() + 1.0, pre_eps
            )
            return pre_norm, residual_out

        return _pattern

    @property
    def replacement(self):
        post_eps = self.post_eps
        pre_eps = self.pre_eps

        def _replacement(x, residual, post_weight, pre_weight):
            from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
                fused_rmsnorm_add_rmsnorm,
            )

            residual_out = torch.empty_like(x)
            pre_norm = fused_rmsnorm_add_rmsnorm(
                x,
                residual,
                post_weight,
                pre_weight,
                post_eps,
                pre_eps,
                residual_out,
            )
            return pre_norm, residual_out

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        width = self.hidden_size
        return [
            self.empty_bf16(5, width),
            self.empty_bf16(5, width),
            self.empty_bf16(width),
            self.empty_bf16(width),
        ]


class MuseSandwichNormFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "muse_sandwich_norm_fusion_pass")

        from vllm.model_executor.models.muse_glimmer import _text_config

        text_config = _text_config(config.model_config.hf_config)
        for fused_add in (True, False):
            self.register(
                MuseSandwichNormPattern(
                    text_config.hidden_size,
                    text_config.post_norm_eps,
                    text_config.rms_norm_eps,
                    fused_add,
                )
            )

        self.dump_patterns(config, self.pm_pass)
