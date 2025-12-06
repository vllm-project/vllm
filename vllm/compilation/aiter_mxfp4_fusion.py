# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_i32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int32, device="cuda")


def empty_fp4(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.uint8, device="cuda")


class SiluMulMXFP4GemmPattern:
    def __init__(self):
        pass

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                torch.ops._C.silu_and_mul.default, result=result_silu_mul, input=input
            )
            at2 = auto_functionalized(
                torch.ops.vllm.gemm_with_dynamic_quant.default,
                result=result,
                x=at1[1],
                weight=weight,
                weight_scale=scale,
                x_scales=None,
            )
            return at2[1]

        def replacement(
            result: torch.Tensor,
            result_silu_mul: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                torch.ops.vllm.silu_and_mul_mxfp4_gemm.default,
                result=result,
                x=input,
                weight=weight,
                weight_scale=scale,
            )
            return at[1]

        inputs = [
            empty_bf16(5, 4),  # result
            empty_bf16(5, 4),  # result_silu_mul
            empty_bf16(5, 4),  # input
            empty_fp4(5, 4),  # weight
            empty_fp4(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


ADD_RMS_OP = torch.ops._C.fused_add_rms_norm.default


class AddRMSNormMXFP4GemmPattern:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.FUSED_OP = torch.ops.vllm.add_rmsnorm_mxfp4_gemm.default
        self.QUANT_F4GEMM_OP = torch.ops.vllm.gemm_with_dynamic_quant.default

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight_rms: torch.Tensor,
            weight_gemm: torch.Tensor,
            scale: torch.Tensor,
        ):
            at1 = auto_functionalized(
                ADD_RMS_OP,
                input=input,
                residual=residual,
                weight=weight_rms,
                epsilon=self.epsilon,
            )
            at2 = auto_functionalized(
                self.QUANT_F4GEMM_OP,
                result=result,
                x=at1[1],
                weight=weight_gemm,
                weight_scale=scale,
                x_scales=None,
            )
            return at2[1], at1[2]

        def replacement(
            result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            weight_rms: torch.Tensor,
            weight_gemm: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                residual=residual,
                residual_out=residual,
                weight_rms=weight_rms,
                weight_gemm=weight_gemm,
                scale=scale,
                epsilon=self.epsilon,
            )
            return at[1], at[2]

        inputs = [
            empty_bf16(4, 4),  # result
            empty_bf16(4, 4),  # input
            empty_bf16(4, 4),  # residual
            empty_bf16(1, 4),  # weight_rms
            empty_fp4(4, 4),  # weight_gemm
            empty_fp4(1, 1),  # scale
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class MXFP4FusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_fusion_pass"
        )

        SiluMulMXFP4GemmPattern().register(self.patterns)

        for epsilon in [1e-5, 1e-6]:
            AddRMSNormMXFP4GemmPattern(epsilon).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(self, SiluMulMXFP4GemmPattern)
