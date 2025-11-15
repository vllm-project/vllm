# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.activation_quant_fusion import ActivationQuantPattern
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .fusion import empty_bf16
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherSiluAndMul
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()

if rocm_aiter_ops.is_enabled():
    AITER_RMS_GROUP_QUANT_OP = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant.default
    AITER_RMS_ADD_GROUP_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default
    )

    AITER_RMS_OP = torch.ops.vllm.rocm_aiter_rms_norm.default
    AITER_RMS_ADD_OP = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default

    AITER_GROUP_FP8_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default

    FUSED_SILU_MUL_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant.default
    )


class AiterRMSFp8GroupQuantPattern:
    def __init__(self, epsilon: float, quant_dtype: torch.dtype, use_triton: bool):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype
        self.use_triton = use_triton

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            at1 = AITER_RMS_OP(x=input, weight=weight, variance_epsilon=self.epsilon)

            at2 = AITER_GROUP_FP8_QUANT_OP(at1, 128, self.use_triton)

            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            at = AITER_RMS_GROUP_QUANT_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            return at[0], at[1]

        inputs = [
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class AiterFusedAddRMSFp8GroupQuantPattern:
    def __init__(self, epsilon: float, quant_dtype: torch.dtype, use_triton: bool):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype
        self.use_triton = use_triton

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ):
            at1 = AITER_RMS_ADD_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
            )

            at2 = AITER_GROUP_FP8_QUANT_OP(at1[0], 128, self.use_triton)

            # result, scale, residual
            return at2[0], at2[1], at1[1]

        def replacement(
            input: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
        ):
            at = AITER_RMS_ADD_GROUP_QUANT_OP(
                x=input,
                residual=residual,
                weight=weight,
                variance_epsilon=self.epsilon,
                group_size=128,
            )

            # result, scale, residual
            return at[0], at[1], at[2]

        inputs = [
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(1, 5),  # weight
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class RocmAiterRMSNormFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses rms_norm & quant custom ops into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_fp8_group_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            # Fuse rms_norm + dynamic group fp8 quant
            for use_triton in [False, True]:
                AiterRMSFp8GroupQuantPattern(epsilon, FP8_DTYPE, use_triton).register(
                    self.patterns
                )

                AiterFusedAddRMSFp8GroupQuantPattern(
                    epsilon, FP8_DTYPE, use_triton
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        fusion_patterns = [
            AiterRMSFp8GroupQuantPattern,
            AiterFusedAddRMSFp8GroupQuantPattern,
        ]
        return self.hash_source(self, *fusion_patterns)


class AiterSiluMulFp8GroupQuantPattern(ActivationQuantPattern):
    def __init__(self, use_triton: bool):
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.use_triton = use_triton

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
        ):
            at1 = self.silu_and_mul_matcher(input)
            at2 = AITER_GROUP_FP8_QUANT_OP(at1, 128, self.use_triton)
            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
        ):
            at = FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
            return at[0], at[1]

        inputs = [
            self.silu_and_mul_matcher.inputs()[0],
        ]

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class RocmAiterSiluMulFp8GroupQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_silu_mul_fp8_group_quant_fusion_pass"
        )

        for use_triton in [False, True]:
            AiterSiluMulFp8GroupQuantPattern(use_triton).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self):
        fusion_patterns = [
            ActivationQuantPattern,
            AiterSiluMulFp8GroupQuantPattern,
        ]
        return VllmInductorPass.hash_source(self, *fusion_patterns)
