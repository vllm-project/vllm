# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.model_executor.layers.quantization.utils.fp8_utils  # noqa: F401
from vllm.compilation.activation_quant_fusion import ActivationQuantPattern
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform

from .fusion import (
    QUANT_OPS,
    FusedRMSQuantKey,
    RMSNormQuantPattern,
    empty_bf16,
    empty_fp32,
)
from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherSiluAndMul
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterRMSNormQuantPattern(RMSNormQuantPattern):
    RMS_OP = torch.ops.vllm.rocm_aiter_rms_norm.default
    RMS_ADD_OP = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default

    def __init__(self, epsilon, key):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        assert key.quant in QUANT_OPS, f"unsupported quantization scheme {key.quant}"
        self.QUANT_OP = QUANT_OPS[key.quant]


class AiterRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
        FusedRMSQuantKey(
            kFp8DynamicTokenSym,
            False,
        ): torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant.default,
    }

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        assert key in self.FUSED_OPS, (
            f"unsupported fused aiter rmsnorm+quant op for {key}"
        )
        self.FUSED_OP = self.FUSED_OPS[key]

        super().__init__(epsilon, key)

    def register(self, pm_pass):
        def pattern(
            result: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            rms_out = self.RMS_OP(
                x=input,
                weight=weight,
                variance_epsilon=self.epsilon,
            )

            at = auto_functionalized(
                self.QUANT_OP, result=result, input=rms_out, scale=scale, scale_ub=None
            )

            return at[1], at[2]

        def replacement(
            result: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                x=input,
                weight=weight,
                y_scale=scale,
                epsilon=self.epsilon,
            )

            return at[1], at[2]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # input
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class AiterFusedAddRMSNormDynamicQuantPattern(AiterRMSNormQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
        FusedRMSQuantKey(
            kFp8DynamicTokenSym,
            True,
        ): torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant.default,
    }

    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )

        assert key in self.FUSED_OPS, (
            f"unsupported fused aiter rmsnorm+quant op for {key}"
        )
        self.FUSED_OP = self.FUSED_OPS[key]

        super().__init__(epsilon, key)

    def register(self, pm_pass):
        def pattern(
            result: torch.Tensor,
            rms_result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            residual_out: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.RMS_ADD_OP,
                out=rms_result,
                x=input,
                residual=residual,
                residual_out=residual_out,
                weight=weight,
                variance_epsilon=self.epsilon,
            )

            at1 = auto_functionalized(
                self.QUANT_OP, result=result, input=at[1], scale=scale, scale_ub=None
            )

            return at1[1], at[2], at1[2]

        def replacement(
            result: torch.Tensor,
            rms_result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            residual_out: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                x=input,
                residual=residual,
                residual_out=residual_out,
                weight=weight,
                y_scale=scale,
                epsilon=self.epsilon,
            )
            # result, residual, scale
            return at[1], at[2], at[3]

        inputs = [
            torch.empty(5, 4, device="cuda", dtype=self.quant_dtype),  # result
            empty_bf16(5, 4),  # result_rms
            empty_bf16(5, 4),  # input
            empty_bf16(5, 4),  # residual
            empty_bf16(5, 4),  # residual_out
            empty_bf16(1, 5),  # weight
            empty_fp32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class AiterRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm & group fp8 quant custom
    ops into an aiter rms_norm_group_fp8_quant op.
    """

    RMS_GROUP_QUANT_OP = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant.default

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, quant_op: OpOverload):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype
        self.quant_op = quant_op

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            at1 = self.RMS_OP(x=input, weight=weight, variance_epsilon=self.epsilon)

            at2 = self.quant_op(at1, 128)

            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
        ):
            at = self.RMS_GROUP_QUANT_OP(
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


class AiterFusedAddRMSFp8GroupQuantPattern(AiterRMSNormQuantPattern):
    """
    This pattern fuses aiter rms_norm_with_add & group fp8 quant custom ops
    into a aiter rms_norm_with_add_group_fp8_quant op.
    """

    RMS_ADD_GROUP_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default
    )

    def __init__(self, epsilon: float, quant_dtype: torch.dtype, quant_op: OpOverload):
        self.epsilon = epsilon
        self.quant_dtype = quant_dtype
        self.quant_op = quant_op

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            rms_result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            residual_out: torch.Tensor,
            weight: torch.Tensor,
        ):
            at1 = auto_functionalized(
                self.RMS_ADD_OP,
                out=rms_result,
                x=input,
                residual=residual,
                residual_out=residual_out,
                weight=weight,
                variance_epsilon=self.epsilon,
            )

            at2 = self.quant_op(at1[1], 128)

            # result, scale, residual
            return at2[0], at2[1], at1[2]

        def replacement(
            rms_result: torch.Tensor,
            input: torch.Tensor,
            residual: torch.Tensor,
            residual_out: torch.Tensor,
            weight: torch.Tensor,
        ):
            at = self.RMS_ADD_GROUP_QUANT_OP(
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


class RocmAiterRMSNormFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & vllm/aiter quant custom ops
    into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    AITER_GROUP_FP8_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default
    TRITON_GROUP_FP8_QUANT_OP = torch.ops.vllm.triton_per_token_group_quant_fp8.default

    QUANT_OPS = [AITER_GROUP_FP8_QUANT_OP, TRITON_GROUP_FP8_QUANT_OP]

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_rms_norm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            for quant_op in self.QUANT_OPS:
                #  Fuse aiter rms_norm + aiter dynamic group fp8 quant
                AiterRMSFp8GroupQuantPattern(epsilon, FP8_DTYPE, quant_op).register(
                    self.patterns
                )
                # Fuse aiter fused_add_rms_norm + aiter dynamic group fp8 quant
                AiterFusedAddRMSFp8GroupQuantPattern(
                    epsilon, FP8_DTYPE, quant_op
                ).register(self.patterns)

            # Fuse aiter rms_norm + vllm built-in dynamic per-token fp8 quant
            AiterRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + vllm built-in dynamic per-token fp8 quant
            AiterFusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

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
    """
    This pattern fuses aiter silu_and_mul & group fp8 quant custom
    ops into an aiter silu_and_mul_group_fp8_quant op.
    """

    FUSED_SILU_MUL_QUANT_OP = (
        torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant.default
    )

    def __init__(self, quant_op: OpOverload):
        self.silu_and_mul_matcher = MatcherSiluAndMul()
        self.quant_op = quant_op

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
        ):
            at1 = self.silu_and_mul_matcher(input)
            at2 = self.quant_op(at1, 128)
            return at2[0], at2[1]

        def replacement(
            input: torch.Tensor,
        ):
            at = self.FUSED_SILU_MUL_QUANT_OP(x=input, group_size=128)
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

    AITER_GROUP_FP8_QUANT_OP = torch.ops.vllm.rocm_aiter_group_fp8_quant.default
    TRITON_GROUP_FP8_QUANT_OP = torch.ops.vllm.triton_per_token_group_quant_fp8.default

    QUANT_OPS = [AITER_GROUP_FP8_QUANT_OP, TRITON_GROUP_FP8_QUANT_OP]

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rocm_aiter_silu_mul_fp8_group_quant_fusion_pass"
        )

        for quant_op in self.QUANT_OPS:
            AiterSiluMulFp8GroupQuantPattern(quant_op).register(self.patterns)

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
