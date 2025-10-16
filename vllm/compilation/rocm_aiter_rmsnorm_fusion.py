# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.envs as envs

# add this import to make sure the custom ops are registered
import vllm.model_executor.layers.layernorm  # noqa: F401
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8DynamicTokenSym,
)
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from .fusion import (
    FP8_DTYPE,
    QUANT_OPS,
    FusedRMSQuantKey,
    RMSNormQuantPattern,
    empty_bf16,
    empty_fp32,
)
from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def is_rocm_aiter_rmsnorm_enabled() -> bool:
    return (
        current_platform.is_rocm()
        and envs.VLLM_ROCM_USE_AITER_RMSNORM
        and envs.VLLM_ROCM_USE_AITER
    )


def rocm_aiter_rmsnorm_fused_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> None:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
        out, input, y_scale, weight, epsilon, use_model_sensitive_rmsnorm=0
    )


def rocm_aiter_rmsnorm_fused_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> None:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_add_dynamicquant(
        out,
        input,
        residual,
        residual_out,
        y_scale,
        weight,
        epsilon,
        use_model_sensitive_rmsnorm=0,
    )


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> None:
    pass


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_dynamic_quant_impl,
        mutates_args=["out", "y_scale"],
        fake_impl=rocm_aiter_rmsnorm_fused_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm_fused_add_dynamic_quant",
        op_func=rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl,
        mutates_args=["out", "residual_out", "y_scale"],
        fake_impl=rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )


class RMSNormAiterQuantPattern(RMSNormQuantPattern):
    def __init__(self, epsilon, key):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        assert key.quant in QUANT_OPS, f"unsupported quantization scheme {key.quant}"
        self.QUANT_OP = QUANT_OPS[key.quant]


class RMSNormAiterDynamicQuantPattern(RMSNormAiterQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

    ROCM_AITER_RMS_OP = torch.ops.vllm.rocm_aiter_rms_norm.default

    ROCM_AITER_FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
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

        assert key in self.ROCM_AITER_FUSED_OPS, (
            f"unsupported fused aiter rmsnorm+quant op for {key}"
        )
        self.FUSED_OP = self.ROCM_AITER_FUSED_OPS[key]

        super().__init__(epsilon, key)

    def register(self, pm_pass):
        def pattern(
            result: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            rms_out = self.ROCM_AITER_RMS_OP(
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
                input=input,
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


class FusedAddRMSNormAiterDynamicQuantPattern(RMSNormAiterQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

    ROCM_AITER_RMS_ADD_OP = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default

    ROCM_AITER_FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
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

        assert key in self.ROCM_AITER_FUSED_OPS, (
            f"unsupported fused aiter rmsnorm+quant op for {key}"
        )
        self.FUSED_OP = self.ROCM_AITER_FUSED_OPS[key]

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
                self.ROCM_AITER_RMS_ADD_OP,
                output=rms_result,
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
                input=input,
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


class RMSNormAiterQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses aiter rms_norm & quant custom ops into a fused rms_norm_quant op.
    It also supports aiter fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="aiter_rmsnorm_quant_fusion_pass"
        )

        for epsilon in [1e-5, 1e-6]:
            # Fuse aiter rms_norm + dynamic per-token fp8 quant
            RMSNormAiterDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

            # Fuse aiter fused_add_rms_norm + dynamic per-token fp8 quant
            FusedAddRMSNormAiterDynamicQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(
            self,
            RMSNormQuantPattern,
            RMSNormAiterDynamicQuantPattern,
            FusedAddRMSNormAiterDynamicQuantPattern,
        )
