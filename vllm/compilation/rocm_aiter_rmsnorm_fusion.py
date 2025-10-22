# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from itertools import product
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
from vllm.utils.torch_utils import direct_register_custom_op

from .fusion import (
    FP8_DTYPE,
    FusedRMSQuantKey,
    RMSNormQuantPattern,
    empty_bf16,
    empty_fp32,
)
from .inductor_pass import enable_fake_mode
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


def is_rocm_aiter_enabled() -> bool:
    return current_platform.is_rocm() and envs.VLLM_ROCM_USE_AITER


def rocm_aiter_rmsnorm_fused_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
        out, input, y_scale, weight, epsilon, use_model_sensitive_rmsnorm=0
    )

    return out, y_scale


def rocm_aiter_rmsnorm_fused_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return out, y_scale


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    residual_out = torch.empty_like(residual)

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

    return out, residual_out, y_scale


def rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_scale: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return out, torch.empty_like(residual), y_scale


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
        mutates_args=["out", "y_scale"],
        fake_impl=rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def aiter_rms_pattern(epsilon: float):
    return lambda input, weight: torch.ops.vllm.rocm_aiter_rms_norm.default(
        x=input,
        weight=weight,
        variance_epsilon=epsilon,
    )


def vllm_rms_pattern(epsilon: float):
    return lambda result, input, weight: auto_functionalized(
        torch.ops._C.rms_norm.default,
        result=result,
        input=input,
        weight=weight,
        epsilon=epsilon,
    )[1]


def aiter_rms_add_pattern(epsilon: float):
    return (
        lambda input,
        residual,
        weight: torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default(
            x=input,
            residual=residual,
            weight=weight,
            variance_epsilon=epsilon,
        )
    )


def vllm_rms_add_pattern(epsilon: float):
    return lambda input, residual, weight: auto_functionalized(
        torch.ops._C.fused_add_rms_norm.default,
        input=input,
        residual=residual,
        weight=weight,
        epsilon=epsilon,
    )[1:3]


def aiter_per_token_quant_pattern():
    return lambda out, input, scale: auto_functionalized(
        torch.ops.vllm.rocm_aiter_per_token_quant.default,
        out=out,
        x=input,
        scale=scale,
    )[1:3]


def vllm_per_token_quant_pattern():
    return lambda out, input, scale: auto_functionalized(
        torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,
        result=out,
        input=input,
        scale=scale,
        scale_ub=None,
    )[1:3]


def create_inplace_rms_norm_and_quant_pattern_and_replacement(
    rms_norm_op: OpOverload,
    quant_op: OpOverload,
    fused_op: OpOverload,
    epsilon: float,
    quant_dtype: torch.dtype,
):
    inputs = [
        torch.empty(5, 4, device="cuda", dtype=quant_dtype),
        empty_bf16(5, 4),  # input
        empty_bf16(1, 5),  # weight
        empty_fp32(5, 1),  # scale
    ]

    def replacement(result, input, weight, scale):
        return fused_op(
            out=result,
            input=input,
            weight=weight,
            y_scale=scale,
            epsilon=epsilon,
        )

    def pattern(
        result: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ):
        rms_out = rms_norm_op(
            input=input,
            weight=weight,
        )
        out, scales_out = quant_op(
            out=result,
            input=rms_out,
            scale=scale,
        )

        return out, scales_out

    return pattern, replacement, inputs


def create_non_inplace_rms_norm_and_quant_pattern_and_replacement(
    rms_norm_op: OpOverload,
    quant_op: OpOverload,
    fused_op: OpOverload,
    epsilon: float,
    quant_dtype: torch.dtype,
):
    inputs = [
        torch.empty(5, 4, device="cuda", dtype=quant_dtype),
        empty_bf16(5, 4),  # result_rms
        empty_bf16(5, 4),  # input
        empty_bf16(1, 5),  # weight
        empty_fp32(5, 1),  # scale
    ]

    def replacement(rms_result, result, input, weight, scale):
        return fused_op(
            out=result,
            input=input,
            weight=weight,
            y_scale=scale,
            epsilon=epsilon,
        )

    def pattern(
        rms_result: torch.Tensor,
        result: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ):
        rms_out = rms_norm_op(
            result=rms_result,
            input=input,
            weight=weight,
        )
        out, scales_out = quant_op(
            out=result,
            input=rms_out,
            scale=scale,
        )

        return out, scales_out

    return pattern, replacement, inputs


def create_rms_norm_and_quant_pattern_and_replacement(
    rms_norm_pattern_generator: Callable,
    quant_pattern_generator: Callable,
    fused_op: OpOverload,
    epsilon: float,
    quant_dtype: torch.dtype,
):
    rms_norm_op = rms_norm_pattern_generator(epsilon)
    quant_op = quant_pattern_generator()
    # aiter's rms op is not inplace and doesn't
    # require a result buffer. Therefore, we need
    # to handle that case by returning pattern
    # without a result buffer.

    if rms_norm_pattern_generator == aiter_rms_pattern:
        return create_inplace_rms_norm_and_quant_pattern_and_replacement(
            rms_norm_op, quant_op, fused_op, epsilon, quant_dtype
        )
    return create_non_inplace_rms_norm_and_quant_pattern_and_replacement(
        rms_norm_op, quant_op, fused_op, epsilon, quant_dtype
    )


def create_rms_norm_fadd_and_quant_pattern_and_replacement(
    rms_norm_fadd_pattern_generator: Callable,
    quant_pattern_generator: Callable,
    fused_op: OpOverload,
    epsilon: float,
    quant_dtype: torch.dtype,
):
    rms_norm_fadd_op = rms_norm_fadd_pattern_generator(epsilon)
    quant_op = quant_pattern_generator()

    inputs = [
        torch.empty(5, 4, device="cuda", dtype=quant_dtype),  # result
        empty_bf16(5, 4),  # input
        empty_bf16(5, 4),  # residual
        empty_bf16(1, 5),  # weight
        empty_fp32(5, 1),  # scale
    ]

    def replacement(result, input, residual, weight, scale):
        return fused_op(
            out=result,
            input=input,
            residual=residual,
            weight=weight,
            y_scale=scale,
            epsilon=epsilon,
        )

    def pattern(
        result: torch.Tensor,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
    ):
        rms_norm_fadd_out, residual_out = rms_norm_fadd_op(
            input=input,
            residual=residual,
            weight=weight,
        )
        out, scales_out = quant_op(
            out=result,
            input=rms_norm_fadd_out,
            scale=scale,
        )

        return out, residual_out, scales_out

    return pattern, replacement, inputs


QUANT_OPS: dict[QuantKey, list[OpOverload]] = {
    kFp8DynamicTokenSym: [aiter_per_token_quant_pattern, vllm_per_token_quant_pattern]
}
RMS_PATTERNS = [aiter_rms_pattern, vllm_rms_pattern]
RMS_ADD_PATTERNS = [aiter_rms_add_pattern, vllm_rms_add_pattern]
ROCM_AITER_FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
    FusedRMSQuantKey(
        kFp8DynamicTokenSym,
        False,
    ): torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8DynamicTokenSym,
        True,
    ): torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant.default,  # noqa: E501
}


class RMSNormAiterQuantPattern(RMSNormQuantPattern):
    def __init__(self, epsilon, key):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype

        assert key.quant in QUANT_OPS, f"unsupported quantization scheme {key.quant}"
        self.QUANT_OPS = QUANT_OPS[key.quant]
        assert key in ROCM_AITER_FUSED_OPS, (
            f"unsupported fused aiter rmsnorm+quant op for {key}"
        )
        self.FUSED_OP = ROCM_AITER_FUSED_OPS[key]


class RMSNormAiterDynamicQuantPattern(RMSNormAiterQuantPattern):
    """AITER RMSNorm + Dynamic Quantization pattern."""

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
        self.RMS_PATTERNS = RMS_PATTERNS
        super().__init__(epsilon, key)

    def register(self, pm_pass):
        for rms_pattern, quant_pattern in product(self.RMS_PATTERNS, self.QUANT_OPS):
            pattern, replacement, inputs = (
                create_rms_norm_and_quant_pattern_and_replacement(
                    rms_pattern,
                    quant_pattern,
                    self.FUSED_OP,
                    self.epsilon,
                    self.quant_dtype,
                )
            )

            pm.register_replacement(
                pattern,
                replacement,
                inputs,
                pm.fwd_only,
                pm_pass,
            )


class FusedAddRMSNormAiterDynamicQuantPattern(RMSNormAiterQuantPattern):
    """AITER RMSNorm Fused Add + Dynamic Quantization pattern."""

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
        self.RMS_ADD_PATTERNS = RMS_ADD_PATTERNS

        super().__init__(epsilon, key)

    def register(self, pm_pass):
        for rms_fadd_pattern, quant_pattern in product(
            self.RMS_ADD_PATTERNS, self.QUANT_OPS
        ):
            pattern, replacement, inputs = (
                create_rms_norm_fadd_and_quant_pattern_and_replacement(
                    rms_fadd_pattern,
                    quant_pattern,
                    self.FUSED_OP,
                    self.epsilon,
                    self.quant_dtype,
                )
            )
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
