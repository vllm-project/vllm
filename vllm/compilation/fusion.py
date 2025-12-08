# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Quant,
    kStaticTensorScale,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
    should_use_deepgemm_for_fp8_linear_for_nk,
)

from .inductor_pass import enable_fake_mode
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherQuantFP8, MatcherRMSNorm
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def empty_bf16(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_i32(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int32, device="cuda")


def empty_i64(*args, **kwargs):
    return torch.empty(*args, **kwargs, dtype=torch.int64, device="cuda")


RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}
if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Quant] = torch.ops._C.scaled_fp4_quant.default
if current_platform.is_cuda():
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501


class FusedRMSQuantKey(NamedTuple):
    """
    Named tuple for identifying the type of RMSNorm + quant fusion.
    quant: type of quantization
    fused_add: does the op also perform the residual add
    """

    quant: QuantKey
    fused_add: bool

    def __str__(self):
        return (
            f"FusedQuantKey({self.quant}, with"
            f"{'' if self.fused_add else 'out'} residual)"
        )


FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
    FusedRMSQuantKey(
        kFp8StaticTensorSym, False
    ): torch.ops._C.rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8StaticTensorSym, True
    ): torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, False
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, True
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic128Sym, False
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic128Sym, True
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic64Sym, False
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic64Sym, True
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
}


class RMSNormQuantPattern:
    def __init__(self, epsilon: float, key: FusedRMSQuantKey):
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None

        # groupwise FP8 linear uses col major scales if deepgemm and cutlass
        using_deepgemm = should_use_deepgemm_for_fp8_linear_for_nk(
            self.model_dtype,
            config.model_config.hf_config.intermediate_size,
            config.model_config.hf_config.hidden_size,
        )
        use_col_major_scales = using_deepgemm or cutlass_block_fp8_supported()
        use_e8m0 = is_deep_gemm_e8m0_used() if using_deepgemm else False

        assert key in FUSED_OPS, f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]

        self.rmsnorm_matcher = (
            MatcherRMSNorm(epsilon)
            if not key.fused_add
            else MatcherFusedAddRMSNorm(epsilon)
        )
        self.quant_matcher = MatcherQuantFP8(
            key.quant, use_col_major_scales=use_col_major_scales, use_e8m0=use_e8m0
        )


class RMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass):
        # Cannot use methods, as the self argument affects tracing
        def pattern(input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            result_rms = self.rmsnorm_matcher(input, weight)
            return self.quant_matcher(result_rms, scale)[0]

        def replacement(input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty(
                input.shape, device=input.device, dtype=self.quant_dtype
            )
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result
            return at[1]

        inputs = [
            # input, weight
            *self.rmsnorm_matcher.inputs(),
            self.quant_matcher.inputs()[1],  # scale
        ]
        pattern(*inputs)

        pm.register_replacement(pattern, replacement, inputs, pm.fwd_only, pm_pass)


class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(self, epsilon: float, quant_dtype: torch.dtype, symmetric=True):
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ):
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, _ = self.quant_matcher(result_rms, scale)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                residual=residual,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result, residual
            return at[1], at[2]

        inputs = [
            # input, weight, residual
            *self.rmsnorm_matcher.inputs(),
            self.quant_matcher.inputs()[1],  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
        )


class FusedAddRMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor):
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)
            return result, residual, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(
                input, transposed=self.quant_matcher.use_col_major_scales
            )
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual,
                group_size=self.group_shape[1],
                is_scale_transposed=self.quant_matcher.use_col_major_scales,
            )

            # result, residual, scale
            return at[1], at[3], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class RMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric=True,
    ):
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor, weight: torch.Tensor):
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(input: torch.Tensor, weight: torch.Tensor):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(
                input, transposed=self.quant_matcher.use_col_major_scales
            )
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None,
                group_size=self.group_shape[1],
                is_scale_transposed=self.quant_matcher.use_col_major_scales,
            )

            # result, scale
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class RMSNormDynamicQuantPattern(RMSNormQuantPattern):
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
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor, weight: torch.Tensor):
            result_rms = self.rmsnorm_matcher(input, weight)
            # result, scale
            return self.quant_matcher(result_rms)

        def replacement(input: torch.Tensor, weight: torch.Tensor):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(input)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None,
            )

            # result, scale
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):
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
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass):
        def pattern(input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor):
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ):
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(input)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual,
            )

            # result, residual, scale
            return at[1], at[3], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            self.rmsnorm_matcher.inputs(),
            pm.fwd_only,
            pm_pass,
        )


class RMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses rms_norm & quant custom ops into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig):
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            # Fuse fused_add_rms_norm + fp8 group quant
            FusedAddRMSNormGroupQuantPattern(
                epsilon, FP8_DTYPE, group_shape=GroupShape(1, 128)
            ).register(self.patterns)

            # Fuse rms_norm + fp8 group quant
            RMSNormGroupQuantPattern(
                epsilon, FP8_DTYPE, group_shape=GroupShape(1, 128)
            ).register(self.patterns)

            FusedAddRMSNormGroupQuantPattern(
                epsilon, FP8_DTYPE, group_shape=GroupShape(1, 64)
            ).register(self.patterns)

            # Fuse rms_norm + fp8 group quant
            RMSNormGroupQuantPattern(
                epsilon, FP8_DTYPE, group_shape=GroupShape(1, 64)
            ).register(self.patterns)

            # Fuse fused_add_rms_norm + static fp8 quant
            FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

            # Fuse rms_norm + static fp8 quant
            RMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

            # Fuse fused_add_rms_norm + dynamic per-token fp8 quant
            FusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

            # Fuse rms_norm + dynamic per-token fp8 quant
            RMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph):
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> Any:
        return self.hash_source(
            self,
            RMSNormGroupQuantPattern,
            RMSNormQuantPattern,
            RMSNormStaticQuantPattern,
            RMSNormDynamicQuantPattern,
            FusedAddRMSNormStaticQuantPattern,
            FusedAddRMSNormDynamicQuantPattern,
            FusedAddRMSNormGroupQuantPattern,
        )
