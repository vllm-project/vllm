# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product
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
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform

from .inductor_pass import enable_fake_mode
from .matcher_utils import (
    MatcherFusedAddRMSNorm,
    MatcherQuantFP8,
    MatcherRMSNorm,
)
from .vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def empty_bf16(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(*args, **kwargs, dtype=torch.bfloat16, device="cuda")


def empty_fp32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(*args, **kwargs, dtype=torch.float32, device="cuda")


def empty_i32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(*args, **kwargs, dtype=torch.int32, device="cuda")


def empty_i64(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(*args, **kwargs, dtype=torch.int64, device="cuda")


RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}

if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.default
if current_platform.is_cuda():
    QUANT_OPS.update(
        dict.fromkeys(
            [kFp8Dynamic128Sym, kFp8Dynamic64Sym],
            torch.ops._C.per_token_group_fp8_quant.default,
        )
    )

if current_platform.is_rocm():
    # Import to trigger custom op registrations via direct_register_custom_op
    from vllm.model_executor.layers.quantization.utils import fp8_utils  # noqa: F401

    QUANT_OPS.update(
        dict.fromkeys(
            [kFp8Dynamic128Sym, kFp8Dynamic64Sym],
            torch.ops.vllm.per_token_group_quant_fp8.default,
        )
    )


class FusedRMSQuantKey(NamedTuple):
    """
    Named tuple for identifying the type of RMSNorm + quant fusion.
    quant: type of quantization
    fused_add: does the op also perform the residual add
    """

    quant: QuantKey
    fused_add: bool

    def __str__(self) -> str:
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
    def __init__(
        self,
        epsilon: float,
        key: FusedRMSQuantKey,
        is_e8m0: bool = False,
        has_col_major_scales: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None

        assert key in FUSED_OPS, f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]

        self.rmsnorm_matcher = (
            MatcherRMSNorm(epsilon)
            if not key.fused_add
            else MatcherFusedAddRMSNorm(epsilon)
        )
        self.quant_matcher = MatcherQuantFP8(
            key.quant, is_e8m0=is_e8m0, has_col_major_scales=has_col_major_scales
        )


class RMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None:
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Cannot use methods, as the self argument affects tracing
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            result_rms = self.rmsnorm_matcher(input, weight)
            return self.quant_matcher(result_rms, scale)[0]

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
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
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None:
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, _ = self.quant_matcher(result_rms, scale)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = False,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        super().__init__(
            epsilon, key, is_e8m0=is_e8m0, has_col_major_scales=has_col_major_scales
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)
            return result, residual, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(input, self.has_col_major_scales)
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
                is_scale_transposed=self.has_col_major_scales,
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
        symmetric: bool = True,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.has_col_major_scales = has_col_major_scales
        self.group_shape = group_shape
        super().__init__(
            epsilon, key, is_e8m0=is_e8m0, has_col_major_scales=has_col_major_scales
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            result, scale = self.quant_matcher(result_rms)
            return result, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Return the same pattern if input is not contiguous.
            # This has the same effect as skipping replacement.
            if not input.is_contiguous():
                return pattern(input, weight)

            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(
                input, transposed=self.has_col_major_scales
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
                is_scale_transposed=self.quant_matcher.has_col_major_scales,
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
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = self.rmsnorm_matcher(input, weight)
            # result, scale
            return self.quant_matcher(result_rms)  # type: ignore[no-any-return]

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual = self.rmsnorm_matcher(input, weight, residual)
            result, scale = self.quant_matcher(result_rms)

            return result, residual, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
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

            for group_shape, has_col_major_scales, is_e8m0 in product(
                [GroupShape(1, 128), GroupShape(1, 64)], [True, False], [True, False]
            ):
                # Fuse fused_add_rms_norm + fp8 group quant
                FusedAddRMSNormGroupQuantPattern(
                    epsilon,
                    FP8_DTYPE,
                    group_shape=group_shape,
                    has_col_major_scales=has_col_major_scales,
                    is_e8m0=is_e8m0,
                ).register(self.patterns)

                # Fuse rms_norm + fp8 group quant
                RMSNormGroupQuantPattern(
                    epsilon,
                    FP8_DTYPE,
                    group_shape=group_shape,
                    has_col_major_scales=has_col_major_scales,
                    is_e8m0=is_e8m0,
                ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
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
