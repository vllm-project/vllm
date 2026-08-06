# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload

import vllm.ir.ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    get_fp8_min_max,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    MatcherQuantFP8,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


_RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
_FLASHINFER_ADD_RMSNORM_FP4QUANT: Any | None = None


def _flashinfer_fused_add_rms_norm_nvfp4_quant(
    result: torch.Tensor,
    result_block_scale: torch.Tensor,
    residual: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    block_scale_unswizzled: torch.Tensor,
    is_sf_swizzled_layout: bool,
    epsilon: float,
) -> None:
    """FlashInfer fused add + RMSNorm + NVFP4 quantization."""
    assert _FLASHINFER_ADD_RMSNORM_FP4QUANT is not None
    _FLASHINFER_ADD_RMSNORM_FP4QUANT(
        input,
        residual,
        weight,
        y_fp4=result.view(torch.float4_e2m1fn_x2),
        block_scale=result_block_scale.view(torch.float8_e4m3fn),
        global_scale=input_global_scale.reshape(1),
        eps=epsilon,
        block_size=16,
        scale_format="e4m3",
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        output_both_sf_layouts=False,
        block_scale_unswizzled=block_scale_unswizzled,
    )


def _flashinfer_fused_add_rms_norm_nvfp4_quant_fake(
    result: torch.Tensor,
    result_block_scale: torch.Tensor,
    residual: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    block_scale_unswizzled: torch.Tensor,
    is_sf_swizzled_layout: bool,
    epsilon: float,
) -> None:
    return None


_FLASHINFER_NVFP4_RMS_QUANT_OP: OpOverload | None = None
if (
    current_platform.is_cuda()
    and hasattr(torch, "float4_e2m1fn_x2")
    and has_flashinfer()
):
    try:
        from flashinfer import (
            add_rmsnorm_fp4quant as _FLASHINFER_ADD_RMSNORM_FP4QUANT,
        )
    except ImportError:
        pass
    else:
        # FlashInfer requires block_scale_unswizzled to have the full shape for
        # TVM-FFI validation, but does not write it when output_both_sf_layouts=False.
        direct_register_custom_op(
            op_name="flashinfer_fused_add_rms_norm_nvfp4_quant",
            op_func=_flashinfer_fused_add_rms_norm_nvfp4_quant,
            mutates_args=["result", "result_block_scale", "residual"],
            fake_impl=_flashinfer_fused_add_rms_norm_nvfp4_quant_fake,
        )
        _FLASHINFER_NVFP4_RMS_QUANT_OP = (
            torch.ops.vllm.flashinfer_fused_add_rms_norm_nvfp4_quant.default
        )


# TODO: extend rmsnorm quant kernels to support mixed input/weight dtypes,
# and remove this check.
def _rms_input_weight_dtype_match(match: pm.Match) -> bool:
    """Prevent fusion when rms_norm input and weight dtypes differ."""
    for node in match.nodes:
        if node.target == _RMS_NORM_OP:
            # rms_norm(x, weight, epsilon, variance_size)
            x, weight = node.args[0], node.args[1]
        elif node.target == _FUSED_ADD_RMS_NORM_OP:
            # fused_add_rms_norm(x, residual, weight, epsilon, variance_size)
            x, weight = node.args[0], node.args[2]
        else:
            continue
        if isinstance(x, fx.Node) and isinstance(weight, fx.Node):
            return x.meta["val"].dtype == weight.meta["val"].dtype
    return True


def empty_bf16(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.bfloat16, device=current_platform.device_type
    )


def empty_fp32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.float32, device=current_platform.device_type
    )


def empty_i32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.int32, device=current_platform.device_type
    )


def empty_i64(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.int64, device=current_platform.device_type
    )


RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}
if hasattr(torch.ops._C, "per_token_group_fp8_quant"):
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.out


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
}
# rms_norm_per_block_quant is CUDA-only; guard it like per_token_group_fp8_quant above.
if hasattr(torch.ops._C, "rms_norm_per_block_quant"):
    _rms_norm_per_block_quant = torch.ops._C.rms_norm_per_block_quant.default
    FUSED_OPS[FusedRMSQuantKey(kFp8Dynamic128Sym, False)] = _rms_norm_per_block_quant  # noqa: E501
    FUSED_OPS[FusedRMSQuantKey(kFp8Dynamic128Sym, True)] = _rms_norm_per_block_quant  # noqa: E501
    FUSED_OPS[FusedRMSQuantKey(kFp8Dynamic64Sym, False)] = _rms_norm_per_block_quant  # noqa: E501
    FUSED_OPS[FusedRMSQuantKey(kFp8Dynamic64Sym, True)] = _rms_norm_per_block_quant  # noqa: E501


class RMSNormQuantPattern:
    def __init__(
        self,
        epsilon: float,
        key: FusedRMSQuantKey,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None

        assert key in FUSED_OPS, f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]

        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
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
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            return self.quant_matcher(result_rms, scale)[0]

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
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
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            self.quant_matcher.inputs()[1],  # scale
        ]
        pattern(*inputs)

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


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
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
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
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
            self.quant_matcher.inputs()[1],  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        self.is_e8m0 = is_e8m0
        self.has_col_major_scales = has_col_major_scales
        self.is_tma_aligned = is_tma_aligned
        super().__init__(
            epsilon,
            key,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result = torch.empty(
                result_rms.shape,
                device=result_rms.device,
                dtype=self.quant_matcher.quant_key.dtype,
            )
            assert scale is not None
            fp8_min, fp8_max = get_fp8_min_max()

            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=result_rms,
                output_q=result,
                output_s=scale,
                group_size=self.quant_matcher.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.quant_matcher.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )

            return result, residual, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)

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

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
            self.quant_matcher.empty_f32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class RMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        self.has_col_major_scales = has_col_major_scales
        self.is_tma_aligned = is_tma_aligned
        super().__init__(
            epsilon,
            key,
            has_col_major_scales=self.has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            result = torch.empty(
                result_rms.shape,
                device=result_rms.device,
                dtype=self.quant_matcher.quant_key.dtype,
            )
            assert scale is not None
            fp8_min, fp8_max = get_fp8_min_max()

            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=result_rms,
                output_q=result,
                output_s=scale,
                group_size=self.quant_matcher.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.quant_matcher.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )

            return result, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
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
                is_scale_transposed=self.has_col_major_scales,
            )

            # result, scale
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            [
                empty_bf16(5, 16),  # input
                empty_bf16(16),  # weight
                self.quant_matcher.empty_f32(1, 1),  # scale
            ],
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
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
            [
                empty_bf16(5, 16),  # input
                empty_bf16(16),  # weight
            ],
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
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

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormNvfp4QuantPattern:
    """Fuse add-RMSNorm with NVFP4 quantization for either scale layout."""

    def __init__(self, epsilon: float, is_sf_swizzled_layout: bool) -> None:
        assert _FLASHINFER_NVFP4_RMS_QUANT_OP is not None
        self.epsilon = epsilon
        self.is_sf_swizzled_layout = is_sf_swizzled_layout
        self.FUSED_OP = _FLASHINFER_NVFP4_RMS_QUANT_OP

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            result: torch.Tensor,
            result_block_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, updated_residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            at = auto_functionalized(
                torch.ops._C.scaled_fp4_quant.out,
                input=result_rms,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=self.is_sf_swizzled_layout,
                output=result,
                output_scale=result_block_scale,
            )
            return at[1], updated_residual, at[2]

        def replacement(
            result: torch.Tensor,
            result_block_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden_size = input.shape[-1]
            num_tokens = input.numel() // hidden_size
            # This full-size dummy is required by FlashInfer's TVM-FFI tensor
            # validation even though output_both_sf_layouts=False leaves it untouched.
            block_scale_unswizzled = torch.empty(
                (num_tokens, hidden_size // 16),
                dtype=torch.float8_e4m3fn,
                device=input.device,
            )
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                result_block_scale=result_block_scale,
                residual=residual,
                input=input,
                weight=weight,
                input_global_scale=input_global_scale,
                block_scale_unswizzled=block_scale_unswizzled,
                is_sf_swizzled_layout=self.is_sf_swizzled_layout,
                epsilon=self.epsilon,
            )
            # result, updated residual, block scale in the requested layout
            return at[1], at[3], at[2]

        inputs = [
            torch.empty(
                (5, 32), dtype=torch.uint8, device=current_platform.device_type
            ),
            (
                empty_i32(128, 4)
                if self.is_sf_swizzled_layout
                else torch.empty(
                    (5, 4),
                    dtype=torch.uint8,
                    device=current_platform.device_type,
                )
            ),
            empty_bf16(5, 64),
            empty_bf16(64),
            empty_bf16(5, 64),
            empty_fp32(1),
        ]
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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
            if (
                self.pass_config.fuse_add_rms_norm_nvfp4
                and _FLASHINFER_NVFP4_RMS_QUANT_OP is not None
                and current_platform.has_device_capability(100)
            ):
                for is_sf_swizzled_layout in (True, False):
                    FusedAddRMSNormNvfp4QuantPattern(
                        epsilon, is_sf_swizzled_layout
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

            # Only register group quant patterns on CUDA/ROCm where the C++ op exists
            for group_shape in [GroupShape(1, 128), GroupShape(1, 64)]:
                for has_col_major_scales in [True, False]:
                    for is_e8m0 in [True, False]:
                        for is_tma_aligned in [False, True]:
                            # Fuse fused_add_rms_norm + fp8 group quant
                            FusedAddRMSNormGroupQuantPattern(
                                epsilon,
                                FP8_DTYPE,
                                group_shape=group_shape,
                                is_e8m0=is_e8m0,
                                has_col_major_scales=has_col_major_scales,
                                is_tma_aligned=is_tma_aligned,
                            ).register(self.patterns)

                            # Fuse rms_norm + fp8 group quant
                            RMSNormGroupQuantPattern(
                                epsilon,
                                FP8_DTYPE,
                                group_shape=group_shape,
                                is_e8m0=is_e8m0,
                                has_col_major_scales=has_col_major_scales,
                                is_tma_aligned=is_tma_aligned,
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
            FusedAddRMSNormNvfp4QuantPattern,
            _flashinfer_fused_add_rms_norm_nvfp4_quant,
            _flashinfer_fused_add_rms_norm_nvfp4_quant_fake,
        )
