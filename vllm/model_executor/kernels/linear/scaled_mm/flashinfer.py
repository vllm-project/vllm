# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    flashinfer_scaled_fp8_mm,
    has_flashinfer,
    has_flashinfer_cutedsl,
    is_flashinfer_fp8_blockscale_gemm_supported,
    should_use_flashinfer_for_blockscale_fp8_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op, eager_bypass_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledDynamicMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
)
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel, fp8_gemm_nt
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class FlashInferFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."

        if not has_flashinfer():
            return False, "requires FlashInfer to be installed."

        if compute_capability is not None and compute_capability < 100:
            return False, "requires compute capability 100 and above."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return False, "requires per tensor activation and weight scales."

        return True, None

    def input_quant_key(self) -> QuantKey | None:
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        return flashinfer_scaled_fp8_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )


class FlashInferFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # FlashInfer accepts BF16 input and handles FP8 conversion internally.
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                (
                    "Supports only dynamic per token group activation "
                    "quantization with group_shape=(1,128)."
                ),
            )

        if not should_use_flashinfer_for_blockscale_fp8_gemm(
            is_flashinfer_fp8_blockscale_gemm_supported(),
            config.out_dtype,
            config.input_dtype,
            config.weight_quant_key.dtype,
            config.weight_shape,
        ):
            return (
                False,
                "The provided metadata is not supported.",
            )

        return True, None

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "only cuda devices are supported."

        if not is_flashinfer_fp8_blockscale_gemm_supported():
            return False, "FlashInfer block-scale FP8 GEMM is not available."

        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # A is BF16 — FlashInfer handles FP8 conversion internally.
        # As is a placeholder (apply_input_quant=False) and is not used here.
        return _flashinfer_fp8_blockscale_gemm_op(
            A,  # BF16 input
            B,  # FP8 weight
            Bs,  # Weight scales
        )


class FlashInferFp8DeepGEMMDynamicBlockScaledKernel(
    Fp8BlockScaledDynamicMMLinearKernel
):
    """Conditional FlashInfer / DeepGEMM FP8 block-scaled GEMM.

    Dispatches between two kernels based on input batch size:
    - Small batches (M < 32): FlashInfer's swapAB trick for better utilisation.
    - Large batches (M >= 32): DeepGEMM for peak throughput.

    apply_input_quant is False because FlashInfer accepts BF16 input and
    handles FP8 conversion internally.  The DeepGEMM branch therefore
    quantises BF16→FP8 inside apply_mm via a closure before dispatching to
    the DeepGEMM kernel — keeping both branches compatible with the single
    BF16 tensor operand list passed by torch.cond.
    """

    base_type: ClassVar[type[FlashInferFp8BlockScaledMMKernel]] = (
        FlashInferFp8BlockScaledMMKernel
    )
    fallback_type: ClassVar[type[DeepGemmFp8BlockScaledMMKernel]] = (
        DeepGemmFp8BlockScaledMMKernel
    )
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.base: FlashInferFp8BlockScaledMMKernel
        self.fallback: DeepGemmFp8BlockScaledMMKernel

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # DeepGEMM need post-processing; both kernels share the same
        # parameter tensor layout so processing once is sufficient.
        self.fallback.process_weights_after_loading(layer)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        group_size = self.weight_group_shape.col
        use_deep_gemm_e8m0 = self.fallback.use_deep_gemm_e8m0

        return _dynamic_flashinfer_deepgemm_blockscale_gemm_op(
            A, B, Bs, group_size, use_deep_gemm_e8m0
        )


class FlashInferMxfp8DeepGemmBlockScaledMMKernel(DeepGemmFp8BlockScaledMMKernel):
    """Block-FP8 linear on SM107: FlashInfer's small-M MXFP8 CuTe-DSL GEMM for
    up to `MXFP8_MAX_M` rows, DeepGEMM above.

    The UE8M0 128x128 weight scales DeepGEMM uses are repeated to one scale per
    32 values; both GEMMs read the same FP8 weight. Small-M activations are
    quantized to MXFP8 (one UE8M0 scale per 32 values).
    """

    apply_input_quant: ClassVar[bool] = False

    @classmethod
    def is_supported(cls, compute_capability=None):
        supported, reason = super().is_supported(compute_capability)
        if not supported:
            return supported, reason
        if not current_platform.is_device_capability(107):
            return False, "requires sm_107 (Rubin)."
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module."
        if DeepGemmQuantScaleFMT.from_oracle() != DeepGemmQuantScaleFMT.UE8M0:
            return False, "requires DeepGEMM UE8M0 scales."
        return True, None

    @classmethod
    def can_implement(cls, config):
        can_implement, reason = super().can_implement(config)
        if not can_implement:
            return can_implement, reason
        if config.weight_quant_key.scale.group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 weight blocks."
        if config.weight_shape[0] % 128 != 0:
            return False, "mm_mxfp8 requires N divisible by 128."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_bmm", False):
            super().process_weights_after_loading(layer)
            return
        Fp8BlockScaledMMLinearKernel.process_weights_after_loading(self, layer)
        params = self._get_layer_params(layer)
        weight_scale = params.block_scale
        dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=params.weight,
            ws=weight_scale,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=self.use_deep_gemm_e8m0,
        )
        # Requantization above rewrote weight_scale in place to powers of two.
        layer.weight_scale_mxfp8 = _block_scale_to_mxfp8(weight_scale, *dg_weight.shape)
        replace_parameter(layer, params.WEIGHT, dg_weight)
        replace_parameter(layer, params.block_scale_attr, dg_weight_scale)
        layer.deep_gemm_warmup_provider = self

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        output = torch.ops.vllm.flashinfer_mxfp8_deepgemm_blockscale_gemm(
            x.view(-1, x.shape[-1]),
            params.weight,
            params.block_scale,
            layer.weight_scale_mxfp8,
        )
        if bias is not None:
            output = output + bias
        return output.view(*x.shape[:-1], params.weight.shape[0])


def _block_scale_to_mxfp8(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Power-of-two `[N/128, K/128]` block scales -> swizzled MXFP8 scales."""
    if scale.dtype == torch.float32:
        e8m0 = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    else:
        e8m0 = scale.view(torch.uint8)
    per_32 = e8m0.repeat_interleave(128, dim=0)[:n].repeat_interleave(
        128 // MXFP8_BLOCK_SIZE, dim=1
    )[:, : k // MXFP8_BLOCK_SIZE]
    return swizzle_mxfp8_scale(per_32.contiguous(), M=n, K=k)


# Measured on Vera Rubin: up to 16 rows the MXFP8 path (quant + GEMM) beats DeepGEMM
# on every Mistral Large 4 projection; from 24 rows it is slower.
MXFP8_MAX_M = 16


def _flashinfer_mxfp8_deepgemm_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_mxfp8: torch.Tensor,
) -> torch.Tensor:
    if input.shape[0] <= MXFP8_MAX_M:
        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input, is_sf_swizzled_layout=True
        )
        return vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale_mxfp8,
            out_dtype=torch.bfloat16,
            backend="cute-dsl",
        )
    q_input, input_scale = per_token_group_quant_fp8_packed_for_deepgemm(
        input, group_size=128, use_ue8m0=True
    )
    output = torch.empty(
        (input.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=input.device
    )
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
        is_deep_gemm_e8m0_used=True,
    )
    return output


def _flashinfer_mxfp8_deepgemm_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_mxfp8: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_mxfp8_deepgemm_blockscale_gemm",
    _flashinfer_mxfp8_deepgemm_blockscale_gemm_impl,
    fake_impl=_flashinfer_mxfp8_deepgemm_blockscale_gemm_fake,
)


def _flashinfer_fp8_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # The sm90 binding takes only a data pointer and assumes packed rows, so a
    # strided input (e.g. a column slice) misreads every row after the first.
    return flashinfer_fp8_blockscale_gemm(
        input=input.contiguous(),
        weight=weight,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _flashinfer_fp8_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Required fake/meta implementation for torch.compile graph tracing."""
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_fp8_blockscale_gemm",
    _flashinfer_fp8_blockscale_gemm_impl,
    fake_impl=_flashinfer_fp8_blockscale_gemm_fake,
)
_flashinfer_fp8_blockscale_gemm_op = eager_bypass_custom_op(
    "flashinfer_fp8_blockscale_gemm", _flashinfer_fp8_blockscale_gemm_impl
)


def _dynamic_flashinfer_deepgemm_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """Conditional FlashInfer FP8 blockscale GEMM with batch-size-dependent selection.

    This function switches between two optimized kernels based on the input batch size:
    - For small batches (M < 32): Uses FlashInfer's DeepGEMM swapAB optimization.
    - For larger batches (M >= 32): Uses the official DeepGEMM kernel.

    The conditional logic must use torch.cond() instead of a simple if-else statement
    to maintain compatibility with torch.compile graph compilation.

    This batch-size-dependent selection is essential for maintaining model accuracy.
    Benchmarks on GSM8K show a significant accuracy gap (88% vs 95%) for DeepSeek-V3.1
    when using FlashInfer's DeepGEMM on M>=32. The M < 32 strategy fixes the accuracy
    drop.

    Args:
        input: Input tensor of shape (batch_size, input_dim) in FP8 format
        weight: Weight tensor of shape (output_dim, input_dim) in FP8 format
        weight_scale: Scale factors for weight quantization (per-group)
        group_size: Quantization group size for the weight tensor
        use_deep_gemm_e8m0: Whether to use the E8M0 format in DeepGEMM quantization

    Returns:
        Output tensor of shape (batch_size, output_dim) in bfloat16 format

    """

    def run_flashinfer_deepgemm_swapAB(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        # Packed rows required, see _flashinfer_fp8_blockscale_gemm_impl.
        return flashinfer_fp8_blockscale_gemm(
            input=input.contiguous(),
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )

    def run_deepgemm(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        q_input, input_scale = per_token_group_quant_fp8(
            input,
            group_size=group_size,
            column_major_scales=True,
            use_ue8m0=use_deep_gemm_e8m0,
        )
        output = torch.empty(
            (q_input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=q_input.device,
        )
        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
        return output

    if envs.VLLM_BATCH_INVARIANT:
        return run_deepgemm(input, weight, weight_scale)

    condition = input.shape[0] < 32

    # PyTorch's torch.compile cannot handle input-dependent control flow in standard
    # Python conditionals. torch.cond() explicitly registers both code paths in the
    # computation graph, allowing torch.compile to capture both branches.
    # without torch.cond, the M < 32 condition won't be able to be captured by torch
    # compile
    return torch.cond(
        condition,
        run_flashinfer_deepgemm_swapAB,
        run_deepgemm,
        (input, weight, weight_scale),
    )


def _dynamic_flashinfer_deepgemm_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """Required fake/meta implementation for torch.compile graph tracing."""
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "dynamic_flashinfer_deepgemm_blockscale_gemm",
    _dynamic_flashinfer_deepgemm_blockscale_gemm_impl,
    fake_impl=_dynamic_flashinfer_deepgemm_blockscale_gemm_fake,
)
_dynamic_flashinfer_deepgemm_blockscale_gemm_op = eager_bypass_custom_op(
    "dynamic_flashinfer_deepgemm_blockscale_gemm",
    _dynamic_flashinfer_deepgemm_blockscale_gemm_impl,
)
