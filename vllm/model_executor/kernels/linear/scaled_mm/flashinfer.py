# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    flashinfer_scaled_fp8_mm,
    has_flashinfer,
    is_flashinfer_fp8_blockscale_gemm_supported,
    should_use_flashinfer_for_blockscale_fp8_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledDynamicMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
)
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel
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
        act_scale_descriptor = config.activation_quant_key.scale
        # flashinfer does not require input fp8 op.
        # since flashinfer for block_scaled_mm
        # is used dynamically with deepgemm.
        # the quant_fp8 is instantiated identical to
        # deepgemm input quant.
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=is_deep_gemm_e8m0_used(),
            tma_aligned_scales=envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES,
            column_major_scales=True,
        )

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
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
        return torch.ops.vllm.flashinfer_fp8_blockscale_gemm(
            A,  # BF16 input
            B,  # FP8 weight
            Bs,  # Weight scales
        )


class FlashInferFp8DeepGEMMDynamicBlockScaledKernel(
    Fp8BlockScaledDynamicMMLinearKernel
):
    """
    Conditional FlashInfer / DeepGEMM FP8 block-scaled GEMM.

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

    @classmethod
    def is_supported(cls, compute_capability=None):
        if envs.VLLM_BATCH_INVARIANT:
            return False, "Always use deepgemm for batch invariant"
        return super().is_supported(compute_capability)

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        # Use DeepGEMM's quant_fp8 (with TMA-aligned, column-major scales)
        # for the fallback branch quantisation step inside apply_mm.
        self.quant_fp8 = self.fallback.quant_fp8

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # DeepGEMM may need post-processing; both kernels share the same
        # parameter tensor layout so processing once is sufficient.
        self.fallback.process_weights_after_loading(layer)

    def predicate(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        return A.shape[0] < 32

    def _deepgemm_apply_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # DeepGEMM, which requires FP8 input.
        # Both input_scale and input_scale_ub are None for dynamic block-scaled
        # quantisation (static quantisation is rejected by can_implement).
        A_fp8, As_fp8 = self.quant_fp8(A, None, None, use_triton=False)
        return self.fallback.apply_block_scaled_mm(A_fp8, B, As_fp8, Bs)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # FlashInfer branch receives BF16 A directly.
        # DeepGEMM branch quantises BF16→FP8 inside _deepgemm_apply_mm.
        return torch.cond(
            self.predicate(A, B, As, Bs),
            self.base.apply_block_scaled_mm,
            self._deepgemm_apply_mm,
            [A, B, As, Bs],
        )


def _flashinfer_fp8_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return flashinfer_fp8_blockscale_gemm(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _flashinfer_fp8_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_fp8_blockscale_gemm",
    _flashinfer_fp8_blockscale_gemm_impl,
    fake_impl=_flashinfer_fp8_blockscale_gemm_fake,
)
