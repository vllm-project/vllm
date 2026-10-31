# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 linear kernel for RDNA3/RDNA3.5 (gfx11) with an int8-activation decode
path.

Routes on the number of activation rows M:
  1 <= M <= MAX_W4A8_BATCH_SIZE (decode, speculative verify):
      HIP W4A8 GEMV (``_rocm_C.mxfp4_w4a8_gemv_rdna3``). Activations are quantized
      to int8 per 32-element group and reduced with int8 dot instructions.
  otherwise (prefill, profiling):
      dequantize the weight and run a high-precision GEMM on the unquantized
      activations (weight-only).

Both paths read the checkpoint layout directly ([N, K/2] uint8 E2M1 codes,
[N, K/32] uint8 E8M0 scales), so there is no weight repack.

Selected when int8 activations are requested, e.g. with
``--quantization-config.linear.activation int8_per_group_32``. For W4A4
checkpoints this replaces the MXFP4 activation QDQ: decode uses int8
activations and prefill uses the model dtype, which kept accuracy where mixing
MXFP4-QDQ prefill with int8 decode did not.
"""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import dequant_mxfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8Dynamic32Sym,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_quark
from vllm.utils.torch_utils import direct_register_custom_op

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

MAX_W4A8_BATCH_SIZE = 8
_MXFP4_GROUP_SIZE = 32


def _rdna3_mxfp4_w4a8_apply_impl(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    import vllm._custom_ops as ops

    M = x_2d.shape[0]
    if 0 < M <= MAX_W4A8_BATCH_SIZE and x_2d.dtype in (torch.float16, torch.bfloat16):
        out = ops.mxfp4_w4a8_gemv_rdna3(x_2d, weight, weight_scale)
        if bias is not None:
            out.add_(bias)
        return out

    return F.linear(x_2d, dequant_mxfp4(weight, weight_scale, x_2d.dtype), bias)


def _rdna3_mxfp4_w4a8_apply_fake(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty(
        (x_2d.shape[0], weight.shape[0]), dtype=x_2d.dtype, device=x_2d.device
    )


direct_register_custom_op(
    op_name="rdna3_mxfp4_w4a8_apply",
    op_func=_rdna3_mxfp4_w4a8_apply_impl,
    mutates_args=[],
    fake_impl=_rdna3_mxfp4_w4a8_apply_fake,
)


class Rdna3W4A8MxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 GEMM for RDNA3/RDNA3.5: W4A8 int8-dot GEMV for M <= 8, dequant +
    high-precision GEMM above."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "only targets ROCm"
        from vllm.platforms.rocm import on_gfx11

        if not on_gfx11():
            return False, "requires RDNA3/RDNA3.5 (gfx11)"
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "mxfp4_w4a8_gemv_rdna3")
        ):
            return False, "_rocm_C.mxfp4_w4a8_gemv_rdna3 is not built"
        return True, None

    @classmethod
    def _supports_quant_scheme(
        cls,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key == kInt8Dynamic32Sym

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kInt8Dynamic32Sym:
            return False, "only supports dynamic per-32-group int8 activations"
        if not has_quark():
            return False, "amd-quark package not available (needed for M > 8)"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError(
                "Rdna3W4A8MxFp4LinearKernel expects uint8 packed weights and "
                f"uint8 E8M0 scales, got {weight.dtype} and {weight_scale.dtype}"
            )
        N, k_half = weight.shape
        K = 2 * k_half
        if K % _MXFP4_GROUP_SIZE != 0 or weight_scale.shape != (
            N,
            K // _MXFP4_GROUP_SIZE,
        ):
            raise ValueError(
                "Rdna3W4A8MxFp4LinearKernel expects weight [N, K/2] and "
                f"weight_scale [N, K/32] with K % 32 == 0, got "
                f"{tuple(weight.shape)} and {tuple(weight_scale.shape)}"
            )
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.contiguous(), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = torch.ops.vllm.rdna3_mxfp4_w4a8_apply(
            x_2d, layer.weight, layer.weight_scale, bias
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
