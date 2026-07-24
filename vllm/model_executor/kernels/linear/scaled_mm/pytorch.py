# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.config import CompilationMode, get_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


def _get_num_tokens(output_shape: list) -> int:
    # torch._scaled_mm works with 2D tensors, so input tensors are
    # flattened if they are 3D. If output_shape is 3D, num_tokens is
    # the product of all dims except the last (hidden dim).
    return math.prod(output_shape[:-1])


class TorchFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    Base class for FP8 linear kernels using Torch.
    Each subclass represents a kernel variant for
    specific device capabilities and torch versions.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            return False, "requires ROCm, CUDA, CPU or XPU."

        if compute_capability is not None and compute_capability < 89:
            return False, "requires compute capability 89 and above."

        return True, None

    def get_output_padding(self) -> int | None:
        # Note: we pad the input because torch._scaled_mm is more performant
        # for matrices with batch dimension > 16.
        # This could change in the future.
        # We also don't pad when using torch.compile,
        # as it breaks with dynamic shapes.
        #
        # The perf gain is still relevant as of 16/1/2026
        # torch version == 2.9.0. More details in the link below:
        # https://github.com/vllm-project/vllm/issues/32269
        vllm_config = get_current_vllm_config().compilation_config
        pad_output = vllm_config.mode < CompilationMode.VLLM_COMPILE
        return 17 if pad_output else None


class PerTensorTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
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
        # torch._scaled_mm under torch.compile does not support 0-D scales
        if As.dim() == 0:
            As = As.view(1)
        if Bs.dim() == 0:
            Bs = Bs.view(1)

        output = torch._scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
        # A fix for discrepancy in scaled_mm which returns tuple
        # for torch < 2.5 and a single value in torch >= 2.5
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        num_tokens = _get_num_tokens(output_shape)
        return torch.narrow(output, 0, 0, num_tokens).view(*output_shape)


class RowWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "requires ROCm."

        from vllm.platforms.rocm import on_mi3xx

        if not on_mi3xx():
            return False, "requires MI3xx."

        if compute_capability is not None and compute_capability < 94:
            return False, "requires compute capability 94 and above."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if c.out_dtype == torch.float16:
            # hipblaslt rowwise _scaled_mm only supports BFloat16
            return False, "supports BFloat16 output data type only."

        if per_tensor_activation_scales or per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

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
        #  Note:
        #  For now it has only been validated on ROCm platform.
        #  fp8 rowwise scaling in torch._scaled_mm is introduced in
        #  https://github.com/pytorch/pytorch/pull/144432 using
        #  hipBLASLt and ROCm 6.3, which only exists in torch 2.7 and above.
        #
        #  For CUDA platform please validate if the torch._scaled_mm supports
        #  rowwise scaled GEMM before using it

        # torch._scaled_mm rowwise requires scale_a = (m, 1), scale_b = (1, n).
        # CompressedTensors stores weight_scale as (n, 1), so `.t()` yields (1, n).
        # ModelOpt FP8_PER_CHANNEL_PER_TOKEN stores it as 1-D (n,); reshape to
        # (1, n) so both paths satisfy the rowwise contract.
        scale_b = Bs.view(1, -1) if Bs.dim() == 1 else Bs.t()
        if As.dim() == 1:
            As = As.view(-1, 1)

        # Fused GEMM_DQ Rowwise GEMM
        output = torch._scaled_mm(
            A,
            B,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=scale_b,
            bias=bias,
        )

        num_tokens = _get_num_tokens(output_shape)
        return torch.narrow(output, 0, 0, num_tokens).view(*output_shape)


class ChannelWiseTorchFP8ScaledMMLinearKernel(TorchFP8ScaledMMLinearKernel):
    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if per_tensor_activation_scales and per_tensor_weight_scales:
            return False, "cannot be used with per tensor activation and weight scales."

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
        # Use unfused DQ due to limitations with scaled_mm

        # Symmetric quantized GEMM by definition computes the following:
        #   C = (s_x * X) (s_w * W) + bias
        # This is equivalent to dequantizing the weights and activations
        # before applying a GEMM.
        #
        # In order to compute quantized operands, a quantized kernel
        # will rewrite the above like so:
        #   C = s_w * s_x * (X * W) + bias
        #
        # For the scaled_mm fallback case, we break this down, since it
        # does not support s_w being a vector.

        # Input scaling factors are no longer optional in _scaled_mm starting
        # from pytorch 2.5. Allocating a dummy tensor to pass as scales
        dummy_tensor = torch.ones(1, dtype=torch.float32, device=A.device)

        # GEMM
        # This computes C = (X * W).
        # Output in fp32 to allow subsequent ops to happen in-place
        output = torch._scaled_mm(
            A,
            B,
            scale_a=dummy_tensor,
            scale_b=dummy_tensor,
            out_dtype=torch.float32,
        )
        # A fix for discrepancy in scaled_mm which returns tuple
        # for torch < 2.5 and a single value in torch >= 2.5
        if type(output) is tuple and len(output) == 2:
            output = output[0]

        # Unpad (undo num_token_padding)
        num_tokens = _get_num_tokens(output_shape)
        output = torch.narrow(output, 0, 0, num_tokens)
        x_scale = torch.narrow(As, 0, 0, num_tokens)

        # DQ
        # C = sw * sx * (X * W) + bias
        output = output * x_scale * Bs.t()
        if bias is not None:
            output = output + bias
        return output.to(out_dtype).view(*output_shape)


class BlockWiseTorchFP8ScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """FP8 block-scaled linear kernel using ``torch._scaled_mm``.

    Implements the DeepSeek-style block-scaled path of ``torch._scaled_mm``
    (v1), which dispatches on the shapes of the scale tensors. For
    ``A = [M, K]`` and ``B = [K, N]`` (both fp8) the op's block path
    requires, with float32 scales:
      * 1x128 activation: ``scale_a = [M, ceil(K / 128)]``
      * 128x128 weight:   ``scale_b = [ceil(K / 128), ceil(N / 128)]``

    The op supports this path on CUDA (cuBLASLt) and XPU (oneDNN). The
    logical scale shapes are identical across both; only the physical scale
    layout differs: CUDA requires cuBLAS-specific strides (outer-dim-major
    for 1x128), while XPU accepts either row- or column-major and normalizes
    internally. We therefore request column-major activation scales on CUDA
    (matching the op's requirement) and default row-major on XPU.
    """

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        # CUDA's block path requires outer-dim-major (column-major)
        # activation scales; XPU accepts either. Weight scales are handled
        # via a transpose in apply_block_scaled_mm.
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
            column_major_scales=current_platform.is_cuda_alike(),
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # torch._scaled_mm implements the fp8 128-block path on CUDA
        # (cuBLASLt) and XPU (oneDNN).
        if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
            return False, "requires CUDA, ROCm or XPU."
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        # torch._scaled_mm's DeepSeek block path is the 1x128 activation /
        # 128x128 weight pair with float32 scales.
        act_group_shape = config.activation_quant_key.scale.group_shape
        if act_group_shape != GroupShape(1, 128):
            return (
                False,
                "requires 1x128 (per-token-group) activation quantization.",
            )
        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return (
                False,
                "requires 128x128 block weight quantization.",
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # B is [N, K] from the checkpoint; B.t() is the [K, N] operand.
        # Bs is [ceil(N/128), ceil(K/128)] from the checkpoint; Bs.t() gives
        # the [ceil(K/128), ceil(N/128)] scale the op expects for B.t().
        output = torch._scaled_mm(
            A,
            B.t(),
            scale_a=As,
            scale_b=Bs.t(),
            out_dtype=self.config.out_dtype,
        )
        if type(output) is tuple and len(output) == 2:
            output = output[0]
        return output

