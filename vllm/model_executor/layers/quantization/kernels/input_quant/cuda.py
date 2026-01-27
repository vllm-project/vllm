# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    get_tma_aligned_size,
)

from .InputQuantKernel import (
    _FP8_DTYPE,
    _FP8_MAX,
    _FP8_MIN,
    InputQuantConfig,
    InputQuantKernel,
)
from .pytorch import PytorchInputQuantKernel
from .triton import TritonInputQuantKernel


def per_token_group_quant_fp8_cuda(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool = False,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA implementation of per-token-group FP8 quantization.

    This function uses the CUDA kernel for per-token-group quantization.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor. Note that only `torch.float8_e4m3fn`
            is supported for now.
        column_major_scales: Outputs scales in column major.
        out_q: Optional output tensor. If not provided, function will create.
        use_ue8m0: Whether to use UE8M0 format for scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor.
    """
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"
    assert x.is_contiguous(), "CUDA kernel requires contiguous input tensor"

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor in either row- or column-major format.
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    torch.ops._C.per_token_group_fp8_quant(
        x, x_q, x_s, group_size, eps, _FP8_MIN, _FP8_MAX, use_ue8m0
    )
    return x_q, x_s


class CudaInputQuantKernel(InputQuantKernel[InputQuantConfig]):
    @classmethod
    def is_supported(cls):
        return current_platform.is_cuda_alike(), "Not supported on Non cuda platfrom"

    @classmethod
    def can_implement(cls, config: InputQuantConfig):
        if config.group_shape.is_per_group() and config.static:
            return (
                False,
                "Cuda group quantization does not support static quantization.",
            )

        if config.group_shape.is_per_group() and current_platform.is_rocm():
            return (
                False,
                ("Cuda group quantization only supported on Cuda platform."),
            )

        return True, ""

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type[InputQuantKernel[InputQuantConfig]]]:
        return [TritonInputQuantKernel, PytorchInputQuantKernel]

    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            fall_backs = self.ordered_fallback_kernels()
            for kernel in fall_backs:
                if kernel.is_supported()[0] and kernel.can_implement(self.config)[0]:
                    return kernel(self.config).apply(x, scale, scale_ub)

            raise ValueError(
                f"No suitable fallback kernel found for quantization. "
                f"Input contiguous: {x.is_contiguous()},"
                f"config: {self.config}"
            )

        assert not self.is_static_quant, (
            "Cuda group quantization does not support static quantization."
        )
        assert scale is None, "Dynamic group quantization does not use scale"

        return per_token_group_quant_fp8_cuda(
            x,
            group_size=self.group_size,
            column_major_scales=self.is_column_major_scales,
            tma_aligned_scales=self.config.tma_aligned_scales,
            dtype=_FP8_DTYPE,
            use_ue8m0=self.use_ue8m0,
        )

    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (scale is not None) == self.is_static_quant
        assert scale_ub is None or (
            not self.is_static_quant
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.group_shape.is_per_token(),
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.is_static_quant
            else None,
        )
