# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)

from .cuda import CudaInputQuantKernel
from .InputQuantKernel import InputQuantConfig, InputQuantKernel
from .pytorch import PytorchInputQuantKernel
from .triton import TritonInputQuantKernel


def per_token_group_quant_fp8_packed_for_deepgemm(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    use_ue8m0: bool | None = None,
    out_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 per-token-group quantization for DeepGEMM.

    Returns:
        (x_q, x_s_packed)
            x_q: FP8 activations, same shape as `x`.
            x_s_packed: Int32 tensor with logical shape
                        [mn, ceil(num_groups_per_row / 4)], laid out with
                        TMA-aligned stride along the packed-K dimension
    """
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    # for DeepGEMM UE8M0-packed layout we *require* UE8M0 scales.
    assert use_ue8m0, (
        "per_token_group_quant_fp8_packed_for_deepgemm requires UE8M0 scales."
    )

    dtype = current_platform.fp8_dtype()
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min, fp8_max = finfo.min, finfo.max

    # compute DeepGEMM-style packed scale tensor shape.
    hidden_dim = x.shape[-1]
    mn = x.numel() // hidden_dim
    num_groups_per_row = hidden_dim // group_size
    k_num_packed_sf_k = (num_groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4

    x_s_packed = torch.empty_strided(
        (mn, k_num_packed_sf_k),
        (1, tma_aligned_mn),
        device=x.device,
        dtype=torch.int32,
    )

    # CUDA kernel path only (DeepGEMM + E8M0 is CUDA-specific).
    assert current_platform.is_cuda(), (
        "per_token_group_quant_fp8_packed_for_deepgemm is only valid on CUDA "
        "platforms using DeepGEMM."
    )

    x_contiguous = x.contiguous()
    if out_q is not None:
        x_q_local = out_q
    else:
        x_q_local = torch.empty_like(x_contiguous, device=x.device, dtype=dtype)

    torch.ops._C.per_token_group_fp8_quant_packed(
        x_contiguous,
        x_q_local,
        x_s_packed,
        group_size,
        eps,
        fp8_min,
        fp8_max,
    )

    # return a tensor with the original logical shape.
    x_q = x_q_local.view_as(x)
    return x_q, x_s_packed


class DeepGemmInputQuantKernel(InputQuantKernel[InputQuantConfig]):
    @classmethod
    def is_supported(cls):
        if not is_deep_gemm_supported():
            return (
                False,
                "DeepGEMM currently only supported on Hopper and Blackwell GPUs.",
            )

    @classmethod
    def can_implement(cls, config: InputQuantConfig):
        # DeepGEMM kernel only supports per-group quantization with UE8M0 scale format.
        # Other configurations fall back to CudaInputQuantKernel
        # (check ordered_fallback_kernels).
        if config.group_shape.is_per_group() and (
            DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        ):
            return True, ""

        return (
            False,
            "Only DeepGemmQuant orcal is implemented for dyanmic group input quant.",
        )

    @classmethod
    def ordered_fallback_kernels(cls):
        return [CudaInputQuantKernel, TritonInputQuantKernel, PytorchInputQuantKernel]

    def apply_group_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return per_token_group_quant_fp8_packed_for_deepgemm(
            x,
            group_size=self.group_shape.col,
            use_ue8m0=True,
        )

    def apply_per_token_per_tensor_quant(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Currently there is no per_tensor, per_token deep_gemm kernel implementation.
        raise NotImplementedError
