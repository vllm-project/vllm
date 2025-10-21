# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform


def use_swizzle_gemm(n: int, k: int, dtype: torch.dtype) -> bool:
    multiple_of: int = 64

    if dtype == current_platform.fp8_dtype():
        multiple_of = 128

    return n % multiple_of == 0 and k % multiple_of == 0


if current_platform.is_rocm():
    from aiter.tuned_gemm import tgemm as aiter_tgemm
else:
    aiter_tgemm = None


class aiter_ops:
    @staticmethod
    def rocm_aiter_tuned_gemm(
        input: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [N, K]
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        scale_a: torch.Tensor | None = None,
        scale_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return aiter_tgemm.mm(
            input, weight, otype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias
        )
