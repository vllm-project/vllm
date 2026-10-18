# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A shared BF16-to-FP32 projection with one fixed reduction per weight: the
CuTe DSL K256 kernel, or (VLLM_BI_BF16_GEMM=tensor_core) DeepGEMM without
split-K."""

import functools

import torch

from vllm import envs
from vllm.utils.torch_utils import direct_register_custom_op

from .ll_bf16 import LLBf16Gemm


class FixedBf16Gemm(LLBf16Gemm):
    def dispatch(
        self, *, M: int, K: int, N: int, output_dtype: torch.dtype = torch.float32
    ) -> LLBf16Gemm.CompileKey:
        return self.CompileKey(
            backend="splitk", split_k=1, num_stages=4, output_dtype=output_dtype
        )


_KERNEL = FixedBf16Gemm()


@functools.cache
def _deterministic_deep_gemm():
    """DeepGEMM with split-K pinned off, so a row's bits ignore M."""
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    deep_gemm.use_deterministic_algorithms(True)
    return deep_gemm


def _uses_tensor_core(weight: torch.Tensor) -> bool:
    n, k = weight.shape
    return envs.VLLM_BI_BF16_GEMM == "tensor_core" and n % 64 == 0 and k % 128 == 0


# Per weight shape, the largest M whose DeepGEMM kernels are compiled.
_WARMED_M: dict[torch.Size, int] = {}


def _compile_tensor_core_configs(
    a: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
) -> None:
    """DeepGEMM picks its tile config from M and JIT-compiles each new one
    (~2 s, a serving stall); run every smaller M once when a weight first
    sees a larger batch, which the max-token profile run does at startup."""
    done = _WARMED_M.get(weight.shape, 0)
    if a.shape[0] <= done or torch.cuda.is_current_stream_capturing():
        return
    _WARMED_M[weight.shape] = a.shape[0]
    deep_gemm = _deterministic_deep_gemm()
    for m in range(done + 1, a.shape[0]):
        deep_gemm.bf16_gemm_nt(a[:m], weight, out[:m])


def _bf16_fixed_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    a, weight = a.contiguous(), weight.contiguous()
    _KERNEL._validate_inputs(a, weight, torch.float32)
    if a.shape[0] == 0:
        return torch.empty((0, weight.shape[0]), device=a.device, dtype=torch.float32)
    if _uses_tensor_core(weight):
        out = torch.empty(
            (a.shape[0], weight.shape[0]), device=a.device, dtype=torch.float32
        )
        _compile_tensor_core_configs(a, weight, out)
        _deterministic_deep_gemm().bf16_gemm_nt(a, weight, out)
        return out
    return _KERNEL(a, weight, torch.float32)


def _bf16_fixed_mm_fake(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (a.shape[0], weight.shape[0]), device=a.device, dtype=torch.float32
    )


direct_register_custom_op(
    op_name="bf16_fixed_mm",
    op_func=_bf16_fixed_mm,
    fake_impl=_bf16_fixed_mm_fake,
)


def bf16_fixed_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute live BF16 operands using a fixed K256, single-split reduction."""
    return torch.ops.vllm.bf16_fixed_mm(a, weight)
