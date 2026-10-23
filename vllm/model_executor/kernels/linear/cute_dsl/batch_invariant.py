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

# Widest projection that takes the split-K path under VLLM_BI_BF16_SPLITK.
_SPLITK_MAX_N = 384


class SplitKBf16Gemm(LLBf16Gemm):
    """Cluster split-K with a slice count fixed by the environment: each row
    sums the same K256 tiles in the same order whatever M is."""

    def __init__(self, split_k: int) -> None:
        super().__init__()
        self.split_k = split_k

    def dispatch(
        self, *, M: int, K: int, N: int, output_dtype: torch.dtype = torch.float32
    ) -> LLBf16Gemm.CompileKey:
        return self.CompileKey(
            backend="splitk",
            split_k=self.split_k,
            num_stages=4,
            output_dtype=output_dtype,
        )


@functools.cache
def _splitk_kernel(split_k: int) -> SplitKBf16Gemm:
    return SplitKBf16Gemm(split_k)


def _uses_splitk(weight: torch.Tensor) -> bool:
    return envs.VLLM_BI_BF16_SPLITK > 0 and weight.shape[0] <= _SPLITK_MAX_N


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


def compile_tensor_core_rows(weight: torch.Tensor, rows: int) -> None:
    """Compile ``weight``'s DeepGEMM configs for up to ``rows`` rows now, for a
    weight whose largest batches (such as speculative verification at the
    lm_head) arrive only while serving."""
    if (
        not _uses_tensor_core(weight)
        or _uses_splitk(weight)
        or rows <= _WARMED_M.get(weight.shape, 0)
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    a = weight.new_zeros((rows, weight.shape[1]))
    out = torch.empty((rows, weight.shape[0]), device=a.device, dtype=torch.float32)
    _compile_tensor_core_configs(a, weight, out)
    _deterministic_deep_gemm().bf16_gemm_nt(a, weight, out)


def _bf16_fixed_mm(a: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    a, weight = a.contiguous(), weight.contiguous()
    _KERNEL._validate_inputs(a, weight, torch.float32)
    if a.shape[0] == 0:
        return torch.empty((0, weight.shape[0]), device=a.device, dtype=torch.float32)
    if _uses_splitk(weight):
        # Every slice owns at least one K256 tile; the count depends on K only.
        split_k = min(envs.VLLM_BI_BF16_SPLITK, -(-weight.shape[1] // 256))
        return _splitk_kernel(split_k)(a, weight, torch.float32)
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
