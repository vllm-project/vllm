"""Tests for sparse cutlass kernels

Run `pytest tests/kernels/test_semi_structured.py`.
"""
from typing import Optional, Tuple, Type

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    sparse_cutlass_supported)
from vllm.platforms import current_platform

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.bfloat16)


def to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float16)


def prune_to_2_4(tensor):
    # Reshape tensor to [N, 4] where N is number of groups of 4
    original_shape = tensor.shape
    reshaped = tensor.reshape(-1, 4)

    # Get indices of top 2 absolute values in each group of 4
    _, indices = torch.topk(torch.abs(reshaped), k=2, dim=1)

    # Create binary mask
    mask = torch.zeros_like(reshaped)
    mask.scatter_(dim=1,
                  index=indices,
                  src=torch.ones_like(indices, dtype=mask.dtype))

    # Apply mask and reshape back
    pruned = reshaped * mask

    # Turn all -0.0 to 0.0
    pruned[pruned == -0.0] = 0.0

    return pruned.reshape(original_shape)


def make_rand_sparse_tensors(
        dtype: torch.dtype, m: int, n: int, k: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device='cuda') * 5
    b = torch.randn((n, k), device='cuda').t() * 5

    b = prune_to_2_4(b.t()).t()

    if dtype == torch.int8:
        a, b = to_int8(a), to_int8(b)
    elif dtype == torch.float8_e4m3fn:
        a, b = to_fp8(a), to_fp8(b)
    elif dtype == torch.float16:
        a, b = to_fp16(a), to_fp16(b)
    elif dtype == torch.bfloat16:
        a, b = to_bf16(a), to_bf16(b)
    else:
        raise ValueError("unsupported dtype")

    b_compressed, e = ops.cutlass_sparse_compress(b.t())

    # Compressed B, Metadata, Original A, B
    return b_compressed, e, a, b


def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: Type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    output = (scale_a * (scale_b * (torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))))).to(out_dtype)
    if bias is not None:
        output = output + bias

    return output


@pytest.mark.skipif(not sparse_cutlass_supported(),
                    reason="Sparse FP8 is not yet supported on this GPU type.")
# Test working with a subset of A and B for sparse matmul
def test_cutlass_sparse_subset():

    big_m = 1024
    m, n, k = 512, 512, 512

    # Create tensors
    b_comp, e, whole_a, b = make_rand_sparse_tensors(torch.float8_e4m3fn,
                                                     big_m, n, k)
    a = whole_a[0:m, 0:k]
    scale_a = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, 1), device="cuda", dtype=torch.float32) / 10

    out = ops.cutlass_scaled_sparse_mm(a,
                                       b_comp,
                                       e,
                                       scale_a,
                                       scale_b,
                                       out_dtype=torch.bfloat16)
    baseline = baseline_scaled_mm(a,
                                  b,
                                  scale_a,
                                  scale_b,
                                  out_dtype=torch.bfloat16)

    torch.testing.assert_close(out, baseline, rtol=1e-1, atol=1e0)
