# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/quantization/test_awq_triton.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.awq_triton import (
    AWQ_FUSED_FP32_SUPPORTED,
    AWQ_TRITON_SUPPORTED_GROUP_SIZES,
    awq_dequantize_triton,
    awq_gemm_fused_fp32,
    awq_gemm_triton,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="AWQ Triton kernels require CUDA/ROCm or XPU.",
)

device = current_platform.device_type


def reverse_awq_order(t: torch.Tensor):
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        t.shape[-1],
        dtype=torch.int32,
        device=t.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    t = t[:, reverse_order_tensor] & 0xF
    return t


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def awq_dequantize_torch(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
@pytest.mark.parametrize("qweight_rows", [3584, 18944, 128, 256, 512, 1024])
@pytest.mark.parametrize("qweight_cols", [448, 576, 4736, 16, 32, 64, 128])
@pytest.mark.parametrize("group_size", AWQ_TRITON_SUPPORTED_GROUP_SIZES)
def test_dequantize(qweight_rows, qweight_cols, group_size):
    if group_size == -1:
        group_size = qweight_rows

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    set_random_seed(0)

    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (qweight_rows, qweight_cols),
        dtype=qweight_dtype,
        device=device,
    )
    scales = torch.rand(scales_rows, scales_cols, dtype=scales_dtype, device=device)
    zeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (zeros_rows, zeros_cols),
        dtype=zeros_dtype,
        device=device,
    )

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)

    assert not torch.any(torch.isinf(iweights_triton)) and not torch.any(
        torch.isnan(iweights_triton)
    )

    iweights_torch = awq_dequantize_torch(qweight, scales, zeros, group_size)

    torch.testing.assert_close(iweights_triton, iweights_torch)


# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
@pytest.mark.parametrize("N", [1, 2, 4, 8, 14, 17, 23, 32])
@pytest.mark.parametrize("K", [128])
@pytest.mark.parametrize("M", [16, 24, 32])
@pytest.mark.parametrize("group_size", AWQ_TRITON_SUPPORTED_GROUP_SIZES)
@pytest.mark.parametrize("splitK", [1, 8])
def test_gemm(N, K, M, splitK, group_size):
    if group_size == -1:
        group_size = K

    split_k_iters = splitK

    input_rows = N
    input_cols = K
    input_dtype = torch.float32
    qweight_rows = input_cols
    qweight_cols = M // 8
    scales_rows = qweight_rows // group_size
    scales_cols = M
    scales_dtype = torch.float32
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols

    set_random_seed(0)

    input = torch.rand((input_rows, input_cols), dtype=input_dtype, device=device)
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (qweight_rows, qweight_cols), device=device
    )
    qzeros = torch.randint(
        0, torch.iinfo(torch.int32).max, (qzeros_rows, qzeros_cols), device=device
    )
    scales = torch.rand((scales_rows, scales_cols), dtype=scales_dtype, device=device)

    output_triton = awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)

    assert not torch.any(torch.isinf(output_triton)) and not torch.any(
        torch.isnan(output_triton)
    )

    dequantized_weights = awq_dequantize_triton(qweight, scales, qzeros)

    output_torch = torch.matmul(input, dequantized_weights)

    assert not torch.any(torch.isinf(output_torch)) and not torch.any(
        torch.isnan(output_torch)
    )

    torch.testing.assert_close(
        output_triton.cpu(), output_torch.cpu(), atol=1e-1, rtol=1e-1
    )


fused_fp32_skip = pytest.mark.skipif(
    not AWQ_FUSED_FP32_SUPPORTED,
    reason="awq_gemm_fused_fp32 requires CUDA on SM89.",
)


def _assert_bit_identical(a: torch.Tensor, b: torch.Tensor) -> None:
    """torch.testing.assert_close(atol=0, rtol=0) still treats +0.0 and
    -0.0 as equal; compare raw bytes to catch that and any other
    bit-level divergence."""
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _make_fused_gemm_inputs(m: int, n: int, k: int, group_size: int):
    input = torch.rand((m, k), dtype=torch.float16, device=device)
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (k, n // 8), dtype=torch.int32, device=device
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (k // group_size, n // 8),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((k // group_size, n), dtype=torch.float16, device=device)
    return input, qweight, scales, qzeros


# M=127/128/129 straddle the kernel's (32,32)->(128,64) tiling switch at M=128.
@fused_fp32_skip
@pytest.mark.parametrize("M", [1, 32, 127, 128, 129, 256])
@pytest.mark.parametrize("N", [32, 64, 96, 160])
@pytest.mark.parametrize("K", [128, 256])
def test_awq_gemm_fused_fp32(M, N, K):
    group_size = 128

    set_random_seed(0)

    input, qweight, scales, qzeros = _make_fused_gemm_inputs(M, N, K, group_size)

    output_fused = awq_gemm_fused_fp32(input, qweight, scales, qzeros)

    assert not torch.any(torch.isinf(output_fused))
    assert not torch.any(torch.isnan(output_fused))

    dequantized_weights = awq_dequantize_triton(qweight, scales, qzeros)
    output_ref = torch.matmul(input, dequantized_weights)

    torch.testing.assert_close(output_fused, output_ref, atol=1e-1, rtol=1e-1)


@fused_fp32_skip
def test_awq_gemm_fused_fp32_rejects_non_exact_group_count():
    # K=16416, 128 groups: 16416 // 128 == 128 (integer-division truncation)
    # but 16416 != 128 * 128, so the pre-fix code silently read 32 rows past
    # the end of scales/qzeros for the last (partial) group.
    m, k, n, num_groups = 32, 16416, 32, 128
    input = torch.rand((m, k), dtype=torch.float16, device=device)
    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (k, n // 8), dtype=torch.int32, device=device
    )
    scales = torch.rand((num_groups, n), dtype=torch.float16, device=device)
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (num_groups, n // 8),
        dtype=torch.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="exact multiple"):
        awq_gemm_fused_fp32(input, qweight, scales, qzeros)


@fused_fp32_skip
@pytest.mark.parametrize("N", [32, 64, 96, 160])
@pytest.mark.parametrize("K", [128, 256])
def test_awq_gemm_fused_fp32_batch_invariant(N, K):
    """The kernel switches (BLOCK_SIZE_M, BLOCK_SIZE_N) tiling at M=128; a
    fixed row's output must be bit-identical regardless of how many other
    rows share its launch, or batch-invariant mode's guarantee is broken."""
    group_size = 128
    set_random_seed(0)

    qweight = torch.randint(
        0, torch.iinfo(torch.int32).max, (K, N // 8), dtype=torch.int32, device=device
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K // group_size, N // 8),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((K // group_size, N), dtype=torch.float16, device=device)
    row = torch.rand((1, K), dtype=torch.float16, device=device)

    reference = awq_gemm_fused_fp32(row, qweight, scales, qzeros)

    for m in (32, 127, 128, 129, 256):
        batch = torch.rand((m, K), dtype=torch.float16, device=device)
        batch[0] = row[0]
        output = awq_gemm_fused_fp32(batch, qweight, scales, qzeros)
        _assert_bit_identical(output[0], reference[0])
