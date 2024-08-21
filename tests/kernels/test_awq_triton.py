"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/test_awq_triton.py`.
"""
import pytest
import torch

from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton, awq_gemm_triton)

device = "cuda"

dequantize_threshold = 0.5
# This seems large, but this is using float16 with splitK and large sizes.
gemm_threshold = 10


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
def awq_dequantize_torch(qweight: torch.Tensor, scales: torch.Tensor,
                         qzeros: torch.Tensor) -> torch.Tensor:
    bits = 4
    group_size = 128
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None],
                                         shifts[None, None, :]).to(torch.int8)

    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(qzeros[:, :, None],
                                      shifts[None, None, :]).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_torch(input: torch.Tensor, qweight: torch.Tensor,
                   scales: torch.Tensor, qzeros: torch.Tensor,
                   split_k_iters: int) -> torch.Tensor:
    input_rows, input_cols = input.shape
    qweight_rows, qweight_cols = qweight.shape
    scales_rows, scales_cols = scales.shape
    weights = awq_dequantize_torch(qweight, scales, qzeros)
    return torch.matmul(input, weights)


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
@pytest.mark.parametrize("qweight_rows", [3584, 18944, 128, 256, 512, 1024])
@pytest.mark.parametrize("qweight_cols", [448, 576, 4736, 16, 32, 64, 128])
def test_dequantize(qweight_rows, qweight_cols):
    group_size = 128

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    torch.manual_seed(0)

    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          torch.iinfo(torch.int32).max,
                          (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)

    assert (not torch.any(torch.isinf(iweights_triton))
            and not torch.any(torch.isnan(iweights_triton)))

    iweights_torch = awq_dequantize_torch(qweight, scales, zeros)

    diff = iweights_torch - iweights_triton
    error = torch.sqrt(torch.sum(diff * diff / torch.numel(diff)))

    assert error < dequantize_threshold


# input   - [N, K]
# qweight - [K, M // 8]
# qzeros  - [K // G, M // 8]
# scales  - [K // G, M]
@pytest.mark.parametrize("N", [1, 2, 4, 8, 14, 16, 32, 64, 128])
@pytest.mark.parametrize("K", [3584, 18944, 128, 256, 512, 1024])
@pytest.mark.parametrize("M", [448, 576, 4736, 16, 32, 64, 128])
@pytest.mark.parametrize("splitK", [1, 8, 16])
def test_gemm(N, K, M, splitK):
    split_k_iters = splitK
    group_size = 128

    input_rows = N
    input_cols = K
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = M // 8
    scales_rows = qweight_rows // group_size
    scales_cols = M
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols

    torch.manual_seed(0)

    input = torch.rand((input_rows, input_cols),
                       dtype=input_dtype,
                       device=device)
    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            device=device)
    qzeros = torch.randint(0,
                           torch.iinfo(torch.int32).max,
                           (qzeros_rows, qzeros_cols),
                           device=device)
    scales = torch.rand((scales_rows, scales_cols),
                        dtype=scales_dtype,
                        device=device)

    output_triton = awq_gemm_triton(input, qweight, scales, qzeros,
                                    split_k_iters)

    assert (not torch.any(torch.isinf(output_triton))
            and not torch.any(torch.isnan(output_triton)))

    output_torch = awq_gemm_torch(input.cpu(), qweight.cpu(), scales.cpu(),
                                  qzeros.cpu(), split_k_iters)

    diff = output_torch.cpu() - output_triton.cpu()

    error = torch.sqrt(torch.sum(diff * diff / torch.numel(diff)))

    assert error < gemm_threshold
