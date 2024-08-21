"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/test_awq_triton.py`.
"""
import argparse

import pytest
import torch

from vllm.model_executor.layers.quantization.awq import torch_awq_dequantize
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton, awq_gemm_triton)

device = "cuda"

dequantize_threshold = 0.5
# This seems large, but this is using float16 with splitK and large sizes.
gemm_threshold = 6


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
    return (iweights - zeros) * scales, zeros


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
    print(f"awq_gemm_torch:input_rows = {input_rows} input_cols = {input_cols}"
          f" qweight_rows = {qweight_rows} qweight_cols = {qweight_cols}"
          f" scales_rows = {scales_rows} scales_cols = {scales_cols}")
    weights = torch_awq_dequantize(qweight, scales, qzeros)
    return torch.matmul(input, weights)


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32


@pytest.mark.parametrize("qweight_rows", [3584, 18944, 128, 256, 512, 1024])
@pytest.mark.parametrize("qweight_cols", [448, 576, 4736, 16, 32, 64, 128])
def test_dequantize(qweight_rows, qweight_cols):
    print("=" * 10 + " TESTING DEQUANTIZE" + "=" * 10)

    group_size = 128

    print(f"qweight_rows = {qweight_rows}, qweight_cols = {qweight_cols}")
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    torch.manual_seed(0)

    qweight = torch.randint(0,
                            10000000, (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          10000000, (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)
    print(f"qweight = {qweight}")

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)

    print(f"Triton result:iweights_triton = {iweights_triton}")
    print("Any infs in triton result? -->"
          f"{torch.any(torch.isinf(iweights_triton))}")

    iweights_torch = torch_awq_dequantize(qweight, scales, zeros)
    print(f"Torch result:iweights_torch = {iweights_torch}")

    diff = iweights_torch - iweights_triton
    error = torch.sum(torch.sqrt(diff * diff))
    print(f"error = {error}")

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
    print("=" * 10 + " TESTING GEMM " + "=" * 10)

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
    print(f"input_rows = {input_rows} input_cols = {input_cols}"
          f" qweight_rows = {qweight_rows} qweight_cols = {qweight_cols}"
          f" scales_rows = {scales_rows} scales_cols = {scales_cols}")

    torch.manual_seed(2)
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

    # NOTE: Use to see more data and accuracy during testing.
    # import numpy as np
    # import sys
    # torch.set_printoptions(precision = 3,
    #                        threshold=10000000000000000000000000000,
    #                        sci_mode = False)
    # np.set_printoptions(threshold=sys.maxsize)

    output_torch = awq_gemm_torch(input.cpu(), qweight.cpu(), scales.cpu(),
                                  qzeros.cpu(), split_k_iters)
    print(f"output_torch = {output_torch}")

    output_triton = awq_gemm_triton(input, qweight, scales, qzeros,
                                    split_k_iters)

    print(f"output_triton = {output_triton}")
    print(f"output_triton.shape = {output_triton.shape}")
    print(f"Any infs in triton result? --> "
          f"{torch.any(torch.isinf(output_triton))}")

    diff = output_torch.cpu() - output_triton.cpu()
    error = torch.sum(torch.sqrt(diff * diff) / torch.numel(diff))
    print(f"error = {error}")

    assert error < gemm_threshold


def main():
    parser = argparse.ArgumentParser(
        description="awq_triton test driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test")
    known_args, unknown_args = parser.parse_known_args()
    if known_args.test is not None:
        if known_args.test == "dequantize":
            qweight_rows = 3584
            qweight_cols = 576
            small_test_size = False
            if small_test_size:
                qweight_rows = 256
                qweight_cols = 128
            test_dequantize(qweight_rows, qweight_cols)
        elif known_args.test == "gemm":
            small_test_size = True
            N = 1
            K = 256 if small_test_size else 3584
            M = 32 if small_test_size else 448
            splitK = 1
            test_gemm(N, K, M, splitK)
        else:
            print(f"Unknown test {known_args.test}")
    else:
        print("No test provided.")
        parser.print_help()


if __name__ == '__main__':
    main()
