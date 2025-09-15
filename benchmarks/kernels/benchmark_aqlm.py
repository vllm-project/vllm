# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.aqlm import (
    dequantize_weight, generic_dequantize_gemm, get_int_dtype,
    optimized_dequantize_gemm)
from vllm.utils import FlexibleArgumentParser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def torch_mult(
        input: torch.Tensor,  #  [..., in_features]
        weights: torch.Tensor,
        scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
) -> torch.Tensor:
    output = F.linear(input, weights)
    return output


def dequant_out_scale(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: torch.IntTensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:

    weights = ops.aqlm_dequant(codes, codebooks, output_partition_sizes)

    if bias is None:
        output = F.linear(input, weights, bias)
        orig_shape = output.shape
        flattened_output = output.view(-1, output.size(-1))
        f_scales = scales.view(-1, scales.shape[0])
        b_scales = f_scales.expand(flattened_output.shape[0], -1)
        flattened_output *= b_scales
        return flattened_output.view(orig_shape)
    else:
        b_scales = scales.view(scales.shape[:-3] + (-1, )).expand(
            -1, weights.shape[1])
        weights *= b_scales
        return F.linear(input, weights, bias)


def dequant_weight_scale(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: torch.IntTensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:

    weights = ops.aqlm_dequant(codes, codebooks, output_partition_sizes)

    b_scales = scales.view(scales.shape[:-3] + (-1, )).expand(
        -1, weights.shape[1])
    weights *= b_scales
    return F.linear(input, weights, bias)


def dequant_no_scale(
    input: torch.Tensor,  #  [..., in_features]
    codes: torch.IntTensor,  #  [num_out_groups, num_in_groups, num_codebooks]
    codebooks: torch.
    Tensor,  #  [num_codebooks, codebook_size, out_group_size, in_group_size]
    scales: torch.Tensor,  #  [num_out_groups, 1, 1, 1]
    output_partition_sizes: torch.IntTensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:

    weights = ops.aqlm_dequant(codes, codebooks, output_partition_sizes)

    return F.linear(input, weights, bias)


# Compare the optimized 1x16 and 2x8 cuda decompression/dequant kernels against
# the generic pytorch version.
# Just visual comparison.
def dequant_test(k: int, parts: torch.Tensor, nbooks: int, bits: int) -> None:

    n = int(parts.sum().item())

    device = torch.device('cuda:0')

    code_range = (1 << bits) // 2
    ingroups = 8

    codes = torch.randint(-code_range,
                          code_range,
                          size=(n, k // ingroups, nbooks),
                          dtype=get_int_dtype(bits),
                          device=device)

    codebooks = torch.randn(size=(parts.shape[0] * nbooks, 1 << bits, 1, 8),
                            dtype=torch.float16,
                            device=device)

    count = 0
    for index in range(16):
        for i in range(8):
            for book in range(nbooks):
                codebooks[book, index, 0, i] = count * (10**book)
            count += 1

    print("codes shape", codes.shape)

    for i in range(16):
        for book in range(nbooks):
            codes[0, i, book] = i
            codes[0, -i, book] = i

    weights = dequantize_weight(codes, codebooks, None)
    weights2 = ops.aqlm_dequant(codes, codebooks, parts)

    print("weights shape:", weights.shape)
    print("weights2 shape:", weights2.shape)

    print("weights are:", weights)
    print("weights2 are:", weights2)

    print("first 128 weights are", weights[0, 0:128].to(torch.int32))
    print("first 128 weights2 are:", weights2[0, 0:128].to(torch.int32))

    print("last 128 weights are", weights[0, -128:])
    print("last 128 weights2 are:", weights2[0, -128:])


def main():

    parser = FlexibleArgumentParser(description="Benchmark aqlm performance.")

    # Add arguments
    parser.add_argument("--nbooks",
                        type=int,
                        default=1,
                        help="Number of codebooks (default: 1)")
    parser.add_argument("--bits",
                        type=int,
                        default=16,
                        help="Number of bits per code element (default: 16)")
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Run the decompression/dequant tester rather than benchmarking "
        "(default: False)")

    # Parse the arguments
    args = parser.parse_args()

    # Extract values
    nbooks = args.nbooks
    bits = args.bits

    if args.test:
        dequant_test(4096, torch.tensor((4096, )), nbooks, bits)
        return

    # Otherwise, benchmark.
    methods = [
        ops.aqlm_gemm,
        dequant_out_scale,
        generic_dequantize_gemm,
        optimized_dequantize_gemm,
        dequant_weight_scale,
        torch_mult,
        dequant_no_scale,
    ]

    filename = f"./aqlm_benchmark_{nbooks}x{bits}.csv"
    print(f"writing benchmarks to file {filename}")
    with open(filename, "w") as f:
        sys.stdout = f

        print('m | k | n | n parts', end='')
        for method in methods:
            print(f" | {method.__name__.replace('_', ' ')} (µs)", end='')
        print('')

        # These are reasonable prefill sizes.
        ksandpartions = ((4096, (4096, 4096, 4096)), (4096, (4096, )),
                         (4096, (11008, 11008)), (11008, (4096, )))

        # reasonable ranges for m.
        for m in [
                1, 2, 4, 8, 10, 12, 14, 16, 24, 32, 48, 52, 56, 64, 96, 112,
                128, 256, 512, 1024, 1536, 2048, 3072, 4096
        ]:
            print(f'{m}', file=sys.__stdout__)
            for ksp in ksandpartions:
                run_grid(m, ksp[0], torch.tensor(ksp[1]), nbooks, bits,
                         methods)

        sys.stdout = sys.__stdout__


def run_grid(m: int, k: int, parts: torch.Tensor, nbooks: int, bits: int,
             methods):

    # I didn't see visible improvements from increasing these, but feel free :)
    num_warmup_trials = 1
    num_trials = 1

    num_calls = 100

    # warmup.
    for method in methods:
        for _ in range(num_warmup_trials):
            run_timing(
                num_calls=num_calls,
                m=m,
                k=k,
                parts=parts,
                nbooks=nbooks,
                bits=bits,
                method=method,
            )

    n = parts.sum().item()
    print(f'{m} | {k} | {n} | {parts.tolist()}', end='')

    for method in methods:
        best_time_us = 1e20
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                m=m,
                k=k,
                parts=parts,
                nbooks=nbooks,
                bits=bits,
                method=method,
            )

            kernel_dur_us = 1000 * kernel_dur_ms

            if kernel_dur_us < best_time_us:
                best_time_us = kernel_dur_us

        print(f' | {kernel_dur_us:.0f}', end='')

    print('')


def run_timing(num_calls: int, m: int, k: int, parts: torch.Tensor,
               nbooks: int, bits: int, method) -> float:

    n = int(parts.sum().item())

    device = torch.device('cuda:0')

    input = torch.randn((1, m, k), dtype=torch.float16, device=device)

    code_range = (1 << bits) // 2
    ingroups = 8

    codes = torch.randint(-code_range,
                          code_range,
                          size=(n, k // ingroups, nbooks),
                          dtype=get_int_dtype(bits),
                          device=device)

    codebooks = torch.randn(size=(parts.shape[0] * nbooks, 1 << bits, 1, 8),
                            dtype=torch.float16,
                            device=device)

    scales = torch.randn(size=(n, 1, 1, 1), dtype=torch.float16, device=device)

    # for comparison to just a pytorch mult.
    weights = torch.randn((n, k), dtype=torch.float16, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if method is torch_mult:
        for i in range(num_calls):
            torch_mult(input, weights, scales)
    else:
        for i in range(num_calls):
            method(input, codes, codebooks, scales, parts, None)

    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    sys.exit(main())
