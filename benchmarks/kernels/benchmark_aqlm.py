import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm.model_executor.layers.quantization.aqlm import dequantize_partioned_gemm
from vllm._C import ops

import torch
import torch.nn.functional as F

def main():
    methods = [
        dequantize_partioned_gemm, ops.aqlm_gemm
    ]

    filename = "./benchmark.csv"
    print(f"writing benchmarks to file {filename}")
    with open(filename, "a") as f:
        sys.stdout = f

        print('m | k | n', end='')
        for method in methods:
            print(f' | {method.__name__}', end='')
        print('')

        # These are reasonable prefill sizes.
        ksandpartions = ((4096, (4096, 4096, 4096)), (4096, (4096, )),
                         (4096, (11008, 11008)), (11008, (4096, )))

        # reasonable ranges for m.
        for m in [
                1, 2, 4, 8, #16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
                #2048, 3072, 4096
        ]:
            print(f'{m}', file=sys.__stdout__)
            for ksp in ksandpartions:
                run_grid(m, ksp[0], torch.tensor(ksp[1]), methods)

        sys.stdout = sys.__stdout__


def run_grid(m: int, k: int, parts: torch.tensor, methods):

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
                method=method,
            )

    n = parts.sum().item()
    print(f'{m} | {k} | {n}:{parts.tolist()}', end='')

    for method in methods:
        best_time_us = 1e20
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                m=m,
                k=k,
                parts=parts,
                method=method,
            )

            kernel_dur_us = 1000 * kernel_dur_ms

            if kernel_dur_us < best_time_us:
                best_time_us = kernel_dur_us

        print(f' | {kernel_dur_us:.0f}', end='')

    print('')


def run_timing(num_calls: int, m: int, k: int, parts: torch.tensor,
               method) -> float:

    n = parts.sum().item()

    device = torch.device('cuda:0')

    input = torch.randn((1, m, k), dtype=torch.float16, device=device)

    codes = torch.randint(-32768,
                          32768,
                          size=(n, k // 8, 1),
                          dtype=torch.int16,
                          device=device)

    codebooks = torch.randn(size=(parts.shape[0], 65536, 1, 8),
                            dtype=torch.float16,
                            device=device)

    scales = torch.randn(size=(n, 1, 1, 1), dtype=torch.float16, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        output = method(input, codes, codebooks, scales, parts, None)

    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    sys.exit(main())
