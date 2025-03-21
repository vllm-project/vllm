"""Benchmarks for the AWQ Triton kernel.

Run `python benchmarks/kernels/benchmark_awq_triton.py`.
"""
import argparse
import contextlib

import torch
import triton
import triton.profiler as proton

from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton, awq_gemm_triton)
from vllm.utils import is_hip

device = "cuda"
use_proton = False
benchmark_unit = "gbps"  # can be gpbs or ms
warmup = 25
rep = 100

# (qweight_rows, qweight_cols)
dequantize_benchmark_vals = [
    (256, 128),
    (128, 128),
    (3584, 448 * 8),
    (3584, 576 * 8),
    (3584, 4736 * 8),
    (18944, 448 * 8),
]

# (N, K, M, splitK)
gemm_benchmark_vals = [
    (1, 256, 128, 1),
    (1, 256, 128, 8),
    (128, 128, 128, 1),
    (128, 128, 128, 8),
    (1, 18944, 448 * 8, 8),
    (1, 3584, 448 * 8, 8),
    (1, 3584, 4736 * 8, 8),
    (1, 3584, 576 * 8, 8),
    (1, 18944, 448 * 8, 8),
    (14, 3584, 448 * 8, 8),
    (14, 3584, 4736 * 8, 8),
    (14, 3584, 576 * 8, 8),
]


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


def get_line_vals():
    if is_hip():
        return ["triton"]
    return ["triton", "cuda"]


def get_line_names():
    if is_hip():
        return ["Triton"]
    return ["Triton", "CUDA"]


def get_ylabel():
    if benchmark_unit == "gbps":
        return "GB/s"
    return "gbps"


def awq_dequantize_cuda(qweight: torch.Tensor, scales: torch.Tensor,
                        qzeros: torch.Tensor) -> torch.Tensor:
    K = qweight.shape[0]
    M = scales.shape[1]
    G = qweight.shape[0] // scales.shape[0]
    with proton.scope(
            f"cuda_awq_dequantize M={M}, K={K}, G={G}", {
                "bytes": (qweight.element_size() * M * K +
                          scales.element_size() * K // G * M +
                          qzeros.element_size() * K // G * M),
                "flops":
                2 * M * K
            }) if use_proton else dummy_context_mgr():
        return torch.ops._C.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)


def generate_dequantize_benchmark_vals():
    return dequantize_benchmark_vals


dequantize_benchmark_obj = triton.testing.Benchmark(
    x_names=["qweight_rows", "qweight_cols"],
    x_vals=generate_dequantize_benchmark_vals(),
    x_log=True,
    line_arg="provider",
    line_vals=get_line_vals(),
    line_names=get_line_names(),
    styles=[("blue", "-"), ("green", "-")],
    ylabel=get_ylabel(),
    plot_name="Dequantize performance",
    args={},
)


@triton.testing.perf_report(dequantize_benchmark_obj)
def bench_dequantize(qweight_rows, qweight_cols, provider):
    group_size = 128

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols
    qzeros_dtype = torch.int32

    torch.manual_seed(0)

    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    qzeros = torch.randint(0,
                           torch.iinfo(torch.int32).max,
                           (qzeros_rows, qzeros_cols),
                           dtype=qzeros_dtype,
                           device=device)
    scales = torch.rand((scales_rows, scales_cols),
                        dtype=scales_dtype,
                        device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_cuda(qweight, scales, qzeros),
            quantiles=quantiles,
            warmup=warmup,
            rep=rep)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_triton(qweight, scales, qzeros),
            quantiles=quantiles,
            warmup=warmup,
            rep=rep)

    K = qweight.shape[0]
    M = scales.shape[0]
    if benchmark_unit == "gbps":
        perf = lambda ms: 2 * M * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    return ms, max_ms, min_ms


def awq_gemm_cuda(input: torch.Tensor, qweight: torch.Tensor,
                  qzeros: torch.Tensor, scales: torch.Tensor,
                  split_k_iters: int) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    bytes_per_elem = input.element_size()
    with proton.scope(f"cuda_awq_gemm M={M}, N={N}, K={K}", {
            "bytes": bytes_per_elem * (M * K + N * K),
            "flops": 2. * M * N * K
    }) if use_proton else dummy_context_mgr():
        return torch.ops._C.awq_gemm(input, qweight, qzeros, scales,
                                     split_k_iters)


def generate_gemm_benchmark_vals():
    # (N, K, M, splitK)
    return [
        (1, 256, 128, 1),
        (1, 256, 128, 8),
        (128, 128, 128, 1),
        (128, 128, 128, 8),
        (1, 18944, 448 * 8, 8),
        (1, 3584, 448 * 8, 8),
        (1, 3584, 4736 * 8, 8),
        (1, 3584, 576 * 8, 8),
        (1, 18944, 448 * 8, 8),
        (14, 3584, 448 * 8, 8),
        (14, 3584, 4736 * 8, 8),
        (14, 3584, 576 * 8, 8),
    ]


gemm_benchmark_obj = triton.testing.Benchmark(
    x_names=["N", "K", "M", "splitK"],
    x_vals=generate_gemm_benchmark_vals(),
    x_log=True,
    line_arg="provider",
    line_vals=get_line_vals(),
    line_names=get_line_names(),
    styles=[("blue", "-"), ("green", "-")],
    ylabel=get_ylabel(),
    plot_name="GEMM performance",
    args={},
)


@triton.testing.perf_report(gemm_benchmark_obj)
def bench_gemm(N, K, M, splitK, provider):
    group_size = 128

    input_rows = N
    input_cols = K
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = M // 8
    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = M
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols
    qzeros_dtype = torch.int32

    torch.manual_seed(0)
    input = torch.rand((input_rows, input_cols),
                       dtype=input_dtype,
                       device=device)
    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    qzeros = torch.randint(0,
                           torch.iinfo(torch.int32).max,
                           (qzeros_rows, qzeros_cols),
                           dtype=qzeros_dtype,
                           device=device)
    scales = torch.rand((scales_rows, scales_cols),
                        dtype=scales_dtype,
                        device=device)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_gemm_cuda(input, qweight, scales, qzeros, splitK),
            quantiles=quantiles,
            warmup=warmup,
            rep=rep)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_gemm_triton(input, qweight, scales, qzeros, splitK),
            quantiles=quantiles,
            warmup=warmup,
            rep=rep)
    if benchmark_unit == "gbps":
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    return ms, max_ms, min_ms


def main():
    parser = argparse.ArgumentParser(
        description="awq_triton bench driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bench", choices=["dequantize", "gemm"])
    parser.add_argument("--proton", action="store_true")
    parser.add_argument("--gemm_N", type=int)
    parser.add_argument("--gemm_M", type=int)
    parser.add_argument("--gemm_K", type=int)
    parser.add_argument("--gemm_splitK", type=int)
    parser.add_argument("--dequantize_rows", type=int)
    parser.add_argument("--dequantize_cols", type=int)
    parser.add_argument("--benchmark_unit", choices=["gbps", "ms"])
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rep", type=int)
    known_args, unknown_args = parser.parse_known_args()

    benchmark = known_args.bench

    global use_proton
    use_proton = known_args.proton

    global benchmark_unit
    benchmark_unit = known_args.benchmark_unit

    global warmup
    if known_args.warmup is not None:
        warmup = known_args.warmup

    global rep
    if known_args.warmup is not None:
        rep = known_args.rep

    if known_args.bench is not None:
        benchmark = known_args.bench

    if benchmark == "gemm":
        if use_proton:
            proton.start("awq_gemm", hook="triton")
            proton.activate(0)

        if (known_args.gemm_N is not None and known_args.gemm_M is not None
                and known_args.gemm_K is not None
                and known_args.gemm_splitK is not None):
            global gemm_benchmark_obj
            gemm_benchmark_obj.x_vals = [
                (known_args.gemm_N, known_args.gemm_K, known_args.gemm_M,
                 known_args.gemm_splitK)
            ]

        bench_gemm.run(save_path=".", show_plots=True, print_data=True)
        if use_proton:
            proton.deactivate(0)
            proton.finalize()
    elif benchmark == "dequantize":
        if (known_args.dequantize_rows is not None
                and known_args.dequantize_cols is not None):
            global dequantize_benchmark_obj
            dequantize_benchmark_obj.x_vals = [(known_args.dequantize_rows,
                                                known_args.dequantize_cols)]
        if use_proton:
            proton.start("awq_dequantize", hook="triton")
            proton.activate(0)
        bench_dequantize.run(save_path=".", show_plots=True, print_data=True)
        if use_proton:
            proton.deactivate(0)
            proton.finalize()
    else:
        print(f"Unknown bench {known_args.bench}")


if __name__ == "__main__":
    main()
