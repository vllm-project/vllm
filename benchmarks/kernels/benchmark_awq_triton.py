"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/test_awq_triton.py`.
"""
import argparse
import contextlib

import torch
import triton
import triton.profiler as proton

from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton, awq_gemm_triton)


@contextlib.contextmanager
def dummy_context_mgr():
    yield None


device = "cuda"
use_proton = False


def awq_dequantize_cuda(qweight: torch.Tensor, scales: torch.Tensor,
                        qzeros: torch.Tensor) -> torch.Tensor:
    K = qweight.shape
    M = scales.shape[1]
    G = qweight.shape[0] // scales.shape[0]
    bytes_per_elem = qweight.element_size()
    with proton.scope(
            f"cuda_awq_dequantize M={M}, N={N}, K={K}", {
                "bytes":
                qweight.element_size() * M * K +
                scales.element_size * K // G * M +
                qzeros.element_size() * K // G * M,
                "flops":
                M * K + 2 * K // G * M
            }) if use_proton else dummy_context_mgr():
        return torch.ops._C.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)


def generate_dequantize_benchmark_vals():
    # (qweight_rows, qweight_cols)
    return [
        (256, 128),
        (128, 128),
        (3584, 448 * 8),
        (3584, 576 * 8),
        (3584, 4736 * 8),
        (18944, 448 * 8),
    ]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["qweight_rows", "qweight_cols"],
        x_vals=generate_dequantize_benchmark_vals(),
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "cuda"],
        line_names=["Triton", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="Dequantize performance",
        args={},
    ))
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
            quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_dequantize_triton(qweight, scales, qzeros),
            quantiles=quantiles)

    K = qweight.shape[0]
    M = scales.shape[0]
    perf = lambda ms: 2 * M * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N", "K", "M", "splitK"],
        x_vals=generate_gemm_benchmark_vals(),
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "cuda"],
        line_names=["Triton", "CUDA"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="GEMM performance",
        args={},
    ))
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
            quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: awq_gemm_triton(input, qweight, scales, qzeros, splitK),
            quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def main():
    parser = argparse.ArgumentParser(
        description="awq_triton bench driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bench")
    parser.add_argument("--proton", action="store_true")
    known_args, unknown_args = parser.parse_known_args()

    benchmark = known_args.bench
    use_proton = known_args.proton

    if known_args.bench is not None:
        benchmark = known_args.bench

    if benchmark == "gemm":
        if use_proton:
            proton.start("awq_gemm", hook="triton")
        proton.activate(0)
        bench_gemm.run(save_path=".", show_plots=True, print_data=True)
        proton.deactivate(0)
        if use_proton:
            proton.finalize()
    elif benchmark == "dequantize":
        if use_proton:
            proton.start("awq_dequantize", hook="triton")
        proton.activate(0)
        bench_dequantize.run(save_path=".", show_plots=True, print_data=True)
        proton.deactivate(0)
        if use_proton:
            proton.finalize()
    else:
        print(f"Unknown bench {known_args.bench}")


if __name__ == "__main__":
    main()
