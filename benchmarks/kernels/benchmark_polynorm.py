# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

from vllm import _custom_ops as vllm_ops
from vllm.triton_utils import triton


def polynorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    def norm(x, eps: float):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    x = x.float()
    return (
        (
            weight[0] * norm(x**3, eps)
            + weight[1] * norm(x**2, eps)
            + weight[2] * norm(x, eps)
            + bias
        )
        .to(weight.dtype)
        .view(orig_shape)
    )


def polynorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    out = torch.empty_like(x)
    vllm_ops.poly_norm(out, x, weight, bias, eps)
    output = out

    output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_dim):
    dtype = torch.bfloat16
    x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device="cuda")
    weight = torch.ones(3, dtype=dtype, device="cuda")
    bias = torch.ones(1, dtype=dtype, device="cuda")

    output_naive = polynorm_naive(x, weight, bias)
    output_vllm = polynorm_vllm(x, weight, bias)

    if torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [2**i for i in range(0, 7, 2)]
seq_length_range = [2**i for i in range(6, 11, 1)]
dim_range = [2048, 4096]
configs = list(itertools.product(dim_range, batch_size_range, seq_length_range))


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["dim", "batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["naive", "vllm"],
            line_names=["Naive", "vLLM"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name="polynorm-perf",
            args={},
        )
    )
    def benchmark(dim, batch_size, seq_len, provider):
        dtype = torch.bfloat16
        hidden_dim = dim * 4

        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device="cuda")
        weight = torch.ones(3, dtype=dtype, device="cuda")
        bias = torch.ones(1, dtype=dtype, device="cuda")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "naive":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: polynorm_naive(x, weight, bias),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: polynorm_vllm(x, weight, bias),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=8192,
        help="Intermediate size of MLP",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/polnorm/",
        help="Path to save polnorm benchmark results",
    )

    args = parser.parse_args()

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
    )

    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
