# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utility script to benchmark vLLM's RMSNorm implementation against FlashInfer's 
and HuggingFace's. Can be used to generate performance data for a range of batch sizes
and hidden sizes, with or without residual connection.

Optionally, users can enable CUDA graphs and multiple calls
per benchmark run to help stabilize measurements.
This is particularly important for avoiding the launch overhead
which dominates the latency for small batch sizes.

Note that batch_size does not account for num_heads,
so in practice the effective batch size for the kernel may be num_requests * num_heads.
Users can set the batch_size argument accordingly, to account for those cases.

Example usage:

# Measure GPT-OSS-shape RMSNorm without residuals, including the launch overhead
python3 benchmarks/kernels/benchmark_rmsnorm.py --hidden-sizes 2880 --use-residual False

# Measure+Profile DeepSeek-shape RMSNorm (with residuals, on by default) for low-latency

nsys profile --cuda-graph-trace=node \
    python3 benchmarks/kernels/benchmark_rmsnorm.py \
    --hidden-sizes 7168 \
    --calls-per-run 20 \
    --use-cuda-graph

"""

import itertools
import os
from collections.abc import Callable

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from torch import nn

from vllm import _custom_ops as vllm_ops
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.triton_utils import triton


# TODO(luka): use standalone_compile utility
def with_dyn_arg(fn: Callable, arg_index: int, dim_index: int):
    def inner(*args):
        torch._dynamo.mark_dynamic(args[arg_index], dim_index)
        return fn(*args)

    return inner


def bench_compile(fn: Callable):
    # recompile for different shapes
    fwd = torch.compile(fn, fullgraph=True, dynamic=False)

    return fwd
    # First dim is explicitly dynamic to simulate vLLM usage
    # return with_dyn_arg(fwd, 0, 0)


torch._dynamo.config.recompile_limit = 8888


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_flashinfer(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        fused_add_rmsnorm(x, residual, weight, eps)
        output = (x, residual)
    else:
        output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_vllm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def rmsnorm_torch_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    norm = RMSNorm(x.shape[-1], eps=eps).to(x.device)
    norm.weight = nn.Parameter(weight)

    orig_shape = x.shape

    def f(x, residual):
        x = x.view(-1, x.shape[-1])
        if residual is not None:
            residual = residual.view(-1, residual.shape[-1])
        output = norm.forward_native(x, residual)
        if isinstance(output, tuple):
            output = (output[0].view(orig_shape), output[1].view(orig_shape))
        else:
            output = output.view(orig_shape)
        return output

    f_compiled = bench_compile(norm.forward_native)

    return lambda x, weight, residual: f_compiled(x, residual)


def calculate_diff(batch_size, hidden_size, use_residual):
    dtype = torch.bfloat16
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_flashinfer = rmsnorm_flashinfer(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_vllm = rmsnorm_vllm(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_torch = rmsnorm_torch_func(
        x.clone(), weight, residual.clone() if residual is not None else None
    )(x.clone(), None, residual.clone() if residual is not None else None)

    if use_residual:
        output_naive = output_naive[0]
        output_flashinfer = output_flashinfer[0]
        output_vllm = output_vllm[0]
        output_torch = output_torch[0]

    print(f"Naive output={output_naive}")
    print(f"FlashInfer output={output_flashinfer}")
    print(f"vLLM output={output_vllm}")
    print(f"Torch output={output_torch}")

    if (
        torch.allclose(output_naive, output_flashinfer, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_vllm, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_torch, atol=1e-2, rtol=1e-2)
    ):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


def get_benchmark(args):
    provider_titles = (
        ["HuggingFace", "FlashInfer", "vLLM", "Torch"]
        if not args.omit_huggingface
        else ["FlashInfer", "vLLM", "Torch"]
    )
    providers = [x.lower() for x in provider_titles]
    use_residual = args.use_residual

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "hidden_size"],
            x_vals=[
                list(_) for _ in itertools.product(args.batch_sizes, args.hidden_sizes)
            ],
            line_arg="provider",
            line_vals=providers,
            line_names=provider_titles,
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("orange", "-")][
                : len(providers)
            ],
            ylabel="us",
            plot_name=f"rmsnorm-perf-{'with' if use_residual else 'without'}-residual",
            args={},
        )
    )
    def benchmark(batch_size, hidden_size, provider):
        dtype = torch.bfloat16

        input_buffers = [
            dict(
                x=torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda"),
                weight=torch.ones(hidden_size, dtype=dtype, device="cuda"),
                residual=torch.randn(
                    batch_size, hidden_size, dtype=dtype, device="cuda"
                )
                if use_residual
                else None,
            )
            for _ in range(args.calls_per_run)
        ]

        quantiles = [0.5, 0.2, 0.8]

        provider_fn = {
            "huggingface": rmsnorm_naive,
            "flashinfer": rmsnorm_flashinfer,
            "vllm": rmsnorm_vllm,
            "torch": rmsnorm_torch_func(**input_buffers[0]),
        }[provider]
        run_fn = lambda: [provider_fn(**inputs) for inputs in input_buffers]

        if args.use_cuda_graph:
            if args.calls_per_run <= 1:
                print(
                    "Warning: Using CUDA graph with calls_per_run <= 1"
                    " may not provide meaningful results."
                )

            # Warmup for graph capture
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    run_fn()
            torch.cuda.current_stream().wait_stream(s)

            # Capture the graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run_fn()

            # Redefine run_fn to replay the graph
            run_fn = lambda: g.replay()

        torch.cuda.nvtx.range_push(
            f"RMSNorm benchmark: {provider=}, {hidden_size=},"
            f" {batch_size=} {use_residual=} {dtype=}"
        )

        ms, min_ms, max_ms = triton.testing.do_bench(run_fn, quantiles=quantiles)

        torch.cuda.nvtx.range_pop()

        scale = (
            1000 / args.calls_per_run
        )  # Convert to microseconds and account for calls per run
        return scale * ms, scale * max_ms, scale * min_ms

    return benchmark


@default_vllm_config()
def main(args):
    os.makedirs(args.save_path, exist_ok=True)

    # Run correctness test
    calculate_diff(
        batch_size=args.batch_sizes[-1],
        hidden_size=args.hidden_sizes[-1],
        use_residual=args.use_residual,
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[2**i for i in range(0, 17, 1)]
    )  # 1 to 65536
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[2880, 4096, 7168],  # GPT-OSS, Qwen3.5, DeepSeek V3
    )
    parser.add_argument(
        "--use-residual",
        type=bool,
        default=True,
        help="Whether to use residual connection. Defaults to True",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./configs/rmsnorm/",
        help="Path to save rmsnorm benchmark results",
    )
    parser.add_argument(
        "--omit-huggingface",
        action="store_true",
        help="Whether to omit HuggingFace implementation in the benchmark.",
    )
    parser.add_argument(
        "--calls-per-run",
        type=int,
        default=1,
        help="""Number of calls to the RMSNorm function per benchmark run.
        Each call will use its own input buffers.
        Used to help stabilize low-latency kernel measurements and account for PDL.
        Can be used in combination with use-cuda-graph.
        """,
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="""Use a cuda graph for each kernel invocation. Should be used alongside
        calls-per-run, as it will otherwise not have anything to record in each graph.
        """,
    )

    args = parser.parse_args()

    main(args)
