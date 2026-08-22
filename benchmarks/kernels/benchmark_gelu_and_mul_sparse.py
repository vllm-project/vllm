# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import torch

import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

NUM_TOKENS = [1, 2, 8, 32, 128, 512, 2048]
INTERMEDIATE_SIZES = [2048, 4096, 8192, 13824, 16384, 32768]
CONFIGS = list(itertools.product(NUM_TOKENS, INTERMEDIATE_SIZES))
STD_MULTIPLIER = 1.6448536269514722


@default_vllm_config()
def benchmark_gelu_and_mul_sparse(
    num_tokens: int,
    intermediate_size: int,
    provider: str,
    dtype: torch.dtype,
):
    set_random_seed(0)
    torch.set_default_device("cuda")
    x = torch.randn(num_tokens, 2 * intermediate_size, dtype=dtype)
    native = ir.ops.gelu_and_mul_sparse.impls["native"].impl_fn
    triton_impl = ir.ops.gelu_and_mul_sparse.impls["triton"].impl_fn
    args = (x, STD_MULTIPLIER, "tanh")

    expected = native(*args)
    actual = triton_impl(*args)
    torch.testing.assert_close(
        actual,
        expected,
        **ir.ops.gelu_and_mul_sparse.get_tolerance(actual.dtype),
    )

    if provider == "triton":
        fn = lambda: triton_impl(*args)
    elif provider == "native_compiled":
        compiled = torch.compile(native, fullgraph=True)
        compiled(*args)
        fn = lambda: compiled(*args)
    else:
        fn = lambda: native(*args)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, max_ms, min_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the Gemma3n sparse GELU activation."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "intermediate_size"],
            x_vals=CONFIGS,
            line_arg="provider",
            line_vals=["triton", "native_compiled", "native_eager"],
            line_names=["Triton", "Native (torch.compile)", "Native (eager)"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="ms",
            plot_name="gelu-and-mul-sparse-performance",
            args={},
        )
    )
    perf_report(
        lambda num_tokens, intermediate_size, provider: (
            benchmark_gelu_and_mul_sparse(
                num_tokens, intermediate_size, provider, dtype
            )
        )
    ).run(print_data=True)
