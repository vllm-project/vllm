# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# benchmark custom activation op performance
import itertools

import torch

import vllm.model_executor.layers.activation  # noqa F401
from vllm.model_executor.custom_op import op_registry
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

batch_size_range = [1, 16, 128]
seq_len_range = [1, 16, 64, 1024, 4096]
intermediate_size = [3072, 9728, 12288]
configs = list(itertools.product(batch_size_range, seq_len_range, intermediate_size))


def benchmark_activation(
    batch_size: int,
    seq_len: int,
    intermediate_size: int,
    provider: str,
    func_name: str,
    dtype: torch.dtype,
):
    device = "cuda"
    num_tokens = batch_size * seq_len
    dim = intermediate_size
    set_random_seed(42)
    torch.set_default_device(device)

    if func_name == "gelu_and_mul":
        layer = op_registry[func_name](approximate="none")
    elif func_name == "gelu_and_mul_tanh":
        layer = op_registry["gelu_and_mul"](approximate="tanh")
    elif func_name == "fatrelu_and_mul":
        threshold = 0.5
        layer = op_registry[func_name](threshold)
    else:
        layer = op_registry[func_name]()

    x = torch.randn(num_tokens, dim, dtype=dtype, device=device)
    compiled_layer = torch.compile(layer.forward_native)

    if provider == "custom":
        fn = lambda: layer(x)
    elif provider == "compiled":
        fn = lambda: compiled_layer(x)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, max_ms, min_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the custom activation op.")
    parser.add_argument(
        "--func-name",
        type=str,
        choices=[
            "mul_and_silu",
            "silu_and_mul",
            "gelu_and_mul",
            "gelu_and_mul_tanh",
            "fatrelu_and_mul",
            "swigluoai_and_mul",
            "gelu_new",
            "gelu_fast",
            "quick_gelu",
        ],
        default="silu_and_mul",
    )
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    args = parser.parse_args()
    assert args

    func_name = args.func_name
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "intermediate_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["custom", "compiled"],
            line_names=["Custom OP", "Compiled"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name=f"{func_name}-op-performance",
            args={},
        )
    )

    perf_report(
        lambda batch_size, seq_len, intermediate_size, provider: benchmark_activation(
            batch_size, seq_len, intermediate_size, provider, func_name, dtype
        )
    ).run(print_data=True)
