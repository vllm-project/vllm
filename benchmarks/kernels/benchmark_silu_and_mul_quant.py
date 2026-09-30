# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the fused SiLU-and-mul + static per-tensor FP8 quant kernel
(torch.ops._C.silu_and_mul_quant) against the Inductor-compiled native chain
that runs when the manual fusion is not wired."""

import itertools

import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

num_tokens_range = [1, 16, 32, 64, 256, 1024, 4096, 16384]
intermediate_size_range = [8192, 10752, 14336, 28672]
configs = list(itertools.product(num_tokens_range, intermediate_size_range))


@default_vllm_config()
def benchmark(num_tokens: int, intermediate_size: int, provider: str, dtype):
    torch.set_default_device("cuda")
    set_random_seed(42)
    x = torch.randn(num_tokens, 2 * intermediate_size, dtype=dtype)
    scale = torch.tensor([0.02], dtype=torch.float32)
    fp8_dtype = current_platform.fp8_dtype()

    if provider == "custom":
        out = torch.empty(num_tokens, intermediate_size, dtype=fp8_dtype)

        def fn():
            torch.ops._C.silu_and_mul_quant(out, x, scale)

    else:
        act = SiluAndMul()
        quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

        def native(x):
            return quant.forward_native(act.forward_native(x), scale)[0]

        compiled = torch.compile(native)
        compiled(x)

        def fn():
            compiled(x)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1e3, max_ms * 1e3, min_ms * 1e3


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark silu_and_mul_quant.")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16"], default="bfloat16"
    )
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "intermediate_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["custom", "compiled"],
            line_names=["silu_and_mul_quant (us)", "Inductor native (us)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="silu_and_mul_quant-performance",
            args={},
        )
    )
    perf_report(
        lambda num_tokens, intermediate_size, provider: benchmark(
            num_tokens, intermediate_size, provider, dtype
        )
    ).run(print_data=True)
