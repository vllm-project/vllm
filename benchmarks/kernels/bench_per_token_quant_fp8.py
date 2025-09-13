# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from typing import Callable
from unittest.mock import patch

import pandas as pd
import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.triton_utils import triton
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, FlexibleArgumentParser


def with_triton_mode(fn):
    """Temporarily force the Triton fallback path"""

    def wrapped(*args, **kwargs):
        with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
            return fn(*args, **kwargs)

    return wrapped


# TODO(luka): use standalone_compile utility
def with_dyn_arg(fn: Callable, arg_index: int, dim_index: int):
    def inner(*args):
        torch._dynamo.mark_dynamic(args[arg_index], dim_index)
        return fn(*args)

    return inner


def bench_compile(fn: Callable):
    # recompile for different shapes
    fwd = torch.compile(fn, fullgraph=True, dynamic=False)

    # First dim is explicitly dynamic to simulate vLLM usage
    return with_dyn_arg(fwd, 0, 0)


torch._dynamo.config.recompile_limit = 8888


def calculate_diff(
    batch_size: int,
    hidden_size: int,
    group_shape: GroupShape,
    dtype: torch.dtype,
):
    """Calculate the difference between Inductor and CUDA implementations."""
    device = torch.device("cuda")
    x = torch.rand((batch_size * hidden_size, 4096), dtype=dtype, device=device)

    quant_fp8 = QuantFP8(False, group_shape, column_major_scales=False)

    torch_out, torch_scale = bench_compile(quant_fp8.forward_native)(x)
    torch_eager_out, torch_eager_scale = quant_fp8.forward_native(x)
    cuda_out, cuda_scale = quant_fp8.forward_cuda(x)

    out_allclose = lambda o1, o2: torch.allclose(
        o1.to(torch.float32),
        o2.to(torch.float32),
        rtol=1e-3,
        atol=1e-5,
    )
    scale_allclose = lambda s1, s2: torch.allclose(s1, s2, rtol=1e-3, atol=1e-5)

    if (
        out_allclose(cuda_out, torch_out)
        and scale_allclose(cuda_scale, torch_scale)
        and out_allclose(cuda_out, torch_eager_out)
        and scale_allclose(cuda_scale, torch_eager_scale)
    ):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


hidden_sizes = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]
batch_sizes = [1, 16, 32, 64, 128]
group_shapes = [
    GroupShape.PER_TENSOR,
    GroupShape.PER_TOKEN,
    GroupShape(1, 64),
    GroupShape(1, 128),
]
column_major_scales = [True, False]

config_gen = itertools.product(
    group_shapes,
    column_major_scales,
    batch_sizes,
    hidden_sizes,
)

# filter out column-major scales for non-group, reverse order
configs = list(c[::-1] for c in config_gen if (c[0].is_per_group() or not c[1]))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["hidden_size", "batch_size", "col_major", "group_shape"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["torch", "cuda", "triton"],
        line_names=["Torch (Compiled)", "CUDA", "Triton"],
        styles=[("blue", "-"), ("green", "-"), ("black", "-")],
        ylabel="us",
        plot_name="QuantFP8 performance",
        args={},
    )
)
def benchmark_quantization(
    batch_size,
    hidden_size,
    provider,
    group_shape: GroupShape,
    col_major: bool,
    dtype: torch.dtype,
):
    device = torch.device("cuda")

    x = torch.randn(batch_size * hidden_size, 4096, device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    quant_fp8 = QuantFP8(False, group_shape, column_major_scales=col_major)

    if provider == "torch":
        fn = lambda: bench_compile(quant_fp8.forward_native)(x.clone())
    elif provider == "cuda":
        fn = lambda: quant_fp8.forward_cuda(x.clone())
    elif provider == "triton":
        if not group_shape.is_per_group():
            # Triton only supported for per-group
            return 0, 0, 0

        fn = lambda: with_triton_mode(quant_fp8.forward_cuda)(x.clone())

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


# TODO(luka) extract to utils
def compute_geomean_speedups(
    df: pd.DataFrame,
    baseline_col: str,
    speedup_cols: list[str],
    groupby_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute geometric mean speedups over a baseline column.

    Args:
        df: Input dataframe
        baseline_col: Column to use as baseline
        speedup_cols: Columns to compute speedups for
        groupby_cols: Columns to group by. If None, compute over entire df.

    Returns:
        pd.DataFrame with geometric mean speedups
    """
    from scipy.stats import gmean

    def geo_speedup(group: pd.DataFrame) -> pd.Series:
        ratios = {
            col: (group[baseline_col] / group[col]).values for col in speedup_cols
        }
        return pd.Series({col: gmean(vals) for col, vals in ratios.items()})

    if groupby_cols is None:
        result = geo_speedup(df).to_frame().T
    else:
        result = df.groupby(groupby_cols).apply(geo_speedup).reset_index()

    return result


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the various implementations of QuantFP8 (dynamic-only)"
    )
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="half"
    )

    args = parser.parse_args()
    assert args

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    if args.check:
        for group_shape in group_shapes:
            group_size = group_shape[1]
            print(f"{group_size=}")
            calculate_diff(
                batch_size=4, hidden_size=4096, group_shape=group_shape, dtype=dtype
            )

    df = benchmark_quantization.run(print_data=True, dtype=dtype, return_df=True)

    # Print geomean speedups
    geo_table_grouped = compute_geomean_speedups(
        df,
        baseline_col="Torch (Compiled)",
        speedup_cols=["CUDA", "Triton"],
        groupby_cols=["col_major", "group_shape"],
    )

    print("Speedup over Torch (Compiled)")
    print(geo_table_grouped.to_string(index=False))
