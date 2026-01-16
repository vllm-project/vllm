# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Callable
from unittest.mock import patch

import pandas as pd
import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE


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
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)

    quant_fp8 = QuantFP8(False, group_shape, column_major_scales=False)

    torch_out, torch_scale = bench_compile(quant_fp8.forward_native)(x)
    torch_eager_out, torch_eager_scale = quant_fp8.forward_native(x)
    cuda_out, cuda_scale = quant_fp8.forward_cuda(x)

    try:
        torch.testing.assert_close(
            cuda_out.to(torch.float32),
            torch_out.to(torch.float32),
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(cuda_scale, torch_scale, rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(
            cuda_out.to(torch.float32),
            torch_eager_out.to(torch.float32),
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(cuda_scale, torch_eager_scale, rtol=1e-3, atol=1e-5)
        print("✅ All implementations match")
    except AssertionError as e:
        print("❌ Implementations differ")
        print(e)


configs = []


def benchmark_quantization(
    batch_size,
    hidden_size,
    provider,
    group_shape: GroupShape,
    col_major: bool,
    dtype: torch.dtype,
):
    device = torch.device("cuda")

    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)

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
        result = (
            df.groupby(groupby_cols)
            .apply(geo_speedup, include_groups=False)
            .reset_index()
        )

    return result


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the various implementations of QuantFP8 (dynamic-only)"
    )
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument(
        "--dtype", type=str, choices=["half", "bfloat16", "float"], default="bfloat16"
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[896, 1024, 2048, 4096, 7168],
        help="Hidden sizes to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 128, 512, 1024],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Group sizes for GroupShape(1,N) to benchmark. "
        "Use 0 for PER_TENSOR, -1 for PER_TOKEN (default: 0,-1,64,128)",
    )
    parser.add_argument(
        "--no-column-major",
        action="store_true",
        help="Disable column-major scales testing",
    )

    args = parser.parse_args()
    assert args

    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    hidden_sizes = args.hidden_sizes
    batch_sizes = args.batch_sizes

    if args.group_sizes is not None:
        group_shapes = []
        for size in args.group_sizes:
            if size == 0:
                group_shapes.append(GroupShape.PER_TENSOR)
            elif size == -1:
                group_shapes.append(GroupShape.PER_TOKEN)
            else:
                group_shapes.append(GroupShape(1, size))
    else:
        group_shapes = [
            GroupShape.PER_TENSOR,
            GroupShape.PER_TOKEN,
            GroupShape(1, 64),
            GroupShape(1, 128),
        ]

    column_major_scales = [False] if args.no_column_major else [True, False]

    config_gen = itertools.product(
        group_shapes,
        column_major_scales,
        batch_sizes,
        hidden_sizes,
    )

    # filter out column-major scales for non-group, reverse order
    configs.extend(c[::-1] for c in config_gen if (c[0].is_per_group() or not c[1]))

    print(f"Running {len(configs)} configurations:")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Group shapes: {[str(g) for g in group_shapes]}")
    print(f"  Column major scales: {column_major_scales}")
    print()

    if args.check:
        for group_shape in group_shapes:
            group_size = group_shape[1]
            print(f"{group_size=}")
            calculate_diff(
                batch_size=4, hidden_size=4096, group_shape=group_shape, dtype=dtype
            )

    benchmark = triton.testing.perf_report(
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
    )(benchmark_quantization)

    df = benchmark.run(print_data=True, dtype=dtype, return_df=True)

    # Print geomean speedups
    geo_table_grouped = compute_geomean_speedups(
        df,
        baseline_col="Torch (Compiled)",
        speedup_cols=["CUDA", "Triton"],
        groupby_cols=["col_major", "group_shape"],
    )

    print("Speedup over Torch (Compiled)")
    print(geo_table_grouped.to_string(index=False))
