# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2024 The vLLM team.

import itertools

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed

# -----------------------------------------------------------------------------
# Configuration ranges
# -----------------------------------------------------------------------------
# Interpreted as a 2D tensor [M, N] where:
#   M ~ "tokens" or rows
#   N ~ "hidden dimension" or columns
m_range = [64, 256, 1024, 4096]
n_range = [64, 256, 1024, 4096]
configs = list(itertools.product(m_range, n_range))


# -----------------------------------------------------------------------------
# Helper: build scale / group_shape for different quantization modes
# -----------------------------------------------------------------------------
def build_scale_and_group_shape(
    mode: str,
    m: int,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, tuple[int, int] | None]:
    """Build scale tensor and group_shape depending on quantization mode.

    Modes:
      - "dynamic_per_tensor": dynamic per-tensor (scale=None)
      - "static_per_tensor":  static per-tensor (0D/1D scale)
      - "static_per_channel": static per-channel (1D scale, group_shape=(-1, 1))
      - "static_group_1x128": static group quant (group_m=1, group_n=128)
    """
    if mode == "dynamic_per_tensor":
        # No scale provided -> dynamic quantization.
        return None, None

    if mode == "static_per_tensor":
        # Per-tensor scale: 0D or length-1 tensor.
        scale = torch.ones(1, dtype=torch.float32, device=device)
        # group_shape (-1, -1) indicates full-extent (per-tensor).
        group_shape = (-1, -1)
        return scale, group_shape

    if mode == "static_per_channel":
        # 1D scale over N channels (per-column / per-channel).
        scale = torch.ones(n, dtype=torch.float32, device=device)
        # (-1, 1) means per-channel (across N dimension).
        group_shape = (-1, 1)
        return scale, group_shape

    if mode == "static_group_1x128":
        # group_shape = (1, 128) -> groups of 1 row x 128 columns.
        group_m, group_n = 1, 128
        if n % group_n != 0:
            # Return None to let the caller decide how to handle
            # (e.g., skip or return NaN)
            return None, None
        # scale.shape = [M/group_m, N/group_n] = [M, N/128]
        scale = torch.ones(
            m // group_m,
            n // group_n,
            dtype=torch.float32,
            device=device,
        )
        group_shape = (group_m, group_n)
        return scale, group_shape

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# Wrapper around ops.scaled_fp8_quant
# -----------------------------------------------------------------------------
def run_scaled_fp8_quant(
    x: torch.Tensor,
    mode: str,
    use_per_token_if_dynamic: bool = False,
    num_token_padding: int | None = None,
    scale_ub_value: float | None = None,
):
    """Thin wrapper around vLLM's scaled_fp8_quant.

    Args:
        x: Input tensor, must be 2D [M, N].
        mode: Quantization mode, see build_scale_and_group_shape.
        use_per_token_if_dynamic: Whether dynamic quant uses per-token scale.
        num_token_padding: Optional padding for the first dimension of output.
        scale_ub_value: Optional scalar upper bound used in dynamic per-token
                        scaling (will be turned into a tensor if provided).

    Returns:
        (x_fp8, scale): quantized tensor and scaling factor tensor.
        If a mode is incompatible with (M, N) (e.g. group size mismatch),
        returns (None, None) so that the caller can skip this config.
    """
    assert x.ndim == 2, "scaled_fp8_quant expects a 2D tensor [M, N]."
    m, n = x.shape
    device = x.device

    # Build scale and group_shape according to mode
    scale, group_shape = build_scale_and_group_shape(
        mode=mode,
        m=m,
        n=n,
        device=device,
    )

    # If None is returned (e.g. static_group_1x128 with incompatible N),
    # we signal the caller to skip this configuration.
    if mode == "static_group_1x128" and scale is None and group_shape is None:
        return None, None

    # Optional scale upper bound for dynamic per-token case
    scale_ub = None
    if scale_ub_value is not None:
        scale_ub = torch.tensor(
            [scale_ub_value],
            dtype=torch.float32,
            device=device,
        )

    # Let scaled_fp8_quant allocate output and scale when None is passed.
    x_fp8, scale_out = ops.scaled_fp8_quant(
        input=x,
        scale=scale,
        num_token_padding=num_token_padding,
        scale_ub=scale_ub,
        use_per_token_if_dynamic=use_per_token_if_dynamic,
        output=None,
        group_shape=group_shape,
    )

    return x_fp8, scale_out


# -----------------------------------------------------------------------------
# Benchmark function
# -----------------------------------------------------------------------------
def benchmark_scaled_fp8_quant(
    m: int,
    n: int,
    mode: str,
    dtype: torch.dtype,
):
    """Benchmark scaled_fp8_quant on a tensor of shape [m, n]."""
    device = "cuda"
    set_random_seed(42)
    torch.set_default_device(device)

    x = torch.randn(m, n, dtype=dtype, device=device)

    use_per_token_if_dynamic = mode == "dynamic_per_tensor"
    num_token_padding = None
    scale_ub_value = None

    def fn():
        return run_scaled_fp8_quant(
            x,
            mode=mode,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
            num_token_padding=num_token_padding,
            scale_ub_value=scale_ub_value,
        )

    # For invalid (m, n, mode) configurations
    # (e.g., group_1x128 with n not divisible by 128),
    # run_scaled_fp8_quant returns (None, None). We handle this here and return NaN.
    x_fp8, scale_out = fn()
    if x_fp8 is None and scale_out is None:
        # Return NaN to leave this data point blank in results
        return float("nan"), float("nan"), float("nan")

    # Otherwise, perform the actual benchmark with do_bench_cudagraph
    # (avoiding warmup time in measurement)
    def bench_fn():
        return run_scaled_fp8_quant(
            x,
            mode=mode,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
            num_token_padding=num_token_padding,
            scale_ub_value=scale_ub_value,
        )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        bench_fn, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, max_ms, min_ms


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = FlexibleArgumentParser(
        description=(
            "Benchmark vLLM scaled_fp8_quant kernel under different "
            "quantization modes and tensor sizes."
        )
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
        help="Input dtype for the tensor to be quantized.",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="scaled-fp8-quant",
        help="Name for the performance plot.",
    )

    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    line_vals = [
        "dynamic_per_tensor",
        "static_per_tensor",
        "static_per_channel",
        "static_group_1x128",
    ]
    line_names = [
        "dynamic_per_tensor",
        "static_per_tensor",
        "static_per_channel",
        "static_group_1x128",
    ]

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n"],
            x_vals=configs,
            line_arg="mode",
            line_vals=line_vals,
            line_names=line_names,
            styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("purple", "-")],
            ylabel="ms",
            plot_name=args.plot_name,
            args={
                "dtype": dtype,
            },
        )
    )

    def run_bench(
        m,
        n,
        mode,
        dtype,
    ):
        return benchmark_scaled_fp8_quant(
            m=m,
            n=n,
            mode=mode,
            dtype=dtype,
        )

    perf_report(run_bench).run(print_data=True)


if __name__ == "__main__":
    main()
