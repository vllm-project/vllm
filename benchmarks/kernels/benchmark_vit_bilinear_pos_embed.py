# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Benchmarks the fused Triton bilinear position-embedding kernel against
# the pure-PyTorch (native) implementation used in Qwen3-VL ViT models.
#
# == Usage Examples ==
#
# Default benchmark:
#   python3 benchmark_vit_bilinear_pos_embed.py
#
# Custom parameters:
#   python3 benchmark_vit_bilinear_pos_embed.py --hidden-dim 1152 \
#       --num-grid-per-side 48 --save-path ./configs/vit_pos_embed/

import itertools

import torch

from vllm.model_executor.models.qwen3_vl import (
    pos_embed_interpolate_native,
    triton_pos_embed_interpolate,
)
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

# (h, w) configurations to benchmark
h_w_configs = [
    (16, 16),
    (32, 32),
    (48, 48),
    (64, 64),
    (128, 128),
    (32, 48),
    (60, 80),
]

# Temporal dimensions
t_range = [1]

configs = list(itertools.product(t_range, h_w_configs))


def get_benchmark(
    num_grid_per_side: int,
    spatial_merge_size: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["t", "h_w"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["native", "triton"],
            line_names=["Native (PyTorch)", "Triton"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="us",
            plot_name=(
                f"vit-bilinear-pos-embed-"
                f"grid{num_grid_per_side}-"
                f"dim{hidden_dim}-"
                f"{dtype}"
            ),
            args={},
        )
    )
    def benchmark(t, h_w, provider):
        h, w = h_w

        torch.manual_seed(42)
        embed_weight = (
            torch.randn(
                num_grid_per_side * num_grid_per_side,
                hidden_dim,
                device=device,
                dtype=dtype,
            )
            * 0.25
        )

        quantiles = [0.5, 0.2, 0.8]

        if provider == "native":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: pos_embed_interpolate_native(
                    embed_weight,
                    t,
                    h,
                    w,
                    num_grid_per_side,
                    spatial_merge_size,
                    dtype,
                ),
                quantiles=quantiles,
            )
        else:
            assert HAS_TRITON, "Triton not available"
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_pos_embed_interpolate(
                    embed_weight,
                    t,
                    h,
                    w,
                    num_grid_per_side,
                    spatial_merge_size,
                    dtype,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark bilinear position embedding interpolation."
    )
    parser.add_argument(
        "--num-grid-per-side",
        type=int,
        default=48,
        help="Position embedding grid size (default: 48 for Qwen3-VL)",
    )
    parser.add_argument(
        "--spatial-merge-size",
        type=int,
        default=2,
        help="Spatial merge size (default: 2)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=1152,
        help="Embedding hidden dimension (default: 1152 for Qwen3-VL)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda:0", "cuda:1"],
        default="cuda:0",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./vit_pos_embed/",
    )
    args = parser.parse_args()

    dtype = torch.bfloat16

    bench = get_benchmark(
        args.num_grid_per_side,
        args.spatial_merge_size,
        args.hidden_dim,
        dtype,
        args.device,
    )
    bench.run(print_data=True, save_path=args.save_path)
