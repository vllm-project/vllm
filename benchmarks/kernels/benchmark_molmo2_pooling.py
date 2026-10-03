# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import statistics
from collections.abc import Callable

import torch
from tabulate import tabulate

from vllm.model_executor.models.molmo2 import _prepare_molmo2_pooling
from vllm.utils.argparse_utils import FlexibleArgumentParser

SHAPES = {
    "image": (4, (256, 550, 1024)),
    "video": (9, (256, 1024, 4096)),
}


def time_cuda(function: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def interquartile_range(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[2] - quartiles[0]


def assert_outputs_close(
    actual_outputs: tuple[torch.Tensor, ...],
    expected_outputs: tuple[torch.Tensor, ...],
) -> None:
    for actual, expected in zip(actual_outputs, expected_outputs):
        for actual_chunk, expected_chunk in zip(actual.split(64), expected.split(64)):
            torch.testing.assert_close(
                actual_chunk, expected_chunk, rtol=1e-2, atol=1e-2
            )


def upstream_native(
    image_features: torch.Tensor,
    token_pooling: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, _, dim = image_features.shape
    valid = token_pooling >= 0
    batch_idx = torch.arange(
        batch_size,
        dtype=torch.long,
        device=token_pooling.device,
    )
    batch_idx = torch.tile(
        batch_idx.view(batch_size, 1, 1),
        [1, token_pooling.shape[1], token_pooling.shape[2]],
    )
    to_pool = image_features.reshape(batch_size, -1, dim)[
        batch_idx, torch.clamp(token_pooling, min=0)
    ]
    to_pool = to_pool * valid.to(image_features.dtype)[..., None]
    to_pool = to_pool.reshape(-1, token_pooling.shape[-1], dim)

    denom = valid.reshape(-1, valid.shape[-1]).float().sum(-1).clamp_min(1)
    query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
    return (
        to_pool,
        query,
        valid.reshape(-1, 1, 1, valid.shape[-1]),
        valid.any(-1),
    )


def make_inputs(
    batch_size: int,
    groups: int,
    pool_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    num_crops = 5 if pool_size == 4 else 8
    num_patches = 729
    image_features = torch.randn(
        batch_size,
        num_crops,
        num_patches,
        2304,
        device="cuda",
        dtype=dtype,
    )
    token_pooling = torch.randint(
        0,
        num_crops * num_patches,
        (batch_size, groups, pool_size),
        device="cuda",
    )
    token_pooling[:, ::11, -1] = -1
    return image_features, token_pooling


@torch.inference_mode()
def benchmark_case(
    batch_size: int,
    groups: int,
    pool_size: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
    trials: int,
) -> tuple[float, float, float, float]:
    image_features, token_pooling = make_inputs(batch_size, groups, pool_size, dtype)
    baseline = lambda: upstream_native(image_features, token_pooling)
    optimized = lambda: _prepare_molmo2_pooling(
        image_features,
        token_pooling,
        masked_average=True,
    )
    assert_outputs_close(optimized(), baseline())

    timings: dict[str, list[float]] = {"upstream": [], "optimized": []}
    functions = {"upstream": baseline, "optimized": optimized}
    for trial in range(trials):
        order = (
            ("upstream", "optimized") if trial % 2 == 0 else ("optimized", "upstream")
        )
        for name in order:
            timings[name].append(time_cuda(functions[name], warmup, repeats))

    return (
        statistics.median(timings["upstream"]),
        interquartile_range(timings["upstream"]),
        statistics.median(timings["optimized"]),
        interquartile_range(timings["optimized"]),
    )


def main(args) -> None:
    rows = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for kind, (pool_size, group_sizes) in SHAPES.items():
            for batch_size in args.batch_sizes:
                for groups in group_sizes:
                    upstream_us, upstream_iqr, optimized_us, optimized_iqr = (
                        benchmark_case(
                            batch_size,
                            groups,
                            pool_size,
                            dtype,
                            args.warmup,
                            args.repeats,
                            args.trials,
                        )
                    )
                    rows.append(
                        [
                            kind,
                            dtype_name,
                            batch_size,
                            groups,
                            pool_size,
                            upstream_us,
                            upstream_iqr,
                            optimized_us,
                            optimized_iqr,
                            upstream_us / optimized_us,
                        ]
                    )
                    gc.collect()
                    torch.accelerator.empty_cache()

    print("Molmo2 pooling preparation (median CUDA-event latency)")
    print(
        tabulate(
            rows,
            headers=[
                "kind",
                "dtype",
                "batch",
                "groups",
                "K",
                "upstream (us)",
                "upstream IQR",
                "optimized (us)",
                "optimized IQR",
                "speedup",
            ],
            floatfmt=("", "", "d", "d", "d", ".2f", ".2f", ".2f", ".2f", ".2f"),
        )
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--dtypes",
        choices=["float16", "bfloat16"],
        nargs="+",
        default=["float16", "bfloat16"],
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--trials", type=int, default=10)
    main(parser.parse_args())
