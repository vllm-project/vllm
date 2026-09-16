# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import statistics
from collections.abc import Callable

import torch
from tabulate import tabulate

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.molmo2_pooling import Molmo2PoolingPreparation
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
    with set_current_vllm_config(VllmConfig()):
        op = Molmo2PoolingPreparation(masked_average=True)

    native = lambda: op.forward_native(image_features, token_pooling)
    fused = lambda: op(image_features, token_pooling)
    assert_outputs_close(fused(), native())

    timings: dict[str, list[float]] = {"native": [], "fused": []}
    functions = {"native": native, "fused": fused}
    for trial in range(trials):
        order = ("native", "fused") if trial % 2 == 0 else ("fused", "native")
        for name in order:
            timings[name].append(time_cuda(functions[name], warmup, repeats))

    return (
        statistics.median(timings["native"]),
        interquartile_range(timings["native"]),
        statistics.median(timings["fused"]),
        interquartile_range(timings["fused"]),
    )


def main(args) -> None:
    rows = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for kind, (pool_size, group_sizes) in SHAPES.items():
            for batch_size in args.batch_sizes:
                for groups in group_sizes:
                    native_us, native_iqr, fused_us, fused_iqr = benchmark_case(
                        batch_size,
                        groups,
                        pool_size,
                        dtype,
                        args.warmup,
                        args.repeats,
                        args.trials,
                    )
                    rows.append(
                        [
                            kind,
                            dtype_name,
                            batch_size,
                            groups,
                            pool_size,
                            native_us,
                            native_iqr,
                            fused_us,
                            fused_iqr,
                            native_us / fused_us,
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
                "native (us)",
                "native IQR",
                "fused (us)",
                "fused IQR",
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
