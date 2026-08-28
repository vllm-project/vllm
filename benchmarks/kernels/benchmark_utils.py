# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import math
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
import torch.profiler as tpf

from vllm.utils.platform_utils import cuda_get_device_properties


def _compute_num_rotate_copies(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    num_iters: int,
) -> int:
    """Estimate how many input copies are needed to overflow the L2 cache,
    matching the logic in ``aiter.test_common.perftest``."""
    gpu_id = torch.accelerator.current_device_index()
    input_size = (
        sum(
            element.nbytes
            for element in args
            if isinstance(element, torch.Tensor) and element.device.index == gpu_id
        )
        + 1
    )

    function(*args)
    torch.accelerator.synchronize()

    (l2_cache_size,) = cuda_get_device_properties(
        gpu_id, ("L2_cache_size",), init_cuda=True
    )
    free_memory = torch.accelerator.get_memory_info(gpu_id)[0]
    cache_size = min(
        l2_cache_size * 64 * 128,
        (free_memory + input_size) * 0.9,
    )
    cache_size = max(cache_size, 0)
    num_copies = int((cache_size + input_size - 1) // input_size)
    return min(num_copies, num_iters)


def _build_rotated_args(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    num_iters: int,
) -> list[tuple[Any, ...]]:
    """Build a list of deep-copied argument tuples for L2 cache rotation."""
    num_copies = _compute_num_rotate_copies(function, args, num_iters)
    rotated = [tuple(copy.deepcopy(arg) for arg in args) for _ in range(num_copies - 1)]
    rotated.append(args)
    return rotated


def _print_results(
    label: str,
    median_us: float,
    minimum_us: float,
    maximum_us: float,
    baseline_us: float,
) -> None:
    ratio = median_us / baseline_us if baseline_us > 0 else float("inf")
    print(
        f"    {label}:  median={median_us:.2f}us, "
        f"min={minimum_us:.2f}us, max={maximum_us:.2f}us"
    )
    print(f"    {label}/rocblas: {ratio:.2f}x")


def _get_trace_perf_us(
    profiler: tpf.profile,
    num_iters: int,
) -> float:
    """Extract average GPU kernel time in us from a profiler trace.

    Mirrors ``aiter.test_common.get_trace_perf``: filters CUDA events, drops
    the first iteration as warmup, removes IQR outliers, and returns
    ``device_time_sum / actual_iters``.
    """
    if num_iters <= 1:
        raise ValueError("num_iters must be greater than one")
    profiler_warmup_iters = 1

    columns = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    rows = []
    for event in profiler.events():
        rows.append([getattr(event, column, None) for column in columns])
    dataframe = pd.DataFrame(rows, columns=columns)

    device_dataframe = dataframe[
        dataframe["device_type"].apply(
            lambda device_type: str(device_type).split(".")[-1] == "CUDA"
        )
    ].reset_index(drop=True)

    if device_dataframe.empty:
        raise RuntimeError(
            "Profiler captured no GPU events; the benchmarked kernel may have "
            "failed to launch"
        )

    kernel_names = device_dataframe["name"].tolist()
    total_device_events = len(kernel_names)

    kernels_per_iter = 1
    for candidate_count in range(1, total_device_events // 2 + 1):
        pattern = kernel_names[:candidate_count]
        full_repeats = total_device_events // candidate_count
        matches = all(
            kernel_names[i] == pattern[i % candidate_count]
            for i in range(full_repeats * candidate_count)
        )
        if matches:
            kernels_per_iter = candidate_count
            break

    actual_complete_iters = total_device_events // kernels_per_iter
    if actual_complete_iters != num_iters:
        raise RuntimeError(
            "Profiler captured an unexpected number of complete iterations: "
            f"expected {num_iters}, got {actual_complete_iters} "
            f"({total_device_events} GPU events, {kernels_per_iter} per iteration)"
        )
    usable_events = actual_complete_iters * kernels_per_iter
    device_dataframe = device_dataframe.iloc[:usable_events]

    grouped = device_dataframe.groupby(
        device_dataframe.index // kernels_per_iter,
        sort=False,
    ).agg({"self_device_time_total": "sum"})

    grouped = grouped.iloc[profiler_warmup_iters:].reset_index(drop=True)

    if len(grouped) > 30:
        q1 = grouped["self_device_time_total"].quantile(0.25)
        q3 = grouped["self_device_time_total"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        grouped = grouped[
            (grouped["self_device_time_total"] >= lower)
            & (grouped["self_device_time_total"] <= upper)
        ]

    if grouped.empty:
        raise RuntimeError("Profiler captured no complete measured iterations")

    average_us = grouped["self_device_time_total"].sum() / len(grouped)
    if not math.isfinite(average_us) or average_us <= 0:
        raise RuntimeError(f"Profiler returned invalid GPU time: {average_us!r} us")
    return average_us


def _profiler_bench_us(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    num_warmup: int,
    num_iters: int,
) -> float:
    """Profile *function* with rotated inputs and return avg GPU time in us.

    Matches the aiter tuner methodology:
    1. Build rotated input copies to overflow L2 cache.
    2. Warmup.
    3. Profile ``num_iters`` calls with input rotation.
    4. Extract GPU kernel time via ``_get_trace_perf_us``.
    """
    rotated_args = _build_rotated_args(function, args, num_iters)
    num_rotate = len(rotated_args)

    for _ in range(num_warmup):
        function(*args)
    torch.accelerator.synchronize()

    with tpf.profile(
        activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for iteration in range(num_iters):
            current_args = rotated_args[iteration % num_rotate]
            function(*current_args)
        torch.accelerator.synchronize()

    return _get_trace_perf_us(profiler, num_iters)
