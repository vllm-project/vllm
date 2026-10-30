# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import bisect
import math
import statistics
from collections import defaultdict
from collections.abc import Callable

import torch


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def cupti_kernel_details(
    runner: Callable, repeats: int, flush: torch.Tensor | None
) -> dict:
    """Expose per-kernel activities using FlashInfer's CUPTI tracing method.

    FlashInfer's public helper returns only the first-to-last GPU span. This
    separate trace retains kernel names/durations and their sum, allowing a
    host-induced gap inside an eager sequence to be distinguished from work.
    """
    from cupti import cupti

    kernels, launches, intervals = [], [], []
    configurations = {}

    def completed(activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                fields = (
                    "grid_x",
                    "grid_y",
                    "grid_z",
                    "block_x",
                    "block_y",
                    "block_z",
                    "cluster_x",
                    "cluster_y",
                    "cluster_z",
                    "registers_per_thread",
                    "static_shared_memory",
                    "dynamic_shared_memory",
                )
                configurations[str(activity.name)] = {
                    field: int(getattr(activity, field))
                    for field in fields
                    if getattr(activity, field, None) is not None
                }
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                kernels.append(
                    (
                        str(activity.name)
                        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL
                        else str(activity.kind),
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                    )
                )
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append((activity.start, activity.correlation_id))

    kinds = (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )
    try:
        for kind in kinds:
            cupti.activity_enable(kind)
        cupti.activity_register_callbacks(lambda: (8 * 1024 * 1024, 0), completed)
        for _ in range(repeats):
            if flush is not None:
                flush.zero_()
            torch.accelerator.synchronize()
            start = cupti.get_timestamp()
            runner()
            end = cupti.get_timestamp()
            torch.accelerator.synchronize()
            intervals.append((start, end))
        cupti.activity_flush_all(0)
    finally:
        for kind in kinds:
            cupti.activity_disable(kind)
        cupti.finalize()

    launches.sort()
    starts = [item[0] for item in launches]
    by_id = defaultdict(list)
    for kernel in kernels:
        by_id[kernel[3]].append(kernel)
    by_name = defaultdict(list)
    spans, sums, counts = [], [], []
    for start, end in intervals:
        lo, hi = bisect.bisect_left(starts, start), bisect.bisect_right(starts, end)
        ids = {correlation for _, correlation in launches[lo:hi]}
        selected = [kernel for key in ids for kernel in by_id[key]]
        if not selected:
            raise RuntimeError("CUPTI returned no kernels for a timed invocation")
        spans.append((max(k[2] for k in selected) - min(k[1] for k in selected)) / 1000)
        sums.append(sum((k[2] - k[1]) / 1000 for k in selected))
        counts.append(len(selected))
        for name, begin, finish, _ in selected:
            by_name[name].append((finish - begin) / 1000)
    return {
        "gpu_span_us": summarize(spans),
        "gpu_kernel_sum_us": summarize(sums),
        "kernels_per_invocation": summarize(counts),
        "kernels": {
            name: {
                **summarize(times),
                "calls_per_invocation": len(times) / repeats,
                "launch_config": configurations.get(name, {}),
            }
            for name, times in by_name.items()
        },
    }
