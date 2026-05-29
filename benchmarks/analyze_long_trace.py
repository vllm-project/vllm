#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze long torch-profiler traces for blockscale SplitK zero-init fusion.

This script promotes the one-off analysis used for the Qwen3-Next
blockscale SplitK zero-init investigation into a reusable tool. It expects a
results directory laid out like:

    results/
      splitk/
        dp0_*.pt.trace.json
      splitk_fused/
        dp0_*.pt.trace.json

The analysis is intentionally trace-format-light: it uses only event names,
timestamps, durations, and coarse categories. That keeps it useful across
minor torch-profiler schema changes.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CONTAMINATION_PATTERNS = (
    "## Call CompiledFxGraph",
    "compile_attempt_0",
    "_compile.compile_inner",
    "InductorBenchmarker.benchmark",
    "OutputGraph.call_user_compiler",
    "_recursive_pre_grad_passes",
    "_recursive_post_grad_passes",
    "compile_fx_inner",
    "fx_codegen_and_compile",
    "GraphLowering",
    "Scheduler.codegen",
    "CachingAutotuner.benchmark_all_configs",
    "(dynamo_timed)",
)


@dataclass(frozen=True)
class Event:
    name: str
    ts: float
    dur: float
    cat: str

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass(frozen=True)
class ModeTrace:
    mode: str
    path: Path
    events: list[dict]
    kernels: list[Event]
    runtime_events: list[Event]


@dataclass(frozen=True)
class BurstSummary:
    start: float
    end: float
    kernels: list[Event]
    fmoe_count: int
    total_us_per_fmoe: float
    kernels_per_fmoe: float
    fill_us_per_fmoe: float
    fill_count_per_fmoe: float
    fmoe_durations: list[float]
    by_kernel_us_per_fmoe: dict[str, float]


@dataclass(frozen=True)
class SliceSummary:
    wall_us: float
    gpu_busy_us: float
    gpu_idle_us: float
    cpu_launch_us: float
    n_kernels: int
    n_fmoe: int


@dataclass(frozen=True)
class GemmBlock:
    pre_op: Event | None
    quant: Event
    fill: Event | None
    gemm: Event

    @property
    def minimal_sum_us(self) -> float:
        return self.quant.dur + (self.fill.dur if self.fill else 0.0) + self.gemm.dur

    @property
    def minimal_wall_us(self) -> float:
        return self.gemm.end - self.quant.ts

    @property
    def with_pre_sum_us(self) -> float:
        return (self.pre_op.dur if self.pre_op else 0.0) + self.minimal_sum_us

    @property
    def with_pre_wall_us(self) -> float:
        start = self.pre_op.ts if self.pre_op else self.quant.ts
        return self.gemm.end - start


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _event_from_json(raw: dict) -> Event | None:
    name = raw.get("name")
    ts = raw.get("ts")
    dur = raw.get("dur")
    if not isinstance(name, str) or not _is_number(ts) or not _is_number(dur):
        return None
    if dur <= 0:
        return None
    cat = raw.get("cat")
    return Event(name=name, ts=float(ts), dur=float(dur), cat=str(cat or ""))


def _is_kernel(raw: dict, event: Event) -> bool:
    cat = event.cat.lower()
    if "memcpy" in cat or "memset" in cat:
        return False
    if "kernel" in cat and "runtime" not in cat:
        return True
    args = raw.get("args")
    if not isinstance(args, dict):
        return False
    has_device = "device" in args or "Device" in args
    has_stream = "stream" in args or "stream id" in args or "stream_id" in args
    # Torch profiler kernel events sometimes carry device/stream metadata even
    # when the category changes between releases.
    return has_device and has_stream and "cuda_runtime" not in cat


def _is_runtime(event: Event) -> bool:
    cat = event.cat.lower()
    name = event.name.lower()
    if "cuda_runtime" in cat or "hip_runtime" in cat:
        return True
    if "runtime" in cat and ("cuda" in name or "hip" in name):
        return True
    return name.startswith(("cuda", "hip")) and "kernel" not in cat


def find_trace_path(results_dir: Path, mode: str) -> Path:
    candidates = sorted((results_dir / mode).glob("*.pt.trace.json"))
    if not candidates:
        raise FileNotFoundError(f"No *.pt.trace.json files under {results_dir / mode}")

    preferred = [
        path
        for path in candidates
        if path.name.startswith("dp0_") and "async_llm" not in path.name
    ]
    if preferred:
        return preferred[0]

    non_async = [path for path in candidates if "async_llm" not in path.name]
    return non_async[0] if non_async else candidates[0]


def load_mode_trace(results_dir: Path, mode: str) -> ModeTrace:
    path = find_trace_path(results_dir, mode)
    with path.open() as f:
        payload = json.load(f)
    raw_events = payload.get("traceEvents", [])
    if not isinstance(raw_events, list):
        raise ValueError(f"{path} does not contain a traceEvents list")

    kernels: list[Event] = []
    runtime_events: list[Event] = []
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        event = _event_from_json(raw)
        if event is None:
            continue
        if _is_kernel(raw, event):
            kernels.append(event)
        elif _is_runtime(event):
            runtime_events.append(event)

    kernels.sort(key=lambda event: (event.ts, event.end))
    runtime_events.sort(key=lambda event: (event.ts, event.end))
    return ModeTrace(mode, path, raw_events, kernels, runtime_events)


def contamination_counts(events: Iterable[dict]) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    for event in events:
        name = event.get("name")
        if not isinstance(name, str):
            continue
        for pattern in CONTAMINATION_PATTERNS:
            if pattern in name:
                counts[pattern] += 1
                break
    return counts


def longest_dense_burst(kernels: list[Event], gap_us: float) -> list[Event]:
    if not kernels:
        return []

    best: list[Event] = []
    current: list[Event] = [kernels[0]]
    current_end = kernels[0].end

    for event in kernels[1:]:
        if event.ts - current_end <= gap_us:
            current.append(event)
            current_end = max(current_end, event.end)
            continue
        if _burst_wall_us(current) > _burst_wall_us(best):
            best = current
        current = [event]
        current_end = event.end

    if _burst_wall_us(current) > _burst_wall_us(best):
        best = current
    return best


def _burst_wall_us(burst: list[Event]) -> float:
    if not burst:
        return 0.0
    return max(event.end for event in burst) - min(event.ts for event in burst)


def summarize_burst(
    kernels: list[Event],
    *,
    fmoe_substring: str,
    fill_substring: str,
) -> BurstSummary:
    if not kernels:
        return BurstSummary(
            start=math.nan,
            end=math.nan,
            kernels=[],
            fmoe_count=0,
            total_us_per_fmoe=math.nan,
            kernels_per_fmoe=math.nan,
            fill_us_per_fmoe=math.nan,
            fill_count_per_fmoe=math.nan,
            fmoe_durations=[],
            by_kernel_us_per_fmoe={},
        )

    fmoe_events = [event for event in kernels if fmoe_substring in event.name]
    denom = len(fmoe_events)
    by_name: collections.Counter[str] = collections.Counter()
    for event in kernels:
        by_name[event.name] += event.dur

    fill_events = [event for event in kernels if fill_substring in event.name]
    if denom:
        by_kernel = {name: dur / denom for name, dur in by_name.items()}
        total_us_per_fmoe = sum(event.dur for event in kernels) / denom
        kernels_per_fmoe = len(kernels) / denom
        fill_us_per_fmoe = sum(event.dur for event in fill_events) / denom
        fill_count_per_fmoe = len(fill_events) / denom
    else:
        by_kernel = {name: math.nan for name in by_name}
        total_us_per_fmoe = math.nan
        kernels_per_fmoe = math.nan
        fill_us_per_fmoe = math.nan
        fill_count_per_fmoe = math.nan

    return BurstSummary(
        start=min(event.ts for event in kernels),
        end=max(event.end for event in kernels),
        kernels=kernels,
        fmoe_count=denom,
        total_us_per_fmoe=total_us_per_fmoe,
        kernels_per_fmoe=kernels_per_fmoe,
        fill_us_per_fmoe=fill_us_per_fmoe,
        fill_count_per_fmoe=fill_count_per_fmoe,
        fmoe_durations=[event.dur for event in fmoe_events],
        by_kernel_us_per_fmoe=by_kernel,
    )


def events_in_window(events: list[Event], start: float, end: float) -> list[Event]:
    # The traces are small enough that a linear scan is simpler and fast enough.
    return [event for event in events if start <= event.ts < end]


def decode_slices(
    trace: ModeTrace,
    *,
    lm_head_substring: str,
    fmoe_substring: str,
) -> list[SliceSummary]:
    boundaries = [
        event for event in trace.kernels if lm_head_substring in event.name
    ]
    boundaries.sort(key=lambda event: event.ts)
    if len(boundaries) < 2:
        return []

    slices: list[SliceSummary] = []
    for left, right in zip(boundaries, boundaries[1:]):
        start = left.ts
        end = right.ts
        if end <= start:
            continue
        kernels = events_in_window(trace.kernels, start, end)
        runtime_events = events_in_window(trace.runtime_events, start, end)
        gpu_busy_us = sum(event.dur for event in kernels)
        wall_us = end - start
        slices.append(
            SliceSummary(
                wall_us=wall_us,
                gpu_busy_us=gpu_busy_us,
                gpu_idle_us=max(0.0, wall_us - gpu_busy_us),
                cpu_launch_us=sum(event.dur for event in runtime_events),
                n_kernels=len(kernels),
                n_fmoe=sum(1 for event in kernels if fmoe_substring in event.name),
            )
        )
    return slices


def modal_fmoe_slices(slices: list[SliceSummary]) -> list[SliceSummary]:
    if not slices:
        return []
    counts = collections.Counter(slice_.n_fmoe for slice_ in slices)
    modal_n_fmoe, _ = counts.most_common(1)[0]
    return [slice_ for slice_ in slices if slice_.n_fmoe == modal_n_fmoe]


def is_dynamic_group_quant(event: Event) -> bool:
    return "dynamic_per_group_scaled_quant_kernel" in event.name


def is_fill_functor(event: Event) -> bool:
    return "FillFunctor<c10::BFloat16>" in event.name


def is_ck_blockscale_gemm(event: Event) -> bool:
    return "ck::kernel_gemm_xdl_cshuffle_v3" in event.name


def find_gemm_blocks(kernels: list[Event]) -> tuple[list[GemmBlock], list[tuple[int, str]]]:
    """Find quant -> optional FillFunctor -> CK blockscale GEMM motifs.

    ``pre_op`` is the immediately preceding GPU kernel. In this workload that
    is usually the RMSNorm/Silu-like producer feeding the quant kernel, but
    keeping it positional avoids baking in every Inductor-generated Triton name.
    """
    blocks: list[GemmBlock] = []
    misses: list[tuple[int, str]] = []
    for idx, event in enumerate(kernels):
        if not is_dynamic_group_quant(event):
            continue
        next_idx = idx + 1
        fill = None
        if next_idx < len(kernels) and is_fill_functor(kernels[next_idx]):
            fill = kernels[next_idx]
            next_idx += 1
        if next_idx >= len(kernels) or not is_ck_blockscale_gemm(kernels[next_idx]):
            target = kernels[next_idx].name if next_idx < len(kernels) else "<eof>"
            misses.append((idx, target))
            continue
        blocks.append(
            GemmBlock(
                pre_op=kernels[idx - 1] if idx > 0 else None,
                quant=event,
                fill=fill,
                gemm=kernels[next_idx],
            )
        )
    return blocks, misses


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def bootstrap_ci_delta(
    baseline_values: list[float],
    fused_values: list[float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float] | None:
    if not baseline_values or not fused_values or iterations <= 0:
        return None
    rng = random.Random(seed)
    deltas = []
    for _ in range(iterations):
        base_mean = mean(rng.choice(baseline_values) for _ in baseline_values)
        fused_mean = mean(rng.choice(fused_values) for _ in fused_values)
        deltas.append(fused_mean - base_mean)
    return percentile(deltas, 2.5), percentile(deltas, 97.5)


def stats_line(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} mean={mean(values):.3f} "
        f"std={statistics.pstdev(values):.3f} "
        f"min={min(values):.3f} p50={percentile(values, 50):.3f} "
        f"p99={percentile(values, 99):.3f} max={max(values):.3f}"
    )


def format_delta(baseline: float, fused: float) -> str:
    delta = fused - baseline
    pct = delta / baseline * 100.0 if baseline else math.nan
    return f"{baseline:.3f} -> {fused:.3f}  delta={delta:+.3f} ({pct:+.2f}%)"


def mean_attr(items: list[GemmBlock], attr: str) -> float:
    return mean(getattr(item, attr) for item in items)


def mean_component(items: list[GemmBlock], attr: str) -> float:
    values = [getattr(item, attr).dur for item in items if getattr(item, attr) is not None]
    return mean(values)


def print_gemm_block_summary(
    baseline_mode: str,
    fused_mode: str,
    blocks_by_mode: dict[str, list[GemmBlock]],
    misses_by_mode: dict[str, list[tuple[int, str]]],
) -> None:
    baseline = blocks_by_mode[baseline_mode]
    fused = blocks_by_mode[fused_mode]
    print("## Isolated Quant/Zero-Fill/GEMM Blocks")
    print(
        "- motif: immediate pre-op -> dynamic_per_group_scaled_quant_kernel "
        "-> optional FillFunctor<BFloat16> -> ck::kernel_gemm_xdl_cshuffle_v3"
    )
    for mode in (baseline_mode, fused_mode):
        print(
            f"- {mode}: {len(blocks_by_mode[mode])} blocks, "
            f"unmatched quant kernels={len(misses_by_mode[mode])}"
        )

    for label, attr in (
        ("quant(+fill)+gemm wall_us", "minimal_wall_us"),
        ("quant(+fill)+gemm summed_kernel_us", "minimal_sum_us"),
        ("pre-op+quant(+fill)+gemm wall_us", "with_pre_wall_us"),
        ("pre-op+quant(+fill)+gemm summed_kernel_us", "with_pre_sum_us"),
    ):
        base = mean_attr(baseline, attr)
        fuse = mean_attr(fused, attr)
        print(f"- {label}: {format_delta(base, fuse)}")

    component_attrs = (
        ("pre-op", "pre_op"),
        ("quant", "quant"),
        ("fill", "fill"),
        ("gemm", "gemm"),
    )
    for label, attr in component_attrs:
        base = mean_component(baseline, attr)
        fuse = mean_component(fused, attr)
        if math.isnan(fuse):
            print(f"- {label} component: {base:.3f} -> 0.000  delta={-base:+.3f}")
        else:
            print(f"- {label} component: {format_delta(base, fuse)}")
    print()


def print_contamination(traces: dict[str, ModeTrace]) -> None:
    print("## Contamination Check")
    for mode, trace in traces.items():
        counts = contamination_counts(trace.events)
        if counts:
            summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        else:
            summary = "CLEAN"
        print(f"- {mode}: {summary}")
    print()


def print_burst_summary(
    baseline_mode: str,
    fused_mode: str,
    bursts: dict[str, BurstSummary],
    *,
    top_k: int,
) -> None:
    baseline = bursts[baseline_mode]
    fused = bursts[fused_mode]
    print("## Dense-Burst Per-FMOE Summary")
    print(f"- {baseline_mode}: {baseline.fmoe_count} fmoe kernels")
    print(f"- {fused_mode}: {fused.fmoe_count} fmoe kernels")
    print(
        "- total us/fmoe: "
        f"{format_delta(baseline.total_us_per_fmoe, fused.total_us_per_fmoe)}"
    )
    print(
        "- kernels/fmoe: "
        f"{format_delta(baseline.kernels_per_fmoe, fused.kernels_per_fmoe)}"
    )
    print(
        "- FillFunctor us/fmoe: "
        f"{format_delta(baseline.fill_us_per_fmoe, fused.fill_us_per_fmoe)}"
    )
    print(
        "- FillFunctor count/fmoe: "
        f"{format_delta(baseline.fill_count_per_fmoe, fused.fill_count_per_fmoe)}"
    )
    print()

    names = set(baseline.by_kernel_us_per_fmoe) | set(fused.by_kernel_us_per_fmoe)
    deltas = []
    for name in names:
        base = baseline.by_kernel_us_per_fmoe.get(name, 0.0)
        fuse = fused.by_kernel_us_per_fmoe.get(name, 0.0)
        deltas.append((abs(fuse - base), fuse - base, base, fuse, name))
    print(f"## Top {top_k} Kernel Deltas (us/fmoe)")
    for _, delta, base, fuse, name in sorted(deltas, reverse=True)[:top_k]:
        print(f"- {delta:+8.3f}  {base:8.3f} -> {fuse:8.3f}  {name}")
    print()


def print_fmoe_distribution(bursts: dict[str, BurstSummary]) -> None:
    print("## FMOE Kernel Duration Distribution")
    for mode, burst in bursts.items():
        print(f"- {mode}: {stats_line(burst.fmoe_durations)}")
    print()


def print_decode_summary(
    baseline_mode: str,
    fused_mode: str,
    slices_by_mode: dict[str, list[SliceSummary]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    baseline = modal_fmoe_slices(slices_by_mode[baseline_mode])
    fused = modal_fmoe_slices(slices_by_mode[fused_mode])
    if not baseline or not fused:
        print("## Decode-Slice Summary")
        print("- skipped: lm-head boundaries were not found in both traces")
        print()
        return

    print("## Decode-Slice Summary (Modal FMOE Count)")
    print(f"- {baseline_mode}: {len(baseline)} slices, modal n_fmoe={baseline[0].n_fmoe}")
    print(f"- {fused_mode}: {len(fused)} slices, modal n_fmoe={fused[0].n_fmoe}")

    metrics = (
        ("wall_us", lambda slice_: slice_.wall_us),
        ("gpu_busy_us", lambda slice_: slice_.gpu_busy_us),
        ("gpu_idle_us", lambda slice_: slice_.gpu_idle_us),
        ("cpu_launch_us", lambda slice_: slice_.cpu_launch_us),
        ("n_kernels", lambda slice_: float(slice_.n_kernels)),
    )
    for name, getter in metrics:
        base_values = [getter(slice_) for slice_ in baseline]
        fused_values = [getter(slice_) for slice_ in fused]
        base_mean = mean(base_values)
        fused_mean = mean(fused_values)
        ci = bootstrap_ci_delta(
            base_values,
            fused_values,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        ci_text = f" CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"- {name}: {format_delta(base_mean, fused_mean)}{ci_text}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing splitk/ and splitk_fused/ trace subdirs.",
    )
    parser.add_argument("--baseline-mode", default="splitk")
    parser.add_argument("--fused-mode", default="splitk_fused")
    parser.add_argument("--fmoe-substring", default="aiter::fmoe")
    parser.add_argument("--fill-substring", default="FillFunctor")
    parser.add_argument(
        "--lm-head-substring",
        default="aiter::wv_splitk_small_fp16_bf16_kernel",
        help="Kernel substring used to delimit decode steps.",
    )
    parser.add_argument("--burst-gap-us", type=float, default=200.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = (args.baseline_mode, args.fused_mode)
    traces = {mode: load_mode_trace(args.results_dir, mode) for mode in modes}

    print("# Blockscale SplitK Zero-Init Long-Trace Analysis")
    print(f"results_dir: {args.results_dir}")
    for mode, trace in traces.items():
        print(f"{mode}: {trace.path}")
        print(
            f"  events={len(trace.events)} kernels={len(trace.kernels)} "
            f"runtime_events={len(trace.runtime_events)}"
        )
    print()

    print_contamination(traces)

    bursts = {}
    for mode, trace in traces.items():
        burst = longest_dense_burst(trace.kernels, args.burst_gap_us)
        bursts[mode] = summarize_burst(
            burst,
            fmoe_substring=args.fmoe_substring,
            fill_substring=args.fill_substring,
        )
    print_burst_summary(args.baseline_mode, args.fused_mode, bursts, top_k=args.top_k)
    print_fmoe_distribution(bursts)

    blocks_by_mode: dict[str, list[GemmBlock]] = {}
    misses_by_mode: dict[str, list[tuple[int, str]]] = {}
    for mode, trace in traces.items():
        blocks, misses = find_gemm_blocks(trace.kernels)
        blocks_by_mode[mode] = blocks
        misses_by_mode[mode] = misses
    print_gemm_block_summary(
        args.baseline_mode,
        args.fused_mode,
        blocks_by_mode,
        misses_by_mode,
    )

    slices_by_mode = {
        mode: decode_slices(
            trace,
            lm_head_substring=args.lm_head_substring,
            fmoe_substring=args.fmoe_substring,
        )
        for mode, trace in traces.items()
    }
    print_decode_summary(
        args.baseline_mode,
        args.fused_mode,
        slices_by_mode,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
