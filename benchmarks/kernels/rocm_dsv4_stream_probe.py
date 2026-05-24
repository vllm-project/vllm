# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe ROCm stream ownership/overlap for DeepSeek-V4 CSA-like kernels.

This is intentionally standalone: it does not instantiate the model or require
serving metadata.  It answers whether representative kernels used around DSV4
CSA decode honor the current HIP stream and whether independent streams overlap
when forced outside vLLM's graph/runtime path.

Repro commands used on ROCm:

  # Control: stream scheduling outside torch.compile, then graph replay.
  HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER=1 \
  PYTHONPATH=/path/to/vllm \
  rocprofv3 --runtime-trace --group-by-queue \
    --output-directory /tmp/vllm_rocm_dsv4_rocprof_graph \
    --output-file graph --output-format json csv -- \
  .venv/bin/python benchmarks/kernels/rocm_dsv4_stream_probe.py \
    --scenario aiter_vs_bf16_mm_decode \
    --repeats 5 --profile-repeats 0 --warmup 1 --mode graph

  # Repro: stream scheduling inside torch.compile, then graph replay.
  HIP_VISIBLE_DEVICES=0 VLLM_ROCM_USE_AITER=1 \
  PYTHONPATH=/path/to/vllm \
  rocprofv3 --runtime-trace --group-by-queue \
    --output-directory /tmp/vllm_rocm_dsv4_rocprof_compile_pair_graph \
    --output-file compile_pair_graph --output-format json csv -- \
  .venv/bin/python benchmarks/kernels/rocm_dsv4_stream_probe.py \
    --scenario aiter_vs_bf16_mm_decode \
    --repeats 5 --profile-repeats 0 --warmup 1 --mode compile_pair_graph

Finding from the repro:
  - graph mode preserves separate ROCm queues for the representative AITER and
    BF16 GEMM branches, but the tested decode-sized kernels did not produce
    useful timestamp overlap, which points to kernel resource contention rather
    than Python stream ownership as the first bottleneck;
  - compile_pair_graph collapses the same representative kernels onto ROCm
    stream 0 / queue 1 during graph replay, exposing the non-overlap failure
    mode seen in vLLM-like compiled scheduling.
"""

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

# Registers vLLM ROCm AITER custom ops.
import vllm._aiter_ops  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops

KernelFn = Callable[[], Any]


def _fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fnuz", torch.int8)


def _make_bf16_mm(
    m: int,
    n: int,
    k: int,
    device: torch.device,
) -> KernelFn:
    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        torch.mm(a, b, out=out)
        return out

    return run


def _make_aiter_fp8_block_gemm(
    m: int,
    n: int,
    k: int,
    device: torch.device,
) -> KernelFn:
    dtype = _fp8_dtype()
    a = torch.empty((m, k), device=device, dtype=dtype)
    b = torch.empty((n, k), device=device, dtype=dtype)
    a_scales = torch.ones((m, (k + 127) // 128), device=device, dtype=torch.float32)
    b_scales = torch.ones((n, (k + 127) // 128), device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        out = rocm_aiter_ops.gemm_a8w8_blockscale(
            a,
            b,
            a_scales,
            b_scales,
            [1, 128],
            output_dtype=torch.bfloat16,
        )
        return out

    return run


def _make_topk(
    rows: int,
    cols: int,
    topk: int,
    device: torch.device,
) -> KernelFn:
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    values = torch.empty((rows, topk), device=device, dtype=x.dtype)
    indices = torch.empty((rows, topk), device=device, dtype=torch.long)

    def run() -> torch.Tensor:
        torch.topk(x, topk, dim=-1, out=(values, indices))
        return values

    return run


def _event_time_ms(
    fn0: KernelFn,
    fn1: KernelFn,
    concurrent: bool,
    iterations: int,
) -> float:
    torch.cuda.synchronize()
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current = torch.cuda.current_stream()

    start.record(current)
    for _ in range(iterations):
        _run_pair(fn0, fn1, concurrent, stream0, stream1)
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _run_pair(
    fn0: KernelFn,
    fn1: KernelFn,
    concurrent: bool,
    stream0: torch.cuda.Stream,
    stream1: torch.cuda.Stream,
) -> tuple[Any, Any]:
    current = torch.cuda.current_stream()
    if concurrent:
        stream0.wait_stream(current)
        stream1.wait_stream(current)
        with torch.cuda.stream(stream0):
            result0 = fn0()
        with torch.cuda.stream(stream1):
            result1 = fn1()
        current.wait_stream(stream0)
        current.wait_stream(stream1)
        return result0, result1
    else:
        result0 = fn0()
        result1 = fn1()
        return result0, result1


def _capture_pair_graph(
    fn0: KernelFn,
    fn1: KernelFn,
    concurrent: bool,
) -> tuple[torch.cuda.CUDAGraph, tuple[torch.cuda.Stream, ...]]:
    torch.cuda.synchronize()
    # Run once outside capture so the caching allocator and any JIT/autotune
    # setup do not become capture-time side effects.
    warm_stream0 = torch.cuda.Stream()
    warm_stream1 = torch.cuda.Stream()
    _run_pair(fn0, fn1, concurrent, warm_stream0, warm_stream1)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    capture_stream.wait_stream(current)
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        _run_pair(fn0, fn1, concurrent, stream0, stream1)
    current.wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph, (capture_stream, stream0, stream1)


def _graph_time_ms(
    fn0: KernelFn,
    fn1: KernelFn,
    concurrent: bool,
    iterations: int,
) -> float:
    graph, _streams = _capture_pair_graph(fn0, fn1, concurrent)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current = torch.cuda.current_stream()
    start.record(current)
    for _ in range(iterations):
        graph.replay()
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _capture_runner_graph(
    runner: KernelFn,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    torch.cuda.synchronize()
    runner()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    capture_stream.wait_stream(current)
    with torch.cuda.stream(capture_stream), torch.cuda.graph(graph):
        runner()
    current.wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph, capture_stream


def _graph_runner_time_ms(runner: KernelFn, iterations: int) -> float:
    graph, _capture_stream = _capture_runner_graph(runner)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current = torch.cuda.current_stream()
    start.record(current)
    for _ in range(iterations):
        graph.replay()
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _compile_kernel(fn: KernelFn) -> KernelFn:
    compiled = torch.compile(
        fn,
        backend="inductor",
        dynamic=False,
        fullgraph=False,
    )
    compiled()
    torch.cuda.synchronize()
    return compiled


def _call_time_ms(fn: KernelFn, iterations: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current = torch.cuda.current_stream()
    start.record(current)
    for _ in range(iterations):
        fn()
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _compile_pair_runner(
    fn0: KernelFn,
    fn1: KernelFn,
    concurrent: bool,
    disable_scheduler_compile: bool = False,
) -> KernelFn:
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()

    def pair_run() -> tuple[Any, Any]:
        return _run_pair(fn0, fn1, concurrent, stream0, stream1)

    if disable_scheduler_compile:
        pair_run = torch.compiler.disable(pair_run)
    return _compile_kernel(pair_run)


def _profile_runner_trace(
    runner: KernelFn,
    profile_dir: Path,
    scenario: str,
    mode: str,
    iterations: int,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iterations):
            runner()
        torch.cuda.synchronize()

    trace = profile_dir / f"{scenario}.{mode}.trace.json"
    prof.export_chrome_trace(str(trace))
    return trace


def _profile_runner_graph_trace(
    runner: KernelFn,
    profile_dir: Path,
    scenario: str,
    mode: str,
    iterations: int,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    graph, _capture_stream = _capture_runner_graph(runner)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()

    trace = profile_dir / f"{scenario}.{mode}.trace.json"
    prof.export_chrome_trace(str(trace))
    return trace


def _profile_trace(
    fn0: KernelFn,
    fn1: KernelFn,
    profile_dir: Path,
    scenario: str,
    mode: str,
    use_graph: bool,
    iterations: int,
) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    graph = None
    if use_graph:
        graph, _streams = _capture_pair_graph(fn0, fn1, concurrent=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        if graph is not None:
            for _ in range(iterations):
                graph.replay()
        else:
            for _ in range(iterations):
                _run_pair(fn0, fn1, True, stream0, stream1)
                current.wait_stream(stream0)
                current.wait_stream(stream1)
        torch.cuda.synchronize()

    trace = profile_dir / f"{scenario}.{mode}.trace.json"
    prof.export_chrome_trace(str(trace))
    return trace


def _summarize_trace(trace: Path) -> dict[str, object]:
    with trace.open() as f:
        data = json.load(f)
    stream_counts: Counter[str] = Counter()
    stream_kernel_counts: Counter[str] = Counter()
    stream_durations_us: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    kernel_names: Counter[tuple[str, str]] = Counter()

    for event in data.get("traceEvents", []):
        name = event.get("name", "")
        if name in ("hipEventRecord", "hipStreamWaitEvent"):
            event_counts[name] += 1
        args = event.get("args") or {}
        stream = args.get("stream")
        if stream is None:
            continue
        stream = str(stream)
        stream_counts[stream] += 1
        if event.get("cat") in ("kernel", "gpu_memcpy"):
            stream_kernel_counts[stream] += 1
            stream_durations_us[stream] += event.get("dur", 0.0) or 0.0
            kernel_names[(stream, name[:96])] += 1

    return {
        "streams": dict(stream_counts),
        "stream_kernel_counts": dict(stream_kernel_counts),
        "stream_durations_us": dict(stream_durations_us),
        "event_counts": dict(event_counts),
        "top_kernel_names": [
            {"stream": stream, "name": name, "count": count}
            for (stream, name), count in kernel_names.most_common(12)
        ],
    }


def _scenario(
    name: str,
    device: torch.device,
) -> tuple[KernelFn, KernelFn, KernelFn, KernelFn]:
    if name == "bf16_mm_pair":
        return (
            _make_bf16_mm(256, 2048, 7168, device),
            _make_bf16_mm(256, 768, 7168, device),
            _make_bf16_mm(256, 2048, 7168, device),
            _make_bf16_mm(256, 768, 7168, device),
        )
    if name == "aiter_vs_bf16_mm_decode":
        return (
            _make_aiter_fp8_block_gemm(4, 2048, 7168, device),
            _make_bf16_mm(4, 768, 7168, device),
            _make_aiter_fp8_block_gemm(4, 2048, 7168, device),
            _make_bf16_mm(4, 768, 7168, device),
        )
    if name == "aiter_pair_decode":
        return (
            _make_aiter_fp8_block_gemm(4, 2048, 7168, device),
            _make_aiter_fp8_block_gemm(4, 768, 7168, device),
            _make_aiter_fp8_block_gemm(4, 2048, 7168, device),
            _make_aiter_fp8_block_gemm(4, 768, 7168, device),
        )
    if name == "topk_vs_bf16_mm":
        return (
            _make_topk(64, 8192, 128, device),
            _make_bf16_mm(64, 768, 7168, device),
            _make_topk(64, 8192, 128, device),
            _make_bf16_mm(64, 768, 7168, device),
        )
    raise ValueError(f"unknown scenario: {name}")


def _mode_enabled(selected: str, mode: str) -> bool:
    if selected == "all":
        return True
    if selected == "both":
        return mode in ("eager", "graph")
    return selected == mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[
            "bf16_mm_pair",
            "aiter_vs_bf16_mm_decode",
            "aiter_pair_decode",
            "topk_vs_bf16_mm",
        ],
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--profile-compile-pair",
        action="store_true",
        help=(
            "Also profile the torch.compile wrapper that includes Python "
            "stream scheduling. ROCTracer may be unstable for this mode on "
            "some ROCm builds, so timing is collected by default without a "
            "trace."
        ),
    )
    parser.add_argument(
        "--profile-compile",
        action="store_true",
        help=(
            "Profile compiled branch modes when running --mode all. Explicit "
            "--mode compile and --mode compile_graph runs are profiled by "
            "default when --profile-repeats is positive."
        ),
    )
    parser.add_argument(
        "--profile-aggregate",
        action="store_true",
        help=(
            "Profile aggregate modes such as --mode both or --mode all. On "
            "some ROCm stacks, multiple profiler sessions in one process can "
            "crash during ROCTracer cleanup, so aggregate modes collect timing "
            "only unless this is set."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=[
            "eager",
            "graph",
            "compile",
            "compile_graph",
            "compile_pair",
            "compile_pair_graph",
            "disabled_compile_pair_graph",
            "both",
            "all",
        ],
        default="both",
        help=(
            "Measure eager forced streams, graph replay, torch.compile branch "
            "functions, compiled branches under graph replay, a compiled whole "
            "pair scheduler, the compiled scheduler under graph replay, a "
            "torch.compiler.disable-protected scheduler under graph replay, or "
            "all modes."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/tmp/vllm_rocm_dsv4_stream_probe"),
    )
    args = parser.parse_args()

    scenarios = args.scenario or [
        "bf16_mm_pair",
        "aiter_vs_bf16_mm_decode",
        "aiter_pair_decode",
        "topk_vs_bf16_mm",
    ]
    device = torch.device("cuda")
    print(f"pid={os.getpid()} device={torch.cuda.get_device_name(0)!r}")

    for name in scenarios:
        fn0, fn1, profile_fn0, profile_fn1 = _scenario(name, device)
        for _ in range(args.warmup):
            _event_time_ms(fn0, fn1, concurrent=True, iterations=1)
        seq_ms = _event_time_ms(
            fn0, fn1, concurrent=False, iterations=args.repeats
        )
        conc_ms = _event_time_ms(
            fn0, fn1, concurrent=True, iterations=args.repeats
        )
        overlap_pct = 100.0 * (1.0 - conc_ms / seq_ms) if seq_ms else 0.0
        results: dict[str, object] = {"scenario": name}
        if _mode_enabled(args.mode, "eager"):
            should_profile_eager = args.profile_repeats > 0 and (
                args.mode == "eager" or args.profile_aggregate
            )
            eager_result = {
                "sequential_ms": seq_ms,
                "concurrent_ms": conc_ms,
                "overlap_pct": overlap_pct,
            }
            if should_profile_eager:
                trace = _profile_trace(
                    profile_fn0,
                    profile_fn1,
                    args.profile_dir,
                    name,
                    "eager",
                    use_graph=False,
                    iterations=args.profile_repeats,
                )
                eager_result.update({
                    "trace": str(trace),
                    **_summarize_trace(trace),
                })
            elif args.profile_repeats > 0:
                eager_result["profile_note"] = (
                    "eager trace skipped in aggregate mode; use --mode eager "
                    "or pass --profile-aggregate"
                )
            results["eager"] = eager_result
        if _mode_enabled(args.mode, "graph"):
            try:
                should_profile_graph = args.profile_repeats > 0 and (
                    args.mode == "graph" or args.profile_aggregate
                )
                graph_seq_ms = _graph_time_ms(
                    fn0,
                    fn1,
                    concurrent=False,
                    iterations=args.repeats,
                )
                graph_conc_ms = _graph_time_ms(
                    fn0,
                    fn1,
                    concurrent=True,
                    iterations=args.repeats,
                )
                graph_overlap_pct = (
                    100.0 * (1.0 - graph_conc_ms / graph_seq_ms)
                    if graph_seq_ms
                    else 0.0
                )
                graph_result = {
                    "sequential_ms": graph_seq_ms,
                    "concurrent_ms": graph_conc_ms,
                    "overlap_pct": graph_overlap_pct,
                }
                if should_profile_graph:
                    graph_trace = _profile_trace(
                        profile_fn0,
                        profile_fn1,
                        args.profile_dir,
                        name,
                        "graph",
                        use_graph=True,
                        iterations=args.profile_repeats,
                    )
                    graph_result.update({
                        "trace": str(graph_trace),
                        **_summarize_trace(graph_trace),
                    })
                elif args.profile_repeats > 0:
                    graph_result["profile_note"] = (
                        "graph trace skipped in aggregate mode; use --mode "
                        "graph or pass --profile-aggregate"
                    )
                results["graph"] = graph_result
            except Exception as exc:
                results["graph"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if _mode_enabled(args.mode, "compile"):
            try:
                should_profile_compile = args.profile_repeats > 0 and (
                    args.mode == "compile"
                    or (args.profile_aggregate and args.profile_compile)
                )
                compiled_fn0 = _compile_kernel(fn0)
                compiled_fn1 = _compile_kernel(fn1)
                compiled_profile_fn0 = _compile_kernel(profile_fn0)
                compiled_profile_fn1 = _compile_kernel(profile_fn1)
                compile_seq_ms = _event_time_ms(
                    compiled_fn0,
                    compiled_fn1,
                    concurrent=False,
                    iterations=args.repeats,
                )
                compile_conc_ms = _event_time_ms(
                    compiled_fn0,
                    compiled_fn1,
                    concurrent=True,
                    iterations=args.repeats,
                )
                compile_overlap_pct = (
                    100.0 * (1.0 - compile_conc_ms / compile_seq_ms)
                    if compile_seq_ms
                    else 0.0
                )
                compile_result = {
                    "sequential_ms": compile_seq_ms,
                    "concurrent_ms": compile_conc_ms,
                    "overlap_pct": compile_overlap_pct,
                }
                if should_profile_compile:
                    compile_trace = _profile_trace(
                        compiled_profile_fn0,
                        compiled_profile_fn1,
                        args.profile_dir,
                        name,
                        "compile",
                        use_graph=False,
                        iterations=args.profile_repeats,
                    )
                    compile_result.update({
                        "trace": str(compile_trace),
                        **_summarize_trace(compile_trace),
                    })
                else:
                    compile_result["profile_note"] = (
                        "compiled branch trace skipped in aggregate mode; "
                        "use --mode compile or pass --profile-compile"
                    )
                results["compile"] = compile_result
            except Exception as exc:
                results["compile"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if _mode_enabled(args.mode, "compile_graph"):
            try:
                should_profile_compile_graph = args.profile_repeats > 0 and (
                    args.mode == "compile_graph"
                    or (args.profile_aggregate and args.profile_compile)
                )
                compiled_fn0 = _compile_kernel(fn0)
                compiled_fn1 = _compile_kernel(fn1)
                compiled_profile_fn0 = _compile_kernel(profile_fn0)
                compiled_profile_fn1 = _compile_kernel(profile_fn1)
                compile_graph_seq_ms = _graph_time_ms(
                    compiled_fn0,
                    compiled_fn1,
                    concurrent=False,
                    iterations=args.repeats,
                )
                compile_graph_conc_ms = _graph_time_ms(
                    compiled_fn0,
                    compiled_fn1,
                    concurrent=True,
                    iterations=args.repeats,
                )
                compile_graph_overlap_pct = (
                    100.0 * (1.0 - compile_graph_conc_ms / compile_graph_seq_ms)
                    if compile_graph_seq_ms
                    else 0.0
                )
                compile_graph_result = {
                    "sequential_ms": compile_graph_seq_ms,
                    "concurrent_ms": compile_graph_conc_ms,
                    "overlap_pct": compile_graph_overlap_pct,
                }
                if should_profile_compile_graph:
                    compile_graph_trace = _profile_trace(
                        compiled_profile_fn0,
                        compiled_profile_fn1,
                        args.profile_dir,
                        name,
                        "compile_graph",
                        use_graph=True,
                        iterations=args.profile_repeats,
                    )
                    compile_graph_result.update({
                        "trace": str(compile_graph_trace),
                        **_summarize_trace(compile_graph_trace),
                    })
                else:
                    compile_graph_result["profile_note"] = (
                        "compiled graph trace skipped in aggregate mode; "
                        "use --mode compile_graph or pass --profile-compile"
                    )
                results["compile_graph"] = compile_graph_result
            except Exception as exc:
                results["compile_graph"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if _mode_enabled(args.mode, "compile_pair"):
            try:
                compiled_seq_pair = _compile_pair_runner(fn0, fn1, concurrent=False)
                compiled_conc_pair = _compile_pair_runner(fn0, fn1, concurrent=True)
                compile_pair_seq_ms = _call_time_ms(
                    compiled_seq_pair, iterations=args.repeats
                )
                compile_pair_conc_ms = _call_time_ms(
                    compiled_conc_pair, iterations=args.repeats
                )
                compile_pair_overlap_pct = (
                    100.0 * (1.0 - compile_pair_conc_ms / compile_pair_seq_ms)
                    if compile_pair_seq_ms
                    else 0.0
                )
                compile_pair_result = {
                    "sequential_ms": compile_pair_seq_ms,
                    "concurrent_ms": compile_pair_conc_ms,
                    "overlap_pct": compile_pair_overlap_pct,
                }
                if args.profile_compile_pair and args.profile_repeats > 0:
                    compiled_profile_pair = _compile_pair_runner(
                        profile_fn0,
                        profile_fn1,
                        concurrent=True,
                    )
                    compile_pair_trace = _profile_runner_trace(
                        compiled_profile_pair,
                        args.profile_dir,
                        name,
                        "compile_pair",
                        iterations=args.profile_repeats,
                    )
                    compile_pair_result.update({
                        "trace": str(compile_pair_trace),
                        **_summarize_trace(compile_pair_trace),
                    })
                else:
                    compile_pair_result["profile_note"] = (
                        "compile_pair trace skipped; pass --profile-compile-pair "
                        "to profile the compiled Python stream scheduler"
                    )
                results["compile_pair"] = compile_pair_result
            except Exception as exc:
                results["compile_pair"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if _mode_enabled(args.mode, "compile_pair_graph"):
            try:
                compiled_seq_pair = _compile_pair_runner(fn0, fn1, concurrent=False)
                compiled_conc_pair = _compile_pair_runner(fn0, fn1, concurrent=True)
                compile_pair_graph_seq_ms = _graph_runner_time_ms(
                    compiled_seq_pair, iterations=args.repeats
                )
                compile_pair_graph_conc_ms = _graph_runner_time_ms(
                    compiled_conc_pair, iterations=args.repeats
                )
                compile_pair_graph_overlap_pct = (
                    100.0
                    * (
                        1.0
                        - compile_pair_graph_conc_ms / compile_pair_graph_seq_ms
                    )
                    if compile_pair_graph_seq_ms
                    else 0.0
                )
                compile_pair_graph_result = {
                    "sequential_ms": compile_pair_graph_seq_ms,
                    "concurrent_ms": compile_pair_graph_conc_ms,
                    "overlap_pct": compile_pair_graph_overlap_pct,
                }
                if args.profile_repeats > 0 and (
                    args.mode == "compile_pair_graph" or args.profile_aggregate
                ):
                    compiled_profile_pair = _compile_pair_runner(
                        profile_fn0,
                        profile_fn1,
                        concurrent=True,
                    )
                    compile_pair_graph_trace = _profile_runner_graph_trace(
                        compiled_profile_pair,
                        args.profile_dir,
                        name,
                        "compile_pair_graph",
                        iterations=args.profile_repeats,
                    )
                    compile_pair_graph_result.update({
                        "trace": str(compile_pair_graph_trace),
                        **_summarize_trace(compile_pair_graph_trace),
                    })
                elif args.profile_repeats > 0:
                    compile_pair_graph_result["profile_note"] = (
                        "compile_pair_graph trace skipped in aggregate mode; "
                        "use --mode compile_pair_graph or pass "
                        "--profile-aggregate"
                    )
                results["compile_pair_graph"] = compile_pair_graph_result
            except Exception as exc:
                results["compile_pair_graph"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if _mode_enabled(args.mode, "disabled_compile_pair_graph"):
            try:
                disabled_seq_pair = _compile_pair_runner(
                    fn0,
                    fn1,
                    concurrent=False,
                    disable_scheduler_compile=True,
                )
                disabled_conc_pair = _compile_pair_runner(
                    fn0,
                    fn1,
                    concurrent=True,
                    disable_scheduler_compile=True,
                )
                disabled_pair_graph_seq_ms = _graph_runner_time_ms(
                    disabled_seq_pair, iterations=args.repeats
                )
                disabled_pair_graph_conc_ms = _graph_runner_time_ms(
                    disabled_conc_pair, iterations=args.repeats
                )
                disabled_pair_graph_overlap_pct = (
                    100.0
                    * (
                        1.0
                        - disabled_pair_graph_conc_ms / disabled_pair_graph_seq_ms
                    )
                    if disabled_pair_graph_seq_ms
                    else 0.0
                )
                disabled_pair_graph_result = {
                    "sequential_ms": disabled_pair_graph_seq_ms,
                    "concurrent_ms": disabled_pair_graph_conc_ms,
                    "overlap_pct": disabled_pair_graph_overlap_pct,
                }
                if args.profile_repeats > 0 and (
                    args.mode == "disabled_compile_pair_graph"
                    or args.profile_aggregate
                ):
                    disabled_profile_pair = _compile_pair_runner(
                        profile_fn0,
                        profile_fn1,
                        concurrent=True,
                        disable_scheduler_compile=True,
                    )
                    disabled_pair_graph_trace = _profile_runner_graph_trace(
                        disabled_profile_pair,
                        args.profile_dir,
                        name,
                        "disabled_compile_pair_graph",
                        iterations=args.profile_repeats,
                    )
                    disabled_pair_graph_result.update({
                        "trace": str(disabled_pair_graph_trace),
                        **_summarize_trace(disabled_pair_graph_trace),
                    })
                elif args.profile_repeats > 0:
                    disabled_pair_graph_result["profile_note"] = (
                        "disabled_compile_pair_graph trace skipped in aggregate "
                        "mode; use --mode disabled_compile_pair_graph or pass "
                        "--profile-aggregate"
                    )
                results["disabled_compile_pair_graph"] = disabled_pair_graph_result
            except Exception as exc:
                results["disabled_compile_pair_graph"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
