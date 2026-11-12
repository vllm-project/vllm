# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU kernel profiles of `vllm bench forward` cases.

With `--profile-dir`, each case's graph is replayed a few more times under
`torch.profiler` after timing. Each rank writes a Perfetto trace and a kernel
summary (`<case>.rank<r>.kernels.json`): per-kernel time per forward, sorted.

Summarize one file or compare two (e.g. before/after a change):
`python -m vllm.benchmarks.forward.profile A.kernels.json [B.kernels.json]`.

The same replays run between `cudaProfilerStart`/`Stop` in an NVTX range per
case (with or without `--profile-dir`), so Nsight Systems can record just them:
`nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat
--cuda-graph-trace=node vllm bench forward ...`.
"""

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from typing import Any

import regex as re
import torch

from vllm.platforms import current_platform
from vllm.profiler.wrapper import get_default_torch_profiler_activities

_GPU_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
# Kernels named after a generic launcher are identified by their first
# template argument instead.
_GENERIC = re.compile(r"(^|::)(device_kernel|Kernel|kernel|KernelWrapper)$")
# CuTe DSL kernel names embed per-shape layouts after these markers.
_CUTE_DSL_SUFFIX = re.compile(r"_object_at_|_tensorptr")


def _match_back(text: str, end: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket opening the one closing at `end`, skipping
    brackets of the other kind nested inside parentheses."""
    depth, parens = 0, 0
    for i in range(end, -1, -1):
        ch = text[i]
        if open_ch == "<" and ch in "()":
            parens += 1 if ch == ")" else -1
            continue
        if parens:
            continue
        if ch == close_ch:
            depth += 1
        elif ch == open_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def short_name(name: str) -> str:
    """`void ns::foo<T, 4>(args)` -> `ns::foo`: return type, template and
    function arguments dropped. Generic launchers (e.g.
    `cutlass::device_kernel<Op<...>>`) keep their first template argument."""
    if name.startswith("kernel_cutlass_"):
        return _CUTE_DSL_SUFFIX.split(name, maxsplit=1)[0]
    head = name.strip()
    if head.endswith(")"):
        head = head[: max(_match_back(head, len(head) - 1, "(", ")"), 0)].rstrip()
    template = ""
    if head.endswith(">"):
        start = _match_back(head, len(head) - 1, "<", ">")
        if start > 0:
            head, template = head[:start], head[start + 1 : -1]
    base = head.split()[-1] if head.split() else name
    if _GENERIC.search(base) and template:
        arg = re.split(r"[<>(,]", template, maxsplit=1)[0].strip()
        if arg:
            return f"{base}<{arg.removeprefix('void ')}>"
    return base


def summarize(events: list[dict], replays: int, skip: int) -> dict[str, Any]:
    """Per-forward kernel times from the GPU activities of `replays` replays.

    Each replay first runs `skip` activities that aren't part of the timed
    forward (the KV state restore and the rank rendezvous); they are reported
    separately as `pre_forward_us`.
    """
    acts = sorted(
        (e for e in events if e.get("cat") in _GPU_CATEGORIES and "dur" in e),
        key=lambda e: e["ts"],
    )
    if not acts:
        raise ValueError(
            "No GPU activity recorded. Another profiler (e.g. nsys) may hold "
            "CUPTI; don't combine --profile-dir with it."
        )
    if len(acts) % replays:
        raise ValueError(
            f"{len(acts)} GPU activities don't split into {replays} replays"
        )
    per_replay = len(acts) // replays
    if per_replay <= skip:
        raise ValueError(f"{per_replay} activities per replay, {skip} to skip")
    kernels: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_us": 0.0, "calls": 0, "example": ""}
    )
    spans, busy, pre = [], [], []
    for r in range(replays):
        chunk = acts[r * per_replay : (r + 1) * per_replay]
        pre.append(sum(e["dur"] for e in chunk[:skip]))
        forward = chunk[skip:]
        spans.append(
            max(e["ts"] + e["dur"] for e in forward) - min(e["ts"] for e in forward)
        )
        busy.append(sum(e["dur"] for e in forward))
        for e in forward:
            name = short_name(e["name"]) if e["cat"] == "kernel" else e["name"]
            k = kernels[name]
            k["total_us"] += e["dur"]
            k["calls"] += 1
            k["example"] = k["example"] or e["name"][:300]
    total = sum(k["total_us"] for k in kernels.values())
    rows = [
        {
            "name": name,
            "us": round(k["total_us"] / replays, 2),
            "pct": round(100 * k["total_us"] / total, 2),
            "calls": k["calls"] // replays,
            "example": k["example"],
        }
        for name, k in kernels.items()
    ]
    rows.sort(key=lambda row: -row["us"])
    return {
        "replays": replays,
        # First to last forward activity; busy can exceed it when streams overlap.
        "span_us": round(statistics.median(spans), 2),
        "busy_us": round(statistics.median(busy), 2),
        "pre_forward_us": round(statistics.median(pre), 2),
        "activities": per_replay - skip,
        "kernels": rows,
    }


def profile_graph(
    graph: torch.cuda.CUDAGraph,
    name: str,
    prefix: str | None,
    skip: int,
    replays: int = 3,
) -> dict[str, Any] | None:
    """Replay `replays` times between `cudaProfilerStart`/`Stop`, in an NVTX
    range named after the case, for `nsys --capture-range=cudaProfilerApi`.

    With `prefix`, also trace them with `torch.profiler` and write
    `prefix.trace.json.gz` and `prefix.kernels.json`. Every rank must call
    this (replays may contain collectives).
    """

    def run() -> None:
        # `cudaProfilerStart`/`Stop` and NVTX ranges are CUDA-only (for
        # Nsight Systems); other accelerators just replay.
        if current_platform.is_xpu():
            for _ in range(replays):
                graph.replay()
                torch.accelerator.synchronize()
        else:
            torch.cuda.profiler.start()
            with torch.cuda.nvtx.range(f"forward/{name}"):
                for _ in range(replays):
                    graph.replay()
                    torch.accelerator.synchronize()
            torch.cuda.profiler.stop()

    if prefix is None:
        run()
        return None
    activities = get_default_torch_profiler_activities()
    with torch.profiler.profile(activities=activities) as prof:
        run()
    trace = f"{prefix}.trace.json.gz"
    prof.export_chrome_trace(trace)
    with gzip.open(trace, "rt") as f:
        events = json.load(f)["traceEvents"]
    summary = {"trace": trace, **summarize(events, replays, skip)}
    with open(f"{prefix}.kernels.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def format_summary(summary: dict[str, Any], top: int = 20) -> str:
    lines = [
        (
            f"span {summary['span_us']:.1f} us, busy {summary['busy_us']:.1f} us, "
            f"{summary['activities']} activities/forward"
        ),
        f"{'us':>10} {'%':>6} {'calls':>6}  kernel",
    ]
    for row in summary["kernels"][:top]:
        lines.append(
            f"{row['us']:10.1f} {row['pct']:6.2f} {row['calls']:6d}  {row['name']}"
        )
    return "\n".join(lines)


def format_diff(a: dict[str, Any], b: dict[str, Any], top: int = 20) -> str:
    rows_a = {row["name"]: row for row in a["kernels"]}
    rows_b = {row["name"]: row for row in b["kernels"]}
    deltas = []
    for name in rows_a.keys() | rows_b.keys():
        ra, rb = rows_a.get(name), rows_b.get(name)
        us_a, us_b = ra["us"] if ra else 0.0, rb["us"] if rb else 0.0
        calls_a, calls_b = ra["calls"] if ra else 0, rb["calls"] if rb else 0
        deltas.append((us_b - us_a, us_a, us_b, calls_a, calls_b, name))
    deltas.sort(key=lambda d: -abs(d[0]))
    lines = [
        (
            f"span {a['span_us']:.1f} -> {b['span_us']:.1f} us "
            f"({b['span_us'] - a['span_us']:+.1f}), busy {a['busy_us']:.1f} -> "
            f"{b['busy_us']:.1f} us ({b['busy_us'] - a['busy_us']:+.1f})"
        ),
        f"{'delta us':>10} {'A us':>10} {'B us':>10} {'A n':>5} {'B n':>5}  kernel",
    ]
    for delta, us_a, us_b, calls_a, calls_b, name in deltas[:top]:
        lines.append(
            f"{delta:+10.1f} {us_a:10.1f} {us_b:10.1f} {calls_a:5d} {calls_b:5d}  "
            f"{name}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summaries", nargs="+", help="One or two *.kernels.json.")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if len(args.summaries) > 2:
        parser.error("Pass one summary, or two to compare.")
    loaded = []
    for path in args.summaries:
        with open(path) as f:
            loaded.append(json.load(f))
    if len(loaded) == 1:
        print(format_summary(loaded[0], args.top))
    else:
        print(format_diff(loaded[0], loaded[1], args.top))


if __name__ == "__main__":
    main()
