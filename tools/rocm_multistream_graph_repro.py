# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce a ROCm graph replay hang with side-stream allocations.

This reproducer isolates the lower-level failure seen while enabling
DeepSeek-V4 CSA decode multi-stream on ROCm. The allocating mode captures a
side-stream GEMM that creates its output tensor inside the CUDA graph. On the
MI355X test system this reaches ``capture ok`` and then hangs at the first graph
replay with GPUs at 100% busy and 0% memory bandwidth.

Run from the repo root:

    HIP_VISIBLE_DEVICES=0 timeout 90s \\
      .venv/bin/python tools/rocm_multistream_graph_repro.py --mode allocating

The graph-safe variant preallocates all GEMM outputs and uses ``out=``:

    HIP_VISIBLE_DEVICES=0 timeout 90s \\
      .venv/bin/python tools/rocm_multistream_graph_repro.py --mode preallocated

On ROCm, ``torch.cuda.Event(external=True)`` is not a workaround; it raises
``RuntimeError: External events are disallowed in rocm``.
"""

from __future__ import annotations

import argparse

import torch


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    main_out = torch.empty((1024, 1024), device="cuda", dtype=torch.float16)
    return a, b, main_out


def run_allocating(replays: int) -> None:
    aux_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()
    a, b, main_out = _make_inputs()

    def work() -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        start_event.record(current_stream)
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(start_event)
            aux_out = torch.mm(a, b)
            done_event.record(aux_stream)
        torch.mm(a, b, out=main_out)
        current_stream.wait_event(done_event)
        aux_out.record_stream(current_stream)
        return main_out.float().mean() + aux_out.float().mean()

    for _ in range(3):
        y = work()
        torch.cuda.synchronize()
    print("warmup ok", float(y))

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        y = work()
    print("capture ok")

    for i in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        print("replay", i, float(y))


def run_preallocated(replays: int) -> None:
    aux_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()
    a, b, main_out = _make_inputs()
    aux_out = torch.empty_like(main_out)

    def work() -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        start_event.record(current_stream)
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(start_event)
            torch.mm(a, b, out=aux_out)
            done_event.record(aux_stream)
        torch.mm(a, b, out=main_out)
        current_stream.wait_event(done_event)
        aux_out.record_stream(current_stream)
        return main_out.float().mean() + aux_out.float().mean()

    for _ in range(3):
        y = work()
        torch.cuda.synchronize()
    print("warmup ok", float(y))

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        y = work()
    print("capture ok")

    for i in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        print("replay", i, float(y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("allocating", "preallocated"),
        required=True,
    )
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")

    if args.mode == "allocating":
        run_allocating(args.replays)
    else:
        run_preallocated(args.replays)


if __name__ == "__main__":
    main()
