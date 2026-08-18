# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the sparse-indexer top-k kernels with CUDA graphs.

The benchmark follows the selector dispatch used by ``SparseAttnIndexer``:
cooperative top-k for at most 32 rows and persistent top-k otherwise. Inputs
are built from captured model logits, and every result is checked against
``torch.topk`` before it is timed.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_topk.py \
        --capture /tmp/gvr_real_matrix_valid/gvr_b32_call21.pt \
        --extra-capture /tmp/gvr_real_matrix_valid/gvr_b32_call0.pt
"""

import argparse
import math
import statistics
from collections.abc import Callable

import torch
import vllm._C_stable_libtorch  # noqa: F401

ROWS = (1, 8, 32, 128, 1024, 8192, 16384)
KV_LENGTHS = (10_000, 50_000, 100_000, 200_000)
WORKSPACE_SIZE = 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--extra-capture")
    parser.add_argument("--label", default="current")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float16", "float32"),
        default=("float16", "float32"),
    )
    parser.add_argument("--rows", nargs="+", type=int, default=ROWS)
    parser.add_argument("--kv-lengths", nargs="+", type=int, default=KV_LENGTHS)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--backend",
        choices=("auto", "cooperative", "persistent", "decode"),
        default="auto",
    )
    return parser.parse_args()


def _load_logits(path: str) -> torch.Tensor:
    capture = torch.load(path, map_location="cpu", weights_only=True)
    logits = capture["logits"]
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError(f"Expected 2D FP32 logits in {path}, got {logits}")
    return logits


def _make_source(args: argparse.Namespace) -> torch.Tensor:
    source = _load_logits(args.capture)
    if args.extra_capture:
        extra = _load_logits(args.extra_capture)
        if extra.shape[0] != source.shape[0]:
            raise ValueError("Captures must contain the same number of rows")
        source = torch.cat((source, extra), dim=1)
    required = max(args.kv_lengths)
    if source.shape[1] < required:
        repeats = math.ceil(required / source.shape[1])
        source = source.repeat(1, repeats)
    return source[:, :required].contiguous()


def _selector(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    top_k: int,
    kv_length: int,
    backend_override: str,
) -> tuple[str, Callable[[], None]]:
    use_cooperative = logits.shape[0] <= 32
    use_decode = logits.dtype == torch.float16 and (
        (kv_length <= 32768 and logits.shape[0] >= 512)
        or (32768 < kv_length <= 131072 and logits.shape[0] >= 256)
    )
    backend = (
        ("cooperative" if use_cooperative else "decode" if use_decode else "persistent")
        if backend_override == "auto"
        else backend_override
    )

    if backend == "cooperative":
        backend = "cooperative"

        def launch() -> None:
            torch.ops._C.cooperative_topk(
                logits, lengths, output, workspace, top_k, kv_length
            )

    elif backend == "decode":
        backend = "decode"

        def launch() -> None:
            torch.ops._C.top_k_per_row_decode(
                logits,
                1,
                lengths,
                output,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                top_k,
            )

    elif backend == "persistent":
        backend = "persistent"

        def launch() -> None:
            torch.ops._C.persistent_topk(
                logits, lengths, output, workspace, top_k, kv_length
            )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return backend, launch


def _check_correctness(logits: torch.Tensor, output: torch.Tensor, top_k: int) -> None:
    checked_rows = min(logits.shape[0], 32)
    actual = torch.gather(logits[:checked_rows], 1, output[:checked_rows].long())
    actual = actual.sort(dim=1, descending=True).values
    expected = torch.topk(logits[:checked_rows], top_k, dim=1).values
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).sum().item()
        raise AssertionError(f"Top-k selected values differ at {mismatch} positions")
    if not bool(((output >= 0) & (output < logits.shape[1])).all()):
        raise AssertionError("Top-k returned an out-of-range index")


def _graph_parameters(num_elements: int) -> tuple[int, int]:
    graph_calls = min(100, max(1, 100_000_000 // num_elements))
    graph_elements = graph_calls * num_elements
    replays = min(20, max(3, 300_000_000 // graph_elements))
    return graph_calls, replays


def _benchmark_graph(
    launch: Callable[[], None],
    graph_calls: int,
    replays: int,
    samples: int,
) -> float:
    for _ in range(3):
        launch()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            launch()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / (graph_calls * replays))
    return statistics.median(timings)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    torch.accelerator.set_device_index(0)
    source = _make_source(args).to("cuda")
    dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
    results = []

    for dtype_name in args.dtypes:
        dtype = dtype_by_name[dtype_name]
        for kv_length in args.kv_lengths:
            base = source[:, :kv_length].to(dtype)
            for rows in args.rows:
                repeats = math.ceil(rows / base.shape[0])
                logits = base.repeat(repeats, 1)[:rows].contiguous()
                lengths = torch.full(
                    (rows,), kv_length, dtype=torch.int32, device="cuda"
                )
                output = torch.empty(
                    (rows, args.top_k), dtype=torch.int32, device="cuda"
                )
                workspace = torch.empty(
                    WORKSPACE_SIZE, dtype=torch.uint8, device="cuda"
                )
                backend, launch = _selector(
                    logits,
                    lengths,
                    output,
                    workspace,
                    args.top_k,
                    kv_length,
                    args.backend,
                )
                launch()
                torch.accelerator.synchronize()
                _check_correctness(logits, output, args.top_k)

                graph_calls, replays = _graph_parameters(rows * kv_length)
                latency_us = _benchmark_graph(
                    launch, graph_calls, replays, args.samples
                )
                results.append(
                    (
                        dtype_name,
                        rows,
                        kv_length,
                        backend,
                        latency_us,
                        graph_calls,
                        replays,
                    )
                )
                print(
                    f"RESULT,{args.label},{dtype_name},{rows},{kv_length},"
                    f"{backend},{latency_us:.3f},{graph_calls},{replays}",
                    flush=True,
                )
                del logits, lengths, output, workspace, launch
                torch.accelerator.empty_cache()

    print("\n| dtype | rows | KV | backend | us | graph calls | replays |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for dtype_name, rows, kv_length, backend, latency, calls, replays in results:
        print(
            f"| {dtype_name} | {rows} | {kv_length} | {backend} | "
            f"{latency:.3f} | {calls} | {replays} |"
        )


if __name__ == "__main__":
    main()
