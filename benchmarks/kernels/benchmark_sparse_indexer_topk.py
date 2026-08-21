# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the sparse-indexer top-k kernels with CUDA graphs.

The default follows the selector dispatch used by ``SparseAttnIndexer``.
Individual backends can also be selected to compare FP32 performance before
and after a kernel change. ``persistent_topk`` and ``filtered_topk`` both call
the public ``persistent_topk`` op with shapes that force the named internal
implementation. Every result is checked against ``torch.topk`` before timing.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_topk.py \
        --synthetic --dtypes float32 --backend all

    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_topk.py \
        --synthetic --backend top_k_per_row_prefill \
        --prefill-pattern causal --rows 128 512 1024 2048
"""

import argparse
import importlib
import math
from collections.abc import Callable

import torch

ROWS = (1, 8, 32, 128, 1024, 8192, 16384)
KV_LENGTHS = (10_000, 50_000, 100_000, 200_000)
WORKSPACE_SIZE = 1024 * 1024
BACKENDS = (
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "cooperative_topk",
    "persistent_topk",
    "filtered_topk",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use reproducible synthetic FP32 logits.",
    )
    parser.add_argument("--extra-capture")
    parser.add_argument(
        "--library",
        help="Load this _C_stable_libtorch library instead of the installed one.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="current")
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float16", "float32"),
        default=("float16", "float32"),
    )
    parser.add_argument("--rows", nargs="+", type=int, default=ROWS)
    parser.add_argument("--kv-lengths", nargs="+", type=int, default=KV_LENGTHS)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help=(
            "Allocate this row stride and pass it as the captured max sequence length."
        ),
    )
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=20,
        help="Target duration in milliseconds for each CUDA-graph measurement.",
    )
    parser.add_argument(
        "--prefill-pattern",
        choices=("full", "causal"),
        default="full",
        help=(
            "Use a common full row range or causal row ends for the explicit "
            "prefill backend."
        ),
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=("auto", "all", *BACKENDS),
        default=("auto",),
    )
    args = parser.parse_args()
    if args.extra_capture and not args.capture:
        parser.error("--extra-capture requires --capture")
    if "all" in args.backend and len(args.backend) != 1:
        parser.error("--backend all cannot be combined with another backend")
    if "auto" in args.backend and len(args.backend) != 1:
        parser.error("--backend auto cannot be combined with another backend")
    return args


def _load_logits(path: str) -> torch.Tensor:
    capture = torch.load(path, map_location="cpu", weights_only=True)
    logits = capture["logits"]
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError(f"Expected 2D FP32 logits in {path}, got {logits}")
    return logits


def _make_source(args: argparse.Namespace) -> torch.Tensor:
    required = max(max(args.kv_lengths), args.max_seq_len or 0)
    if args.capture:
        source = _load_logits(args.capture)
        if args.extra_capture:
            extra = _load_logits(args.extra_capture)
            if extra.shape[0] != source.shape[0]:
                raise ValueError("Captures must contain the same number of rows")
            source = torch.cat((source, extra), dim=1)
    else:
        generator = torch.Generator().manual_seed(args.seed)
        source_rows = min(max(args.rows), 32)
        source = torch.randn(source_rows, required, generator=generator)
    if source.shape[1] < required:
        repeats = math.ceil(required / source.shape[1])
        source = source.repeat(1, repeats)
    return source[:, :required].contiguous()


def _workspace_backend(logits: torch.Tensor) -> str:
    properties = torch.cuda.get_device_properties(logits.device)
    has_filtered_topk = properties.shared_memory_per_block_optin >= 128 * 1024
    if (logits.shape[0] > 32 or logits.dtype == torch.float16) and has_filtered_topk:
        return "filtered_topk"
    return "persistent_topk"


def _selector(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    top_k: int,
    max_seq_len: int,
    requested_backend: str,
    prefill_pattern: str,
) -> tuple[str, Callable[[], None], torch.Tensor, torch.Tensor] | None:
    use_cooperative = logits.shape[0] <= 32
    use_decode = logits.dtype == torch.float16 and (
        (32768 < max_seq_len <= 65536 and logits.shape[0] >= 768)
        or (65536 < max_seq_len <= 100000 and logits.shape[0] >= 512)
        or (100000 < max_seq_len <= 131072 and logits.shape[0] >= 1024)
    )
    if requested_backend == "auto":
        backend = (
            "cooperative_topk"
            if use_cooperative
            else "top_k_per_row_decode"
            if use_decode
            else _workspace_backend(logits)
        )
    else:
        backend = requested_backend

    starts = torch.zeros_like(lengths)
    ends = lengths

    if backend == "cooperative_topk":
        if logits.shape[0] > 32:
            return None

        def launch() -> None:
            torch.ops._C.cooperative_topk(
                logits, lengths, output, workspace, top_k, max_seq_len
            )

    elif backend == "top_k_per_row_decode":

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

    elif backend == "top_k_per_row_prefill":
        if prefill_pattern == "causal":
            ends = lengths - torch.arange(
                logits.shape[0] - 1,
                -1,
                -1,
                dtype=torch.int32,
                device=logits.device,
            )
            if int(ends[0]) - int(starts[0]) < top_k:
                raise ValueError(
                    "The first causal prefill row has fewer valid values than top-k"
                )

        def launch() -> None:
            torch.ops._C.top_k_per_row_prefill(
                logits,
                starts,
                ends,
                output,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                top_k,
            )

    elif backend in ("persistent_topk", "filtered_topk"):
        if _workspace_backend(logits) != backend:
            return None

        def launch() -> None:
            torch.ops._C.persistent_topk(
                logits, lengths, output, workspace, top_k, max_seq_len
            )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return backend, launch, starts, ends


def _check_correctness(
    logits: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> None:
    checked_rows = (
        torch.linspace(
            0,
            logits.shape[0] - 1,
            min(logits.shape[0], 32),
            device=logits.device,
        )
        .round()
        .long()
        .unique()
    )
    for row_tensor in checked_rows:
        row = int(row_tensor)
        row_start = int(row_starts[row])
        row_end = int(row_ends[row])
        indices = output[row].long()
        if not bool(((indices >= row_start) & (indices < row_end)).all()):
            raise AssertionError(f"Top-k returned an out-of-range index in row {row}")
        actual = logits[row, indices].sort(descending=True).values
        expected = torch.topk(logits[row, row_start:row_end], top_k).values
        if not torch.equal(actual, expected):
            mismatch = (actual != expected).sum().item()
            raise AssertionError(
                f"Top-k selected values differ at {mismatch} positions in row {row}"
            )


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if args.library:
        torch.ops.load_library(args.library)
    else:
        importlib.import_module("vllm._C_stable_libtorch")
    triton_testing = importlib.import_module("triton.testing")
    torch.accelerator.set_device_index(0)
    source = _make_source(args).to("cuda")
    dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
    results = []

    requested_backends = BACKENDS if args.backend == ["all"] else args.backend
    for dtype_name in args.dtypes:
        dtype = dtype_by_name[dtype_name]
        for kv_length in args.kv_lengths:
            max_seq_len = args.max_seq_len or kv_length
            if kv_length > max_seq_len:
                raise ValueError(
                    f"KV length {kv_length} exceeds max sequence length {max_seq_len}"
                )
            base = source[:, :max_seq_len].to(dtype)
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
                for requested_backend in requested_backends:
                    selected = _selector(
                        logits,
                        lengths,
                        output,
                        workspace,
                        args.top_k,
                        max_seq_len,
                        requested_backend,
                        args.prefill_pattern,
                    )
                    if selected is None:
                        continue
                    backend, launch, row_starts, row_ends = selected
                    launch()
                    torch.accelerator.synchronize()
                    _check_correctness(
                        logits,
                        output,
                        args.top_k,
                        row_starts,
                        row_ends,
                    )

                    latency_us = 1000 * triton_testing.do_bench_cudagraph(
                        launch,
                        rep=args.rep_ms,
                        return_mode="median",
                    )
                    results.append(
                        (
                            dtype_name,
                            rows,
                            kv_length,
                            backend,
                            latency_us,
                        )
                    )
                    print(
                        f"RESULT,{args.label},{dtype_name},{rows},{kv_length},"
                        f"{backend},{latency_us:.3f}",
                        flush=True,
                    )
                    del launch
                del logits, lengths, output, workspace
                torch.accelerator.empty_cache()

    print("\n| dtype | rows | KV | backend | us |")
    print("|---|---:|---:|---|---:|")
    for dtype_name, rows, kv_length, backend, latency in results:
        print(f"| {dtype_name} | {rows} | {kv_length} | {backend} | {latency:.3f} |")

    latency_by_shape = {
        (dtype_name, rows, kv_length, backend): latency
        for dtype_name, rows, kv_length, backend, latency in results
    }
    speedups = []
    paired_shapes = set()
    for _, rows, kv_length, backend, _ in results:
        key = (rows, kv_length, backend)
        fp16 = latency_by_shape.get(("float16", *key))
        fp32 = latency_by_shape.get(("float32", *key))
        if fp16 is not None and fp32 is not None and key not in paired_shapes:
            speedups.append((*key, fp16, fp32, fp32 / fp16))
            paired_shapes.add(key)

    if speedups:
        print("\n| rows | KV | backend | FP16 us | FP32 us | FP32 / FP16 |")
        print("|---:|---:|---|---:|---:|---:|")
        for rows, kv_length, backend, fp16, fp32, speedup in speedups:
            print(
                f"| {rows} | {kv_length} | {backend} | {fp16:.3f} | "
                f"{fp32:.3f} | {speedup:.3f}x |"
            )


if __name__ == "__main__":
    main()
