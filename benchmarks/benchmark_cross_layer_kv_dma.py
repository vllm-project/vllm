# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend  # noqa: E402
from vllm.v1.simple_kv_offload.cuda_mem_ops import (  # noqa: E402
    build_params,
    copy_blocks,
    pin_tensor,
)

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "int8": torch.int8,
}


@dataclass(frozen=True)
class BenchResult:
    copy_path: str
    layout: str
    direction: str
    num_layers: int
    num_blocks: int
    transfer_blocks: int
    page_size_per_layer_bytes: int
    bytes_per_iteration: int
    descriptors_per_iteration: int
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_gib_per_s: float
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-layer vs cross-layer KV DMA copies."
    )
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument(
        "--transfer-blocks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Block counts copied in each measured iteration.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["store", "load"],
        default=["store", "load"],
        help="store is GPU->CPU, load is CPU->GPU.",
    )
    parser.add_argument(
        "--copy-path",
        choices=["backend", "direct"],
        default="backend",
        help=(
            "backend uses DmaCopyBackend.launch_copy(), matching the "
            "SimpleCPUOffloadWorker copy path. direct calls copy_blocks() "
            "directly for the raw DMA microbenchmark."
        ),
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Skip cudaHostRegister on CPU buffers.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for JSON results.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path for CSV results.",
    )
    return parser.parse_args()


def page_size_per_layer_bytes(args: argparse.Namespace) -> int:
    dtype = DTYPES[args.dtype]
    return (
        args.block_size
        * args.num_kv_heads
        * args.head_size
        * 2  # K and V
        * dtype.itemsize
    )


def make_per_layer_caches(
    *,
    num_layers: int,
    num_blocks: int,
    page_size_bytes: int,
    device: torch.device,
    pin_memory: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    gpu_caches = {
        f"layer_{idx}": torch.empty(
            (num_blocks, page_size_bytes), dtype=torch.int8, device=device
        )
        for idx in range(num_layers)
    }
    cpu_caches = {
        name: torch.empty_like(tensor, device="cpu")
        for name, tensor in gpu_caches.items()
    }
    if pin_memory:
        for tensor in cpu_caches.values():
            pin_tensor(tensor)
    return gpu_caches, cpu_caches


def make_cross_layer_caches(
    *,
    num_layers: int,
    num_blocks: int,
    page_size_bytes: int,
    device: torch.device,
    pin_memory: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    cross_page_size = page_size_bytes * num_layers
    gpu_caches = {
        "cross_layer": torch.empty(
            (num_blocks, cross_page_size), dtype=torch.int8, device=device
        )
    }
    cpu_caches = {
        "cross_layer": torch.empty_like(gpu_caches["cross_layer"], device="cpu")
    }
    if pin_memory:
        pin_tensor(cpu_caches["cross_layer"])
    return gpu_caches, cpu_caches


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = round((len(values) - 1) * pct)
    return sorted(values)[idx]


def timed_copy_ms(
    *,
    src_blocks: list[int],
    dst_blocks: list[int],
    params,
    stream: torch.cuda.Stream,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        copy_blocks(src_blocks, dst_blocks, params)
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def timed_backend_copy_ms(
    *,
    src_blocks: list[int],
    dst_blocks: list[int],
    backend: DmaCopyBackend,
    is_store: bool,
    event_idx: int,
    events_list: list[tuple[int, torch.Event]],
) -> float:
    start = time.perf_counter()
    backend.launch_copy(
        src_blocks,
        dst_blocks,
        is_store=is_store,
        event_idx=event_idx,
        events_list=events_list,
    )

    while not events_list or events_list[-1][0] != event_idx:
        time.sleep(0)
    events_list[-1][1].synchronize()
    return (time.perf_counter() - start) * 1000.0


def run_one(
    *,
    layout: str,
    direction: str,
    gpu_caches: dict[str, torch.Tensor],
    cpu_caches: dict[str, torch.Tensor],
    backend: DmaCopyBackend | None,
    stream: torch.cuda.Stream,
    args: argparse.Namespace,
    page_size_bytes: int,
    transfer_blocks: int,
) -> BenchResult:
    if direction == "store":
        src_caches, dst_caches = gpu_caches, cpu_caches
    else:
        src_caches, dst_caches = cpu_caches, gpu_caches
    params = None
    if args.copy_path == "direct":
        params = build_params(src_caches, dst_caches, stream)

    src_blocks = list(range(transfer_blocks))
    dst_blocks = list(range(transfer_blocks))
    events_list: list[tuple[int, torch.Event]] = []
    event_idx = 0

    def copy_once() -> float:
        nonlocal event_idx
        if args.copy_path == "backend":
            assert backend is not None
            event_idx += 1
            return timed_backend_copy_ms(
                src_blocks=src_blocks,
                dst_blocks=dst_blocks,
                backend=backend,
                is_store=direction == "store",
                event_idx=event_idx,
                events_list=events_list,
            )

        assert params is not None
        return timed_copy_ms(
            src_blocks=src_blocks,
            dst_blocks=dst_blocks,
            params=params,
            stream=stream,
        )

    for _ in range(args.warmup_iters):
        copy_once()

    samples_ms = [copy_once() for _ in range(args.iters)]

    bytes_per_iteration = transfer_blocks * page_size_bytes * args.num_layers
    mean_ms = statistics.mean(samples_ms)
    descriptors = transfer_blocks * (args.num_layers if layout == "per_layer" else 1)
    gib = bytes_per_iteration / float(1 << 30)
    mean_gib_per_s = gib / (mean_ms / 1000.0) if mean_ms > 0 else float("inf")

    return BenchResult(
        copy_path=args.copy_path,
        layout=layout,
        direction=direction,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        transfer_blocks=transfer_blocks,
        page_size_per_layer_bytes=page_size_bytes,
        bytes_per_iteration=bytes_per_iteration,
        descriptors_per_iteration=descriptors,
        mean_ms=mean_ms,
        median_ms=statistics.median(samples_ms),
        p90_ms=percentile(samples_ms, 0.90),
        p99_ms=percentile(samples_ms, 0.99),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        mean_gib_per_s=mean_gib_per_s,
        iterations=args.iters,
    )


def print_results(results: Iterable[BenchResult]) -> None:
    rows = list(results)
    print(
        "copy_path,layout,direction,blocks,descriptors,bytes,"
        "mean_ms,p90_ms,p99_ms,mean_gib_s"
    )
    for r in rows:
        print(
            f"{r.copy_path},{r.layout},{r.direction},{r.transfer_blocks},"
            f"{r.descriptors_per_iteration},{r.bytes_per_iteration},"
            f"{r.mean_ms:.4f},{r.p90_ms:.4f},{r.p99_ms:.4f},"
            f"{r.mean_gib_per_s:.2f}"
        )

    by_key = {(r.direction, r.transfer_blocks, r.layout): r for r in rows}
    print("\nSpeedup: per_layer_mean_ms / cross_layer_mean_ms")
    for direction in sorted({r.direction for r in rows}):
        for transfer_blocks in sorted({r.transfer_blocks for r in rows}):
            per = by_key.get((direction, transfer_blocks, "per_layer"))
            cross = by_key.get((direction, transfer_blocks, "cross_layer"))
            if per is None or cross is None or cross.mean_ms == 0:
                continue
            descriptor_reduction = (
                per.descriptors_per_iteration / cross.descriptors_per_iteration
            )
            print(
                f"{direction:5s} blocks={transfer_blocks:<4d} "
                f"speedup={per.mean_ms / cross.mean_ms:.2f}x "
                f"descriptor_reduction={descriptor_reduction:.1f}x"
            )


def write_outputs(args: argparse.Namespace, results: list[BenchResult]) -> None:
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n"
        )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)


def main() -> int:
    args = parse_args()
    if not current_platform.is_cuda_alike():
        print("This benchmark requires CUDA or ROCm.", file=sys.stderr)
        return 1
    if max(args.transfer_blocks) > args.num_blocks:
        raise ValueError("--transfer-blocks cannot exceed --num-blocks")

    device = torch.device("cuda:0")
    pin_memory = not args.no_pin_memory
    page_size = page_size_per_layer_bytes(args)

    print(
        "Config: "
        f"layers={args.num_layers}, num_blocks={args.num_blocks}, "
        f"page_size_per_layer={page_size} bytes, dtype={args.dtype}, "
        f"pin_memory={pin_memory}, copy_path={args.copy_path}"
    )

    stream = torch.cuda.Stream()
    results: list[BenchResult] = []

    cache_factories = {
        "per_layer": make_per_layer_caches,
        "cross_layer": make_cross_layer_caches,
    }
    for layout, factory in cache_factories.items():
        gpu_caches, cpu_caches = factory(
            num_layers=args.num_layers,
            num_blocks=args.num_blocks,
            page_size_bytes=page_size,
            device=device,
            pin_memory=pin_memory,
        )
        try:
            backend = None
            if args.copy_path == "backend":
                backend = DmaCopyBackend()
                backend.init(
                    gpu_caches,
                    cpu_caches,
                    device,
                    torch.cuda.Stream(),
                    torch.cuda.Stream(),
                )
            for direction in args.directions:
                for transfer_blocks in args.transfer_blocks:
                    results.append(
                        run_one(
                            layout=layout,
                            direction=direction,
                            gpu_caches=gpu_caches,
                            cpu_caches=cpu_caches,
                            backend=backend,
                            stream=stream,
                            args=args,
                            page_size_bytes=page_size,
                            transfer_blocks=transfer_blocks,
                        )
                    )
        finally:
            if backend is not None:
                backend.shutdown()
            del gpu_caches
            del cpu_caches
            torch.cuda.empty_cache()

    print_results(results)
    write_outputs(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
