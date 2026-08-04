# SPDX-License-Identifier: Apache-2.0
"""Benchmark Stage2 in-process vLLM MUSA connector transfers."""

# Standard
from dataclasses import dataclass
from typing import Callable
import argparse
import os
import time

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.musa_connectors import VLLMPagedMemMUSAConnectorV2
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.platform.musa.native_kv_transfer import ENV_MUSA_NATIVE_KV_TRANSFER


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark result."""

    name: str
    seconds_per_iter: float


def compare_results(
    torch_result: BenchmarkResult,
    native_result: BenchmarkResult,
    *,
    min_speedup: float,
) -> tuple[bool, str]:
    """Compare native transfer against the torch fallback."""
    speedup = torch_result.seconds_per_iter / native_result.seconds_per_iter
    passed = speedup >= min_speedup
    return (
        passed,
        "torch={:.6f}s native={:.6f}s speedup={:.3f}x required>={:.3f}x".format(
            torch_result.seconds_per_iter,
            native_result.seconds_per_iter,
            speedup,
            min_speedup,
        ),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument(
        "--memory-device",
        choices=("cpu", "musa"),
        default="cpu",
        help="Device backing the LMCache contiguous benchmark buffer.",
    )
    parser.add_argument("--min-speedup", type=float, default=1.2)
    return parser.parse_args()


def main() -> int:
    """Run torch fallback and native opt-in transfer benchmarks."""
    args = parse_args()
    _require_musa()
    device = torch.device("musa:0")
    total_slots = args.num_blocks * args.block_size
    if args.num_tokens > total_slots:
        raise ValueError("--num-tokens must be <= num_blocks * block_size")

    source = _make_kvcaches(args, device)
    destination = [torch.zeros_like(layer) for layer in source]
    slot_mapping = torch.randperm(total_slots, device=device, dtype=torch.long)[
        : args.num_tokens
    ]
    hidden_dim = args.num_heads * args.head_size
    alloc_bytes = 2 * args.num_layers * args.num_tokens * hidden_dim * 2
    allocator = _make_allocator(args.memory_device, alloc_bytes, device)
    memobj = allocator.allocate(
        torch.Size([2, args.num_layers, args.num_tokens, hidden_dim]),
        torch.bfloat16,
        MemoryFormat.KV_2LTD,
    )
    if memobj is None:
        raise RuntimeError(
            f"Failed to allocate {args.memory_device} LMCache benchmark buffer"
        )
    memory_tensor = memobj.tensor
    if memory_tensor is None:
        raise RuntimeError(
            f"{args.memory_device} LMCache benchmark buffer has no tensor view"
        )
    print(
        f"memory_device={args.memory_device} "
        f"memory_tensor_device={memory_tensor.device}"
    )
    conn = VLLMPagedMemMUSAConnectorV2.from_metadata(
        _make_metadata(args),
        use_gpu=False,
        device=device,
    )

    old_env = os.environ.get(ENV_MUSA_NATIVE_KV_TRANSFER)
    try:
        torch_result = _run_one_mode(
            args=args,
            native_enabled=False,
            conn=conn,
            memobj=memobj,
            source=source,
            destination=destination,
            slot_mapping=slot_mapping,
        )
        for layer in destination:
            layer.zero_()
        native_result = _run_one_mode(
            args=args,
            native_enabled=True,
            conn=conn,
            memobj=memobj,
            source=source,
            destination=destination,
            slot_mapping=slot_mapping,
        )
        _assert_copied_slots_match(source, destination, slot_mapping, hidden_dim)
        passed, summary = compare_results(
            torch_result,
            native_result,
            min_speedup=args.min_speedup,
        )
        print(summary)
        return 0 if passed else 1
    finally:
        memobj.ref_count_down()
        allocator.close()
        if old_env is None:
            os.environ.pop(ENV_MUSA_NATIVE_KV_TRANSFER, None)
        else:
            os.environ[ENV_MUSA_NATIVE_KV_TRANSFER] = old_env


def _require_musa() -> None:
    """Fail clearly when the benchmark is run away from a MUSA host."""
    if not hasattr(torch, "musa") or not torch.musa.is_available():  # type: ignore[attr-defined]
        raise RuntimeError("torch.musa is not available; run on a MUSA host")


def _make_allocator(
    memory_device: str,
    alloc_bytes: int,
    device: torch.device,
) -> MemoryAllocatorInterface:
    """Create the LMCache benchmark buffer allocator."""
    size = max(alloc_bytes * 2, 64 * 1024 * 1024)
    if memory_device == "musa":
        return GPUMemoryAllocator(size=size, device=device)
    return PinMemoryAllocator(size=size)


def _make_metadata(args: argparse.Namespace) -> LMCacheMetadata:
    """Create metadata matching the synthetic benchmark KV cache."""
    return LMCacheMetadata(
        model_name="musa-stage2-bench",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(
            args.num_layers,
            2,
            args.num_tokens,
            args.num_heads,
            args.head_size,
        ),
    )


def _make_kvcaches(
    args: argparse.Namespace,
    device: torch.device,
) -> list[torch.Tensor]:
    """Allocate synthetic non-MLA vLLM MUSA paged KV caches."""
    return [
        torch.randn(
            2,
            args.num_blocks,
            args.block_size,
            args.num_heads,
            args.head_size,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(args.num_layers)
    ]


def _run_one_mode(
    *,
    args: argparse.Namespace,
    native_enabled: bool,
    conn: VLLMPagedMemMUSAConnectorV2,
    memobj: MemoryObj,
    source: list[torch.Tensor],
    destination: list[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> BenchmarkResult:
    """Run one benchmark mode."""
    os.environ[ENV_MUSA_NATIVE_KV_TRANSFER] = "1" if native_enabled else "0"

    def _transfer_once() -> None:
        conn.from_gpu(
            memobj,
            start=0,
            end=args.num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=source,
        )
        conn.to_gpu(
            memobj,
            start=0,
            end=args.num_tokens,
            slot_mapping=slot_mapping,
            kvcaches=destination,
        )

    return _time_call(
        name="native" if native_enabled else "torch",
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        fn=_transfer_once,
    )


def _time_call(
    name: str,
    iters: int,
    warmup_iters: int,
    fn: Callable[[], None],
) -> BenchmarkResult:
    """Time a synchronized MUSA callable."""
    for _ in range(warmup_iters):
        fn()
    torch.musa.synchronize()  # type: ignore[attr-defined]
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.musa.synchronize()  # type: ignore[attr-defined]
    return BenchmarkResult(
        name=name,
        seconds_per_iter=(time.perf_counter() - start) / iters,
    )


def _assert_copied_slots_match(
    source: list[torch.Tensor],
    destination: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    hidden_dim: int,
) -> None:
    """Verify destination slots match source slots after connector round-trip."""
    for src_layer, dst_layer in zip(source, destination, strict=True):
        src_k = src_layer[0].reshape(-1, hidden_dim)
        src_v = src_layer[1].reshape(-1, hidden_dim)
        dst_k = dst_layer[0].reshape(-1, hidden_dim)
        dst_v = dst_layer[1].reshape(-1, hidden_dim)
        torch.testing.assert_close(dst_k[slot_mapping], src_k[slot_mapping])
        torch.testing.assert_close(dst_v[slot_mapping], src_v[slot_mapping])


if __name__ == "__main__":
    raise SystemExit(main())
