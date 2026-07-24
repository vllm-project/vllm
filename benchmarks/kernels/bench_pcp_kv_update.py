# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark replicated PCP cache updates with collectives or peer stores.

This is a communication microbenchmark for the stage-1 replicated-cache
design.  It compares two ways to make each PCP rank's local ``kv_c``, ``k_pe``,
and ``indexer_k`` payload visible in every rank's cache:

* ``all_gather``: three NCCL all-gathers followed by one representative local
  cache-insertion kernel;
* ``direct``: one Triton kernel that stores each local payload into every
  rank's CUDA VMM cache view, followed by the production release/acquire fence.

Both paths produce the same component-major, replicated BF16 cache image.  The
benchmark intentionally excludes model work (RMSNorm, RoPE, quantization, slot
mapping, and attention) so it measures only update transport, insertion, and
visibility.  Widths are configurable when a different semantic layout is
needed.

Example on one four-GPU node::

    torchrun --standalone --nproc-per-node=4 \
      benchmarks/kernels/bench_pcp_kv_update.py \
      --local-tokens 1,8,32,128,512,2048 \
      --warmup 20 --repetitions 100 \
      --json-output /tmp/pcp-kv-update.json

Only rank zero writes JSON.  Latency percentiles are computed from the slowest
rank in every repetition, rather than from independently pooled rank samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.cuda_vmm import (
    create_rank_major_peer_view,
)
from vllm.model_executor.layers.attention.pcp_peer_cache import (
    PCPPeerCacheFence,
    make_rank_major_tensor_view,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _insert_gathered_kernel(
    gathered_kv_c_ptr,
    gathered_k_pe_ptr,
    gathered_indexer_k_ptr,
    local_cache_ptr,
    component_stride,
    kv_c_numel,
    k_pe_numel,
    indexer_k_numel,
    kv_c_offset,
    k_pe_offset,
    indexer_k_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy three gathered source-rank components into one local cache."""
    source_rank = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    destination = local_cache_ptr + source_rank * component_stride

    kv_c_mask = offsets < kv_c_numel
    kv_c = tl.load(
        gathered_kv_c_ptr + source_rank * kv_c_numel + offsets,
        mask=kv_c_mask,
    )
    tl.store(destination + kv_c_offset + offsets, kv_c, mask=kv_c_mask)

    k_pe_mask = offsets < k_pe_numel
    k_pe = tl.load(
        gathered_k_pe_ptr + source_rank * k_pe_numel + offsets,
        mask=k_pe_mask,
    )
    tl.store(destination + k_pe_offset + offsets, k_pe, mask=k_pe_mask)

    indexer_k_mask = offsets < indexer_k_numel
    indexer_k = tl.load(
        gathered_indexer_k_ptr + source_rank * indexer_k_numel + offsets,
        mask=indexer_k_mask,
    )
    tl.store(
        destination + indexer_k_offset + offsets,
        indexer_k,
        mask=indexer_k_mask,
    )


@triton.jit
def _direct_fanout_kernel(
    local_kv_c_ptr,
    local_k_pe_ptr,
    local_indexer_k_ptr,
    peer_cache_ptr,
    destination_stride,
    component_stride,
    source_rank: tl.constexpr,
    world_size: tl.constexpr,
    kv_c_numel,
    k_pe_numel,
    indexer_k_numel,
    kv_c_offset,
    k_pe_offset,
    indexer_k_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Fan one source rank's components out to every replicated cache."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    kv_c_mask = offsets < kv_c_numel
    k_pe_mask = offsets < k_pe_numel
    indexer_k_mask = offsets < indexer_k_numel
    kv_c = tl.load(local_kv_c_ptr + offsets, mask=kv_c_mask)
    k_pe = tl.load(local_k_pe_ptr + offsets, mask=k_pe_mask)
    indexer_k = tl.load(local_indexer_k_ptr + offsets, mask=indexer_k_mask)

    for destination_rank in range(world_size):
        destination = (
            peer_cache_ptr
            + destination_rank * destination_stride
            + source_rank * component_stride
        )
        tl.store(destination + kv_c_offset + offsets, kv_c, mask=kv_c_mask)
        tl.store(destination + k_pe_offset + offsets, k_pe, mask=k_pe_mask)
        tl.store(
            destination + indexer_k_offset + offsets,
            indexer_k,
            mask=indexer_k_mask,
        )


def _parse_local_tokens(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "local token sizes must be comma-separated integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("local token sizes must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("local token sizes must be unique")
    return values


def _percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summarize_slowest_rank(
    local_samples_ms: list[float], cpu_group: dist.ProcessGroup
) -> tuple[dict[str, float] | None, list[float] | None]:
    per_rank_samples: list[list[float] | None] = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(per_rank_samples, local_samples_ms, group=cpu_group)
    if dist.get_rank(cpu_group) != 0:
        return None, None

    samples = [sample for sample in per_rank_samples if sample is not None]
    slowest_ms = [max(values) for values in zip(*samples, strict=True)]
    summary = {
        "p50_us": _percentile(slowest_ms, 0.50) * 1000.0,
        "p90_us": _percentile(slowest_ms, 0.90) * 1000.0,
        "p99_us": _percentile(slowest_ms, 0.99) * 1000.0,
        "min_us": min(slowest_ms) * 1000.0,
        "max_us": max(slowest_ms) * 1000.0,
    }
    return summary, [value * 1000.0 for value in slowest_ms]


def _time_cuda(operation: Any, repetitions: int) -> list[float]:
    samples_ms: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return samples_ms


def _make_component(
    local_tokens: int,
    width: int,
    rank: int,
    component_id: int,
    device: torch.device,
) -> torch.Tensor:
    # Exactly representable BF16 integers make the preflight comparison strict.
    values = torch.arange(local_tokens * width, device=device, dtype=torch.int64)
    values = (values + rank * 101 + component_id * 17) % 251
    return values.to(torch.bfloat16).view(local_tokens, width)


def _launch_insert(
    gathered: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cache: torch.Tensor,
    numels: tuple[int, int, int],
    offsets: tuple[int, int, int],
    max_numel: int,
    world_size: int,
) -> None:
    _insert_gathered_kernel[(triton.cdiv(max_numel, 256), world_size)](
        *gathered,
        cache,
        cache.stride(0),
        *numels,
        *offsets,
        BLOCK_SIZE=256,
    )


def _launch_direct(
    components: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    peer_cache: torch.Tensor,
    numels: tuple[int, int, int],
    offsets: tuple[int, int, int],
    max_numel: int,
    rank: int,
    world_size: int,
    fence: PCPPeerCacheFence,
) -> None:
    _direct_fanout_kernel[(triton.cdiv(max_numel, 256),)](
        *components,
        peer_cache,
        peer_cache.stride(0),
        peer_cache.stride(1),
        rank,
        world_size,
        *numels,
        *offsets,
        BLOCK_SIZE=256,
    )
    fence()


def _run_shape(
    args: argparse.Namespace,
    local_tokens: int,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group: dist.ProcessGroup,
) -> dict[str, Any] | None:
    widths = (args.kv_c_width, args.k_pe_width, args.indexer_k_width)
    components = tuple(
        _make_component(local_tokens, width, rank, component_id, device)
        for component_id, width in enumerate(widths)
    )
    numels = tuple(component.numel() for component in components)
    offsets = (0, numels[0], numels[0] + numels[1])
    component_stride = sum(numels)
    max_numel = max(numels)

    gathered = tuple(
        torch.empty(
            (world_size * local_tokens, width),
            dtype=torch.bfloat16,
            device=device,
        )
        for width in widths
    )
    collective_cache = torch.empty(
        (world_size, component_stride), dtype=torch.bfloat16, device=device
    )
    peer_allocation = create_rank_major_peer_view(
        (world_size, component_stride),
        dtype=torch.bfloat16,
        group=cpu_group,
        require_native_atomics=True,
        device=device,
    )
    assert peer_allocation.local_view is not None
    requested_local_cache = peer_allocation.local_view[:world_size]
    peer_cache = make_rank_major_tensor_view(peer_allocation, requested_local_cache)
    fence = PCPPeerCacheFence(cpu_group, device)

    def collective_update() -> None:
        for output, source in zip(gathered, components, strict=True):
            dist.all_gather_into_tensor(output, source)
        _launch_insert(
            gathered,
            collective_cache,
            numels,
            offsets,
            max_numel,
            world_size,
        )

    def direct_update() -> None:
        _launch_direct(
            components,
            peer_cache,
            numels,
            offsets,
            max_numel,
            rank,
            world_size,
            fence,
        )

    try:
        collective_update()
        direct_update()
        torch.cuda.synchronize(device)
        if not torch.equal(collective_cache, requested_local_cache):
            raise AssertionError(
                f"rank {rank}: collective and direct cache images differ"
            )

        for _ in range(args.warmup):
            collective_update()
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        collective_samples = _time_cuda(collective_update, args.repetitions)
        collective_summary, collective_slowest = _summarize_slowest_rank(
            collective_samples, cpu_group
        )

        for _ in range(args.warmup):
            direct_update()
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        direct_samples = _time_cuda(direct_update, args.repetitions)
        direct_summary, direct_slowest = _summarize_slowest_rank(
            direct_samples, cpu_group
        )
        dist.barrier(group=cpu_group)

        if rank != 0:
            return None
        assert collective_summary is not None
        assert collective_slowest is not None
        assert direct_summary is not None
        assert direct_slowest is not None
        logical_payload_bytes = component_stride * torch.bfloat16.itemsize
        return {
            "local_tokens": local_tokens,
            "width_elements": {
                "kv_c": widths[0],
                "k_pe": widths[1],
                "indexer_k": widths[2],
            },
            "logical_payload_bytes_per_rank": logical_payload_bytes,
            "collective": {
                "latency": collective_summary,
                "slowest_rank_samples_us": collective_slowest,
                "kv_update_collective_calls_per_iteration": 3,
                "cache_insert_kernel_calls_per_iteration": 1,
                "collective_input_bytes_per_rank": logical_payload_bytes,
                "remote_payload_bytes_received_per_rank": (
                    (world_size - 1) * logical_payload_bytes
                ),
                "replicated_cache_bytes_written_per_rank": (
                    world_size * logical_payload_bytes
                ),
            },
            "direct": {
                "latency": direct_summary,
                "slowest_rank_samples_us": direct_slowest,
                "kv_update_collective_calls_per_iteration": 0,
                "peer_store_kernel_calls_per_iteration": 1,
                "fence_kernel_calls_per_iteration": 2,
                "logical_peer_store_bytes_issued_per_rank": (
                    world_size * logical_payload_bytes
                ),
                "logical_remote_peer_store_bytes_issued_per_rank": (
                    (world_size - 1) * logical_payload_bytes
                ),
                "fence_publish_atomic_bytes_per_rank": world_size * 4,
                "fence_wait_atomic_read_bytes": "data-dependent",
            },
            "direct_p50_speedup_x": (
                collective_summary["p50_us"] / direct_summary["p50_us"]
            ),
            "direct_p50_latency_reduction_percent": (
                (1.0 - direct_summary["p50_us"] / collective_summary["p50_us"]) * 100.0
            ),
        }
    finally:
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        fence.close()
        peer_allocation.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-tokens",
        type=_parse_local_tokens,
        default=_parse_local_tokens("1,8,32,128,512,2048"),
        help="Comma-separated local token counts (default: %(default)s).",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--kv-c-width", type=int, default=512)
    parser.add_argument("--k-pe-width", type=int, default=64)
    parser.add_argument("--indexer-k-width", type=int, default=128)
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=4,
        help="Fail unless torchrun launches this many ranks (default: 4).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Also write rank-zero JSON to this path.",
    )
    parser.add_argument(
        "--min-direct-p50-latency-reduction-percent",
        type=float,
        help=(
            "Fail rank zero when any shape misses this direct-path p50 "
            "latency-reduction threshold."
        ),
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if min(args.kv_c_width, args.k_pe_width, args.indexer_k_width) <= 0:
        parser.error("component widths must be positive")
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if (
        args.min_direct_p50_latency_reduction_percent is not None
        and not 0.0 <= args.min_direct_p50_latency_reduction_percent < 100.0
    ):
        parser.error("--min-direct-p50-latency-reduction-percent must be in [0, 100)")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA GPUs")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch this benchmark with torchrun")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise RuntimeError(
            f"expected {args.expected_world_size} ranks, got {world_size}"
        )

    results: list[dict[str, Any]] = []
    try:
        for local_tokens in args.local_tokens:
            if rank == 0:
                print(
                    f"benchmarking local_tokens={local_tokens}",
                    file=sys.stderr,
                    flush=True,
                )
            result = _run_shape(args, local_tokens, rank, world_size, device, cpu_group)
            if result is not None:
                results.append(result)
    finally:
        dist.barrier(group=cpu_group)
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()

    if rank == 0:
        report = {
            "benchmark": "pcp_replicated_kv_update",
            "scope": (
                "raw BF16 semantic payload transport, replicated insertion, "
                "and visibility; excludes model transforms and attention"
            ),
            "world_size": world_size,
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "min_direct_p50_latency_reduction_percent": (
                args.min_direct_p50_latency_reduction_percent
            ),
            "device": torch.cuda.get_device_name(device),
            "results": results,
        }
        encoded = json.dumps(report, indent=2, sort_keys=True)
        print(encoded)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(encoded + "\n", encoding="utf-8")
        threshold = args.min_direct_p50_latency_reduction_percent
        if threshold is not None:
            failures = [
                (result["local_tokens"], result["direct_p50_latency_reduction_percent"])
                for result in results
                if result["direct_p50_latency_reduction_percent"] < threshold
            ]
            if failures:
                details = ", ".join(
                    f"tokens={tokens}: {reduction:.2f}%"
                    for tokens, reduction in failures
                )
                raise RuntimeError(
                    f"direct p50 latency reduction missed {threshold:.2f}%: {details}"
                )


if __name__ == "__main__":
    main()
