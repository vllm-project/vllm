# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the NIXL host-buffer KV copies across KV cache groups.

`NixlBaseConnectorWorker.sync_recved_kv_to_device` / `save_kv_to_host` move a
request's blocks between the CPU transfer buffer and the device KV cache. Block
ids are unique across KV cache groups, so the per-group copies can be issued as
one. This measures what that coalescing saves: the copied bytes are identical,
only the number of launches changes.
"""

import time

import torch
from tabulate import tabulate

from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

logger = init_logger(__name__)


def _make_caches(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Host transfer buffers and device KV caches, shaped as the worker builds them."""
    shape = (num_blocks, block_size, num_heads, head_size)
    host = {
        f"layer.{i}": torch.randn(shape, dtype=dtype, device="cpu")
        for i in range(num_layers)
    }
    device_caches = {
        f"layer.{i}": torch.zeros(shape, dtype=dtype, device=device)
        for i in range(num_layers)
    }
    return host, device_caches


def _split_into_groups(block_ids: list[int], num_groups: int) -> list[list[int]]:
    """Partition block ids into groups, mirroring the global BlockPool id space."""
    per_group = len(block_ids) // num_groups
    return [block_ids[i * per_group : (i + 1) * per_group] for i in range(num_groups)]


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * q))]


@torch.inference_mode()
def _time_paired(
    host: dict[str, torch.Tensor],
    device_caches: dict[str, torch.Tensor],
    group_block_ids: list[list[int]],
    direction: str,
    num_iters: int,
) -> tuple[list[float], list[float]]:
    """Per-group and coalesced latencies, measured alternately.

    Interleaving the two variants keeps drift in machine state from landing on
    one of them: each iteration times both, back to back.
    """
    src, dst = (host, device_caches) if direction == "h2d" else (device_caches, host)
    merged = [[b for group in group_block_ids for b in group]]

    def _run(batches: list[list[int]]) -> float:
        start = time.perf_counter()
        for ids in batches:
            copy_kv_blocks(src, dst, ids, ids, direction)
        torch.accelerator.synchronize()
        return time.perf_counter() - start

    for _ in range(5):  # warmup
        _run(group_block_ids)
        _run(merged)

    per_group: list[float] = []
    coalesced: list[float] = []
    for i in range(num_iters):
        # Alternate which variant goes first so ordering cannot favour either.
        if i % 2:
            per_group.append(_run(group_block_ids))
            coalesced.append(_run(merged))
        else:
            coalesced.append(_run(merged))
            per_group.append(_run(group_block_ids))
    return sorted(per_group), sorted(coalesced)


@torch.inference_mode()
def _count_device_ops(
    host: dict[str, torch.Tensor],
    device_caches: dict[str, torch.Tensor],
    group_block_ids: list[list[int]],
    direction: str,
    coalesced: bool,
) -> dict[str, int]:
    """Device-op counts for one request-sync, via the profiler.

    Counts do not depend on how busy the machine is, so this is the part of
    the comparison that reproduces exactly.
    """
    from torch.profiler import ProfilerActivity, profile

    src, dst = (host, device_caches) if direction == "h2d" else (device_caches, host)
    batches = (
        [[b for group in group_block_ids for b in group]]
        if coalesced
        else group_block_ids
    )

    for ids in batches:  # warm up allocator/autograd caches
        copy_kv_blocks(src, dst, ids, ids, direction)
    torch.accelerator.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for ids in batches:
            copy_kv_blocks(src, dst, ids, ids, direction)
        torch.accelerator.synchronize()

    counts: dict[str, int] = {}
    for evt in prof.key_averages():
        if evt.count and (evt.self_device_time_total or evt.device_time_total):
            counts[evt.key] = counts.get(evt.key, 0) + evt.count
    return counts


def _report_op_counts(host, device_caches, block_ids, groups):
    print("\nDevice-op counts per request-sync (profiler; independent of load):")
    rows = []
    for num_groups in groups:
        group_block_ids = _split_into_groups(block_ids, num_groups)
        for direction in ("h2d", "d2h"):
            before = _count_device_ops(
                host, device_caches, group_block_ids, direction, coalesced=False
            )
            after = _count_device_ops(
                host, device_caches, group_block_ids, direction, coalesced=True
            )
            for key in sorted(set(before) | set(after)):
                b, a = before.get(key, 0), after.get(key, 0)
                if b != a:
                    rows.append([num_groups, direction, key, b, a])
    print(
        tabulate(
            rows,
            headers=["groups", "direction", "op", "per-group", "coalesced"],
        )
    )


def main(args):
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]
    torch.manual_seed(args.seed)

    host, device_caches = _make_caches(
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_heads=args.num_heads,
        head_size=args.head_size,
        dtype=dtype,
        device="cuda",
    )
    block_ids = list(range(args.blocks_per_request))

    rows = []
    for num_groups in args.groups:
        group_block_ids = _split_into_groups(block_ids, num_groups)
        for direction in ("h2d", "d2h"):
            per_group, coalesced = _time_paired(
                host,
                device_caches,
                group_block_ids,
                direction,
                num_iters=args.iters,
            )
            # Interference only ever adds time, so the minimum is the most
            # stable estimator here; the p10 spread above it says how noisy
            # the machine was while sampling.
            best_before = per_group[0]
            best_after = coalesced[0]
            spread = (_percentile(coalesced, 0.5) - best_after) / best_after
            rows.append(
                [
                    num_groups,
                    direction,
                    best_before * 1e6,
                    best_after * 1e6,
                    (best_before - best_after) / best_before * 100.0,
                    spread * 100.0,
                ]
            )

    print(
        f"layers={args.num_layers} num_blocks={args.num_blocks} "
        f"block_size={args.block_size} heads={args.num_heads} "
        f"head_size={args.head_size} dtype={args.dtype} "
        f"blocks/request={args.blocks_per_request} iters={args.iters}"
    )
    print(
        tabulate(
            rows,
            headers=[
                "groups",
                "direction",
                "per-group min (µs)",
                "coalesced min (µs)",
                "saved (%)",
                "p50 over min (%)",
            ],
            floatfmt=".3f",
        )
    )

    if args.op_counts:
        _report_op_counts(host, device_caches, block_ids, args.groups)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--blocks-per-request", type=int, default=60)
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[1, 2, 3, 6],
        help="KV cache group counts to sweep (1 = non-hybrid model).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["half", "bfloat16", "float"],
        default="bfloat16",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--op-counts",
        action="store_true",
        help="Also report profiler op counts, which do not vary with load.",
    )
    parser.add_argument("--seed", type=int, default=0)

    main(parser.parse_args())
