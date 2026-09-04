# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched benchmark for fused RoPE and packed KV-cache update.

Each CUDA-event sample times the caller-owned Q-out operator and the equivalent
``rotary_embedding`` plus ``reshape_and_cache_flash`` composition. The paths
use independent copies of the same packed inputs, and their order is balanced
then shuffled to limit first-run and clock drift bias. Cache views come from
the standard allocator and cover every current ``KVCacheLayout`` by default.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tabulate import tabulate

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
    compute_layout_strides,
    create_kv_cache_views,
)

MAX_POS = 4096
SEED = 0


@dataclass(frozen=True)
class Case:
    q_heads: int
    kv_heads: int
    head_size: int = 128
    rope_tokens: int = 1
    cache_tokens: int = 1


CASES = {
    "mha": Case(32, 32),
    "gqa": Case(32, 8),
    "tinyllama-gqa": Case(32, 4, head_size=64),
    "mqa": Case(32, 1),
    "padded-gqa": Case(32, 8, rope_tokens=8, cache_tokens=5),
}
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
Operation = Callable[[], None]


def _make_cache_views(
    layout: KVCacheLayout,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_layers = 1 if layout.is_layer_compact else 2
    cache_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    physical_cache = torch.zeros(
        num_layers * num_blocks * cache_spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    )
    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        cache_spec, num_blocks, num_layers, layout
    )
    cache_tensor = KVCacheTensor(
        size=physical_cache.numel(),
        layers=[str(index) for index in range(num_layers)],
        layer_stride=layer_stride,
        block_stride=block_stride,
    )
    packed_cache = create_kv_cache_views(
        physical_cache,
        cache_spec,
        num_blocks,
        layout,
        cache_tensor,
    )[-1]

    key_cache, value_cache = packed_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


def _make_operations(
    case: Case,
    layout: KVCacheLayout,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    is_neox: bool,
    seed: int,
    device: torch.device,
) -> dict[str, Operation]:
    num_slots = num_blocks * block_size
    if case.cache_tokens > num_slots:
        raise ValueError(
            f"Case needs {case.cache_tokens} cache slots, but only "
            f"{num_slots} were allocated"
        )

    set_random_seed(seed)
    packed_qkv = torch.randn(
        case.rope_tokens,
        (case.q_heads + 2 * case.kv_heads) * head_size,
        dtype=dtype,
        device=device,
    )
    angles = torch.randn(
        MAX_POS,
        head_size // 2,
        dtype=torch.float32,
        device=device,
    )
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    positions = torch.randperm(MAX_POS, device=device)[: case.rope_tokens]
    slot_mapping = torch.randperm(num_slots, device=device)[: case.cache_tokens]
    scale = torch.ones(1, dtype=torch.float32, device=device)

    def make_state() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        qkv = packed_qkv.clone()
        query_flat, key_flat, value_flat = qkv.split(
            [
                case.q_heads * head_size,
                case.kv_heads * head_size,
                case.kv_heads * head_size,
            ],
            dim=-1,
        )
        query = query_flat.view(case.rope_tokens, case.q_heads, head_size)
        key = key_flat.view(case.rope_tokens, case.kv_heads, head_size)
        value = value_flat.view(case.rope_tokens, case.kv_heads, head_size)
        key_cache, value_cache = _make_cache_views(
            layout,
            num_blocks,
            block_size,
            case.kv_heads,
            head_size,
            dtype,
            device,
        )
        return query, key, value, key_cache, value_cache

    (
        q_out_query,
        q_out_key,
        q_out_value,
        q_out_key_cache,
        q_out_value_cache,
    ) = make_state()
    (
        unfused_query,
        unfused_key,
        unfused_value,
        unfused_key_cache,
        unfused_value_cache,
    ) = make_state()

    q_out_buffer = torch.empty(
        case.rope_tokens,
        case.q_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    def q_out() -> None:
        ops.fused_rope_and_reshape_cache_flash_q_out(
            q_out_query,
            q_out_key,
            q_out_value,
            q_out_buffer,
            positions,
            cos_sin_cache,
            is_neox,
            q_out_key_cache,
            q_out_value_cache,
            slot_mapping,
        )

    def unfused() -> None:
        ops.rotary_embedding(
            positions,
            unfused_query,
            unfused_key,
            head_size,
            cos_sin_cache,
            is_neox,
        )
        ops.reshape_and_cache_flash(
            unfused_key,
            unfused_value,
            unfused_key_cache,
            unfused_value_cache,
            slot_mapping,
            "auto",
            scale,
            scale,
        )

    return {
        "q_out": q_out,
        "unfused": unfused,
    }


_ORDER_CYCLE = tuple(itertools.permutations(("q_out", "unfused")))


def _balanced_orders(count: int, rng: random.Random) -> list[tuple[str, ...]]:
    offset = rng.randrange(len(_ORDER_CYCLE))
    orders = [
        _ORDER_CYCLE[(offset + index) % len(_ORDER_CYCLE)] for index in range(count)
    ]
    rng.shuffle(orders)
    return orders


def _measure_arms(
    operations: dict[str, Operation],
    warmup: int,
    samples: int,
    repeats: int,
    rng: random.Random,
) -> tuple[list[str], dict[str, list[float]]]:
    for order in _balanced_orders(warmup, rng):
        for name in order:
            for _ in range(repeats):
                operations[name]()
    torch.accelerator.synchronize()

    records = []
    sample_orders = _balanced_orders(samples, rng)
    for order in sample_orders:
        record = {}
        for name in order:
            start = torch.Event(enable_timing=True)
            end = torch.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                operations[name]()
            end.record()
            record[name] = (start, end)
        records.append(record)
    torch.accelerator.synchronize()

    samples_us = {name: [] for name in operations}
    for record in records:
        for name, (start, end) in record.items():
            elapsed_us = start.elapsed_time(end) * 1000.0 / repeats
            if elapsed_us <= 0:
                raise RuntimeError(
                    "CUDA event resolution was too low; increase --repeats"
                )
            samples_us[name].append(elapsed_us)
    return ["->".join(order) for order in sample_orders], samples_us


def _summary(samples: list[float]) -> str:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return (
        f"{statistics.median(samples):.3f} "
        f"[{percentile(0.1):.3f}, {percentile(0.9):.3f}]"
    )


@torch.inference_mode()
def main(args) -> None:
    if not torch.accelerator.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.samples < 1 or args.repeats < 1:
        raise ValueError("--samples and --repeats must be positive")
    if args.num_blocks < 1:
        raise ValueError("--num-blocks must be positive")

    device = torch.device("cuda")
    rng = random.Random(args.seed)
    rows = []
    raw_results = []
    for case_name in args.cases:
        case = CASES[case_name]
        head_size = args.head_size or case.head_size
        for layout_name in args.layouts:
            layout = KVCacheLayout[layout_name]
            operations = _make_operations(
                case,
                layout,
                head_size,
                args.block_size,
                args.num_blocks,
                DTYPES[args.dtype],
                args.rope_style == "neox",
                args.seed,
                device,
            )
            orders, samples_us = _measure_arms(
                operations,
                args.warmup,
                args.samples,
                args.repeats,
                rng,
            )
            versus_unfused = [
                unfused / q_out
                for q_out, unfused in zip(samples_us["q_out"], samples_us["unfused"])
            ]
            rows.append(
                [
                    case_name,
                    layout.name,
                    case.rope_tokens,
                    case.cache_tokens,
                    f"{case.q_heads}/{case.kv_heads}",
                    head_size,
                    _summary(samples_us["unfused"]),
                    _summary(samples_us["q_out"]),
                    _summary(versus_unfused),
                ]
            )
            raw_results.append(
                {
                    "case": case_name,
                    "layout": layout.name,
                    "head_size": head_size,
                    "orders": orders,
                    "q_out_us": samples_us["q_out"],
                    "unfused_us": samples_us["unfused"],
                }
            )

    print(
        f"Device: {current_platform.get_device_name()}\n"
        f"dtype={args.dtype} head_size={args.head_size or 'per-case'} "
        f"rope={args.rope_style} "
        f"warmup={args.warmup} samples={args.samples} "
        f"repeats/sample={args.repeats}"
    )
    print(
        tabulate(
            rows,
            headers=[
                "case",
                "layout",
                "T",
                "slots",
                "Q/KV heads",
                "D",
                "unfused median [p10, p90] (us)",
                "Q-out fused median [p10, p90] (us)",
                "Q-out vs unfused [p10, p90]",
            ],
        )
    )
    if args.raw_json:
        print("\nRaw paired samples (microseconds per invocation):")
        for result in raw_results:
            print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=[layout.name for layout in KVCacheLayout],
        default=[layout.name for layout in KVCacheLayout],
    )
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument(
        "--head-size",
        type=int,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        default=None,
        help="Override the per-case head size.",
    )
    parser.add_argument("--rope-style", choices=["neox", "interleaved"], default="neox")
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print paired per-sample timings after the summary table.",
    )
    main(parser.parse_args())
