# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

SCENARIOS = {
    "single-60k": [60_000],
    "single-300k": [300_000],
    "skew-2": [60_000, 300_000],
    "skew-4": [60_000, 100_000, 180_000, 300_000],
    "skew-8": [
        60_000,
        60_000,
        80_000,
        100_000,
        140_000,
        180_000,
        240_000,
        300_000,
    ],
}
DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def make_page_table(
    seq_lens: list[int],
    block_size: int,
    seq_starts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if seq_starts is None:
        seq_starts = [0] * len(seq_lens)
    blocks_per_req = [
        math.ceil((start + length) / block_size)
        for start, length in zip(seq_starts, seq_lens)
    ]
    total_blocks = sum(blocks_per_req)
    block_table = torch.zeros(
        (len(seq_lens), max(blocks_per_req)), dtype=torch.int32, device="cuda"
    )
    physical_blocks = torch.randperm(total_blocks, dtype=torch.int32, device="cuda")
    block_offset = 0
    for req_id, num_blocks in enumerate(blocks_per_req):
        block_table[req_id, :num_blocks] = physical_blocks[
            block_offset : block_offset + num_blocks
        ]
        block_offset += num_blocks

    cu_seq_lens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device="cuda")
    cu_seq_lens[1:] = torch.tensor(seq_lens, dtype=torch.int32, device="cuda").cumsum(
        dim=0
    )
    return block_table, cu_seq_lens, total_blocks


def make_cache_gather(
    seq_lens: list[int],
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
) -> tuple[Callable[[], None], int]:
    seq_starts_list = [
        13 + (17 * req_id) % block_size for req_id in range(len(seq_lens))
    ]
    block_table, cu_seq_lens, total_blocks = make_page_table(
        seq_lens, block_size, seq_starts_list
    )
    src_cache = torch.empty(
        (total_blocks, block_size, entry_size), dtype=dtype, device="cuda"
    )
    dst = torch.empty((sum(seq_lens), entry_size), dtype=dtype, device="cuda")
    seq_starts = torch.tensor(seq_starts_list, dtype=torch.int32, device="cuda")

    def run() -> None:
        ops.cp_gather_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            len(seq_lens),
            seq_starts,
        )

    bytes_moved = 2 * dst.numel() * dst.element_size()
    return run, bytes_moved


def make_fp8_upconvert(
    seq_lens: list[int],
    block_size: int,
) -> tuple[Callable[[], None], int]:
    entry_bytes = 656
    output_elements = 576
    seq_starts_list = [
        13 + (17 * req_id) % block_size for req_id in range(len(seq_lens))
    ]
    block_table, cu_seq_lens, total_blocks = make_page_table(
        seq_lens, block_size, seq_starts_list
    )
    src_cache = torch.empty(
        (total_blocks, block_size, entry_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    dst = torch.empty(
        (sum(seq_lens), output_elements), dtype=torch.bfloat16, device="cuda"
    )
    seq_starts = torch.tensor(seq_starts_list, dtype=torch.int32, device="cuda")

    def run() -> None:
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens[:-1],
            len(seq_lens),
            seq_starts,
        )

    bytes_moved = sum(seq_lens) * (entry_bytes + output_elements * 2)
    return run, bytes_moved


def make_maybe_dequant_gather(
    seq_lens: list[int],
    block_size: int,
    entry_size: int,
) -> tuple[Callable[[], None], int]:
    seq_starts_list = [
        13 + (17 * req_id) % block_size for req_id in range(len(seq_lens))
    ]
    block_table, cu_seq_lens, total_blocks = make_page_table(
        seq_lens, block_size, seq_starts_list
    )
    src_cache = torch.empty(
        (total_blocks, block_size, entry_size),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    dst = torch.empty((sum(seq_lens), entry_size), dtype=torch.bfloat16, device="cuda")
    token_to_seq = torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device="cuda"),
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
    )
    seq_starts = torch.tensor(seq_starts_list, dtype=torch.int32, device="cuda")
    scale = torch.tensor(0.1, dtype=torch.float32, device="cuda")

    def run() -> None:
        ops.gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            sum(seq_lens),
            "fp8",
            scale,
            seq_starts,
        )

    bytes_moved = sum(seq_lens) * entry_size * 3
    return run, bytes_moved


@torch.inference_mode()
def run_scenario(
    variant: str,
    name: str,
    seq_lens: list[int],
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> None:
    if variant == "cache":
        run, bytes_moved = make_cache_gather(seq_lens, block_size, entry_size, dtype)
    elif variant == "fp8-upconvert":
        run, bytes_moved = make_fp8_upconvert(seq_lens, block_size)
    else:
        run, bytes_moved = make_maybe_dequant_gather(seq_lens, block_size, entry_size)

    latency_ms = triton.testing.do_bench(
        run, warmup=warmup_ms, rep=rep_ms, return_mode="median"
    )
    bandwidth_gbps = bytes_moved / latency_ms / 1e6
    lengths = ",".join(str(seq_len) for seq_len in seq_lens)
    print(
        f"{variant:15s} {name:10s} batch={len(seq_lens):2d} "
        f"total={sum(seq_lens):7d} latency={latency_ms * 1e3:9.2f} us "
        f"bandwidth={bandwidth_gbps:8.1f} GB/s lengths=[{lengths}]"
    )


def main() -> None:
    parser = FlexibleArgumentParser(description="Benchmark cp_gather variants")
    parser.add_argument(
        "--variant",
        choices=["all", "cache", "fp8-upconvert", "maybe-dequant"],
        default="all",
    )
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--dtype", choices=DTYPES, default="fp8")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--entry-size", type=int, default=576)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    variants = (
        ("cache", "fp8-upconvert", "maybe-dequant")
        if args.variant == "all"
        else (args.variant,)
    )
    scenarios = (
        SCENARIOS
        if args.scenario == "all"
        else {args.scenario: SCENARIOS[args.scenario]}
    )
    for variant in variants:
        for name, seq_lens in scenarios.items():
            run_scenario(
                variant,
                name,
                seq_lens,
                args.block_size,
                args.entry_size,
                DTYPES[args.dtype],
                args.warmup_ms,
                args.rep_ms,
            )
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
