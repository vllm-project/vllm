# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep the ROCm sparse MLA decode chain (split-K partial + reduce).

The split count the decode picks is shared by every sparse MLA consumer, so
judging it needs a per-shape picture rather than a single representative point.

A shape here is (heads per rank, decode rows, SWA length, topk length, cache
block sizes). Decode rows are ``concurrency * query_len``, where query_len is 1
without speculative decoding and up to 1 + spec tokens with it, so the sweep
walks concurrency and query_len separately even though the kernel only sees the
product.

Usage:
    python benchmarks/kernels/benchmark_rocm_dsv4_sparse_decode.py
    python benchmarks/kernels/benchmark_rocm_dsv4_sparse_decode.py \
        --profile dsv41_flash --heads 32 --concurrency 1,8,32 --query-len 1,4
    python benchmarks/kernels/benchmark_rocm_dsv4_sparse_decode.py \
        --sweep-splits 1,2,3,4,6,8,16 --csv splits.csv
"""

import argparse
import contextlib
import csv
import itertools
import json
from dataclasses import dataclass, field

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM


@dataclass(frozen=True)
class Profile:
    """Segment and cache geometry a model presents to the decode kernel."""

    name: str
    swa_len: int
    topk_len: int
    main_block_size: int
    extra_block_size: int
    heads: tuple[int, ...]


# Sliding window is 128 for both models; topk width and the SWA cache block size
# are what differ. Head counts are per rank: V4.1-Flash has 64 query heads and
# V4-Pro 128, so the listed values cover TP=2 and TP=4 for Flash and the TP=4
# and TP=8 shapes of Pro that land on the same head counts.
PROFILES = {
    "dsv41_flash": Profile("dsv41_flash", 128, 512, 32, 128, (32, 16)),
    "dsv4_pro": Profile("dsv4_pro", 128, 1024, 16, 128, (32, 16)),
}


@dataclass
class Case:
    profile: Profile
    heads: int
    concurrency: int
    query_len: int

    @property
    def rows(self) -> int:
        return self.concurrency * self.query_len


@dataclass
class Inputs:
    q: torch.Tensor
    main_cache: torch.Tensor
    main_indices: torch.Tensor
    main_indptr: torch.Tensor
    extra_cache: torch.Tensor
    extra_indices: torch.Tensor
    extra_indptr: torch.Tensor
    scale: float
    kwargs: dict = field(default_factory=dict)


def _pack_cache(kv: torch.Tensor, block_size: int, use_fnuz: bool) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros(
        (num_blocks, block_size, 584), dtype=torch.uint8, device=kv.device
    )
    quantize_and_insert_k_cache(
        kv,
        cache,
        torch.arange(num_tokens, dtype=torch.int64, device=kv.device),
        block_size=block_size,
        use_fnuz=use_fnuz,
    )
    return cache


def _ragged(
    rows: int, per_row: int, pool: int, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ragged (indices, indptr) with per_row random slots drawn from pool."""
    indices = torch.randint(
        0,
        pool,
        (rows * per_row,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    indptr = torch.arange(0, rows * per_row + 1, per_row, dtype=torch.int32)
    return indices, indptr.to(device)


def build_inputs(case: Case, device: torch.device) -> Inputs:
    gen = torch.Generator(device=device).manual_seed(0)
    rows = case.rows
    prof = case.profile
    use_fnuz = current_platform.is_fp8_fnuz()

    q = (
        torch.randn(
            rows,
            case.heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        * 0.125
    )

    # Distinct KV pools sized so the gather is not served entirely from L2.
    main_pool = max(prof.swa_len * 4, rows * prof.swa_len // 4, 1024)
    extra_pool = max(prof.topk_len * 4, rows * prof.topk_len // 4, 4096)
    main_kv = (
        torch.randn(
            main_pool, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.125
    )
    extra_kv = (
        torch.randn(
            extra_pool, HEAD_DIM, dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.125
    )

    main_indices, main_indptr = _ragged(rows, prof.swa_len, main_pool, device, gen)
    extra_indices, extra_indptr = _ragged(rows, prof.topk_len, extra_pool, device, gen)

    return Inputs(
        q=q,
        main_cache=_pack_cache(main_kv, prof.main_block_size, use_fnuz),
        main_indices=main_indices,
        main_indptr=main_indptr,
        extra_cache=_pack_cache(extra_kv, prof.extra_block_size, use_fnuz=False),
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        scale=HEAD_DIM**-0.5,
    )


@contextlib.contextmanager
def forced_splits(splits: int | None):
    """Pin the split count, bypassing the split heuristic.

    The heuristic trades per-workgroup work against occupancy, so measuring a
    shape at every split count is what says whether its choice was right.
    """
    if splits is None:
        yield
        return
    originals = (sparse._decode_gfx950_num_splits, sparse._decode_num_splits)
    sparse._decode_gfx950_num_splits = lambda *a, **k: splits
    sparse._decode_num_splits = lambda *a, **k: splits
    try:
        yield
    finally:
        sparse._decode_gfx950_num_splits, sparse._decode_num_splits = originals


def run_decode(inputs: Inputs, adaptive_splits: bool) -> torch.Tensor:
    return sparse._rocm_sparse_attn_decode_ragged_triton(
        q=inputs.q,
        main_cache=inputs.main_cache,
        main_indices=inputs.main_indices,
        main_indptr=inputs.main_indptr,
        scale=inputs.scale,
        attn_sink=None,
        nope_head_dim=NOPE_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        extra_cache=inputs.extra_cache,
        extra_indices=inputs.extra_indices,
        extra_indptr=inputs.extra_indptr,
        adaptive_splits=adaptive_splits,
    )


def bytes_moved(case: Case) -> int:
    """KV bytes the decode must read at minimum: 576 B per fp8_ds_mla token."""
    return case.rows * (case.profile.swa_len + case.profile.topk_len) * 576


def graph_replay_time_us(inputs: Inputs, adaptive_splits: bool) -> float:
    """Device time with the launcher's host work captured away.

    Production captures decode into CUDA graphs, so eager wrapper time and
    replay time answer different questions: at low row counts the chain is
    launch-bound and only replay reflects what the kernels cost.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_decode(inputs, adaptive_splits)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_decode(inputs, adaptive_splits)
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    return triton.testing.do_bench(graph.replay, warmup=25, rep=100) * 1e3


def measure(
    case: Case,
    device: torch.device,
    adaptive_splits: bool,
    inputs: Inputs | None = None,
) -> dict:
    if inputs is None:
        inputs = build_inputs(case, device)
    run_decode(inputs, adaptive_splits)  # compile before timing
    torch.accelerator.synchronize()
    eager_us = (
        triton.testing.do_bench(
            lambda: run_decode(inputs, adaptive_splits), warmup=25, rep=100
        )
        * 1e3
    )
    replay_us = graph_replay_time_us(inputs, adaptive_splits)

    heads_blocks = -(-case.heads // 16)
    splits = sparse._decode_gfx950_num_splits(
        case.rows,
        heads_blocks,
        float(case.profile.swa_len),
        float(case.profile.topk_len),
    )
    return {
        "profile": case.profile.name,
        "heads": case.heads,
        "conc": case.concurrency,
        "qlen": case.query_len,
        "rows": case.rows,
        "splits": splits,
        "eager_us": eager_us,
        "replay_us": replay_us,
        "gbps": bytes_moved(case) / (replay_us * 1e-6) / 1e9,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p.add_argument("--heads", type=str, default=None, help="comma list, per rank")
    p.add_argument("--concurrency", type=str, default="1,2,4,8,16,32,64,128")
    p.add_argument("--query-len", type=str, default="1,2,3,4,5,6")
    p.add_argument("--adaptive-splits", action="store_true")
    p.add_argument(
        "--sweep-splits",
        type=str,
        default=None,
        metavar="V1,V2",
        help="search the split count against the heuristic's choice",
    )
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--json", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not current_platform.is_rocm():
        print("ROCm only")
        return 1

    device = torch.device("cuda")
    profiles = (
        list(PROFILES.values()) if args.profile == "all" else [PROFILES[args.profile]]
    )
    concs = [int(v) for v in args.concurrency.split(",")]
    qlens = [int(v) for v in args.query_len.split(",")]
    split_candidates = (
        [int(v) for v in args.sweep_splits.split(",")] if args.sweep_splits else []
    )

    rows_out = []
    if split_candidates:
        header = (
            f"{'profile':>12}{'heads':>7}{'conc':>6}{'qlen':>6}{'rows':>6}"
            f"{'heur':>6}{'heur us':>9}{'best':>6}{'best us':>9}{'gain':>7}"
        )
    else:
        header = (
            f"{'profile':>12}{'heads':>7}{'conc':>6}{'qlen':>6}{'rows':>6}"
            f"{'splits':>8}{'eager us':>10}{'replay us':>11}{'GB/s':>9}"
        )
    print(header)
    print("-" * len(header))
    for prof in profiles:
        heads = (
            [int(v) for v in args.heads.split(",")] if args.heads else list(prof.heads)
        )
        for h, conc, qlen in itertools.product(heads, concs, qlens):
            case = Case(prof, h, conc, qlen)
            inputs = build_inputs(case, device)
            r = measure(case, device, args.adaptive_splits, inputs)

            if not split_candidates:
                rows_out.append(r)
                print(
                    f"{r['profile']:>12}{r['heads']:>7}{r['conc']:>6}{r['qlen']:>6}"
                    f"{r['rows']:>6}{r['splits']:>8}{r['eager_us']:>10.2f}"
                    f"{r['replay_us']:>11.2f}{r['gbps']:>9.1f}"
                )
                continue

            times = {r["splits"]: r["replay_us"]}
            for s in split_candidates:
                with forced_splits(s):
                    run_decode(inputs, args.adaptive_splits)  # compile
                    times[s] = graph_replay_time_us(inputs, args.adaptive_splits)
            best = min(times, key=times.get)
            r["best_splits"] = best
            r["best_us"] = times[best]
            r["per_split_us"] = {s: round(t, 2) for s, t in sorted(times.items())}
            rows_out.append(r)
            print(
                f"{r['profile']:>12}{r['heads']:>7}{r['conc']:>6}{r['qlen']:>6}"
                f"{r['rows']:>6}{r['splits']:>6}{r['replay_us']:>9.2f}"
                f"{best:>6}{times[best]:>9.2f}{r['replay_us'] / times[best]:>6.2f}x"
            )

    if args.csv and rows_out:
        fields = list(dict.fromkeys(k for r in rows_out for k in r))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows_out:
                row = dict(r)
                if "per_split_us" in row:
                    row["per_split_us"] = json.dumps(row["per_split_us"])
                writer.writerow(row)
        print(f"\nwrote {args.csv}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows_out, f, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
