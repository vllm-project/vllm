# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FlashInfer and PR #49847 ReplaySSM MTP kernel calls.

Run this same file with the PYTHONPATH of the implementation under test.  The
FlashInfer mode uses native autotuning and can pass PDL explicitly; the
``pr49847`` mode imports that PR's fused scatter/precompute + verify/flush call.
Both modes use the same dense Mamba2 geometry and ring-buffer contents.

Examples::

    python compare_mtp_kernels.py --backend flashinfer --enable-pdl
    python compare_mtp_kernels.py --backend pr49847
    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
      python compare_mtp_kernels.py --backend flashinfer --profile
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["flashinfer", "pr49847"], required=True)
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    parser.add_argument("--spec-lengths", default="2,4,8")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--enable-pdl", action="store_true")
    parser.add_argument(
        "--tune-flashinfer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--l2-flush", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _make_inputs(args: argparse.Namespace, batch: int, spec_len: int) -> dict:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    slots = batch + 1
    ring_len = args.window + spec_len

    A_base = -torch.rand(args.nheads, device=device) - 0.5
    A = A_base[:, None, None].expand(args.nheads, args.head_dim, args.dstate)
    D = torch.randn(args.nheads, device=device, dtype=dtype)[:, None].expand(
        args.nheads, args.head_dim
    )
    dt_bias = torch.randn(args.nheads, device=device, dtype=dtype)[:, None].expand(
        args.nheads, args.head_dim
    )
    dt_base = torch.randn(batch, spec_len, args.nheads, device=device, dtype=dtype)
    dt = dt_base[..., None].expand(batch, spec_len, args.nheads, args.head_dim)
    state_initial = torch.randn(
        slots,
        args.nheads,
        args.head_dim,
        args.dstate,
        device=device,
        dtype=torch.float32,
    )
    x_cache_initial = torch.randn(
        slots,
        args.nheads,
        ring_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    B_cache_initial = torch.randn(
        slots,
        args.ngroups,
        ring_len,
        args.dstate,
        device=device,
        dtype=dtype,
    )
    dt_cache_initial = torch.randn(
        slots,
        args.nheads,
        ring_len,
        device=device,
        dtype=torch.float32,
    ).abs()
    return {
        "state_initial": state_initial,
        "state": state_initial.clone(),
        "x_cache_initial": x_cache_initial,
        "x_cache": x_cache_initial.clone(),
        "B_cache_initial": B_cache_initial,
        "B_cache": B_cache_initial.clone(),
        "dt_cache_initial": dt_cache_initial,
        "dt_cache": dt_cache_initial.clone(),
        "x": torch.randn(
            batch,
            spec_len,
            args.nheads,
            args.head_dim,
            device=device,
            dtype=dtype,
        ),
        "dt": dt,
        "dt_base": dt_base,
        "B": torch.randn(
            batch,
            spec_len,
            args.ngroups,
            args.dstate,
            device=device,
            dtype=dtype,
        ),
        "C": torch.randn(
            batch,
            spec_len,
            args.ngroups,
            args.dstate,
            device=device,
            dtype=dtype,
        ),
        "A": A,
        "D": D,
        "dt_bias": dt_bias,
        "indices": torch.arange(1, slots, device=device, dtype=torch.int32),
        "query_start_loc": torch.arange(
            0, (batch + 1) * spec_len, spec_len, device=device, dtype=torch.int32
        ),
        "ring_len": ring_len,
    }


def _make_case(
    args: argparse.Namespace, batch: int, spec_len: int, flush: bool
) -> tuple[Callable[[], None], Callable[[], None]]:
    tensors = _make_inputs(args, batch, spec_len)
    slots = batch + 1
    pnat = args.window if flush else 0

    if args.backend == "flashinfer":
        from flashinfer.mamba.checkpointing_ssu import (
            allocate_checkpointing_ssu_scratch,
            checkpointing_ssu,
        )

        scratch = allocate_checkpointing_ssu_scratch(
            batch,
            args.nheads,
            spec_len,
            args.window,
            torch.bfloat16,
            "cuda",
        )
        ring_start = torch.zeros(slots, device="cuda", dtype=torch.int32)
        prev = torch.full((slots,), pnat, device="cuda", dtype=torch.int32)
        out = torch.empty_like(tensors["x"])

        def run() -> None:
            checkpointing_ssu(
                tensors["state"],
                tensors["x_cache"],
                tensors["B_cache"],
                tensors["dt_cache"],
                ring_start,
                prev,
                tensors["x"],
                tensors["dt"],
                tensors["A"],
                tensors["B"],
                tensors["C"],
                out,
                D=tensors["D"],
                dt_bias=tensors["dt_bias"],
                dt_softplus=True,
                state_batch_indices=tensors["indices"],
                enable_pdl=args.enable_pdl,
                cb_scaled=scratch[0],
                cumAdt_vec=scratch[1],
                cb_old=scratch[2],
                algorithm="auto",
            )

        def reset() -> None:
            tensors["state"].copy_(tensors["state_initial"])
            tensors["x_cache"].copy_(tensors["x_cache_initial"])
            tensors["B_cache"].copy_(tensors["B_cache_initial"])
            tensors["dt_cache"].copy_(tensors["dt_cache_initial"])
            ring_start.zero_()
            prev.fill_(pnat)

        if args.tune_flashinfer:
            from flashinfer.autotuner import autotune

            with autotune(True):
                run()
            reset()
    else:
        from vllm.model_executor.layers.mamba.ops import (
            selective_state_update_replayssm_spec as spec_ops,
        )

        write_pos = torch.zeros(slots, device="cuda", dtype=torch.int32)
        write_pos[tensors["indices"].long()] = pnat
        post_origin = torch.zeros_like(write_pos)
        is_flush = torch.zeros(slots, device="cuda", dtype=torch.int8)
        is_flush[tensors["indices"].long()] = int(flush)
        block_spec = 1 << (spec_len - 1).bit_length()
        bc_pre = torch.empty(
            batch,
            args.ngroups,
            tensors["ring_len"],
            block_spec,
            device="cuda",
            dtype=torch.float32,
        )
        out = torch.empty(
            batch * spec_len,
            args.nheads,
            args.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def run() -> None:
            spec_ops.selective_state_update_replayssm_spec(
                tensors["state"],
                tensors["x_cache"],
                tensors["dt_cache"],
                tensors["B_cache"],
                tensors["x"].flatten(0, 1),
                tensors["dt_base"].flatten(0, 1),
                tensors["B"].flatten(0, 1),
                tensors["C"].flatten(0, 1),
                tensors["A"],
                write_pos=write_pos,
                post_conv_state_pos=post_origin,
                is_flush=is_flush,
                query_start_loc=tensors["query_start_loc"],
                state_batch_indices=tensors["indices"],
                max_cache_len=tensors["ring_len"],
                spec_query_len=spec_len,
                D=tensors["D"],
                dt_bias=tensors["dt_bias"][:, 0],
                dt_softplus=True,
                out=out,
                bc_pre=bc_pre,
                null_block_id=0,
            )

        def reset() -> None:
            tensors["state"].copy_(tensors["state_initial"])
            tensors["x_cache"].copy_(tensors["x_cache_initial"])
            tensors["B_cache"].copy_(tensors["B_cache_initial"])
            tensors["dt_cache"].copy_(tensors["dt_cache_initial"])
            write_pos.zero_()
            write_pos[tensors["indices"].long()] = pnat
            post_origin.zero_()
            is_flush.zero_()
            is_flush[tensors["indices"].long()] = int(flush)

    return run, reset


def _time_case(
    args: argparse.Namespace, batch: int, spec_len: int, flush: bool
) -> dict:
    run, reset = _make_case(args, batch, spec_len, flush)
    l2 = (
        torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.float32)
        if args.l2_flush
        else None
    )
    for _ in range(args.warmup):
        reset()
        run()
    torch.accelerator.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    label = f"{args.backend}/b{batch}/t{spec_len}/{'flush' if flush else 'verify'}"
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for start, end in zip(starts, ends):
        reset()
        if l2 is not None:
            l2.zero_()
        start.record()
        torch.cuda.nvtx.range_push(label)
        run()
        torch.cuda.nvtx.range_pop()
        end.record()
    torch.accelerator.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    times_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    result = {
        "backend": args.backend,
        "batch": batch,
        "spec_len": spec_len,
        "path": "flush" if flush else "verify",
        "enable_pdl": args.enable_pdl,
        "l2_flush": args.l2_flush,
        "median_us": statistics.median(times_us),
        "p95_us": _percentile(times_us, 0.95),
        "p99_us": _percentile(times_us, 0.99),
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.enable_pdl and args.backend != "flashinfer":
        raise ValueError("--enable-pdl is only valid for the FlashInfer backend")
    spec_lengths = _csv_ints(args.spec_lengths)
    if max(spec_lengths) > args.window:
        raise ValueError("spec length must not exceed the logical replay window")
    results = [
        _time_case(args, batch, spec_len, flush)
        for batch in _csv_ints(args.batch_sizes)
        for spec_len in spec_lengths
        for flush in (False, True)
    ]
    payload = {"args": vars(args), "results": results}
    print("RESULT_JSON " + json.dumps(payload), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
