# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark collective versus direct final PCP hidden-state restoration.

Run on one peer-connected four-GPU node:

    torchrun --standalone --nproc-per-node=4 \
      benchmarks/kernels/bench_pcp_hidden_restore.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

from vllm.v1.worker.gpu.pcp_hidden_restore import PCPHiddenStateRestorer


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _measure_pair_us(
    first_operation: Callable[[], None],
    second_operation: Callable[[], None],
    *,
    cpu_group: dist.ProcessGroup,
    warmup: int,
    iterations: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    for _ in range(warmup):
        first_operation()
        second_operation()
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    first_times = []
    second_times = []
    for step in range(iterations):
        dist.barrier(group=cpu_group)
        operations = (
            (first_operation, first_times),
            (second_operation, second_times),
        )
        if step & 1:
            operations = operations[::-1]
        for operation, times in operations:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)

    def aggregate(local_times: list[float]) -> tuple[float, float]:
        all_times: list[list[float] | None] = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(all_times, local_times, group=cpu_group)
        step_maxima = [
            max(rank_times[step] for rank_times in all_times if rank_times is not None)
            for step in range(iterations)
        ]
        return statistics.median(step_maxima), _percentile(step_maxima, 0.95)

    return aggregate(first_times), aggregate(second_times)


def _run_case(
    *,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    cpu_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    rank = dist.get_rank(cpu_group)
    world_size = dist.get_world_size(cpu_group)
    device = torch.device(f"cuda:{rank}")
    padded_tokens = (num_tokens + world_size - 1) // world_size

    local_global_rows = rank + world_size * torch.arange(padded_tokens, device=device)
    local_global_rows[local_global_rows >= num_tokens] = -1
    hidden_states = torch.randn(
        (padded_tokens, hidden_size),
        dtype=dtype,
        device=device,
    )

    gathered = torch.empty(
        (world_size * padded_tokens, hidden_size),
        dtype=dtype,
        device=device,
    )
    restore_idx = torch.empty(num_tokens, dtype=torch.int64, device=device)
    global_rows = torch.arange(num_tokens, device=device)
    restore_idx.copy_(
        global_rows.remainder(world_size) * padded_tokens
        + torch.div(global_rows, world_size, rounding_mode="floor")
    )
    collective_output = torch.empty(
        (num_tokens, hidden_size),
        dtype=dtype,
        device=device,
    )

    def collective_restore() -> None:
        dist.all_gather_into_tensor(gathered, hidden_states, group=nccl_group)
        torch.index_select(gathered, 0, restore_idx, out=collective_output)

    restorer = PCPHiddenStateRestorer(
        group=cpu_group,
        device=device,
        max_num_tokens=num_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
    )

    def direct_restore() -> None:
        restorer.restore(
            hidden_states,
            local_global_rows,
            num_global_tokens=num_tokens,
        )

    collective_restore()
    direct_restore()
    torch.cuda.synchronize()
    if not torch.equal(collective_output, restorer.local_output[:num_tokens]):
        raise AssertionError("Direct and collective restore outputs differ.")

    (collective_median, collective_p95), (direct_median, direct_p95) = _measure_pair_us(
        collective_restore,
        direct_restore,
        cpu_group=cpu_group,
        warmup=warmup,
        iterations=iterations,
    )
    restorer.close()

    return {
        "tokens": num_tokens,
        "hidden_size": hidden_size,
        "dtype": str(dtype).removeprefix("torch."),
        "collective_median_us": collective_median,
        "collective_p95_us": collective_p95,
        "direct_median_us": direct_median,
        "direct_p95_us": direct_p95,
        "speedup": collective_median / direct_median,
        "latency_reduction_pct": 100
        * (collective_median - direct_median)
        / collective_median,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,16,64,256,1024,4096")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    cpu_group = dist.group.WORLD
    nccl_group = dist.new_group(backend="nccl")
    try:
        if dist.get_world_size() != 4:
            raise ValueError("This benchmark currently requires PCP=4.")
        dtype = getattr(torch, args.dtype)
        rows = [
            _run_case(
                num_tokens=num_tokens,
                hidden_size=args.hidden_size,
                dtype=dtype,
                cpu_group=cpu_group,
                nccl_group=nccl_group,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for num_tokens in map(int, args.tokens.split(","))
        ]
        if rank == 0:
            payload = {
                "world_size": dist.get_world_size(),
                "device": torch.cuda.get_device_name(),
                "warmup": args.warmup,
                "iterations": args.iterations,
                "results": rows,
            }
            rendered = json.dumps(payload, indent=2)
            print(rendered)
            if args.output_json:
                with open(args.output_json, "w") as output:
                    output.write(rendered + "\n")
    finally:
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
