# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the complete direct DCP Output/LSE operation under CUDA graphs.

Example:
    torchrun --nproc-per-node=4 benchmarks/kernels/benchmark_dcp_a2a_lse_reduce.py \
        --heads 64 --rows 1,8,32,64,128 --output result.json
Run unchanged on the base and candidate revisions, with exclusive GPU ownership.

"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from vllm.v1.attention.ops.dcp import DirectDCPA2AWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--rows", default="1,8,32,64,128")
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [int(value) for value in args.rows.split(",")]
    if min(rows) <= 0 or min(args.trials, args.replays, args.unroll) <= 0:
        parser.error("rows and timing counts must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.accelerator.set_device_index(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        world, rank = dist.get_world_size(), dist.get_rank()
        if world <= 1 or args.heads <= 0 or args.heads % world:
            raise ValueError("heads must divide evenly across multiple DCP ranks")
        workspace = DirectDCPA2AWorkspace(
            dist.group.WORLD,
            device,
            max(rows),
            args.heads // world,
            512,
            torch.bfloat16,
        )
        records = []
        for num_rows in rows:
            torch.manual_seed(100 + rank)
            partial_output = torch.randn(
                num_rows, args.heads, 512, device=device, dtype=torch.bfloat16
            )
            partial_lse = torch.randn(num_rows, args.heads, device=device)
            workspace.lse_reduce(partial_output, partial_lse, False)
            torch.accelerator.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.unroll):
                    workspace.lse_reduce(partial_output, partial_lse, False)
            for _ in range(10):
                graph.replay()
            torch.accelerator.synchronize()
            samples = []
            for _ in range(args.trials):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / (args.replays * args.unroll),
                    device=device,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples.append(elapsed.item())
            record = {
                "rows": num_rows,
                "median_us": statistics.median(samples),
                "samples_us": samples,
            }
            records.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
        if rank == 0 and args.output:
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": world,
                        "heads": args.heads,
                        "head_dim": 512,
                        "torch_version": torch.__version__,
                        "trials": args.trials,
                        "replays": args.replays,
                        "unroll": args.unroll,
                        "records": records,
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
