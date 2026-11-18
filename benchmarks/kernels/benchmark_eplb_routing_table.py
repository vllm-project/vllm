# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare hash and table lookup inside the same EPLB map-and-record kernel.

The table reproduces the hash exactly for power-of-two replica counts dividing
1024. This isolates lookup cost from a change in replica selection or atomic
contention. Table construction, allocation and compilation are outside timing.
Graph replay measures a warm working set; --tables rotates independent layer
maps/tables to expose the larger table working set. This is not a serving test.

Example:
    python -m benchmarks.kernels.benchmark_eplb_routing_table \
        --device npu --tokens 1 32 128 1024 4096 --blocks 128 256 512

"""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.eplb_triton import (
    _eplb_map_and_record_i32_kernel,
)
from vllm.triton_utils import triton


def benchmark(args):
    if args.device == "npu":
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    runtime = torch.get_device_module(device)
    graph_type = runtime.NPUGraph if args.device == "npu" else runtime.CUDAGraph
    generator = torch.Generator().manual_seed(0)
    order = random.Random(0)
    experts, topk, rows = args.experts, args.topk, 1024
    redundant = args.redundant_experts
    num_physical = experts + redundant
    replicas = torch.ones(experts, dtype=torch.int64)
    replicas[:redundant] = 2
    mapping = torch.full((experts, redundant + 1), -1, dtype=torch.int64)
    mapping[:, 0] = torch.arange(experts)
    if redundant:
        mapping[:redundant, 1] = torch.arange(experts, num_physical)
    hashed_rows = (torch.arange(rows, dtype=torch.int64) * 2654435769) & 0xFFFFFFFF
    table = mapping.gather(1, (hashed_rows[:, None] % replicas).T).T.int()
    mappings = [mapping.to(device).clone() for _ in range(args.tables)]
    tables = [table.to(device).contiguous().clone() for _ in range(args.tables)]
    replicas = replicas.to(device)
    results = []
    for tokens in args.tokens:
        ids_cpu = torch.randint(experts, (tokens, topk), generator=generator)
        hashed = (torch.arange(tokens, dtype=torch.int64) * 2654435769) & 0xFFFFFFFF
        expected = mapping[ids_cpu, hashed[:, None] % replicas.cpu()[ids_cpu]]
        ids = ids_cpu.to(device=device, dtype=torch.int32)
        real_tokens = torch.tensor(tokens, device=device, dtype=torch.int32)
        for recording in (False, True):
            gate = torch.tensor(recording, device=device)
            expected_load = torch.bincount(expected.flatten(), minlength=num_physical)
            if not recording:
                expected_load.zero_()
            for block in args.blocks:
                arms = {}
                for name in ("hash", "table"):
                    out = torch.empty_like(ids)
                    load = torch.zeros(num_physical, dtype=torch.int32, device=device)

                    def launch(
                        index=0,
                        ids=ids,
                        block=block,
                        out=out,
                        load=load,
                        gate=gate,
                        real_tokens=real_tokens,
                        use_table=name == "table",
                    ):
                        _eplb_map_and_record_i32_kernel[
                            (triton.cdiv(ids.numel(), block),)
                        ](
                            ids,
                            replicas,
                            mappings[index],
                            out,
                            load,
                            gate,
                            real_tokens,
                            experts,
                            mapping.shape[1],
                            num_physical,
                            ids.numel(),
                            topk,
                            BLOCK_SIZE=block,
                            replica_routing_table=tables[index] if use_table else None,
                            ROUTING_TABLE_ROWS=rows if use_table else 0,
                        )

                    launch()
                    torch.testing.assert_close(
                        out.cpu().long(), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        load.cpu().long(), expected_load, rtol=0, atol=0
                    )
                    for _ in range(10):
                        launch()
                    runtime.synchronize()
                    graph = graph_type()
                    with runtime.graph(graph):
                        for iteration in range(args.iterations):
                            launch(iteration % args.tables)
                    runtime.synchronize()
                    arms[name] = (graph, load, out)
                samples = {name: [] for name in arms}
                for _ in range(args.repeats):
                    names = list(arms)
                    order.shuffle(names)
                    for name in names:
                        graph, load, _ = arms[name]
                        load.zero_()
                        runtime.synchronize()
                        start, end = (
                            runtime.Event(enable_timing=True),
                            runtime.Event(enable_timing=True),
                        )
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(
                            start.elapsed_time(end) * 1000 / args.iterations
                        )
                medians = {k: statistics.median(v) for k, v in samples.items()}
                result = {
                    "tokens": tokens,
                    "recording": recording,
                    "block": block,
                    "median_us": medians,
                    "speedup": medians["hash"] / medians["table"],
                    "samples_us": samples,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    report = {
        "device": args.device,
        "device_name": runtime.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "experts": experts,
        "topk": topk,
        "redundant_experts": redundant,
        "tables": args.tables,
        "table_bytes": table.numel() * table.element_size() * args.tables,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 32, 128, 1024, 4096]
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--redundant-experts", type=int, default=32)
    parser.add_argument("--blocks", type=int, nargs="+", default=[256])
    parser.add_argument("--tables", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.tokens + args.blocks + [args.tables, args.iterations, args.repeats])
        < 1
    ):
        parser.error(
            "token counts, blocks, tables, iterations and repeats must be positive"
        )
    if not 0 <= args.redundant_experts <= args.experts:
        parser.error("redundant experts must be between zero and logical experts")
    benchmark(args)
