# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark production MNNVL reduce-scatter backends on SM100/SM103.

The multimem candidate includes the device-to-device copy into its persistent
symmetric input buffer. Timings use pointer-distinct CUDA graphs and report the
median of the per-sample maximum across ranks.

Run the default BF16 message-size sweep with:

    .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 \
        benchmarks/kernels/benchmark_multimem_reduce_scatter.py
"""

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm._custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

_DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


@dataclass
class Candidate:
    name: str
    runs: list[Callable[[], None]]
    outputs: list[torch.Tensor]


def parse_size(value: str) -> int:
    suffixes = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}
    value = value.strip().lower()
    multiplier = suffixes.get(value[-1], 1)
    number = value[:-1] if multiplier != 1 else value
    return int(float(number) * multiplier)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message-bytes",
        type=parse_size,
        nargs="+",
        default=[
            16 << 20,
            32 << 20,
            64 << 20,
            68 << 20,
            72 << 20,
            76 << 20,
            96 << 20,
        ],
        help="Full input bytes on each rank. K/M/G suffixes are accepted.",
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="+",
        default=[8],
        help="Maximum CTAs used by each multimem candidate.",
    )
    parser.add_argument("--num-workspaces", type=int, default=4)
    parser.add_argument("--warmup-replays", type=int, default=8)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="Flush approximately twice L2 capacity before each replay.",
    )
    return parser.parse_args()


def make_lamport_run(
    comm: CustomAllreduce,
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> Callable[[], None]:
    def run() -> None:
        ops.mnnvl_lamport_reduce_scatter(
            comm._ptr,
            input_tensor,
            output,
            comm.mnnvl_lamport_rs_local_ptr,
            comm.mnnvl_lamport_rs_epoch_ptr,
            comm.mnnvl_buffer_size,
        )

    return run


def make_multimem_run(
    comm: CustomAllreduce,
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    blocks: int,
) -> Callable[[], None]:
    def run() -> None:
        ops.mnnvl_multimem_reduce_scatter(
            comm._ptr,
            input_tensor,
            output,
            comm.mnnvl_multimem_rs_local_ptr,
            comm.mnnvl_multimem_rs_multicast_ptr,
            comm.mnnvl_multimem_rs_buffer_size,
            blocks,
        )

    return run


def make_nccl_run(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    device_group: dist.ProcessGroup,
) -> Callable[[], None]:
    def run() -> None:
        dist.reduce_scatter_single(output, input_tensor, group=device_group)

    return run


def build_candidates(
    comm: CustomAllreduce,
    inputs: list[torch.Tensor],
    local_elements: int,
    blocks_values: list[int],
    device_group: dist.ProcessGroup,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    lamport_outputs = [
        input_tensor.new_empty(local_elements) for input_tensor in inputs
    ]
    candidates.append(
        Candidate(
            "lamport",
            [
                make_lamport_run(comm, input_tensor, output)
                for input_tensor, output in zip(inputs, lamport_outputs)
            ],
            lamport_outputs,
        )
    )

    for blocks in blocks_values:
        outputs = [input_tensor.new_empty(local_elements) for input_tensor in inputs]
        candidates.append(
            Candidate(
                f"multimem_b{blocks}",
                [
                    make_multimem_run(comm, input_tensor, output, blocks)
                    for input_tensor, output in zip(inputs, outputs)
                ],
                outputs,
            )
        )

    nccl_outputs = [input_tensor.new_empty(local_elements) for input_tensor in inputs]
    candidates.append(
        Candidate(
            "nccl",
            [
                make_nccl_run(input_tensor, output, device_group)
                for input_tensor, output in zip(inputs, nccl_outputs)
            ],
            nccl_outputs,
        )
    )
    return candidates


def check_correctness(
    candidates: list[Candidate],
    inputs: list[torch.Tensor],
    rank: int,
    world_size: int,
    cpu_group: dist.ProcessGroup,
    device_group: dist.ProcessGroup,
    graphs: dict[str, list[torch.cuda.CUDAGraph]] | None = None,
) -> None:
    expected_outputs = []
    for input_tensor in inputs:
        expected = input_tensor.float().clone()
        dist.all_reduce(expected, group=device_group)
        expected_outputs.append(expected.to(input_tensor.dtype).chunk(world_size)[rank])

    if inputs[0].dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    elif inputs[0].dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 5e-2, 1.25e-1

    for candidate in candidates:
        for workspace_index, expected in enumerate(expected_outputs):
            dist.barrier(group=cpu_group)
            if graphs is None:
                candidate.runs[workspace_index]()
            else:
                graphs[candidate.name][workspace_index].replay()
            torch.accelerator.synchronize()
            error = None
            try:
                torch.testing.assert_close(
                    candidate.outputs[workspace_index],
                    expected,
                    rtol=rtol,
                    atol=atol,
                )
            except AssertionError as assertion_error:
                error = assertion_error
            correct = torch.tensor(
                error is None,
                dtype=torch.int32,
                device=candidate.outputs[workspace_index].device,
            )
            dist.all_reduce(correct, op=dist.ReduceOp.MIN, group=device_group)
            if not correct.item():
                location = f"{candidate.name} workspace {workspace_index}"
                detail = str(error) if error is not None else "failed on another rank"
                raise AssertionError(
                    f"{location} produced an incorrect shard: {detail}"
                )
    dist.barrier(group=cpu_group)


def capture_graph(
    run: Callable[[], None],
    cpu_group: dist.ProcessGroup,
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    dist.barrier(group=cpu_group)
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier(group=cpu_group)
    return graph, stream


def capture_candidates(
    candidates: list[Candidate],
    cpu_group: dist.ProcessGroup,
) -> tuple[dict[str, list[torch.cuda.CUDAGraph]], list[object]]:
    graphs = {}
    keepalive: list[object] = []
    for candidate in candidates:
        bundles = [capture_graph(run, cpu_group) for run in candidate.runs]
        graphs[candidate.name] = [graph for graph, _ in bundles]
        keepalive.extend(bundles)
    return graphs, keepalive


def benchmark_graphs(
    graphs: dict[str, list[torch.cuda.CUDAGraph]],
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
    flush: torch.Tensor | None,
) -> dict[str, float]:
    names = list(graphs)
    for round_index in range(warmup_replays):
        for candidate_index in range(len(names)):
            name = names[(round_index + candidate_index) % len(names)]
            device_barrier()
            graphs[name][round_index % len(graphs[name])].replay()
    torch.accelerator.synchronize()

    timings = {name: [] for name in names}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for sample_index in range(samples):
        for candidate_index in range(len(names)):
            name = names[(sample_index + candidate_index) % len(names)]
            if flush is not None:
                flush.zero_()
            device_barrier()
            start.record()
            graphs[name][sample_index % len(graphs[name])].replay()
            end.record()
            end.synchronize()

            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000,
                dtype=torch.float64,
                device=torch.accelerator.current_device_index(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
            timings[name].append(elapsed.item())

    return {name: statistics.median(values) for name, values in timings.items()}


def benchmark_size(
    args: argparse.Namespace,
    comm: CustomAllreduce,
    message_bytes: int,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    device_barrier: Callable[[], None],
) -> dict[str, object]:
    element_size = dtype.itemsize
    if message_bytes % (element_size * world_size) != 0:
        raise ValueError("Message bytes must divide evenly into TP shards")
    num_elements = message_bytes // element_size
    local_elements = num_elements // world_size
    if local_elements * element_size % 16 != 0:
        raise ValueError("Each output shard must be a multiple of 16 bytes")

    inputs = []
    for workspace_index in range(args.num_workspaces):
        generator = torch.Generator(device=device)
        generator.manual_seed(1000 + rank * 101 + workspace_index)
        input_tensor = torch.empty(num_elements, dtype=dtype, device=device)
        input_tensor.uniform_(-1.0, 1.0, generator=generator)
        inputs.append(input_tensor)

    candidates = build_candidates(
        comm,
        inputs,
        local_elements,
        args.blocks,
        device_group,
    )
    torch.accelerator.synchronize()
    dist.barrier(group=cpu_group)
    check_correctness(
        candidates,
        inputs,
        rank,
        world_size,
        cpu_group,
        device_group,
    )
    graphs, graph_keepalive = capture_candidates(candidates, cpu_group)
    check_correctness(
        candidates,
        inputs,
        rank,
        world_size,
        cpu_group,
        device_group,
        graphs,
    )

    flush = None
    if args.cold_l2:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        flush = torch.empty(2 * l2_bytes, dtype=torch.uint8, device=device)
    times = benchmark_graphs(
        graphs,
        args.warmup_replays,
        args.samples,
        device_group,
        device_barrier,
        flush,
    )
    graph_keepalive.clear()

    bus_bytes = message_bytes * (world_size - 1) / world_size
    fastest = min(times, key=times.__getitem__)
    return {
        "message_bytes": message_bytes,
        "shard_bytes": message_bytes // world_size,
        "latency_us": times,
        "logical_bus_gbps": {
            name: bus_bytes / (latency_us * 1e3) for name, latency_us in times.items()
        },
        "fastest": fastest,
    }


def main() -> None:
    args = parse_args()
    if not args.message_bytes or min(args.message_bytes) <= 0:
        raise ValueError("--message-bytes values must be positive")
    if not args.blocks or min(args.blocks) <= 0 or max(args.blocks) > 8:
        raise ValueError("--blocks values must be in [1, 8]")
    if min(args.num_workspaces, args.samples) <= 0:
        raise ValueError("Workspace and sample counts must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl")
    device_group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")
    rank = dist.get_rank(device_group)
    world_size = dist.get_world_size(device_group)
    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(device)
    if world_size not in (2, 4, 8) or capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            "This benchmark targets TP2/TP4/TP8 SM100/SM103, "
            f"got TP{world_size} SM{capability}"
        )

    max_message_bytes = max(args.message_bytes)
    comm = CustomAllreduce(
        group=cpu_group,
        device=device,
        max_mnnvl_reduce_scatter_size=max_message_bytes,
        max_mnnvl_multimem_reduce_scatter_size=max_message_bytes,
    )
    if comm.disabled:
        raise RuntimeError("The production custom all-reduce backend is unavailable")
    comm._init_mnnvl_multimem_reduce_scatter_buffer()
    if not comm.mnnvl_multimem_rs_multicast_ptr:
        raise RuntimeError("The production MNNVL multimem backend is unavailable")

    dtype = _DTYPES[args.dtype]
    warmup = torch.zeros(1, dtype=torch.float32, device=device)
    dist.all_reduce(warmup, group=device_group)
    sync = torch.zeros_like(warmup)

    def device_barrier() -> None:
        dist.all_reduce(sync, group=device_group)

    results = [
        benchmark_size(
            args,
            comm,
            message_bytes,
            dtype,
            device,
            rank,
            world_size,
            device_group,
            cpu_group,
            device_barrier,
        )
        for message_bytes in args.message_bytes
    ]
    if rank == 0:
        metadata = {
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(device),
            "sm": f"{capability[0]}{capability[1]}",
            "num_sms": torch.cuda.get_device_properties(device).multi_processor_count,
            "dtype": args.dtype,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "blocks": args.blocks,
            "num_workspaces": args.num_workspaces,
            "cold_l2": args.cold_l2,
        }
        print(json.dumps({"metadata": metadata, "results": results}, indent=2))

    comm.close()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
