# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import statistics
from collections.abc import Callable

import torch
import torch.distributed as dist

import vllm._custom_ops as ops
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 128, 1024])
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--graph-repeats", type=int, default=20)
    parser.add_argument("--warmup-replays", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    return parser.parse_args()


def capture_graph(op: Callable[[], None], repeats: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeats):
            op()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def max_rank_graph_time(
    graph: torch.cuda.CUDAGraph,
    repeats: int,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> float:
    for _ in range(warmup_replays):
        graph.replay()
    torch.accelerator.synchronize()

    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        dist.barrier(group=cpu_group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            start.elapsed_time(end) / repeats,
            dtype=torch.float64,
            device=torch.accelerator.current_device_index(),
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
        timings.append(elapsed.item())
    return statistics.median(timings)


def check_outputs(
    comm: CustomAllreduce,
    local: torch.Tensor,
    reduce_input: torch.Tensor,
    device_group: dist.ProcessGroup,
) -> None:
    expected_gather = torch.empty(
        (local.shape[0] * dist.get_world_size(), local.shape[1]),
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(expected_gather, local, group=device_group)
    gathered = comm.custom_all_gather(local)
    assert gathered is not None
    torch.testing.assert_close(gathered, expected_gather)

    expected_scatter = torch.empty_like(local)
    dist.reduce_scatter_tensor(
        expected_scatter,
        reduce_input.clone(),
        group=device_group,
    )
    scattered = comm.custom_reduce_scatter(reduce_input)
    assert scattered is not None
    torch.testing.assert_close(scattered, expected_scatter)


def benchmark_shape(
    comm: CustomAllreduce,
    global_tokens: int,
    hidden_size: int,
    graph_repeats: int,
    warmup_replays: int,
    samples: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> dict[str, float | int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    padded_tokens = (global_tokens + world_size - 1) // world_size * world_size
    local_tokens = padded_tokens // world_size
    local = torch.full(
        (local_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=torch.accelerator.current_device_index(),
    )
    reduce_input = torch.full(
        (padded_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=local.device,
    )
    check_outputs(comm, local, reduce_input, device_group)

    custom_gather_out = torch.empty(
        (padded_tokens, hidden_size),
        dtype=local.dtype,
        device=local.device,
    )
    custom_scatter_out = torch.empty_like(local)
    nccl_gather_out = torch.empty_like(custom_gather_out)
    nccl_scatter_out = torch.empty_like(local)

    def custom_ag() -> None:
        ops.mnnvl_lamport_all_gather(
            comm._ptr,
            local,
            custom_gather_out,
            comm.mnnvl_lamport_ag_local_ptr,
            comm.mnnvl_lamport_ag_multicast_ptr,
            comm.mnnvl_lamport_ag_epoch_ptr,
            comm.mnnvl_buffer_size,
        )

    def custom_rs() -> None:
        ops.mnnvl_lamport_reduce_scatter(
            comm._ptr,
            reduce_input,
            custom_scatter_out,
            comm.mnnvl_lamport_rs_local_ptr,
            comm.mnnvl_lamport_rs_epoch_ptr,
            comm.mnnvl_buffer_size,
        )

    def nccl_ag() -> None:
        dist.all_gather_into_tensor(nccl_gather_out, local, group=device_group)

    def nccl_rs() -> None:
        dist.reduce_scatter_tensor(
            nccl_scatter_out,
            reduce_input,
            group=device_group,
        )

    graphs = {
        "custom_ag_us": capture_graph(custom_ag, graph_repeats),
        "nccl_ag_us": capture_graph(nccl_ag, graph_repeats),
        "custom_rs_us": capture_graph(custom_rs, graph_repeats),
        "nccl_rs_us": capture_graph(nccl_rs, graph_repeats),
    }
    times = {
        name: max_rank_graph_time(
            graph,
            graph_repeats,
            warmup_replays,
            samples,
            device_group,
            cpu_group,
        )
        * 1000
        for name, graph in graphs.items()
    }
    torch.testing.assert_close(custom_gather_out, nccl_gather_out)
    torch.testing.assert_close(custom_scatter_out, nccl_scatter_out)
    return {
        "global_tokens": global_tokens,
        "padded_tokens": padded_tokens,
        "local_bytes": local.nbytes,
        "full_bytes": reduce_input.nbytes,
        **times,
        "ag_speedup": times["nccl_ag_us"] / times["custom_ag_us"],
        "rs_speedup": times["nccl_rs_us"] / times["custom_rs_us"],
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl")
    device_group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")

    comm = CustomAllreduce(
        group=cpu_group,
        device=torch.device("cuda", local_rank),
    )
    assert not comm.disabled
    assert comm.world_size == 16
    assert comm.mnnvl_only
    assert comm.mnnvl_multicast_ptr

    results = [
        benchmark_shape(
            comm,
            tokens,
            args.hidden_size,
            args.graph_repeats,
            args.warmup_replays,
            args.samples,
            device_group,
            cpu_group,
        )
        for tokens in args.tokens
    ]
    if dist.get_rank() == 0:
        print(json.dumps(results, indent=2), flush=True)

    comm.close()
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
