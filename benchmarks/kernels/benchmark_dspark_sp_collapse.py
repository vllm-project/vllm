# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the DeepSeek-V4.1 draft tail's gather/collapse ordering.

Run with .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4
benchmarks/kernels/benchmark_dspark_sp_collapse.py. No model weights are needed.
The timed boundary includes mHC post, collapse, and the production SP gathers.
CUDA events measure collective latency across ranks; each sample uses the slowest
rank. Input generation, compilation, allocation and L2 flushing are outside timing.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.models.common.ops.sequence_parallel import sp_all_gather


def benchmark(args, tokens, flush, sync_input, sync_output):
    group = get_tp_group()
    rank, size = group.rank_in_group, group.world_size
    local_tokens = (tokens + size - 1) // size
    hidden, hc = args.hidden_size, 4
    device = sync_input.device
    torch.manual_seed(42 + rank)
    graphs = {"before": [], "after": []}
    keepalive = []
    comm = group.device_communicator

    for _ in range(args.workspaces):
        x = torch.randn(local_tokens, hidden, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            local_tokens, hc, hidden, dtype=torch.bfloat16, device=device
        )
        post = torch.rand(local_tokens, hc, 1, device=device)
        res = torch.rand(local_tokens, hc, hc, device=device)
        pre = torch.rand(local_tokens, hc, device=device)

        def before(x=x, residual=residual, post=post, res=res, pre=pre):
            states = mhc_post_tilelang(x, residual, post, res)
            states = sp_all_gather(states)[:tokens]
            full_pre = sp_all_gather(pre)[:tokens]
            return hc_collapse_triton(states, full_pre)

        def after(x=x, residual=residual, post=post, res=res, pre=pre):
            states = mhc_post_tilelang(x, residual, post, res)
            states = hc_collapse_triton(states, pre)
            return sp_all_gather(states)[:tokens]

        expected, actual = before(), after()
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        captured_outputs = {}
        for name, op in (("before", before), ("after", after)):
            dist.barrier(group=group.cpu_group)
            with group.graph_capture() as context:
                for _ in range(3):
                    op()
                context.stream.synchronize()
                dist.barrier(group=group.cpu_group)
                graph = torch.cuda.CUDAGraph()
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph, stream=context.stream):
                    start.record()
                    output = op()
                    end.record()
            graphs[name].append((graph, start, end))
            captured_outputs[name] = output
            keepalive.extend((context, output))
        keepalive.extend((x, residual, post, res, pre, expected, actual))
        # Verify graph replay as well as eager execution before timing.
        for name in graphs:
            graphs[name][-1][0].replay()
        torch.accelerator.synchronize()
        for output in captured_outputs.values():
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    for index in range(args.warmup):
        for name in ("before", "after"):
            comm.pynccl_comm.all_reduce(sync_input, sync_output)
            graphs[name][index % args.workspaces][0].replay()
    torch.accelerator.synchronize()
    timings = {name: [] for name in graphs}
    for index in range(args.samples):
        order = ("before", "after") if index % 2 == 0 else ("after", "before")
        for name in order:
            flush.zero_()
            comm.pynccl_comm.all_reduce(sync_input, sync_output)
            graph, start, end = graphs[name][index % args.workspaces]
            graph.replay()
            end.synchronize()
            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000, dtype=torch.float64, device=device
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group.device_group)
            timings[name].append(elapsed.item())
    latency = {name: statistics.median(values) for name, values in timings.items()}
    # Logical receive payload per rank, excluding its own shard.
    before_bytes = (size - 1) * local_tokens * hc * (hidden * 2 + 4)
    after_bytes = (size - 1) * local_tokens * hidden * 2
    ca = comm.ca_comm
    custom_gathers = {
        name: ca is not None and ca.should_custom_all_gather(tensor)
        for name, tensor in (
            ("before_states", residual),
            ("before_pre_mix", pre),
            ("after_states", x),
        )
    }
    return {
        "tokens": tokens,
        "local_tokens": local_tokens,
        "hidden_size": hidden,
        "hc_mult": hc,
        "before_us": latency["before"],
        "after_us": latency["after"],
        "speedup": latency["before"] / latency["after"],
        "before_receive_bytes_per_rank": before_bytes,
        "after_receive_bytes_per_rank": after_bytes,
        "before_payload_GBps": before_bytes / (latency["before"] * 1000),
        "after_payload_GBps": after_bytes / (latency["after"] * 1000),
        "custom_gathers": custom_gathers,
        "bitwise_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 7, 32, 128, 512, 2048]
    )
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--workspaces", type=int, default=4)
    parser.add_argument("--nccl-only", action="store_true")
    args = parser.parse_args()
    assert min(*args.tokens, args.hidden_size, args.samples, args.workspaces) > 0
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment()
    set_custom_all_reduce(not args.nccl_only)
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    sync_input = torch.zeros(1, device=device)
    sync_output = torch.empty_like(sync_input)
    flush = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "world_size": dist.get_world_size(),
                    "args": vars(args),
                }
            ),
            flush=True,
        )
    for tokens in args.tokens:
        result = benchmark(args, tokens, flush, sync_input, sync_output)
        if dist.get_rank() == 0:
            print(json.dumps(result), flush=True)
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
