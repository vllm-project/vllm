# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone test for UCX DP allreduce.

Single-node quick test with 4 ranks:
  torchrun --nproc-per-node=4 \
    vllm/distributed/device_communicators/test_ucx_allreduce.py

Cross-node test (4 nodes x 4 ranks):
  torchrun --nproc-per-node=4 --nnodes=4 \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    --node-rank=$NODE_RANK \
    vllm/distributed/device_communicators/test_ucx_allreduce.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def test_gloo_baseline(rank, world_size, group, n_iters=100):
    """Measure Gloo TCP allreduce latency."""
    tensor = torch.zeros(4, world_size, dtype=torch.int32)
    # warmup
    for _ in range(10):
        tensor.zero_()
        tensor[0][rank] = 1
        dist.all_reduce(tensor, group=group)

    latencies = []
    for _ in range(n_iters):
        tensor.zero_()
        tensor[0][rank] = rank + 1
        tensor[1][rank] = (rank + 1) * 10
        tensor[2][rank] = 1
        tensor[3][rank] = 2
        t0 = time.monotonic()
        dist.all_reduce(tensor, group=group)
        latencies.append(time.monotonic() - t0)

    for i in range(world_size):
        assert tensor[0][i].item() == i + 1, f"Gloo: rank {i} col 0 wrong"
        assert tensor[1][i].item() == (i + 1) * 10, f"Gloo: rank {i} col 1 wrong"

    return latencies


def test_ucx_allreduce(rank, world_size, gloo_group, n_iters=100):
    """Measure UCX RDMA allreduce latency."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from ucx_dp_communicator import UCXDPCommunicator

    comm = UCXDPCommunicator(rank, world_size, max_msg_bytes=1024)
    comm.bootstrap(gloo_group)
    print(f"[rank {rank}] UCX communicator initialized")

    tensor = torch.zeros(4, world_size, dtype=torch.int32)
    # warmup
    for _ in range(10):
        tensor.zero_()
        tensor[0][rank] = 1
        comm.allreduce_inplace(tensor)

    latencies = []
    for _ in range(n_iters):
        tensor.zero_()
        tensor[0][rank] = rank + 1
        tensor[1][rank] = (rank + 1) * 10
        tensor[2][rank] = 1
        tensor[3][rank] = 2
        t0 = time.monotonic()
        comm.allreduce_inplace(tensor)
        latencies.append(time.monotonic() - t0)

    for i in range(world_size):
        assert tensor[0][i].item() == i + 1, (
            f"UCX: rank {i} col 0 = {tensor[0][i].item()}, expected {i + 1}"
        )
        assert tensor[1][i].item() == (i + 1) * 10, (
            f"UCX: rank {i} col 1 = {tensor[1][i].item()}, expected {(i + 1) * 10}"
        )

    comm.finalize()
    return latencies


def percentile(data, p):
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return data[min(idx, len(data) - 1)]


def print_stats(name, latencies, rank):
    if latencies is None:
        print(f"[rank {rank}] {name}: SKIPPED")
        return
    us = [t * 1e6 for t in latencies]
    print(
        f"[rank {rank}] {name}: "
        f"p50={percentile(us, 50):.1f}us  "
        f"p95={percentile(us, 95):.1f}us  "
        f"p99={percentile(us, 99):.1f}us  "
        f"mean={sum(us) / len(us):.1f}us  "
        f"min={min(us):.1f}us  "
        f"max={max(us):.1f}us"
    )


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gloo_group = dist.group.WORLD

    print(f"[rank {rank}] world_size={world_size}")

    n_iters = int(os.environ.get("TEST_ITERS", "200"))

    gloo_latencies = test_gloo_baseline(rank, world_size, gloo_group, n_iters)
    print_stats("Gloo TCP", gloo_latencies, rank)

    ucx_latencies = test_ucx_allreduce(rank, world_size, gloo_group, n_iters)
    print_stats("UCX RDMA", ucx_latencies, rank)

    if ucx_latencies and rank == 0:
        gloo_p50 = percentile(gloo_latencies, 50) * 1e6
        ucx_p50 = percentile(ucx_latencies, 50) * 1e6
        gloo_p99 = percentile(gloo_latencies, 99) * 1e6
        ucx_p99 = percentile(ucx_latencies, 99) * 1e6
        print("\n=== Speedup ===")
        print(f"  P50: {gloo_p50:.1f}us -> {ucx_p50:.1f}us ({gloo_p50 / ucx_p50:.1f}x)")
        print(f"  P99: {gloo_p99:.1f}us -> {ucx_p99:.1f}us ({gloo_p99 / ucx_p99:.1f}x)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
