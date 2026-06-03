# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness test for UCX allreduce — exercises cross-node paths.

Tests multiple patterns:
1. Each rank writes its column (the actual DP sync pattern)
2. Each rank writes a unique value — verifies every peer's data arrives
3. Sequential rounds — verifies round counter stays in sync
4. Stress test with many rapid calls

Single-node quick test:
  torchrun --nproc-per-node=4 test_ucx_correctness.py

Cross-node (4 nodes x 4 ranks):
  torchrun --nproc-per-node=4 --nnodes=4 \
    --master-addr=$MASTER_ADDR --master-port=29500 \
    --node-rank=$NODE_RANK test_ucx_correctness.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def load_ucx_communicator(rank, world_size, gloo_group):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from ucx_dp_communicator import UCXDPCommunicator

    comm = UCXDPCommunicator(rank, world_size, max_msg_bytes=1024)
    comm.bootstrap(gloo_group)
    return comm


def test_column_pattern(comm, rank, world_size, n_iters=50):
    """The actual DP sync pattern: each rank fills its column."""
    failures = 0
    for it in range(n_iters):
        tensor = torch.zeros(4, world_size, dtype=torch.int32)
        tensor[0][rank] = rank + 1
        tensor[1][rank] = (rank + 1) * 10
        tensor[2][rank] = 1
        tensor[3][rank] = it % 3

        comm.allreduce_inplace(tensor)

        for r in range(world_size):
            expected = [r + 1, (r + 1) * 10, 1, it % 3]
            actual = [tensor[row][r].item() for row in range(4)]
            if actual != expected:
                print(
                    f"[rank {rank}] FAIL iter={it} col={r}: "
                    f"expected {expected} got {actual}"
                )
                failures += 1
                if failures > 5:
                    return failures
    return failures


def test_unique_values(comm, rank, world_size, n_iters=50):
    """Each rank writes a unique value in every cell of its
    column."""
    failures = 0
    for it in range(n_iters):
        tensor = torch.zeros(4, world_size, dtype=torch.int32)
        val = rank * 1000 + it
        for row in range(4):
            tensor[row][rank] = val + row

        comm.allreduce_inplace(tensor)

        for r in range(world_size):
            base = r * 1000 + it
            for row in range(4):
                expected = base + row
                actual = tensor[row][r].item()
                if actual != expected:
                    print(
                        f"[rank {rank}] FAIL iter={it} "
                        f"row={row} col={r}: "
                        f"expected {expected} got {actual}"
                    )
                    failures += 1
                    if failures > 5:
                        return failures
    return failures


def test_gloo_comparison(comm, rank, world_size, gloo_group, n_iters=50):
    """Run same allreduce via both UCX and Gloo, compare."""
    failures = 0
    for it in range(n_iters):
        tensor_ucx = torch.zeros(4, world_size, dtype=torch.int32)
        tensor_gloo = torch.zeros(4, world_size, dtype=torch.int32)

        val = rank * 100 + it
        for row in range(4):
            tensor_ucx[row][rank] = val + row
            tensor_gloo[row][rank] = val + row

        comm.allreduce_inplace(tensor_ucx)
        dist.all_reduce(tensor_gloo, group=gloo_group)

        if not torch.equal(tensor_ucx, tensor_gloo):
            diff_mask = tensor_ucx != tensor_gloo
            print(
                f"[rank {rank}] FAIL iter={it}: "
                f"UCX != Gloo\n"
                f"  UCX:  {tensor_ucx.tolist()}\n"
                f"  Gloo: {tensor_gloo.tolist()}\n"
                f"  Diff: {diff_mask.nonzero().tolist()}"
            )
            failures += 1
            if failures > 5:
                return failures
    return failures


def test_rapid_fire(comm, rank, world_size, n_iters=200):
    """Many rapid consecutive allreduces — stress the round
    counter."""
    failures = 0
    for it in range(n_iters):
        tensor = torch.zeros(4, world_size, dtype=torch.int32)
        tensor[0][rank] = it + 1

        comm.allreduce_inplace(tensor)

        total = tensor[0].sum().item()
        expected = (it + 1) * world_size
        if total != expected:
            print(
                f"[rank {rank}] FAIL iter={it}: "
                f"sum={total} expected={expected} "
                f"row0={tensor[0].tolist()}"
            )
            failures += 1
            if failures > 5:
                return failures
    return failures


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gloo_group = dist.group.WORLD

    print(f"[rank {rank}] world_size={world_size}")

    comm = load_ucx_communicator(rank, world_size, gloo_group)
    print(f"[rank {rank}] UCX communicator ready")

    tests = [
        (
            "column_pattern",
            lambda: test_column_pattern(comm, rank, world_size),
        ),
        (
            "unique_values",
            lambda: test_unique_values(comm, rank, world_size),
        ),
        (
            "gloo_comparison",
            lambda: test_gloo_comparison(comm, rank, world_size, gloo_group),
        ),
        (
            "rapid_fire",
            lambda: test_rapid_fire(comm, rank, world_size),
        ),
    ]

    all_pass = True
    for name, fn in tests:
        dist.barrier(group=gloo_group)
        t0 = time.monotonic()
        failures = fn()
        elapsed = time.monotonic() - t0
        status = "PASS" if failures == 0 else f"FAIL ({failures} failures)"
        print(f"[rank {rank}] {name}: {status} ({elapsed:.2f}s)")
        if failures > 0:
            all_pass = False

    comm.finalize()
    dist.barrier(group=gloo_group)

    if rank == 0:
        result = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
        print(f"\n{result}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
