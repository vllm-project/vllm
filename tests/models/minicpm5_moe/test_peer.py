# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the host IPC protocol without creating XPU contexts."""

import os
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.minicpm5_moe.xpu.specialized_decode.peer import agree, exchange_fd


def _peer_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    read_fd, write_fd = os.pipe()
    imported = None
    try:
        os.write(write_fd, str(rank).encode())
        imported = exchange_fd(read_fd, rank, dist.group.WORLD)
        os.close(read_fd)
        read_fd = -1
        assert os.read(imported, 1) == str(1 - rank).encode()
        try:
            agree(
                ["injected cache mismatch"] if rank == 1 else [],
                dist.group.WORLD,
                "cache layout",
            )
        except ValueError as error:
            assert "rank1: injected cache mismatch" in str(error)
        else:
            raise AssertionError("Both ranks must refuse a one-rank failure")
        agree([], dist.group.WORLD, "normal control")
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)
        if imported is not None:
            os.close(imported)
        dist.destroy_process_group()


def test_peer_fd_ownership_and_collective_failure_agreement(tmp_path):
    mp.spawn(_peer_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
