# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise eager cache snapshots through multiple actual process boundaries."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.deepseek_v4_1.common.pipeline_transfer import (
    restore_cache_blocks,
    snapshot_cache_blocks,
)


def _relay_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        expected = torch.arange(6 * 3 * 4, dtype=torch.uint8).reshape(6, 3, 4)
        cache = expected.clone() if rank == 0 else torch.zeros_like(expected)
        table = torch.tensor([[1, 4, -1]])
        if rank:
            ids = torch.empty(2, dtype=torch.int64)
            blocks = torch.empty(2, 3, 4, dtype=torch.uint8)
            dist.recv(ids, src=rank - 1)
            dist.recv(blocks, src=rank - 1)
            restore_cache_blocks(cache, ids, blocks)
        torch.testing.assert_close(cache[[1, 4]], expected[[1, 4]], rtol=0, atol=0)
        if rank < 2:
            ids, blocks = snapshot_cache_blocks(cache, [table], max_bytes=1024)
            dist.send(ids, dst=rank + 1)
            dist.send(blocks, dst=rank + 1)
    finally:
        dist.destroy_process_group()


def test_cache_blocks_survive_two_pipeline_hops(tmp_path):
    mp.spawn(_relay_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3)
