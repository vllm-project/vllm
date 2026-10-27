# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops.shardq import (
    fold_split,
    make_shardq_split,
    shard_q,
    unshard_out,
)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_fold_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    per_rank = fold_split(qsl, world)
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late piece of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


def _round_trip(rank, world, port, use_sm, q_dtype):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.manual_seed(0)
    lens = [700, 64, 3001]
    heads, dim = 4, 576
    full = torch.randn(sum(lens), world * heads, dim, device="cuda").to(q_dtype)
    split = make_shardq_split(
        np.r_[0, np.cumsum(lens)], world, rank, torch.device("cuda")
    )
    local_heads = full[:, rank * heads : (rank + 1) * heads].contiguous()
    group = dist.group.WORLD
    for _ in range(2):  # The second pass reuses the symmetric buffers.
        sharded = shard_q(local_heads, split, group, use_sm=use_sm)
        # Bitwise, as the exchanges only move bytes (and fp8 lacks torch.equal).
        expected = full[split.local_idx]
        assert torch.equal(sharded.view(torch.uint8), expected.view(torch.uint8))
        # Attention outputs are a head-padded slice; unshard reads its strides.
        padded = torch.zeros(sharded.shape[0], world * heads + 8, 512, device="cuda")
        padded[:, : world * heads] = sharded[..., :512].float()
        out = padded.bfloat16()[:, : world * heads]
        restored = unshard_out(out, split, group, use_sm=use_sm)
        assert torch.equal(restored, local_heads[..., :512].bfloat16())
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
def test_shard_q_round_trip(use_sm, q_dtype):
    world = 2
    mp.spawn(_round_trip, args=(world, get_open_port(), use_sm, q_dtype), nprocs=world)
