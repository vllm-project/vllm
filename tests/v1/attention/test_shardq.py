# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import (
    build_shardq_split,
    fold_split,
    indexer_rows,
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


@pytest.mark.parametrize("num_spec", [0, 2])
def test_build_split_after_indexer_decodes(monkeypatch, num_spec):
    """The MLA and indexer builders derive the same split from the batch:
    requests of at most 1 + num_spec tokens stay head-sharded, and the tokens
    after them are split once they reach sparse_mla_shard_q_min_tokens."""
    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=1)
    )
    query_lens = [1 + num_spec] * 3 + [100, 5000, 4000]
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[q + 300 for q in query_lens], query_lens=query_lens),
        block_size=16,
        device=torch.device("cpu"),
    )
    attention_config = SimpleNamespace(sparse_mla_shard_q_min_tokens=9100)
    vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
        ),
        attention_config=attention_config,
    )
    built = build_shardq_split(metadata, vllm_config, torch.device("cpu"))
    assert built is not None
    num_decode_tokens, split = built
    assert num_decode_tokens == 3 * (1 + num_spec)
    assert sum(split.counts) == 9100
    attention_config.sparse_mla_shard_q_min_tokens = 9101
    assert build_shardq_split(metadata, vllm_config, torch.device("cpu")) is None


@pytest.mark.parametrize("num_decode_tokens", [0, 1, 37])
def test_indexer_rows_cover_decodes_then_local_prefill(num_decode_tokens):
    """Mixed steps: every rank scores the decode rows in place, then its own
    prefill tokens compactly; across ranks each prefill row is scored once."""
    world, lens = 4, [700, 64, 3001]
    qsl = np.r_[0, np.cumsum(lens)]
    num_prefill = int(qsl[-1])
    prefill_rows = []
    for rank in range(world):
        split = make_shardq_split(qsl, world, rank, torch.device("cpu"))
        host, rows = indexer_rows(split, num_decode_tokens)
        np.testing.assert_array_equal(host, rows.numpy())
        assert np.all(np.diff(host) > 0)
        np.testing.assert_array_equal(
            host[:num_decode_tokens], np.arange(num_decode_tokens)
        )
        # Row p of the buffer holds the p-th scored row: this rank's prefill
        # tokens follow the decode rows in the order the attention reads them.
        np.testing.assert_array_equal(
            host[num_decode_tokens:] - num_decode_tokens, split.local_idx_cpu
        )
        prefill_rows.append(host[num_decode_tokens:])
    np.testing.assert_array_equal(
        np.sort(np.concatenate(prefill_rows)),
        np.arange(num_decode_tokens, num_decode_tokens + num_prefill),
    )


def _round_trip(rank, world, port, use_sm, q_dtype, num_leading_rows):
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
        restored = unshard_out(
            out, split, group, use_sm=use_sm, num_leading_rows=num_leading_rows
        )
        assert restored.shape[0] == num_leading_rows + sum(lens)
        assert torch.equal(
            restored[num_leading_rows:], local_heads[..., :512].bfloat16()
        )
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("num_leading_rows", [0, 33])
def test_shard_q_round_trip(use_sm, q_dtype, num_leading_rows):
    world = 2
    mp.spawn(
        _round_trip,
        args=(world, get_open_port(), use_sm, q_dtype, num_leading_rows),
        nprocs=world,
    )
