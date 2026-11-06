# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ token split and exchanges (vllm/v1/attention/ops/shardq.py)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.ops import shardq
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    allocate_shardq_buffers,
    build_shardq_split,
    shard_q,
    unshard_out,
    zigzag_split,
)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("lens", [[4096], [65, 300, 7000], [1000, 64, 64, 129]])
def test_zigzag_split_partitions_tokens(world, lens):
    qsl = np.r_[0, np.cumsum(lens)]
    token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
    per_rank = np.split(token_ids, np.cumsum(num_tokens_per_rank)[:-1])
    merged = np.sort(np.concatenate(per_rank))
    np.testing.assert_array_equal(merged, np.arange(qsl[-1]))
    for idx in per_rank:
        assert np.all(np.diff(idx) > 0)
    # Each rank takes one early and one late chunk of every request, so its
    # summed causal positions stay within a couple of tokens per request of
    # the mean.
    positions = [
        sum(int((idx[(idx >= a) & (idx < b)] - a).sum()) for a, b in zip(qsl, qsl[1:]))
        for idx in per_rank
    ]
    assert max(positions) - min(positions) <= sum(4 * n for n in lens)


@pytest.mark.parametrize("num_spec", [0, 2])
def test_build_shardq_split_after_indexer_decodes(monkeypatch, num_spec):
    """The MLA and indexer builders derive the same split from the batch:
    requests of at most 1 + num_spec tokens stay head-sharded, and the tokens
    after them are split once they reach sparse_mla_shard_q_min_tokens and a
    request is longer than the top-k."""
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
    hf_config = SimpleNamespace(index_topk=5299)
    vllm_config = SimpleNamespace(
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None
        ),
        attention_config=attention_config,
        model_config=SimpleNamespace(hf_text_config=hf_config),
    )
    split = build_shardq_split(metadata, vllm_config)
    assert split is not None
    assert split.num_decode_tokens == 3 * (1 + num_spec)
    assert split.num_tokens_per_rank.sum() == 9100
    hf_config.index_topk = 5300
    assert build_shardq_split(metadata, vllm_config) is None
    hf_config.index_topk = 5299
    attention_config.sparse_mla_shard_q_min_tokens = 9101
    assert build_shardq_split(metadata, vllm_config) is None
    # Five tokens split into eight chunks, three of them empty, leaving rank 2
    # without tokens.
    attention_config.sparse_mla_shard_q_min_tokens = 1
    short = create_common_attn_metadata(
        BatchSpec(seq_lens=[5300], query_lens=[5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    assert build_shardq_split(short, vllm_config) is None


def _round_trip(rank, world, port, use_sm, q_dtype, num_decode_tokens):
    torch.accelerator.set_device_index(rank)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
        )
        ensure_model_parallel_initialized(world, 1)
        torch.manual_seed(0)
        lens = [700, 64, 3001]
        heads, dim = 4, 576
        num_tokens = num_decode_tokens + sum(lens)
        vllm_config.scheduler_config.max_num_batched_tokens = num_tokens
        if use_sm:
            allocate_shardq_buffers(
                vllm_config, heads, dim, 512, q_dtype, torch.bfloat16
            )
        all_heads = world * heads
        full = torch.randn(num_tokens, all_heads, dim, device="cuda").to(q_dtype)
        token_ids, num_tokens_per_rank = zigzag_split(np.array(lens), world)
        shardq = ShardQMetadata(
            num_decode_tokens=num_decode_tokens,
            num_tokens_per_rank=num_tokens_per_rank.tolist(),
            token_ids=torch.from_numpy(token_ids).cuda(),
            rank_offsets=torch.from_numpy(
                np.r_[0, np.cumsum(num_tokens_per_rank)]
            ).cuda(),
        )
        local_heads = full[:, rank * heads : (rank + 1) * heads].contiguous()
        for _ in range(2):  # The second pass reuses the symmetric buffers.
            decode_q, sharded = shard_q(local_heads, shardq)
            # Bitwise, as the exchanges only move bytes (and fp8 lacks
            # torch.equal).
            assert torch.equal(
                decode_q.view(torch.uint8),
                local_heads[:num_decode_tokens].view(torch.uint8),
            )
            expected = full[num_decode_tokens:][shardq.local_token_ids]
            assert torch.equal(sharded.view(torch.uint8), expected.view(torch.uint8))
            # Attention outputs are a head-padded slice; unshard reads its
            # strides.
            padded = torch.zeros(sharded.shape[0], all_heads + 8, 512, device="cuda")
            padded[:, :all_heads] = sharded[..., :512].float()
            out = padded.bfloat16()[:, :all_heads]
            restored = unshard_out(out, shardq, decode_q[..., :512].bfloat16())
            assert torch.equal(restored, local_heads[..., :512].bfloat16())
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("use_sm", [False, True])
@pytest.mark.parametrize("num_decode_tokens", [0, 33])
def test_shard_q_round_trip(use_sm, q_dtype, num_decode_tokens):
    """The exchanges move each rank's tokens to it with all heads and back,
    through symmetric memory or NCCL, leaving the decode tokens head-sharded."""
    world = 2
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    mp.spawn(
        _round_trip,
        args=(world, get_open_port(), use_sm, q_dtype, num_decode_tokens),
        nprocs=world,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_indexer_upload_survives_next_build(monkeypatch):
    """Building the next step must not change a still-queued upload."""
    from vllm.v1.attention.backends.mla.indexer import upload_shardq_indices

    monkeypatch.setattr(
        shardq, "get_tp_group", lambda: SimpleNamespace(world_size=2, rank_in_group=0)
    )

    def upload(lens):
        split = shardq.ShardQSplit(0, *zigzag_split(np.array(lens), 2))
        qsl = torch.tensor(np.r_[0, np.cumsum(lens)], dtype=torch.int32)
        return upload_shardq_indices(split, qsl, torch.device("cuda"))[0], split

    torch.cuda._sleep(1 << 30)  # keep both copies queued
    first, split = upload([3000, 1200])
    upload([700])
    torch.cuda.synchronize()
    np.testing.assert_array_equal(first.cpu().numpy(), split.local_token_ids)
