# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.models.common.ops import sequence_parallel as sp
from vllm.v1.worker.gpu.dsa_sequence_parallel import DSAQueryLayout


@pytest.mark.parametrize("zigzag", [False, True])
@pytest.mark.parametrize("model_rows", [1, 7, 8, 13, 32])
def test_dsa_shards_restore_global_token_order(monkeypatch, zigzag, model_rows):
    tp = 4
    # Different values in every dimension expose a row or feature permutation.
    x = torch.arange(model_rows * 6).reshape(model_rows, 2, 3)
    monkeypatch.setattr(sp, "get_tensor_model_parallel_world_size", lambda: tp)
    shards = []
    for rank in range(tp):
        monkeypatch.setattr(
            sp, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        shards.append(sp.sp_shard(x, zigzag))
    monkeypatch.setattr(sp, "sp_all_gather", lambda _: torch.cat(shards))
    torch.testing.assert_close(sp.sp_gather_global(shards[0], model_rows, zigzag), x)


def test_zigzag_segments_keep_original_causal_prefix():
    # One history-bearing request split around a gap, followed by another.
    offsets = np.array([0, 7, 8])
    layout = DSAQueryLayout.from_offsets(offsets, 8, 2, 0, True, False)
    assert layout.token_indices.tolist() == [0, 1, 6, 7]
    assert layout.req_indices.tolist() == [0, 0, 1]
    assert layout.query_start_loc.tolist() == [0, 2, 3, 4]
    assert layout.trailing_tokens.tolist() == [5, 0, 0]


@pytest.mark.parametrize("zigzag", [False, True])
def test_verify_drafts_after_rejection_preserves_each_tokens_causal_prefix(zigzag):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata

    # Two verification groups (bonus + two drafts) alongside a chunked prefill.
    cm = create_common_attn_metadata(
        BatchSpec(seq_lens=[33, 43, 67], query_lens=[3, 3, 7]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    # Async speculation can leave the host ahead of the accepted GPU lengths.
    cm.seq_lens_cpu_upper_bound = cm.seq_lens + torch.tensor([1, 2, 0])
    expected = torch.cat(
        (torch.arange(30, 33), torch.arange(40, 43), torch.arange(60, 67))
    )
    actual = torch.full_like(expected, -1)
    for rank in range(2):
        layout = DSAQueryLayout.from_offsets(
            cm.query_start_loc_cpu.numpy(), cm.num_actual_tokens, 2, rank, zigzag, False
        )
        local = layout.local_metadata(cm, buffers=None, capacity=0)
        starts = local.compute_num_computed_tokens()
        positions = torch.cat(
            [torch.arange(left, right) for left, right in zip(starts, local.seq_lens)]
        )
        actual[local.slot_mapping] = positions
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "zigzag, rank, seq_lens, segments",
    [(False, 0, [32, 33, 32], [3, 2]), (True, 1, [32, 32], [1, 3])],
)
def test_fp8_verify_segments_require_paged_kv(zigzag, rank, seq_lens, segments):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder,
        _sp_decode_in_prefill,
    )
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    cm = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=[3] * len(seq_lens)),
        block_size=16,
        device=torch.device("cpu"),
    )
    cm.is_prefilling = torch.zeros(cm.num_reqs, dtype=torch.bool)
    cm.seq_lens_cpu_upper_bound = cm.seq_lens + 2
    layout = DSAQueryLayout.from_offsets(
        cm.query_start_loc_cpu.numpy(), cm.num_actual_tokens, 2, rank, zigzag, False
    )
    local = layout.local_metadata(cm, buffers=None, capacity=0)
    assert local.query_start_loc_cpu.diff().tolist() == segments
    num_decodes, num_prefills, _, _ = split_decodes_and_prefills(
        local,
        decode_threshold=3,
        require_uniform=FlashMLASparseMetadataBuilder.require_uniform_decodes,
    )
    assert (num_decodes, num_prefills) == (1, 1)
    # The separate prefill gather would read a third KV page for this row.
    assert local.seq_lens[num_decodes] == 32
    assert local.seq_lens_cpu_upper_bound is not None
    assert local.seq_lens_cpu_upper_bound[num_decodes] == 34
    assert _sp_decode_in_prefill(local, num_decodes)
    # A real prompt with that same query shape has exact host lengths.
    assert local.is_prefilling is not None
    local.is_prefilling[num_decodes] = True
    assert not _sp_decode_in_prefill(local, num_decodes)


@pytest.mark.parametrize("zigzag", [False, True])
def test_decode_replay_keeps_capture_shape_on_partially_empty_rank(zigzag):
    for rank in range(4):
        captured = DSAQueryLayout.from_offsets(np.arange(9), 8, 4, rank, zigzag, True)
        replay = DSAQueryLayout.from_offsets(
            np.array([0, 1, 2, 2]), 8, 4, rank, zigzag, True
        )
        assert captured.token_indices.shape == replay.token_indices.shape
        assert torch.equal(captured.query_start_loc, replay.query_start_loc)
        assert all(row < 2 for row in replay.token_indices.tolist())


def test_local_queries_reuse_global_cache_and_keep_capture_buffers():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    offsets = torch.tensor([0, 7, 8], dtype=torch.int32)
    seq_lens = torch.tensor([107, 48], dtype=torch.int32)
    cm = CommonAttentionMetadata(
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=7,
        max_seq_len=107,
        block_table_tensor=torch.tensor([[3, 4], [9, 8]], dtype=torch.int32),
        slot_mapping=torch.arange(8) + 24,
        positions=torch.tensor([100, 101, 102, 103, 104, 105, 106, 47]),
    )
    layout = DSAQueryLayout.from_offsets(offsets.numpy(), 8, 2, 0, True, False)
    local = layout.local_metadata(cm, buffers=None, capacity=0)
    assert local.dsa_sp_global is cm
    assert local.seq_lens.tolist() == [102, 107, 48]
    assert local.slot_mapping.tolist() == [24, 25, 30, 31]
    assert local.positions is not None
    assert local.positions.tolist() == [100, 101, 106, 47]
    assert local.block_table_tensor.tolist() == [[3, 4], [3, 4], [9, 8]]

    # TP4 rank 2 owns no real rows when a one-token decode replays bucket 8.
    capture_offsets = torch.arange(9, dtype=torch.int32)
    cm = cm.replace(
        query_start_loc=capture_offsets,
        query_start_loc_cpu=capture_offsets,
        seq_lens=torch.arange(1, 9),
        seq_lens_cpu_upper_bound=torch.arange(1, 9),
        block_table_tensor=torch.arange(8)[:, None],
        num_reqs=8,
        max_query_len=1,
    )
    captured = DSAQueryLayout.from_offsets(capture_offsets.numpy(), 8, 4, 2, True, True)
    buffers: dict[str, torch.Tensor] = {}
    first = captured.local_metadata(cm, buffers, 8)
    replay = DSAQueryLayout.from_offsets(np.array([0, 1]), 8, 4, 2, True, True)
    second = replay.local_metadata(cm, buffers, 8)
    assert first.block_table_tensor.data_ptr() == second.block_table_tensor.data_ptr()
    assert first.slot_mapping.data_ptr() == second.slot_mapping.data_ptr()
    assert second.slot_mapping.tolist() == [-1, -1]
    assert second.seq_lens.tolist() == [1, 1]
