# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
from vllm.model_executor.layers.sparse_attn_indexer import (
    _build_pcp_candidate_a2a_send_buffer,
    _exchange_pcp_candidates_to_origins,
    _pcp_candidate_a2a_selector_input,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_pcp_routed_prefill_chunks_from_gathered,
)


def _metadata(
    world: int,
    capacity: int,
    rows_by_rank: list[list[tuple[int, int, int]]],
) -> torch.Tensor:
    """Create rank-major (query_start, query_len, seq_len) descriptors."""
    metadata = torch.zeros((world * capacity, 4), dtype=torch.int32)
    for rank, rows in enumerate(rows_by_rank):
        for req_idx, (query_start, query_len, seq_len) in enumerate(rows):
            metadata[rank * capacity + req_idx] = torch.tensor(
                (1, query_start, query_len, seq_len), dtype=torch.int32
            )
    return metadata


def _flatten(chunks, field: str) -> torch.Tensor:
    return torch.cat([getattr(chunk, field) for chunk in chunks])


def test_pcp_dcp_route_aligns_uneven_dual_chunk_swap_rows() -> None:
    # Exact DualChunkSwap segmentation for 13 fresh-prefill tokens at PCP=4:
    # rank 0: [0,1], rank 1: [2,3]+[12], rank 2: [4,5]+[10,11],
    # rank 3: [6,7]+[8,9]. Runtime Q payloads are padded to four rows/rank.
    world = 4
    capacity = 4
    metadata = _metadata(
        world,
        capacity,
        [
            [(0, 2, 2)],
            [(0, 2, 4), (2, 1, 13)],
            [(0, 2, 6), (2, 2, 12)],
            [(0, 2, 8), (2, 2, 10)],
        ],
    )
    block_table = torch.arange(world * capacity, dtype=torch.int32).unsqueeze(1)

    plans = [
        build_pcp_routed_prefill_chunks_from_gathered(
            metadata,
            block_table,
            req_capacity_per_rank=capacity,
            dcp_rank=rank,
            dcp_world_size=world,
            workspace_size=1024,
            # Force several compute chunks; their route row order must still
            # align before the one packed candidate collective.
            max_logits_bytes=128,
        )
        for rank in range(world)
    ]
    assert len(plans[0]) > 1

    expected_sources = torch.tensor(
        [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int32
    )
    expected_source_rows = torch.tensor(
        [0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64
    )
    # Rank-major route order follows the two source segments on each rank.
    expected_global_causal_lens = [
        1,
        2,
        3,
        4,
        13,
        5,
        6,
        11,
        12,
        7,
        8,
        9,
        10,
    ]

    for rank, chunks in enumerate(plans):
        torch.testing.assert_close(_flatten(chunks, "source_ranks"), expected_sources)
        torch.testing.assert_close(
            _flatten(chunks, "source_token_indices"), expected_source_rows
        )
        # Every owner has identical row order and shape, but owner-local causal
        # counts differ exactly according to token-level DCP ownership.
        local_causal_counts = _flatten(chunks, "cu_seqlen_ke") - _flatten(
            chunks, "cu_seqlen_ks"
        )
        expected_counts = torch.tensor(
            [
                max(0, (length + world - 1 - rank) // world)
                for length in expected_global_causal_lens
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(local_causal_counts, expected_counts)

        local_source_rows = _flatten(chunks, "source_token_indices")[
            _flatten(chunks, "source_ranks") == rank
        ]
        torch.testing.assert_close(
            local_source_rows,
            torch.arange(expected_sources.eq(rank).sum(), dtype=torch.int64),
        )

    # Candidate merge alignment is rank invariant.
    for field in ("source_ranks", "source_token_indices"):
        reference = _flatten(plans[0], field)
        for plan in plans[1:]:
            torch.testing.assert_close(_flatten(plan, field), reference)


def test_pcp_dcp_route_handles_mixed_continued_and_zero_token_sources() -> None:
    world = 4
    capacity = 3
    # Query starts may be nonzero because replicated decode rows precede
    # prefill rows in a mixed batch. Rank 1 contributes no prefill rows.
    metadata = _metadata(
        world,
        capacity,
        [
            [(1, 2, 102)],
            [],
            [(0, 1, 1)],
            [(1, 2, 42), (3, 1, 80)],
        ],
    )
    block_table = torch.arange(world * capacity * 2, dtype=torch.int32).reshape(
        world * capacity, 2
    )

    chunks = build_pcp_routed_prefill_chunks_from_gathered(
        metadata,
        block_table,
        req_capacity_per_rank=capacity,
        dcp_rank=1,
        dcp_world_size=world,
        workspace_size=1024,
        max_logits_bytes=1024 * 1024,
    )

    torch.testing.assert_close(
        _flatten(chunks, "source_ranks"),
        torch.tensor([0, 0, 2, 3, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        _flatten(chunks, "source_token_indices"),
        torch.tensor([1, 2, 0, 1, 2, 3], dtype=torch.int64),
    )
    assert not _flatten(chunks, "source_ranks").eq(1).any()

    # Continued rows preserve their nonzero historical causal prefix.
    local_counts = _flatten(chunks, "cu_seqlen_ke") - _flatten(chunks, "cu_seqlen_ks")
    expected_global_causal_lens = [101, 102, 1, 41, 42, 80]
    torch.testing.assert_close(
        local_counts,
        torch.tensor(
            [max(0, (length + 2) // 4) for length in expected_global_causal_lens],
            dtype=torch.int32,
        ),
    )


def test_pcp_dcp_route_capacity_allows_two_segments_per_global_request() -> None:
    # max_num_seqs=2 permits four local DualChunkSwap segment rows. Exercise
    # three valid rows so a mistaken max_num_seqs-only capacity would reject
    # an otherwise supported batch.
    world = 2
    capacity = 4
    metadata = _metadata(
        world,
        capacity,
        [
            [(0, 1, 1), (1, 1, 4), (2, 1, 8)],
            [(0, 1, 2), (1, 1, 6), (2, 1, 10)],
        ],
    )
    chunks = build_pcp_routed_prefill_chunks_from_gathered(
        metadata,
        torch.zeros((world * capacity, 1), dtype=torch.int32),
        req_capacity_per_rank=capacity,
        dcp_rank=0,
        dcp_world_size=world,
        workspace_size=128,
        max_logits_bytes=1024,
    )
    assert _flatten(chunks, "source_ranks").numel() == 6


def test_owner_peer_prefill_reads_global_cache_without_dcp_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    local_cache = torch.empty((4, 64, 132), dtype=torch.uint8)
    global_peer_cache = torch.empty((15, 64, 132), dtype=torch.uint8)
    owner_block_tables = torch.zeros((4, 1, 2), dtype=torch.int32)
    translated_block_table = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    local_cu_seq_lens = torch.tensor([0, 3], dtype=torch.int32)
    chunk = DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros((1, 2), dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([0, 0], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([1, 3], dtype=torch.int32),
        cu_seq_lens=local_cu_seq_lens,
        token_to_seq=torch.zeros(3, dtype=torch.int32),
        total_seq_lens=3,
        token_start=0,
        token_end=2,
        num_reqs=1,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=3,
        max_local_total_seq_lens=3,
        pcp_owner_block_tables=owner_block_tables,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([3], dtype=torch.int32),
        max_seq_len=3,
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=2,
        prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
    )

    k_quant = torch.empty((3, 128), dtype=torch.uint8)
    k_scale = torch.empty((3, 4), dtype=torch.uint8)
    monkeypatch.setattr(
        sparse_indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer_name: metadata}),
    )
    monkeypatch.setattr(
        sparse_indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(get_simultaneous=lambda *_specs: (k_quant, k_scale)),
    )
    monkeypatch.setattr(
        sparse_indexer.current_platform, "fp8_dtype", lambda: torch.uint8
    )
    monkeypatch.setattr(sparse_indexer.current_platform, "is_xpu", lambda: False)

    translated_calls = []

    def fake_translate(block_tables, **kwargs):
        translated_calls.append((block_tables, kwargs))
        return translated_block_table

    gather_calls = []

    def fake_gather(cache, dst_k, dst_scale, block_table, cu_seq_lens):
        gather_calls.append((cache, dst_k, dst_scale, block_table, cu_seq_lens))

    monkeypatch.setattr(
        sparse_indexer, "build_rotated_dcp_peer_block_table", fake_translate
    )
    monkeypatch.setattr(
        sparse_indexer.ops, "cp_gather_indexer_k_quant_cache", fake_gather
    )
    monkeypatch.setattr(
        sparse_indexer,
        "fp8_fp4_mqa_logits",
        lambda *_args, **_kwargs: torch.zeros((2, 3), dtype=torch.float32),
    )
    monkeypatch.setattr(
        sparse_indexer.ops, "top_k_per_row_prefill", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        sparse_indexer,
        "_merge_dcp_topk_global",
        lambda *_args, **_kwargs: pytest.fail(
            "Direct peer prefill must not run the DCP candidate merge"
        ),
    )

    topk_indices = torch.full((2, 2), -1, dtype=torch.int32)
    result = sparse_indexer.sparse_attn_indexer(
        hidden_states=torch.empty((2, 16)),
        k_cache_prefix=layer_name,
        kv_cache=local_cache,
        q_quant=torch.empty((2, 128), dtype=torch.uint8),
        q_scale=None,
        k=None,
        weights=torch.ones((2, 1)),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2,
        head_dim=128,
        max_model_len=4096,
        total_seq_lens=3,
        topk_indices_buffer=topk_indices,
        skip_k_cache_insert=True,
        use_pcp=False,
        dcp_rank=3,
        dcp_world_size=4,
        cp_kv_cache_interleave_size=64,
        skip_topk_buffer_clear=True,
        pcp_peer_kv_cache=global_peer_cache,
        pcp_peer_block_stride=4,
    )
    second_result = sparse_indexer.sparse_attn_indexer(
        hidden_states=torch.empty((2, 16)),
        k_cache_prefix=layer_name,
        kv_cache=local_cache,
        q_quant=torch.empty((2, 128), dtype=torch.uint8),
        q_scale=None,
        k=None,
        weights=torch.ones((2, 1)),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2,
        head_dim=128,
        max_model_len=4096,
        total_seq_lens=3,
        topk_indices_buffer=topk_indices,
        skip_k_cache_insert=True,
        use_pcp=False,
        dcp_rank=3,
        dcp_world_size=4,
        cp_kv_cache_interleave_size=64,
        skip_topk_buffer_clear=True,
        pcp_peer_kv_cache=global_peer_cache,
        pcp_peer_block_stride=4,
    )

    assert result is topk_indices
    assert second_result is topk_indices
    # Per-forward metadata is shared across Indexer layers. The peer page
    # table is invariant across those layers and must be translated once.
    assert len(translated_calls) == 1
    assert translated_calls[0][0] is owner_block_tables
    assert translated_calls[0][1] == {
        "local_rank": 0,
        "peer_block_stride": 4,
        "cp_kv_cache_interleave_size": 64,
        "block_size": 64,
    }
    assert chunk.pcp_peer_block_table_key == (4, 64, 64)
    assert len(gather_calls) == 2
    assert gather_calls[0][0] is global_peer_cache
    assert gather_calls[0][1].data_ptr() == k_quant.data_ptr()
    assert gather_calls[0][1].shape == k_quant.shape
    assert gather_calls[0][2].data_ptr() == k_scale.data_ptr()
    assert gather_calls[0][2].shape == k_scale.shape
    assert gather_calls[0][3] is translated_block_table
    assert gather_calls[0][4] is local_cu_seq_lens


def test_pcp_candidate_a2a_layout_is_padded_and_owner_ordered() -> None:
    world = 3
    source_stride = 4
    topk = 2
    source_ranks = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int32)
    source_rows = torch.tensor([1, 3, 0, 1, 2], dtype=torch.int64)

    sends = []
    packed_by_owner = []
    for owner in range(world):
        packed = torch.empty((source_ranks.shape[0], topk, 2), dtype=torch.float32)
        for route_row in range(source_ranks.shape[0]):
            for candidate in range(topk):
                packed[route_row, candidate, 0] = (
                    owner * 100 + route_row * 10 + candidate
                )
                packed[route_row, candidate, 1] = (
                    owner * 1000 + route_row * 10 + candidate
                )
        packed_by_owner.append(packed)
        sends.append(
            _build_pcp_candidate_a2a_send_buffer(
                packed,
                source_ranks,
                source_rows,
                source_stride,
                world,
            )
        )

    # Simulate equal-split all_to_all_single: destination receives one
    # destination-indexed block from each source owner, in source-owner order.
    for destination in range(world):
        received = torch.stack(
            [sends[owner][destination] for owner in range(world)], dim=0
        )
        selector_input = _pcp_candidate_a2a_selector_input(received)
        assert selector_input.shape == (source_stride, world * topk, 2)

        valid_rows = source_rows[source_ranks == destination]
        for source_row in range(source_stride):
            if source_row not in valid_rows:
                assert torch.isneginf(selector_input[source_row, :, 0]).all()
                assert (selector_input[source_row, :, 1] == -1).all()
                continue
            route_row = int(
                ((source_ranks == destination) & (source_rows == source_row))
                .nonzero()
                .item()
            )
            # The A2A selector row preserves stable owner-major candidate
            # ordering and score/global-id tie semantics.
            expected_owner_ordered_row = torch.cat(
                [packed_by_owner[owner][route_row] for owner in range(world)]
            )
            assert torch.equal(selector_input[source_row], expected_owner_ordered_row)
            for owner in range(world):
                expected_scores = torch.tensor(
                    [
                        owner * 100 + route_row * 10 + candidate
                        for candidate in range(topk)
                    ],
                    dtype=torch.float32,
                )
                torch.testing.assert_close(
                    selector_input[source_row, owner * topk : (owner + 1) * topk, 0],
                    expected_scores,
                )


@pytest.mark.parametrize(
    "source_ranks,source_rows,error",
    [
        (
            torch.tensor([0, 0], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int64),
            "duplicate",
        ),
        (
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 0], dtype=torch.int64),
            "out-of-range",
        ),
        (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int64),
            "out-of-range",
        ),
    ],
)
def test_pcp_candidate_a2a_layout_fails_closed(
    source_ranks: torch.Tensor,
    source_rows: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _build_pcp_candidate_a2a_send_buffer(
            torch.zeros((2, 1, 2), dtype=torch.float32),
            source_ranks,
            source_rows,
            source_stride=2,
            dcp_world_size=2,
        )


def _pcp_candidate_a2a_nccl_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        sparse_indexer.get_dcp_group = lambda: SimpleNamespace(
            device_group=dist.group.WORLD
        )
        source_stride = 5
        topk = 3
        source_ranks = torch.tensor(
            [0, 0, 1, 2, 2, 2, 3],
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        source_rows = torch.tensor(
            [0, 4, 2, 0, 1, 4, 3],
            dtype=torch.int64,
            device=f"cuda:{rank}",
        )
        route = torch.arange(
            source_ranks.numel(), dtype=torch.float32, device=f"cuda:{rank}"
        )[:, None]
        candidate = torch.arange(topk, dtype=torch.float32, device=f"cuda:{rank}")[
            None, :
        ]
        packed = torch.empty(
            (source_ranks.numel(), topk, 2),
            dtype=torch.float32,
            device=f"cuda:{rank}",
        )
        packed[..., 0] = rank * 10_000 + route * 100 + candidate
        packed[..., 1] = rank * 1_000_000 + route * 100 + candidate
        send_buffer = _build_pcp_candidate_a2a_send_buffer(
            packed,
            source_ranks,
            source_rows,
            source_stride,
            world_size,
        )
        actual = _exchange_pcp_candidates_to_origins(send_buffer)
        assert actual.shape == (source_stride, world_size * topk, 2)

        host_ranks = source_ranks.cpu()
        host_rows = source_rows.cpu()
        for source_row in range(source_stride):
            matches = ((host_ranks == rank) & (host_rows == source_row)).nonzero()
            if matches.numel() == 0:
                assert torch.isneginf(actual[source_row, :, 0]).all()
                assert (actual[source_row, :, 1] == -1).all()
                continue
            route_row = int(matches.item())
            expected = torch.empty_like(actual[source_row])
            for owner in range(world_size):
                owner_slice = slice(owner * topk, (owner + 1) * topk)
                expected[owner_slice, 0] = (
                    owner * 10_000 + route_row * 100 + candidate[0]
                )
                expected[owner_slice, 1] = (
                    owner * 1_000_000 + route_row * 100 + candidate[0]
                )
            assert torch.equal(actual[source_row], expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=4)
def test_pcp_candidate_a2a_real_nccl_pcp4() -> None:
    world_size = 4
    if torch.cuda.device_count() < world_size:
        pytest.skip("production PCP candidate A2A requires four CUDA GPUs")
    mp.spawn(
        _pcp_candidate_a2a_nccl_worker,
        args=(world_size, get_open_port()),
        nprocs=world_size,
    )


@pytest.mark.parametrize(
    "metadata,workspace,error",
    [
        (
            torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0]], dtype=torch.int32),
            16,
            "non-negative and non-empty",
        ),
        (
            torch.tensor([[1, 0, 1, 100], [0, 0, 0, 0]], dtype=torch.int32),
            10,
            "exceeds the KV gather workspace",
        ),
    ],
)
def test_pcp_dcp_route_fails_closed_on_invalid_layouts(
    metadata: torch.Tensor,
    workspace: int,
    error: str,
) -> None:
    with pytest.raises((ValueError, RuntimeError), match=error):
        build_pcp_routed_prefill_chunks_from_gathered(
            metadata,
            torch.zeros((2, 1), dtype=torch.int32),
            req_capacity_per_rank=1,
            dcp_rank=0,
            dcp_world_size=2,
            workspace_size=workspace,
            max_logits_bytes=1024,
        )


@pytest.mark.parametrize(
    "metadata,error",
    [
        (
            torch.tensor(
                [[1, 0, 2, 2], [1, 1, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=torch.int32,
            ),
            "duplicate source query row",
        ),
        (
            torch.tensor(
                [[1, 2, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=torch.int32,
            ),
            "exceed the fixed source payload",
        ),
    ],
)
def test_pcp_dcp_route_validates_fixed_source_rows(
    metadata: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_pcp_routed_prefill_chunks_from_gathered(
            metadata,
            torch.zeros((4, 1), dtype=torch.int32),
            req_capacity_per_rank=2,
            dcp_rank=0,
            dcp_world_size=2,
            workspace_size=16,
            max_logits_bytes=1024,
            source_token_capacity_per_rank=3,
        )
