# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request scheduling of MLA chunked-context prefill.

These cover the metadata contract `_compute_prefill_context` relies on: every
context row is gathered exactly once, no chunk exceeds the workspace, and a
chunk never covers a prefill without context (which is why the partial no
longer needs an empty-span masking pass).
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.attention.mla_attention as mla_attention
from vllm.model_executor.layers.attention.mla_attention import (
    build_dcp_kv_final_layout_dst_rows,
    build_mla_chunked_context_metadata,
    init_mla_context_partial,
    reorg_kvcache,
)

BLOCK_SIZE = 16


def build_chunked_context(
    context_lens: list[int],
    query_lens: list[int],
    workspace_size: int,
    block_size: int = BLOCK_SIZE,
    dcp_world_size: int = 1,
    dcp_local_block_size: int = 1,
    dcp_manager=None,
):
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    if dcp_world_size == 1:
        workspace_rows = workspace_size
    elif dcp_manager is not None and dcp_manager.use_direct_kv_gather:
        workspace_rows = workspace_size // dcp_world_size
    else:
        workspace_rows = workspace_size + workspace_size // dcp_world_size
    return build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc,
        chunked_prefill_workspace=torch.empty((workspace_rows, 1)),
        chunked_prefill_workspace_size=workspace_size,
        block_size=block_size,
        align_chunk_to_block=True,
        device=torch.device("cpu"),
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=dcp_local_block_size,
        dcp_virtual_block_size=dcp_local_block_size * dcp_world_size,
        dcp_manager=dcp_manager,
    )


@pytest.mark.parametrize(
    "context_lens,workspace_size",
    [
        # Heterogeneous batch: the long request must not shrink the chunk the
        # short ones share.
        ([960, 16, 16, 16, 16, 16, 16, 16], 1024),
        # A single request that does not fit is split on its own.
        ([2048, 320], 1024),
        # Every request needs its own chunk.
        ([1024, 1024, 1024], 1024),
        # Unaligned tails.
        ([37, 1000, 5], 1024),
        ([1], 1024),
    ],
)
def test_chunks_gather_every_context_row_exactly_once(context_lens, workspace_size):
    """Chunks tile the batch's context without gaps, overlap, or overflow.

    This is the invariant the whole schedule rests on: attention over a chunk
    only sees the rows that chunk gathered, so a row counted twice or dropped
    silently corrupts the context partial.
    """
    query_lens = [4] * len(context_lens)
    metadata = build_chunked_context(context_lens, query_lens, workspace_size)
    assert metadata is not None

    gathered: dict[int, list[tuple[int, int]]] = {}
    previous_request_start = -1
    for chunk in metadata.chunks:
        assert chunk.num_context_tokens <= workspace_size
        assert chunk.num_context_tokens == int(chunk.cu_seq_lens[-1])
        assert chunk.token_to_seq.shape[0] == chunk.num_context_tokens
        # Chunks are emitted in request order, which is what lets the partial
        # treat only a chunk's first request as a continuation.
        assert chunk.request_slice.start >= previous_request_start
        previous_request_start = chunk.request_slice.start

        starts = chunk.starts.tolist()
        seq_lens = chunk.seq_lens.tolist()
        assert len(starts) == len(seq_lens) == chunk.num_requests
        for offset, (start, length) in enumerate(zip(starts, seq_lens)):
            assert length > 0, "a chunk must not cover an empty context span"
            gathered.setdefault(chunk.request_slice.start + offset, []).append(
                (start, length)
            )

    expected = {i: length for i, length in enumerate(context_lens) if length > 0}
    assert gathered.keys() == expected.keys()
    for request, spans in gathered.items():
        cursor = 0
        for start, length in spans:
            assert start == cursor
            cursor += length
        assert cursor == expected[request]


def test_continuation_is_confined_to_a_chunks_first_request():
    """Only a chunk's first request may continue an earlier chunk.

    `accumulate_mla_context_chunk` merges the continuation token slice and
    writes the rest, so a chunk that continues a request other than its first,
    or reports the wrong boundary, folds the partial into the wrong tokens.
    """
    context_lens = [2048, 32, 32]
    query_lens = [3, 5, 7]
    metadata = build_chunked_context(context_lens, query_lens, 1024)
    assert metadata is not None
    query_start_loc = [0, 3, 8, 15]

    assert [chunk.is_continuation for chunk in metadata.chunks] == [
        False,
        True,
        False,
    ]
    for chunk in metadata.chunks:
        starts = chunk.starts.tolist()
        assert chunk.is_continuation == (starts[0] > 0)
        # Requests after the first always start at the beginning of their
        # context, so they can only ever be initialized, never merged.
        assert all(start == 0 for start in starts[1:])
        request_start = chunk.request_slice.start
        request_end = chunk.request_slice.stop
        assert chunk.continuation_token_end == query_start_loc[request_start + 1]
        assert chunk.token_slice == slice(
            query_start_loc[request_start], query_start_loc[request_end]
        )
        assert chunk.query_start_loc.tolist() == [
            offset - chunk.token_slice.start
            for offset in query_start_loc[request_start : request_end + 1]
        ]


def test_tail_splitting_minimizes_chunks():
    """A tail request may fill one chunk and continue at the next chunk's head.

    Without tail splitting these contexts need three chunks. Splitting on the
    block boundary reduces that to the workspace lower bound of two.
    """
    metadata = build_chunked_context([768, 512, 512], [3, 5, 7], 1024)
    assert metadata is not None

    assert len(metadata.chunks) == 2
    first, second = metadata.chunks
    assert first.starts.tolist() == [0, 0]
    assert first.seq_lens.tolist() == [768, 256]
    assert not first.is_continuation

    assert second.starts.tolist() == [256, 0]
    assert second.seq_lens.tolist() == [256, 512]
    assert second.is_continuation


def test_prefills_without_context_are_skipped_and_neutralized():
    """Context-free prefills get neutral partials instead of chunk work.

    If the full-query partial leaves either an internal or trailing gap
    uninitialized, the final merge combines the suffix with undefined scratch.
    """
    metadata = build_chunked_context([64, 0, 64, 0], [4, 4, 4, 4], 1024)
    assert metadata is not None

    covered = {
        request
        for chunk in metadata.chunks
        for request in range(chunk.request_slice.start, chunk.request_slice.stop)
    }
    assert covered == {0, 2}
    assert metadata.empty_token_slices == [slice(4, 8), slice(12, 16)]

    output, output_lse = init_mla_context_partial(
        metadata,
        attn_output=torch.empty(4, 2, 3),
        attn_softmax_lse=torch.empty(2, 4),
        num_tokens=16,
    )
    assert output.shape == (16, 2, 3)
    assert output_lse.shape == (2, 16)
    for token_slice in metadata.empty_token_slices:
        assert not torch.count_nonzero(output[token_slice])
        assert torch.isneginf(output_lse[:, token_slice]).all()


def test_no_context_needs_no_chunks():
    assert build_chunked_context([0, 0], [4, 4], 1024) is None


def test_dcp_chunks_fit_the_per_rank_row_budget():
    """Under DCP each rank gathers its own shard, so the budget is 1/world.

    The local starts and lengths must tile the rank's padded context the same
    way the global ones tile the real context, since `reorg_kvcache` unpads
    using them.
    """
    dcp_world_size, interleave, workspace_size = 2, 64, 1024
    context_lens = [3000, 200, 200]
    metadata = build_chunked_context(
        context_lens,
        [4] * len(context_lens),
        workspace_size,
        block_size=128,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=interleave,
    )
    assert metadata is not None
    virtual_block_size = interleave * dcp_world_size

    local_cursor: dict[int, int] = {}
    for chunk in metadata.chunks:
        assert chunk.num_local_context_tokens <= workspace_size // dcp_world_size
        assert chunk.local_starts == chunk.starts.tolist()
        assert chunk.padded_local_cu_seq_lens.tolist() == [0] + list(
            torch.tensor(chunk.padded_local_seq_lens).cumsum(0)
        )
        assert (
            chunk.padded_local_token_to_seq.shape[0] == chunk.num_local_context_tokens
        )
        for offset, (start, length) in enumerate(
            zip(chunk.local_starts, chunk.padded_local_seq_lens)
        ):
            request = chunk.request_slice.start + offset
            assert start == local_cursor.get(request, 0)
            local_cursor[request] = start + length

    for request, length in enumerate(context_lens):
        padded_local = -(-length // virtual_block_size) * interleave
        assert local_cursor[request] == padded_local


def test_dcp_reorg_uses_each_chunks_local_starts():
    """Reorganization drops DCP padding at each request's own chunk offset."""
    dcp_world_size, interleave, workspace_size = 2, 64, 1024
    metadata = build_chunked_context(
        [3000, 200, 200],
        [4, 4, 4],
        workspace_size,
        block_size=128,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=interleave,
    )
    assert metadata is not None

    for chunk in metadata.chunks:
        assert chunk.padded_local_seq_lens is not None
        assert chunk.local_context_lens_allranks is not None
        assert chunk.local_starts is not None

        toks = chunk.num_local_context_tokens
        rank_buffers = [torch.full((toks, 1, 1), -1) for _ in range(dcp_world_size)]
        expected: list[torch.Tensor] = []
        src_token_idx = 0
        for request, (padded_len, local_lens, local_start) in enumerate(
            zip(
                chunk.padded_local_seq_lens,
                chunk.local_context_lens_allranks,
                chunk.local_starts,
            )
        ):
            for rank, local_len in enumerate(local_lens):
                actual_len = min(max(0, local_len - local_start), padded_len)
                values = (
                    request * 100_000
                    + rank * 10_000
                    + torch.arange(local_start, local_start + actual_len)
                ).view(-1, 1, 1)
                rank_buffers[rank][src_token_idx : src_token_idx + actual_len] = values
                expected.append(values)
            src_token_idx += padded_len

        allgathered = torch.cat(rank_buffers)
        reorganized, _ = reorg_kvcache(
            allgathered,
            allgathered,
            padded_local_chunk_seq_lens_lst=chunk.padded_local_seq_lens,
            local_context_lens_allranks=chunk.local_context_lens_allranks,
            local_starts=chunk.local_starts,
            sum_seq_len=chunk.num_context_tokens,
            max_seq_len=chunk.max_seq_len,
            toks=toks,
        )

        torch.testing.assert_close(reorganized, torch.cat(expected))


@pytest.mark.skip_global_cleanup
def test_dcp_final_layout_dst_rows_maps_padding_and_continuations():
    """The publish map is compact, disjoint, and skips padded local rows."""
    padded_local_seq_lens = [4, 3]
    local_context_lens_allranks = [[4, 3], [7, 5]]
    local_starts = [0, 4]

    rank_0 = build_dcp_kv_final_layout_dst_rows(
        padded_local_seq_lens,
        local_context_lens_allranks,
        local_starts,
        [7, 4],
        dcp_rank=0,
    )
    rank_1 = build_dcp_kv_final_layout_dst_rows(
        padded_local_seq_lens,
        local_context_lens_allranks,
        local_starts,
        [7, 4],
        dcp_rank=1,
    )

    assert rank_0.tolist() == [0, 1, 2, 3, 7, 8, 9]
    assert rank_1.tolist() == [4, 5, 6, -1, 10, -1, -1]
    valid_rows = sorted(row for row in [*rank_0.tolist(), *rank_1.tolist()] if row >= 0)
    assert valid_rows == list(range(11))

    zero_valid_maps = [
        build_dcp_kv_final_layout_dst_rows(
            padded_local_seq_lens=[2],
            local_context_lens_allranks=[[4, 4, 3, 2]],
            local_starts=[2],
            output_seq_lens=[5],
            dcp_rank=rank,
        ).tolist()
        for rank in range(4)
    ]
    assert zero_valid_maps == [[0, 1], [2, 3], [4, -1], [-1, -1]]


@pytest.mark.skip_global_cleanup
def test_dcp_final_layout_validates_each_request_exact_coverage():
    """Equal batch totals cannot hide a gap in one request and overlap in another."""
    with pytest.raises(ValueError, match="exactly cover request 0"):
        build_dcp_kv_final_layout_dst_rows(
            padded_local_seq_lens=[2, 2],
            local_context_lens_allranks=[[1, 1], [2, 2]],
            local_starts=[0, 0],
            output_seq_lens=[3, 3],
            dcp_rank=0,
        )

    mismatched_manager = SimpleNamespace(
        use_direct_kv_gather=True,
        group=SimpleNamespace(rank_in_group=0, world_size=4),
    )
    with pytest.raises(ValueError, match="world sizes differ"):
        build_chunked_context(
            [128],
            [4],
            1024,
            dcp_world_size=2,
            dcp_local_block_size=64,
            dcp_manager=mismatched_manager,
        )


@pytest.mark.skip_global_cleanup
def test_dcp_final_layout_publish_matches_reorg(monkeypatch):
    """All source-rank maps jointly produce the existing compact layout."""
    dcp_world_size, interleave, workspace_size = 2, 64, 1024
    monkeypatch.setattr(mla_attention, "np_to_pinned_tensor", torch.from_numpy)
    metadata_by_rank = []
    for rank in range(dcp_world_size):
        dcp_manager = SimpleNamespace(
            use_direct_kv_gather=True,
            group=SimpleNamespace(rank_in_group=rank, world_size=dcp_world_size),
        )
        metadata = build_chunked_context(
            [3000, 200, 200],
            [4, 4, 4],
            workspace_size,
            block_size=128,
            dcp_world_size=dcp_world_size,
            dcp_local_block_size=interleave,
            dcp_manager=dcp_manager,
        )
        assert metadata is not None
        assert metadata.workspace.shape[0] == workspace_size // dcp_world_size
        assert all(
            chunk.num_local_context_tokens <= workspace_size // dcp_world_size
            for chunk in metadata.chunks
        )
        metadata_by_rank.append(metadata)

    for chunk_index, chunk in enumerate(metadata_by_rank[0].chunks):
        assert chunk.padded_local_seq_lens is not None
        assert chunk.local_context_lens_allranks is not None
        assert chunk.local_starts is not None

        toks = chunk.num_local_context_tokens
        compact = torch.full((chunk.num_context_tokens,), -1, dtype=torch.int64)
        expected: list[list[torch.Tensor]] = []
        for rank in range(dcp_world_size):
            local = torch.full((toks,), -2, dtype=torch.int64)
            src_token_idx = 0
            for request, (padded_len, local_lens, local_start) in enumerate(
                zip(
                    chunk.padded_local_seq_lens,
                    chunk.local_context_lens_allranks,
                    chunk.local_starts,
                )
            ):
                actual_len = min(max(0, local_lens[rank] - local_start), padded_len)
                values = (
                    request * 100_000
                    + rank * 10_000
                    + torch.arange(local_start, local_start + actual_len)
                )
                local[src_token_idx : src_token_idx + actual_len] = values
                if rank == 0:
                    expected.append([])
                expected[request].append(values)
                src_token_idx += padded_len

            dst_rows = metadata_by_rank[rank].chunks[chunk_index].final_layout_dst_rows
            assert dst_rows is not None
            valid = dst_rows >= 0
            assert torch.all(compact[dst_rows[valid]] == -1)
            compact[dst_rows[valid]] = local[valid]

        assert not torch.any(compact == -1)
        torch.testing.assert_close(
            compact,
            torch.cat([torch.cat(request) for request in expected]),
        )
