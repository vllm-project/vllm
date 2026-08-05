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

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
    build_mla_chunked_context_metadata,
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
):
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
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
        assert chunk.request_start >= previous_request_start
        previous_request_start = chunk.request_start

        starts = chunk.starts.tolist()
        seq_lens = chunk.seq_lens.tolist()
        assert len(starts) == len(seq_lens) == chunk.num_requests
        for offset, (start, length) in enumerate(zip(starts, seq_lens)):
            assert length > 0, "a chunk must not cover an empty context span"
            gathered.setdefault(chunk.request_start + offset, []).append(
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

    `accumulate_mla_context_chunk` merges exactly `continuation_token_end -
    token_start` tokens and writes the rest, so a chunk that continues a
    request other than its first, or reports the wrong boundary, folds the
    partial into the wrong token slice.
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
        assert chunk.continuation_token_end == query_start_loc[chunk.request_start + 1]
        assert chunk.token_start == query_start_loc[chunk.request_start]
        assert chunk.token_end == query_start_loc[chunk.request_end]
        assert chunk.query_start_loc.tolist() == [
            offset - chunk.token_start
            for offset in query_start_loc[chunk.request_start : chunk.request_end + 1]
        ]


def test_prefills_without_context_are_skipped_and_reported():
    """A context-free prefill is never chunked, only reported as a gap.

    No chunk covering it means its partial is undefined, so the builder has to
    hand the impl the token range to neutralize; if it stops, the final merge
    combines the suffix with uninitialized scratch.
    """
    metadata = build_chunked_context([64, 0, 64, 0], [4, 4, 4, 4], 1024)
    assert metadata is not None

    covered = {
        request
        for chunk in metadata.chunks
        for request in range(chunk.request_start, chunk.request_end)
    }
    assert covered == {0, 2}
    # Request 1 sits inside the covered token range; request 3 is past its end.
    assert metadata.prefill_tokens_with_context == 12
    assert metadata.empty_token_ranges == [(4, 8)]


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
            request = chunk.request_start + offset
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
        expected = []
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


@pytest.mark.parametrize("max_num_seqs", [1, 32, 256])
def test_workspace_size_does_not_scale_with_max_num_seqs(max_num_seqs):
    """The workspace only has to hold one page, not one page per request.

    Hybrid models inflate the attention page size to cover their state page
    (Kimi K3 reaches ~11000 rows at DEP16); a `max_num_seqs * block_size` floor
    would blow far past the deliberate 64k cap and force users to hand-tune
    `max_num_seqs`.
    """
    block_size = 11000
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=163840),
    )
    workspace_size = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
        vllm_config
    )
    assert workspace_size == 64 * 1024
    assert workspace_size >= block_size
