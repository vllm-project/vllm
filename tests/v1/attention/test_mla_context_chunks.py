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

import vllm.utils.gpu_sync_debug as gsd
from vllm.model_executor.layers.attention.mla_attention import (
    build_mla_chunked_context_metadata,
    init_mla_context_partial,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dcp_world_size", [1, 2])
def test_sparse_context_lengths_do_not_force_gpu_sync(monkeypatch, dcp_world_size):
    """Sparse MLA must pin computed context lengths before the H2D copy, and
    building the DCP context row indices must not sync either."""
    builder = SimpleNamespace(
        _context_lens_cpu=torch.empty(2, dtype=torch.int32, pin_memory=True),
        chunked_prefill_workspace=torch.empty((2048, 1)),
        chunked_prefill_workspace_size=1024,
        kv_cache_spec=SimpleNamespace(block_size=BLOCK_SIZE),
        device=torch.device("cuda"),
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=1,
        dcp_virtual_block_size=dcp_world_size,
        dcp_manager=None,
    )
    common_metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([17, 2052, 324], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 5, 9], dtype=torch.int32),
    )
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()
    metadata = gsd.with_gpu_sync_check(
        SparseMLACommonMetadataBuilder._build_chunked_context_fields
    )(
        builder,
        common_metadata,
        num_decodes=1,
        num_prefills=2,
        prefill_query_lens_cpu=torch.tensor([4, 4], dtype=torch.int32),
    )
    assert metadata is not None
    assert metadata.context_lens.device.type == "cuda"
    assert metadata.context_lens.tolist() == [2048, 320]


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
        # A zero-context request between active requests must not become an
        # empty TRTLLM ragged-attention row.
        ([4, 0, 2], 1024),
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
    assert metadata.context_lens.tolist() == context_lens

    gathered: dict[int, list[tuple[int, int]]] = {}
    previous_request_start = -1
    for chunk in metadata.chunks:
        assert chunk.all_rows_active
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
    way the global ones tile the real context, since the context row indices
    unpad using them.
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
    assert metadata.context_lens.tolist() == context_lens
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


@pytest.mark.parametrize(
    "dcp_world_size,interleave,context_lens,workspace_size",
    [
        # The long request spans several chunks.
        (2, 64, [3000, 200, 200], 1024),
        # A single one-token context: one rank holds it, the other only padding.
        (2, 1, [1], 1024),
        # Empty contexts and contexts that leave ranks with fewer rows.
        (4, 16, [1000, 0, 77, 5, 64, 300], 1024),
        (8, 1, [129, 7, 8, 9, 63, 64, 65], 512),
        # Lengths around the virtual block size.
        (8, 64, [1, 511, 512, 513, 4000], 1024),
        # Many requests in one chunk.
        (16, 8, [5] * 40 + [2000], 2048),
    ],
)
def test_dcp_context_row_indices_unshard_the_context(
    dcp_world_size, interleave, context_lens, workspace_size
):
    """``context_row_indices`` picks each chunk's context rows out of the
    all-gathered DCP workspace and skips the padding: request by request, each
    rank's rows of the chunk in turn. They are the chunk's span of the
    request's context, and the chunks cover every context row once."""
    metadata = build_chunked_context(
        context_lens,
        [4] * len(context_lens),
        workspace_size,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=interleave,
    )
    assert metadata is not None

    # Token t of request i is i * 100_000 + t. Blocks of `interleave` tokens
    # go to the ranks in turn, so each rank holds a shard in local order.
    def local_tokens(request: int, rank: int) -> torch.Tensor:
        tokens = torch.arange(context_lens[request])
        shard = tokens[(tokens // interleave) % dcp_world_size == rank]
        return request * 100_000 + shard

    cursor = [0] * len(context_lens)
    for chunk in metadata.chunks:
        assert chunk.padded_local_seq_lens is not None
        assert chunk.local_context_lens_allranks is not None
        assert chunk.local_starts is not None
        assert chunk.context_row_indices is not None
        assert chunk.context_row_indices.dtype == torch.int64

        # Each rank's local gather: every request's padded local rows from its
        # local start on, -1 past the end of the rank's shard.
        requests = range(chunk.request_slice.start, chunk.request_slice.stop)
        rank_rows = []
        for rank in range(dcp_world_size):
            rows = []
            for request, start, padded, local_lens in zip(
                requests,
                chunk.local_starts,
                chunk.padded_local_seq_lens,
                chunk.local_context_lens_allranks,
            ):
                shard = local_tokens(request, rank)
                assert shard.numel() == local_lens[rank]
                padded_rows = torch.full((padded,), -1)
                kept = shard[start : start + padded]
                padded_rows[: kept.numel()] = kept
                rows.append(padded_rows)
            rank_rows.append(torch.cat(rows))
            assert rank_rows[-1].numel() == chunk.num_local_context_tokens
        allgathered = torch.cat(rank_rows)

        expected = []
        for request, start, padded, length in zip(
            requests,
            chunk.local_starts,
            chunk.padded_local_seq_lens,
            chunk.seq_lens.tolist(),
        ):
            tokens = torch.cat(
                [
                    local_tokens(request, rank)[start : start + padded]
                    for rank in range(dcp_world_size)
                ]
            )
            span = torch.arange(cursor[request], cursor[request] + length)
            assert torch.equal(tokens.sort().values, request * 100_000 + span)
            cursor[request] += length
            expected.append(tokens)

        assert torch.equal(allgathered[chunk.context_row_indices], torch.cat(expected))
    assert cursor == context_lens


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dims", [(512, 64), (384, 48), (512, 0)])
@pytest.mark.parametrize("num_rows", [0, 33, 1000])
def test_gather_dcp_context_rows_copies_rows(num_rows, dims, dtype):
    """The gather equals indexing the latent and rope parts of the rows, also
    for NoPE MLA layers, which have no rope part."""
    from vllm.v1.attention.ops.mla_dcp_context import gather_dcp_context_rows

    kv_lora_rank, rope_dim = dims
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    gathered = torch.randn(
        (1200, kv_lora_rank + rope_dim), device="cuda", generator=generator
    ).to(dtype)
    row_indices = torch.randint(
        0, 1200, (num_rows,), device="cuda", generator=generator
    )

    latent, k_pe = gather_dcp_context_rows(gathered, row_indices, kv_lora_rank)

    expected = gathered[row_indices].unsqueeze(1)
    assert latent.shape == (num_rows, 1, kv_lora_rank)
    assert k_pe.shape == (num_rows, 1, rope_dim)
    assert latent.dtype == k_pe.dtype == dtype
    assert latent.is_contiguous() and k_pe.is_contiguous()
    assert torch.equal(_bits(latent), _bits(expected[..., :kv_lora_rank]))
    assert torch.equal(_bits(k_pe), _bits(expected[..., kv_lora_rank:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope_dim", [64, 0])
def test_gather_dcp_context_rows_warmup_covers_the_context_loop(dtype, rope_dim):
    """The JIT warmup compiles the variant that the context loop launches: rows
    of the all-gather part of the workspace, int64 row indices taken at any
    offset into the batch's indices, and any number of rows."""
    from vllm.v1.attention.ops.mla_dcp_context import (
        _gather_context_rows,
        gather_dcp_context_rows,
    )

    kv_lora_rank = 512
    kernel = _gather_context_rows.kernel

    def compiled_variants() -> set:
        return {key for caches in kernel.device_caches.values() for key in caches[0]}

    kernel.device_caches.clear()
    _gather_context_rows.warmup(
        dtype=dtype, kv_lora_rank=kv_lora_rank, rope_dim=rope_dim
    )
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    workspace_rows, dcp_world_size = 1024, 4
    workspace = torch.zeros(
        (workspace_rows + workspace_rows // dcp_world_size, kv_lora_rank + rope_dim),
        dtype=dtype,
        device="cuda",
    )
    gathered = workspace[workspace_rows // dcp_world_size :]
    row_indices = torch.arange(workspace_rows, device="cuda")
    for start, num_rows in ((0, 1), (1, 33), (3, 1000)):
        gather_dcp_context_rows(
            gathered, row_indices[start : start + num_rows], kv_lora_rank
        )
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


def test_chunked_context_metadata_owns_context_lengths():
    """Reusing CPU scratch must not overwrite an earlier batch's context lengths."""
    context_lens = torch.tensor([16, 32], dtype=torch.int32)
    metadata = build_mla_chunked_context_metadata(
        context_lens_cpu=context_lens,
        prefill_query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        chunked_prefill_workspace=torch.empty((64, 1)),
        chunked_prefill_workspace_size=64,
        block_size=BLOCK_SIZE,
        align_chunk_to_block=True,
        device=torch.device("cpu"),
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
    )
    context_lens.zero_()
    assert metadata is not None
    assert metadata.context_lens.tolist() == [16, 32]
