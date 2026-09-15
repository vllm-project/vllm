# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints for Mamba2 align-mode prefix caching."""

import torch

from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

compute_chunk_metadata = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata


def _chunk_ends(cu_chunk_seqlen: list[int]) -> list[int]:
    """Chunk end offsets. `cu_chunk_seqlen[1:]` already ends with the total."""
    return cu_chunk_seqlen[1:]


def test_no_checkpoint_offsets_preserves_legacy_chunking():
    """Passing no offsets must reproduce the pre-feature chunk layout."""
    num_computed = torch.tensor([0, 300])
    qsl = torch.tensor([0, 700, 1000])

    cu, seq_idx, last, ckpt = compute_chunk_metadata(256, 2, num_computed, qsl)

    # Row 0: 700 fresh tokens from sequence position 0 -> 256, 256, 188.
    # Row 1: 300 new tokens resuming at 300. 300 % 256 == 44, so it finishes
    # its partial chunk with 256 - 44 == 212 tokens, then 88.
    assert cu == [0, 256, 512, 700, 912, 1000]
    assert seq_idx == [0, 0, 0, 1, 1]
    assert last == [2, 4]
    assert ckpt == [-1, -1]


def test_checkpoint_forces_a_chunk_boundary_off_the_chunk_grid():
    """A checkpoint at a non-multiple of chunk_size still lands on a boundary.

    This is the case the design exists for: align-mode block sizes carry no
    chunk_size factor, so the checkpoint routinely falls off the grid.
    """
    num_computed = torch.tensor([0])
    qsl = torch.tensor([0, 700])

    cu, seq_idx, last, ckpt = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[100]
    )

    num_chunks = len(cu) - 1
    assert 100 in _chunk_ends(cu)
    assert cu[ckpt[0] + 1] == 100
    assert seq_idx == [0] * num_chunks
    assert last == [num_chunks - 1]


def test_chunking_realigns_to_the_grid_after_a_checkpoint():
    """After the forced break, chunks resume breaking on chunk_size.

    100 -> then 156 to reach the 256 grid, then full chunks. Without the
    realignment the second chunk would span 100..356 and cross a boundary.
    """
    num_computed = torch.tensor([0])
    qsl = torch.tensor([0, 700])

    cu, _, _, _ = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[100]
    )

    assert cu == [0, 100, 256, 512, 700]


def test_checkpoint_on_the_grid_adds_no_extra_chunk():
    """An already-aligned checkpoint reuses the existing boundary."""
    num_computed = torch.tensor([0])
    qsl = torch.tensor([0, 700])

    cu, _, _, ckpt = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[256]
    )

    assert cu == [0, 256, 512, 700]
    assert cu[ckpt[0] + 1] == 256


def test_checkpoint_on_a_resumed_request_is_relative_to_the_query():
    """Offsets count from the query start, not the sequence start."""
    num_computed = torch.tensor([300])
    qsl = torch.tensor([0, 500])

    cu, _, _, ckpt = compute_chunk_metadata(
        256, 1, num_computed, qsl, checkpoint_offsets_p=[212]
    )

    # Row resumes at sequence position 300. 300 % 256 == 44, so the partial
    # chunk is 212 tokens and already ends exactly at the checkpoint (query
    # offset 212 == sequence position 512) — no extra split needed.
    assert cu == [0, 212, 468, 500]
    assert ckpt == [0]


def test_only_the_requested_rows_checkpoint():
    """A zero offset leaves that row's chunking and index untouched."""
    num_computed = torch.tensor([0, 0])
    qsl = torch.tensor([0, 700, 1400])

    cu, _, last, ckpt = compute_chunk_metadata(
        256, 2, num_computed, qsl, checkpoint_offsets_p=[0, 100]
    )

    # Row 0 chunks 256/256/188 (indices 0-2); row 1 breaks at its offset 100,
    # which is absolute position 700 + 100 == 800.
    assert ckpt == [-1, 3]
    assert cu == [0, 256, 512, 700, 800, 956, 1212, 1400]
    assert last == [2, 6]
