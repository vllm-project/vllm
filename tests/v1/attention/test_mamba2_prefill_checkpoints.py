# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints for Mamba2 align-mode prefix caching."""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

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


@pytest.mark.parametrize("hash_block_size", [16, 64, 256])
def test_checkpoint_position_is_reachable_at_alignment_one(hash_block_size):
    """Alignment 1 accepts every hash-block-aligned checkpoint.

    This is the property Task 1's chunk split buys: with the KDA-style
    alignment of 16 these cases would be declined whenever the offset is not
    a multiple of 16.
    """
    seq_len = 5 * hash_block_size + 7
    query_start = 0
    position = get_mamba_prefill_checkpoint_position(
        seq_len, hash_block_size, drop_eagle_block=False
    )

    assert is_mamba_prefill_checkpoint_valid(
        query_start=query_start,
        query_end=seq_len,
        checkpoint_position=position,
        hash_block_size=hash_block_size,
        mamba_block_size=hash_block_size * 4,
        checkpoint_alignment=1,
    )


def test_checkpoint_is_declined_without_an_alignment():
    """`prefill_checkpoint_alignment=None` means the backend cannot export."""
    assert not is_mamba_prefill_checkpoint_valid(
        query_start=0,
        query_end=1031,
        checkpoint_position=1024,
        hash_block_size=256,
        mamba_block_size=1024,
        checkpoint_alignment=None,
    )


MAMBA_BLOCK_SIZE = 256
CHUNK_SIZE = 256
DEVICE = torch.device("cpu")


def _create_mamba2_builder(
    mamba_cache_mode: str = "align",
    num_prefill_checkpoint_blocks: int = 1,
    prefill_checkpoint_alignment: int | None = 1,
) -> Mamba2AttentionMetadataBuilder:
    vllm_config = create_vllm_config(block_size=MAMBA_BLOCK_SIZE)
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    # get_mamba_chunk_size() reads `mamba_chunk_size` then `chunk_size`.
    vllm_config.model_config.hf_text_config.mamba_chunk_size = CHUNK_SIZE
    spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=prefill_checkpoint_alignment,
    )
    return Mamba2AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(builder, seq_lens, query_lens):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        MAMBA_BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    # create_common_attn_metadata leaves is_prefilling unset; the builder
    # asserts on it. Every row here is a full prefill.
    common = common.replace(is_prefilling=torch.ones(len(seq_lens), dtype=torch.bool))
    return builder.build(common_prefix_len=0, common_attn_metadata=common)


def test_builder_emits_compacted_checkpoint_tensors():
    """Only checkpointing rows appear, and a chunk ends on the checkpoint."""
    builder = _create_mamba2_builder()
    # Row 0 (seq_len 900) checkpoints at 768; row 1 (seq_len 100) is too short
    # for a checkpoint at all.
    meta = _build(builder, seq_lens=[900, 100], query_lens=[900, 100])

    assert meta.checkpoint_chunk_idx is not None
    assert meta.checkpoint_chunk_idx.numel() == 1
    assert meta.checkpoint_block_idx.numel() == 1
    # The checkpoint's token offset is the end of its chunk. Asserting it here
    # also proves the chunk split actually landed on the checkpoint.
    assert meta.cu_chunk_seqlen_p[meta.checkpoint_chunk_idx + 1].item() == 768


def test_builder_emits_nothing_when_the_predicate_declines():
    """A spec that cannot export must produce no checkpoint metadata.

    This is the invariant that stops `MambaManager` from allocating a block
    the model never writes.
    """
    builder = _create_mamba2_builder(
        num_prefill_checkpoint_blocks=0, prefill_checkpoint_alignment=None
    )
    meta = _build(builder, seq_lens=[900], query_lens=[900])

    assert meta.checkpoint_chunk_idx is None
    assert meta.checkpoint_block_idx is None


def test_builder_emits_nothing_outside_align_mode():
    builder = _create_mamba2_builder(mamba_cache_mode="all")
    meta = _build(builder, seq_lens=[900], query_lens=[900])

    assert meta.checkpoint_chunk_idx is None
