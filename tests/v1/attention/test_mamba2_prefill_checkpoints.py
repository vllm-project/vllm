# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Internal prefill checkpoints for Mamba2 align-mode prefix caching."""

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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

compute_chunk_metadata = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata


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


def test_builder_emits_one_checkpoint_entry_per_prefill_row():
    """Declined rows stay in place, masked, and a chunk ends on the checkpoint."""
    builder = _create_mamba2_builder()
    # Row 0 (seq_len 900) checkpoints at 768; row 1 (seq_len 100) is too short
    # for a checkpoint at all.
    meta = _build(builder, seq_lens=[900, 100], query_lens=[900, 100])

    assert meta.checkpoint_chunk_idx is not None
    assert meta.checkpoint_meta is not None
    assert meta.checkpoint_chunk_idx.numel() == 2
    # The exporter masks row 1 off by its zero offset and null block, but
    # still loads its chunk index first, so the placeholder must be in range.
    assert meta.checkpoint_meta.checkpoint_offsets.tolist() == [768, 0]
    assert meta.checkpoint_meta.state_indices[1].item() == NULL_BLOCK_ID
    assert meta.checkpoint_chunk_idx[1].item() == 0
    # The checkpoint's token offset is the end of its chunk. Asserting it here
    # also proves the chunk split actually landed on the checkpoint.
    ckpt_chunk = meta.checkpoint_chunk_idx[0]
    assert meta.cu_chunk_seqlen_p[ckpt_chunk + 1].item() == 768


def test_update_block_table_regathers_checkpoint_blocks():
    """Groups sharing a spec reuse one group's metadata via update_block_table.

    Checkpoint destinations are block-table entries, so each group needs its
    own: layers of different groups share physical tensors, and reusing the
    first group's indices writes every group's checkpoint into its blocks.
    """
    builder = _create_mamba2_builder()
    meta = _build(builder, seq_lens=[900, 100], query_lens=[900, 100])
    assert meta.checkpoint_meta is not None

    # A second group's table: same shape as the first, disjoint block ids.
    other_table = torch.arange(1000, 1016, dtype=torch.int32).reshape(2, 8)
    updated = builder.update_block_table(
        meta, other_table, torch.zeros(1000, dtype=torch.int64)
    )

    # Row 0 checkpoints in column cdiv(900, 256) - 2 == 2; row 1 declined.
    assert updated.checkpoint_meta is not None
    assert updated.checkpoint_meta.state_indices.tolist() == [
        other_table[0, 2].item(),
        NULL_BLOCK_ID,
    ]


def test_builder_emits_nothing_outside_align_mode():
    builder = _create_mamba2_builder(mamba_cache_mode="none")
    meta = _build(builder, seq_lens=[900], query_lens=[900])

    assert meta.checkpoint_chunk_idx is None
