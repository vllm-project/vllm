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
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)

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


def test_builder_emits_nothing_outside_align_mode():
    builder = _create_mamba2_builder(mamba_cache_mode="all")
    meta = _build(builder, seq_lens=[900], query_lens=[900])

    assert meta.checkpoint_chunk_idx is None


def _call_get_kv_cache_spec(monkeypatch, mamba_cache_mode: str) -> MambaSpec:
    """Drive MambaMixer2.get_kv_cache_spec without building a real layer.

    Patching the base implementation means `super()` returns a known spec, so
    none of the layer's instance state (or distributed init) is touched.
    """
    base = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_cache_mode=mamba_cache_mode,
    )
    monkeypatch.setattr(MambaBase, "get_kv_cache_spec", lambda self, cfg: base)
    vllm_config = create_vllm_config(block_size=MAMBA_BLOCK_SIZE)
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    layer = object.__new__(MambaMixer2)
    return MambaMixer2.get_kv_cache_spec(layer, vllm_config)


def test_align_mode_opts_into_unaligned_checkpoints(monkeypatch):
    """Align mode requests one checkpoint block at any token position."""
    spec = _call_get_kv_cache_spec(monkeypatch, "align")

    assert spec.num_prefill_checkpoint_blocks == 1
    assert spec.prefill_checkpoint_alignment == 1


@pytest.mark.parametrize("mode", ["all", "none"])
def test_non_align_modes_do_not_opt_in(monkeypatch, mode):
    """`all` already caches every boundary; `none` has no prefix cache."""
    spec = _call_get_kv_cache_spec(monkeypatch, mode)

    assert spec.num_prefill_checkpoint_blocks == 0
    assert spec.prefill_checkpoint_alignment is None


@pytest.mark.parametrize("hash_block_size", [16, 64, 256])
def test_eagle_block_drop_moves_the_checkpoint_one_block_earlier(hash_block_size):
    """Eagle prunes the last matching block, so the checkpoint backs off one.

    Getting this wrong writes a checkpoint at a position the cache will not
    look for, so the block is registered but never hit.
    """
    seq_len = 19 * hash_block_size + 7

    plain = get_mamba_prefill_checkpoint_position(
        seq_len, hash_block_size, drop_eagle_block=False
    )
    dropped = get_mamba_prefill_checkpoint_position(
        seq_len, hash_block_size, drop_eagle_block=True
    )

    assert plain == 19 * hash_block_size
    assert dropped == plain - hash_block_size


@pytest.mark.parametrize("mamba_block_size", [16, 256, 2096])
def test_valid_checkpoint_always_leaves_room_for_the_conv_window(mamba_block_size):
    """A valid checkpoint is never close enough to the query start to underflow.

    The mixer reads the `conv_state.shape[-1]` tokens preceding the checkpoint
    (conv_kernel - 1 + num_spec, so 6 for conv_kernel 4 with num_spec 3). If an
    offset below that were reachable, the window would index off the front of
    the batch and silently wrap. It is not: the predicate's
    `checkpoint_col > initial_state_col` term forces the checkpoint past a
    block boundary the query start has not reached, so the offset is at least
    half a block. K3 guards this explicitly (kda.py:424); this test is why we
    do not need to.
    """
    candidates = (1, 2, 4, 8, 16, 128, 1048, 2096)
    hash_sizes = [h for h in candidates if mamba_block_size % h == 0]
    checked = 0
    for hash_block_size in hash_sizes:
        for drop_eagle_block in (False, True):
            for query_end in range(mamba_block_size + 1, 4 * mamba_block_size, 7):
                position = get_mamba_prefill_checkpoint_position(
                    query_end, hash_block_size, drop_eagle_block=drop_eagle_block
                )
                for query_start in range(0, query_end, hash_block_size):
                    if not is_mamba_prefill_checkpoint_valid(
                        query_start=query_start,
                        query_end=query_end,
                        checkpoint_position=position,
                        hash_block_size=hash_block_size,
                        mamba_block_size=mamba_block_size,
                        checkpoint_alignment=1,
                    ):
                        continue
                    checked += 1
                    assert position - query_start >= mamba_block_size // 2, (
                        f"offset {position - query_start} too small: "
                        f"block={mamba_block_size} hash={hash_block_size} "
                        f"start={query_start} end={query_end} drop={drop_eagle_block}"
                    )
    assert checked > 0, "no valid checkpoints exercised; test proves nothing"
