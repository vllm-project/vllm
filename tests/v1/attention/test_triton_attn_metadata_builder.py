# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


# 64 segments only while num_seqs * num_kv_heads * 16 < 188 SMs on SM12.0 and
# the 64-segment scratch holds every query token (e.g. multi-query verify).
@pytest.mark.parametrize(
    "builder_cls",
    [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder],
)
@pytest.mark.parametrize(
    "capability,num_kv_heads,num_seqs,query_len,segments",
    [
        ((12, 0), 1, 11, 1, 64),
        ((12, 0), 1, 12, 1, 16),
        ((12, 0), 8, 1, 1, 64),
        ((12, 0), 8, 2, 1, 16),
        ((9, 0), 1, 1, 1, 16),
        ((12, 0), 1, 2, 4, 64),
        ((12, 0), 1, 10, 4, 16),
    ],
)
def test_split_k_segments_follow_sm_occupancy(
    monkeypatch, builder_cls, capability, num_kv_heads, num_seqs, query_len, segments
):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap: cap == capability
    )
    monkeypatch.setattr(current_platform, "num_compute_units", lambda: 188)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: num_kv_heads,
            get_head_size=lambda: 128,
            rswa_window=None,
        ),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=num_kv_heads, head_size=128, dtype=torch.bfloat16
    )
    builder = builder_cls(spec, ["layer.0"], config, "cpu")
    batch = BatchSpec(seq_lens=[128] * num_seqs, query_lens=[query_len] * num_seqs)
    metadata = builder.build(0, create_common_attn_metadata(batch, 16, "cpu"))
    assert metadata.num_par_softmax_segments == segments
    assert metadata.softmax_segm_output.shape[2] == segments
    assert metadata.softmax_segm_max.shape[0] >= num_seqs * query_len
    # DiffKV re-allocates the output; it must stay sized like the base buffers.
    assert builder.softmax_segm_output.shape[0] == builder.softmax_segm_max.shape[0]
    assert (
        metadata.softmax_segm_output.data_ptr()
        == builder.softmax_segm_output.data_ptr()
    )


def _draft_builder(builder_cls, rswa_window: int | None):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 8,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: 128,
            rswa_window=rswa_window,
        ),
        parallel_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        speculative_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    return builder_cls(spec, ["layer.0"], config, "cpu")


def _draft_decode_batch(seq_lens: list[int]):
    batch = BatchSpec(seq_lens=seq_lens, query_lens=[1] * len(seq_lens))
    return create_common_attn_metadata(batch, 16, "cpu")


@pytest.mark.parametrize(
    "builder_cls",
    [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder],
)
@pytest.mark.parametrize("rswa_window", [None, 64])
def test_draft_decode_update_flag_matches_build_writes(builder_cls, rswa_window):
    """The update is a no-op, so the builder may claim support only when
    build() writes no builder-owned tensor that a captured graph reads."""
    builder = _draft_builder(builder_cls, rswa_window)
    common_metadata = _draft_decode_batch([5, 9])
    common_metadata.rswa_prefix_lens = torch.tensor([3, 7], dtype=torch.int32)

    def owned_bytes():
        # Bytes, not values: uninitialized scratch may hold NaNs.
        return {
            name: value.view(torch.uint8).clone()
            for name, value in vars(builder).items()
            if isinstance(value, torch.Tensor)
        }

    before = owned_bytes()
    builder.build(0, common_metadata)
    after = owned_bytes()

    written = [name for name in before if not torch.equal(before[name], after[name])]
    assert builder.supports_draft_decode_metadata_update is (not written)
    assert builder.supports_draft_decode_metadata_update is (rswa_window is None)


@pytest.mark.parametrize(
    "builder_cls",
    [TritonAttentionMetadataBuilder, TritonAttentionDiffKVMetadataBuilder],
)
def test_draft_decode_update_matches_fresh_build(builder_cls):
    """A replayed graph reads metadata built at capture; after an in-place
    update it must match a fresh build of the replayed batch."""
    torch.manual_seed(0)
    builder = _draft_builder(builder_cls, rswa_window=None)
    captured = _draft_decode_batch([100, 120])
    metadata = builder.build(0, captured)

    replayed = _draft_decode_batch([128, 30])
    # The replayed batch arrives through the same persistent input buffers.
    captured.seq_lens.copy_(replayed.seq_lens)
    captured.block_table_tensor.copy_(replayed.block_table_tensor)
    captured.slot_mapping.copy_(replayed.slot_mapping)
    builder.update_draft_decode_metadata(metadata)

    expected = _draft_builder(builder_cls, rswa_window=None).build(0, replayed)
    assert torch.equal(metadata.seq_lens, expected.seq_lens)
    assert torch.equal(metadata.block_table, expected.block_table)
    assert torch.equal(metadata.slot_mapping, expected.slot_mapping)
    assert torch.equal(metadata.query_start_loc, expected.query_start_loc)
