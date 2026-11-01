# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadataBuilder

BLOCK_SIZE = 16


def _draft_builder(builder_cls):
    vllm_config = create_vllm_config(block_size=BLOCK_SIZE)
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    with set_current_vllm_config(vllm_config):
        return builder_cls(kv_cache_spec, ["layer.0"], vllm_config, "cpu")


def _draft_decode_batch(seq_lens: list[int]):
    batch = BatchSpec(seq_lens=seq_lens, query_lens=[1] * len(seq_lens))
    return create_common_attn_metadata(batch, BLOCK_SIZE, "cpu")


def test_unified_builder_opts_into_draft_decode_metadata_update():
    """The update is a no-op, so the builder may claim support only when
    build() writes no builder-owned tensor that a captured graph reads.
    ROCM_ATTN shares build() but stays opted out until it is validated."""
    unified = _draft_builder(RocmAiterUnifiedAttentionMetadataBuilder)
    parent = _draft_builder(RocmAttentionMetadataBuilder)
    assert unified.supports_draft_decode_metadata_update is True
    assert parent.supports_draft_decode_metadata_update is False

    def owned_bytes():
        # Bytes, not values: uninitialized scratch may hold NaNs.
        return {
            name: value.view(torch.uint8).clone()
            for name, value in vars(unified).items()
            if isinstance(value, torch.Tensor)
        }

    before = owned_bytes()
    unified.build(0, _draft_decode_batch([5, 9]))
    after = owned_bytes()
    written = [name for name in before if not torch.equal(before[name], after[name])]
    assert written == []


def test_unified_draft_decode_update_matches_fresh_build():
    """A replayed graph reads metadata built at capture; after an in-place
    update it must match a fresh build of the replayed batch."""
    torch.manual_seed(0)
    builder = _draft_builder(RocmAiterUnifiedAttentionMetadataBuilder)
    captured = _draft_decode_batch([100, 120])
    metadata = builder.build(0, captured)

    replayed = _draft_decode_batch([128, 30])
    # The replayed batch arrives through the same persistent input buffers.
    captured.seq_lens.copy_(replayed.seq_lens)
    captured.block_table_tensor.copy_(replayed.block_table_tensor)
    captured.slot_mapping.copy_(replayed.slot_mapping)
    builder.update_draft_decode_metadata(metadata)

    expected = _draft_builder(RocmAiterUnifiedAttentionMetadataBuilder).build(
        0, replayed
    )
    assert torch.equal(metadata.seq_lens, expected.seq_lens)
    assert torch.equal(metadata.block_table, expected.block_table)
    assert torch.equal(metadata.slot_mapping, expected.slot_mapping)
    assert torch.equal(metadata.query_start_loc, expected.query_start_loc)
