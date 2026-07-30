# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec


class _NonCausalMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    supports_non_causal_multi_token_decode = True


def _metadata(
    query_start_loc: list[int], num_tokens: int | None = None
) -> CommonAttentionMetadata:
    num_reqs = len(query_start_loc) - 1
    num_tokens = query_start_loc[-1] if num_tokens is None else num_tokens
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.arange(1, num_reqs + 1, dtype=torch.int32) * 100 + 8,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(
            end - start for start, end in zip(query_start_loc, query_start_loc[1:])
        ),
        max_seq_len=num_reqs * 100 + 8,
        block_table_tensor=torch.arange(num_reqs * 3, dtype=torch.int32).view(
            num_reqs, 3
        ),
        slot_mapping=torch.arange(num_tokens),
        causal=False,
        seq_lens_cpu_upper_bound=None,
    )


def _builder(marked: bool = True) -> _NonCausalMLAMetadataBuilder:
    builder = object.__new__(_NonCausalMLAMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.reorder_batch_threshold = 1
    builder.query_len_support = QueryLenSupport.SINGLE_ONLY
    builder.non_causal_multi_token_decode = marked
    builder.dcp_world_size = 1
    builder.metadata_cls = MLACommonMetadata
    builder.model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_head_size=lambda: 576
    )
    return builder


def test_noncausal_block_uses_decode_without_cpu_lengths():
    common_metadata = _metadata([0, 8, 16])
    metadata = _builder().build(0, common_metadata)

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 16
    assert metadata.num_prefills == 0
    assert metadata.prefill is None
    assert metadata.decode is not None
    assert metadata.decode.block_table.shape == (2, 3)
    assert metadata.decode.seq_lens.shape == (2,)
    assert torch.equal(metadata.decode.block_table, common_metadata.block_table_tensor)
    assert torch.equal(metadata.decode.seq_lens, common_metadata.seq_lens)
    assert not metadata.causal


def test_noncausal_support_is_explicit_and_uniform():
    with pytest.raises(ValueError, match="explicitly supported"):
        _builder(marked=False).build(0, _metadata([0, 8, 16]))
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 3, 8]))


def test_noncausal_block_allows_trailing_cudagraph_padding():
    common_metadata = _metadata([0, 8, 16, 16], num_tokens=24)
    common_metadata.seq_lens[-1] = 0

    metadata = _builder().build(0, common_metadata)

    assert metadata.num_decodes == 3
    assert metadata.num_decode_tokens == 24
    assert metadata.decode is not None
    assert metadata.decode.seq_lens.tolist() == [108, 208, 0]


def test_noncausal_block_rejects_non_trailing_padding():
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 8, 8, 16], num_tokens=24))


def test_noncausal_decode_metadata_keeps_live_request_buffers():
    common_metadata = _metadata([0, 8, 16])
    metadata = _builder().build(0, common_metadata)

    assert metadata.decode is not None
    assert metadata.decode.seq_lens.data_ptr() == common_metadata.seq_lens.data_ptr()
    assert (
        metadata.decode.block_table.data_ptr()
        == common_metadata.block_table_tensor.data_ptr()
    )


def test_mla_cache_marker_is_promoted_to_group_capability():
    kwargs = {
        "block_size": 64,
        "num_kv_heads": 1,
        "head_size": 576,
        "dtype": torch.bfloat16,
    }
    marked = MLAAttentionSpec(**kwargs, non_causal_multi_token_decode=True)
    unmarked = MLAAttentionSpec(**kwargs)

    assert MLAAttentionSpec.merge([marked, marked]).non_causal_multi_token_decode
    assert MLAAttentionSpec.merge([marked, unmarked]).non_causal_multi_token_decode
    assert not MLAAttentionSpec.merge(
        [unmarked, unmarked]
    ).non_causal_multi_token_decode
