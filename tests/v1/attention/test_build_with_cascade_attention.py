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
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder


@pytest.mark.parametrize(
    "builder_cls", [TritonAttentionMetadataBuilder, RocmAttentionMetadataBuilder]
)
def test_builder_build_with_cascade_attention(builder_cls):
    """common_prefix_len > 0 triggers the rocm_attn and triton_attn UnboundError"""
    device = torch.device("cpu")
    vllm_config = create_vllm_config(model_name="facebook/opt-125m")
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    batch_spec = BatchSpec(seq_lens=[64, 64], query_lens=[32, 32])
    common_prefix_len = 16
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device
    )

    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)

    attn_metadata = builder.build(
        common_prefix_len=common_prefix_len,
        common_attn_metadata=common_attn_metadata,
    )

    assert attn_metadata.use_cascade is True
    assert attn_metadata.prefix_scheduler_metadata is None
    assert attn_metadata.prefix_kv_lens.item() == common_prefix_len
    assert torch.equal(
        attn_metadata.suffix_kv_lens.cpu(),
        common_attn_metadata.seq_lens.cpu() - common_prefix_len,
    )
    assert attn_metadata.cu_prefix_query_lens.tolist() == [
        0,
        common_attn_metadata.num_actual_tokens,
    ]