# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")
MODEL_NAME = "facebook/opt-125m"


def _create_builder() -> TritonAttentionMetadataBuilder:
    vllm_config = create_vllm_config(model_name=MODEL_NAME, block_size=BLOCK_SIZE)
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    return TritonAttentionMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def test_triton_builder_initializes_prefix_scheduler_metadata_for_cascade():
    builder = _create_builder()
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[16], query_lens=[1]),
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    metadata = builder.build(common_prefix_len=1, common_attn_metadata=common)

    assert metadata.use_cascade is True
    assert metadata.common_prefix_len == 1
    assert metadata.prefix_scheduler_metadata is None
    assert metadata.prefix_kv_lens.tolist() == [1]
