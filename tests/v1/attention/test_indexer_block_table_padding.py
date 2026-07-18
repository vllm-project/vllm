# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_expanded_block_table_matches_multigroup_block_table_padding():
    """Regression: expanded_block_table_buffer width must match block_table_tensor.

    MultiGroupBlockTable pads max_num_blocks_per_req to a multiple of
    (128 // block_size). Without the same padding in the MLA indexer buffer,
    MTP flatten decode can fail with expand size mismatch (e.g. 1669 vs 1670).
    """
    device = torch.device("cuda")
    max_model_len = 106816
    block_size = 64
    # ceil(106816 / 64) = 1669 -> padded to 1670 for block_size=64
    expected_blocks_per_req = 1670

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    vllm_config = create_vllm_config(
        max_model_len=max_model_len,
        block_size=block_size,
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    assert builder.expanded_block_table_buffer.shape[1] == expected_blocks_per_req
