# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)


@pytest.mark.parametrize(
    ("config_block_size", "compress_ratio", "expected"),
    [
        (32, 1, 64),
        (64, 1, 64),
        (128, 1, 128),
        (64, 2, 128),
        (128, 2, 128),
    ],
)
def test_sm120_compressed_cache_page_widens_to_64_states(
    config_block_size: int, compress_ratio: int, expected: int
):
    attention = DeepseekV4FlashInferSM120Attention.__new__(
        DeepseekV4FlashInferSM120Attention
    )
    attention.is_kv_source = True
    attention.kv_cache_dtype = "fp8_ds_mla"
    attention.kv_cache_torch_dtype = torch.uint8
    attention.head_dim = 512
    attention.kv_bytes_per_token = 584
    attention.kv_page_alignment = 576
    attention.compress_ratio = compress_ratio
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=config_block_size)
    )

    spec = attention.get_kv_cache_spec(vllm_config)
    selected_block_size = min(spec.block_size, 64 * compress_ratio)

    assert spec.block_size == expected
    assert spec.get_num_kernel_states(selected_block_size) == 64
