# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_sparse_swa_reorder_threshold_matches_spec_decode_threshold():
    vllm_config = create_vllm_config(
        block_size=256,
        hf_config_override={
            "sliding_window": 128,
            "compress_ratios": [1, 4, 128],
        },
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=2,
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )

    builder = DeepseekSparseSWAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )

    assert builder.decode_threshold == 3
    assert builder.reorder_batch_threshold == builder.decode_threshold
