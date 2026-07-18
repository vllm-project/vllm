# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def test_sparse_swa_opts_out_of_reorder_batch_vote():
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
    # sparse_swa deliberately opts OUT of the runner's reorder-batch vote
    # (upstream #47327): the runner reduces thresholds with min_none_high, so
    # publishing a real threshold here would drag flashmla_sparse's 128-1024
    # dense-MHA routing threshold down for the whole model. indexer.py uses the
    # same None opt-out. decode_threshold above stays the local spec value.
    assert builder.reorder_batch_threshold is None
