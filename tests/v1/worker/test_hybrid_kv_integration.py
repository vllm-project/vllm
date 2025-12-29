#! SPDX-License-Identifier: Apache-2.0
#! SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.hybrid_attn_layer import HybridAttentionLayer
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec
from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec


def test_hybrid_kv_cache_spec_includes_ssm_group():
    """HybridAttentionLayer should register both sliding-window and SSM KV specs."""
    vllm_config = create_vllm_config(add_mock_model_methods=True)

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config

    # Enable a model-level sliding window so the attention KV spec is
    # SlidingWindowSpec instead of FullAttentionSpec.
    cache_config.sliding_window = 128

    num_heads = model_config.get_num_attention_heads(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()

    with set_current_vllm_config(vllm_config):
        _ = HybridAttentionLayer(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0,
            num_kv_heads=num_kv_heads,
            ssm_state_size=head_size,
            ssm_conv_kernel_size=3,
            ssm_intermediate_size=4 * head_size,
            cache_config=cache_config,
            prefix="layer0.attn",
        )

    kv_specs = get_kv_cache_spec(vllm_config)

    assert "layer0.attn" in kv_specs
    assert "layer0.attn.ssm" in kv_specs

    attn_spec = kv_specs["layer0.attn"]
    ssm_spec = kv_specs["layer0.attn.ssm"]

    assert isinstance(attn_spec, SlidingWindowSpec)
    assert isinstance(ssm_spec, MambaSpec)


