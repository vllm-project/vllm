# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3 tests' model geometry, read from the shipped checkpoint's config
values.
"""

from types import SimpleNamespace

import pytest

from vllm.models.glm53.nvidia.ops.launch import Glm53Geometry

#: The ``config.json`` fields of zai-org/GLM-5.3 that the decode segment reads.
GLM53_CONFIG = {
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "index_topk_freq": 4,
    "index_skip_topk_offset": 3,
    "index_topk_pattern": None,
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_topk": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "vocab_size": 154880,
    "quantization_config": {"weight_block_size": [128, 128]},
}


@pytest.fixture
def hf_config() -> SimpleNamespace:
    """The shipped checkpoint's config fields."""
    return SimpleNamespace(**GLM53_CONFIG)


@pytest.fixture
def geometry(hf_config: SimpleNamespace) -> Glm53Geometry:
    """GLM-5.3's geometry on TP8."""
    return Glm53Geometry.from_config(hf_config, world=8)
