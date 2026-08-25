# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.orthrus import OrthrusConfig

pytestmark = pytest.mark.skip_global_cleanup


def test_orthrus_config_registered_without_remote_code(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["OrthrusLM"],
                "model_type": "orthrus",
                "vocab_size": 151936,
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "max_position_embeddings": 40960,
                "block_size": 16,
                "mask_token_id": 151665,
            }
        )
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, OrthrusConfig)
    assert config.model_type == "orthrus"
    assert config.block_size == 16
    assert config.mask_token_id == 151665


def test_orthrus_diff_projections_map_to_their_own_fused_param():
    """Pins the checkpoint-name -> fused-parameter contract.

    Orthrus names its diffusion projections ``q_proj_diff`` etc., which
    contain ``q_proj`` as a substring, so the autoregressive and diffusion
    projections must stay separable. This pins both spellings to the fused
    parameter and shard they belong to, guarding future edits to
    ``STACKED_PARAMS_MAPPING`` or its matching rule.
    """
    from vllm.model_executor.models.orthrus import resolve_stacked_weight_name

    prefix = "model.layers.0.self_attn"
    for proj, shard in (("q", "q"), ("k", "k"), ("v", "v")):
        ar = resolve_stacked_weight_name(f"{prefix}.{proj}_proj.weight")
        assert ar == (f"{prefix}.qkv_proj.weight", shard)

        diff = resolve_stacked_weight_name(f"{prefix}.{proj}_proj_diff.weight")
        assert diff == (f"{prefix}.qkv_proj_diff.weight", shard)

    # o_proj_diff is not a fused parameter and must load directly.
    assert resolve_stacked_weight_name(f"{prefix}.o_proj_diff.weight") is None
