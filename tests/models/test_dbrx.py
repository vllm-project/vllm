# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DBRX model weight loading.

Regression test for https://github.com/vllm-project/vllm/issues/49004
"""

import torch
from transformers import DbrxConfig
from transformers.models.dbrx.configuration_dbrx import (
    DbrxAttentionConfig,
    DbrxFFNConfig,
)

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models.dbrx import DbrxModel


def test_dbrx_load_weights_maps_checkpoint_names_to_routed_experts(dist_init):
    """Test that DbrxModel.load_weights correctly maps checkpoint names.

    This is a regression test for https://github.com/vllm-project/vllm/issues/49004
    where DbrxModel.load_weights was not updated for the FusedMoE/MoERunner
    refactor in PR #41184. Expert weights now live under a routed_experts
    submodule.

    Without the fix, load_weights raises:
        KeyError: 'blocks.0.ffn.experts.w13_weight'
    because it maps mlp.w1 -> w13 instead of -> routed_experts.w13

    With the fix, checkpoint names like 'blocks.0.ffn.experts.mlp.w1' are
    correctly mapped to 'blocks.0.ffn.experts.routed_experts.w13_weight'.
    """
    # Create minimal DBRX config to avoid OOM
    attn_config = DbrxAttentionConfig(kv_n_heads=1)
    attn_config.rope_theta = 10000.0

    ffn_config = DbrxFFNConfig(
        ffn_hidden_size=64,
        moe_num_experts=4,
        moe_top_k=1,
    )
    dbrx_config = DbrxConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        vocab_size=128,
        max_seq_len=256,
        attn_config=attn_config,
        ffn_config=ffn_config,
    )

    # Set up VllmConfig - internal components use get_current_vllm_config()
    model_config = ModelConfig(model="facebook/opt-125m")
    model_config.hf_config = dbrx_config
    model_config.hf_text_config = dbrx_config
    model_config.dtype = torch.float32
    vllm_config = VllmConfig(model_config=model_config)

    # dist_init already initializes distributed/model parallel
    with set_current_vllm_config(vllm_config):
        # Construct the real DbrxModel
        model = DbrxModel(vllm_config=vllm_config, prefix="")
        params = dict(model.named_parameters(remove_duplicate=False))

        # Verify the model registers params under routed_experts
        expert_params = [k for k in params if "experts" in k]
        assert any("routed_experts.w13_weight" in p for p in expert_params), (
            "Model must have routed_experts.w13_weight param"
        )
        assert any("routed_experts.w2_weight" in p for p in expert_params), (
            "Model must have routed_experts.w2_weight param"
        )

        # Verify the OLD (buggy) un-prefixed names do NOT exist
        assert not any(
            p.endswith("experts.w13_weight") and "routed_experts" not in p
            for p in params
        ), "Un-prefixed experts.w13_weight should not exist"
        assert not any(
            p.endswith("experts.w2_weight") and "routed_experts" not in p
            for p in params
        ), "Un-prefixed experts.w2_weight should not exist"

        # Derive checkpoint tensor shape from config
        # DbrxExperts.weight_loader reshapes to [-1, ffn_hidden_size, d_model]
        num_experts = dbrx_config.ffn_config.moe_num_experts
        ffn_hidden_size = dbrx_config.ffn_config.ffn_hidden_size
        d_model = dbrx_config.d_model
        ckpt_shape = (num_experts, ffn_hidden_size, d_model)

        # Create checkpoint-format weights (HuggingFace DBRX format uses mlp.w1/v1/w2)
        checkpoint_weights = [
            ("blocks.0.ffn.experts.mlp.w1", torch.randn(ckpt_shape)),
            ("blocks.0.ffn.experts.mlp.v1", torch.randn(ckpt_shape)),
            ("blocks.0.ffn.experts.mlp.w2", torch.randn(ckpt_shape)),
        ]

        # Call the real load_weights - this raises KeyError without the fix
        loaded_params = model.load_weights(checkpoint_weights)

        # Verify the loaded params have routed_experts prefix
        assert any("routed_experts.w13_weight" in p for p in loaded_params), (
            "Loaded params must include routed_experts.w13_weight"
        )
        assert any("routed_experts.w2_weight" in p for p in loaded_params), (
            "Loaded params must include routed_experts.w2_weight"
        )
