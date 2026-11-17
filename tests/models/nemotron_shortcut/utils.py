# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A small Nemotron Shortcut config and module builder for the tests."""

import torch

from vllm.platforms import current_platform
from vllm.transformers_utils.configs.nemotron_shortcut import NemotronShortcutTextConfig


def make_config(**overrides) -> NemotronShortcutTextConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        mlp_hidden_act="relu2",
        activation_clamp_value=4.0,
        layer_types=["linear_attention", "mlp", "shortcut_moe_gdp"],
        gdp_num_heads=4,
        gdp_num_groups=4,
        gdp_head_dim=8,
        gdp_key_head_dim=16,
        gdp_num_householder=2,
        gdp_conv_kernel=4,
        gdp_norm_eps=1e-5,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_latent_size=8,
        moe_shared_expert_intermediate_size=16,
        moe_use_norm_before_up_proj=True,
        moe_shortcut_post_norm=False,
        routed_scaling_factor=2.5,
        wide_residual_num_streams=2,
        wide_residual_learned_retention=True,
        wide_residual_retention_max_forget=0.2,
        residual_in_fp32=False,
    )
    kwargs.update(overrides)
    return NemotronShortcutTextConfig(**kwargs)


def build(factory):
    """Build a module on the platform device with small finite parameters."""
    torch.manual_seed(0)
    with torch.device(current_platform.device_type):
        module = factory()
    for param in module.parameters():
        param.data.uniform_(-0.1, 0.1)
    return module
