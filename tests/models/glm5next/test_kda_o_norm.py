# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On ROCm, GLM-5.3-Flash's KDA output norm runs the fused gated-RMSNorm kernel."""

from types import SimpleNamespace

import torch
from transformers import Glm5NextTextConfig

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


def test_o_norm_kernel_with_custom_ops_off(dist_init):
    config = Glm5NextTextConfig(
        hidden_size=256, linear_num_heads=2, linear_head_dim=64, num_hidden_layers=1
    )
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=["none"]
        )
    )
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    with set_current_vllm_config(vllm_config):
        layer = Glm5NextLinearAttention(
            config, vllm_config, prefix="model.layers.0.linear_attn"
        )

    # ROCm keeps the fused kernel; other platforms follow `custom_ops`.
    forward = layer.o_norm._forward_method.__func__
    is_native = forward is FusedRMSNormGated.forward_native
    assert is_native != current_platform.is_rocm()
    enabled = vllm_config.compilation_config.enabled_custom_ops
    assert ("fused_rms_norm_gated" in enabled) == current_platform.is_rocm()
