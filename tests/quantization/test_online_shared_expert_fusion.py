# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests ROCm AITER fused shared experts with online quantization."""

import pytest
import torch

from tests.quantization.utils import load_model_without_vllm_runner
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import current_platform

MODEL_NAME = "amd/Qwen3.5-35B-A3B-MXFP4"


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Fused shared-expert online quantization is a ROCm AITER feature.",
)
def test_online_quantization(monkeypatch, dist_init, workspace_init) -> None:
    """Quantize and fuse Qwen3.5 shared-expert weights into the MoE."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    rocm_aiter_ops.refresh_env_variables()

    logged_messages: list[str] = []
    logged_warnings: list[str] = []

    def record_info(message: str, *args: object) -> None:
        logged_messages.append(message % args)

    def record_warning(message: str, *args: object) -> None:
        logged_warnings.append(message % args)

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info", record_info
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.utils.logger.warning", record_warning
    )

    online_quant_args = {"targets": {"*shared_expert*": "mxfp4"}}
    model, vllm_config = load_model_without_vllm_runner(
        MODEL_NAME,
        model_config_kwargs={
            "quantization_config": online_quant_args,
            "hf_overrides": {
                "text_config": {
                    "num_hidden_layers": 1,
                    "layer_types": ["linear_attention"],
                }
            },
        },
    )

    mlp = model.language_model.model.layers[0].mlp
    experts = mlp.experts
    routed_experts = experts.routed_experts
    online_config = vllm_config.quant_config.online_quantization_config

    assert vllm_config.model_config.hf_text_config.num_hidden_layers == 1
    assert mlp.is_fused_shared_expert_enabled
    assert mlp.shared_expert is None
    assert experts.expert_map_manager.num_fused_shared_experts == 1
    assert routed_experts.w13_weight.dtype == torch.float4_e2m1fn_x2
    assert routed_experts.w13_weight_scale.dtype == torch.uint8
    assert routed_experts.w2_weight.dtype == torch.float4_e2m1fn_x2
    assert routed_experts.w2_weight_scale.dtype == torch.uint8
    assert online_config is not None
    assert not any(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
        "cannot be enabled" in warning
        for warning in logged_warnings
    )
    assert logged_messages == [
        "Quantized 2 layers of types: mlp.shared_expert.down_proj: 1 "
        "(from targets: *shared_expert*, mxfp4); "
        "mlp.shared_expert.gate_up_proj: 1 "
        "(from targets: *shared_expert*, mxfp4)"
    ]
