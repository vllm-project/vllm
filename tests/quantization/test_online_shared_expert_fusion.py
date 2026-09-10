# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests ROCm AITER fused shared experts with online quantization."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.quantization.utils import load_model_without_vllm_runner
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization.online.moe_shared_expert import (
    OnlineMxfp4SharedExpertLoader,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.platforms import current_platform

_QUARK_MXFP4_CONFIG: dict[str, Any] = {
    "global_quant_config": {
        "input_tensors": {
            "dtype": "fp4",
            "is_dynamic": True,
            "qscheme": "per_group",
            "ch_axis": -1,
            "group_size": 32,
            "block_size": None,
            "symmetric": None,
            "round_method": "half_even",
            "scale_type": "float",
            "scale_format": "e8m0",
            "scale_calculation_mode": "even",
            "mx_element_dtype": None,
            "observer_cls": "PerBlockMXObserver",
            "is_scale_quant": False,
            "enable_buffer_reuse": False,
            "max_input_numel": 4194304,
        },
        "output_tensors": None,
        "weight": {
            "dtype": "fp4",
            "is_dynamic": False,
            "qscheme": "per_group",
            "ch_axis": -1,
            "group_size": 32,
            "block_size": None,
            "symmetric": None,
            "round_method": "half_even",
            "scale_type": "float",
            "scale_format": "e8m0",
            "scale_calculation_mode": "even",
            "mx_element_dtype": None,
            "observer_cls": "PerBlockMXObserver",
            "is_scale_quant": False,
            "enable_buffer_reuse": False,
            "max_input_numel": 4194304,
        },
        "bias": None,
        "target_device": None,
    },
    "algo_config": None,
    "softmax_quant_spec": None,
    "quant_method": "quark",
    "layer_type_quant_config": {},
    "layer_quant_config": {},
    "exclude": ["re:^(?!.*\\.mlp\\.experts(?:\\.|$)).*$"],
    "kv_cache_quant_config": {},
    "kv_cache_post_rope": False,
    "quant_mode": "eager_mode",
    "version": "0.12+9d3d471cdf1",
    "export": {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
}

_QUARK_MXFP4_LAYER_CONFIG: dict[str, Any] = _QUARK_MXFP4_CONFIG["global_quant_config"]
_QUARK_MXFP4_CONFIG = {
    **_QUARK_MXFP4_CONFIG,
    "global_quant_config": {
        "input_tensors": {},
        "output_tensors": None,
        "weight": {},
        "bias": None,
        "target_device": None,
    },
    "layer_quant_config": {"*mlp.experts*": _QUARK_MXFP4_LAYER_CONFIG},
}


def _write_minimal_moe_config(model_path: Path, architecture: str) -> None:
    """Write a compact MoE config with one shared expert."""
    config = {
        "architectures": [architecture],
        "hidden_size": 256,
        "intermediate_size": 512,
        "moe_intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "max_position_embeddings": 64,
        "q_lora_rank": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 4,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "n_shared_experts": 1,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "rope_theta": 10000,
        "quantization_config": _QUARK_MXFP4_CONFIG,
    }
    if architecture == "AXK1ForCausalLM":
        config["model_type"] = "axk1"
    elif architecture == "Glm4MoeForCausalLM":
        config["model_type"] = "glm4_moe"
    elif architecture == "Qwen3NextForCausalLM":
        config.update(
            model_type="qwen3_next",
            layer_types=["linear_attention"],
            head_dim=64,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            shared_expert_intermediate_size=256,
        )
    else:
        config["model_type"] = "deepseek_v3" if "V3" in architecture else "deepseek_v2"
        if architecture == "GlmMoeDsaForCausalLM":
            config.update(
                index_topk=1,
                index_kpool=1,
                index_n_heads=1,
                index_head_dim=32,
                index_kv_lora_rank=32,
            )
    (model_path / "config.json").write_text(json.dumps(config))


def test_online_shared_expert_quantization_fusion_tp() -> None:
    """TP shared-expert weights are sharded on each projection's TP dimension."""
    loader = OnlineMxfp4SharedExpertLoader()
    w13 = torch.arange(48).reshape(8, 6)
    w2 = torch.arange(48).reshape(6, 8)

    assert torch.equal(loader._tp_shard(w13, "w1", tp_size=2, tp_rank=0), w13[:4])
    assert torch.equal(loader._tp_shard(w2, "w2", tp_size=2, tp_rank=0), w2[:, :4])
    assert torch.equal(loader._tp_shard(w13, "w3", tp_size=2, tp_rank=1), w13[4:])
    assert torch.equal(loader._tp_shard(w2, "w2", tp_size=2, tp_rank=1), w2[:, 4:])


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Fused shared-expert online quantization is a ROCm AITER feature.",
)
@pytest.mark.parametrize(
    "architecture",
    [
        "AXK1ForCausalLM",
        "DeepseekForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Qwen3NextForCausalLM",
    ],
)
def test_online_quantization(
    architecture: str,
    tmp_path: Path,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    """Online MXFP4 fuses each architecture's shared expert into its MoE."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    rocm_aiter_ops.refresh_env_variables()
    _write_minimal_moe_config(tmp_path, architecture)
    expected_shared_expert_name = (
        "shared_expert" if architecture == "Qwen3NextForCausalLM" else "shared_experts"
    )
    target_pattern = f"*{expected_shared_expert_name}*"

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

    model, vllm_config = load_model_without_vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        quantization="quark",
        model_config_kwargs={
            "quantization_config": resolve_quantization_config(
                "quark", {"targets": {target_pattern: "mxfp4"}}
            )
        },
        model_loader_cls=DummyModelLoader,
    )
    expected_model_cls, resolved_architecture = get_model_architecture(
        vllm_config.model_config
    )
    assert resolved_architecture == architecture
    assert isinstance(model, expected_model_cls)

    fused_moes = [
        module
        for module in model.modules()
        if getattr(module, "is_fused_shared_expert_enabled", False)
        and hasattr(module, "experts")
    ]
    assert len(fused_moes) == 1, f"{architecture} did not enable FSE"
    moe = fused_moes[0]
    shared_expert_name = (
        "shared_expert" if hasattr(moe, "shared_expert") else "shared_experts"
    )
    assert shared_expert_name == expected_shared_expert_name
    assert getattr(moe, shared_expert_name) is None

    routed_experts = moe.experts.routed_experts
    assert isinstance(routed_experts.quant_method, QuarkOCP_MX_MoEMethod)
    shared_expert_prefix = routed_experts.moe_config.shared_expert_prefix
    assert shared_expert_prefix is not None
    assert shared_expert_prefix.endswith(f"mlp.{shared_expert_name}")
    expert_map_manager = routed_experts.expert_map_manager
    assert expert_map_manager.num_fused_shared_experts == 1
    assert expert_map_manager.map_global_to_local(moe.n_routed_experts) == (
        moe.n_routed_experts
    )
    assert routed_experts.w13_weight.dtype == torch.float4_e2m1fn_x2
    assert routed_experts.w13_weight_scale.dtype == torch.uint8
    assert routed_experts.w2_weight.dtype == torch.float4_e2m1fn_x2
    assert routed_experts.w2_weight_scale.dtype == torch.uint8
    assert vllm_config.quant_config.online_quantization_config is not None
    assert not any(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
        "cannot be enabled" in warning
        for warning in logged_warnings
    )
    assert logged_messages == [
        "Quantized 2 layers of types: "
        f"mlp.{shared_expert_name}.down_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4); "
        f"mlp.{shared_expert_name}.gate_up_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4)"
    ]
