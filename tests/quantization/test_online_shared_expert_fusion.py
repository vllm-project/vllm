# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests ROCm AITER fused shared experts with online quantization."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.quantization.utils import load_model_without_vllm_runner
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import ModelConfig
from vllm.config.quantization import (
    QuantizationConfigArgs,
    resolve_quantization_config,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm AITER shared-expert tests require ROCm.",
)

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


class _StubAttention(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)


@pytest.fixture(autouse=True)
def reset_aiter_shared_expert_topk_metadata(monkeypatch: pytest.MonkeyPatch):
    """Isolate AITER's process-global shared-expert routing buffer."""
    monkeypatch.setattr(rocm_aiter_moe, "aiter_topK_meta_data", None)
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    yield
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()


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
    elif architecture == "MiniMaxM3SparseForCausalLM":
        text_config = {
            **config,
            "model_type": "minimax_m3_text",
            "dense_intermediate_size": 512,
            "hidden_act": "swigluoai",
            "num_local_experts": 4,
            "n_shared_experts": 1,
            "moe_layer_freq": [1],
            "sparse_attention_config": None,
            "quantization_config": {
                **_QUARK_MXFP4_CONFIG,
                "layer_quant_config": {
                    "*block_sparse_moe.experts*": _QUARK_MXFP4_LAYER_CONFIG
                },
                "exclude": [r"re:^(?!.*\.block_sparse_moe\.experts(?:\.|$)).*$"],
            },
        }
        config = {
            "architectures": [architecture],
            "model_type": "minimax_m3_vl",
            "text_config": text_config,
        }
    elif architecture == "Glm4MoeForCausalLM":
        config["model_type"] = "glm4_moe"
    elif architecture == "Glm4MoeLiteForCausalLM":
        config["model_type"] = "glm4_moe_lite"
    elif architecture in ("Qwen3NextForCausalLM", "Qwen3_5MoeForCausalLM"):
        config.update(
            model_type=(
                "qwen3_next"
                if architecture == "Qwen3NextForCausalLM"
                else "qwen3_5_moe_text"
            ),
            layer_types=[
                "linear_attention"
                if architecture == "Qwen3NextForCausalLM"
                else "full_attention"
            ],
            head_dim=64,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            shared_expert_intermediate_size=256,
        )
    elif architecture == "Qwen4ExpForCausalLM":
        config.update(
            model_type="qwen4_exp_text",
            layer_types=["full_attention"],
            head_dim=64,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            shared_expert_intermediate_size=256,
            eos_token_id=1,
            hc_count=2,
            hc_lowrank=4,
            ple_layer_ids=[],
        )
    elif architecture == "Glm5NextForCausalLM":
        config.update(
            model_type="glm5_next",
            layer_types=["linear_attention"],
            mlp_layer_types=["sparse"],
            qk_rope_head_dim=0,
            index_topk=1,
            index_kpool=1,
            pad_token_id=None,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            mhc=False,
        )
    else:
        config["model_type"] = (
            "deepseek_v3"
            if "V3" in architecture
            else "deepseek_v32"
            if architecture == "DeepseekV32ForCausalLM"
            else "deepseek_v2"
        )
        if architecture in ("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"):
            config.update(
                index_topk=1,
                index_kpool=1,
                index_n_heads=1,
                index_head_dim=32,
                index_kv_lora_rank=32,
            )
    (model_path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize(
    "load_path", ["load_weights", "fused_gate_up", "weight_loader"]
)
def test_online_shared_expert_loads_bf16_weights_into_mxfp4_slot(
    default_vllm_config,
    dist_init,
    load_path: str,
) -> None:
    """A BF16 shared expert is quantized while routed MXFP4 weights are loaded.

    `weight_loader` covers DeepSeek-style model loaders, which bypass
    `RoutedExperts.load_weights`.
    """
    default_vllm_config.model_config = ModelConfig()
    hidden_size = intermediate_size = 64
    num_routed_experts = 2
    device = current_platform.device_type
    online_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*shared_expert*": "mxfp4"})
    )
    quant_config = QuarkConfig(_QUARK_MXFP4_CONFIG)
    quant_config.online_quantization_config = online_config

    with torch.device(device):
        runner = FusedMoEFactory(
            num_experts=num_routed_experts,
            top_k=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_shared_experts=1,
            fuse_shared_experts=True,
            shared_expert_prefix="model.layers.0.mlp.shared_expert",
            prefix="model.layers.0.mlp.experts",
            quant_config=quant_config,
        )
        layer = runner.routed_experts
        assert isinstance(layer.quant_method, QuarkOCP_MX_MoEMethod)

        routed_gate, _ = mxfp4_quantize(
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        )
        routed_up, _ = mxfp4_quantize(
            torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        )
        routed_down, _ = mxfp4_quantize(
            torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)
        )
        shared_gate = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        shared_up = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
        shared_down = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)

        weights = [
            ("0.gate_proj.weight", routed_gate),
            ("0.up_proj.weight", routed_up),
            ("0.down_proj.weight", routed_down),
        ]
        if load_path == "load_weights":
            weights += [
                ("2.gate_proj.weight", shared_gate),
                ("2.up_proj.weight", shared_up),
                ("2.down_proj.weight", shared_down),
            ]
        elif load_path == "fused_gate_up":
            weights += [
                ("2.gate_up_proj.weight", torch.cat([shared_gate, shared_up])),
                ("2.down_proj.weight", shared_down),
            ]
        list(layer.load_weights(weights))

        if load_path == "weight_loader":
            for stem, shard_id, weight in (
                ("w13", "w1", shared_gate),
                ("w13", "w3", shared_up),
                ("w2", "w2", shared_down),
            ):
                param = getattr(layer, f"{stem}_weight")
                param.weight_loader(
                    param, weight, f"{layer.layer_name}.{stem}_weight", shard_id, 2
                )

    expected_gate, expected_gate_scale = mxfp4_quantize(shared_gate)
    expected_up, expected_up_scale = mxfp4_quantize(shared_up)
    expected_down, expected_down_scale = mxfp4_quantize(shared_down)
    assert torch.equal(layer.w13_weight[2, :intermediate_size], expected_gate)
    assert torch.equal(layer.w13_weight[2, intermediate_size:], expected_up)
    assert torch.equal(layer.w2_weight[2], expected_down)
    assert torch.equal(
        layer.w13_weight_scale[2, :intermediate_size], expected_gate_scale
    )
    assert torch.equal(layer.w13_weight_scale[2, intermediate_size:], expected_up_scale)
    assert torch.equal(layer.w2_weight_scale[2], expected_down_scale)


def test_online_shared_expert_reload_compatibility(
    default_vllm_config,
    dist_init,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reload a BF16 shared weight into its MXFP4 slot via meta staging."""
    default_vllm_config.model_config = ModelConfig()
    online_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*shared_expert*": "mxfp4"})
    )
    quant_config = QuarkConfig(_QUARK_MXFP4_CONFIG)
    quant_config.online_quantization_config = online_config

    with torch.device(current_platform.device_type):
        layer = FusedMoEFactory(
            num_experts=2,
            top_k=1,
            hidden_size=64,
            intermediate_size=64,
            n_shared_experts=1,
            fuse_shared_experts=True,
            shared_expert_prefix="model.layers.0.mlp.shared_expert",
            prefix="model.layers.0.mlp.experts",
            quant_config=quant_config,
        ).routed_experts
        monkeypatch.setattr(
            layer.quant_method, "process_weights_after_loading", lambda _: None
        )

        routed_weights = []
        for expert_id in range(2):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                weight, scale = mxfp4_quantize(
                    torch.randn(64, 64, dtype=torch.bfloat16)
                )
                prefix = f"{expert_id}.{projection}"
                routed_weights.extend(
                    [(f"{prefix}.weight", weight), (f"{prefix}.weight_scale", scale)]
                )

        shared_gate = torch.randn(64, 64, dtype=torch.bfloat16)
        shared_up = torch.randn(64, 64, dtype=torch.bfloat16)
        shared_down = torch.randn(64, 64, dtype=torch.bfloat16)

        def checkpoint_weights(gate: torch.Tensor):
            return [
                *routed_weights,
                ("2.gate_proj.weight", gate),
                ("2.up_proj.weight", shared_up),
                ("2.down_proj.weight", shared_down),
            ]

        list(layer.load_weights(checkpoint_weights(shared_gate)))
        old_weight = layer.w13_weight[2, :64].clone()
        reloaded_gate = shared_gate + 1
        expected_weight, expected_scale = mxfp4_quantize(reloaded_gate)
        assert not torch.equal(old_weight, expected_weight)

        record_metadata_for_reloading(layer)
        initialize_layerwise_reload(layer)

        assert layer.w13_weight.is_meta
        list(layer.load_weights(checkpoint_weights(reloaded_gate)))
        finalize_layerwise_reload(layer, default_vllm_config.model_config)
        assert not layer.w13_weight.is_meta
        assert torch.equal(layer.w13_weight[2, :64], expected_weight)
        assert torch.equal(layer.w13_weight_scale[2, :64], expected_scale)


# TODO: Add DeepseekV4ForConditionalGeneration once it uses
# resolve_layer_fused_shared_expert.
@pytest.mark.parametrize(
    "architecture",
    [
        "AXK1ForCausalLM",
        "DeepseekForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "Glm5NextForCausalLM",
        "GlmMoeDsaForCausalLM",
        "MiniMaxM3SparseForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen4ExpForCausalLM",
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
    if architecture == "Glm5NextForCausalLM":
        from vllm.models.glm5next.common import model as glm5_next_model

        for attention in ("Glm5NextLinearAttention", "Glm5NextMLAAttention"):
            monkeypatch.setattr(glm5_next_model, attention, _StubAttention)
        monkeypatch.setattr(
            glm5_next_model, "_fused_shared_experts_tuned", lambda _: True
        )
    elif architecture == "Glm4MoeLiteForCausalLM":
        from vllm.model_executor.models import glm4_moe_lite

        monkeypatch.setattr(glm4_moe_lite, "Glm4MoeLiteAttention", _StubAttention)
        monkeypatch.setattr(glm4_moe_lite, "Glm4MoeLiteMLAAttention", _StubAttention)
    elif architecture == "DeepseekV32ForCausalLM":
        from vllm.models.deepseek_v32.amd import model as deepseek_v32_model

        monkeypatch.setattr(
            deepseek_v32_model, "DeepseekV32MLAAttention", _StubAttention
        )
    elif architecture == "Qwen3_5MoeForCausalLM":
        from vllm.model_executor.models import qwen3_5

        monkeypatch.setattr(qwen3_5, "Qwen3NextAttention", _StubAttention)
    elif architecture == "Qwen4ExpForCausalLM":
        from vllm.models.qwen4_exp.amd import model as qwen4_exp_model

        monkeypatch.setattr(qwen4_exp_model, "Qwen3NextAttention", _StubAttention)
    _write_minimal_moe_config(tmp_path, architecture)
    expected_shared_expert_name = (
        "shared_expert" if architecture.startswith("Qwen") else "shared_experts"
    )
    target_pattern = f"*{expected_shared_expert_name}*"

    logged_messages: list[str] = []
    logged_warnings: list[str] = []

    def record_info(message: str, *args: object, **_kwargs: object) -> None:
        if "Quantizing " in message and "of types" in message:
            logged_messages.append(message % args)

    def record_warning(message: str, *args: object, **_kwargs: object) -> None:
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
    assert shared_expert_prefix.endswith(shared_expert_name)
    assert routed_experts._fused_shared_expert_quantizer is not None
    expert_map_manager = routed_experts.expert_map_manager
    assert expert_map_manager.num_fused_shared_experts == 1
    assert expert_map_manager.map_global_to_local(
        routed_experts.moe_config.num_experts
    ) == (routed_experts.moe_config.num_experts)
    expected_weight_dtype = {
        Mxfp4MoeBackend.AITER_MXFP4_BF16: torch.float4_e2m1fn_x2,
        Mxfp4MoeBackend.AITER_MXFP4_MXFP4: torch.float4_e2m1fn_x2,
        Mxfp4MoeBackend.EMULATION: torch.uint8,
    }[routed_experts.quant_method.mxfp4_backend]
    assert routed_experts.w13_weight.dtype == expected_weight_dtype
    assert routed_experts.w13_weight_scale.dtype == torch.uint8
    assert routed_experts.w2_weight.dtype == expected_weight_dtype
    assert routed_experts.w2_weight_scale.dtype == torch.uint8
    assert vllm_config.quant_config is not None
    assert vllm_config.quant_config.online_quantization_config is not None
    assert not any(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
        "cannot be enabled" in warning
        for warning in logged_warnings
    )
    if architecture == "MiniMaxM3SparseForCausalLM":
        # TODO: Use resolve_layer_fused_shared_expert in MiniMax M3 and remove
        # this branch once its virtual fused projections are registered.
        assert logged_messages == ["Quantizing 0 layers of types: "]
    else:
        assert logged_messages == [
            "Quantizing 2 layers of types: "
            f"mlp.{shared_expert_name}.down_proj: 1 "
            f"(from targets: {target_pattern}, mxfp4); "
            f"mlp.{shared_expert_name}.gate_up_proj: 1 "
            f"(from targets: {target_pattern}, mxfp4)"
        ]


def test_online_gate_up_proj_target_disables_fse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dist_init,
    workspace_init,
) -> None:
    """A partial shared-expert target quantizes while disabling FSE."""
    architecture = "DeepseekForCausalLM"
    target_pattern = "*shared_experts.gate_up_proj*"
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "1")
    rocm_aiter_ops.refresh_env_variables()
    _write_minimal_moe_config(tmp_path, architecture)

    logged_messages: list[str] = []
    logged_warnings: list[str] = []

    def record_info(message: str, *args: object, **_kwargs: object) -> None:
        if "Quantizing " in message and "of types" in message:
            logged_messages.append(message % args)

    def record_warning(message: str, *args: object, **_kwargs: object) -> None:
        logged_warnings.append(message % args)

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.base_loader.logger.info", record_info
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.utils.logger.warning", record_warning
    )

    model, _ = load_model_without_vllm_runner(
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

    moe = next(
        module
        for module in model.modules()
        if hasattr(module, "is_fused_shared_expert_enabled")
        and hasattr(module, "shared_experts")
    )
    assert not moe.is_fused_shared_expert_enabled
    assert moe.shared_experts is not None
    assert logged_messages == [
        "Quantizing 1 layers of types: "
        "mlp.shared_experts.gate_up_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4)"
    ]
    assert logged_warnings == [
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but cannot be "
        "enabled - skipping for this layer: online quantization targets only "
        "part of the shared expert at model.layers.0.mlp.shared_experts."
    ]
