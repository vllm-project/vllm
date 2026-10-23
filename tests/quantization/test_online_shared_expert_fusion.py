# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests ROCm AITER fused shared experts with online quantization."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

import vllm.envs as envs
from tests.quantization.utils import load_model_without_vllm_runner
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import ModelConfig
from vllm.config.quantization import (
    QuantizationConfigArgs,
    resolve_quantization_config,
)
from vllm.logger import _print_warning_once
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.mxfp4 import Mxfp4OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.requantization import (
    is_requantization_source,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkNvfp4MoEMethod,
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    mxfp4_quantize,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
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


class _SingleRankGroup:
    """Minimal stand-in for a size-1 `GroupCoordinator`."""

    world_size = 1
    rank_in_group = 0
    rank = 0
    local_rank = 0
    ranks = [0]
    is_first_rank = True
    is_last_rank = True
    # Read at model construction, never used for communication here.
    device_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    cpu_group = None


@pytest.fixture
def single_rank_parallel_state(monkeypatch: pytest.MonkeyPatch):
    """Single rank parallel groups without initializing a process group.

    This is faster than the `dist_init` fixture.
    """
    for name in ("_WORLD", "_TP", "_ETP", "_DCP", "_PP", "_DP", "_EP", "_PCP"):
        monkeypatch.setattr(
            f"vllm.distributed.parallel_state.{name}", _SingleRankGroup()
        )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


@pytest.fixture(autouse=True)
def reset_aiter_shared_expert_topk_metadata(monkeypatch: pytest.MonkeyPatch):
    """Isolate AITER's process-global shared-expert routing buffer."""
    _print_warning_once.cache_clear()  # `logger.warning_once` dedupes across tests
    monkeypatch.setattr(rocm_aiter_moe, "aiter_topK_meta_data", None)
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    yield
    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()


def _write_minimal_moe_config(
    model_path: Path,
    architecture: str,
    quantization_config: dict[str, Any] | None = None,
) -> None:
    """Write a compact MoE config with one shared expert.

    `quantization_config` overrides the checkpoint quantization of MiniMax-M3.
    """
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
            "intermediate_size": 256,
            "dense_intermediate_size": 512,
            "shared_intermediate_size": 256,
            "hidden_act": "swigluoai",
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "scoring_func": "sigmoid",
            "use_routing_bias": True,
            "routed_scaling_factor": 2.0,
            "num_local_experts": 4,
            "moe_layer_freq": [1],
            "sparse_attention_config": None,
            "quantization_config": quantization_config
            or {
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
    single_rank_parallel_state,
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
    single_rank_parallel_state,
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
    caplog_vllm,
    single_rank_parallel_state,
    workspace_init,
) -> None:
    """Online MXFP4 fuses each architecture's shared expert into its MoE."""
    _enable_aiter(monkeypatch, fse=True)
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
    assert routed_experts.shared_expert_online_loader is not None
    assert routed_experts.shared_expert_online_loader.quantizer is not None
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
        for warning in _fse_warnings(caplog_vllm)
    )
    parent_name = (
        "block_sparse_moe" if architecture == "MiniMaxM3SparseForCausalLM" else "mlp"
    )
    assert _online_quantization_logs(caplog_vllm) == [
        "Quantizing 2 layers of types: "
        f"{parent_name}.{shared_expert_name}.down_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4); "
        f"{parent_name}.{shared_expert_name}.gate_up_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4)"
    ]


_MINIMAX_NVFP4_ROUTED_MXFP8_SHARED_QUANT_CONFIG: dict[str, Any] = {
    # Mirrors MiniMax-M3.1's checkpoint: MXFP8 linears and shared experts,
    # compressed-tensors NVFP4 routed experts.
    "quant_method": "mxfp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [1, 32],
    "ignored_layers": [
        "lm_head",
        "model.embed_tokens",
        "vision_tower",
        "multi_modal_projector",
        "patch_merge_mlp",
        "model.layers.0.block_sparse_moe.gate",
    ],
    "group_size": 16,
    "moe_quant_algo": "NVFP4",
    "moe_quant_format": "nvfp4-pack-quantized",
}

def _quark_moe_config(
    routed: dict[str, Any], shared: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Quark MiniMax checkpoint quantizing the routed experts as `routed` and the
    shared experts as `shared` (excluded if None). Other layers are excluded."""
    modules = "experts" if shared is None else "(?:experts|shared_experts)"
    return {
        **_QUARK_MXFP4_CONFIG,
        "global_quant_config": routed,
        "layer_quant_config": (
            {} if shared in (None, routed) else {"*shared_experts*": shared}
        ),
        "exclude": [f"re:^(?!.*\\.block_sparse_moe\\.{modules}(?:\\.|$)).*$"],
    }


# NVFP4: fp4 values with per-16 e4m3 scales, plus a per-tensor fp8_e4m3 scale.
_NVFP4_VALUES = {
    "dtype": "fp4",
    "is_dynamic": False,
    "qscheme": "per_group",
    "ch_axis": -1,
    "group_size": 16,
    "scale_format": "e4m3",
    "observer_cls": "PerBlockMXObserver",
}
_NVFP4_GLOBAL_SCALE = {
    "dtype": "fp8_e4m3",
    "is_dynamic": False,
    "qscheme": "per_tensor",
    "observer_cls": "PerTensorMinMaxObserver",
}
_QUARK_NVFP4_LAYER_CONFIG: dict[str, Any] = {
    "input_tensors": [{**_NVFP4_VALUES, "is_dynamic": True}, _NVFP4_GLOBAL_SCALE],
    "output_tensors": None,
    "weight": [_NVFP4_VALUES, _NVFP4_GLOBAL_SCALE],
    "bias": None,
    "target_device": None,
}
_QUARK_MXFP8_LAYER_CONFIG: dict[str, Any] = {
    **_QUARK_MXFP4_LAYER_CONFIG,
    **{
        name: {**_QUARK_MXFP4_LAYER_CONFIG[name], "dtype": "fp8_e4m3"}
        for name in ("input_tensors", "weight")
    },
}
_QUARK_NVFP4_ROUTED_BF16_SHARED_QUANT_CONFIG = _quark_moe_config(
    _QUARK_NVFP4_LAYER_CONFIG
)
_QUARK_NVFP4_ROUTED_NVFP4_SHARED_QUANT_CONFIG = _quark_moe_config(
    _QUARK_NVFP4_LAYER_CONFIG, _QUARK_NVFP4_LAYER_CONFIG
)
_QUARK_MXFP4_ROUTED_MXFP8_SHARED_QUANT_CONFIG = _quark_moe_config(
    _QUARK_MXFP4_LAYER_CONFIG, _QUARK_MXFP8_LAYER_CONFIG
)
# BF16 (excluded) routed experts, MXFP8 shared experts.
_QUARK_BF16_ROUTED_MXFP8_SHARED_QUANT_CONFIG = {
    **_QUARK_MXFP4_CONFIG,
    "global_quant_config": _QUARK_MXFP8_LAYER_CONFIG,
    "layer_quant_config": {},
    "exclude": ["re:^(?!.*\\.block_sparse_moe\\.shared_experts(?:\\.|$)).*$"],
}

_FSE_ROUTED_SHARED_TARGETS = {
    "*block_sparse_moe.experts*": "mxfp4",
    "*shared_experts*": "mxfp4",
}

_ONLINE_QUANTIZATION_LOGGER = "vllm.model_executor.model_loader.base_loader"
_FSE_LOGGER = "vllm.model_executor.layers.fused_moe.utils"


def _enable_aiter(
    monkeypatch: pytest.MonkeyPatch, *, moe: bool = True, fse: bool | None = None
) -> None:
    """Enable AITER; `fse` sets or unsets fused shared experts, None keeps it."""
    # Other tests reload `envs`, which can leave these flags pinned as attributes
    # that shadow the environment variables.
    for name in (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
    ):
        monkeypatch.delitem(vars(envs), name, raising=False)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    if moe:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    if fse is not None:
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", str(int(fse)))
    rocm_aiter_ops.refresh_env_variables()


def _use_checkpoint_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only resolve the user's quantization config, as the CLI does.

    The checkpoint's own "mxfp8" method is detected into `model_config.quantization`,
    which the helper would otherwise resolve as the online MXFP8 shorthand.
    """
    monkeypatch.setattr(
        "tests.quantization.utils.resolve_quantization_config",
        lambda _quantization, quantization_config: quantization_config,
    )


def _logged_messages(
    caplog: pytest.LogCaptureFixture, logger: str, level: int
) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger and r.levelno == level
    ]


def _online_quantization_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        message
        for message in _logged_messages(
            caplog, _ONLINE_QUANTIZATION_LOGGER, logging.INFO
        )
        if "Quantizing " in message and "of types" in message
    ]


def _fse_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _logged_messages(caplog, _FSE_LOGGER, logging.WARNING)


_ROUTED_MXFP4 = {"*block_sparse_moe.experts*": "mxfp4"}
_ROUTED_SHARED_MXFP4 = {**_ROUTED_MXFP4, "*shared_expert*": "mxfp4"}
# (targets, fse_requested, routed_method_cls, id) of routed experts requantization.
_ROUTED_TARGET_CASES = [
    (_ROUTED_SHARED_MXFP4, True, Mxfp4OnlineMoEMethod, "fse_routed_shared_mxfp4"),
    ({"*block_sparse_moe.experts*": "mxfp8"}, False, Mxfp8OnlineMoEMethod, "routed_mxfp8"),
    (_ROUTED_MXFP4, False, Mxfp4OnlineMoEMethod, "routed_mxfp4"),
    (_ROUTED_SHARED_MXFP4, False, Mxfp4OnlineMoEMethod, "routed_shared_mxfp4"),
    # FSE refused: the shared expert, kept as stored, does not match.
    (_ROUTED_MXFP4, True, Mxfp4OnlineMoEMethod, "fse_refused_routed_mxfp4"),
]


@pytest.mark.parametrize(
    "checkpoint_quant_config, source_cls, targets, fse_requested, routed_method_cls",
    [
        *(
            pytest.param(
                config,
                source_cls,
                targets,
                fse,
                method_cls,
                id=f"{config_id}-{case_id}",
                marks=pytest.mark.skip(
                    # TODO: enable once `QuarkLinearMethod` implements
                    # `dequantize_weight`.
                    reason="Requantizing Quark NVFP4 shared experts is not supported."
                )
                if config is _QUARK_NVFP4_ROUTED_NVFP4_SHARED_QUANT_CONFIG
                and "*shared_expert*" in targets
                else (),
            )
            for config, source_cls, config_id in [
                (
                    _MINIMAX_NVFP4_ROUTED_MXFP8_SHARED_QUANT_CONFIG,
                    CompressedTensorsW4A4Nvfp4MoEMethod,
                    "nvfp4_routed_mxfp8_shared",
                ),
                (
                    _QUARK_NVFP4_ROUTED_BF16_SHARED_QUANT_CONFIG,
                    QuarkNvfp4MoEMethod,
                    "quark_nvfp4_routed_bf16_shared",
                ),
                (
                    _QUARK_NVFP4_ROUTED_NVFP4_SHARED_QUANT_CONFIG,
                    QuarkNvfp4MoEMethod,
                    "quark_nvfp4_routed_nvfp4_shared",
                ),
            ]
            for targets, fse, method_cls, case_id in _ROUTED_TARGET_CASES
        ),
        # Routed experts are not requantized. With FSE, the MXFP8 shared expert is
        # staged and quantized to MXFP4 into its slot after loading. Without, the
        # shared expert linears are requantized from Quark MXFP8.
        *(
            pytest.param(
                _QUARK_MXFP4_ROUTED_MXFP8_SHARED_QUANT_CONFIG,
                None,
                {"*shared_expert*": "mxfp4"},
                fse,
                QuarkOCP_MX_MoEMethod,
                id=f"quark_mxfp4_routed_mxfp8_shared-{case_id}",
            )
            for fse, case_id in ((True, "fse_shared_mxfp4"), (False, "shared_mxfp4"))
        ),
    ],
)
def test_fse_online_quantization_routed_shared(
    checkpoint_quant_config: dict[str, Any],
    source_cls: type,
    targets: dict[str, str],
    fse_requested: bool,
    routed_method_cls: type,
    tmp_path: Path,
    monkeypatch,
    caplog_vllm,
    single_rank_parallel_state,
    workspace_init,
) -> None:
    """Routed and/or shared experts are (re)quantized online. The shared experts
    are fused into the routed experts only with
    `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`.
    """
    architecture = "MiniMaxM3SparseForCausalLM"
    _enable_aiter(monkeypatch, fse=fse_requested)
    _write_minimal_moe_config(tmp_path, architecture, checkpoint_quant_config)
    _use_checkpoint_quantization(monkeypatch)
    model, vllm_config = load_model_without_vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        model_config_kwargs={
            "quantization_config": QuantizationConfigArgs(targets=targets)
        },
        model_loader_cls=DummyModelLoader,
    )
    expected_model_cls, resolved_architecture = get_model_architecture(
        vllm_config.model_config
    )
    assert resolved_architecture == architecture
    assert isinstance(model, expected_model_cls)

    # FSE requested with only the routed experts online quantized: FSE is
    # rightfully refused, as the shared expert (kept as stored) does not run with
    # the routed experts' online keys.
    if fse_requested and "*shared_expert*" not in targets:
        assert any(
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
            "cannot be enabled" in warning
            and "do not match the routed expert keys" in warning
            for warning in _fse_warnings(caplog_vllm)
        ), _fse_warnings(caplog_vllm)
        return

    assert not any(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but "
        "cannot be enabled" in warning
        for warning in _fse_warnings(caplog_vllm)
    ), _fse_warnings(caplog_vllm)

    moes = [
        module
        for module in model.modules()
        if hasattr(module, "is_fused_shared_expert_enabled")
        and hasattr(module, "experts")
    ]
    assert len(moes) == 1
    moe = moes[0]
    assert moe.is_fused_shared_expert_enabled == fse_requested, architecture
    assert (moe.shared_experts is None) == fse_requested

    routed_experts = moe.experts.routed_experts
    assert isinstance(routed_experts.quant_method, routed_method_cls)
    # Without source, the routed experts are not requantized: the MXFP8 shared
    # expert is staged, and quantized to MXFP4 into its slot after loading.
    assert isinstance(
        getattr(routed_experts.quant_method, "requantization_source", None),
        source_cls or type(None),
    )
    num_fused_shared_experts = 1 if fse_requested else 0
    assert (
        routed_experts.expert_map_manager.num_fused_shared_experts
        == num_fused_shared_experts
    )
    num_slots = routed_experts.moe_config.num_logical_experts + num_fused_shared_experts
    assert routed_experts.w13_weight.shape[0] == num_slots
    assert routed_experts.w2_weight.shape[0] == num_slots
    if routed_method_cls is Mxfp8OnlineMoEMethod:
        # MXFP8 scales are stored flattened as [num_slots * rows, K // 32].
        assert routed_experts.w13_weight_scale.shape[0] == (
            num_slots * routed_experts.w13_weight.shape[1]
        )
        assert routed_experts.w2_weight_scale.shape[0] == (
            num_slots * routed_experts.w2_weight.shape[1]
        )
    else:
        assert routed_experts.w13_weight_scale.shape[0] == num_slots
        assert routed_experts.w2_weight_scale.shape[0] == num_slots
    # Staging and source parameters are released after requantization.
    for name in (
        "online_w13_weight",
        "online_w2_weight",
        "online_w13_weight_scale",
        "online_w2_weight_scale",
        "w13_weight_packed",
        "w2_weight_packed",
        "w13_weight_global_scale",
        "w2_weight_global_scale",
    ):
        assert not hasattr(routed_experts, name), name

    routed_target = "*block_sparse_moe.experts*"
    shared_target = "*shared_expert*"
    quantized = []
    if routed_target in targets:
        quantized += [
            f"block_sparse_moe.experts: 1 "
            f"(from targets: {routed_target}, {targets[routed_target]})"
        ]
    if shared_target in targets:
        quantized += [
            f"block_sparse_moe.shared_experts.{projection}: 1 "
            f"(from targets: {shared_target}, {targets[shared_target]})"
            for projection in ("down_proj", "gate_up_proj")
        ]
    assert _online_quantization_logs(caplog_vllm) == [
        f"Quantizing {len(quantized)} layers of types: " + "; ".join(quantized)
    ]

    if fse_requested:
        # The staged shared experts were quantized into their slots once.
        with pytest.raises(NotImplementedError, match="Reloading fused shared"):
            routed_experts.weight_loader(
                routed_experts.w2_weight,
                torch.zeros(1),
                f"{routed_experts.layer_name}.w2_weight",
                "w2",
                routed_experts.moe_config.num_logical_experts,
            )
    if fse_requested and source_cls is None:
        # Requantized routed experts parameters are replaced after processing, and
        # are restored for reloads by the layerwise reload machinery only.
        shared_prefix = routed_experts.moe_config.shared_expert_prefix
        with pytest.raises(NotImplementedError, match="Reloading fused shared"):
            model.load_weights(
                [(f"{shared_prefix}.down_proj.weight", routed_experts.w2_weight[0])]
            )


def test_fse_online_quantization_bf16_routed_quantized_shared_raises(
    tmp_path: Path,
    monkeypatch,
    single_rank_parallel_state,
    workspace_init,
) -> None:
    """Routed experts quantized online from BF16 can not take a staged shared
    expert stored quantized (MXFP8)."""
    architecture = "MiniMaxM3SparseForCausalLM"
    _enable_aiter(monkeypatch, fse=True)
    _write_minimal_moe_config(
        tmp_path, architecture, _QUARK_BF16_ROUTED_MXFP8_SHARED_QUANT_CONFIG
    )
    _use_checkpoint_quantization(monkeypatch)
    with pytest.raises(NotImplementedError, match="quantized online from high"):
        load_model_without_vllm_runner(
            str(tmp_path),
            dtype="bfloat16",
            model_config_kwargs={
                "quantization_config": QuantizationConfigArgs(
                    targets=_ROUTED_SHARED_MXFP4
                )
            },
            model_loader_cls=DummyModelLoader,
        )


def test_fse_online_quantization_routed_shared_loads_checkpoint_weights(
    default_vllm_config,
    single_rank_parallel_state,
) -> None:
    """Routed NVFP4 and shared MXFP8 weights land in their MXFP4 expert slots.

    The weights are loaded through `MiniMaxM3Model.load_weights` with checkpoint names
    (`w1/w2/w3`, `shared_experts.*`, `weight_scale_inv`), as the model loader does.
    """
    from vllm.models.minimax_m3.amd.model import MiniMaxM3Model

    default_vllm_config.model_config = ModelConfig()
    hidden_size = intermediate_size = 64
    num_routed_experts = 2
    shared_slot = num_routed_experts
    device = current_platform.device_type
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        moe_quant_algo="NVFP4",
    )
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets=_FSE_ROUTED_SHARED_TARGETS)
    )

    def nvfp4_weight(rows: int, cols: int):
        packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
        scale = (torch.rand(rows, cols // 16) * 4 + 0.5).to(torch.float8_e4m3fn)
        global_scale = torch.rand(1, dtype=torch.float32) + 0.5
        return packed, scale, global_scale

    def dequantize_nvfp4(packed, scale, global_scale):
        return dequantize_to_dtype(
            packed[None],
            scale[None],
            (1.0 / global_scale).reshape(1),
            torch.bfloat16,
            block_size=16,
            swizzle=False,
        )[0]

    with torch.device(device):
        runner = FusedMoEFactory(
            num_experts=num_routed_experts,
            top_k=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            n_shared_experts=1,
            fuse_shared_experts=True,
            shared_expert_prefix="model.layers.0.block_sparse_moe.shared_experts",
            prefix="model.layers.0.block_sparse_moe.experts",
            quant_config=quant_config,
        )
        layer = runner.routed_experts
        assert isinstance(layer.quant_method, Mxfp4OnlineMoEMethod)

        shapes = {
            "gate_proj": (intermediate_size, hidden_size),
            "up_proj": (intermediate_size, hidden_size),
            "down_proj": (hidden_size, intermediate_size),
        }
        routed = {
            (expert, projection): nvfp4_weight(*shape)
            for expert in range(num_routed_experts)
            for projection, shape in shapes.items()
        }
        shared = {
            projection: mxfp8_e4m3_quantize(
                torch.randn(*shape, dtype=torch.bfloat16),
                is_sf_swizzled_layout=False,
            )
            for projection, shape in shapes.items()
        }

        ckpt_names = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        weights = []
        for (expert, projection), (packed, scale, global_scale) in routed.items():
            prefix = f"block_sparse_moe.experts.{expert}.{ckpt_names[projection]}"
            weights += [
                (f"{prefix}.weight_packed", packed),
                (f"{prefix}.weight_scale", scale),
                (f"{prefix}.weight_global_scale", global_scale),
                (f"{prefix}.input_global_scale", torch.ones(1)),
            ]
        for projection, (weight, scale) in shared.items():
            prefix = f"block_sparse_moe.shared_experts.{projection}"
            weights += [
                (f"{prefix}.weight", weight),
                (f"{prefix}.weight_scale_inv", scale),
            ]

        # Only the parts of `MiniMaxM3Model` that `load_weights` uses.
        class ModelStub(torch.nn.Module):
            get_expert_mapping = MiniMaxM3Model.get_expert_mapping
            load_weights = MiniMaxM3Model.load_weights

        model = ModelStub()
        model.config = SimpleNamespace(
            num_local_experts=num_routed_experts, n_shared_experts=1
        )
        model.is_fused_shared_expert_enabled = True
        model.block_sparse_moe = torch.nn.Module()
        model.block_sparse_moe.experts = runner
        model.load_weights(weights)
        layer.quant_method.process_weights_after_loading(layer)

    def check_slot(slot: int, gate, up, down) -> None:
        for rows, reference in (
            (slice(0, intermediate_size), gate),
            (slice(intermediate_size, 2 * intermediate_size), up),
        ):
            expected, expected_scale = mxfp4_quantize(reference)
            assert torch.equal(layer.w13_weight[slot, rows], expected)
            assert torch.equal(layer.w13_weight_scale[slot, rows], expected_scale)
        expected, expected_scale = mxfp4_quantize(down)
        assert torch.equal(layer.w2_weight[slot], expected)
        assert torch.equal(layer.w2_weight_scale[slot], expected_scale)

    for expert in range(num_routed_experts):
        check_slot(
            expert,
            *(dequantize_nvfp4(*routed[expert, projection]) for projection in shapes),
        )
    check_slot(
        shared_slot,
        *(dequant_mxfp8_to_bf16(*shared[projection]) for projection in shapes),
    )


def test_online_gate_up_proj_target_disables_fse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog_vllm,
    single_rank_parallel_state,
    workspace_init,
) -> None:
    """A partial shared-expert target quantizes while disabling FSE."""
    architecture = "DeepseekForCausalLM"
    target_pattern = "*shared_experts.gate_up_proj*"
    _enable_aiter(monkeypatch, fse=True)
    _write_minimal_moe_config(tmp_path, architecture)

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
    assert _online_quantization_logs(caplog_vllm) == [
        "Quantizing 1 layers of types: "
        "mlp.shared_experts.gate_up_proj: 1 "
        f"(from targets: {target_pattern}, mxfp4)"
    ]
    # The untargeted down_proj is unquantized: its keys do not match.
    [warning] = _fse_warnings(caplog_vllm)
    assert warning.startswith(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled but cannot be "
        "enabled - skipping for this layer: shared expert (weight, activation) keys "
    )
    assert (
        "at model.layers.0.mlp.shared_experts do not match the routed expert keys"
        in warning
    )


@pytest.mark.parametrize("num_shared", [0, 1, 2])
@pytest.mark.parametrize("experts_per_chunk", [1, 3, 4])
def test_online_moe_quantize_weights_places_routed_then_shared_experts(
    num_shared: int, experts_per_chunk: int
):
    """Chunked routed experts come first and the staged shared experts last."""
    import types

    from vllm.model_executor.layers.quantization.online.moe_base import (
        OnlineMoEMethodBase,
    )

    num_routed, w13_shape, w2_shape = 4, (6, 8), (8, 3)
    num_experts = num_routed + num_shared

    def expert_values(shape: tuple[int, ...]) -> torch.Tensor:
        # Every element of expert `e` is `e`, so the slot is recoverable.
        values = torch.arange(num_experts, dtype=torch.float32)
        return values.view(-1, 1, 1).expand(num_experts, *shape).contiguous()

    w13_all, w2_all = expert_values(w13_shape), expert_values(w2_shape)

    def dequantize_weight(layer):
        for start in range(0, num_routed, experts_per_chunk):
            experts = slice(start, min(start + experts_per_chunk, num_routed))
            yield experts, w13_all[experts], w2_all[experts]

    method = SimpleNamespace(
        requantization_source=SimpleNamespace(
            dequantize_weight=dequantize_weight,
        ),
        _pad_requantized_weights=lambda layer, w13, w2, n: (w13, w2),
    )
    method.iter_weights_for_quantization = types.MethodType(
        OnlineMoEMethodBase.iter_weights_for_quantization, method
    )
    layer = SimpleNamespace(
        local_num_experts=num_experts,
        expert_map_manager=SimpleNamespace(num_fused_shared_experts=num_shared),
        orig_dtype=torch.float32,
        shared_expert_online_loader=SimpleNamespace(
            dequantize_staging=lambda layer: (
                w13_all[num_routed:],
                w2_all[num_routed:],
            ),
            release_staging=lambda layer: None,
        ),
    )

    def quantize(w13, w2):
        return w13 * 2, w13.mean(-1), w2 * 2, w2.mean(-1)

    w13_q, w13_scale, w2_q, w2_scale = OnlineMoEMethodBase.quantize_weights(
        method, layer, quantize
    )

    assert torch.equal(w13_q, w13_all * 2)
    assert torch.equal(w13_scale, w13_all.mean(-1))
    assert torch.equal(w2_q, w2_all * 2)
    assert torch.equal(w2_scale, w2_all.mean(-1))


if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
else:
    ROCM_GFX950 = False


@pytest.mark.parametrize("requantize", [False, True], ids=["no_requant", "requant"])
def test_nvfp4_moe_requantization_on_rocm(
    requantize: bool,
    tmp_path: Path,
    default_vllm_config,
    single_rank_parallel_state,
    workspace_init,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVFP4 routed experts must be requantized online on ROCm.

    NVFP4 -> MXFP4 requantization with `--moe-backend aiter` skips the NVFP4 oracle:
    `aiter` is not an NVFP4 backend.
    """
    _enable_aiter(monkeypatch)
    if not requantize:
        _write_minimal_moe_config(
            tmp_path,
            "MiniMaxM3SparseForCausalLM",
            _MINIMAX_NVFP4_ROUTED_MXFP8_SHARED_QUANT_CONFIG,
        )
        _use_checkpoint_quantization(monkeypatch)
        with pytest.raises(
            NotImplementedError,
            match="NVFP4 MoE experts .* is not supported on ROCm",
        ):
            load_model_without_vllm_runner(
                str(tmp_path),
                dtype="bfloat16",
                model_loader_cls=DummyModelLoader,
            )
        return

    if not ROCM_GFX950:
        pytest.skip("AITER MXFP4 MoE requires gfx950")
    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.moe_backend = "aiter"

    nvfp4_oracle_logger = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.nvfp4.logger", nvfp4_oracle_logger
    )

    hidden_size = intermediate_size = 256
    num_experts, top_k = 8, 2
    device = current_platform.device_type
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        moe_quant_algo="NVFP4",
    )
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*block_sparse_moe.experts*": "mxfp4"})
    )

    with torch.device(device):
        runner = FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            router_logits_dtype=torch.float32,
            prefix="model.layers.0.block_sparse_moe.experts",
            quant_config=quant_config,
        )
        layer = runner.routed_experts
        assert layer.moe_config.moe_backend == "aiter"

    # The requantization target selected the AITER MXFP4 backend, and the NVFP4
    # oracle did not run for the checkpoint method: no log from it.
    assert isinstance(layer.quant_method, Mxfp4OnlineMoEMethod)
    assert "AITER" in layer.quant_method.mxfp4_backend.name
    assert isinstance(
        layer.quant_method.requantization_source, CompressedTensorsW4A4Nvfp4MoEMethod
    )
    assert nvfp4_oracle_logger.mock_calls == []


def test_nvfp4_oracle_only_skipped_for_requantized_layers(
    default_vllm_config,
    single_rank_parallel_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the targeted layer's NVFP4 checkpoint method skips backend selection."""
    _enable_aiter(monkeypatch)
    default_vllm_config.model_config = ModelConfig()
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        moe_quant_algo="NVFP4",
    )
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(
            targets={"*layers.0.block_sparse_moe.experts*": "mxfp4"}
        )
    )

    real_select = nvfp4_oracle.select_nvfp4_moe_backend
    selected: list[tuple[str, NvFp4MoeBackend]] = []

    def select(config, weight_key, activation_key):
        skipped = is_requantization_source()
        if skipped:
            backend, experts_cls = real_select(config, weight_key, activation_key)
        else:
            # No native NVFP4 MoE backend on ROCm: pretend one was selected.
            backend, experts_cls = NvFp4MoeBackend.VLLM_CUTLASS, Mock()
        selected.append(("skipped" if skipped else "selected", backend))
        return backend, experts_cls

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.compressed_tensors."
        "compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4."
        "select_nvfp4_moe_backend",
        select,
    )

    layers = []
    with torch.device(current_platform.device_type):
        for layer_idx in range(2):
            runner = FusedMoEFactory(
                num_experts=4,
                top_k=1,
                hidden_size=64,
                intermediate_size=64,
                params_dtype=torch.bfloat16,
                prefix=f"model.layers.{layer_idx}.block_sparse_moe.experts",
                quant_config=quant_config,
            )
            layers.append(runner.routed_experts)

    assert selected == [
        ("skipped", NvFp4MoeBackend.EMULATION),
        ("selected", NvFp4MoeBackend.VLLM_CUTLASS),
    ]
    assert isinstance(layers[0].quant_method, Mxfp4OnlineMoEMethod)
    assert isinstance(
        layers[0].quant_method.requantization_source,
        CompressedTensorsW4A4Nvfp4MoEMethod,
    )
    assert isinstance(layers[1].quant_method, CompressedTensorsW4A4Nvfp4MoEMethod)
    assert not is_requantization_source()
