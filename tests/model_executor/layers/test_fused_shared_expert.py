# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from torch import nn

import vllm.config as vllm_config_module
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.model_executor.layers.fused_moe.layer import determine_expert_counts
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v4 import quant_config as deepseek_v4_quant_config


def test_determine_expert_counts_fuse_shared_experts_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.layer.rocm_aiter_ops."
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.layer.envs."
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
        False,
    )

    common_args = (8, 0, 2, True)
    assert determine_expert_counts(*common_args, True)[2] == 2
    assert determine_expert_counts(*common_args, False)[2] == 0
    assert determine_expert_counts(*common_args, None)[2] == 0


def test_resolve_fused_shared_expert_fusion_skips_compatibility_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "is_shared_expert_quant_fse_compatible",
        lambda *_: pytest.fail(
            "compatibility must not be checked when FSE is disabled"
        ),
    )

    assert not fused_moe_utils.resolve_fused_shared_expert_fusion(
        object(), "model.layers.0.mlp"
    )


def test_resolve_fused_shared_expert_fusion_normalizes_unavailable_aiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: None,
    )

    assert (
        fused_moe_utils.resolve_fused_shared_expert_fusion(
            object(), "model.layers.0.mlp"
        )
        is False
    )


def test_resolve_fused_shared_expert_fusion_passes_module_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = object()
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )

    calls: list[tuple[object, str, str]] = []

    def check_compatibility(
        config: object, expert_prefix: str, shared_expert_prefix: str
    ) -> tuple[bool, str | None]:
        calls.append((config, expert_prefix, shared_expert_prefix))
        return True, None

    monkeypatch.setattr(
        fused_moe_utils, "is_shared_expert_quant_fse_compatible", check_compatibility
    )

    assert fused_moe_utils.resolve_fused_shared_expert_fusion(
        quant_config,
        "model.layers.0.mlp",
        shared_expert_name="shared_expert",
    )
    assert calls == [
        (
            quant_config,
            "model.layers.0.mlp.experts",
            "model.layers.0.mlp.shared_expert",
        )
    ]


def test_resolve_fused_shared_expert_fusion_rejects_incompatible_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "is_shared_expert_quant_fse_compatible",
        lambda *_: (False, "shared experts are excluded"),
    )

    with pytest.raises(ValueError, match="shared experts are excluded"):
        fused_moe_utils.resolve_fused_shared_expert_fusion(
            object(), "model.layers.0.mlp"
        )


def test_deepseek_v4_shared_expert_fse_uses_mtp_quantization_config_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeepseekV4Config:
        expert_dtype = "fp4"

    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        quantization_config={
            "layer_quant_config": {
                "mtp.0.ffn.shared_experts.w1": {"weight": {"dtype": "fp4"}}
            },
            "global_quant_config": {"weight": {"dtype": "fp8"}},
        },
    )
    monkeypatch.setattr(
        deepseek_v4_quant_config, "DeepseekV4FP8Config", DeepseekV4Config
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        DeepseekV4Config(),
        "model.layers.2.ffn.experts",
        "model.layers.2.ffn.shared_experts",
    )

    assert compatible
    assert reason is None


def test_resolve_model_fused_shared_expert_fusion_requires_consistent_layers() -> None:
    class MoE(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.is_fused_shared_expert_enabled = enabled

    class Layer(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.mlp = MoE(enabled)

    enabled_layers = nn.ModuleList([Layer(True)])
    disabled_layers = nn.ModuleList([Layer(False)])
    mixed_layers = nn.ModuleList([Layer(True), Layer(False)])
    empty_layers = nn.ModuleList()
    pipeline_layers = nn.ModuleList([Layer(True), PPMissingLayer()])

    assert fused_moe_utils.resolve_model_fused_shared_expert_fusion(
        enabled_layers, MoE, "mlp"
    )
    assert not fused_moe_utils.resolve_model_fused_shared_expert_fusion(
        disabled_layers, MoE, "mlp"
    )
    assert not fused_moe_utils.resolve_model_fused_shared_expert_fusion(
        empty_layers, MoE, "mlp"
    )
    assert fused_moe_utils.resolve_model_fused_shared_expert_fusion(
        pipeline_layers, MoE, "mlp"
    )
    with pytest.raises(NotImplementedError, match="1 enabled and 1 disabled layers"):
        fused_moe_utils.resolve_model_fused_shared_expert_fusion(
            mixed_layers, MoE, "mlp"
        )


@pytest.mark.parametrize(
    ("exclude", "expected"),
    [([], True), (["*.shared_expert.*"], False)],
)
def test_quark_shared_expert_fse_compatibility(
    exclude: list[str], expected: bool
) -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        QuarkConfig(
            {
                "exclude": exclude,
                "global_quant_config": {},
                "layer_quant_config": {},
            }
        ),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert compatible is expected
    if expected:
        assert reason is None
    else:
        assert (
            reason
            == "Quark excludes shared experts at model.layers.0.mlp.shared_expert"
        )


def test_quark_shared_expert_fse_requires_matching_layer_quant_configs() -> None:
    global_quant_config = {"weight": {"dtype": "fp4"}}
    quant_config = QuarkConfig(
        {
            "exclude": [],
            "global_quant_config": global_quant_config,
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.gate_up_proj": {
                    "weight": {"dtype": "fp8"}
                },
                "model.layers.0.mlp.shared_expert.down_proj": {
                    "weight": {"dtype": "fp8"}
                },
            },
        }
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert not compatible
    assert reason == (
        "Quark uses different quantization configurations for routed and "
        "shared experts at model.layers.0.mlp.shared_expert"
    )


def test_quark_layer_config_from_name_checks_packed_projections() -> None:
    quant_config = QuarkConfig(
        {
            "global_quant_config": {"weight": {"dtype": "fp4"}},
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.w1": {"weight": {"dtype": "fp4"}},
                "model.layers.0.mlp.shared_expert.w3": {"weight": {"dtype": "fp8"}},
            },
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}

    with pytest.raises(ValueError, match="requires all to use the same scheme"):
        quant_config.get_layer_quant_config_from_name(
            "model.layers.0.mlp.shared_expert.gate_up_proj"
        )


def test_quark_packed_layer_config_must_match_global_config() -> None:
    quant_config = QuarkConfig(
        {
            "global_quant_config": {"weight": {"dtype": "fp4"}},
            "layer_type_quant_config": {},
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.w1": {"weight": {"dtype": "fp8"}},
            },
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}

    with pytest.raises(ValueError, match="requires all to use the same scheme"):
        quant_config._find_matched_config(
            "model.layers.0.mlp.shared_expert.gate_up_proj", nn.Module()
        )


def test_non_quark_shared_expert_fse_is_incompatible() -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        object(),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "shared-expert FSE quantization compatibility is not implemented for object"
    )


@pytest.mark.parametrize(
    ("linear", "moe", "ignore", "expected"),
    [
        ("mxfp4", "mxfp4", [], True),
        ("mxfp4", "fp8_per_tensor", [], False),
        (None, "mxfp4", [], False),
        ("mxfp4", None, [], False),
        ("mxfp4", "mxfp4", ["re:.*\\.experts"], False),
        ("mxfp4", "mxfp4", ["re:.*shared_expert\\.gate_up_proj"], False),
    ],
)
def test_online_shared_expert_fse_requires_matching_linear_and_moe_configs(
    linear: str | None,
    moe: str | None,
    ignore: list[str],
    expected: bool,
) -> None:
    quant_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(linear=linear, moe=moe, ignore=ignore)
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert compatible is expected
    assert (reason is None) is expected
