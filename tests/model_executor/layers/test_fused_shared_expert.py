# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

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
        QuarkConfig({"exclude": exclude}),
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
