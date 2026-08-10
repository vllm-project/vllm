# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)


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
        lambda *_: pytest.fail("compatibility must not be checked when FSE is disabled"),
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
    enabled_layer = SimpleNamespace(is_fused_shared_expert_enabled=True)
    disabled_layer = SimpleNamespace(is_fused_shared_expert_enabled=False)

    assert fused_moe_utils.resolve_model_fused_shared_expert_fusion([enabled_layer])
    assert not fused_moe_utils.resolve_model_fused_shared_expert_fusion([disabled_layer])
    assert not fused_moe_utils.resolve_model_fused_shared_expert_fusion([])
    with pytest.raises(
        NotImplementedError, match="1 enabled and 1 disabled layers"
    ):
        fused_moe_utils.resolve_model_fused_shared_expert_fusion(
            [enabled_layer, disabled_layer]
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
        assert reason == "Quark excludes shared experts at model.layers.0.mlp.shared_expert"


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
