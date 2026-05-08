# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM"),
        ("vllm.model_executor.models.AXK1", "AXK1ForCausalLM"),
    ],
)
@pytest.mark.parametrize("shared_experts_enabled", [False, True])
def test_get_expert_mapping_matches_load_weights_expert_count(
    monkeypatch, module_name, class_name, shared_experts_enabled
):
    module = pytest.importorskip(module_name)
    model_cls = getattr(module, class_name)

    captured_kwargs = {}

    def fake_make_mapping(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(
        module,
        "fused_moe_make_expert_params_mapping",
        fake_make_mapping,
    )
    monkeypatch.setattr(
        module.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: shared_experts_enabled,
    )

    model = SimpleNamespace(
        num_routed_experts=8,
        num_shared_experts=2,
        num_redundant_experts=3,
    )

    assert model_cls.get_expert_mapping(model) == []

    expected_num_experts = 10 if shared_experts_enabled else 8
    assert captured_kwargs["num_experts"] == expected_num_experts
    assert captured_kwargs["num_redundant_experts"] == 3


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("vllm.model_executor.models.deepseek_v2", "DeepseekV2ForCausalLM"),
        ("vllm.model_executor.models.AXK1", "AXK1ForCausalLM"),
    ],
)
def test_get_expert_mapping_handles_missing_expert_metadata(
    monkeypatch, module_name, class_name
):
    module = pytest.importorskip(module_name)
    model_cls = getattr(module, class_name)

    captured_kwargs = {}

    def fake_make_mapping(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(
        module,
        "fused_moe_make_expert_params_mapping",
        fake_make_mapping,
    )
    monkeypatch.setattr(
        module.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )

    model = SimpleNamespace(
        num_routed_experts=None,
        num_shared_experts=None,
        num_redundant_experts=None,
    )

    assert model_cls.get_expert_mapping(model) == []
    assert captured_kwargs["num_experts"] == 0
    assert captured_kwargs["num_redundant_experts"] == 0
