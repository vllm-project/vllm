# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How Nemotron Shortcut picks its experts implementation (no GPU needed).

Every path must end in an experts class that applies the clamp: a generic
ReLU² backend would run and silently drop it.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.models.nemotron_shortcut.nvidia.moe import experts as moe_experts
from vllm.models.nemotron_shortcut.nvidia.moe import flashinfer as moe_flashinfer
from vllm.models.nemotron_shortcut.nvidia.moe.config import NemotronShortcutMoEConfig


def _moe_config(**overrides) -> NemotronShortcutMoEConfig:
    values = dict(
        num_experts=64,
        experts_per_token=4,
        hidden_dim=256,
        intermediate_size=1024,
        num_local_experts=64,
        num_logical_experts=64,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.RELU2_NO_MUL,
        activation_clamp_value=4.0,
        in_dtype=torch.bfloat16,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_backend="triton",
    )
    values.update(overrides)
    return NemotronShortcutMoEConfig(**values)


def _supported(monkeypatch, cls, supported: bool = True) -> None:
    monkeypatch.setattr(cls, "_supports_current_device", lambda: supported)


@pytest.mark.parametrize(
    "override",
    [
        {"pcp_size": 2},
        {"sp_size": 2},
        {"enable_eplb": True},
        {"all2all_backend": "deepep_low_latency"},
        {"placement": "round_robin"},
    ],
)
def test_expert_parallel_rejects_dispatching_tokens(override):
    """The experts run every token, so EP must not redistribute tokens."""
    options = dict(
        use_ep=True,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        placement="linear",
    )
    options.update(override)
    layer = SimpleNamespace(expert_placement_strategy=options.pop("placement"))
    options["is_sequence_parallel"] = override.get("sp_size", 1) != 1
    config = SimpleNamespace(moe_parallel_config=SimpleNamespace(**options))
    with pytest.raises(NotImplementedError):
        moe_experts.NemotronShortcutRoutedExperts._get_quant_method(
            layer, "test", None, config
        )


def test_mixed_precision_resolves_each_expert_group(monkeypatch):
    """One checkpoint may mix NVFP4 experts, MXFP8 experts and BF16 experts."""
    quant = moe_experts.ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.moe.experts": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                },
                "model.language_model.layers.1.moe.experts": {"quant_algo": "MXFP8"},
                "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
            },
        }
    )
    monkeypatch.setattr(moe_experts.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        moe_experts.current_platform, "has_device_capability", lambda _: True
    )
    config = _moe_config()
    for index, name, args in (
        (0, "NemotronShortcutModelOptNvFp4MoEMethod", (quant.nvfp4_config, config)),
        (1, "NemotronShortcutModelOptMxfp8MoEMethod", (config,)),
        (2, "NemotronShortcutMoEMethod", (config,)),
    ):
        factory = Mock()
        monkeypatch.setattr(moe_experts, name, factory)
        result = moe_experts.NemotronShortcutRoutedExperts._get_quant_method(
            SimpleNamespace(),
            f"language_model.model.layers.{index}.moe.experts",
            quant,
            config,
        )
        assert result is factory.return_value
        factory.assert_called_once_with(*args)


@pytest.mark.parametrize("flashinfer", [False, True])
def test_unquantized_experts_prefer_flashinfer(monkeypatch, flashinfer):
    _supported(
        monkeypatch, moe_flashinfer.NemotronShortcutFlashInferExperts, flashinfer
    )
    method = moe_experts.NemotronShortcutMoEMethod(_moe_config(moe_backend="auto"))
    expected = (
        moe_flashinfer.NemotronShortcutFlashInferExperts
        if flashinfer
        else moe_experts.NemotronShortcutTritonExperts
    )
    assert type(method.moe_kernel.fused_experts) is expected


@pytest.mark.parametrize(
    "backend,cute_dsl,expected",
    [
        ("auto", True, "cute_dsl"),
        ("auto", False, "triton"),
        ("triton", True, "triton"),
        ("flashinfer_cutedsl", False, "CuTe-DSL MXFP8 experts are unavailable"),
        ("marlin", True, "does not implement"),
        ("flashinfer_trtllm", True, "does not implement"),
    ],
)
def test_mxfp8_experts_selection(monkeypatch, backend, cute_dsl, expected):
    """Only backends that apply the clamp are chosen; the others are refused."""
    _supported(
        monkeypatch, moe_flashinfer.NemotronShortcutCuteDslMxfp8Experts, cute_dsl
    )
    _supported(monkeypatch, moe_experts.NemotronShortcutMxfp8TritonExperts)
    config = _moe_config(moe_backend=backend)
    classes = {
        "cute_dsl": moe_flashinfer.NemotronShortcutCuteDslMxfp8Experts,
        "triton": moe_experts.NemotronShortcutMxfp8TritonExperts,
    }
    if expected not in classes:
        with pytest.raises(ValueError, match=expected):
            moe_experts.select_mxfp8_experts(config)
        return
    assert moe_experts.select_mxfp8_experts(config) is classes[expected]


@pytest.mark.parametrize("num_experts,local,offset", [(64, 32, 32), (10, 3, None)])
def test_nvfp4_expert_parallel_requires_even_partitions(
    monkeypatch, num_experts, local, offset
):
    monkeypatch.setattr(moe_flashinfer, "has_cute_dsl_options", lambda *_: True)
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        use_ep=True,
        ep_size=2 if offset else 3,
        ep_rank=1,
        dp_size=2 if offset else 3,
        dp_rank=1,
    )
    config = _moe_config(
        num_experts=num_experts,
        num_logical_experts=num_experts,
        experts_per_token=2,
        num_local_experts=local,
        moe_parallel_config=parallel,
    )
    quant = FusedMoEQuantConfig.make(quant_dtype="nvfp4")
    if offset is None:
        with pytest.raises(NotImplementedError, match="split evenly"):
            moe_flashinfer.NemotronShortcutCuteDslNvFp4Experts(config, quant)
        return
    experts = moe_flashinfer.NemotronShortcutCuteDslNvFp4Experts(config, quant)
    assert experts.local_expert_offset == offset
    assert experts.clamp == 4.0


def test_experts_claim_only_the_non_gated_relu2():
    for cls in (
        moe_experts.NemotronShortcutTritonExperts,
        moe_experts.NemotronShortcutMxfp8TritonExperts,
        moe_flashinfer.NemotronShortcutFlashInferExperts,
        moe_flashinfer.NemotronShortcutCuteDslMxfp8Experts,
    ):
        assert cls._supports_activation(MoEActivation.RELU2_NO_MUL)
        assert not cls._supports_activation(MoEActivation.SILU)
