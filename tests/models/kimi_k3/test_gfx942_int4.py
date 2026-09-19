# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.mxfp4 import (
    _moe_weight_override_is_int4,
    _use_k3_situ_int4_gfx942,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)

pytestmark = pytest.mark.cpu_test


class _FakeRocmModule(ModuleType):
    on_gfx942: Callable[[], bool]


class _FakeAiterOpsModule(ModuleType):
    rocm_aiter_ops: SimpleNamespace


def _vllm_config(moe_weight: str | None):
    quantization_config = (
        None
        if moe_weight is None
        else QuantizationConfigArgs(moe={"weight": moe_weight})
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization_config=quantization_config)
    )


@pytest.mark.parametrize(
    ("moe_weight", "expected"),
    [
        (None, False),
        ("mxfp4", False),
        ("int4_per_group_32", True),
    ],
)
def test_moe_weight_override_is_int4(monkeypatch, moe_weight, expected):
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: _vllm_config(moe_weight),
    )
    assert _moe_weight_override_is_int4() is expected


@pytest.mark.parametrize("int4_requested", [False, True])
def test_k3_situ_int4_gfx942_requires_explicit_override(int4_requested):
    moe = SimpleNamespace(
        activation=MoEActivation.SITU,
        activation_situ_linear_beta=25.0,
    )
    fake_rocm = _FakeRocmModule("vllm.platforms.rocm")
    fake_rocm.on_gfx942 = lambda: True
    fake_aiter_ops = _FakeAiterOpsModule("vllm._aiter_ops")
    fake_aiter_ops.rocm_aiter_ops = SimpleNamespace(is_fused_moe_enabled=lambda: True)
    with (
        patch("vllm.platforms.current_platform.is_rocm", return_value=True),
        patch.dict(
            sys.modules,
            {
                "vllm.platforms.rocm": fake_rocm,
                "vllm._aiter_ops": fake_aiter_ops,
            },
        ),
        patch(
            "vllm.model_executor.layers.quantization.mxfp4."
            "_moe_weight_override_is_int4",
            return_value=int4_requested,
        ),
    ):
        assert _use_k3_situ_int4_gfx942(moe) is int4_requested


def test_int4_per_group_32_is_not_an_online_moe_method():
    # --quantization-config.moe.weight int4_per_group_32 is a load-time
    # MXFP4 requantization for gfx942 Kimi-K3. It must not fail-close in
    # the online MoE table, or Mxfp4MoEMethod never runs.
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(moe={"weight": "int4_per_group_32"})
    )
    assert (
        config.resolve_quant_method_cls(
            Mock(spec=RoutedExperts), "model.layers.0.mlp.experts"
        )
        is None
    )


def test_unknown_online_moe_weight_still_raises():
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(moe={"weight": "fp8_per_token"})
    )
    with pytest.raises(ValueError, match="online quantization"):
        config.resolve_quant_method_cls(
            Mock(spec=RoutedExperts), "model.layers.0.mlp.experts"
        )
