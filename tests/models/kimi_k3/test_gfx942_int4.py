# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _moe_weight_override_is_int4,
    _use_k3_situ_int4_gfx942,
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


def test_gfx942_int4_preserves_native_k3_intermediate_size():
    method = object.__new__(Mxfp4MoEMethod)
    method.is_k3_situ_int4_gfx942 = True

    with patch.object(
        FusedMoEMethodBase,
        "maybe_roundup_sizes",
        return_value=(7168, 384),
    ):
        hidden_size, intermediate_size = method.maybe_roundup_sizes(
            hidden_size=7168,
            intermediate_size_per_partition=384,
            act_dtype=torch.bfloat16,
            moe_parallel_config=Mock(),
        )

    assert hidden_size == 7168
    assert intermediate_size == 384
