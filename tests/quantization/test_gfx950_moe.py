# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for MXFP4 MoE oracle backend selection on mi355x (GFX950).

These tests run on real hardware — no mocks. Skipped on non-GFX950 platforms.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _requires_qwen38_tep8_emulation,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform

ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()


def set_rocm_aiter(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    value = "1" if enabled else "0"
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", value)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", value)
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", enabled)
    monkeypatch.setattr(rocm_aiter_ops, "_FMOE_ENABLED", enabled)


@pytest.fixture
def enable_rocm_aiter(monkeypatch: pytest.MonkeyPatch):
    set_rocm_aiter(monkeypatch, True)


@pytest.fixture
def disable_rocm_aiter(monkeypatch: pytest.MonkeyPatch):
    set_rocm_aiter(monkeypatch, False)


def _make_w4a4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


def _make_qwen38_tep8_moe_config() -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=512,
        experts_per_token=9,
        hidden_dim=2560,
        intermediate_size=640,
        num_local_experts=64,
        num_logical_experts=512,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            pcp_size=1,
            pcp_rank=0,
            dp_size=1,
            dp_rank=0,
            ep_size=8,
            ep_rank=0,
            sp_size=1,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )


def test_qwen38_tep8_requires_emulation_only_on_gfx950(monkeypatch):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle

    config = _make_qwen38_tep8_moe_config()
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)

    assert _requires_qwen38_tep8_emulation(config, kMxfp4Dynamic)

    config.moe_parallel_config.ep_size = 4
    config.num_local_experts = 128
    assert not _requires_qwen38_tep8_emulation(config, kMxfp4Dynamic)

    config = _make_qwen38_tep8_moe_config()
    monkeypatch.setattr(mxfp4_oracle.current_platform, "is_rocm", lambda: False)
    assert not _requires_qwen38_tep8_emulation(config, kMxfp4Dynamic)


@pytest.mark.parametrize(
    "requested_backend,expected_backend",
    [
        ("auto", Mxfp4MoeBackend.EMULATION),
        ("aiter", Mxfp4MoeBackend.AITER_MXFP4_MXFP4),
    ],
)
def test_qwen38_tep8_auto_fallback_respects_explicit_backend(
    requested_backend,
    expected_backend,
    monkeypatch,
):
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle

    class SupportedExperts:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return True, None

    config = _make_qwen38_tep8_moe_config()
    config.moe_backend = requested_backend
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    monkeypatch.setattr(
        mxfp4_oracle, "backend_to_kernel_cls", lambda backend: [SupportedExperts]
    )

    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )

    assert backend == expected_backend
    assert experts_cls is SupportedExperts


@pytest.fixture
def mxfp4_oracle_config():
    """Stub the config the oracle reads (``model_config.quantization_config``)
    so backend dispatch resolves without a real model / user override."""
    from unittest.mock import patch

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
    ) as mock_get_config:
        mock_get_config.return_value.model_config.quantization_config = None
        yield


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(not ROCM_AITER_SUPPORTED, reason="Requires supported AITER")
def test_w4a4_dispatches_to_aiter(mxfp4_oracle_config, enable_rocm_aiter):
    """With AITER enabled + GFX950, W4A4 selects AITER_MXFP4_MXFP4."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is not None


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
def test_w4a4_falls_back_without_aiter(
    mxfp4_oracle_config,
    disable_rocm_aiter,
):
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
def test_w4a4_dispatches_to_emulation_with_moe_backend(mxfp4_oracle_config):
    """With --moe-backend emulation, W4A4 selects EMULATION."""
    config = _make_w4a4_moe_config(moe_backend="emulation")
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None
