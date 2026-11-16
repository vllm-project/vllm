# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP4 MoE oracle backend selection on mi355x (GFX950).

Contract tests run without a GPU; hardware selection tests require GFX950.
"""

import pytest
import torch

from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _backend_activation_key,
    convert_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform

ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False


def test_situ_mxfp8_backend_quantization_contract():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterMxfp4Mxfp8Experts,
    )

    backend = Mxfp4MoeBackend.AITER_MXFP4_MXFP8
    scales = torch.ones(8, 16, dtype=torch.uint8)
    quant_config = make_mxfp4_moe_quant_config(backend, scales, scales)

    assert backend in map_mxfp4_backend("aiter")
    assert _backend_activation_key(backend) == kMxfp8Dynamic
    assert quant_config is not None
    assert quant_config.quant_dtype == "mxfp8"
    assert quant_config.weight_quant_dtype == "mxfp4"
    assert AiterMxfp4Mxfp8Experts._supports_quant_scheme(kMxfp4Static, kMxfp8Dynamic)
    assert not AiterMxfp4Mxfp8Experts._supports_quant_scheme(
        kMxfp4Static, kMxfp4Dynamic
    )
    assert AiterMxfp4Mxfp8Experts._supports_activation(MoEActivation.SITU)
    assert not AiterMxfp4Mxfp8Experts._supports_activation(MoEActivation.SILU)


def test_situ_mxfp8_custom_op_schema_matches():
    from vllm._aiter_ops import (
        _rocm_aiter_fused_moe_fake,
        _rocm_aiter_fused_moe_impl,
    )

    real = torch.library.infer_schema(_rocm_aiter_fused_moe_impl, mutates_args={})
    fake = torch.library.infer_schema(_rocm_aiter_fused_moe_fake, mutates_args={})

    assert real == fake
    assert "q_dtype_a2" in real


def test_situ_mxfp8_keeps_tp8_intermediate_size():
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.AITER_MXFP4_MXFP8,
        3584,
        384,
        activation=MoEActivation.SITU,
    ) == (3584, 384)


@pytest.mark.parametrize("num_tokens", [1, 255, 256, 257])
def test_situ_mxfp8_forwards_both_stage_dtypes(monkeypatch, num_tokens):
    import sys
    from enum import IntEnum
    from types import ModuleType

    from vllm._aiter_ops import _rocm_aiter_fused_moe_impl

    activation_type = IntEnum("activation_type", {"Situv2": 1})
    quant_type = IntEnum("quant_type", {"per_1x32": 3})
    aiter_module = ModuleType("aiter")
    fused_module = ModuleType("aiter.fused_moe")

    def fused_moe(hidden_states, *args, **kwargs):
        assert kwargs["quant_dtype_a"] == torch.float8_e4m3fn
        assert kwargs["quant_dtype_a2"] == torch.float8_e4m3fn
        return hidden_states.clone()

    monkeypatch.setattr(aiter_module, "ActivationType", activation_type, raising=False)
    monkeypatch.setattr(aiter_module, "QuantType", quant_type, raising=False)
    monkeypatch.setattr(fused_module, "fused_moe", fused_moe, raising=False)
    monkeypatch.setitem(sys.modules, "aiter", aiter_module)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fused_module)
    hidden = torch.ones(num_tokens, 128, dtype=torch.bfloat16)
    weights = torch.empty(0, dtype=torch.uint8)
    output = _rocm_aiter_fused_moe_impl(
        hidden,
        weights,
        weights,
        torch.ones(num_tokens, 1),
        torch.zeros(num_tokens, 1, dtype=torch.int32),
        activation_method=activation_type.Situv2,
        quant_method=quant_type.per_1x32,
        q_dtype_a=torch.float8_e4m3fn,
        q_dtype_a2=torch.float8_e4m3fn,
    )

    assert torch.equal(output, hidden)


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


@pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
@pytest.mark.skipif(not ROCM_AITER_SUPPORTED, reason="Requires supported AITER")
def test_situ_mxfp8_layout_ignores_legacy_switches(enable_rocm_aiter, monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    weights = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
    w13 = weights.repeat(2 * 768).reshape(2, 768, 256)
    w2 = weights.repeat(2 * 384).reshape(2, 512, 192)
    w13_scale = torch.full((2, 768, 16), 127, dtype=torch.uint8, device="cuda")
    w2_scale = torch.full((2, 512, 12), 127, dtype=torch.uint8, device="cuda")
    converted = []

    for enabled, bound in (("0", "0"), ("1", "4096")):
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE_SITUV2", enabled)
        monkeypatch.setenv("AITER_SITUV2_A8W4", enabled)
        monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", bound)
        result = convert_weight_to_mxfp4_moe_kernel_format(
            Mxfp4MoeBackend.AITER_MXFP4_MXFP8,
            torch.nn.Module(),
            w13,
            w2,
            w13_scale,
            w2_scale,
            activation=MoEActivation.SITU,
        )
        assert result[0].is_shuffled
        assert result[1].is_shuffled
        converted.append(result[:4])

    for disabled_tensor, enabled_tensor in zip(*converted):
        assert torch.equal(
            disabled_tensor.view(torch.uint8), enabled_tensor.view(torch.uint8)
        )


def _make_w4a4_moe_config(moe_backend: MoEBackend = "auto") -> FusedMoEConfig:
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


@pytest.mark.parametrize("requested_backend", ["auto", "aiter"])
def test_qwen38_tep8_selects_native_aiter(requested_backend, monkeypatch):
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

    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is SupportedExperts


@pytest.fixture
def mxfp4_oracle_config():
    """Stub the config the oracle reads (`model_config.quantization_config`)
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
