# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 pre-route fusion requires ROCm",
)


def test_preroute_factory_owns_model_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.kimi_k3.amd.ops.moe_preroute import (
        KimiK3PrerouteBf16,
    )

    monkeypatch.setattr(
        KimiK3PrerouteBf16,
        "is_backend_available",
        staticmethod(lambda: True),
    )
    supported = {
        "use_latent_moe": True,
        "tensor_parallel_size": 8,
        "shared_experts": object(),
        "routed_projection": object(),
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "lora_enabled": False,
    }
    assert KimiK3PrerouteBf16.create_if_supported(**supported) is not None

    unsupported = (
        {"use_latent_moe": False},
        {"tensor_parallel_size": 4},
        {"shared_experts": None},
        {"routed_projection": None},
        {"situ_beta": None},
        {"situ_linear_beta": None},
        {"lora_enabled": True},
    )
    for override in unsupported:
        config = supported | override
        assert KimiK3PrerouteBf16.create_if_supported(**config) is None


def _relative_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (error / reference).item()


def test_amd_moe_preroute_bf16_matches_reference() -> None:
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.utils import is_flydsl_available

    from vllm.models.kimi_k3.amd.ops.moe_preroute import (
        KimiK3PrerouteBf16,
    )

    if not is_flydsl_available() or get_gfx_runtime() != "gfx950":
        pytest.skip("requires FlyDSL on gfx950")

    torch.manual_seed(20260729)
    hidden = torch.randn(
        (1, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    routed_weight = torch.randn(
        (3584, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_gate_up_weight = torch.randn(
        (1536, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_down_weight = torch.randn(
        (7168, 768),
        device="cuda",
        dtype=torch.bfloat16,
    )
    original_hidden = hidden.clone()

    preroute = KimiK3PrerouteBf16(
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    output = preroute(
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )
    assert output is not None
    routed, shared = output

    routed_reference = F.linear(
        hidden.float(),
        routed_weight.float(),
    ).to(torch.bfloat16)
    gate_up_reference = F.linear(
        hidden.float(),
        shared_gate_up_weight.float(),
    ).to(torch.bfloat16)
    gate, up = gate_up_reference.float().chunk(2, dim=-1)
    activated = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    shared_reference = F.linear(
        activated.float(),
        shared_down_weight.float(),
    ).to(torch.bfloat16)

    assert _relative_rmse(routed, routed_reference) < 2e-4
    assert _relative_rmse(shared, shared_reference) < 2e-4
    torch.testing.assert_close(
        hidden,
        original_hidden,
        atol=0,
        rtol=0,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = preroute(
            hidden,
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
    assert captured is not None
    captured_routed, captured_shared = captured
    graph.replay()
    expected_routed = captured_routed.clone()
    expected_shared = captured_shared.clone()
    graph.replay()
    torch.testing.assert_close(
        captured_routed,
        expected_routed,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        captured_shared,
        expected_shared,
        atol=0,
        rtol=0,
    )


def test_fp8_factory_rejects_incomplete_model_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.kimi_k3.amd.ops.moe_preroute import (
        KimiK3PrerouteFp8Weights,
    )

    monkeypatch.setattr(
        KimiK3PrerouteFp8Weights,
        "is_backend_available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        KimiK3PrerouteFp8Weights,
        "supports_weights",
        staticmethod(lambda *_: True),
    )
    source_weights = (torch.empty(1), torch.empty(1), torch.empty(1))
    unsupported = (
        {"use_latent_moe": False},
        {"tensor_parallel_size": 4},
        {"source_weights": None},
        {"situ_beta": None},
        {"situ_linear_beta": None},
        {"lora_enabled": True},
    )
    base = {
        "use_latent_moe": True,
        "tensor_parallel_size": 8,
        "source_weights": source_weights,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "lora_enabled": False,
    }
    for override in unsupported:
        assert KimiK3PrerouteFp8Weights.create_if_supported(**(base | override)) is None


def test_amd_moe_preroute_fp8_matches_weight_quantized_reference() -> None:
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.utils import is_flydsl_available

    from vllm.models.kimi_k3.amd.ops.moe_preroute import (
        KimiK3PrerouteFp8Weights,
    )

    if not is_flydsl_available() or get_gfx_runtime() != "gfx950":
        pytest.skip("requires FlyDSL on gfx950")

    torch.manual_seed(20260729)
    hidden = torch.randn((1, 7168), device="cuda", dtype=torch.bfloat16)
    routed_weight = torch.randn(
        (3584, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_gate_up_weight = torch.randn(
        (1536, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_down_weight = torch.randn(
        (7168, 768),
        device="cuda",
        dtype=torch.bfloat16,
    )
    router_weight = torch.randn(
        (896, 7168),
        device="cuda",
        dtype=torch.bfloat16,
    )
    original_hidden = hidden.clone()

    weights = KimiK3PrerouteFp8Weights(
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )
    assert weights.supports_inputs(hidden, router_weight)
    assert not weights.supports_inputs(hidden.expand(8, -1), router_weight)

    routed, shared, router_logits = weights(
        hidden,
        router_weight,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    routed_dequant = weights.routed_weight.float() * weights.routed_scale[:, None]
    shared_gate_up_dequant = (
        weights.shared_gate_up_weight.float() * weights.shared_gate_up_scale[:, None]
    )
    shared_down_dequant = (
        weights.shared_down_weight.float() * weights.shared_down_scale[:, None]
    )
    routed_reference = F.linear(hidden.float(), routed_dequant)
    gate_up_reference = F.linear(hidden.float(), shared_gate_up_dequant)
    gate, up = gate_up_reference.to(torch.bfloat16).float().chunk(2, dim=-1)
    activated = (
        (
            4.0
            * torch.tanh(gate / 4.0)
            * torch.sigmoid(gate)
            * 25.0
            * torch.tanh(up / 25.0)
        )
        .to(torch.bfloat16)
        .float()
    )
    shared_reference = F.linear(activated, shared_down_dequant)
    router_reference = F.linear(hidden, router_weight).float()

    assert _relative_rmse(routed, routed_reference) < 0.035
    assert _relative_rmse(shared, shared_reference) < 0.06
    assert _relative_rmse(router_logits, router_reference) < 0.01
    torch.testing.assert_close(
        router_logits.topk(16, dim=-1).indices,
        router_reference.topk(16, dim=-1).indices,
        atol=0,
        rtol=0,
    )
    assert F.cosine_similarity(shared.float(), shared_reference.float()).item() > 0.998
    torch.testing.assert_close(hidden, original_hidden, atol=0, rtol=0)
