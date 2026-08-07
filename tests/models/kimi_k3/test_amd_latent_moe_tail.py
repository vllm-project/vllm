# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.kimi_k3.amd.linear import (
    KimiAMDLatentMoERunner,
    KimiRoutedOutputTransform,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 AITER latent-MoE tail requires ROCm",
)


def _transform(
    latent_size: int = 3584,
    output_size: int = 7168,
) -> KimiRoutedOutputTransform:
    transform = object.__new__(KimiRoutedOutputTransform)
    nn.Module.__init__(transform)
    transform.norm = SimpleNamespace(
        weight=torch.empty(latent_size, device="meta"),
        variance_epsilon=1.0e-6,
    )
    transform.up_proj = SimpleNamespace(
        weight=torch.empty(output_size, latent_size, device="meta")
    )
    return transform


def test_forward_with_shared_delegates_to_supported_aiter_kernel(monkeypatch):
    latent_moe_tail_module = importlib.import_module("aiter.ops.flydsl.latent_moe_tail")

    transform = _transform()
    routed = torch.empty(1, 3584)
    shared = torch.empty(1, 7168)
    expected = torch.empty_like(shared)
    calls = []

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        latent_moe_tail_module,
        "supports_latent_moe_tail",
        lambda *args: True,
    )

    def fused_tail(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(latent_moe_tail_module, "latent_moe_tail", fused_tail)

    assert transform.forward_with_shared(routed, shared) is expected
    assert len(calls) == 1
    assert calls[0][0] is routed
    assert calls[0][1] is shared
    assert calls[0][2] is transform.norm.weight
    assert calls[0][3] is transform.up_proj.weight
    assert calls[0][4] == transform.norm.variance_epsilon


def test_forward_with_shared_preserves_fallbacks(monkeypatch):
    transform = _transform()
    routed = torch.empty(8, 3584)
    shared = torch.empty(8, 7168)

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    assert transform.forward_with_shared(routed, shared) is None

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    latent_moe_tail_module = importlib.import_module("aiter.ops.flydsl.latent_moe_tail")

    monkeypatch.setattr(
        latent_moe_tail_module,
        "supports_latent_moe_tail",
        lambda *args: False,
    )
    monkeypatch.setattr(
        latent_moe_tail_module,
        "latent_moe_tail",
        lambda *args: pytest.fail("unsupported inputs must use the fallback"),
    )
    assert transform.forward_with_shared(routed, shared) is None


def test_runner_fuses_supported_tail_and_preserves_fallback(monkeypatch):
    runner = object.__new__(KimiAMDLatentMoERunner)
    nn.Module.__init__(runner)
    runner.routed_scaling_factor = 1.0
    transform = _transform(latent_size=2, output_size=4)
    runner.routed_output_transform = transform

    routed = torch.tensor([[1.0, 2.0]])
    shared = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    fused_result = torch.tensor([[7.0, 8.0, 9.0, 10.0]])

    monkeypatch.setattr(
        transform,
        "forward_with_shared",
        lambda routed, shared: fused_result,
    )
    result = runner.apply_routed_output_transform_and_add_shared(shared, routed)
    assert result is fused_result

    fallback_result = torch.tensor([[11.0, 12.0, 13.0, 14.0]])
    monkeypatch.setattr(
        transform,
        "forward_with_shared",
        lambda routed, shared: None,
    )
    monkeypatch.setattr(transform, "forward", lambda routed: fallback_result)
    result = runner.apply_routed_output_transform_and_add_shared(shared, routed)
    torch.testing.assert_close(result, shared + fallback_result)
