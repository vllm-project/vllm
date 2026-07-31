# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.models.kimi_k3.amd import linear
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 FP8 latent-MoE tail requires ROCm",
)


def _transform() -> linear.KimiRoutedOutputTransform:
    transform = object.__new__(linear.KimiRoutedOutputTransform)
    nn.Module.__init__(transform)
    transform.norm = SimpleNamespace(
        weight=torch.empty(1),
        variance_epsilon=1.0e-6,
    )
    transform.up_proj = SimpleNamespace(weight=torch.empty(1))
    transform.register_buffer("_latent_tail_fp8_weight", None, persistent=False)
    transform.register_buffer("_latent_tail_fp8_scale", None, persistent=False)
    return transform


def test_finalize_fp8_weight_owns_prepacked_tensors(monkeypatch):
    transform = _transform()
    packed = torch.empty(1, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, dtype=torch.float32)
    fake_module = ModuleType("aiter.ops.flydsl.latent_moe_tail_fp8")
    vars(fake_module)["quantize_latent_moe_tail_weight"] = lambda weight: (
        packed,
        scale,
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.latent_moe_tail_fp8",
        fake_module,
    )
    monkeypatch.setattr(envs, "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8", True)

    transform.finalize_fp8_weight()

    assert transform._latent_tail_fp8_weight is packed
    assert transform._latent_tail_fp8_scale is scale


def test_fp8_dispatch_uses_prepacked_weight(monkeypatch):
    transform = _transform()
    packed = torch.empty(1, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, dtype=torch.float32)
    transform._latent_tail_fp8_weight = packed
    transform._latent_tail_fp8_scale = scale
    monkeypatch.setattr(envs, "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8", True)
    monkeypatch.setattr(linear.rocm_aiter_ops, "is_enabled", lambda: True)

    expected = torch.empty(1)
    calls = []
    fake_module = ModuleType("aiter.ops.flydsl.latent_moe_tail_fp8")
    vars(fake_module)["supports_latent_moe_tail_fp8"] = lambda *args: True

    def fake_tail(*args):
        calls.append(args)
        return expected

    vars(fake_module)["latent_moe_tail_fp8"] = fake_tail
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.latent_moe_tail_fp8",
        fake_module,
    )
    hidden = torch.empty(1)
    shared = torch.empty(1)

    assert transform.forward_with_shared(hidden, shared) is expected
    assert calls == [
        (
            hidden,
            shared,
            transform.norm.weight,
            packed,
            scale,
            transform.norm.variance_epsilon,
        )
    ]


def test_fp8_unsupported_contract_falls_back_to_bf16(monkeypatch):
    transform = _transform()
    transform._latent_tail_fp8_weight = torch.empty(1, dtype=torch.float8_e4m3fn)
    transform._latent_tail_fp8_scale = torch.empty(1, dtype=torch.float32)
    monkeypatch.setattr(envs, "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8", True)
    monkeypatch.setattr(linear.rocm_aiter_ops, "is_enabled", lambda: True)

    fake_fp8 = ModuleType("aiter.ops.flydsl.latent_moe_tail_fp8")
    vars(fake_fp8)["supports_latent_moe_tail_fp8"] = lambda *args: False
    vars(fake_fp8)["latent_moe_tail_fp8"] = lambda *args: pytest.fail(
        "unexpected FP8 call"
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.latent_moe_tail_fp8",
        fake_fp8,
    )
    expected = torch.empty(1)
    fake_bf16 = ModuleType("aiter.ops.flydsl.latent_moe_tail")
    vars(fake_bf16)["supports_latent_moe_tail"] = lambda *args: True
    vars(fake_bf16)["latent_moe_tail"] = lambda *args: expected
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.latent_moe_tail",
        fake_bf16,
    )

    assert transform.forward_with_shared(torch.empty(1), torch.empty(1)) is expected
