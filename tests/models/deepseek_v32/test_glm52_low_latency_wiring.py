# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring checks for the GLM-5.2 low-latency GEMM plan (CPU-only)."""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as glm52_gemm


class _FakeLinear(nn.Module):
    """Minimal stand-in for a vLLM linear whose quant_method can be swapped."""

    def __init__(self, n: int, k: int, quant_method: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n, k, dtype=torch.bfloat16, device="meta")
        )
        self.quant_method = quant_method
        self.prefix = "fixture.proj"


def _stub_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glm52_gemm, "LinearBase", _FakeLinear)
    monkeypatch.setattr(
        glm52_gemm.shape_dynamic_skinny_gemm,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        glm52_gemm.shape_dynamic_skinny_gemm,
        "request_warmup_configs",
        lambda *args, **kwargs: None,
    )


def _sm10x_root() -> nn.Module:
    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    gate_up = glm52_gemm.GLM52_DENSE_GATE_UP_PROJECTION
    root = nn.Module()
    root.attn = nn.Module()
    root.attn.fused_qkv_a_proj = _FakeLinear(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.mlp = nn.Module()
    root.mlp.gate_up_proj = _FakeLinear(
        gate_up.n, gate_up.k, glm52_gemm.UnquantizedLinearMethod()
    )
    return root


def test_enable_logs_and_installs_on_sm10x(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(glm52_gemm, "_is_sm10x", lambda: True)
    _stub_warmup(monkeypatch)
    root = _sm10x_root()

    with caplog.at_level(logging.INFO, logger="vllm"):
        glm52_gemm.enable_glm52_low_latency_gemm(root, torch.bfloat16)

    assert any(
        "GLM-5.2 low-latency GEMM plan is ENABLED" in record.message
        for record in caplog.records
    ), caplog.text
    assert isinstance(
        root.attn.fused_qkv_a_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert isinstance(
        root.mlp.gate_up_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )


def test_env_var_disables_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv(glm52_gemm.VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM, "1")
    monkeypatch.setattr(glm52_gemm, "_is_sm10x", lambda: True)
    _stub_warmup(monkeypatch)
    root = _sm10x_root()

    with caplog.at_level(logging.INFO, logger="vllm"):
        glm52_gemm.enable_glm52_low_latency_gemm(root, torch.bfloat16)

    assert any(
        "GLM-5.2 low-latency GEMM plan is DISABLED (env override)" in record.message
        for record in caplog.records
    ), caplog.text
    assert (
        type(root.attn.fused_qkv_a_proj.quant_method)
        is glm52_gemm.UnquantizedLinearMethod
    )
    assert (
        type(root.mlp.gate_up_proj.quant_method) is glm52_gemm.UnquantizedLinearMethod
    )


def test_build_glm52_plan_honors_env_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MTP eh_proj path (plain nn.Linear) must also respect the switch."""
    monkeypatch.setenv(glm52_gemm.VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM, "1")
    monkeypatch.setattr(glm52_gemm, "_is_sm10x", lambda: True)
    _stub_warmup(monkeypatch)
    eh = glm52_gemm.GLM52_EH_PROJECTION
    weight = torch.empty(eh.n, eh.k, dtype=torch.bfloat16, device="meta")

    assert glm52_gemm.build_glm52_plan(weight, torch.bfloat16) is None


def _force_platform_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force all platform capability gates on so tests are arch-independent."""
    from vllm.model_executor.models import deepseek_v2 as dv2_mod

    monkeypatch.setattr(dv2_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dv2_mod.current_platform,
        "is_device_capability",
        lambda maj: True,
    )
    monkeypatch.setattr(
        dv2_mod.current_platform,
        "is_device_capability_family",
        lambda fam: True,
    )


def test_glm52_fused_qkv_a_shape_passes_min_latency_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: GLM-5.2 fused-QKV-A must route through the custom op."""
    from vllm.model_executor.models.deepseek_v2 import (
        _can_use_min_latency_fused_qkv_a_gemm,
    )

    _force_platform_gates(monkeypatch)
    monkeypatch.delenv("VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM", raising=False)

    # GLM-5.2 fused-QKV-A weight (N=2624, K=6144) must qualify.
    glm_weight = torch.empty(2624, 6144, dtype=torch.bfloat16, device="meta")
    assert _can_use_min_latency_fused_qkv_a_gemm(glm_weight)

    # DeepSeek-V3 fused-QKV-A weight (N=2112, K=7168) must still qualify.
    dsv3_weight = torch.empty(2112, 7168, dtype=torch.bfloat16, device="meta")
    assert _can_use_min_latency_fused_qkv_a_gemm(dsv3_weight)

    # A random shape must NOT qualify.
    other = torch.empty(2761, 6455, dtype=torch.bfloat16, device="meta")
    assert not _can_use_min_latency_fused_qkv_a_gemm(other)

    # Non-BF16 must not qualify even for the GLM-5.2 shape.
    fp16 = torch.empty(2624, 6144, dtype=torch.float16, device="meta")
    assert not _can_use_min_latency_fused_qkv_a_gemm(fp16)


def test_glm52_fused_qkv_a_respects_disable_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature toggle must also disable the fused-QKV-A custom-op path."""
    from vllm.model_executor.models.deepseek_v2 import (
        _can_use_min_latency_fused_qkv_a_gemm,
    )

    _force_platform_gates(monkeypatch)

    glm_weight = torch.empty(2624, 6144, dtype=torch.bfloat16, device="meta")
    dsv3_weight = torch.empty(2112, 7168, dtype=torch.bfloat16, device="meta")

    # Toggle OFF (disabled): GLM-5.2 QKV-A must NOT qualify.
    monkeypatch.setenv("VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM", "1")
    assert not _can_use_min_latency_fused_qkv_a_gemm(glm_weight)

    # DeepSeek-V3 must remain unaffected by the GLM-5.2 toggle.
    assert _can_use_min_latency_fused_qkv_a_gemm(dsv3_weight)

    # Toggle ON (enabled): GLM-5.2 QKV-A must qualify.
    monkeypatch.delenv("VLLM_DISABLE_GLM52_LOW_LATENCY_GEMM", raising=False)
    assert _can_use_min_latency_fused_qkv_a_gemm(glm_weight)
