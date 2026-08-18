# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for MoE FP8 backend auto-disable (#47169).

On Blackwell, vLLM auto-disables DeepGEMM for some model types (e.g.
``qwen3_5_moe_text``) because the E8M0 scale format degrades accuracy. The FP8
linear path already honors this at ``can_implement``. The MoE backend selector
must honor it too, so selection never lands on DeepGEMM for those models.
"""

from types import SimpleNamespace

import vllm.model_executor.layers.fused_moe.oracle.fp8 as oracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend


def _patch_model_type(monkeypatch, model_type):
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type)
        )
    )
    monkeypatch.setattr(oracle, "get_current_vllm_config", lambda: cfg)


def test_moe_drops_deep_gemm_when_auto_disabled(monkeypatch):
    _patch_model_type(monkeypatch, "qwen3_5_moe_text")
    monkeypatch.setattr(oracle, "should_auto_disable_deep_gemm", lambda mt: True)

    backends = [
        Fp8MoeBackend.DEEPGEMM,
        Fp8MoeBackend.TRITON,
        Fp8MoeBackend.BATCHED_DEEPGEMM,
    ]
    oracle._remove_deep_gemm_if_auto_disabled(backends)

    assert Fp8MoeBackend.DEEPGEMM not in backends
    assert Fp8MoeBackend.BATCHED_DEEPGEMM not in backends
    assert Fp8MoeBackend.TRITON in backends


def test_moe_keeps_deep_gemm_when_not_auto_disabled(monkeypatch):
    _patch_model_type(monkeypatch, "llama")
    monkeypatch.setattr(oracle, "should_auto_disable_deep_gemm", lambda mt: False)

    backends = [Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.TRITON]
    oracle._remove_deep_gemm_if_auto_disabled(backends)

    assert Fp8MoeBackend.DEEPGEMM in backends
    assert Fp8MoeBackend.TRITON in backends
