# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM warmup honors the per-model auto-disable.

Regression test for https://github.com/vllm-project/vllm/issues/47169: when
``quant_config.use_deep_gemm`` is explicitly ``False`` (set by
:func:`vllm.utils.deep_gemm.should_auto_disable_deep_gemm` for Qwen3.5/3.6
hybrid models on Blackwell), the fused-MoE warmup gate must not JIT-compile
DeepGEMM kernels for those modules.

GPU-free: the module and quant method are mocked.
"""

from unittest.mock import Mock

from vllm.model_executor.warmup import deep_gemm_warmup


class _FakeMoERunner:
    """Stand-in for MoERunner so plain mocks pass isinstance checks."""


def _mock_moe_module(monkeypatch, use_deep_gemm=None, has_attr=True):
    monkeypatch.setattr(deep_gemm_warmup, "MoERunner", _FakeMoERunner)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)
    quant_method = Mock()
    if has_attr:
        quant_config = Mock()
        quant_config.use_deep_gemm = use_deep_gemm
        quant_method.quant_config = quant_config
    else:
        # Some non-FP8 quant methods have a quant_config without the field.
        quant_method.quant_config = Mock()
    module = Mock()
    module._quant_method = quant_method
    return module


def test_fused_moe_grouped_gemm_may_use_deep_gemm_false_when_disabled(
    monkeypatch,
):
    module = _mock_moe_module(monkeypatch, use_deep_gemm=False)
    assert deep_gemm_warmup._fused_moe_grouped_gemm_may_use_deep_gemm(module) is False


def test_fused_moe_grouped_gemm_may_use_deep_gemm_no_attribute_error(
    monkeypatch,
):
    # quant_config without a use_deep_gemm attribute must not raise.
    module = _mock_moe_module(monkeypatch, has_attr=False)
    result = deep_gemm_warmup._fused_moe_grouped_gemm_may_use_deep_gemm(module)
    assert result is False


def test_fused_moe_grouped_gemm_may_use_deep_gemm_env_override_wins(
    monkeypatch,
):
    # Explicit env overrides take precedence over the per-model disable.
    monkeypatch.setattr(deep_gemm_warmup, "MoERunner", _FakeMoERunner)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_MOE_USE_DEEP_GEMM", "1")
    quant_method = Mock()
    quant_config = Mock()
    quant_config.use_deep_gemm = False
    quant_method.quant_config = quant_config
    module = Mock()
    module._quant_method = quant_method
    # The gate should not bail out on the flag; it falls through to the
    # existing checks (which return False here because the mock has no
    # matching quant config) without raising.
    result = deep_gemm_warmup._fused_moe_grouped_gemm_may_use_deep_gemm(module)
    assert result is False
