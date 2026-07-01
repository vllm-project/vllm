# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the DeepGEMM warmup selectors (#47169).

On Blackwell, vLLM auto-disables DeepGEMM for some model types (e.g.
``qwen3_5_text``) by setting ``quant_config.use_deep_gemm=False``, because the
E8M0 scale format degrades accuracy. Inference already honors this: the FP8
linear kernel's ``can_implement`` returns False and falls back to CUTLASS. The
warmup selectors must honor it too, otherwise warmup builds a DeepGEMM kernel
that inference never uses, and on the affected kernels that crashes at startup
with an "Unknown recipe" assertion.
"""

from unittest.mock import Mock

import torch

import vllm.model_executor.warmup.deep_gemm_warmup as warmup


class _FakeFp8LinearMethod:
    def __init__(self, *, use_deep_gemm: bool):
        self.block_quant = True
        self.use_marlin = False
        self.use_deep_gemm = use_deep_gemm


class _FakeMxfp8OnlineLinearMethod:
    pass


def _patch_linear(monkeypatch):
    monkeypatch.setattr(warmup, "LinearBase", torch.nn.Module)
    monkeypatch.setattr(warmup, "Fp8LinearMethod", _FakeFp8LinearMethod)
    monkeypatch.setattr(warmup, "Mxfp8OnlineLinearMethod", _FakeMxfp8OnlineLinearMethod)


def test_dense_selector_skips_when_auto_disabled(monkeypatch):
    _patch_linear(monkeypatch)
    extract = Mock()
    monkeypatch.setattr(warmup, "_extract_data_from_linear_base_module", extract)

    module = torch.nn.Module()
    module.quant_method = _FakeFp8LinearMethod(use_deep_gemm=False)

    assert warmup._fp8_linear_may_use_deep_gemm(module) is False
    extract.assert_not_called()


def test_dense_selector_runs_when_enabled(monkeypatch):
    _patch_linear(monkeypatch)
    monkeypatch.setattr(
        warmup, "get_mk_alignment_for_contiguous_layout", lambda: (128, 128)
    )
    monkeypatch.setattr(
        warmup,
        "_extract_data_from_linear_base_module",
        lambda _: (torch.empty((128, 128)), None, (128, 128)),
    )

    module = torch.nn.Module()
    module.quant_method = _FakeFp8LinearMethod(use_deep_gemm=True)

    assert warmup._fp8_linear_may_use_deep_gemm(module) is True


def test_moe_selector_skips_when_auto_disabled(monkeypatch):
    class _FakeQuantMethod:
        def __init__(self):
            self.quant_config = Mock(use_deep_gemm=False)
            self.get_fused_moe_quant_config = Mock()

    monkeypatch.setattr(warmup.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(warmup.envs, "VLLM_MOE_USE_DEEP_GEMM", True)
    monkeypatch.setattr(warmup, "MoERunner", torch.nn.Module)

    module = torch.nn.Module()
    module._quant_method = _FakeQuantMethod()
    module.routed_experts = Mock()

    assert warmup._fused_moe_grouped_gemm_may_use_deep_gemm(module) is False
    module._quant_method.get_fused_moe_quant_config.assert_not_called()
