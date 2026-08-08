# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's hw-agnostic layer resolution.

`layer_registry._resolve` imports a layer symbol from
`vllm.model_executor.hw_agnostic.layers.<module>` when `VLLM_USE_HW_AGNOSTIC`
is set and the symbol exists, and otherwise falls back to
`vllm.model_executor.layers.<module>`. These tests pin that contract and the
logging that reports which source was used.
"""

import logging
import sys
import types

import pytest

from vllm.model_executor.models.transformers import layer_registry

HW_MODULE = "vllm.model_executor.hw_agnostic.layers.layernorm"


@pytest.fixture
def fake_hw_layernorm(monkeypatch):
    """Inject a hw-agnostic `layernorm` module exposing a sentinel `RMSNorm`."""
    module = types.ModuleType(HW_MODULE)
    module.RMSNorm = type("HwRMSNorm", (), {})
    monkeypatch.setitem(sys.modules, HW_MODULE, module)
    return module


def test_falls_back_to_vllm_when_disabled(monkeypatch, fake_hw_layernorm):
    """Disabled: the vLLM class is used even if a hw-agnostic one exists."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "0")
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    assert layer_registry._resolve("layernorm", "RMSNorm") is VllmRMSNorm


def test_uses_hw_agnostic_when_enabled(monkeypatch, fake_hw_layernorm, caplog):
    """Enabled and available: the hw-agnostic class is used and logged."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    with caplog.at_level(logging.INFO):
        resolved = layer_registry._resolve("layernorm", "RMSNorm")
    assert resolved is fake_hw_layernorm.RMSNorm
    assert "Using hw-agnostic layer: RMSNorm" in caplog.text


def test_falls_back_when_symbol_missing(monkeypatch, caplog):
    """Enabled but the symbol is not ported: fall back to vLLM and warn."""
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    # A hw-agnostic module without the requested attribute triggers fallback.
    empty = types.ModuleType(HW_MODULE)
    monkeypatch.setitem(sys.modules, HW_MODULE, empty)
    from vllm.model_executor.layers.layernorm import RMSNorm as VllmRMSNorm

    with caplog.at_level(logging.WARNING):
        resolved = layer_registry._resolve("layernorm", "RMSNorm")
    assert resolved is VllmRMSNorm
    assert "falling back to vLLM" in caplog.text


def test_act_and_mul_falls_back_for_unknown_activation(
    monkeypatch, default_vllm_config
):
    """An activation with no hw-agnostic equivalent falls back to vLLM's.

    `default_vllm_config` supplies the config context the CustomOp needs.
    """
    monkeypatch.setenv("VLLM_USE_HW_AGNOSTIC", "1")
    from vllm.model_executor.layers.activation import GeluAndMul

    assert isinstance(layer_registry.get_act_and_mul_fn("gelu"), GeluAndMul)
