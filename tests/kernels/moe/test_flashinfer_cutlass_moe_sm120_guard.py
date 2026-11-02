# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""flashinfer_cutlass MoE must refuse the SM120 (cc 12.0) MXFP4xMXFP8
configuration instead of silently serving corrupt output (the TRT-LLM
gen fused-MoE kernels are only guaranteed on SM100/SM103 upstream).
SM121 (cc 12.1) and the NVFP4 expert variants are outside the refusal
scope (no corruption observed: gpt-oss-20b MXFP4xMXFP8 clean on GB10,
DSV4-Flash-0731-NVFP4 clean on RTX 6000D)."""

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_mod
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kNvfp4Dynamic,
)


def _platform(cap: tuple[int, int]) -> MagicMock:
    platform = MagicMock()
    platform.get_device_capability.return_value = cap
    return platform


def _check(monkeypatch: pytest.MonkeyPatch, cap: tuple[int, int], act) -> bool:
    """Stub the base gate to always-pass and call the real guard."""
    monkeypatch.setattr(fi_mod, "current_platform", _platform(cap))
    monkeypatch.setattr(
        mk.FusedMoEExpertsModular,
        "is_supported_config",
        staticmethod(lambda cls, *a, **k: (True, None)),
    )
    supported, _ = FlashInferExperts.is_supported_config(
        FlashInferExperts, MagicMock(), None, act, MagicMock()
    )
    return supported


def test_sm120_mxfp8_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _check(monkeypatch, (12, 0), kMxfp8Dynamic) is False


def test_sm121_mxfp8_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # gpt-oss-20b MXFP4xMXFP8 serves correctly on GB10 (cc 12.1).
    assert _check(monkeypatch, (12, 1), kMxfp8Dynamic) is True


def test_sm120_nvfp4_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # DSV4-Flash-0731-NVFP4 serves correctly on RTX 6000D (cc 12.0).
    assert _check(monkeypatch, (12, 0), kNvfp4Dynamic) is True


def test_sm100_mxfp8_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _check(monkeypatch, (10, 0), kMxfp8Dynamic) is True
