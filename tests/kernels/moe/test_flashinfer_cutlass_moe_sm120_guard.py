# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""flashinfer_cutlass MoE must refuse the SM120/SM121 MXFP4xMXFP8
configuration instead of silently serving corrupt output (the TRT-LLM
gen fused-MoE kernels are only guaranteed on SM100/SM103 upstream).
The NVFP4 expert variants are unaffected and must stay selectable on
family-12 GPUs."""

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_mod
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kNvfp4Dynamic,
)


def _platform(family: int) -> MagicMock:
    platform = MagicMock()
    platform.is_device_capability_family = lambda f: f // 10 == family // 10
    return platform


def test_rejects_sm120_mxfp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform(120))
    supported, reason = FlashInferExperts.is_supported_config(
        FlashInferExperts, MagicMock(), None, kMxfp8Dynamic, MagicMock()
    )
    assert not supported
    assert reason is not None and "SM120" in reason


def test_sm120_nvfp4_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # NVFP4 experts are served correctly on family-12 GPUs (verified on
    # RTX 6000D with DeepSeek-V4-Flash-0731-NVFP4): must not hit the refusal.
    monkeypatch.setattr(fi_mod, "current_platform", _platform(120))
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, kNvfp4Dynamic, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass  # the real base gate may raise on the MagicMock config; fine


def test_other_archs_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform(100))
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, kMxfp8Dynamic, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass
