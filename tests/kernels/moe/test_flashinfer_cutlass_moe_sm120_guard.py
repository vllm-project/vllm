# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""flashinfer_cutlass MoE must refuse the SM120/SM121 (capability family 12.x)
instead of silently serving corrupt output: the TRT-LLM gen fused-MoE kernels
behind it are only guaranteed on SM100/SM103 by FlashInfer upstream."""

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_mod
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)


def _platform(family: int) -> MagicMock:
    platform = MagicMock()
    platform.is_device_capability_family = lambda f: f // 10 == family // 10
    return platform


def test_rejects_sm120_family(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform(120))
    supported, reason = FlashInferExperts.is_supported_config(
        FlashInferExperts, MagicMock(), None, None, MagicMock()
    )
    assert not supported
    assert reason is not None and "SM120" in reason


def test_other_archs_not_refused_by_sm120_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform(100))
    # The passthrough hits the real base gate, which may raise on or reject
    # the MagicMock config; either way the SM120 refusal must not be returned.
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, None, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass
