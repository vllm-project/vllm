# SPDX-License-Identifier: Apache-2.0
"""Regression test: DFlashGroupedConv.kernel_projection must stay BF16.

DFlash2 checkpoints store kernel_projection in BF16 (modules_to_not_convert).
If it inherits the FP8 quant_config, the BF16 weight is misread as e4m3 and the
draft forward becomes NaN (acceptance rate 0%). It must be built with
quant_config=None, like CandidateSelector.hidden_projection.
"""
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models import qwen3_dflash2 as mod


class _DummyQuantConfig:
    """Stand-in for an FP8 QuantizationConfig (non-None)."""

    def get_quant_method(self, layer, prefix):  # pragma: no cover - not called
        raise AssertionError("kernel_projection must not be quantized")


def _build_conv(quant_config):
    captured = {}

    real_linear = mod.ReplicatedLinear

    def spy_linear(*args, **kwargs):
        captured["quant_config"] = kwargs.get("quant_config")
        # Avoid touching distributed/TP state: return a lightweight stub.
        return torch.nn.Identity()

    with patch.object(mod, "ReplicatedLinear", spy_linear):
        mod.DFlashGroupedConv(
            hidden_size=5120,
            taps=2,
            group_size=16,
            block_size=9,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.attention_conv",
            quant_config=quant_config,
        )
    return captured


def test_kernel_projection_is_bf16_under_fp8():
    captured = _build_conv(_DummyQuantConfig())
    assert captured["quant_config"] is None, (
        "kernel_projection must be forced to BF16 (quant_config=None) even when "
        "the draft checkpoint is FP8; otherwise the BF16 weight is misread as "
        "e4m3 and the draft forward produces NaN."
    )


def test_kernel_projection_bf16_without_quant():
    captured = _build_conv(None)
    assert captured["quant_config"] is None