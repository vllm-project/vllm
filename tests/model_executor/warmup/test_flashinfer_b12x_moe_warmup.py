# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the FlashInfer b12x MoE warmup helpers.

These cover the shape/ordering logic and the model-detection walk; the actual
kernel warmup itself is SM120-only and must be validated on hardware.
"""

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.warmup.flashinfer_b12x_moe_warmup import (
    _model_uses_b12x_moe,
    _select_warmup_token_sizes,
)


def test_select_token_sizes_is_descending_and_max_first():
    sizes = _select_warmup_token_sizes(8192, [512, 1024], experts_per_token=8)
    # Largest first so the shared workspace is grown before the micro-path
    # kernels compile against it.
    assert sizes[0] == 8192
    assert sizes == sorted(sizes, reverse=True)
    # No duplicates.
    assert len(sizes) == len(set(sizes))
    # Micro-path candidates and cudagraph sizes are both included.
    assert {1, 2, 64, 512, 1024}.issubset(sizes)
    # The static-region ceiling (640 // experts_per_token) is warmed too, so
    # the static workspace reaches its true high-water mark -- not just the
    # dynamic-path max_tokens.
    assert 640 // 8 in sizes


def test_select_token_sizes_filters_and_dedups():
    # Candidates and cudagraph sizes above max_tokens are dropped; the max
    # itself is always present exactly once.
    sizes = _select_warmup_token_sizes(64, [64, 128, 256])
    assert max(sizes) == 64
    assert all(1 <= s <= 64 for s in sizes)
    assert sizes.count(64) == 1


def test_select_token_sizes_empty_when_no_budget():
    assert _select_warmup_token_sizes(0, []) == []
    assert _select_warmup_token_sizes(-5, [1, 2]) == []


class _Leaf(torch.nn.Module):
    """A leaf module carrying an arbitrary ``quant_method`` attribute."""

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method


def test_model_detection_negative():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), _Leaf(None))
    assert _model_uses_b12x_moe(model) is False
    # A quant_method without a b12x moe_kernel must not match.
    other = _Leaf(SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=object())))
    assert _model_uses_b12x_moe(torch.nn.Sequential(other)) is False


def test_model_detection_positive():
    # Bypass __init__ (which needs full MoE configs); isinstance is all the
    # detector checks.
    experts = object.__new__(FlashInferB12xExperts)
    quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=experts))
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), _Leaf(quant_method))
    assert _model_uses_b12x_moe(model) is True
