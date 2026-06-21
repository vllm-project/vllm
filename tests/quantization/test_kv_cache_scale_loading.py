# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for fp8 KV-cache scale loading in
``BaseKVCacheMethod.process_weights_after_loading``.

These exercise the scalar k_scale/v_scale branch directly with a mock layer, so
they run CPU-only (no model download, no GPU kernels).

Run: pytest tests/quantization/test_kv_cache_scale_loading.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod,
    KVCacheScaleParameter,
)

# -1.0 sentinel meaning "this scale was absent from the checkpoint"
ABSENT = None


def _make_layer(k, v):
    """A minimal attention layer with the attributes the method touches."""
    layer = SimpleNamespace()
    layer.q_scale = KVCacheScaleParameter()
    layer.k_scale = KVCacheScaleParameter()
    layer.v_scale = KVCacheScaleParameter()
    layer.prob_scale = KVCacheScaleParameter()
    if k is not ABSENT:
        layer.k_scale.data.copy_(torch.tensor(k))
    if v is not ABSENT:
        layer.v_scale.data.copy_(torch.tensor(v))
    layer._q_scale = torch.tensor(0.0)
    layer._k_scale = torch.tensor(0.0)
    layer._v_scale = torch.tensor(0.0)
    layer._prob_scale = torch.tensor(0.0)
    layer.kv_cache_dtype = "fp8"
    layer.calculate_kv_scales = False
    return layer


def _process(layer):
    BaseKVCacheMethod(MagicMock()).process_weights_after_loading(layer)


@pytest.mark.parametrize(
    "k,v,expected_k,expected_v",
    [
        (0.05, 0.08, 0.05, 0.08),  # both present -> kept separate
        (ABSENT, ABSENT, 1.0, 1.0),  # neither present -> default 1.0
        (0.05, ABSENT, 0.05, 0.05),  # k-only -> duplicated to v
        (ABSENT, 0.05, 0.05, 0.05),  # v-only -> duplicated to k (regression)
    ],
)
def test_single_or_double_scale_loading(k, v, expected_k, expected_v):
    layer = _make_layer(k, v)
    _process(layer)
    assert layer._k_scale_float == pytest.approx(expected_k)
    assert layer._v_scale_float == pytest.approx(expected_v)


def test_v_only_checkpoint_does_not_crash():
    """Regression: a checkpoint that ships only v_scale (k_scale left at the
    -1.0 sentinel) used to hit ``assert layer.k_scale > 0.0`` and abort weight
    loading. It must now load, duplicating the lone valid scale to both."""
    layer = _make_layer(ABSENT, 0.05)
    _process(layer)  # must not raise
    assert layer._k_scale_float == pytest.approx(0.05)
    assert layer._v_scale_float == pytest.approx(0.05)


def test_corrupt_scales_still_rejected():
    """Both scales non-positive (and not the both-negative default path) is a
    genuinely invalid checkpoint and must still be rejected."""
    layer = _make_layer(0.0, 0.0)
    with pytest.raises(AssertionError):
        _process(layer)
