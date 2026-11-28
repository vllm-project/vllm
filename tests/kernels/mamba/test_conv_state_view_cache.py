# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the Mamba conv-state view cache (#64572)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_view
from vllm.v1.worker.utils import clear_layer_kv_caches

pytestmark = pytest.mark.cpu_test


def _base() -> torch.Tensor:
    # (num_blocks, state_len, dim) in the default SD layout.
    return torch.randn(8, 4, 16)


def test_sd_layout_caches_transpose_and_hits():
    base = _base()
    cache, conv_state = get_conv_state_view(None, base)
    assert cache[0] is base
    assert torch.equal(conv_state, base.transpose(-1, -2))
    # Same base tensor in -> identical cache and view objects (no re-derivation).
    cache2, conv_state2 = get_conv_state_view(cache, base)
    assert cache2 is cache
    assert conv_state2 is conv_state


def test_rebuilds_when_tensor_replaced():
    base_profiling = _base()
    cache, cs_old = get_conv_state_view(None, base_profiling)
    base_real = _base()
    cache2, cs_new = get_conv_state_view(cache, base_real)
    assert cache2[0] is base_real
    assert cs_new is not cs_old


def test_view_aliases_storage():
    base = _base()
    _, conv_state = get_conv_state_view(None, base)
    # An in-place write through the cached (transposed) view must land in base,
    # so the conv kernels keep reading/writing the real state cache.
    conv_state.zero_()
    assert torch.all(base == 0)


def test_ds_layout_returns_base_unchanged():
    base = _base()
    with patch.object(mamba_utils, "is_conv_state_dim_first", return_value=True):
        _, conv_state = get_conv_state_view(None, base)
    # DS layout already stores (dim, state_len): no transpose, no copy.
    assert conv_state is base


def test_clear_layer_kv_caches_drops_conv_state_view():
    """Runner teardown must drop the cached conv-state view, or the detached
    state cache tensor stays alive through it. Exercise the real teardown
    entry point (`clear_layer_kv_caches`) rather than resetting the private
    attribute: the mixer stores the cache as `layer._conv_state_view_cache`.
    """
    base = _base()
    cache, _ = get_conv_state_view(None, base)
    assert cache is not None

    # Mirror the mixer layer: kv_cache tensor plus the conv-state view cache.
    layer = SimpleNamespace(kv_cache=base, _conv_state_view_cache=cache)
    clear_layer_kv_caches([layer])

    assert layer._conv_state_view_cache is None
    assert layer.kv_cache.numel() == 0
