# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The minimal KV cache the CUDA graph profiler bootstraps.

SCALE_CACHE  teardown clears quantized scale views, with or without a kv_cache
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.v1.worker import gpu_model_runner as v1


def _layer(with_kv_cache):
    impl = SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object())
    layer = SimpleNamespace(impl=impl)
    if with_kv_cache:
        layer.kv_cache = torch.empty(4)
    return layer


@pytest.mark.parametrize("with_kv_cache", [True, False])
def test_scale_cache_is_cleared_with_or_without_a_kv_cache(with_kv_cache):
    """Scale views can live on a layer that has no ``kv_cache`` attribute."""
    layer = _layer(with_kv_cache)
    worker_utils.clear_layer_kv_caches([layer])

    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    if with_kv_cache:
        assert isinstance(layer.kv_cache, torch.Tensor)
        assert layer.kv_cache.numel() == 0
    else:
        assert not hasattr(layer, "kv_cache")


def test_scale_cache_shutdown_and_profiling_teardown_share_one_helper(monkeypatch):
    """V1 clears layers the same way on both paths, so neither can drift."""
    layer, seen = _layer(with_kv_cache=False), []
    runner = v1.GPUModelRunner.__new__(v1.GPUModelRunner)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=4)
    runner.compilation_config = SimpleNamespace(static_forward_context={"l": layer})
    runner.kv_caches = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(v1, "clear_layer_kv_caches", lambda ls: seen.append(list(ls)))

    runner._cleanup_profiling_kv_cache()
    assert seen == [[layer]]
    assert runner.cache_config.num_gpu_blocks is None
