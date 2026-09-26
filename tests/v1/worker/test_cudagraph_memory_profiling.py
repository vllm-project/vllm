# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The minimal KV cache the CUDA graph profiler bootstraps.

OVERRIDE     the block override is restored even when config building raises
REGISTRY     V1 still validates the KV cache spec registry
SCALE_CACHE  teardown clears quantized scale views, with or without a kv_cache
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.v1.core import kv_cache_utils
from vllm.v1.worker import gpu_model_runner as v1


@pytest.mark.parametrize("raises", [False, True])
def test_override_is_restored_even_when_config_computation_raises(monkeypatch, raises):
    """A failure here must not leak the profiling override into real sizing."""
    cfg = SimpleNamespace(num_gpu_blocks_override=7)
    runner = SimpleNamespace(vllm_config=None, cache_config=cfg, max_num_reqs=4)
    runner.compilation_config = SimpleNamespace(max_cudagraph_capture_size=8)
    seen = []

    def from_groups(vllm_config, groups, available_memory):
        seen.append(cfg.num_gpu_blocks_override)
        if raises:
            raise RuntimeError("boom")
        return "minimal"

    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda c, s: [])
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_config_from_groups", from_groups)

    if raises:
        with pytest.raises(RuntimeError, match="boom"):
            worker_utils.build_minimal_kv_cache_config(runner, object())
    else:
        assert worker_utils.build_minimal_kv_cache_config(runner, object()) == "minimal"
    # The override was in force while the config was built, and is gone after.
    assert seen == [4]
    assert cfg.num_gpu_blocks_override == 7


def test_registry_v1_validates_the_spec_before_building_the_config(monkeypatch):
    runner = v1.GPUModelRunner.__new__(v1.GPUModelRunner)
    checked: list = []
    spec, cfg, events = object(), SimpleNamespace(num_blocks=1), []
    runner.get_kv_cache_spec = lambda: spec
    runner.cache_config = SimpleNamespace(num_gpu_blocks=None)
    runner.initialize_kv_cache = lambda config, is_profiling: events.append("init")

    registry = v1.KVCacheSpecRegistry
    monkeypatch.setattr(registry, "check_kv_cache_spec_registry", checked.append)
    monkeypatch.setattr(v1, "build_minimal_kv_cache_config", lambda r, s: cfg)

    runner._init_minimal_kv_cache_for_profiling()
    assert checked == [spec] and events == ["init"]


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
