# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup memory profiling for a pinned KV cache size.

SCALE_CACHE  teardown clears quantized scale views, with or without a kv_cache
PINNED_KV    a pinned size is checked against the profile, not trusted
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.utils as worker_utils
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker


def _layer(with_kv_cache):
    impl = SimpleNamespace(_k_scale_cache=object(), _v_scale_cache=object())
    layer = SimpleNamespace(impl=impl)
    if with_kv_cache:
        layer.kv_cache = torch.empty(4)
    return layer


@pytest.mark.parametrize("with_kv_cache", [True, False])
def test_scale_cache_is_cleared_with_or_without_a_kv_cache(with_kv_cache):
    """Scale views can live on a layer that has no `kv_cache` attribute."""
    layer = _layer(with_kv_cache)
    worker_utils.clear_layer_kv_caches([layer])

    assert layer.impl._k_scale_cache is None
    assert layer.impl._v_scale_cache is None
    if with_kv_cache:
        assert isinstance(layer.kv_cache, torch.Tensor)
        assert layer.kv_cache.numel() == 0
    else:
        assert not hasattr(layer, "kv_cache")


# PINNED_KV


@pytest.mark.parametrize(
    "profile_run_peak_gib,fits_gib",
    [
        # The attention ladder's peak is the larger one.
        (0, 4),
        # profile_run peaked 2 GiB above what the profile kept after the
        # ladder reset the counter, so that difference counts too.
        (3, 2),
    ],
)
def test_pinned_kv_cache_memory_must_fit_the_profile(profile_run_peak_gib, fits_gib):
    """10 GiB free, less 4 GiB non-KV, 1 GiB of CUDA graphs and 1 GiB of
    allocator overhead, leaves 4 GiB for the KV cache."""
    worker = SimpleNamespace(
        init_snapshot=SimpleNamespace(free_memory=10 * GiB_bytes),
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        _pinned_allocator_overhead_bytes=GiB_bytes,
        _pinned_profile_run_peak_bytes=profile_run_peak_gib * GiB_bytes,
    )
    profile_result = SimpleNamespace(
        non_kv_cache_memory=4 * GiB_bytes,
        after_profile=SimpleNamespace(torch_peak=GiB_bytes),
    )
    fits = fits_gib * GiB_bytes

    def pin(kv_cache_memory_bytes):
        return gpu_worker.Worker._apply_pinned_kv_cache_memory(
            worker, kv_cache_memory_bytes, profile_result, GiB_bytes
        )

    assert pin(fits) == fits
    assert worker.available_kv_cache_memory_bytes == fits
    with pytest.raises(ValueError, match=f"--kv-cache-memory-bytes={fits} or lower"):
        pin(fits + 1)
