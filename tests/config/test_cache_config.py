# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.cache import CacheConfig


def test_kv_cache_memory_bytes_does_not_affect_hash():
    """kv_cache_memory_bytes only sizes the KV cache allocation (like
    gpu_memory_utilization, which is already ignored); it does not affect
    the compiled computation graph. If it leaks into the hash, setting the
    documented fast-boot knob silently invalidates the torch.compile cache
    and forces a full recompile."""
    base_hash = CacheConfig().compute_hash()
    kv_bytes_hash = CacheConfig(kv_cache_memory_bytes=1 << 30).compute_hash()
    assert base_hash == kv_bytes_hash


def test_gpu_memory_utilization_does_not_affect_hash():
    base_hash = CacheConfig().compute_hash()
    util_hash = CacheConfig(gpu_memory_utilization=0.5).compute_hash()
    assert base_hash == util_hash
