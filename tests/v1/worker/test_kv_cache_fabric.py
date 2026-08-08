# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the opt-in fabric KV cache allocator.

CI-runnable without NVLink/fabric hardware: exercises the disabled path and the
non-CUDA fallback, and (when a GPU is present) the capability-probe fallback.
"""
import torch

from vllm.v1.worker.gpu.kv_cache_fabric import maybe_fabric_kv_tensor


def test_disabled_returns_zeros(monkeypatch):
    import vllm.envs as envs
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_FABRIC", False, raising=False)
    t = maybe_fabric_kv_tensor(1024, torch.device("cpu"))
    assert t.dtype == torch.int8 and t.numel() == 1024
    assert bool((t == 0).all())


def test_non_cuda_device_falls_back(monkeypatch):
    import vllm.envs as envs
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_FABRIC", True, raising=False)
    # CPU device must never attempt a fabric allocation.
    t = maybe_fabric_kv_tensor(2048, torch.device("cpu"))
    assert t.dtype == torch.int8 and t.numel() == 2048


def test_enabled_on_unsupported_gpu_falls_back(monkeypatch):
    """When enabled but the platform can't export fabric handles, must fall back
    to a valid zeroed CUDA tensor rather than raise."""
    if not torch.cuda.is_available():
        return
    import vllm.envs as envs
    import vllm.v1.worker.gpu.kv_cache_fabric as m
    monkeypatch.setattr(envs, "VLLM_KV_CACHE_FABRIC", True, raising=False)
    monkeypatch.setattr(m, "_fabric_supported", lambda dev_id: False)
    t = maybe_fabric_kv_tensor(4096, torch.device("cuda:0"))
    assert t.is_cuda and t.dtype == torch.int8 and t.numel() == 4096
    assert bool((t == 0).all())
