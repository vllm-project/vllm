# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.parallel_state import cleanup_dist_env_and_memory


def test_cleanup_skips_accelerator_empty_cache_for_forced_cpu(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.destroy_model_parallel", lambda: None
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.destroy_distributed_environment",
        lambda: None,
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.gc.collect", lambda: None)
    monkeypatch.setattr(
        "torch.accelerator.empty_cache",
        lambda: (_ for _ in ()).throw(AssertionError("empty_cache should be skipped")),
    )
    monkeypatch.setattr("torch._C._host_emptyCache", lambda: None, raising=False)

    cleanup_dist_env_and_memory()


def test_cleanup_swallows_known_allocator_runtime_error(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_TARGET_DEVICE", "cuda")
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.destroy_model_parallel", lambda: None
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.destroy_distributed_environment",
        lambda: None,
    )
    monkeypatch.setattr("vllm.distributed.parallel_state.gc.collect", lambda: None)
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(is_cpu=lambda: False),
    )

    def raise_known_allocator_error():
        raise RuntimeError("Allocator for npu is not a DeviceAllocator")

    monkeypatch.setattr("torch.accelerator.empty_cache", raise_known_allocator_error)
    monkeypatch.setattr("torch._C._host_emptyCache", lambda: None, raising=False)

    cleanup_dist_env_and_memory()