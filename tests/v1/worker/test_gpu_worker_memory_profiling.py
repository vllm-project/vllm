# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.config import CUDAGraphMode
from vllm.utils.mem_utils import MemorySnapshot
from vllm.v1.worker.gpu_worker import Worker


@contextmanager
def fake_memory_profiling(
    init_snapshot: MemorySnapshot,
    weights_memory: int,
) -> Iterator[SimpleNamespace]:
    profile_result = SimpleNamespace(
        before_profile=SimpleNamespace(torch_peak=100),
        after_profile=SimpleNamespace(free_memory=90_000),
        non_torch_increase=1_000,
        weights_memory=weights_memory,
        torch_peak_increase=0,
        non_kv_cache_memory=0,
    )
    yield profile_result


def make_worker() -> Worker:
    worker = object.__new__(Worker)
    worker.device = "cuda:0"
    worker.init_snapshot = MemorySnapshot(
        torch_peak=0,
        free_memory=100_000,
        total_memory=200_000,
        device="cpu",
        auto_measure=False,
    )
    worker.requested_memory = 100_000
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None,
        gpu_memory_utilization=0.5,
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL),
    )
    worker.model_runner = MagicMock()
    worker.model_runner.model_memory_usage = 4_000
    worker.model_runner.profile_run.return_value = None
    return worker


def patch_memory_profiling_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "memory_stats",
        lambda device: {"allocated_bytes.all.peak": 300},
    )


def test_determine_available_memory_skips_cudagraph_profile_when_env_disabled(
    monkeypatch,
):
    worker = make_worker()
    worker.model_runner.profile_cudagraph_memory.side_effect = AssertionError(
        "CUDA graph memory profiling should not run when disabled"
    )

    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        False,
    )
    patch_memory_profiling_dependencies(monkeypatch)

    available_memory = Worker.determine_available_memory(worker)

    assert available_memory == 94_800
    worker.model_runner.profile_run.assert_called_once_with()
    worker.model_runner.profile_cudagraph_memory.assert_not_called()
    assert worker.cudagraph_memory_estimate == 0
    assert worker.non_torch_memory == 1_000
    assert worker.peak_activation_memory == 200
    assert worker.available_kv_cache_memory_bytes == 94_800


def test_determine_available_memory_applies_cudagraph_estimate_when_env_enabled(
    monkeypatch,
):
    worker = make_worker()
    worker.model_runner.profile_cudagraph_memory.return_value = 8_192

    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        True,
    )
    patch_memory_profiling_dependencies(monkeypatch)

    available_memory = Worker.determine_available_memory(worker)

    assert available_memory == 86_608
    worker.model_runner.profile_run.assert_called_once_with()
    worker.model_runner.profile_cudagraph_memory.assert_called_once_with()
    assert worker.cudagraph_memory_estimate == 8_192
    assert worker.non_torch_memory == 1_000
    assert worker.peak_activation_memory == 200
    assert worker.available_kv_cache_memory_bytes == 86_608


def test_determine_available_memory_skips_cudagraph_profile_when_cudagraphs_disabled(
    monkeypatch,
):
    worker = make_worker()
    worker.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    worker.model_runner.profile_cudagraph_memory.side_effect = AssertionError(
        "CUDA graph memory profiling should not run when cudagraphs are disabled"
    )

    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        True,
    )
    patch_memory_profiling_dependencies(monkeypatch)

    available_memory = Worker.determine_available_memory(worker)

    assert available_memory == 94_800
    worker.model_runner.profile_run.assert_called_once_with()
    worker.model_runner.profile_cudagraph_memory.assert_not_called()
    assert worker.cudagraph_memory_estimate == 0
    assert worker.available_kv_cache_memory_bytes == 94_800
