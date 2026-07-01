# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.multimodal.gpu_ipc_memory import MultiModalGPUMemoryReservation
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker


def _worker_with_mm_config(
    mm_config: SimpleNamespace,
    *,
    api_process_count: int = 1,
) -> Worker:
    worker = object.__new__(Worker)
    worker.model_config = SimpleNamespace(multimodal_config=mm_config)
    worker.parallel_config = SimpleNamespace(_api_process_count=api_process_count)
    return worker


def _mm_config() -> SimpleNamespace:
    return SimpleNamespace()


def test_reserve_mm_ipc_gpu_memory_subtracts_reservation(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _worker_with_mm_config(_mm_config(), api_process_count=2)
    reservation = MultiModalGPUMemoryReservation(
        raw_frame_bytes=10,
        decoder_bytes=20,
        per_server_decoder_bytes=10,
        api_process_count=2,
    )

    monkeypatch.setattr(
        gpu_worker_module,
        "get_mm_gpu_ipc_memory_reservation",
        lambda *_: reservation,
    )

    assert worker._reserve_mm_ipc_gpu_memory(100) == 70


def test_reserve_mm_ipc_gpu_memory_raises_when_reservation_exceeds_available(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _worker_with_mm_config(_mm_config())
    reservation = MultiModalGPUMemoryReservation(
        raw_frame_bytes=60,
        decoder_bytes=50,
        per_server_decoder_bytes=50,
        api_process_count=1,
    )

    monkeypatch.setattr(
        gpu_worker_module,
        "get_mm_gpu_ipc_memory_reservation",
        lambda *_: reservation,
    )

    with pytest.raises(ValueError, match="frontend multimodal GPU decoding"):
        worker._reserve_mm_ipc_gpu_memory(100)


def test_determine_available_memory_preserves_explicit_kv_cache_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = _worker_with_mm_config(_mm_config())
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=GiB_bytes)
    worker.init_snapshot = SimpleNamespace(free_memory=4 * GiB_bytes)

    profile_run_called = False

    def profile_run():
        nonlocal profile_run_called
        profile_run_called = True

    worker.model_runner = SimpleNamespace(profile_run=profile_run)
    monkeypatch.setattr(
        worker,
        "_reserve_mm_ipc_gpu_memory",
        lambda _: pytest.fail("explicit KV cache memory must not be reduced"),
    )

    assert worker.determine_available_memory() == GiB_bytes
    assert profile_run_called
