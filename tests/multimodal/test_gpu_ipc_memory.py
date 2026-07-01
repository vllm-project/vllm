# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from types import SimpleNamespace

import pytest

import vllm.multimodal.gpu_ipc_memory as gpu_ipc_memory
from vllm.multimodal.gpu_ipc_memory import (
    MultiModalGPUMemoryPool,
    get_mm_gpu_ipc_memory_reservation,
    get_mm_gpu_ipc_pool,
    maybe_init_mm_gpu_ipc_pool,
    set_mm_gpu_ipc_pool,
)
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
)
from vllm.utils.mem_constants import GiB_bytes


def _mm_config(
    *,
    mm_ipc_gpu_memory_gb: float = 0,
    video_backend: str | None = None,
    codec_backend: str | None = None,
) -> SimpleNamespace:
    video_kwargs = {}
    if video_backend is not None:
        video_kwargs["video_backend"] = video_backend
    if codec_backend is not None:
        video_kwargs["backend"] = codec_backend
    return SimpleNamespace(
        mm_ipc_gpu_memory_gb=mm_ipc_gpu_memory_gb,
        media_io_kwargs={"video": video_kwargs} if video_kwargs else {},
    )


def _pynvvideocodec_decoder_budget(api_process_count: int = 1) -> int:
    return api_process_count * (
        PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES * PYNVVIDEOCODEC_MAX_RETAINED_DECODERS
        + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
    )


def test_acquire_release_accounting():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    assert pool.available_bytes == 100

    lease = pool.acquire(40)
    assert pool.available_bytes == 60

    lease.release()
    assert pool.available_bytes == 100


def test_acquire_too_large_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(ValueError):
        pool.acquire(101)
    # Nothing should have been reserved.
    assert pool.available_bytes == 100


def test_negative_acquire_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(ValueError):
        pool.acquire(-1)


def test_double_release_is_noop():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    lease = pool.acquire(50)
    lease.release()
    assert pool.available_bytes == 100
    # Releasing again must not inflate the pool past its capacity.
    lease.release()
    assert pool.available_bytes == 100


def test_context_manager_releases_on_exception():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(RuntimeError), pool.acquire(50):
        assert pool.available_bytes == 50
        raise RuntimeError("boom")
    assert pool.available_bytes == 100


def test_acquire_blocks_until_release():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    first = pool.acquire(80)

    acquired = threading.Event()

    def waiter():
        # Needs 50 bytes but only 20 are free; must block until `first`
        # is released.
        with pool.acquire(50):
            acquired.set()

    t = threading.Thread(target=waiter)
    t.start()

    # The waiter cannot proceed yet.
    assert not acquired.wait(timeout=0.2)

    # Releasing the first lease frees enough budget to unblock the waiter.
    first.release()
    assert acquired.wait(timeout=2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert pool.available_bytes == 100


def test_concurrent_acquires_serialize():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    # Each task needs 60 bytes, so only one can hold the budget at a time.
    in_section = []
    max_concurrent = 0
    lock = threading.Lock()

    def task():
        nonlocal max_concurrent
        with pool.acquire(60):
            with lock:
                in_section.append(1)
                max_concurrent = max(max_concurrent, len(in_section))
            time.sleep(0.05)
            with lock:
                in_section.pop()

    threads = [threading.Thread(target=task) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert max_concurrent == 1
    assert pool.available_bytes == 100


def test_zero_total_bytes_rejected():
    with pytest.raises(ValueError):
        MultiModalGPUMemoryPool(total_bytes=0)


def test_global_pool_accessor():
    try:
        assert maybe_init_mm_gpu_ipc_pool(0) is None
        assert get_mm_gpu_ipc_pool() is None

        pool = maybe_init_mm_gpu_ipc_pool(2)
        assert pool is not None
        assert get_mm_gpu_ipc_pool() is pool
        assert pool.total_bytes == 2 * GiB_bytes
    finally:
        set_mm_gpu_ipc_pool(None)


def test_global_pool_splits_budget_across_api_processes():
    try:
        pool = maybe_init_mm_gpu_ipc_pool(2, api_process_count=4)
        assert pool is not None
        assert get_mm_gpu_ipc_pool() is pool
        assert pool.total_bytes == GiB_bytes // 2
    finally:
        set_mm_gpu_ipc_pool(None)


def test_global_pool_rejects_invalid_api_process_count():
    with pytest.raises(ValueError):
        maybe_init_mm_gpu_ipc_pool(2, api_process_count=0)


@pytest.mark.parametrize("video_backend", [None, "opencv"])
def test_mm_gpu_ipc_reservation_raw_frame_budget_only(
    monkeypatch: pytest.MonkeyPatch,
    video_backend: str | None,
):
    monkeypatch.setattr(
        gpu_ipc_memory.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        "opencv",
    )

    reservation = get_mm_gpu_ipc_memory_reservation(
        _mm_config(mm_ipc_gpu_memory_gb=0.25, video_backend=video_backend)
    )

    assert reservation.raw_frame_bytes == int(0.25 * GiB_bytes)
    assert reservation.decoder_bytes == 0
    assert reservation.total_bytes == int(0.25 * GiB_bytes)


@pytest.mark.parametrize(
    ("video_backend", "codec_backend"),
    [
        (PYNVVIDEOCODEC_VIDEO_BACKEND, None),
        (None, PYNVVIDEOCODEC_VIDEO_BACKEND),
    ],
)
def test_mm_gpu_ipc_reservation_includes_pynvvideocodec_decoder_budget(
    monkeypatch: pytest.MonkeyPatch,
    video_backend: str | None,
    codec_backend: str | None,
):
    monkeypatch.setattr(
        gpu_ipc_memory.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        "opencv",
    )

    reservation = get_mm_gpu_ipc_memory_reservation(
        _mm_config(
            mm_ipc_gpu_memory_gb=0.25,
            video_backend=video_backend,
            codec_backend=codec_backend,
        )
    )

    assert reservation.raw_frame_bytes == int(0.25 * GiB_bytes)
    assert reservation.decoder_bytes == _pynvvideocodec_decoder_budget()
    assert reservation.total_bytes == (
        int(0.25 * GiB_bytes) + _pynvvideocodec_decoder_budget()
    )


def test_mm_gpu_ipc_reservation_uses_env_video_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gpu_ipc_memory.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        PYNVVIDEOCODEC_VIDEO_BACKEND,
    )

    reservation = get_mm_gpu_ipc_memory_reservation(_mm_config())

    assert reservation.decoder_bytes == _pynvvideocodec_decoder_budget()
    assert reservation.total_bytes == _pynvvideocodec_decoder_budget()


def test_mm_gpu_ipc_reservation_scales_pynvvideocodec_budget_by_api_servers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gpu_ipc_memory.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        PYNVVIDEOCODEC_VIDEO_BACKEND,
    )

    reservation = get_mm_gpu_ipc_memory_reservation(_mm_config(), api_process_count=3)

    assert reservation.api_process_count == 3
    assert reservation.decoder_bytes == _pynvvideocodec_decoder_budget(
        api_process_count=3
    )
    assert reservation.total_bytes == _pynvvideocodec_decoder_budget(
        api_process_count=3
    )
