# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
)
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


def _mm_config(
    *,
    mm_ipc_gpu_memory_gb: float = 0,
    video_backend: str | None = None,
) -> SimpleNamespace:
    video_kwargs = {} if video_backend is None else {"video_backend": video_backend}
    return SimpleNamespace(
        mm_ipc_gpu_memory_gb=mm_ipc_gpu_memory_gb,
        media_io_kwargs={"video": video_kwargs} if video_kwargs else {},
    )


def _pynvvideocodec_decoder_budget(api_process_count: int = 1) -> int:
    return api_process_count * (
        PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES * PYNVVIDEOCODEC_MAX_RETAINED_DECODERS
        + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
    )


@pytest.mark.parametrize("video_backend", [None, "opencv"])
def test_reserve_mm_ipc_gpu_memory_raw_frame_budget_only(
    monkeypatch: pytest.MonkeyPatch,
    video_backend: str | None,
):
    monkeypatch.setattr(
        gpu_worker_module.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        "opencv",
    )
    worker = _worker_with_mm_config(
        _mm_config(mm_ipc_gpu_memory_gb=0.25, video_backend=video_backend)
    )

    assert worker._reserve_mm_ipc_gpu_memory(GiB_bytes) == int(0.75 * GiB_bytes)


def test_reserve_mm_ipc_gpu_memory_includes_pynvvideocodec_decoder_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gpu_worker_module.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        "opencv",
    )
    worker = _worker_with_mm_config(
        _mm_config(
            mm_ipc_gpu_memory_gb=0.25,
            video_backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
        )
    )
    available_bytes = 4 * GiB_bytes

    assert worker._reserve_mm_ipc_gpu_memory(available_bytes) == (
        available_bytes - int(0.25 * GiB_bytes) - _pynvvideocodec_decoder_budget()
    )


def test_reserve_mm_ipc_gpu_memory_uses_env_video_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gpu_worker_module.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        PYNVVIDEOCODEC_VIDEO_BACKEND,
    )
    worker = _worker_with_mm_config(_mm_config())
    available_bytes = 4 * GiB_bytes

    assert worker._reserve_mm_ipc_gpu_memory(available_bytes) == (
        available_bytes - _pynvvideocodec_decoder_budget()
    )


def test_reserve_mm_ipc_gpu_memory_scales_pynvvideocodec_budget_by_api_servers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gpu_worker_module.envs,
        "VLLM_VIDEO_LOADER_BACKEND",
        PYNVVIDEOCODEC_VIDEO_BACKEND,
    )
    worker = _worker_with_mm_config(_mm_config(), api_process_count=3)
    available_bytes = 8 * GiB_bytes

    assert worker._reserve_mm_ipc_gpu_memory(available_bytes) == (
        available_bytes - _pynvvideocodec_decoder_budget(api_process_count=3)
    )
