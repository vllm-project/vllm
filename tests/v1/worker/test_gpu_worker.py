# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
    PYNVVIDEOCODEC_MAX_RETAINED_DECODERS,
    PYNVVIDEOCODEC_VIDEO_BACKEND,
)
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


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


# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Suspend/resume orchestration: Worker.sleep/wake_up consume the
# SleepModeBackend capability flags (vllm/device_allocator/sleep_mode_backend.py).


class _FlaggedBackend:
    """Minimal stand-in for a SleepModeBackend with configurable flags."""

    _preserves_communicators = True
    _preserves_graphs = True

    def __init__(self):
        self.calls: list[tuple] = []

    def suspend(self, level: int = 1) -> None:
        self.calls.append(("suspend", level))

    def resume(self, tags: list[str] | None = None) -> None:
        self.calls.append(("resume", tags))

    @classmethod
    def preserves_communicators(cls) -> bool:
        return cls._preserves_communicators

    @classmethod
    def preserves_graphs_with_communicators(cls) -> bool:
        return cls._preserves_graphs


class _PreservingBackend(_FlaggedBackend):
    """cumem-shaped: communicators survive suspend untouched."""


class _RebuildingBackend(_FlaggedBackend):
    """checkpoint-shaped: communicators are destroyed by suspend and must be
    rebuilt on resume; captured graphs do not survive the rebuild."""

    _preserves_communicators = False
    _preserves_graphs = False


def _sleep_worker(backend: _FlaggedBackend) -> Worker:
    worker = object.__new__(Worker)
    worker._sleep_mode_backend = backend
    worker._sleep_saved_buffers = {}
    worker.model_runner = SimpleNamespace(
        post_kv_cache_wake_up=lambda: None,
    )
    return worker


def _record_comm_hooks(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    monkeypatch.setattr(
        gpu_worker_module,
        "prepare_communicators_for_suspend",
        lambda: events.append("prepare_comms"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "reinit_communicators_after_resume",
        lambda: events.append("reinit_comms"),
    )


def _stub_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu_worker_module.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.accelerator, "get_memory_info", lambda: (0, 0)
    )


def test_sleep_prepares_communicators_before_suspend(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    _record_comm_hooks(monkeypatch, events)
    _stub_accelerator(monkeypatch)
    backend = _RebuildingBackend()
    worker = _sleep_worker(backend)

    worker.sleep(level=1)

    # Teardown must land before the backend snapshots/frees GPU state.
    assert events == ["prepare_comms"]
    assert backend.calls == [("suspend", 1)]


def test_wake_up_rebuilds_communicators_then_invalidates_graphs(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    _record_comm_hooks(monkeypatch, events)
    backend = _RebuildingBackend()
    worker = _sleep_worker(backend)

    # Graphs embedding comm handles are stale after the rebuild; the draft
    # contract fails loudly until discard+recapture exists.
    with pytest.raises(NotImplementedError, match="invalidates captured"):
        worker.wake_up()

    assert backend.calls == [("resume", None)]
    assert events == ["reinit_comms"]


def test_preserving_backend_skips_comm_orchestration(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    _record_comm_hooks(monkeypatch, events)
    _stub_accelerator(monkeypatch)
    backend = _PreservingBackend()
    worker = _sleep_worker(backend)

    worker.sleep(level=1)
    worker.wake_up()

    # cumem-shaped backends keep today's exact path: no hook is invoked.
    assert events == []
    assert backend.calls == [("suspend", 1), ("resume", None)]


def _group_with_world_size(world_size: int) -> GroupCoordinator:
    group = object.__new__(GroupCoordinator)
    group.world_size = world_size
    group.unique_name = f"test_group_ws{world_size}"
    return group


def test_group_suspend_hooks_noop_for_single_member_groups():
    group = _group_with_world_size(1)
    group.prepare_for_suspend()
    group.reinit_after_resume()


def test_group_suspend_hooks_fail_loudly_for_multi_member_groups():
    group = _group_with_world_size(2)
    with pytest.raises(NotImplementedError, match="nccl_checkpoint"):
        group.prepare_for_suspend()
    with pytest.raises(NotImplementedError, match="nccl_checkpoint"):
        group.reinit_after_resume()
