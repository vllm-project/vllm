# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import vllm.v1.worker.gpu_worker as gpu_worker_module
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


def _plan_platform(name="NVIDIA H100 PCIe", total_mem=80 * GiB_bytes, cap=(9, 0)):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: total_mem,
        get_device_capability=lambda device_id=0: cap,
    )


def _plan_worker(
    config_hash="abc123",
    rank=0,
    world_size=1,
    free_memory=78 * GiB_bytes,
    kv_cache_memory_bytes=None,
):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=rank,
        parallel_config=SimpleNamespace(world_size=world_size),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan and isolate it under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield tmp_path


def _plan_fingerprint(config_hash="abc123", rank=0, world_size=1, **platform_kwargs):
    vllm_config = SimpleNamespace(compute_hash=lambda: config_hash)
    platform = _plan_platform(**platform_kwargs)
    with patch.object(startup_plan, "current_platform", platform):
        return startup_plan.compute_plan_fingerprint(vllm_config, rank, world_size)


def test_startup_plan_fingerprint_stable_and_sensitive():
    base = _plan_fingerprint()
    assert base == _plan_fingerprint(), "same inputs must give the same fingerprint"
    assert base != _plan_fingerprint(config_hash="different")
    assert base != _plan_fingerprint(name="NVIDIA A100-SXM4-80GB")
    assert base != _plan_fingerprint(total_mem=40 * GiB_bytes)
    assert base != _plan_fingerprint(rank=1, world_size=2)
    # The vLLM version is an explicit factor, independent of compute_hash.
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != _plan_fingerprint()


def test_startup_plan_save_load_round_trip(plan_env):
    worker = _plan_worker()
    maybe_save_startup_plan(worker, 50 * GiB_bytes)

    # The plan lands under {VLLM_CACHE_ROOT}/startup_plan/.
    plan_files = list((plan_env / "startup_plan").glob("startup_plan_*.json"))
    assert len(plan_files) == 1

    fp = startup_plan.compute_plan_fingerprint(worker.vllm_config, 0, 1)
    plan = startup_plan._load_plan(fp)
    assert plan is not None
    assert plan["kv_cache_memory_bytes"] == 50 * GiB_bytes
    assert plan["free_memory_baseline"] == 78 * GiB_bytes


def test_startup_plan_load_missing_or_corrupt_returns_none(plan_env):
    assert startup_plan._load_plan("0000000000000000") is None

    plan_dir = plan_env / "startup_plan"
    plan_dir.mkdir()
    (plan_dir / "startup_plan_1111111111111111.json").write_text("{not json")
    assert startup_plan._load_plan("1111111111111111") is None

    (plan_dir / "startup_plan_2222222222222222.json").write_text(
        json.dumps({"schema": 1, "fingerprint": "mismatch"})
    )
    assert startup_plan._load_plan("2222222222222222") is None


def test_startup_plan_free_memory_gate():
    plan = {
        "kv_cache_memory_bytes": 50 * GiB_bytes,
        "free_memory_baseline": 78 * GiB_bytes,
    }
    gate = startup_plan._applicable_kv_cache_memory_bytes
    # Enough free memory: apply.
    assert gate(plan, 78 * GiB_bytes) == 50 * GiB_bytes
    assert gate(plan, 79 * GiB_bytes) == 50 * GiB_bytes
    # Less free memory than when measured: refuse, re-profile.
    assert gate(plan, 77 * GiB_bytes) is None


def test_startup_plan_gate_rejects_malformed_plans():
    gate = startup_plan._applicable_kv_cache_memory_bytes
    assert gate({}, 80 * GiB_bytes) is None
    assert (
        gate({"kv_cache_memory_bytes": -1, "free_memory_baseline": 1}, 80 * GiB_bytes)
        is None
    )
    assert (
        gate({"kv_cache_memory_bytes": "50", "free_memory_baseline": 1}, 80 * GiB_bytes)
        is None
    )


def test_startup_plan_maybe_apply_end_to_end(plan_env):
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    # Same fingerprint + enough memory: applied.
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    # Same fingerprint, less memory: refused.
    worker = _plan_worker(free_memory=60 * GiB_bytes)
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None

    # Different config: fingerprint miss.
    worker = _plan_worker(config_hash="zzz999")
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    worker = _plan_worker(kv_cache_memory_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


def test_startup_plan_disabled_is_a_noop(plan_env, monkeypatch: pytest.MonkeyPatch):
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "0")

    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None

    maybe_save_startup_plan(_plan_worker(config_hash="fresh"), 50 * GiB_bytes)
    fp = startup_plan.compute_plan_fingerprint(
        _plan_worker(config_hash="fresh").vllm_config, 0, 1
    )
    assert startup_plan._load_plan(fp) is None
