# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import maybe_rocm_profiling_fallback
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


def test_load_model_preserves_compiled_graphs_at_runtime(monkeypatch):
    """Profiling must use serving's thread count to keep Dynamo guards valid."""
    from torch._dynamo.testing import CompileCounter

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(gpu_worker, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        gpu_worker, "set_current_vllm_config", lambda config: nullcontext()
    )
    loading_threads = []
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(weight_transfer_config=None),
        model_runner=SimpleNamespace(
            load_model=lambda **kwargs: loading_threads.append(torch.get_num_threads())
        ),
        _maybe_get_memory_pool_context=lambda **kwargs: nullcontext(),
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        gpu_worker.Worker.load_model(worker)
        assert loading_threads == [2]

        counter = CompileCounter()
        compiled = torch.compile(lambda x: x + 1, backend=counter, fullgraph=True)
        x = torch.ones(2)
        compiled(x)
        gpu_worker.set_torch_threads_for_runtime()
        torch.testing.assert_close(compiled(x), x + 1)
        assert counter.frame_count == 1
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not torch.accelerator.is_available()
    or torch.cuda.memory.get_allocator_backend() != "native",
    reason="needs the native CUDA or ROCm allocator",
)
# A fresh allocator: blocks cached by earlier tests could serve the large buffer.
@create_new_process_for_each_test("spawn")
def test_scoped_max_split_keeps_freed_large_blocks_releasable():
    """A small allocation made after a large buffer is freed must not pin the
    buffer's segment: the profiling run (determine_available_memory) grows
    workspaces this way, and a pinned segment survives empty_cache() and is
    counted as consumed memory."""
    large = 512 * 1024 * 1024
    small = 2 * 1024 * 1024  # large pool, so it is served by splitting cached blocks

    def reserved_while_small_is_live(scope) -> int:
        torch.accelerator.empty_cache()
        with scope:
            buf = torch.empty(large, dtype=torch.uint8, device="cuda")
            del buf
            tensor = torch.empty(small, dtype=torch.uint8, device="cuda")
            torch.accelerator.empty_cache()
            reserved = torch.accelerator.memory_reserved()
            del tensor
        torch.accelerator.empty_cache()
        return reserved

    baseline = torch.accelerator.memory_reserved()
    # Without the limit the small tensor is split off the freed block and pins it.
    assert reserved_while_small_is_live(nullcontext()) - baseline >= large
    scoped = gpu_worker.Worker._scoped_allocator_max_split(
        SimpleNamespace(), max_split_size_mb=20
    )
    assert reserved_while_small_is_live(scoped) - baseline < large


@pytest.mark.skipif(
    not current_platform.is_cuda_alike()
    or not torch.accelerator.is_available()
    or torch.cuda.memory.get_allocator_backend() != "native",
    reason="needs the native CUDA or ROCm allocator",
)
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("snapshot_fallback", [False, True])
@pytest.mark.parametrize("suffix", ["", ", "])
def test_scoped_max_split_preserves_allocator_settings(
    monkeypatch, fail, snapshot_fallback, suffix
):
    """Preserve allocator settings across successful and failed profiling runs."""
    if snapshot_fallback:
        monkeypatch.delattr(
            torch._C, "_accelerator_getAllocatorSettings", raising=False
        )

    def settings():
        return torch.cuda.memory._snapshot()["allocator_settings"]

    original = settings()["PYTORCH_CUDA_ALLOC_CONF"]
    configured = (
        "max_split_size_mb:128,garbage_collection_threshold:0.8,"
        "roundup_power2_divisions:[256:1,512:2,>:4],max_non_split_rounding_mb:32"
    )
    try:
        torch._C._accelerator_setAllocatorSettings(configured + suffix)
        before = settings()
        expected_error = pytest.raises(RuntimeError, match="profiling failed")
        with (
            expected_error if fail else nullcontext(),
            gpu_worker.Worker._scoped_allocator_max_split(SimpleNamespace(), 20),
        ):
            scoped = settings()
            assert scoped["max_split_size"] == 20 * 1024 * 1024
            for key in ("garbage_collection_threshold", "roundup_power2_divisions"):
                assert scoped[key] == before[key]
            if fail:
                raise RuntimeError("profiling failed")
        assert settings() == before
    finally:
        torch._C._accelerator_setAllocatorSettings(original)


def test_scoped_max_split_ignores_async_allocator(monkeypatch):
    """Async allocators ignore max_split and may not support memory snapshots."""
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(
        torch.cuda.memory, "get_allocator_backend", lambda: "cudaMallocAsync"
    )
    with (
        patch.object(torch.cuda.memory, "_snapshot") as snapshot,
        patch.object(torch._C, "_accelerator_setAllocatorSettings") as set_settings,
        gpu_worker.Worker._scoped_allocator_max_split(SimpleNamespace(), 20),
    ):
        pass
    snapshot.assert_not_called()
    set_settings.assert_not_called()


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 1])
def test_memory_profile_bounds_decode_logits_rows(monkeypatch, kv_cache_memory_bytes):
    """ROCm decode workspace uses max_num_seqs during startup profiling."""
    from vllm.config import get_current_vllm_config_or_none
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _max_decode_logits_rows

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    class Profiled(Exception):
        pass

    def profile_run():
        # Without the config context, the helper falls back to all 32768
        # batched tokens, which would reserve 128 GiB at 1M context.
        assert _max_decode_logits_rows(32768) == 16 * (1 + 5)
        raise Profiled

    worker = SimpleNamespace(
        vllm_config=config,
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes),
        model_runner=SimpleNamespace(profile_run=profile_run, model_memory_usage=0),
        init_snapshot=SimpleNamespace(free_memory=2),
        _scoped_allocator_max_split=lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _: None)
    monkeypatch.setattr(
        gpu_worker, "memory_profiling", lambda *args, **kwargs: nullcontext()
    )

    assert get_current_vllm_config_or_none() is None
    with pytest.raises(Profiled):
        gpu_worker.Worker.determine_available_memory(worker)
    assert get_current_vllm_config_or_none() is None


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


# Memory accounting of the profiling run (Worker.determine_available_memory).

# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None


class _OrderedHandle:
    """Send handle that logs when it is waited."""

    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def is_completed(self) -> bool:
        return True

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def test_execute_model_waits_previous_pp_send_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Previous device handles are waited before the forward pass; the
    metadata handle is left to the GroupCoordinator's reaper."""
    import torch

    from vllm.sequence import IntermediateTensors

    log: list[str] = []
    previous_tensor_send = _OrderedHandle(log, "prev-tensor")
    metadata_handle = _OrderedHandle(log, "meta")
    tensor_handle = _OrderedHandle(log, "tensor")

    def isend_tensor_dict(tensors, all_gather_group=None, all_gather_tensors=None):
        log.append("isend")
        return [metadata_handle, tensor_handle]

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=False,
        isend_tensor_dict=isend_tensor_dict,
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace())

    def run_model(scheduler_output, intermediate_tensors):
        log.append("forward")
        return IntermediateTensors({"hidden_states": torch.zeros(1)})

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=run_model),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[previous_tensor_send],
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 4}
    )

    assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    assert log == ["wait:prev-tensor", "forward", "isend"]
    assert worker._pp_send_work == [tensor_handle]


def test_jit_monitor_activation_follows_enable_jit_warmup(
    monkeypatch: pytest.MonkeyPatch,
):
    """The post-warmup JIT monitor must stay off when JIT warmup is disabled
    (e.g. by enforce_eager): runtime compilation is then expected, and
    warning/erroring on it would be noise."""
    from vllm.utils import jit_monitor

    calls = []
    monkeypatch.setattr(jit_monitor, "activate", lambda **kwargs: calls.append(kwargs))

    def worker(enable_jit_warmup):
        return SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup)
            ),
            observability_config=SimpleNamespace(
                jit_monitor_mode="warn", jit_monitor_verbose=False
            ),
        )

    gpu_worker.Worker._maybe_activate_jit_monitor(worker(True))
    assert calls == [{"mode": "warn", "verbose": False}]

    calls.clear()
    gpu_worker.Worker._maybe_activate_jit_monitor(worker(False))
    assert calls == []
