# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    PrefetchTransferStats,
    _ModuleOffloader,
)
from vllm.model_executor.offloader.prefetch_diagnostics import (
    PrefetchScheduleRow,
    build_prefetch_schedule_rows,
    log_prefetch_schedule,
)
from vllm.model_executor.offloader.prefetch_helpers import nvtx_range
from vllm.model_executor.offloader.prefetch_tail_copy import (
    is_wraparound_prefetch,
    iter_chunked_tensor_views,
)
from vllm.model_executor.offloader.runtime import PrefetchRuntimeController


class _FakeStream:
    def __init__(self):
        self.waited_events: list[object] = []
        self.waited_streams: list[object] = []
        self.synchronize_count = 0

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)

    def wait_stream(self, stream: object) -> None:
        self.waited_streams.append(stream)

    def synchronize(self) -> None:
        self.synchronize_count += 1


class _FakeEvent:
    def __init__(self, elapsed_ms: float):
        self.elapsed_ms = elapsed_ms
        self.recorded_streams: list[object] = []

    def elapsed_time(self, end_event: "_FakeEvent") -> float:
        return end_event.elapsed_ms - self.elapsed_ms

    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        self.recorded_streams.append(stream)


class _QueryForbiddenEvent(_FakeEvent):
    def query(self) -> bool:
        raise AssertionError("query must not run during CUDA graph capture")


class _RecordingCudaEvent:
    def record(self, stream=None) -> None:
        return


class _RecordingCudaStream:
    def __init__(self):
        self.waited_events: list[object] = []
        self.recorded_events: list[object] = []

    def wait_event(self, event: object) -> None:
        self.waited_events.append(event)

    def record_event(self, event: object) -> None:
        self.recorded_events.append(event)


class _FakeParamOffloader:
    def __init__(self, cpu_storage: torch.Tensor):
        self._cpu_storage = cpu_storage
        self._gpu_buffer = torch.empty_like(cpu_storage)
        self.ensure_count = 0
        self.synced_count = 0

    def ensure_cpu_master_freshness(self) -> None:
        self.ensure_count += 1

    def mark_cpu_master_synced(self) -> None:
        self.synced_count += 1


class _OrderRecordingStream(_FakeStream):
    def __init__(self, calls: list[str]):
        super().__init__()
        self.calls = calls

    def wait_event(self, event: object) -> None:
        self.calls.append("cuda_wait_event")
        super().wait_event(event)


def test_prefetch_offload_nvtx_range_is_opt_in(monkeypatch):
    from vllm.model_executor.offloader import prefetch_helpers

    calls = []

    @contextlib.contextmanager
    def fake_nvtx_range(name):
        calls.append(("enter", name))
        yield
        calls.append(("exit", name))

    monkeypatch.setattr(prefetch_helpers.envs, "VLLM_NVTX_SCOPES_FOR_PROFILING", False)
    monkeypatch.setattr(prefetch_helpers.torch.cuda.nvtx, "range", fake_nvtx_range)

    with nvtx_range("weight_offload.test"):
        calls.append(("body", "disabled"))

    assert calls == [("body", "disabled")]

    monkeypatch.setattr(prefetch_helpers.envs, "VLLM_NVTX_SCOPES_FOR_PROFILING", True)

    with nvtx_range("weight_offload.test"):
        calls.append(("body", "enabled"))

    assert calls == [
        ("body", "disabled"),
        ("enter", "weight_offload.test"),
        ("body", "enabled"),
        ("exit", "weight_offload.test"),
    ]


def test_prefetch_transfer_stats_records_bytes_and_completed_timings():
    stats = PrefetchTransferStats()

    stats.record_copy(1024, _FakeEvent(0.0), _FakeEvent(1.0))
    stats.record_copy(2048, _FakeEvent(1.0), _FakeEvent(3.0))
    stats.record_wait(_FakeEvent(3.0), _FakeEvent(3.5))
    stats.record_wait(_FakeEvent(4.0), _FakeEvent(4.25))

    assert stats.snapshot() == {
        "h2d_bytes": 3072,
        "copy_count": 2,
        "copy_time_s": 0.0,
        "wait_time_s": 0.0,
        "effective_copy_bandwidth_bytes_per_s": 0.0,
    }

    stats.flush_copy_timings()
    stats.flush_wait_timings()

    assert stats.h2d_bytes == 3072
    assert stats.copy_count == 2
    assert stats.copy_time_s == pytest.approx(0.003)
    assert stats.wait_time_s == pytest.approx(0.00075)
    assert stats.effective_copy_bandwidth_bytes_per_s == pytest.approx(1024000)
    assert "effective_bandwidth_bytes_per_s" not in stats.snapshot()


def test_prefetch_h2d_copy_chunks_preserve_tensor_contents():
    src = torch.arange(10, dtype=torch.uint8)
    dst = torch.empty_like(src)

    chunks = list(iter_chunked_tensor_views(dst, src, src.numel(), chunk_bytes=4))
    for dst_chunk, src_chunk, _ in chunks:
        dst_chunk.copy_(src_chunk)

    assert [num_bytes for _, _, num_bytes in chunks] == [4, 4, 2]
    assert torch.equal(dst, src)


def test_prefetch_h2d_copy_chunking_is_disabled_with_zero_chunk_size():
    src = torch.arange(10, dtype=torch.uint8)
    dst = torch.empty_like(src)

    chunks = list(iter_chunked_tensor_views(dst, src, src.numel(), chunk_bytes=0))

    assert len(chunks) == 1
    assert chunks[0][0] is dst
    assert chunks[0][1] is src
    assert chunks[0][2] == 10


def test_prefetch_wraparound_tail_detection():
    assert is_wraparound_prefetch(source_unit_idx=16, target_unit_idx=0)
    assert not is_wraparound_prefetch(source_unit_idx=0, target_unit_idx=16)
    assert not is_wraparound_prefetch(source_unit_idx=0, target_unit_idx=None)


def test_prefetch_start_passes_tail_hint_to_module_onload():
    calls: list[bool] = []

    class _FakeModuleOffloader:
        def start_onload_to_static(self, *, allow_paced_chunking: bool = False) -> bool:
            calls.append(allow_paced_chunking)
            return False

        def ensure_cpu_master_freshness(self) -> None:
            return

        def release_runtime_buffer_tracking(self) -> None:
            return

    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.runtime = PrefetchRuntimeController(unit_count=20, prefetch_step=16)
    offloader.module_offloaders = [_FakeModuleOffloader() for _ in range(20)]

    offloader._start_prefetch(16, is_tail_prefetch=False)
    offloader._start_prefetch(0, is_tail_prefetch=True)

    assert calls == [False, True]


def test_prefetch_transfer_stats_defers_copy_timing_queries_during_capture():
    stats = PrefetchTransferStats()

    stats.record_copy(1024, _FakeEvent(0.0), _QueryForbiddenEvent(1.0))
    stats.flush_copy_timings(skip_query=True)

    assert stats.copy_time_s == 0.0
    assert stats.h2d_bytes == 1024
    assert stats.copy_count == 1
    assert len(stats._pending_copy_events) == 1

    stats._pending_copy_events = [(_FakeEvent(0.0), _FakeEvent(1.0))]
    stats.flush_copy_timings()

    assert stats.h2d_bytes == 1024
    assert stats.copy_count == 1
    assert stats.copy_time_s == pytest.approx(0.001)
    assert stats._pending_copy_events == []


def test_prefetch_wait_waits_for_host_recorded_copy_event(monkeypatch):
    calls: list[str] = []
    compute_stream = _OrderRecordingStream(calls)
    cuda_event = object()

    class _OffloaderWithDelayedEvent:
        _event_valid_for_eager = True
        _copy_done_event = cuda_event

        def wait_until_copy_done_event_recorded(self) -> None:
            calls.append("host_event_recorded")

    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_offloader_ext.torch.cuda.current_stream",
        lambda: compute_stream,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.cuda.is_current_stream_capturing",
        lambda: False,
    )

    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.module_offloaders = [_OffloaderWithDelayedEvent()]
    offloader.runtime = PrefetchRuntimeController(unit_count=1, prefetch_step=1)
    offloader.copy_stream = _FakeStream()
    offloader.transfer_stats = PrefetchTransferStats()

    offloader._wait_for_layer(0)

    assert calls == ["host_event_recorded", "cuda_wait_event"]


def test_prefetch_wait_instrumentation_records_gpu_wait_event_time(monkeypatch):
    fake_stream = _FakeStream()
    monkeypatch.setenv("VLLM_PREFETCH_LOG_TRANSFER_STATS", "1")
    captured_logs = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.logger.info",
        lambda *args, **kwargs: captured_logs.append((args, kwargs)),
    )
    wait_events = iter((_FakeEvent(0.0), _FakeEvent(2.0)))
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.cuda.Stream",
        lambda: object(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_offloader_ext.torch.cuda.current_stream",
        lambda: fake_stream,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.torch.cuda.is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_offloader_ext.torch.cuda.Event",
        lambda *args, **kwargs: next(wait_events),
    )

    offloader = PrefetchOffloader(group_size=1, num_in_group=1, prefetch_step=1)
    captured_logs.clear()
    offloader.runtime = SimpleNamespace(is_pending_in_capture=lambda layer_idx: False)
    offloader.module_offloaders = [
        SimpleNamespace(
            _event_valid_for_eager=True,
            _copy_done_event="copy-done",
            wait_until_copy_done_event_recorded=lambda: None,
        )
    ]

    offloader._wait_for_layer(0)

    assert fake_stream.waited_events == ["copy-done"]
    assert offloader.transfer_stats.wait_time_s == pytest.approx(0.002)
    assert captured_logs == []


def test_prefetch_transfer_stats_reset_clears_all_counters():
    stats = PrefetchTransferStats()

    stats.record_copy(1024, _FakeEvent(0.0), _FakeEvent(2.0))
    stats.record_wait(_FakeEvent(2.0), _FakeEvent(2.25))
    stats.flush_copy_timings()
    stats.flush_wait_timings()
    stats.reset()

    assert stats.snapshot() == {
        "h2d_bytes": 0,
        "copy_count": 0,
        "copy_time_s": 0.0,
        "wait_time_s": 0.0,
        "effective_copy_bandwidth_bytes_per_s": 0.0,
    }


def test_prefetch_transfer_stats_begin_resets_previous_window(monkeypatch):
    monkeypatch.setenv("VLLM_PREFETCH_LOG_TRANSFER_STATS", "1")
    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.transfer_stats = PrefetchTransferStats(h2d_bytes=1024, copy_count=1)
    offloader.transfer_stats.record_copy(2048, _FakeEvent(0.0), _FakeEvent(1.0))

    offloader.begin_forward_stats()

    assert offloader.transfer_stats.snapshot() == {
        "h2d_bytes": 0,
        "copy_count": 0,
        "copy_time_s": 0.0,
        "wait_time_s": 0.0,
        "effective_copy_bandwidth_bytes_per_s": 0.0,
    }


def test_prefetch_transfer_stats_logging_syncs_and_reports_once(monkeypatch):
    monkeypatch.setenv("VLLM_PREFETCH_LOG_TRANSFER_STATS", "1")
    compute_stream = _FakeStream()
    copy_stream = _FakeStream()
    captured_logs = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_offloader_ext.torch.cuda.current_stream",
        lambda: compute_stream,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_offloader_ext.logger.info",
        lambda *args, **kwargs: captured_logs.append((args, kwargs)),
    )

    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader.copy_stream = copy_stream
    offloader.transfer_stats = PrefetchTransferStats()
    offloader.transfer_stats.record_copy(
        2_000_000_000, _FakeEvent(0.0), _FakeEvent(2.0)
    )
    offloader.transfer_stats.record_wait(_FakeEvent(2.0), _FakeEvent(2.5))

    offloader.end_forward_stats()

    assert compute_stream.synchronize_count == 1
    assert copy_stream.synchronize_count == 1
    assert len(captured_logs) == 1
    args, kwargs = captured_logs[0]
    assert kwargs == {}
    assert args == (
        "[PrefetchOffloader] forward_stats: "
        "h2d_gb=%.2f h2d_copy_ops=%d "
        "gpu_copy_time_s=%.6f gpu_wait_time_s=%.6f "
        "gpu_copy_bandwidth_gb_s=%.2f",
        2.0,
        1,
        0.002,
        0.0005,
        1000.0,
    )
    assert offloader.transfer_stats.snapshot()["h2d_bytes"] == 0


def test_prefetch_schedule_logging_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_PREFETCH_LOG_SCHEDULE", raising=False)
    captured_logs = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_diagnostics.logger.info",
        lambda *args, **kwargs: captured_logs.append((args, kwargs)),
    )

    runtime = PrefetchRuntimeController(unit_count=2, prefetch_step=1)
    units = [
        SimpleNamespace(module_index=0, param_names=("weight",)),
        SimpleNamespace(module_index=1, param_names=("weight",)),
    ]

    log_prefetch_schedule(units, runtime)

    assert captured_logs == []


def test_prefetch_schedule_rows_report_initial_slots_and_load_lead():
    runtime = PrefetchRuntimeController(unit_count=20, prefetch_step=16)
    units = [
        SimpleNamespace(module_index=layer_idx, param_names=("weight",))
        for layer_idx in range(20)
    ]

    rows = build_prefetch_schedule_rows(units, runtime)

    assert rows[0] == PrefetchScheduleRow(
        layer_idx=0,
        unit_idx=0,
        slot_idx=0,
        initial=True,
        load_after_layer_idx=None,
        lead_layers=None,
        steady_state_load_after_layer_idx=16,
        steady_state_lead_layers=4,
    )
    assert rows[16] == PrefetchScheduleRow(
        layer_idx=16,
        unit_idx=16,
        slot_idx=0,
        initial=False,
        load_after_layer_idx=0,
        lead_layers=16,
        steady_state_load_after_layer_idx=0,
        steady_state_lead_layers=16,
    )


def test_prefetch_schedule_rows_mark_singleton_slots_as_initial():
    runtime = PrefetchRuntimeController(unit_count=15, prefetch_step=8)
    units = [
        SimpleNamespace(module_index=layer_idx, param_names=("weight",))
        for layer_idx in range(15)
    ]

    rows = build_prefetch_schedule_rows(units, runtime)

    assert rows[7] == PrefetchScheduleRow(
        layer_idx=7,
        unit_idx=7,
        slot_idx=7,
        initial=True,
        load_after_layer_idx=None,
        lead_layers=None,
        steady_state_load_after_layer_idx=None,
        steady_state_lead_layers=None,
    )
    assert rows[14] == PrefetchScheduleRow(
        layer_idx=14,
        unit_idx=14,
        slot_idx=6,
        initial=False,
        load_after_layer_idx=6,
        lead_layers=8,
        steady_state_load_after_layer_idx=6,
        steady_state_lead_layers=8,
    )


def test_prefetch_schedule_rows_include_non_offloaded_layers():
    runtime = PrefetchRuntimeController(unit_count=4, prefetch_step=2)
    units = [
        SimpleNamespace(module_index=2, param_names=("weight",)),
        SimpleNamespace(module_index=3, param_names=("weight",)),
        SimpleNamespace(module_index=6, param_names=("weight",)),
        SimpleNamespace(module_index=7, param_names=("weight",)),
    ]

    rows = build_prefetch_schedule_rows(units, runtime, module_count=8)

    assert [row.layer_idx for row in rows] == list(range(8))
    assert rows[0] == PrefetchScheduleRow(
        layer_idx=0,
        unit_idx=None,
        slot_idx=None,
        initial=False,
        load_after_layer_idx=None,
        lead_layers=None,
        steady_state_load_after_layer_idx=None,
        steady_state_lead_layers=None,
    )
    assert rows[2] == PrefetchScheduleRow(
        layer_idx=2,
        unit_idx=0,
        slot_idx=0,
        initial=True,
        load_after_layer_idx=None,
        lead_layers=None,
        steady_state_load_after_layer_idx=6,
        steady_state_lead_layers=4,
    )
    assert rows[6] == PrefetchScheduleRow(
        layer_idx=6,
        unit_idx=2,
        slot_idx=0,
        initial=False,
        load_after_layer_idx=2,
        lead_layers=4,
        steady_state_load_after_layer_idx=2,
        steady_state_lead_layers=4,
    )


def test_prefetch_schedule_reports_steady_state_for_initial_layers():
    runtime = PrefetchRuntimeController(unit_count=12, prefetch_step=7)
    units = [
        SimpleNamespace(module_index=layer_idx, param_names=("weight",))
        for layer_idx in range(3, 48, 4)
    ]

    rows = build_prefetch_schedule_rows(units, runtime, module_count=48)

    assert rows[3].initial
    assert rows[3].load_after_layer_idx is None
    assert rows[3].steady_state_load_after_layer_idx == 31
    assert rows[3].steady_state_lead_layers == 20
    assert rows[23].initial
    assert rows[23].steady_state_load_after_layer_idx is None
    assert rows[27].initial
    assert rows[27].steady_state_load_after_layer_idx is None


def test_prefetch_schedule_logging_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_PREFETCH_LOG_SCHEDULE", "1")
    captured_logs = []
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_diagnostics.logger.info",
        lambda *args, **kwargs: captured_logs.append((args, kwargs)),
    )

    runtime = PrefetchRuntimeController(unit_count=20, prefetch_step=16)
    units = [
        SimpleNamespace(module_index=layer_idx, param_names=("weight",))
        for layer_idx in range(20)
    ]

    log_prefetch_schedule(units, runtime)

    assert len(captured_logs) == 1
    args, kwargs = captured_logs[0]
    assert kwargs == {}
    assert args[0] == "[PrefetchOffloader] prefetch schedule:\n%s"
    table = args[1]
    assert "layer_idx" in table
    assert "unit_idx" in table
    assert "slot_idx" in table
    assert "initial" in table
    assert "lead_layers" in table
    assert "load_after_unit_idx" not in table
    assert "weights_loaded_when" in table
    assert "steady_state_loaded_when" in table
    assert "0          0         0         True" in table
    assert "16         16        0         False" in table
    assert "0                     16" in table
    assert "initial -> load layer 0 into slot 0" in table
    assert "after layer 0 -> load layer 16 into slot 0" in table
    assert "after layer 16 -> load layer 0 into slot 0" in table


@pytest.mark.parametrize(
    ("use_slab_copy", "expected_copy_count", "expected_h2d_bytes"),
    [
        (True, 1, 24),
        (False, 2, 24),
    ],
)
def test_module_onload_uses_one_slab_copy_for_packable_tensors(
    monkeypatch,
    use_slab_copy: bool,
    expected_copy_count: int,
    expected_h2d_bytes: int,
):
    monkeypatch.setenv("VLLM_PREFETCH_LOG_TRANSFER_STATS", "1")
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.should_pin_memory",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.is_current_stream_capturing",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.Event",
        lambda *args, **kwargs: _RecordingCudaEvent(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.current_stream",
        lambda: _RecordingCudaStream(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.stream",
        lambda stream: contextlib.nullcontext(),
    )

    copy_stream = _RecordingCudaStream()
    transfer_stats = PrefetchTransferStats()
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader.copy_stream = copy_stream
    offloader.transfer_stats = transfer_stats
    offloader.layer_idx = 0
    offloader._buffer_slot_idx = 0
    offloader._copy_done_event = _RecordingCudaEvent()
    offloader._event_valid_for_eager = False
    offloader._copy_thread_error = None
    offloader._copy_done_event_recorded = SimpleNamespace(
        wait=lambda: None, clear=lambda: None, set=lambda: None
    )
    offloader._use_slab_copy = use_slab_copy
    offloader._slab_param_names = ("a", "b")
    offloader._storage_group_infos = ()
    offloader._storage_group_buffers = []
    offloader._direct_param_names = ()
    offloader._buffer_pool = object()
    offloader._cpu_slab = torch.arange(24, dtype=torch.uint8)
    offloader._gpu_slab = torch.empty(24, dtype=torch.uint8)
    # uses_slab_buffers / uses_storage_group_fallback / uses_direct_fallback
    # are properties on the class; the manual instance state above already
    # drives them through self._slab_param_names / etc.
    offloader._param_offloaders = {
        "a": _FakeParamOffloader(torch.ones(4, dtype=torch.float16)),
        "b": _FakeParamOffloader(torch.ones(4, dtype=torch.float32)),
    }

    in_capture = offloader.start_onload_to_static()

    assert in_capture is False
    assert transfer_stats.copy_count == expected_copy_count
    assert transfer_stats.h2d_bytes == expected_h2d_bytes
    assert len(transfer_stats._pending_copy_events) == expected_copy_count
    assert [p.ensure_count for p in offloader._param_offloaders.values()] == [1, 1]
    assert [p.synced_count for p in offloader._param_offloaders.values()] == [1, 1]


def test_module_onload_does_not_record_timing_events_during_capture(monkeypatch):
    monkeypatch.setenv("VLLM_PREFETCH_LOG_TRANSFER_STATS", "1")
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.should_pin_memory",
        lambda: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.Event",
        lambda *args, **kwargs: _RecordingCudaEvent(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.current_stream",
        lambda: _RecordingCudaStream(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch_onload.torch.cuda.stream",
        lambda stream: contextlib.nullcontext(),
    )

    transfer_stats = PrefetchTransferStats()
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader.copy_stream = _RecordingCudaStream()
    offloader.transfer_stats = transfer_stats
    offloader.layer_idx = 0
    offloader._buffer_slot_idx = 0
    offloader._copy_done_event = _RecordingCudaEvent()
    offloader._event_valid_for_eager = True
    offloader._copy_thread_error = None
    offloader._copy_done_event_recorded = SimpleNamespace(
        wait=lambda: None, clear=lambda: None, set=lambda: None
    )
    offloader._use_slab_copy = True
    offloader._slab_param_names = ("a",)
    offloader._storage_group_infos = ()
    offloader._storage_group_buffers = []
    offloader._direct_param_names = ()
    offloader._buffer_pool = object()
    offloader._cpu_slab = torch.arange(8, dtype=torch.uint8)
    offloader._gpu_slab = torch.empty(8, dtype=torch.uint8)
    offloader._param_offloaders = {
        "a": _FakeParamOffloader(torch.ones(4, dtype=torch.float16)),
    }

    in_capture = offloader.start_onload_to_static()

    assert in_capture is True
    assert transfer_stats.copy_count == 1
    assert transfer_stats.h2d_bytes == 8
    assert transfer_stats._pending_copy_events == []
    assert offloader._event_valid_for_eager is False
