# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import zmq

import vllm.v1.metrics.forward_pass_metrics as fpm_module
from vllm.config import ObservabilityConfig
from vllm.utils.network_utils import get_open_port, get_open_ports_list
from vllm.v1.metrics.forward_pass_metrics import (
    FPM_TIMING_SCOPE_MODEL_STEP_CUDA,
    ForwardPassMetrics,
    ForwardPassMetricsExporter,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    decode_forward_pass_metrics,
    encode_forward_pass_metrics,
    forward_pass_metrics_from_iteration,
)
from vllm.v1.metrics.iteration_workload import (
    QueuedIterationWorkload,
    ScheduledIterationWorkload,
)
from vllm.v1.metrics.stats import SchedulerIterationDetails

pytestmark = pytest.mark.skip_global_cleanup


def _record(**kwargs):
    return replace(
        SchedulerIterationDetails(
            7,
            1,
            16,
            2,
            8,
            elapsed_ms=12.5,
            timing_scope=FPM_TIMING_SCOPE_MODEL_STEP_CUDA,
            scheduled_workload=ScheduledIterationWorkload(24, 4.0, 256, 16.0),
            queued_workload=QueuedIterationWorkload(1, 8, 0.0, 2, 128, 0.0),
        ),
        **kwargs,
    )


def test_iteration_mapping_preserves_fpm_v1_wire_contract():
    metrics = forward_pass_metrics_from_iteration(_record(), "worker", 2, 3)
    assert metrics is not None
    payload = encode_forward_pass_metrics(metrics)
    assert msgspec.msgpack.decode(payload) == {
        "version": 1,
        "worker_id": "worker",
        "dp_rank": 2,
        "counter_id": 3,
        "timing_scope": "model_step_cuda",
        "wall_time": 0.0125,
        "scheduled_requests": {
            "num_prefill_requests": 1,
            "sum_prefill_tokens": 16,
            "var_prefill_length": 4.0,
            "sum_prefill_kv_tokens": 24,
            "num_decode_requests": 2,
            "sum_decode_kv_tokens": 256,
            "var_decode_kv_tokens": 16.0,
        },
        "queued_requests": {
            "num_prefill_requests": 1,
            "sum_prefill_tokens": 8,
            "var_prefill_length": 0.0,
            "num_decode_requests": 2,
            "sum_decode_kv_tokens": 128,
            "var_decode_kv_tokens": 0.0,
        },
    }
    assert decode_forward_pass_metrics(payload) == metrics


@pytest.mark.parametrize("elapsed_ms", [None, 0.0])
def test_incomplete_iterations_are_not_exported_as_idle(elapsed_ms):
    record = _record(elapsed_ms=elapsed_ms)
    assert forward_pass_metrics_from_iteration(record, "worker", 0, 0) is None


@pytest.mark.parametrize("idle", [False, True])
def test_dummy_and_explicit_idle_preserve_waiting_work(idle):
    record = _record(
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=0,
        num_generation_tokens=0,
        scheduled_workload=ScheduledIterationWorkload(),
        elapsed_ms=0.0 if idle else 10.0,
        is_idle=idle,
        is_dummy=not idle,
    )
    metrics = forward_pass_metrics_from_iteration(record, "worker", 0, 0)
    assert metrics is not None
    assert metrics.wall_time == (0.0 if idle else 0.01)
    assert metrics.scheduled_requests == ScheduledRequestMetrics()
    assert metrics.queued_requests.sum_prefill_tokens == 8
    assert metrics.queued_requests.sum_decode_kv_tokens == 128


def test_fpm_configuration_enables_collection_without_verbose_logs():
    config = ObservabilityConfig(forward_pass_metrics_port=20380)
    assert config.iteration_timing_enabled
    assert config.enable_iteration_workload_metrics
    assert not config.enable_logging_iteration_details
    assert (
        ForwardPassMetricsExporter.from_vllm_config(
            SimpleNamespace(observability_config=ObservabilityConfig())
        )
        is None
    )


def test_each_engine_publishes_its_global_rank_and_stops_cleanly():
    ports = sorted(get_open_ports_list(2))
    ranks = [port - ports[0] for port in ports]
    subscribers = []
    for port in ports:
        subscriber = zmq.Context.instance().socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        subscribers.append(subscriber)
    exporters = [
        ForwardPassMetricsExporter.from_vllm_config(
            SimpleNamespace(
                observability_config=ObservabilityConfig(
                    forward_pass_metrics_port=ports[0]
                ),
                # Dense DP can reset the execution rank to zero.
                parallel_config=SimpleNamespace(
                    data_parallel_rank=0, data_parallel_index=rank
                ),
                instance_id="worker",
            )
        )
        for rank in ranks
    ]
    try:
        # Silence must not create an idle heartbeat in the exporter.
        assert not subscribers[0].poll(50)
        for rank, subscriber, exporter in zip(ranks, subscribers, exporters):
            assert exporter is not None
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                exporter.record([_record()])
                if subscriber.poll(50):
                    break
            else:
                pytest.fail("timed out waiting for forward-pass metrics")
            topic, sequence, payload = subscriber.recv_multipart()
            metrics = decode_forward_pass_metrics(payload)
            assert metrics is not None
            assert topic == b""
            assert int.from_bytes(sequence, "big") == metrics.counter_id
            assert metrics.worker_id == "worker" and metrics.dp_rank == rank
            assert metrics.wall_time == 0.0125
    finally:
        for exporter in exporters:
            assert exporter is not None
            exporter.shutdown()
        for subscriber in subscribers:
            subscriber.close(linger=0)
    assert all(
        exporter is not None and not exporter._thread.is_alive()
        for exporter in exporters
    )


def test_zmq_exporter_reports_startup_bind_failure():
    port = get_open_port()
    blocker = zmq.Context.instance().socket(zmq.PUB)
    blocker.bind(f"tcp://127.0.0.1:{port}")
    try:
        with pytest.raises(RuntimeError, match="Failed to bind"):
            ForwardPassMetricsExporter(port, "worker", 0)
    finally:
        blocker.close(linger=0)


def test_slow_publisher_drops_without_blocking_and_keeps_sequence_gap(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    sent: queue.Queue[tuple[bytes, bytes, bytes]] = queue.Queue()
    converter = forward_pass_metrics_from_iteration

    def blocked_converter(*args):
        entered.set()
        assert release.wait(2)
        return converter(*args)

    monkeypatch.setattr(fpm_module, "FPM_MAX_QUEUE_SIZE", 1)
    monkeypatch.setattr(
        fpm_module, "forward_pass_metrics_from_iteration", blocked_converter
    )
    monkeypatch.setattr(
        ForwardPassMetricsExporter,
        "_bind_socket",
        lambda *_: SimpleNamespace(
            send_multipart=lambda frames, **_: sent.put(frames),
            close=lambda **_: None,
        ),
    )
    exporter = ForwardPassMetricsExporter(20380, "worker", 0)
    try:
        exporter.record([_record()])
        assert entered.wait(2)
        exporter.record([_record(), _record()])
        release.set()
        assert [int.from_bytes(sent.get(timeout=2)[1], "big") for _ in range(2)] == [
            0,
            1,
        ]
        exporter.record([_record()])
        assert int.from_bytes(sent.get(timeout=2)[1], "big") == 3
    finally:
        release.set()
        exporter.shutdown()


def test_subscriber_sample_fits_two_80_column_lines(capsys):
    from examples.features.forward_pass_metrics.forward_pass_metrics_subscriber import (
        print_metrics,
    )

    count = 2**64 - 1
    metrics = ForwardPassMetrics(
        worker_id="long-worker-identity" * 10,
        wall_time=1.234567e100,
        scheduled_requests=ScheduledRequestMetrics(
            count, count, 1.234567e200, count, count, count, 1.234567e200
        ),
        queued_requests=QueuedRequestMetrics(
            count, count, 1.234567e200, count, count, 1.234567e200
        ),
    )
    print_metrics(count, metrics)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all(len(line) <= 80 for line in lines)
    assert " S " in lines[0] and " Q " in lines[1]
    assert "/-/" in lines[1]  # Queued prefill has no KV-read metric.


@pytest.mark.parametrize("interval", [0.0, 1.0])
def test_subscriber_refresh_keeps_latest_sample_and_idle_tail(
    interval, monkeypatch, capsys
):
    from examples.features.forward_pass_metrics import (
        forward_pass_metrics_subscriber as subscriber,
    )

    clock = [0.0]
    monkeypatch.setattr(subscriber, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    display = subscriber.MetricsDisplay(interval)
    metrics = ForwardPassMetrics(worker_id="worker\n\x1b[31m" * 20, wall_time=0.012)
    display.update(1, metrics)
    initial = capsys.readouterr().out
    assert all(len(line) <= 80 for line in initial.splitlines())
    assert "\x1b" not in initial
    for seq in range(2, 4):
        display.update(seq, metrics)
    updates = capsys.readouterr().out
    assert len(updates.splitlines()) == (2 if interval == 0 else 0)
    if interval:
        assert display.poll_timeout_ms() == 1000
        clock[0] = 1.0
        assert display.poll_timeout_ms() == 0
        display.flush()  # The last sample is visible even without more messages.
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and lines[0].lstrip().startswith("3 ")
    assert display.poll_timeout_ms() == subscriber.POLL_TIMEOUT_MS
    clock[0] = 2.0
    display.update(4, msgspec.structs.replace(metrics, wall_time=0.0))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and "idle (heartbeat)" in lines[0]


def test_subscriber_validates_messages_hidden_by_refresh(monkeypatch, capsys):
    from examples.features.forward_pass_metrics import (
        forward_pass_metrics_subscriber as subscriber,
    )

    frames = deque(
        [
            b"",
            seq.to_bytes(8, "big"),
            encode_forward_pass_metrics(
                ForwardPassMetrics(counter_id=99 if seq == 1 else seq, wall_time=0.012)
            ),
        ]
        for seq in range(3)
    )

    def poll(_timeout):
        if not frames:
            raise KeyboardInterrupt
        return True

    socket = SimpleNamespace(
        connect=lambda _: None,
        setsockopt=lambda *args: None,
        poll=poll,
        recv_multipart=frames.popleft,
        close=lambda **kwargs: None,
    )
    monkeypatch.setattr(subscriber, "time", SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setattr(
        subscriber,
        "parse_args",
        lambda: SimpleNamespace(interval=1.0, endpoint="tcp://localhost:20380"),
    )
    monkeypatch.setattr(
        subscriber.zmq,
        "Context",
        lambda: SimpleNamespace(socket=lambda _: socket, term=lambda: None),
    )
    subscriber.main()
    out = capsys.readouterr()
    assert "payload counter 99" in out.err
    assert "Sequence gap" not in out.err
    assert not any(line.lstrip().startswith("1 ") for line in out.out.splitlines())
    assert any(line.lstrip().startswith("2 ") for line in out.out.splitlines())
