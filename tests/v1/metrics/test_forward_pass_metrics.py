# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import pytest
import zmq

import vllm.v1.metrics.forward_pass_metrics as fpm_module
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.forward_pass_metrics import (
    FPM_TIMING_SCOPE_EXECUTE_MODEL_CUDA,
    ForwardPassMetrics,
    ForwardPassMetricsEmitter,
    ForwardPassMetricsTimer,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    ZmqForwardPassMetricsPublisher,
    decode_forward_pass_metrics,
    encode_forward_pass_metrics,
    is_forward_pass_metrics_output_rank,
    make_forward_pass_metrics_timer,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.skip_global_cleanup

# Cross-repo golden payload mirrored by Dynamo's FPM contract test.
_GOLDEN_FPM_V2 = {
    "version": 2,
    "worker_id": "worker-0",
    "dp_rank": 2,
    "counter_id": 3,
    "timing_scope": "execute_model_cuda",
    "wall_time": 0.004,
    "scheduled_requests": {
        "num_prefill_requests": 1,
        "sum_prefill_tokens": 16,
        "var_prefill_length": 0.0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
        "var_decode_kv_tokens": 0.0,
    },
    "queued_requests": {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "var_prefill_length": 0.0,
        "num_decode_requests": 2,
        "sum_decode_kv_tokens": 128,
        "var_decode_kv_tokens": 0.0,
    },
}


class _FakeTimingEvent:
    def __init__(self, timestamps):
        self._timestamps = timestamps
        self.timestamp = 0.0
        self.complete = False
        self.synchronize_calls = 0

    def record(self):
        self.timestamp = next(self._timestamps)

    def query(self):
        return self.complete

    def synchronize(self):
        self.synchronize_calls += 1
        self.complete = True

    def elapsed_time(self, end_event):
        return end_event.timestamp - self.timestamp


class _FakeMetricsPublisher:
    def __init__(self):
        self.published = []

    def publish(self, metrics):
        self.published.append(metrics)

    def shutdown(self):
        pass


class _FakeScheduler:
    def __init__(self):
        self.requests = {
            "prefill": SimpleNamespace(num_prompt_tokens=8),
            "decode": SimpleNamespace(num_prompt_tokens=32),
        }
        self.waiting = []
        self.skipped_waiting = []
        self.emit = True

    def get_forward_pass_metrics_request_state(self):
        return self.requests, self.waiting, self.skipped_waiting

    def should_emit_forward_pass_metrics(self, scheduler_output):
        return self.emit


def _make_scheduler_output(
    request_id: str,
    *,
    prompt_tokens: int = 0,
    computed_tokens: int = 0,
    context_phase: bool = False,
) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request_id: max(prompt_tokens, 1)}
    output.total_num_scheduled_tokens = max(prompt_tokens, 1)
    if prompt_tokens:
        output.scheduled_new_reqs = [
            NewRequestData(
                req_id=request_id,
                prompt_token_ids=[0] * prompt_tokens,
                mm_features=[],
                sampling_params=None,
                pooling_params=None,
                block_ids=([],),
                num_computed_tokens=computed_tokens,
                lora_request=None,
            )
        ]
    else:
        output.scheduled_cached_reqs = CachedRequestData(
            req_ids=[request_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[computed_tokens],
            num_output_tokens=[0 if context_phase else 1],
        )
    return output


def test_forward_pass_metrics_v2_wire_contract():
    metrics = ForwardPassMetrics(
        worker_id="worker-0",
        dp_rank=2,
        counter_id=3,
        wall_time=0.004,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=1,
            sum_prefill_tokens=16,
        ),
        queued_requests=QueuedRequestMetrics(
            num_decode_requests=2,
            sum_decode_kv_tokens=128,
        ),
    )

    payload = encode_forward_pass_metrics(metrics)
    raw = msgspec.msgpack.decode(payload)

    assert raw == _GOLDEN_FPM_V2
    assert decode_forward_pass_metrics(payload) == metrics

    old_payload = encode_forward_pass_metrics(
        msgspec.structs.replace(metrics, version=1)
    )
    assert decode_forward_pass_metrics(old_payload) is None


def test_timing_tuple_survives_model_output_ipc_codec():
    output = ModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        forward_pass_timing_samples=((7, 0.012),),
    )

    encoded = MsgpackEncoder().encode(output)
    decoded = MsgpackDecoder(ModelRunnerOutput).decode(encoded)

    assert decoded.forward_pass_timing_samples == ((7, 0.012),)
    empty_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert empty_output.forward_pass_timing_samples == ()


def test_timer_hot_drain_never_synchronizes_pending_event():
    timestamps = iter((10.0, 15.0))
    events = []

    def event_factory():
        event = _FakeTimingEvent(timestamps)
        events.append(event)
        return event

    timer = ForwardPassMetricsTimer(num_event_pairs=1, event_factory=event_factory)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.forward_pass_metrics_iteration_id = 4

    timer.start(scheduler_output)
    timer.finish()
    assert timer.drain_samples() == ()
    assert sum(event.synchronize_calls for event in events) == 0

    events[-1].complete = True
    assert timer.drain_samples() == ((4, 0.005),)
    assert sum(event.synchronize_calls for event in events) == 0


def test_timer_cold_flush_waits_for_final_event():
    events = []

    def event_factory():
        event = _FakeTimingEvent(iter((20.0, 27.0)))
        events.append(event)
        return event

    # Each event owns its iterator, so provide explicit timestamps for records.
    timer = ForwardPassMetricsTimer(
        num_event_pairs=1,
        event_factory=event_factory,
    )
    events[0]._timestamps = iter((20.0,))
    events[1]._timestamps = iter((27.0,))
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.total_num_scheduled_tokens = 1
    scheduler_output.forward_pass_metrics_iteration_id = 9

    timer.start(scheduler_output)
    timer.finish()

    assert timer.drain_samples(wait=True) == ((9, 0.007),)
    assert events[1].synchronize_calls == 1


def test_timer_is_disabled_off_and_on_non_output_ranks():
    disabled = SimpleNamespace(
        observability_config=SimpleNamespace(forward_pass_metrics_port=0)
    )
    enabled = SimpleNamespace(
        observability_config=SimpleNamespace(forward_pass_metrics_port=20380)
    )

    assert make_forward_pass_metrics_timer(disabled, is_output_rank=True) is None
    assert make_forward_pass_metrics_timer(enabled, is_output_rank=False) is None


def test_only_executor_output_rank_owns_timing_events():
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=4,
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
        )
    )

    assert is_forward_pass_metrics_output_rank(config, 0)
    assert not is_forward_pass_metrics_output_rank(config, 1)
    assert not is_forward_pass_metrics_output_rank(config, 2)
    assert not is_forward_pass_metrics_output_rank(config, 3)


def test_emitter_joins_delayed_timing_with_original_snapshots():
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter("worker", 0, publisher)
    scheduler = _FakeScheduler()

    prefill = _make_scheduler_output("prefill", prompt_tokens=8, computed_tokens=2)
    emitter.begin_iteration(scheduler, prefill)
    scheduler.waiting = [
        SimpleNamespace(
            status=RequestStatus.WAITING,
            num_tokens=12,
            num_computed_tokens=0,
        )
    ]
    emitter.complete_iteration(
        scheduler,
        prefill,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )

    decode = _make_scheduler_output("decode", computed_tokens=64)
    emitter.begin_iteration(scheduler, decode)
    scheduler.waiting = []
    emitter.complete_iteration(
        scheduler,
        decode,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            forward_pass_timing_samples=(
                (prefill.forward_pass_metrics_iteration_id, 0.01),
            ),
        ),
    )

    assert len(publisher.published) == 1
    metrics = publisher.published[0]
    assert metrics.timing_scope == FPM_TIMING_SCOPE_EXECUTE_MODEL_CUDA
    assert metrics.wall_time == 0.01
    assert metrics.scheduled_requests.num_prefill_requests == 1
    assert metrics.scheduled_requests.sum_prefill_kv_tokens == 2
    assert metrics.queued_requests.num_prefill_requests == 1
    assert metrics.queued_requests.sum_prefill_tokens == 12


def test_emitter_classifies_chunked_prefill_and_queued_decode_states():
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter("worker", 0, publisher)
    scheduler = _FakeScheduler()
    scheduler.waiting = [
        SimpleNamespace(
            status=RequestStatus.PREEMPTED,
            num_tokens=100,
            num_computed_tokens=70,
        )
    ]
    scheduler.skipped_waiting = [
        SimpleNamespace(
            status=RequestStatus.WAITING_FOR_REMOTE_KVS,
            num_tokens=200,
            num_computed_tokens=80,
        )
    ]
    output = _make_scheduler_output("prefill", computed_tokens=4, context_phase=True)

    emitter.begin_iteration(scheduler, output)
    emitter.complete_iteration(
        scheduler,
        output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            forward_pass_timing_samples=(
                (output.forward_pass_metrics_iteration_id, 0.02),
            ),
        ),
    )

    metrics = publisher.published[0]
    assert metrics.scheduled_requests.num_prefill_requests == 1
    assert metrics.scheduled_requests.sum_prefill_kv_tokens == 4
    assert metrics.queued_requests.num_decode_requests == 2
    assert metrics.queued_requests.sum_decode_kv_tokens == 150


def test_emitter_suppresses_iterations_and_bounds_pending_state():
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter(
        "worker", 0, publisher, max_pending_iterations=1
    )
    scheduler = _FakeScheduler()
    suppressed = _make_scheduler_output("prefill", prompt_tokens=8)
    scheduler.emit = False
    emitter.begin_iteration(scheduler, suppressed)
    assert suppressed.forward_pass_metrics_iteration_id is None
    assert not emitter.has_pending_timing()

    scheduler.emit = True
    first = _make_scheduler_output("prefill", prompt_tokens=8)
    second = _make_scheduler_output("decode", computed_tokens=64)
    emitter.begin_iteration(scheduler, first)
    emitter.begin_iteration(scheduler, second)
    emitter.complete_timing_samples(((first.forward_pass_metrics_iteration_id, 0.01),))
    assert publisher.published == []

    emitter.complete_iteration(
        scheduler,
        second,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            forward_pass_timing_samples=(
                (second.forward_pass_metrics_iteration_id, 0.02),
            ),
        ),
    )
    assert len(publisher.published) == 1


def test_engine_idle_flush_waits_for_and_completes_final_timing():
    core = EngineCore.__new__(EngineCore)
    core.forward_pass_metrics_emitter = emitter = MagicMock()
    core.model_executor = executor = MagicMock()
    emitter.has_pending_timing.return_value = True
    executor.drain_forward_pass_timing.return_value = ((11, 0.03),)

    EngineCore._flush_forward_pass_metrics(core)

    executor.drain_forward_pass_timing.assert_called_once_with(wait=True)
    emitter.complete_timing_samples.assert_called_once_with(((11, 0.03),))


def test_zmq_publisher_frames_payload_and_stops_cleanly():
    port = get_open_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(endpoint)
    publisher = ZmqForwardPassMetricsPublisher(endpoint, "worker", 0, 8)
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            publisher.publish(ForwardPassMetrics(worker_id="worker", wall_time=0.01))
            if subscriber.poll(50, zmq.POLLIN):
                break
        else:
            pytest.fail("timed out waiting for forward-pass metrics")
        topic, sequence, payload = subscriber.recv_multipart()
        metrics = decode_forward_pass_metrics(payload)
        assert topic == b""
        assert int.from_bytes(sequence, "big") == metrics.counter_id
        assert metrics.worker_id == "worker"
        assert metrics.wall_time == 0.01
    finally:
        publisher.shutdown()
        subscriber.close(linger=0)
    assert not publisher._thread.is_alive()


def test_zmq_publisher_emits_idle_heartbeat(monkeypatch):
    monkeypatch.setattr(fpm_module, "FPM_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    endpoint = f"tcp://127.0.0.1:{get_open_port()}"
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(endpoint)
    publisher = ZmqForwardPassMetricsPublisher(endpoint, "worker", 2, 8)
    try:
        assert subscriber.poll(2_000, zmq.POLLIN)
        _, _, payload = subscriber.recv_multipart()
        metrics = decode_forward_pass_metrics(payload)
        assert metrics.worker_id == "worker"
        assert metrics.dp_rank == 2
        assert metrics.wall_time == 0.0
    finally:
        publisher.shutdown()
        subscriber.close(linger=0)


def test_zmq_publisher_reports_bind_failure_from_its_thread():
    endpoint = f"tcp://127.0.0.1:{get_open_port()}"
    blocker = zmq.Context.instance().socket(zmq.PUB)
    blocker.bind(endpoint)
    try:
        with pytest.raises(RuntimeError, match="Failed to bind"):
            ZmqForwardPassMetricsPublisher(endpoint, "worker", 0, 8)
    finally:
        blocker.close(linger=0)
