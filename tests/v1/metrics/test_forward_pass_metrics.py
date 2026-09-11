# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import msgspec
import pytest
import zmq

import vllm.v1.metrics.forward_pass_metrics as fpm_module
from vllm.config import DeviceConfig, ObservabilityConfig, VllmConfig
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.metrics.forward_pass_metrics import (
    FPM_TIMING_SCOPE_MODEL_STEP_CUDA,
    ForwardPassMetrics,
    ForwardPassMetricsEmitter,
    ForwardPassMetricsTimer,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    ZmqForwardPassMetricsPublisher,
    decode_forward_pass_metrics,
    encode_forward_pass_metrics,
    make_forward_pass_metrics_timer,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("count", [2, 10_000, 2**64 - 1])
def test_subscriber_sample_fits_two_80_column_lines(count, capsys):
    from examples.features.forward_pass_metrics.forward_pass_metrics_subscriber import (
        print_metrics,
    )

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
    for seq in range(2, 102):
        display.update(seq, metrics)
    updates = capsys.readouterr().out
    assert len(updates.splitlines()) == (100 if interval == 0 else 0)
    if interval:
        assert display.poll_timeout_ms() == 1000
        clock[0] = 1.0
        assert display.poll_timeout_ms() == 0
        display.flush()  # The last sample is visible even without more messages.
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1 and lines[0].lstrip().startswith("101 ")
    assert display.poll_timeout_ms() == subscriber.POLL_TIMEOUT_MS
    clock[0] = 2.0
    display.update(102, msgspec.structs.replace(metrics, wall_time=0.0))
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


# Golden wire payload for the model-step timing scope.
_GOLDEN_FPM_V1 = {
    "version": 1,
    "worker_id": "worker-0",
    "dp_rank": 2,
    "counter_id": 3,
    "timing_scope": "model_step_cuda",
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

    def get_forward_pass_metrics_request_state(self):
        return self.requests, self.waiting, self.skipped_waiting


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


def test_forward_pass_metrics_v1_wire_contract():
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

    assert raw == _GOLDEN_FPM_V1
    assert decode_forward_pass_metrics(payload) == metrics

    unsupported_payload = encode_forward_pass_metrics(
        msgspec.structs.replace(metrics, version=2)
    )
    assert decode_forward_pass_metrics(unsupported_payload) is None


def test_timer_hot_drain_never_synchronizes_pending_event():
    timestamps = iter((10.0, 15.0, 20.0, 27.0))
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
    assert len(events) == 2

    timer.start(scheduler_output)
    timer.finish()
    with timer._drain_lock:
        assert timer.drain_samples() == ()
    with timer._pending_lock:
        assert timer.drain_samples() == ()
    assert timer.drain_samples() == ((4, 0.007),)
    assert sum(event.synchronize_calls for event in events) == 0
    assert len(events) == 2  # Completed event pair is recycled.


def test_sampling_finishes_existing_timing_after_draft_without_waiting():
    from vllm.v1.worker.gpu_worker import Worker

    events = []
    timestamps = iter((10.0, 25.0))

    def event_factory():
        event = _FakeTimingEvent(timestamps)
        events.append(event)
        return event

    timer = ForwardPassMetricsTimer(1, event_factory=event_factory)
    scheduled = _make_scheduler_output("decode", computed_tokens=100)
    scheduled.forward_pass_metrics_iteration_id = 3
    timer.start(scheduled)
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def sample_and_draft(grammar):
        # The output D2H may already be complete, while the draft is not.
        assert events[1].timestamp == 0
        assert timer.drain_into(output).forward_pass_timing_samples == ()
        return output

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            forward_pass_metrics_timer=timer, sample_tokens=sample_and_draft
        )
    )
    returned = Worker.sample_tokens(worker, None)
    assert returned.forward_pass_timing_samples == ()
    assert len(events) == 2
    assert events[1].timestamp == 25
    assert sum(event.synchronize_calls for event in events) == 0
    events[1].complete = True
    assert timer.drain_into(output).forward_pass_timing_samples == ((3, 0.015),)


def test_timer_is_disabled_off_and_on_non_output_ranks():
    disabled = SimpleNamespace(
        observability_config=SimpleNamespace(forward_pass_metrics_port=0)
    )
    enabled = SimpleNamespace(
        observability_config=SimpleNamespace(forward_pass_metrics_port=20380)
    )

    assert make_forward_pass_metrics_timer(disabled, is_output_rank=True) is None
    assert make_forward_pass_metrics_timer(enabled, is_output_rank=False) is None


def test_idle_poll_retries_without_new_outputs_and_keeps_one_rpc(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(fpm_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter("worker", 0, publisher)
    scheduler = _FakeScheduler()
    output = _make_scheduler_output("decode", computed_tokens=64)
    emitter.begin_iteration(scheduler, output)
    emitter.complete_iteration(scheduler, output, ModelRunnerOutput([], {}))
    response: list[tuple[tuple[int, float], ...] | None] = [None]
    calls = []

    def start_poll():
        calls.append(clock[0])
        return lambda: response[0]

    executor = SimpleNamespace(start_forward_pass_timing_poll=start_poll)
    emitter.poll_timing(executor)
    for _ in range(3):
        emitter.poll_timing(executor)
    assert len(calls) == 1  # Outstanding response: do not issue duplicate RPCs.
    response[0] = ()  # GPU event was not ready when the worker queried it.
    emitter.poll_timing(executor)
    assert len(calls) == 1  # Rate limit applies even to empty responses.
    clock[0] += fpm_module.FPM_IDLE_POLL_INTERVAL_SECONDS
    response[0] = ((output.forward_pass_metrics_iteration_id, 0.012),)
    emitter.poll_timing(executor)
    assert len(calls) == 2
    assert len(publisher.published) == 1
    assert publisher.published[0].wall_time == 0.012
    assert not emitter.has_pending_timing()
    emitter.poll_timing(executor)
    assert len(calls) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_idle_input_arrival_does_not_wait_for_timing(enabled):
    from vllm.v1.engine.core import EngineCoreProc

    handled: list[tuple[str, str]] = []
    kwargs_seen = []
    queue_calls = []

    def receive(**kwargs):
        kwargs_seen.append(kwargs)
        return ("request", "new")

    def start_poll():
        queue_calls.append("poll")
        return lambda: None  # The RPC never becomes ready in this test.

    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter("worker", 0, publisher) if enabled else None
    if emitter is not None:
        emitter.begin_iteration(_FakeScheduler(), _make_scheduler_output("decode"))
    core = SimpleNamespace(
        has_work=lambda: bool(handled),
        is_running=lambda: True,
        forward_pass_metrics_emitter=emitter,
        model_executor=SimpleNamespace(start_forward_pass_timing_poll=start_poll),
        input_queue=SimpleNamespace(empty=lambda: True, get=receive),
        process_input_queue_block=True,
        aborts_queue=SimpleNamespace(mutex=nullcontext(), queue=[]),
        _notify_idle_state_callbacks=lambda: None,
        _handle_client_request=lambda *request: handled.append(request),
    )
    EngineCoreProc._process_input_queue(core)
    assert handled == [("request", "new")]
    assert kwargs_seen == (
        [{"timeout": fpm_module.FPM_IDLE_POLL_INTERVAL_SECONDS}]
        if enabled
        else [{"block": True}]
    )
    assert queue_calls == (["poll"] if enabled else [])


@pytest.mark.parametrize("model_consumes_first", [False, True])
def test_timing_rpc_poll_preserves_model_response_order(model_consumes_first):
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

    executor = object.__new__(MultiprocExecutor)
    sent: list = []
    responses: deque = deque()
    writable = [False]

    def receive(**kwargs):
        assert responses, "Would block waiting for an RPC response"
        return WorkerProc.ResponseStatus.SUCCESS, responses.popleft()

    executor.is_failed = False
    executor.output_rank = 0
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(
        can_enqueue=lambda: writable[0], enqueue=sent.append
    )
    executor.response_mqs = [
        SimpleNamespace(can_dequeue=lambda: bool(responses), dequeue=receive)
    ]
    assert executor.start_forward_pass_timing_poll() is None
    assert not sent
    writable[0] = True
    poll = executor.start_forward_pass_timing_poll()
    assert poll is not None and poll() is None
    assert executor.start_forward_pass_timing_poll() is None
    assert len(sent) == 1
    later = executor.collective_rpc(
        "execute_model", non_block=True, unique_reply_rank=0
    )
    responses.extend([((7, 0.01),), "model-output"])
    if model_consumes_first:
        assert later.result() == "model-output"
        assert poll() == ((7, 0.01),)
    else:
        assert poll() == ((7, 0.01),)
        assert later.result() == "model-output"
    assert not responses
    assert not executor.futures_queue


def test_ray_timing_poll_checks_refs_before_get(monkeypatch):
    from vllm.v1.executor import ray_executor

    ready = [False]
    reads = []

    def wait(refs, *, num_returns, timeout):
        assert timeout == 0 and num_returns == len(refs)
        return (refs, []) if ready[0] else ([], refs)

    def result():
        reads.append(True)
        return [((3, 0.01),)]

    monkeypatch.setattr(ray_executor, "ray", SimpleNamespace(wait=wait))
    future = SimpleNamespace(ref_or_refs=["ref"], result=result)
    executor = SimpleNamespace(collective_rpc=lambda *a, **k: future)
    poll = ray_executor.RayDistributedExecutor.start_forward_pass_timing_poll(executor)
    assert poll() is None and not reads
    ready[0] = True
    assert poll() == ((3, 0.01),)
    assert reads == [True]


def test_shutdown_timing_grace_is_bounded(monkeypatch):
    from vllm.v1.engine import core as core_module

    times = iter((0.0, 0.5, 1.1))
    polls = []
    monkeypatch.setattr(
        core_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(times),
            sleep=lambda _: None,
        ),
    )
    core = SimpleNamespace(
        forward_pass_metrics_emitter=SimpleNamespace(
            has_pending_timing=lambda: True,
            poll_timing=lambda _: polls.append(True),
        ),
        model_executor=None,
    )
    core_module.EngineCore._flush_forward_pass_metrics(core)
    assert len(polls) == 2


def test_fpm_requires_model_config():
    with pytest.raises(ValueError, match="support generative models only"):
        VllmConfig(
            device_config=DeviceConfig(device="cuda"),
            observability_config=ObservabilityConfig(forward_pass_metrics_port=20380),
        )


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
    assert metrics.timing_scope == FPM_TIMING_SCOPE_MODEL_STEP_CUDA
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
    # Odd query sizes exercise half-token attention lengths, not prompt lengths.
    new = _make_scheduler_output("new", prompt_tokens=3, computed_tokens=2)
    new.scheduled_new_reqs[0].prompt_token_ids = [0] * 1000
    output.scheduled_new_reqs = new.scheduled_new_reqs
    output.num_scheduled_tokens["new"] = 3
    output.total_num_scheduled_tokens = 4

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
    assert metrics.scheduled_requests.num_prefill_requests == 2
    assert metrics.scheduled_requests.sum_prefill_tokens == 4
    assert metrics.scheduled_requests.sum_prefill_kv_tokens == 6
    # Attention lengths are 4 + 1/2 = 4.5 and 2 + 3/2 = 3.5; Var = 0.25.
    assert metrics.scheduled_requests.var_prefill_length == 0.25
    assert metrics.queued_requests.num_decode_requests == 2
    assert metrics.queued_requests.sum_decode_kv_tokens == 150


def test_async_sd_rejections_correct_only_later_iteration_lengths():
    scheduler = _FakeScheduler()
    scheduler.requests = {
        rid: SimpleNamespace(num_prompt_tokens=10, num_preemptions=0)
        for rid in ("a", "b")
    }
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter(
        "worker", 0, publisher, correct_async_spec_lengths=True
    )

    def schedule(lengths):
        out = SchedulerOutput.make_empty()
        out.num_scheduled_tokens = {"a": 4, "b": 4}
        out.total_num_scheduled_tokens = 8
        out.scheduled_spec_decode_tokens = {"a": [-1] * 3, "b": [-1] * 3}
        out.scheduled_cached_reqs = CachedRequestData(
            req_ids=["a", "b"],
            resumed_req_ids=set(),
            new_token_ids=[[], []],
            all_token_ids={},
            new_block_ids=[None, None],
            num_computed_tokens=lengths,
            num_output_tokens=[1, 1],
        )
        emitter.begin_iteration(scheduler, out)
        return out

    first, second = schedule([100, 200]), schedule([104, 204])
    result = ModelRunnerOutput(
        req_ids=["a", "b"],
        req_id_to_index={"a": 0, "b": 1},
        sampled_token_ids=[[1], [2, 3, 4]],
        forward_pass_timing_samples=((first.forward_pass_metrics_iteration_id, 0.01),),
    )
    emitter.before_update(first, result)
    # Scheduler output processing can truncate tokens at a stop/max-length limit.
    result.sampled_token_ids[1] = [2]
    emitter.complete_iteration(scheduler, first, result)
    assert publisher.published[0].scheduled_requests.sum_decode_kv_tokens == 300
    third = schedule([105, 207])  # first settled; second still optimistic
    result.sampled_token_ids = [[5], [6, 7, 8, 9]]
    result.forward_pass_timing_samples = (
        (second.forward_pass_metrics_iteration_id, 0.02),
    )
    emitter.before_update(second, result)
    emitter.complete_iteration(scheduler, second, result)
    metrics = publisher.published[1].scheduled_requests
    assert metrics.sum_decode_kv_tokens == 304  # 101 + 203
    assert metrics.var_decode_kv_tokens == 2601
    result.forward_pass_timing_samples = (
        (third.forward_pass_metrics_iteration_id, 0.03),
    )
    emitter.complete_iteration(scheduler, third, result)
    assert publisher.published[2].scheduled_requests.sum_decode_kv_tokens == 309


@pytest.mark.parametrize("preempted", [True, False])
def test_async_sd_rejection_does_not_cross_request_generation(preempted):
    scheduler = _FakeScheduler()
    req = SimpleNamespace(num_prompt_tokens=10, num_preemptions=0)
    scheduler.requests = {"decode": req}
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter(
        "worker", 0, publisher, correct_async_spec_lengths=True
    )
    first = _make_scheduler_output("decode", computed_tokens=100)
    first.scheduled_spec_decode_tokens = {"decode": [-1] * 3}
    emitter.begin_iteration(scheduler, first)
    if preempted:
        req.num_preemptions += 1
    else:
        scheduler.requests["decode"] = SimpleNamespace(
            num_prompt_tokens=10, num_preemptions=0
        )
    resumed = _make_scheduler_output("decode", computed_tokens=20)
    emitter.begin_iteration(scheduler, resumed)
    result = ModelRunnerOutput(
        req_ids=["decode"],
        req_id_to_index={"decode": 0},
        sampled_token_ids=[[1]],
        forward_pass_timing_samples=((first.forward_pass_metrics_iteration_id, 0.01),),
    )
    emitter.before_update(first, result)
    emitter.complete_iteration(scheduler, first, result)
    scheduler.requests.clear()  # removed requests must not break the frozen snapshot
    result.forward_pass_timing_samples = (
        (resumed.forward_pass_metrics_iteration_id, 0.02),
    )
    emitter.complete_iteration(scheduler, resumed, result)
    assert publisher.published[-1].scheduled_requests.sum_decode_kv_tokens == 20


def test_async_sd_correction_survives_dropped_sample():
    scheduler = _FakeScheduler()
    scheduler.requests["decode"].num_preemptions = 0
    publisher = _FakeMetricsPublisher()
    emitter = ForwardPassMetricsEmitter(
        "worker",
        0,
        publisher,
        max_pending_iterations=1,
        correct_async_spec_lengths=True,
    )
    first = _make_scheduler_output("decode", computed_tokens=100)
    first.scheduled_spec_decode_tokens = {"decode": [-1] * 3}
    emitter.begin_iteration(scheduler, first)
    second = _make_scheduler_output("decode", computed_tokens=104)
    emitter.begin_iteration(scheduler, second)
    result = ModelRunnerOutput(
        req_ids=["decode"],
        req_id_to_index={"decode": 0},
        sampled_token_ids=[[1]],
    )
    emitter.before_update(first, result)
    emitter.complete_iteration(scheduler, first, result)
    result.forward_pass_timing_samples = (
        (second.forward_pass_metrics_iteration_id, 0.01),
    )
    emitter.before_update(second, result)
    emitter.complete_iteration(scheduler, second, result)
    assert publisher.published[-1].scheduled_requests.sum_decode_kv_tokens == 101
    assert not emitter._spec_requests


def test_zmq_publisher_frames_payload_and_stops_cleanly():
    port = get_open_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    context = zmq.Context.instance()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(endpoint)
    publisher = ZmqForwardPassMetricsPublisher(endpoint, "worker", 0)
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
    publisher = ZmqForwardPassMetricsPublisher(endpoint, "worker", 2)
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
            ZmqForwardPassMetricsPublisher(endpoint, "worker", 0)
    finally:
        blocker.close(linger=0)
