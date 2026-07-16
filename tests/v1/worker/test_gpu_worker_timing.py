# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator

import numpy as np
import pytest

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.metrics.timing import (
    ModelRunnerTiming,
    drain_timing_samples,
)

pytestmark = pytest.mark.cpu_test


class FakeEvent:
    def __init__(self, timestamps: Iterator[float]) -> None:
        self.timestamps = timestamps
        self.timestamp = 0.0
        self.complete = False
        self.elapsed_calls = 0

    def record(self, stream=None) -> None:
        self.timestamp = next(self.timestamps)
        self.complete = False

    def query(self) -> bool:
        return self.complete

    def elapsed_time(self, end_event: "FakeEvent") -> float:
        self.elapsed_calls += 1
        return end_event.timestamp - self.timestamp


class FakeCopyEvent:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


class FakeSchedulerOutput:
    def __init__(self, total_num_scheduled_tokens: int = 1) -> None:
        self.total_num_scheduled_tokens = total_num_scheduled_tokens


class FakeRunner:
    def __init__(self, worker_timing: ModelRunnerTiming) -> None:
        self.worker_timing = worker_timing

    @drain_timing_samples
    def execute_model(self, output: ModelRunnerOutput) -> ModelRunnerOutput:
        return output


def make_collector(timestamps: list[float]):
    timestamp_iter = iter(timestamps)
    events: list[FakeEvent] = []

    def event_factory() -> FakeEvent:
        event = FakeEvent(timestamp_iter)
        events.append(event)
        return event

    return ModelRunnerTiming(None, event_factory), events  # type: ignore[arg-type]


def make_batch(is_prefilling: list[bool]) -> object:
    num_reqs = len(is_prefilling)
    return type(
        "Batch",
        (),
        {
            "is_prefilling_np": np.array(is_prefilling),
            "num_scheduled_tokens": np.array([4] + [1] * (num_reqs - 1)),
            "num_tokens_after_padding": 8,
            "num_reqs": num_reqs,
        },
    )()


def make_async_output(
    output: ModelRunnerOutput, timing: ModelRunnerTiming
) -> tuple[AsyncOutput, FakeCopyEvent]:
    async_output = AsyncOutput.__new__(AsyncOutput)
    async_output.model_runner_output = output
    copy_event = FakeCopyEvent()
    async_output.copy_event = copy_event  # type: ignore[assignment]
    async_output.sampled_token_ids = np.empty((0, 0), dtype=np.int64)
    async_output.num_sampled_tokens_np = np.empty(0, dtype=np.int64)
    async_output.num_nans = None
    async_output.logprobs_tensors = None
    async_output.prompt_logprobs_dict = {}
    async_output.timing = timing
    return async_output, copy_event


def finish_model_output(
    timing: ModelRunnerTiming, output: ModelRunnerOutput
) -> ModelRunnerOutput:
    result = timing.finish_output(output)
    assert isinstance(result, ModelRunnerOutput)
    return result


@pytest.mark.parametrize(("dummy_run", "num_scheduled_tokens"), [(True, 1), (False, 0)])
def test_start_step_ignores_non_model_batches(
    dummy_run: bool, num_scheduled_tokens: int
) -> None:
    timing, events = make_collector([])

    timing.start_step(
        dummy_run,
        FakeSchedulerOutput(num_scheduled_tokens),
    )  # type: ignore[arg-type]

    assert not timing.is_active
    assert events == []


def test_suspended_timing_excludes_warmup_without_advancing_iteration() -> None:
    timing, events = make_collector([0.0, 5.0])
    scheduler_output = FakeSchedulerOutput()

    with timing.suspend():
        timing.start_step(False, scheduler_output)  # type: ignore[arg-type]
        timing.set_step_metadata(make_batch([False]))
        timing.start_proposer()
        timing.end_proposer()
        output = finish_model_output(
            timing, ModelRunnerOutput(req_ids=[], req_id_to_index={})
        )

    assert not timing.is_active
    assert events == []
    assert output.worker_timing_samples == []

    timing.start_step(False, scheduler_output)  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([False]))
    output = finish_model_output(
        timing, ModelRunnerOutput(req_ids=[], req_id_to_index={})
    )
    events[-1].complete = True
    output = timing.drain_into(output)

    assert output.worker_timing_samples[0].iteration_index == 0


def test_execute_model_decorator_drains_terminal_output() -> None:
    timing, events = make_collector([0.0, 5.0, 10.0])
    scheduler_output = FakeSchedulerOutput()
    timing.start_step(False, scheduler_output)  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([False]))
    finish_model_output(
        timing,
        ModelRunnerOutput(req_ids=[], req_id_to_index={}),
    )
    events[-1].complete = True
    timing.start_step(False, scheduler_output)  # type: ignore[arg-type]

    output = FakeRunner(timing).execute_model(
        ModelRunnerOutput(req_ids=[], req_id_to_index={})
    )

    assert not timing.is_active
    assert len(output.worker_timing_samples) == 1


def test_speculative_step_is_drained_without_synchronizing() -> None:
    timing, events = make_collector([0.0, 6.0, 8.0, 10.0])
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    timing.start_step(False, FakeSchedulerOutput())  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([True, False, False]))
    timing.start_proposer()
    timing.end_proposer()
    output = finish_model_output(timing, output)

    assert output.worker_timing_samples == []
    assert sum(event.elapsed_calls for event in events) == 0

    events[-1].complete = True
    output = timing.drain_into(output)

    assert len(output.worker_timing_samples) == 1
    stat = output.worker_timing_samples[0]
    assert stat.phase == "prefill"
    assert stat.num_model_tokens == 8
    assert stat.num_requests == 3
    assert stat.num_prefill_requests == 1
    assert stat.num_prefill_tokens == 4
    assert stat.num_decode_requests == 2
    assert stat.num_decode_tokens == 2
    assert stat.num_requests == stat.num_prefill_requests + stat.num_decode_requests
    assert stat.num_model_tokens > stat.num_prefill_tokens + stat.num_decode_tokens
    assert stat.model_time_seconds == pytest.approx(0.008)
    assert stat.proposer_time_seconds == pytest.approx(0.002)
    assert stat.total_time_seconds == pytest.approx(0.010)
    assert stat.total_time_seconds == pytest.approx(
        stat.model_time_seconds + stat.proposer_time_seconds
    )

    model_aggregate = timing.model_times[("prefill", 8)]
    proposer_aggregate = timing.proposer_times[("prefill", 3)]
    assert model_aggregate.count == 1
    assert model_aggregate.mean_seconds == pytest.approx(0.008)
    assert proposer_aggregate.total_seconds == pytest.approx(0.002)


def test_pure_spec_decode_verification_is_decode() -> None:
    timing, events = make_collector([0.0, 5.0])

    timing.start_step(False, FakeSchedulerOutput())  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([False, False, False]))
    output = finish_model_output(timing, EMPTY_MODEL_RUNNER_OUTPUT)
    events[-1].complete = True
    output = timing.drain_into(output)

    assert output is not EMPTY_MODEL_RUNNER_OUTPUT
    assert EMPTY_MODEL_RUNNER_OUTPUT.worker_timing_samples == []
    stat = output.worker_timing_samples[0]
    assert stat.phase == "decode"
    assert stat.num_prefill_requests == 0
    assert stat.num_decode_requests == 3
    assert stat.model_time_seconds == pytest.approx(0.005)
    assert stat.proposer_time_seconds is None
    assert stat.total_time_seconds == stat.model_time_seconds


def test_async_output_drains_timing_after_copy() -> None:
    timing, events = make_collector([0.0, 5.0])
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    timing.start_step(False, FakeSchedulerOutput())  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([False]))
    async_output, copy_event = make_async_output(output, timing)
    result = timing.finish_output(async_output)
    assert result is async_output
    events[-1].complete = True

    output = async_output.get_output()

    assert copy_event.synchronized
    assert len(output.worker_timing_samples) == 1
    assert output.worker_timing_samples[0].iteration_index == 0


def test_worker_timing_samples_round_trip_through_msgpack() -> None:
    timing, events = make_collector([0.0, 5.0])
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    timing.start_step(False, FakeSchedulerOutput())  # type: ignore[arg-type]
    timing.set_step_metadata(make_batch([False]))
    output = finish_model_output(timing, output)
    events[-1].complete = True
    output = timing.drain_into(output)

    encoded = MsgpackEncoder().encode(output)
    decoded = MsgpackDecoder(ModelRunnerOutput).decode(encoded)

    assert decoded.worker_timing_samples == output.worker_timing_samples
