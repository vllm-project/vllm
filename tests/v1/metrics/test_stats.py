# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import (
    CachedRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.engine import EngineCoreOutputs, FinishReason
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.metrics.iteration import IterationMetricsCollector
from vllm.v1.metrics.iteration_workload import (
    IterationWorkloadAccumulator,
    QueuedIterationWorkload,
    ScheduledIterationWorkload,
)
from vllm.v1.metrics.stats import (
    IterationStats,
    LoRARequestStates,
    PrefillStats,
    PrefixCacheStats,
    PromptTokenStats,
    RequestStateStats,
    SchedulerIterationDetails,
    SchedulerStats,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.utils import compute_iteration_details


@pytest.mark.parametrize("blocker", [None, "global", "timing", "arrival"])
def test_idle_iteration_records_require_quiescence_and_drained_timing(blocker):
    from vllm.v1.engine.core import EngineCoreProc

    scheduler = SimpleNamespace(
        requests={}, waiting=[], kv_holding_waiting=[], make_stats=SchedulerStats
    )
    collector = IterationMetricsCollector(scheduler)
    if blocker == "timing":
        collector.begin_dummy_iteration()

    class InputQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            if not self.empty():
                return super().get(block=False)
            return ("stop",)

    running = [True]
    inputs = InputQueue()

    def poll():
        if blocker == "arrival":
            inputs.put(("stop",))
        return {}

    outputs: queue.Queue[tuple[int, EngineCoreOutputs]] = queue.Queue()
    published: list[SchedulerIterationDetails] = []
    core = SimpleNamespace(
        iteration_metrics=collector,
        forward_pass_metrics_exporter=SimpleNamespace(record=published.extend),
        scheduler=scheduler,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(enable_iteration_workload_metrics=True)
        ),
        input_queue=inputs,
        output_queue=outputs,
        aborts_queue=queue.Queue(),
        process_input_queue_block=True,
        has_work=lambda: blocker == "global",
        is_running=lambda: running[0],
        _poll_iteration_metrics=poll,
        _notify_idle_state_callbacks=lambda: None,
        _handle_client_request=lambda *_: running.__setitem__(0, False),
    )
    EngineCoreProc._process_input_queue(core)
    if blocker is not None:
        assert outputs.empty()
        assert not published
    else:
        owner, output = outputs.get_nowait()
        assert owner == 0
        assert output.scheduler_stats is not None
        (record,) = output.scheduler_stats.iteration_records
        assert record.is_idle and not record.is_dummy
        assert record.elapsed_ms == 0
        assert published == [record]


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    assert repr(iteration_stats).startswith("IterationStats(")


def test_idle_iteration_output_preserves_control_and_mixed_records():
    idle = SchedulerIterationDetails(0, 0, 0, 0, 0, elapsed_ms=0, is_idle=True)
    output = EngineCoreOutputs(scheduler_stats=SchedulerStats(iteration_records=[idle]))
    assert output.is_idle_iteration_output()
    output.start_wave = 0
    assert not output.is_idle_iteration_output()
    output.start_wave = None
    assert output.scheduler_stats is not None
    output.scheduler_stats.iteration_records.append(
        SchedulerIterationDetails(1, 1, 8, 0, 0, elapsed_ms=1)
    )
    assert not output.is_idle_iteration_output()


def test_scheduler_iteration_details_serialization():
    iteration_details = SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
        num_encoder_inputs=2,
        num_encoder_output_tokens=392,
        scheduled_workload=ScheduledIterationWorkload(sum_prefill_kv_tokens=17),
        queued_workload=QueuedIterationWorkload(num_prefill_requests=2),
    )
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            kv_cache_usage=0.5,
            iteration_details=iteration_details,
            iteration_records=[iteration_details],
        )
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.kv_cache_usage == 0.5
    assert decoded.scheduler_stats.iteration_details == iteration_details
    assert decoded.scheduler_stats.iteration_records == [iteration_details]


@pytest.mark.parametrize("timing_first", [False, True])
def test_iteration_timing_keeps_original_batch_when_results_arrive_late(timing_first):
    collector = IterationMetricsCollector()
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"prefill": 8, "decode": 1}
    output.total_num_scheduled_tokens = 9
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["prefill", "decode"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[4, 8],
        num_output_tokens=[0, 1],
    )
    output.scheduled_encoder_input_stats = ScheduledEncoderInputStats(
        num_inputs=2, output_tokens=392
    )
    collector.begin_iteration(output)
    index = output.iteration_metrics_id
    assert index is not None

    output.num_scheduled_tokens.update(prefill=99, decode=7)
    output.scheduled_encoder_input_stats.num_inputs = 9
    output.scheduled_encoder_input_stats.output_tokens = 999
    model_output = ModelRunnerOutput([], {})
    samples = ((index, 12.0),)
    if timing_first:
        collector.complete_timing_samples(samples)
    else:
        collector.complete_iteration(output, model_output)
    assert not collector.take_ready()
    assert collector.has_pending_timing()
    if timing_first:
        collector.complete_iteration(output, model_output)
    else:
        collector.complete_timing_samples(samples)
    (record,) = collector.take_ready()
    assert record.num_ctx_requests == 1
    assert record.num_ctx_tokens == 8
    assert record.num_generation_requests == 1
    assert record.num_generation_tokens == 1
    assert record.num_encoder_inputs == 2
    assert record.num_encoder_output_tokens == 392
    assert record.elapsed_ms == 12
    assert record.timing_scope == "model_step_cuda"
    assert not collector.has_pending_timing()


def test_dummy_tail_timings_survive_more_than_batch_queue_depth():
    collector = IterationMetricsCollector()
    indices = [collector.begin_dummy_iteration() for _ in range(64)]
    assert collector.has_pending_timing()
    assert not collector.take_ready()
    output = SchedulerOutput.make_empty()
    collector.complete_iteration(
        output,
        ModelRunnerOutput(
            [], {}, iteration_timing_samples=tuple((index, 10.0) for index in indices)
        ),
    )
    records = collector.take_ready()
    assert [record.iteration_index for record in records] == indices
    assert all(record.is_dummy and record.elapsed_ms == 10 for record in records)
    assert all(
        record.num_ctx_tokens == record.num_generation_tokens == 0 for record in records
    )
    assert not collector.has_pending_timing()


@pytest.mark.parametrize("dummy", [False, True])
def test_completed_idle_timing_poll_is_consumed_when_inference_resumes(dummy):
    collector = IterationMetricsCollector()
    old_id = collector.begin_dummy_iteration()
    submitted = []

    def start_poll():
        submitted.append(True)
        return lambda: ((old_id, 10.0),)

    collector.poll_timing(SimpleNamespace(start_iteration_timing_poll=start_poll))
    if dummy:
        iteration_id = collector.begin_dummy_iteration()
        collector.complete_timing_samples(((iteration_id, 20.0),))
    else:
        output = SchedulerOutput.make_empty()
        output.num_scheduled_tokens = {"request": 1}
        output.total_num_scheduled_tokens = 1
        output.scheduled_cached_reqs = CachedRequestData(
            req_ids=["request"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[8],
            num_output_tokens=[1],
        )
        collector.begin_iteration(output)
        iteration_id = output.iteration_metrics_id
        collector.complete_iteration(
            output,
            ModelRunnerOutput([], {}, iteration_timing_samples=((iteration_id, 20.0),)),
        )
    records = collector.take_ready()
    assert [record.iteration_index for record in records] == [
        old_id,
        iteration_id,
    ]
    assert [record.elapsed_ms for record in records] == [10, 20]
    assert not collector.has_pending_timing()
    assert submitted == [True]  # No new timing RPC on the active inference path.


@pytest.mark.parametrize("stats_owner", [None, 0, 3])
def test_iteration_records_preserve_scheduler_stats_owner(stats_owner):
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.output_processor import OutputProcessor

    collector = IterationMetricsCollector()
    index = collector.begin_dummy_iteration()
    collector.complete_timing_samples(((index, 10.0),))
    stats = SchedulerStats(num_running_reqs=7, kv_cache_usage=0.5)
    outputs = {}
    if stats_owner is not None:
        outputs[0] = EngineCoreOutputs()
        outputs[stats_owner] = EngineCoreOutputs(scheduler_stats=stats)
    published: list[SchedulerIterationDetails] = []
    core = SimpleNamespace(
        iteration_metrics=collector,
        scheduler=SimpleNamespace(make_stats=lambda: stats),
        forward_pass_metrics_exporter=SimpleNamespace(record=published.extend),
    )
    EngineCore._attach_iteration_records(core, outputs)
    owner = 0 if stats_owner is None else stats_owner
    assert outputs[owner].scheduler_stats is stats
    assert sum(output.scheduler_stats is not None for output in outputs.values()) == 1
    assert stats.num_running_reqs == 7
    assert [record.iteration_index for record in stats.iteration_records] == [index]
    EngineCore._attach_iteration_records(core, outputs)
    assert len(stats.iteration_records) == 1
    assert published == stats.iteration_records

    frontends = {
        i: SimpleNamespace(lora_states=LoRARequestStates(log_stats=True))
        for i in outputs
    }
    frontends[owner].lora_states.request_running("running", "adapter-A")
    frontends[owner].lora_states.request_waiting("waiting", "adapter-B")
    for i, output in outputs.items():
        OutputProcessor.update_scheduler_stats(frontends[i], output.scheduler_stats)
    assert stats.running_lora_adapters == {"adapter-A": 1, "adapter-B": 0}
    assert stats.waiting_lora_adapters == {"adapter-A": 0, "adapter-B": 1}


def _decode_workload_output(lengths: dict[str, int]) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = dict.fromkeys(lengths, 4)
    output.total_num_scheduled_tokens = len(lengths) * 4
    output.scheduled_spec_decode_tokens = {rid: [-1] * 3 for rid in lengths}
    cached = output.scheduled_cached_reqs
    cached.req_ids = list(lengths)
    cached.num_computed_tokens = list(lengths.values())
    cached.num_output_tokens = [1] * len(lengths)
    cached.prefill_req_ids = set()
    workload = IterationWorkloadAccumulator()
    for length in lengths.values():
        workload.add_decode(length, 4)
    output.scheduled_workload = workload.snapshot()
    return output


def test_async_spec_rejections_correct_later_workload_before_stop_truncation():
    scheduler = SimpleNamespace(
        requests={rid: SimpleNamespace(num_preemptions=0) for rid in ("a", "b")},
        waiting=[],
        kv_holding_waiting=[],
    )
    collector = IterationMetricsCollector(scheduler, correct_async_spec_lengths=True)
    first = _decode_workload_output({"a": 100, "b": 200})
    second = _decode_workload_output({"a": 104, "b": 204})
    collector.begin_iteration(first)
    collector.begin_iteration(second)
    result = ModelRunnerOutput(
        ["a", "b"],
        {"a": 0, "b": 1},
        [[1], [2, 3, 4]],
        iteration_timing_samples=((first.iteration_metrics_id, 10.0),),
    )
    collector.before_update(first, result)
    result.sampled_token_ids[1] = [2]  # Scheduler stop handling truncates output.
    collector.complete_iteration(first, result)
    assert collector.take_ready()[0].scheduled_workload.sum_decode_kv_tokens == 300

    third = _decode_workload_output({"a": 105, "b": 207})
    collector.begin_iteration(third)
    result.sampled_token_ids = [[5], [6, 7, 8, 9]]
    result.iteration_timing_samples = ((second.iteration_metrics_id, 20.0),)
    collector.before_update(second, result)
    collector.complete_iteration(second, result)
    workload = collector.take_ready()[0].scheduled_workload
    assert workload.sum_decode_kv_tokens == 304  # 101 + 203
    assert workload.var_decode_kv_tokens == 2601
    result.iteration_timing_samples = ((third.iteration_metrics_id, 30.0),)
    collector.complete_iteration(third, result)
    assert collector.take_ready()[0].scheduled_workload.sum_decode_kv_tokens == 309


@pytest.mark.parametrize("preempted", [False, True])
def test_async_spec_workload_correction_does_not_cross_request_generation(preempted):
    request = SimpleNamespace(num_preemptions=0)
    scheduler = SimpleNamespace(
        requests={"request": request}, waiting=[], kv_holding_waiting=[]
    )
    collector = IterationMetricsCollector(scheduler, correct_async_spec_lengths=True)
    first = _decode_workload_output({"request": 100})
    collector.begin_iteration(first)
    if preempted:
        request.num_preemptions += 1
    else:
        scheduler.requests["request"] = SimpleNamespace(num_preemptions=0)
    resumed = _decode_workload_output({"request": 20})
    collector.begin_iteration(resumed)
    result = ModelRunnerOutput(
        ["request"],
        {"request": 0},
        [[1]],
        iteration_timing_samples=((first.iteration_metrics_id, 10.0),),
    )
    collector.before_update(first, result)
    collector.complete_iteration(first, result)
    collector.take_ready()
    scheduler.requests.clear()
    result.iteration_timing_samples = ((resumed.iteration_metrics_id, 20.0),)
    collector.complete_iteration(resumed, result)
    assert collector.take_ready()[0].scheduled_workload.sum_decode_kv_tokens == 20


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_dummy_and_idle_keep_queue_snapshots_without_double_counting_deferred(policy):
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.request import RequestStatus

    scheduler = create_scheduler(scheduling_policy=policy)
    states = [
        (RequestStatus.WAITING, 0),
        (RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR, 0),
        (RequestStatus.WAITING, 32),
        (RequestStatus.PREEMPTED, 70),
        (RequestStatus.WAITING_FOR_REMOTE_KVS, 80),
        (RequestStatus.WAITING_FOR_STREAMING_REQ, 96),
    ]
    requests = create_requests(num_requests=len(states), num_tokens=100)
    for request, (status, computed) in zip(requests, states):
        request.status = status
        request.num_computed_tokens = computed
        scheduler.add_request(request)
    assert len(scheduler.deferred_waiting) == 3
    collector = IterationMetricsCollector(scheduler)
    index = collector.begin_dummy_iteration()
    assert collector.make_idle_record() is None
    requests[0].append_output_token_ids([1, 2])
    collector.complete_timing_samples(((index, 20.0),))
    (dummy,) = collector.take_ready()
    assert dummy.is_dummy and not dummy.is_idle
    assert dummy.scheduled_workload == ScheduledIterationWorkload()
    assert dummy.queued_workload == QueuedIterationWorkload(
        num_prefill_requests=4,
        sum_prefill_tokens=400,
        num_decode_requests=2,
        sum_decode_kv_tokens=150,
        var_decode_kv_tokens=25,
    )
    idle = collector.make_idle_record()
    assert idle.is_idle and not idle.is_dummy and idle.elapsed_ms == 0
    assert idle.iteration_index > dummy.iteration_index
    assert idle.queued_workload.sum_prefill_tokens == 402


def test_prefix_cache_stats_hits_by_source_accumulate():
    stats = PrefixCacheStats()
    stats.record(
        num_tokens=8,
        num_hits=5,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 3, CacheHitSource.DISK: 2},
    )
    stats.record(
        num_tokens=8,
        num_hits=6,
        preempted=False,
        hits_by_source={CacheHitSource.P2P: 5, CacheHitSource.HOST: 1},
    )
    # Zero counts never create an entry.
    stats.record(
        num_tokens=8,
        num_hits=4,
        preempted=False,
        hits_by_source={CacheHitSource.DISK: 4, CacheHitSource.P2P: 0},
    )

    assert stats.hits_by_source == {
        CacheHitSource.HOST: 4,
        CacheHitSource.DISK: 6,
        CacheHitSource.P2P: 5,
    }
    assert sum(stats.hits_by_source.values()) == stats.hits == 15


def test_prefix_cache_stats_record_hits_by_source():
    stats = PrefixCacheStats()
    stats.record(num_tokens=32, num_hits=16, preempted=False)
    stats.record(
        num_tokens=32,
        num_hits=24,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 16, CacheHitSource.DISK: 8},
    )
    # Preempted re-admissions never contribute to hits or the split.
    stats.record(
        num_tokens=32,
        num_hits=32,
        preempted=True,
        hits_by_source={CacheHitSource.P2P: 32},
    )
    # A connector miscount is reported as unspecified, not dropped or raised.
    stats.record(
        num_tokens=8,
        num_hits=8,
        preempted=False,
        hits_by_source={CacheHitSource.HOST: 4},
    )

    assert stats.requests == 3
    assert stats.hits == 48
    assert stats.preempted_hits == 32
    assert stats.hits_by_source == {
        CacheHitSource.HOST: 16,
        CacheHitSource.DISK: 8,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 8,
    }
    assert sum(stats.hits_by_source.values()) == 48 - 16


def test_prefix_cache_stats_hits_by_source_serialization():
    connector_stats = PrefixCacheStats()
    connector_stats.record(
        num_tokens=16,
        num_hits=8,
        preempted=False,
        hits_by_source={CacheHitSource.P2P: 4, CacheHitSource.HOST: 4},
    )
    outputs = EngineCoreOutputs(
        scheduler_stats=SchedulerStats(
            connector_prefix_cache_stats=connector_stats,
        )
    )

    encoded = MsgpackEncoder().encode(outputs)
    decoded = MsgpackDecoder(EngineCoreOutputs).decode(encoded)

    assert decoded.scheduler_stats is not None
    assert decoded.scheduler_stats.connector_prefix_cache_stats is not None
    assert decoded.scheduler_stats.connector_prefix_cache_stats.hits_by_source == {
        CacheHitSource.P2P: 4,
        CacheHitSource.HOST: 4,
    }


def test_compute_iteration_details_includes_encoder_stats():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_encoder_input_stats = ScheduledEncoderInputStats(
        num_inputs=2,
        output_tokens=392,
    )

    iteration_details = compute_iteration_details(scheduler_output)

    assert iteration_details.num_encoder_inputs == 2
    assert iteration_details.num_encoder_output_tokens == 392


def test_prefill_kv_computed_with_cache():
    """Test that prefill KV compute correctly excludes cached tokens."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 5.0
    req_stats.num_generation_tokens = 50

    # Case 1: With prefix cache (1200 tokens cached)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-001",
        num_prompt_tokens=10000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=1200,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 10000
    assert finished_req.num_cached_tokens == 1200
    assert finished_req.request_id == "test-req-001"

    # Verify calculation: prefill KV = prompt tokens - cached tokens
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 8800  # 10000 - 1200


def test_prefill_kv_computed_no_cache():
    """Test prefill KV compute without prefix caching."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 2.0
    req_stats.num_generation_tokens = 10

    # Case 2: No prefix cache
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-002",
        num_prompt_tokens=2000,
        max_tokens_param=100,
        req_stats=req_stats,
        num_cached_tokens=0,
    )

    finished_req = iteration_stats.finished_requests[0]
    assert finished_req.num_prompt_tokens == 2000
    assert finished_req.num_cached_tokens == 0
    assert finished_req.request_id == "test-req-002"

    # Verify calculation: prefill KV = full prompt when no cache
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 2000


def test_prefill_kv_computed_edge_cases():
    """Test edge cases for prefill KV compute calculation."""
    iteration_stats = IterationStats()
    req_stats = RequestStateStats(arrival_time=0.0)
    req_stats.scheduled_ts = 0.1
    req_stats.first_token_ts = 0.5
    req_stats.last_token_ts = 1.0
    req_stats.num_generation_tokens = 1

    # Case 3: Negative num_cached_tokens (shouldn't happen, but handle gracefully)
    iteration_stats.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-003",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=-1,
    )

    finished_req = iteration_stats.finished_requests[0]
    # max() should handle negative values
    prefill_kv_computed = finished_req.num_prompt_tokens - max(
        finished_req.num_cached_tokens, 0
    )
    assert prefill_kv_computed == 100  # Should treat negative as 0
    assert finished_req.request_id == "test-req-003"

    # Case 4: All tokens cached (shouldn't happen in practice)
    iteration_stats2 = IterationStats()
    iteration_stats2.update_from_finished_request(
        finish_reason=FinishReason.STOP,
        request_id="test-req-004",
        num_prompt_tokens=100,
        max_tokens_param=10,
        req_stats=req_stats,
        num_cached_tokens=100,
    )

    finished_req2 = iteration_stats2.finished_requests[0]
    prefill_kv_computed2 = finished_req2.num_prompt_tokens - max(
        finished_req2.num_cached_tokens, 0
    )
    assert prefill_kv_computed2 == 0  # All cached, nothing computed
    assert finished_req2.request_id == "test-req-004"


def test_prompt_token_stats_all_computed():
    """Test all tokens computed locally, no caching."""
    stats = PromptTokenStats()

    # Case 1: No caching (All tokens computed locally)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1000
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 0
    assert stats.total == 1000


def test_prompt_token_stats_partial_local_cache():
    """Test partial local prefix cache hit."""
    stats = PromptTokenStats()

    # Case 2: Partial local cache
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=300,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 700
    assert stats.local_cache_hit == 300
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 300
    assert stats.total == 1000


def test_prompt_token_stats_partial_external_transfer():
    """Test partial external KV transfer."""
    stats = PromptTokenStats()

    # Case 3: Partial external transfer
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=500,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 500
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 500
    assert stats.cached_tokens == 500
    assert stats.total == 1000


def test_prompt_token_stats_mixed_sources():
    """Test mix of local cache and external transfer."""
    stats = PromptTokenStats()

    # Case 4: Mixed sources
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=400,
        num_external_cached_tokens=200,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 400
    assert stats.local_cache_hit == 400
    assert stats.external_kv_transfer == 200
    assert stats.cached_tokens == 600
    assert stats.total == 1000


def test_prompt_token_stats_full_local_cache_recompute():
    """Test full local cache triggers last token recomputation.

    When all tokens are cached, the scheduler forces the model to recompute
    the last token (num_computed_tokens=1), with the rest from cache.
    """
    stats = PromptTokenStats()

    # Case 5: Full local cache (999 cached, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=999,
        num_external_cached_tokens=0,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 999
    assert stats.external_kv_transfer == 0
    assert stats.cached_tokens == 999
    assert stats.total == 1000


def test_prompt_token_stats_full_external_transfer_recompute():
    """Test full external transfer triggers last token recomputation."""
    stats = PromptTokenStats()

    # Case 6: Full external transfer (999 from external, 1 recomputed)
    prefill_stats = PrefillStats()
    prefill_stats.set(
        num_prompt_tokens=1000,
        num_local_cached_tokens=0,
        num_external_cached_tokens=999,
    )
    stats.update_from_output(prefill_stats)

    assert stats.computed == 1
    assert stats.local_cache_hit == 0
    assert stats.external_kv_transfer == 999
    assert stats.cached_tokens == 999
    assert stats.total == 1000
