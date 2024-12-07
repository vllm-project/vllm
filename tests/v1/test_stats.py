from typing import Optional

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.stats.common import RequestStats, RequestStatsUpdate


def test_lifecycle_updates():
    request_id = "test_request"
    stats = RequestStats(request_id=request_id)

    # Test the below scenario:
    arrived_ts = 0
    input_processed_ts = 1
    queued_ts = 2
    running_ts = 3
    running_2_ts = 4
    decoded_ts = 5
    detokenized_ts = 6
    decoded_2_ts = 7
    preempted_ts = 8
    resumed_ts = 9
    decoded_3_ts = 10
    finished_ts = 11

    # Test ARRIVED
    arrived_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.ARRIVED,
        monotonic_ts_s=arrived_ts,
    )
    stats.update_from(arrived_update)
    assert stats.arrival_ts_s == arrived_ts
    assert stats.last_updated_ts_s == arrived_ts

    # Test INPUT_PROCESSED
    sampling_params = SamplingParams(n=1)
    engine_request = EngineCoreRequest(
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        sampling_params=sampling_params,
        request_id=request_id,
        prompt="test_prompt",
        mm_inputs=None,
        mm_placeholders=None,
        eos_token_id=None,
        arrival_time=arrived_ts,
        lora_request=None,
    )
    input_processed_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.INPUT_PROCESSED,
        monotonic_ts_s=input_processed_ts,
        engine_request=engine_request,
    )
    stats.update_from(input_processed_update)
    assert stats.input_processor_end_ts_s == input_processed_ts
    assert stats.engine_request == engine_request
    assert stats.last_updated_ts_s == input_processed_ts
    assert stats.num_prompt_tokens == 6
    assert stats.sampling_params == sampling_params

    assert stats.first_token_ts_s is None
    assert stats.prefill_ts_s is None

    # Test QUEUED
    queued_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.QUEUED,
        monotonic_ts_s=queued_ts,
    )
    stats.update_from(queued_update)
    assert stats.queued_ts_s == queued_ts
    assert stats.last_updated_ts_s == queued_ts

    # Test RUNNING
    running_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.RUNNING,
        monotonic_ts_s=running_ts,
        new_prefill=True,
        num_computed_tokens=3,
        num_cached_tokens=1,
    )
    stats.update_from(running_update)
    assert stats.prefill_ts_s == running_ts
    assert stats.num_computed_tokens == 3
    assert stats.num_cached_tokens == 1
    assert stats.queue_duration_s == running_ts - queued_ts

    # Test RUNNING again shouldn't update prefill_ts_s
    running_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.RUNNING,
        monotonic_ts_s=running_2_ts,
        new_prefill=False,
        num_computed_tokens=6,
        num_cached_tokens=0,
    )
    stats.update_from(running_update)
    assert stats.prefill_ts_s == running_ts
    assert stats.num_computed_tokens == 6
    # num_cached_tokens is not updated
    assert stats.num_cached_tokens == 1
    assert stats.last_updated_ts_s == running_2_ts
    # prefill_start_ts_s_lst should only contain the first running/resumed
    # running prefill update.
    assert stats.prefill_start_ts_s_lst == [
        running_ts,
    ]

    # Test DECODED
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODED,
        monotonic_ts_s=decoded_ts,
        num_new_tokens=1,
        token_perf_ts_ns=decoded_ts * 1e9,
    )
    stats.update_from(decoded_update)
    assert stats.last_updated_ts_s == decoded_ts
    # Since arrival
    assert stats.first_token_latency_s == decoded_ts - arrived_ts
    assert stats.num_output_tokens == 1
    # Since first scheduled
    assert stats.prefill_latency_s == 2

    # Test DETOKENIZED
    detokenized_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DETOKENIZED,
        monotonic_ts_s=detokenized_ts,
    )
    stats.update_from(detokenized_update)
    assert stats.last_updated_ts_s == detokenized_ts

    # Test another DECODE should yield correct inter token latency
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODED,
        monotonic_ts_s=decoded_2_ts,
        num_new_tokens=1,
        token_perf_ts_ns=decoded_2_ts * 1e9,
    )
    stats.update_from(decoded_update)
    assert stats.output_token_latency_s_lst == [
        decoded_2_ts - decoded_ts,
    ]
    assert stats.num_output_tokens == 2

    # Test PREEMPTED
    preempted_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.PREEMPTED,
        monotonic_ts_s=preempted_ts,
    )
    stats.update_from(preempted_update)
    assert stats.last_updated_ts_s == preempted_ts
    assert stats.preempted_ts_s_lst == [preempted_ts]
    # States should be reset
    assert stats.num_computed_tokens == 0
    assert stats.num_cached_tokens == 0
    assert stats.num_output_tokens == 0
    assert stats.output_token_latency_s_lst == []

    # Test resumed
    resumed_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.RUNNING,
        monotonic_ts_s=resumed_ts,
        new_prefill=True,
        num_computed_tokens=6,
        num_cached_tokens=2,
    )
    stats.update_from(resumed_update)
    # Resumed prefill timestamp should be updated
    assert stats.prefill_ts_s == resumed_ts
    assert stats.num_computed_tokens == 6
    assert stats.num_cached_tokens == 2
    assert stats.prefill_start_ts_s_lst == [
        running_ts,
        resumed_ts,
    ]
    assert stats.last_updated_ts_s == resumed_ts

    # Test another DECODED should yield correct first token latency.
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODED,
        monotonic_ts_s=decoded_3_ts,
        num_new_tokens=1,
        token_perf_ts_ns=decoded_3_ts * 1e9,
    )
    stats.update_from(decoded_update)
    assert stats.first_token_ts_s == decoded_3_ts
    assert stats.num_output_tokens == 1

    # Test FINISHED
    finished_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DETOKENIZED,
        monotonic_ts_s=finished_ts,
        finish_reason="test_reason",
    )
    stats.update_from(finished_update)
    assert stats.last_updated_ts_s == finished_ts
    assert stats.e2e_latency_s == finished_ts - arrived_ts
    assert stats.inference_latency_s == finished_ts - resumed_ts
    assert stats.decode_latency_s == finished_ts - decoded_3_ts
    assert stats.is_finished
    assert stats.finish_reason == "test_reason"


@pytest.mark.parametrize("finish_reason",
                         ["test-decode", "test-detokenize", None])
def test_finish_reason(finish_reason: Optional[str]):
    """
    Test that a request could be finished when decoded and detokenized
    at different times.
    """
    request_id = "test_request"
    r = RequestStats(request_id=request_id)

    # Test FINISHED
    updates = [
        RequestStatsUpdate(
            request_id=request_id,
            type=RequestStatsUpdate.Type.ARRIVED,
            monotonic_ts_s=0,
        ),
        RequestStatsUpdate(
            request_id=request_id,
            type=RequestStatsUpdate.Type.INPUT_PROCESSED,
            monotonic_ts_s=1,
        ),
        RequestStatsUpdate(
            request_id=request_id,
            type=RequestStatsUpdate.Type.QUEUED,
            monotonic_ts_s=2,
        ),
        RequestStatsUpdate(
            request_id=request_id,
            type=RequestStatsUpdate.Type.RUNNING,
            new_prefill=True,
            monotonic_ts_s=3,
            num_computed_tokens=3,
        ),
    ]

    if finish_reason is not None:
        if finish_reason == "test-decode":
            updates.append(
                RequestStatsUpdate(
                    request_id=request_id,
                    type=RequestStatsUpdate.Type.DECODED,
                    monotonic_ts_s=4,
                    finish_reason=finish_reason,
                    token_perf_ts_ns=4 * 1e9,
                    num_new_tokens=1,
                ))
        elif finish_reason == "test-detokenize":
            updates.append(
                RequestStatsUpdate(
                    request_id=request_id,
                    type=RequestStatsUpdate.Type.DETOKENIZED,
                    monotonic_ts_s=4,
                    finish_reason=finish_reason,
                ))

    for update in updates:
        r.update_from(update)

    if finish_reason is not None:
        assert r.finish_reason == finish_reason
        assert r.is_finished
        assert r.e2e_latency_s == 4
    else:
        assert r.finish_reason is None
        assert not r.is_finished
        assert r.e2e_latency_s is None
