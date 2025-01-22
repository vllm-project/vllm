import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.stats.common import RequestStats, RequestStatsUpdate


def make_update(
    request_id: str,
    update_type: RequestStatsUpdate.Type,
    monotonic_ts_s: float,
    **kwargs,
):
    if update_type == RequestStatsUpdate.Type.INPUT_PROCESSED:
        kwargs.setdefault("sampling_params", SamplingParams(n=1))
        kwargs.setdefault("num_prompt_tokens", 10)
    elif update_type == RequestStatsUpdate.Type.PREFILLING:
        kwargs.setdefault("num_computed_tokens", 10)
        kwargs.setdefault("num_cached_tokens", 10)
    elif update_type == RequestStatsUpdate.Type.DETOKENIZED:
        kwargs.setdefault("num_new_tokens", 10)
    elif update_type == RequestStatsUpdate.Type.FINISHED:
        kwargs.setdefault("finish_reason", "test_reason")

    return RequestStatsUpdate(
        request_id=request_id,
        type=update_type,
        monotonic_ts_s=monotonic_ts_s,
        **kwargs,
    )


def test_invalid_request_update():
    request_id = "test_request"
    update_specific_required_fields = {
        RequestStatsUpdate.Type.INPUT_PROCESSED: [
            "sampling_params",
            "num_prompt_tokens",
        ],
        RequestStatsUpdate.Type.PREFILLING: [
            "num_computed_tokens",
            "num_cached_tokens",
        ],
        RequestStatsUpdate.Type.DETOKENIZED: ["num_new_tokens"],
        RequestStatsUpdate.Type.FINISHED: ["finish_reason"],
    }

    # Missing a required field should raise an assertion error.
    for update_type in RequestStatsUpdate.Type:
        required_fields = update_specific_required_fields.get(update_type, [])

        # Try to miss one of the required fields.
        kwargs = {field: object() for field in required_fields}
        for field in required_fields:
            copy_kwargs = kwargs.copy()
            copy_kwargs.pop(field)
            with pytest.raises(ValueError):
                RequestStatsUpdate(
                    request_id=request_id,
                    type=update_type,
                    **copy_kwargs,
                )


def test_invalid_request_update_transition():
    # Test invalid transition type.
    for src in RequestStatsUpdate.Type:
        for dst in RequestStatsUpdate.Type:
            if dst not in RequestStatsUpdate._VALID_TRANSITIONS[src]:
                with pytest.raises(AssertionError):
                    RequestStatsUpdate.check_valid_update(
                        make_update(
                            update_type=dst,
                            request_id="test_request",
                            monotonic_ts_s=1,
                        ),
                        last_update_type=src,
                        last_updated_ts_s=0,
                    )
            else:
                RequestStatsUpdate.check_valid_update(
                    make_update(
                        request_id="test_request",
                        update_type=dst,
                        monotonic_ts_s=1,
                    ),
                    last_update_type=src,
                    last_updated_ts_s=0,
                )

    # Test invalid timestamp.
    with pytest.raises(AssertionError):
        RequestStatsUpdate.check_valid_update(
            make_update(
                request_id="test_request",
                update_type=RequestStatsUpdate.Type.ARRIVED,
                monotonic_ts_s=1,
            ),
            last_update_type=None,
            last_updated_ts_s=2,
        )


def test_lifecycle_updates():
    request_id = "test_request"
    stats = RequestStats(request_id=request_id)

    # Test the below scenario:
    arrived_ts = 0
    input_processed_ts = 1
    queued_ts = 2
    prefilling_ts = 3
    decoded_ts = 5
    detokenized_ts = 6
    decoded_2_ts = 7
    detokenized_2_ts = 8
    preempted_ts = 9
    resumed_ts = 10
    decoded_3_ts = 11
    detokenized_3_ts = 12
    finished_ts = 13

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
    input_processed_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.INPUT_PROCESSED,
        monotonic_ts_s=input_processed_ts,
        sampling_params=sampling_params,
        num_prompt_tokens=6,
    )
    stats.update_from(input_processed_update)
    assert stats.input_processor_end_ts_s == input_processed_ts
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

    # Test PREFILLING
    prefilling_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.PREFILLING,
        monotonic_ts_s=prefilling_ts,
        num_computed_tokens=3,
        num_cached_tokens=1,
    )
    stats.update_from(prefilling_update)
    assert stats.prefill_ts_s == prefilling_ts
    assert stats.num_computed_tokens == 3
    assert stats.num_cached_tokens == 1
    assert stats.queue_duration_s == prefilling_ts - queued_ts

    # Test DECODING
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODING,
        monotonic_ts_s=decoded_ts,
    )
    stats.update_from(decoded_update)
    assert stats.last_updated_ts_s == decoded_ts

    # Test DETOKENIZED
    detokenized_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DETOKENIZED,
        monotonic_ts_s=detokenized_ts,
        num_new_tokens=1,
    )
    stats.update_from(detokenized_update)
    assert stats.last_updated_ts_s == detokenized_ts
    assert stats.num_output_tokens == 1
    # Since arrival
    assert stats.first_token_latency_s == detokenized_ts - arrived_ts
    # Since first scheduled
    assert stats.prefill_latency_s == detokenized_ts - prefilling_ts

    # Test another DECODING and DETOKENIZED should
    # yield correct inter token latency
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODING,
        monotonic_ts_s=decoded_2_ts,
    )
    stats.update_from(decoded_update)

    detokenized_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DETOKENIZED,
        monotonic_ts_s=detokenized_2_ts,
        num_new_tokens=1,
    )
    stats.update_from(detokenized_update)
    assert stats.output_token_latency_s_lst == [
        detokenized_2_ts - detokenized_ts,
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
    # These states should not be reset
    assert stats.num_output_tokens == 2
    assert stats.output_token_latency_s_lst == [
        detokenized_2_ts - detokenized_ts,
    ]
    assert stats.prefill_latency_s == prefilling_ts - arrived_ts
    assert stats.num_prompt_tokens == 6
    assert stats.prefill_start_ts_s_lst == [prefilling_ts]

    # Test resumed
    resumed_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.PREFILLING,
        monotonic_ts_s=resumed_ts,
        num_computed_tokens=6,
        num_cached_tokens=2,
    )
    stats.update_from(resumed_update)
    # prefill timestamp should not be updated since it's a resumed prefill
    assert stats.prefill_ts_s == prefilling_ts
    assert stats.num_computed_tokens == 6
    assert stats.num_cached_tokens == 2
    assert stats.prefill_start_ts_s_lst == [
        prefilling_ts,
        resumed_ts,
    ]
    assert stats.last_updated_ts_s == resumed_ts

    # Test another DECODED/DETOKENIZED should yield correct first token latency.
    decoded_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DECODING,
        monotonic_ts_s=decoded_3_ts,
    )
    detokenized_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.DETOKENIZED,
        monotonic_ts_s=detokenized_3_ts,
        num_new_tokens=1,
    )
    stats.update_from(decoded_update)
    stats.update_from(detokenized_update)
    assert stats.first_token_ts_s == detokenized_ts - arrived_ts
    assert stats.num_output_tokens == 3
    assert stats.output_token_latency_s_lst == [
        detokenized_2_ts - detokenized_ts,
        detokenized_3_ts - detokenized_2_ts,
    ]

    # Test FINISHED
    finished_update = RequestStatsUpdate(
        request_id=request_id,
        type=RequestStatsUpdate.Type.FINISHED,
        monotonic_ts_s=finished_ts,
        finish_reason="test_reason",
    )
    stats.update_from(finished_update)
    assert stats.last_updated_ts_s == finished_ts
    assert stats.e2e_latency_s == finished_ts - arrived_ts
    assert stats.inference_latency_s == finished_ts - prefilling_ts
    assert stats.prefill_latency_s == detokenized_ts - prefilling_ts
    assert stats.decode_latency_s == finished_ts - detokenized_ts
    assert stats.first_token_latency_s == detokenized_ts - arrived_ts
    assert stats.queue_duration_s == prefilling_ts - queued_ts
    assert stats.is_finished
    assert stats.finish_reason == "test_reason"

    # TODO(rickyx): Add model forward/execute time.
    assert stats.model_forward_duration_s == 0.0
    assert stats.model_execute_duration_s == 0.0
