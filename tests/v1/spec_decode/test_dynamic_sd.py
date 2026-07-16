# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Dynamic SD batch-size schedule helpers."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.structured_output import StructuredOutputManager


def _make_lookup(
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]],
    *,
    max_batch_size: int = 256,
    runtime_num_speculative_tokens: int = 3,
) -> list[int]:
    return build_dynamic_sd_schedule_lookup(
        num_speculative_tokens_per_batch_size=num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=max_batch_size,
        vllm_num_speculative_tokens=runtime_num_speculative_tokens,
    )


def _make_scheduler_with_dynamic_sd(
    schedule: list[tuple[int, int, int]],
    *,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    runtime_num_speculative_tokens: int = 3,
    adaptive_verification: bool = False,
) -> Scheduler:
    base_scheduler = create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=runtime_num_speculative_tokens,
    )

    speculative_config = base_scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    speculative_config.num_speculative_tokens_per_batch_size = schedule
    speculative_config.adaptive_verification = adaptive_verification

    return Scheduler(
        vllm_config=base_scheduler.vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        block_size=base_scheduler.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base_scheduler.vllm_config),
    )


def _add_requests_and_schedule(
    scheduler: Scheduler, num_requests: int, *, num_tokens: int = 10
):
    requests = create_requests(num_requests=num_requests, num_tokens=num_tokens)
    for request in requests:
        scheduler.add_request(request)
    return scheduler.schedule()


def test_dynamic_sd_uses_batch_size_schedule():
    dynamic_sd_lookup = _make_lookup(
        [
            (1, 16, 3),
            (32, 128, 2),
            (256, 2048, 0),
        ]
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[16] == 3
    assert dynamic_sd_lookup[17] == 3
    assert dynamic_sd_lookup[31] == 3
    assert dynamic_sd_lookup[32] == 2
    assert dynamic_sd_lookup[128] == 2
    assert dynamic_sd_lookup[129] == 2
    assert dynamic_sd_lookup[255] == 2
    assert dynamic_sd_lookup[256] == 0


def test_dynamic_sd_requires_schedule_starting_at_batch_size_one():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_lookup([(2, 16, 3)])


def test_dynamic_sd_clamps_k_to_runtime_max():
    dynamic_sd_lookup = _make_lookup(
        [(1, 256, 4)],
        runtime_num_speculative_tokens=3,
    )

    assert dynamic_sd_lookup[1] == 3
    assert dynamic_sd_lookup[256] == 3


def test_dynamic_sd_rejects_invalid_schedule_entry():
    with pytest.raises(ValueError, match="3-item sequence"):
        _make_lookup([(1, 16, 3), (32, 64)])  # type: ignore[list-item]


def test_dynamic_sd_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="non-overlapping and sorted"):
        _make_lookup([(1, 16, 3), (16, 32, 2)])


def test_dynamic_sd_rejects_negative_k():
    with pytest.raises(ValueError, match="values must be >= 0"):
        _make_lookup([(1, 16, -1)])


def test_dynamic_sd_rejects_empty_schedule():
    with pytest.raises(ValueError, match="must not be empty"):
        _make_lookup([])


def test_dynamic_sd_requires_schedule_config():
    with pytest.raises(
        ValueError, match="num_speculative_tokens_per_batch_size is required"
    ):
        build_dynamic_sd_schedule_lookup(
            None,
            vllm_max_batch_size=256,
            vllm_num_speculative_tokens=3,
        )


def test_dynamic_sd_lookup_rejects_invalid_batch_size_queries():
    dynamic_sd_lookup = _make_lookup([(1, 256, 3)])

    assert dynamic_sd_lookup[0] == 0
    with pytest.raises(IndexError):
        _ = dynamic_sd_lookup[257]


def test_scheduler_initializes_dynamic_sd_lookup_from_speculative_config():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )

    assert scheduler.dynamic_sd_lookup is not None
    assert scheduler.num_spec_tokens == 3


def test_scheduler_uses_dsd_k_based_on_number_of_scheduled_requests():
    test_cases = [
        (4, 3),
        (64, 2),
        (256, 0),
    ]

    for num_requests, expected_k in test_cases:
        scheduler = _make_scheduler_with_dynamic_sd(
            [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
            max_num_seqs=num_requests,
            max_num_batched_tokens=num_requests * 10,
            runtime_num_speculative_tokens=3,
        )
        output = _add_requests_and_schedule(scheduler, num_requests)

        assert len(output.num_scheduled_tokens) == num_requests
        assert output.num_spec_tokens_to_schedule == expected_k


def test_v2_scheduler_selects_dynamic_sd_k():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 1, 4), (2, 2, 2)],
        max_num_seqs=2,
        max_num_batched_tokens=20,
        runtime_num_speculative_tokens=4,
    )
    scheduler.use_v2_model_runner = True

    output = _add_requests_and_schedule(scheduler, 2)

    assert len(output.num_scheduled_tokens) == 2
    assert output.num_spec_tokens_to_schedule == 2


def test_scheduler_clamps_dsd_k_to_runtime_num_speculative_tokens():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 256, 5)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3


@pytest.mark.parametrize("optimal_k", [0, 2])
def test_adaptive_scheduler_ignores_dynamic_schedule_for_mixed_batch(optimal_k: int):
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 2, optimal_k)],
        max_num_seqs=2,
        max_num_batched_tokens=32,
        runtime_num_speculative_tokens=4,
        adaptive_verification=True,
    )
    decode_req, prefill_req = create_requests(num_requests=2, num_tokens=1)
    scheduler.add_request(decode_req)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[decode_req.request_id],
            req_id_to_index={decode_req.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds([decode_req.request_id], [[10, 11, 12, 13]])
    )

    scheduler.add_request(prefill_req)
    output = scheduler.schedule()

    assert scheduler.dynamic_sd_lookup is None
    assert output.scheduled_spec_decode_tokens == {
        decode_req.request_id: [10, 11, 12, 13],
        prefill_req.request_id: [-1, -1, -1, -1],
    }
    assert output.num_scheduled_tokens == {
        decode_req.request_id: 5,
        prefill_req.request_id: 5,
    }
    assert output.total_num_scheduled_tokens == 10
    assert output.num_spec_tokens_to_schedule == 4


def test_adaptive_scheduler_uses_fixed_k_max():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 2, 2)],
        max_num_seqs=2,
        max_num_batched_tokens=32,
        runtime_num_speculative_tokens=4,
        adaptive_verification=True,
    )
    requests = create_requests(num_requests=2, num_tokens=1)
    for request in requests:
        scheduler.add_request(request)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[request.request_id for request in requests],
            req_id_to_index={
                request.request_id: i for i, request in enumerate(requests)
            },
            sampled_token_ids=[[0], [0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [request.request_id for request in requests],
            [[10, 11, 12, 13], [20, 21, 22, 23]],
        )
    )

    output = scheduler.schedule()

    assert output.scheduled_spec_decode_tokens == {
        requests[0].request_id: [10, 11, 12, 13],
        requests[1].request_id: [20, 21, 22, 23],
    }
    assert output.num_scheduled_tokens == {
        requests[0].request_id: 5,
        requests[1].request_id: 5,
    }
    assert output.total_num_scheduled_tokens == 10
    assert output.num_spec_tokens_to_schedule == 4


def test_scheduler_uses_valid_structured_output_draft_prefix():
    scheduler = create_scheduler(
        num_speculative_tokens=4,
        use_v2_model_runner=True,
    )
    request = create_requests(num_requests=1, num_tokens=1)[0]
    scheduler.add_request(request)
    prefill_output = scheduler.schedule()
    scheduler.update_from_output(
        prefill_output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    grammar = Mock()
    grammar.validate_tokens.return_value = [10, 11, 12]
    request.structured_output_request = Mock(grammar=grammar)
    scheduler.update_draft_token_ids(
        DraftTokenIds(
            [request.request_id],
            [[10, 11, 12, 13]],
        )
    )

    expected_spec_token_ids = [10, 11, 12]
    assert request.spec_token_ids == expected_spec_token_ids
    grammar.validate_tokens.assert_called_once_with([10, 11, 12, 13])

    output = scheduler.schedule()
    assert (
        output.scheduled_spec_decode_tokens[request.request_id]
        == expected_spec_token_ids
    )

    bitmask = np.zeros((len(expected_spec_token_ids) + 1, 1), dtype=np.int32)
    scheduler.structured_output_manager.grammar_bitmask = Mock(return_value=bitmask)
    grammar_output = scheduler.get_grammar_bitmask(output)

    assert grammar_output is not None
    assert grammar_output.grammar_bitmask is bitmask
    assert scheduler.structured_output_manager.grammar_bitmask.call_args.args[2] == {
        request.request_id: expected_spec_token_ids
    }


def test_async_scheduler_output_keeps_full_draft_window_for_grammar_masks():
    scheduler = create_scheduler(
        num_speculative_tokens=4,
        use_v2_model_runner=True,
    )
    request = create_requests(num_requests=1, num_tokens=1)[0]
    grammar = Mock()
    grammar.validate_tokens.return_value = [10, 11, 12]
    request.structured_output_request = Mock(grammar=grammar)
    scheduler.requests[request.request_id] = request

    output = SchedulerOutput.make_empty()
    output.scheduled_spec_decode_tokens = {request.request_id: [-1, -1, -1, -1]}
    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds([request.request_id], [[10, 11, 12, 13]]),
        output,
    )

    assert output.scheduled_spec_decode_tokens == {request.request_id: [10, 11, 12, -1]}


def test_scheduler_falls_back_to_static_k_when_dsd_not_configured():
    scheduler = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=40,
        num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 4)

    assert scheduler.dynamic_sd_lookup is None
    assert output.num_spec_tokens_to_schedule == 3


def test_dynamic_sd_is_disabled_with_data_parallel(caplog_vllm):
    with caplog_vllm.at_level(logging.WARNING, logger="vllm"):
        scheduler = create_scheduler(
            max_num_seqs=256,
            max_num_batched_tokens=2560,
            num_speculative_tokens=3,
            num_speculative_tokens_per_batch_size=[
                (1, 16, 3),
                (64, 128, 2),
                (256, 4096, 0),
            ],
            data_parallel_size=2,
        )

    speculative_config = scheduler.vllm_config.speculative_config
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens_per_batch_size is None
    assert scheduler.dynamic_sd_lookup is None
    assert "Dynamic speculative decoding is not supported with data parallelism" in (
        caplog_vllm.text
    )

    output = _add_requests_and_schedule(scheduler, 256)
    assert len(output.num_scheduled_tokens) == 256
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_uses_static_k_when_no_requests_are_scheduled():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        runtime_num_speculative_tokens=3,
    )
    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 0
    assert output.num_spec_tokens_to_schedule == 3


def test_scheduler_rejects_bad_dsd_config_at_construction():
    with pytest.raises(ValueError, match="must start at 1"):
        _make_scheduler_with_dynamic_sd([(2, 16, 3)])


def test_scheduler_passes_max_num_seqs_as_dsd_runtime_batch_limit():
    scheduler = _make_scheduler_with_dynamic_sd(
        [(1, 16, 3), (64, 128, 2), (256, 4096, 0)],
        max_num_seqs=16,
        max_num_batched_tokens=160,
        runtime_num_speculative_tokens=3,
    )
    output = _add_requests_and_schedule(scheduler, 16)

    assert scheduler.dynamic_sd_lookup is not None
    assert len(scheduler.dynamic_sd_lookup) == 17
    assert len(output.num_scheduled_tokens) == 16
    assert output.num_spec_tokens_to_schedule == 3
