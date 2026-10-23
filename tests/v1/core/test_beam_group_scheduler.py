# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler tests for in-engine beam groups.

Covers atomic admission only. Token selection and KV fork are follow-ups
of https://github.com/vllm-project/vllm/issues/61880.
"""

import pytest

from vllm.v1.core.sched.beam_group import BeamSearchGroup, equal_chunk_size
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _output_for(scheduler_output, token_id: int) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
    )


def test_equal_chunk_size_keeps_slots_together():
    requests = create_requests(num_requests=4, num_tokens=10, same_prompt=True)
    assert (
        equal_chunk_size(
            requests,
            token_budget=8,
            input_budget=8,
            draft_slots=0,
            max_model_len=2048,
            num_sampled_tokens=1,
        )
        == 2
    )
    assert (
        equal_chunk_size(
            requests,
            token_budget=3,
            input_budget=8,
            draft_slots=0,
            max_model_len=2048,
            num_sampled_tokens=1,
        )
        == 0
    )


def test_beam_group_requires_two_slots():
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    with pytest.raises(ValueError, match="at least 2"):
        scheduler.add_beam_group("g", [request])


def test_beam_group_rejected_with_pipeline_parallelism():
    scheduler = create_scheduler(pipeline_parallel_size=2)
    requests = create_requests(num_requests=2, same_prompt=True)
    with pytest.raises(ValueError, match="pipeline parallelism"):
        scheduler.add_beam_group("g", requests)


def test_beam_group_admitted_together_and_decodes_together():
    scheduler = create_scheduler(max_num_seqs=8)
    requests = create_requests(
        num_requests=4,
        num_tokens=6,
        max_tokens=4,
        same_prompt=True,
        req_ids=["b0", "b1", "b2", "b3"],
    )
    normal = create_requests(num_requests=1, num_tokens=6, req_ids=["normal"])[0]

    scheduler.add_beam_group("beams", requests)
    scheduler.add_request(normal)

    assert len(scheduler.waiting) == 1
    assert scheduler.get_num_unfinished_requests() == 5

    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens
    assert scheduled == {"b0": 6, "b1": 6, "b2": 6, "b3": 6, "normal": 6}
    assert [req.req_id for req in output.scheduled_new_reqs] == [
        "b0",
        "b1",
        "b2",
        "b3",
        "normal",
    ]
    assert all(req.status == RequestStatus.RUNNING for req in requests)
    assert "beams" not in scheduler.waiting_beam_group_ids

    scheduler.update_from_output(output, _output_for(output, token_id=7))
    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {
        "b0": 1,
        "b1": 1,
        "b2": 1,
        "b3": 1,
        "normal": 1,
    }
    assert decode.scheduled_new_reqs == []


def test_beam_group_chunks_every_slot_equally():
    scheduler = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8)
    requests = create_requests(
        num_requests=4,
        num_tokens=10,
        same_prompt=True,
        req_ids=["b0", "b1", "b2", "b3"],
    )
    scheduler.add_beam_group("beams", requests)

    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"b0": 2, "b1": 2, "b2": 2, "b3": 2}
    assert all(req.status == RequestStatus.RUNNING for req in requests)

    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=list(first.num_scheduled_tokens),
            req_id_to_index={
                req_id: index for index, req_id in enumerate(first.num_scheduled_tokens)
            },
            sampled_token_ids=[[] for _ in requests],
        ),
    )
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"b0": 2, "b1": 2, "b2": 2, "b3": 2}


def test_beam_group_does_not_take_a_partial_sequence_hole():
    scheduler = create_scheduler(max_num_seqs=4)
    running = create_requests(num_requests=1, num_tokens=4, req_ids=["running"])[0]
    scheduler.add_request(running)
    scheduler.schedule()

    group = create_requests(
        num_requests=4,
        num_tokens=4,
        same_prompt=True,
        req_ids=["b0", "b1", "b2", "b3"],
    )
    waiting = create_requests(num_requests=1, num_tokens=4, req_ids=["waiting"])[0]
    scheduler.add_beam_group("beams", group)
    scheduler.add_request(waiting)

    output = scheduler.schedule()
    assert "b0" not in output.num_scheduled_tokens
    assert "waiting" not in output.num_scheduled_tokens
    assert all(req.status == RequestStatus.WAITING for req in group)
    assert waiting.status == RequestStatus.WAITING
    assert list(scheduler.waiting_beam_group_ids) == ["beams"]


def test_beam_group_allocation_failure_schedules_nobody_in_the_group():
    scheduler = create_scheduler(num_blocks=8, block_size=16, max_num_seqs=8)
    hog = create_requests(num_requests=1, num_tokens=96, req_ids=["hog"])[0]
    scheduler.add_request(hog)
    scheduler.schedule()

    group = create_requests(
        num_requests=2,
        num_tokens=32,
        same_prompt=True,
        block_size=16,
        req_ids=["b0", "b1"],
    )
    scheduler.add_beam_group("beams", group)
    output = scheduler.schedule()

    assert "b0" not in output.num_scheduled_tokens
    assert "b1" not in output.num_scheduled_tokens
    assert all(req.status == RequestStatus.WAITING for req in group)
    assert len(scheduler.running) == 1


def test_aborting_one_beam_slot_aborts_the_group():
    scheduler = create_scheduler()
    requests = create_requests(
        num_requests=3,
        same_prompt=True,
        req_ids=["b0", "b1", "b2"],
    )
    scheduler.add_beam_group("beams", requests)
    scheduler.finish_requests("b1", RequestStatus.FINISHED_ABORTED)

    assert scheduler.beam_groups == {}
    assert scheduler.get_num_unfinished_requests() == 0
    assert all(req.is_finished() for req in requests)


def test_add_request_rejects_a_lone_beam_slot():
    scheduler = create_scheduler()
    (request,) = create_requests(num_requests=1)
    request.beam_group_id = "g"
    with pytest.raises(ValueError, match="add_beam_group"):
        scheduler.add_request(request)


def test_beam_search_group_rejects_uneven_prompts():
    short, long = create_requests(num_requests=2, num_tokens=4)
    long.num_prompt_tokens = 8
    with pytest.raises(ValueError, match="prompt lengths"):
        BeamSearchGroup(group_id="g", requests=[short, long])
