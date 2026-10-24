# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `max_priority_preemptions_per_step` and `priority_prefill_budget_cap`:
under the priority policy, the waiting request of the highest priority preempts
running requests of a lower priority for KV cache blocks, and prefills of a lower
priority sit out the steps of its first tokens.

Unless a test says otherwise, the KV cache holds 32 blocks of 16 tokens (33
with the null block) and prefix caching is off, so requests share no block.
"""

from collections import deque

import pytest

from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .test_scheduler import create_requests_with_priority
from .utils import create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _create_scheduler(**kwargs) -> Scheduler:
    options = dict(
        scheduling_policy="priority",
        max_num_batched_tokens=512,
        max_model_len=24 * BLOCK_SIZE,
        num_blocks=33,
        block_size=BLOCK_SIZE,
    )
    options.update(kwargs)
    return create_scheduler(**options)


def _num_tokens(num_blocks: int) -> int:
    """A prompt that fills `num_blocks` blocks, leaving room in the last one to
    decode 8 tokens."""
    return num_blocks * BLOCK_SIZE - 8


def _request(
    request_id: str,
    num_tokens: int,
    priority: int,
    arrival_time: float,
    max_tokens: int = 64,
    prompt_token: int = 0,
) -> Request:
    return create_requests_with_priority(
        num_requests=1,
        priorities=[priority],
        arrival_times=[arrival_time],
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        starting_idx=prompt_token,
        req_ids=[request_id],
    )[0]


def _model_runner_output(
    scheduler: Scheduler,
    scheduler_output: SchedulerOutput,
    finished_recving: set[str] | None = None,
) -> ModelRunnerOutput:
    """A sampled token for each scheduled request but partial prefills, and the
    async KV loads that are done."""
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[
            [] if scheduler.requests[req_id].is_prefill_chunk else [0]
            for req_id in req_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        kv_connector_output=(
            KVConnectorOutput(finished_recving=finished_recving)
            if finished_recving
            else None
        ),
    )


def _step(
    scheduler: Scheduler, finished_recving: set[str] | None = None
) -> SchedulerOutput:
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output,
        _model_runner_output(scheduler, scheduler_output, finished_recving),
    )
    return scheduler_output


def _run_until_decoding(scheduler: Scheduler, *requests: Request) -> None:
    for request in requests:
        scheduler.add_request(request)
    for _ in range(20):
        if all(
            request.status == RequestStatus.RUNNING and not request.is_prefill_chunk
            for request in requests
        ):
            return
        _step(scheduler)
    raise AssertionError("requests still prefilling after 20 steps")


def _num_free_blocks(scheduler: Scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


def _create_deferring_scheduler(**kwargs) -> Scheduler:
    """An async scheduler with deferred block frees forced on, as in
    test_deferred_block_free.py: their gate (overlapping batches and a KV
    consumer connector) cannot be reached on CPU."""
    scheduler = _create_scheduler(async_scheduling=True, **kwargs)
    scheduler.defer_block_free = True
    return scheduler


def _in_flight_steps(scheduler: Scheduler):
    """`schedule` and `process` for async scheduling: as in the engine, a step's
    output is processed once the next step is scheduled."""
    in_flight: deque[tuple[SchedulerOutput, ModelRunnerOutput]] = deque()

    def schedule() -> SchedulerOutput:
        scheduler_output = scheduler.schedule()
        in_flight.append(
            (scheduler_output, _model_runner_output(scheduler, scheduler_output))
        )
        return scheduler_output

    def process() -> None:
        scheduler.update_from_output(*in_flight.popleft())

    return schedule, process


def _run_three_low_priority_requests(scheduler: Scheduler) -> None:
    """30 of the 32 blocks run: priority 1 holds 10, priority 2 holds 8, 6, 6."""
    _run_until_decoding(
        scheduler,
        _request("p1", _num_tokens(10), priority=1, arrival_time=0),
        _request("big", _num_tokens(8), priority=2, arrival_time=1),
        _request("small", _num_tokens(6), priority=2, arrival_time=2),
        _request("other", _num_tokens(6), priority=2, arrival_time=3),
    )
    assert _num_free_blocks(scheduler) == 2


def test_make_room_is_off_by_default():
    """A waiting request then waits for free blocks whatever its priority."""
    scheduler = _create_scheduler()
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens


@pytest.mark.parametrize(
    "option",
    [{"max_priority_preemptions_per_step": 1}, {"priority_prefill_budget_cap": True}],
)
def test_make_room_requires_the_priority_policy(option):
    with pytest.raises(ValueError, match="the priority scheduling policy"):
        SchedulerConfig.default_factory(policy="fcfs", **option)


def test_preempts_the_lowest_priority_first_and_only_as_many_as_needed():
    """7 blocks with 2 free: one priority 2 request makes room. The priority 1
    request holds more blocks but goes last, and the other priority 2 requests
    keep running."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=4)
    _run_three_low_priority_requests(scheduler)

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=4))
    output = _step(scheduler)

    assert output.preempted_req_ids == {"big"}
    assert output.num_scheduled_tokens.keys() == {"hi", "p1", "small", "other"}


def test_preempts_the_request_of_the_most_blocks_then_the_latest():
    """Within a priority, the request whose sequence takes the most blocks goes
    first; between requests of as many blocks, the latest arrival, though "early"
    is 4 tokens longer."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(12), priority=0, arrival_time=0),
        _request("early", _num_tokens(6) + 4, priority=2, arrival_time=1),
        _request("large", _num_tokens(7), priority=2, arrival_time=2),
        _request("late", _num_tokens(6), priority=2, arrival_time=3),
    )

    scheduler.add_request(_request("hi-1", _num_tokens(8), priority=0, arrival_time=4))
    assert _step(scheduler).preempted_req_ids == {"large"}

    scheduler.add_request(_request("hi-2", _num_tokens(6), priority=0, arrival_time=5))
    assert _step(scheduler).preempted_req_ids == {"late"}


def test_preempting_an_inflight_prefill_frees_its_reservation():
    """The waiting request needs 7 blocks beside the 8 that an in-flight prefill
    still needs, with 8 free. The prefill holds 4 blocks and reserves 8, so
    preempting it frees 12, more than the 8 of the decode: it goes alone."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, max_num_batched_tokens=64
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(12), priority=0, arrival_time=0),
        _request("decode", _num_tokens(8), priority=2, arrival_time=1),
    )
    prefill = _request("prefill", 12 * BLOCK_SIZE, priority=2, arrival_time=2)
    scheduler.add_request(prefill)
    # Admitted, as its 12 blocks fit: 62 of its 192 tokens beside the decodes.
    _step(scheduler)
    assert prefill.is_prefill_chunk and _num_free_blocks(scheduler) == 8

    scheduler.add_request(_request("hi", 7 * BLOCK_SIZE, priority=0, arrival_time=3))
    assert _step(scheduler).preempted_req_ids == {"prefill"}


def test_no_preemption_beyond_the_per_step_cap():
    """15 blocks with 2 free take two preemptions, so a cap of one preempts
    none."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=1)
    _run_three_low_priority_requests(scheduler)

    scheduler.add_request(_request("hi", _num_tokens(15), priority=0, arrival_time=4))
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens


def test_preempted_request_is_not_resumed_in_the_same_step():
    """The room counts the 8 blocks that an in-flight prefill still needs: the
    waiting request of 4 blocks, with 10 free, preempts the decode of 14. The
    prefill takes 4 blocks in the running loop and the waiting request 4, so 16
    stay free, enough to resume the decode; it waits for the next step. The
    prefill is of the decode's priority, so nothing else holds the decode back."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, long_prefill_token_threshold=64
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(4), priority=0, arrival_time=0),
        _request("bg", _num_tokens(14), priority=2, arrival_time=1),
    )
    prefill = _request("prefill", 12 * BLOCK_SIZE, priority=2, arrival_time=2)
    scheduler.add_request(prefill)
    _step(scheduler)
    assert prefill.is_prefill_chunk and _num_free_blocks(scheduler) == 10

    scheduler.add_request(_request("hi", _num_tokens(4), priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert output.num_scheduled_tokens.keys() == {"rt", "prefill", "hi"}
    assert _num_free_blocks(scheduler) == 16


def test_preempted_request_waits_for_the_prefill_it_made_room_for():
    """With prefix caching, the blocks of a preempted request stay cached, and
    resuming it takes them all back. Once the deferred frees land, the waiting
    request is admitted with the first of its three chunks; the preempted
    request then waits for the other two, which the in-flight prefill still
    needs, or the next chunk would find no free block."""
    scheduler = _create_deferring_scheduler(
        max_priority_preemptions_per_step=2,
        long_prefill_token_threshold=2 * BLOCK_SIZE,
        enable_prefix_caching=True,
    )
    schedule, process = _in_flight_steps(scheduler)
    rt = _request("rt", _num_tokens(14), priority=0, arrival_time=0, prompt_token=1)
    bg = _request("bg", _num_tokens(16), priority=2, arrival_time=1, prompt_token=2)
    scheduler.add_request(rt)
    scheduler.add_request(bg)
    schedule()
    for _ in range(30):
        if all(
            request.status == RequestStatus.RUNNING and not request.is_prefill_chunk
            for request in (rt, bg)
        ):
            break
        schedule()
        process()
    assert _num_free_blocks(scheduler) == 2

    hi = _request("hi", 6 * BLOCK_SIZE, priority=0, arrival_time=2, prompt_token=3)
    scheduler.add_request(hi)
    output = schedule()
    assert output.preempted_req_ids == {"bg"}
    assert "hi" not in output.num_scheduled_tokens
    num_computed_tokens = []
    for _ in range(5):
        process()
        output = schedule()
        assert "bg" not in output.num_scheduled_tokens or not hi.is_prefill_chunk
        num_computed_tokens.append(hi.num_computed_tokens)
    # Three chunks of 32 tokens in a row, and only then may "bg" resume.
    assert num_computed_tokens[:3] == [32, 64, 96]


def test_room_includes_the_watermark():
    """With a watermark of 2 blocks, the waiting request of 7 blocks needs 9
    free, so freeing the 6 of one request with 2 free is not enough: a second
    request is preempted, and the waiting request runs."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2, watermark=0.07)
    assert scheduler.kv_cache_manager.watermark_blocks == 2
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(6), priority=2, arrival_time=1),
        _request("bg2", _num_tokens(2), priority=2, arrival_time=2),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg", "bg2"}
    assert "hi" in output.num_scheduled_tokens


def test_blocks_shared_by_prefix_caching_are_not_counted_as_freed():
    """Two priority 2 requests share 7 of their 8 blocks. Preempting one frees a
    single block, so with 1 free, the waiting request of 3 blocks takes both."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, enable_prefix_caching=True
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0, prompt_token=1),
        _request("a", _num_tokens(8), priority=2, arrival_time=1),
    )
    _run_until_decoding(
        scheduler, _request("b", _num_tokens(8), priority=2, arrival_time=2)
    )
    assert _num_free_blocks(scheduler) == 1

    scheduler.add_request(
        _request("hi", _num_tokens(3), priority=0, arrival_time=3, prompt_token=2)
    )
    output = _step(scheduler)
    assert output.preempted_req_ids == {"a", "b"}
    assert "hi" in output.num_scheduled_tokens


def test_freeable_blocks():
    """A released block returns to the free block pool only if no other request
    holds it, it is not the null block and it is of this pool (blocks of a host
    tier are not)."""
    scheduler = _create_scheduler()
    block_pool = scheduler.kv_cache_manager.block_pool
    own, shared = block_pool.get_new_blocks(2)
    block_pool.touch([shared])
    (other_pool_block,) = BlockPool(
        num_gpu_blocks=2, enable_caching=False, hash_block_size=BLOCK_SIZE
    ).get_new_blocks(1)
    blocks = [own, shared, other_pool_block, block_pool.null_block]
    assert scheduler._num_freeable_blocks(blocks) == 1


def test_pending_connector_block_frees_are_awaited(monkeypatch):
    """As in the running loop, blocks that the KV connector's pending transfers
    will free are awaited instead of preempting."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    # The mock connector has no transfers of its own.
    monkeypatch.setattr(scheduler.connector, "has_pending_block_frees", lambda: True)

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    assert not _step(scheduler).preempted_req_ids


@pytest.mark.parametrize("hold", ["defer_prefills", "pause"])
def test_no_room_in_a_step_that_admits_no_new_request(hold):
    """A step that holds prefills back (prefill_schedule_interval=2) or a paused
    scheduler admits no new request, so it makes room for none: the next step
    makes room and admits the waiting request."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        prefill_schedule_interval=2 if hold == "defer_prefills" else 1,
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    if hold == "pause":
        scheduler.set_pause_state(PauseState.PAUSED_NEW)
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens
    scheduler.set_pause_state(PauseState.UNPAUSED)
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert "hi" in output.num_scheduled_tokens


def _session_request(num_tokens: int, max_tokens: int, priority: int) -> Request:
    """An input chunk of the streaming-input session "session"."""
    return Request(
        request_id="session",
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        arrival_time=0,
        priority=priority,
        resumable=True,
    )


def _run_session_first_chunk(scheduler: Scheduler, priority: int) -> None:
    """Run the session's first chunk of 4 blocks; it then waits for its next
    input chunk holding them."""
    session = _session_request(_num_tokens(4), max_tokens=1, priority=priority)
    scheduler.add_request(session)
    _step(scheduler)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ


def test_makes_room_for_a_streaming_input_session():
    """The session's next chunk needs 6 more blocks with 2 free: it makes room
    like a new request, preempting the request of the most blocks of a lower
    priority."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_session_first_chunk(scheduler, priority=0)
    _run_until_decoding(
        scheduler,
        _request("bg", _num_tokens(20), priority=2, arrival_time=1),
        _request("bg2", _num_tokens(6), priority=2, arrival_time=2),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_session_request(100, max_tokens=16, priority=0))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert output.num_scheduled_tokens["session"] == 100


def test_a_request_holding_kv_blocks_is_not_held_back():
    """Requests holding KV blocks are drained first, so a priority 2 session's
    next chunk, which needs 6 more blocks with 12 free, is admitted although a
    priority 0 prefill in flight still needs 8."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, long_prefill_token_threshold=64
    )
    _run_session_first_chunk(scheduler, priority=2)
    _run_until_decoding(
        scheduler, _request("rt", _num_tokens(8), priority=0, arrival_time=1)
    )
    prefill = _request("prefill", 16 * BLOCK_SIZE, priority=0, arrival_time=2)
    scheduler.add_request(prefill)
    _step(scheduler)
    assert prefill.is_prefill_chunk and _num_free_blocks(scheduler) == 16

    scheduler.add_request(_session_request(100, max_tokens=16, priority=2))
    output = _step(scheduler)
    assert "session" in output.num_scheduled_tokens


def test_deferred_frees_count_as_free():
    """With deferred block frees (async scheduling with a KV consumer), the
    preempted request's blocks are freed once the step in flight is processed:
    the waiting request runs a step later, and the deferred blocks count as free
    meanwhile, so nothing else is preempted."""
    scheduler = _create_deferring_scheduler(max_priority_preemptions_per_step=4)
    schedule, process = _in_flight_steps(scheduler)
    scheduler.add_request(_request("rt", _num_tokens(16), priority=0, arrival_time=0))
    scheduler.add_request(_request("bg", _num_tokens(8), priority=2, arrival_time=1))
    scheduler.add_request(_request("bg2", _num_tokens(6), priority=2, arrival_time=2))
    schedule()
    for _ in range(3):  # One step stays in flight, as in the engine.
        schedule()
        process()
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=3))
    output = schedule()
    assert output.preempted_req_ids == {"bg"}
    assert "hi" not in output.num_scheduled_tokens
    # Before the step in flight is processed.
    output = schedule()
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens
    process()
    process()
    output = schedule()
    assert not output.preempted_req_ids
    assert {"hi", "bg2"} <= output.num_scheduled_tokens.keys()


@pytest.mark.parametrize("priority_prefill_budget_cap", [True, False])
def test_lower_priority_prefill_sits_out_a_requests_first_tokens(
    priority_prefill_budget_cap,
):
    """The priority 0 request that fits takes its whole prompt in a step that the
    running priority 2 prefill sits out; then the prefill and a waiting priority
    2 request sit out the steps until it has its first 4 tokens, while the
    priority 2 decode keeps its token, and both go on. The first tokens of the
    decode hold back no prefill of its own priority. With the cap off, the
    prefill is scheduled in every step."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=priority_prefill_budget_cap,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    hi = _request("hi", 100, priority=0, arrival_time=2)
    scheduler.add_request(hi)
    num_scheduled_tokens = _step(scheduler).num_scheduled_tokens
    assert num_scheduled_tokens["hi"] == 100
    assert ("prefill" in num_scheduled_tokens) != priority_prefill_budget_cap
    assert hi.num_output_tokens == 1

    scheduler.add_request(_request("late", 300, priority=2, arrival_time=3))
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(4)]
    assert all(tokens["decode"] == tokens["hi"] == 1 for tokens in steps)
    if priority_prefill_budget_cap:
        assert steps[:3] == [{"decode": 1, "hi": 1}] * 3
        assert steps[3]["prefill"] == 256 and steps[3]["late"] > 0
    else:
        assert all(tokens["prefill"] == 256 for tokens in steps)


def test_budget_cap_holds_nothing_back_for_a_request_that_does_not_fit():
    """A waiting priority 0 request of 10 blocks does not fit beside the priority
    2 prefill in flight, which still needs 10 of the 12 free blocks: the prefill,
    whose blocks it waits for, goes on."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True, max_num_batched_tokens=64
    )
    _run_until_decoding(
        scheduler, _request("decode", _num_tokens(16), priority=2, arrival_time=0)
    )
    scheduler.add_request(
        _request("prefill", 14 * BLOCK_SIZE, priority=2, arrival_time=1)
    )
    # 63 of its 224 tokens beside the decode.
    _step(scheduler)
    assert _num_free_blocks(scheduler) == 12

    scheduler.add_request(_request("hi", _num_tokens(10), priority=0, arrival_time=2))
    for _ in range(2):
        num_scheduled_tokens = _step(scheduler).num_scheduled_tokens
        assert num_scheduled_tokens["prefill"] == 63
        assert "hi" not in num_scheduled_tokens


def test_budget_cap_counts_deferred_frees():
    """With deferred block frees, the blocks of the request preempted to make
    room are free once the step in flight is processed, and the waiting request
    runs a step later. It fits meanwhile, as the deferred blocks count as free,
    so the priority 2 prefill sits out these steps too."""
    scheduler = _create_deferring_scheduler(
        max_priority_preemptions_per_step=2,
        priority_prefill_budget_cap=True,
        long_prefill_token_threshold=64,
    )
    schedule, process = _in_flight_steps(scheduler)
    scheduler.add_request(_request("bg", _num_tokens(16), priority=2, arrival_time=0))
    schedule()
    for _ in range(3):  # One step stays in flight, as in the engine.
        schedule()
        process()
    scheduler.add_request(
        _request("prefill", 12 * BLOCK_SIZE, priority=2, arrival_time=1)
    )
    schedule()
    process()
    assert _num_free_blocks(scheduler) == 12

    # 13 blocks and the 8 that the prefill still needs: 9 more than are free.
    scheduler.add_request(_request("hi", _num_tokens(13), priority=0, arrival_time=2))
    output = schedule()
    assert output.preempted_req_ids == {"bg"}
    assert not output.num_scheduled_tokens
    # Before the step in flight is processed.
    assert not schedule().num_scheduled_tokens
    process()
    process()
    assert schedule().num_scheduled_tokens == {"hi": 64}


@pytest.mark.parametrize("priority_prefill_budget_cap", [True, False])
def test_budget_cap_holds_prefill_while_a_request_loads_its_kv(
    priority_prefill_budget_cap,
):
    """A priority 0 request admitted with an async KV load is an in-flight
    prefill: the running priority 2 prefill sits out the steps while it loads,
    as once its load is done, the waiting loop promotes it only on budget the
    running loop leaves. The priority 2 decode keeps its token. With the cap
    off, the prefill gets its chunk."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=priority_prefill_budget_cap,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
        # Each request loads its first 256 tokens asynchronously.
        use_kv_connector=mock_kv(matched_tokens=256, is_async=True),
    )
    scheduler.add_request(_request("decode", 300, priority=2, arrival_time=0))
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    _step(scheduler, finished_recving={"decode", "prefill"})
    for _ in range(2):
        _step(scheduler)
    hi = _request("hi", 300, priority=0, arrival_time=2)
    scheduler.add_request(hi)
    _step(scheduler)
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    num_scheduled_tokens = _step(scheduler).num_scheduled_tokens
    if priority_prefill_budget_cap:
        assert num_scheduled_tokens == {"decode": 1}
    else:
        assert num_scheduled_tokens == {"decode": 1, "prefill": 256}


def test_budget_cap_holds_nothing_back_for_a_loading_request_without_a_slot():
    """With two running slots, a priority 2 request whose async KV load lands
    first is promoted into the second one, still prefilling. When the load of
    the priority 0 request lands, it has no slot to be promoted into, so it
    holds nothing back: both prefills go on, rather than sit out every step."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        long_prefill_token_threshold=64,
        max_num_seqs=2,
        max_model_len=2048,
        num_blocks=400,
        # Each request loads its first 256 tokens asynchronously.
        use_kv_connector=mock_kv(matched_tokens=256, is_async=True),
    )
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=0))
    _step(scheduler, finished_recving={"prefill"})
    _step(scheduler)
    scheduler.add_request(_request("loading", 1000, priority=2, arrival_time=1))
    _step(scheduler)
    scheduler.add_request(_request("hi", 300, priority=0, arrival_time=2))
    _step(scheduler, finished_recving={"loading"})

    # While a slot is free, "hi" holds the prefill back.
    num_scheduled_tokens = _step(
        scheduler, finished_recving={"hi"}
    ).num_scheduled_tokens
    assert num_scheduled_tokens == {"loading": 64}
    for _ in range(2):
        num_scheduled_tokens = _step(scheduler).num_scheduled_tokens
        assert num_scheduled_tokens == {"prefill": 64, "loading": 64}


def _trace(scheduler: Scheduler, priorities: list[int]) -> list[tuple]:
    """Each step's scheduled tokens, preempted and new requests, for 24 requests
    of mixed sizes arriving every other step, more than 48 blocks hold."""
    workload = deque(
        (
            2 * i,
            _request(
                str(i),
                (37 * i) % 200 + 20,
                priority=priority,
                arrival_time=i,
                max_tokens=(11 * i) % 40 + 4,
            ),
        )
        for i, priority in enumerate(priorities)
    )
    trace = []
    for step in range(160):
        while workload and workload[0][0] == step:
            scheduler.add_request(workload.popleft()[1])
        output = _step(scheduler)
        trace.append(
            (
                output.num_scheduled_tokens,
                sorted(output.preempted_req_ids or ()),
                sorted(req.req_id for req in output.scheduled_new_reqs),
            )
        )
    return trace


@pytest.mark.parametrize("mixed_priorities", [False, True])
def test_make_room_acts_only_across_priorities(mixed_priorities):
    """At one priority (the default 0) neither option acts, and the schedules
    match those with both off, through preemptions in the running loop and
    chunked prefills. Across priorities, the same workload is scheduled
    differently."""
    options = dict(
        num_blocks=48, max_num_batched_tokens=128, long_prefill_token_threshold=64
    )
    priorities = [i % 3 if mixed_priorities else 0 for i in range(24)]
    trace = _trace(_create_scheduler(**options), priorities)
    assert any(preempted for _, preempted, _ in trace)

    trace_with_make_room = _trace(
        _create_scheduler(
            **options,
            max_priority_preemptions_per_step=4,
            priority_prefill_budget_cap=True,
        ),
        priorities,
    )
    assert (trace_with_make_room != trace) == mixed_priorities
