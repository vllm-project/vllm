# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `max_priority_preemptions_per_step` and `priority_prefill_budget_cap`:
under the priority policy, the waiting request of the highest priority preempts
running requests of a lower priority for KV cache blocks, and prefills of a lower
priority sit out the steps of a request's first tokens.

Unless a test says otherwise, the KV cache holds 32 blocks of 16 tokens (33
with the null block) and prefix caching is off, so requests share no block.
"""

from collections import deque

import prometheus_client
import pytest
import torch

from vllm.config import (
    CacheConfig,
    LoRAConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.encoder_cache_manager import EncoderDecoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.prometheus import unregister_vllm_metrics
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

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
    num_encoder_tokens: int = 0,
) -> Request:
    """A request whose prompt is `num_tokens` tokens of `prompt_token`, with an
    encoder input of `num_encoder_tokens` if any."""
    encoder_inputs = {}
    if num_encoder_tokens:
        encoder_inputs = dict(
            mm_hashes_list=[[f"encoder-{request_id}"]],
            mm_positions=[[PlaceholderRange(offset=0, length=num_encoder_tokens)]],
        )
    return create_requests_with_priority(
        num_requests=1,
        priorities=[priority],
        arrival_times=[arrival_time],
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        starting_idx=prompt_token,
        req_ids=[request_id],
        **encoder_inputs,
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


@pytest.mark.parametrize(
    "option, message",
    [
        ({"max_priority_preemptions_per_step": -1}, "greater than or equal to 0"),
        ({"priority_prefill_budget_cap_tokens": 0}, "greater than or equal to 1"),
    ],
)
def test_the_options_take_their_ranges(option, message):
    with pytest.raises(ValueError, match=message):
        SchedulerConfig.default_factory(policy="priority", **option)


# The victims


def test_preempts_the_lowest_priority_first_and_only_as_many_as_needed():
    """7 blocks with 2 free: one priority 2 request makes room, the latest
    arrival, as the running loop would preempt. The priority 1 request holds
    more blocks but goes last, and the older priority 2 requests keep running."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=4)
    _run_three_low_priority_requests(scheduler)

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=4))
    output = _step(scheduler)

    assert output.preempted_req_ids == {"other"}
    assert output.num_scheduled_tokens.keys() == {"hi", "p1", "big", "small"}


def test_victims_go_in_the_running_loops_preemption_order():
    """Within a priority, the latest arrival goes first, as the running loop
    preempts, which spares older work, rather than the request that holds the
    most blocks: the 7 blocks of the latest make room for the first waiting
    request, and the 9 of the earlier one then go for the second."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(14), priority=0, arrival_time=0),
        _request("large", _num_tokens(9), priority=2, arrival_time=1),
        _request("late", _num_tokens(7), priority=2, arrival_time=2),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi-1", _num_tokens(8), priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"late"}
    assert "hi-1" in output.num_scheduled_tokens

    scheduler.add_request(_request("hi-2", _num_tokens(9), priority=0, arrival_time=4))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"large"}
    assert "hi-2" in output.num_scheduled_tokens


@pytest.mark.parametrize(
    "max_preemptions, victims", [(2, set()), (3, {"other", "small", "big"})]
)
def test_no_preemption_beyond_the_per_step_cap(max_preemptions, victims):
    """15 blocks with 2 free take three preemptions (of 6, 6 and 8 blocks), so a
    cap of two preempts none."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=max_preemptions)
    _run_three_low_priority_requests(scheduler)

    scheduler.add_request(_request("hi", _num_tokens(15), priority=0, arrival_time=4))
    output = _step(scheduler)
    assert (output.preempted_req_ids or set()) == victims
    assert ("hi" in output.num_scheduled_tokens) == bool(victims)


@pytest.mark.parametrize("peer_priority, victims", [(0, set()), (1, {"low", "peer"})])
def test_a_running_request_of_the_same_priority_is_no_victim(peer_priority, victims):
    """With 2 blocks free, 14 for the waiting request take both running requests'
    20 and 10 blocks: only when both have a lower priority than it. One of its
    own priority is no victim, beside one of a lower priority too, and the
    lower one alone cannot make room."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("peer", _num_tokens(20), priority=peer_priority, arrival_time=0),
        _request("low", _num_tokens(10), priority=2, arrival_time=1),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi", _num_tokens(14), priority=0, arrival_time=2))
    output = _step(scheduler)
    assert (output.preempted_req_ids or set()) == victims
    assert ("hi" in output.num_scheduled_tokens) == bool(victims)


def test_no_growth_is_counted_when_the_victims_allowed_cannot_make_room(
    monkeypatch,
):
    """The 15 blocks take three preemptions and two are allowed: the decodes'
    growth could only add to the shortfall, so make-room gives up without
    counting it."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_three_low_priority_requests(scheduler)

    def num_growth_blocks(request: Request, num_steps: int) -> int:
        raise AssertionError("counted the decodes' growth")

    monkeypatch.setattr(scheduler, "_num_growth_blocks", num_growth_blocks)
    scheduler.add_request(_request("hi", _num_tokens(15), priority=0, arrival_time=4))
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens


def test_no_growth_is_counted_when_the_request_fits_with_a_block_each_to_spare(
    monkeypatch,
):
    """A running request takes at most a block per KV cache group in a step, so
    with that many blocks to spare the waiting request fits without counting
    the decodes' growth: here 4 blocks of 16 free, beside 2 running requests."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("a", _num_tokens(8), priority=2, arrival_time=0),
        _request("b", _num_tokens(8), priority=2, arrival_time=1),
    )
    assert _num_free_blocks(scheduler) == 16

    def num_growth_blocks(request: Request, num_steps: int) -> int:
        raise AssertionError("counted the decodes' growth")

    monkeypatch.setattr(scheduler, "_num_growth_blocks", num_growth_blocks)
    scheduler.add_request(_request("hi", _num_tokens(12), priority=0, arrival_time=2))
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "hi" in output.num_scheduled_tokens


def test_the_growth_bound_holds_only_where_a_request_takes_a_block_a_group():
    """One block per KV cache group a step, without speculative tokens, from the
    managers that allocate by the token count."""
    assert _create_scheduler()._max_growth_blocks == 1
    assert _create_encoder_decoder_scheduler(num_blocks=65)._max_growth_blocks == 2
    assert _create_scheduler(num_speculative_tokens=2)._max_growth_blocks is None


def test_without_deferred_frees_the_growth_is_counted_once(monkeypatch):
    """Without deferred frees, one step of growth is counted until the victims'
    blocks are free: the sum the fit is checked against, once for each running
    request, and then each victim's own as it is added."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=4)
    _run_three_low_priority_requests(scheduler)
    num_growth_blocks = scheduler._num_growth_blocks
    counted: list[tuple[str, int]] = []

    def counting(request: Request, num_steps: int) -> int:
        counted.append((request.request_id, num_steps))
        return num_growth_blocks(request, num_steps)

    monkeypatch.setattr(scheduler, "_num_growth_blocks", counting)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=4))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"other"}
    assert sorted(counted) == sorted(
        [("p1", 1), ("big", 1), ("small", 1), ("other", 1), ("other", 1)]
    )


def test_preempting_an_inflight_prefill_frees_its_reservation():
    """The waiting request needs 7 blocks beside the 8 that an in-flight prefill
    still needs, with 8 free. The prefill, the latest arrival, holds 4 blocks
    and reserves 8, so preempting it frees 12, enough alone."""
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


def test_preempted_request_is_not_resumed_in_the_same_step():
    """The room counts the 8 blocks that an in-flight prefill still needs: the
    waiting request of 4 blocks, with 10 free, preempts the decode of 14, the
    latest arrival. The prefill takes 4 blocks in the running loop and the
    waiting request 4, so 16 stay free, enough to resume the decode; it waits
    for the next step. The prefill is of the decode's priority, so nothing else
    holds the decode back."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, long_prefill_token_threshold=64
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(4), priority=0, arrival_time=0),
        _request("bg", _num_tokens(14), priority=2, arrival_time=2),
    )
    prefill = _request("prefill", 12 * BLOCK_SIZE, priority=2, arrival_time=1)
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


# The room


def test_room_includes_the_watermark():
    """With a watermark of 2 blocks, the waiting request of 7 blocks needs 9
    free, so with 2 free, freeing the 6 blocks of the latest arrival is not
    enough: the request of 2 is preempted too, and the waiting request runs."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2, watermark=0.07)
    assert scheduler.kv_cache_manager.watermark_blocks == 2
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(6), priority=2, arrival_time=2),
        _request("bg2", _num_tokens(2), priority=2, arrival_time=1),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg", "bg2"}
    assert "hi" in output.num_scheduled_tokens


def test_room_includes_the_decodes_next_blocks():
    """The running loop allocates the decodes' next blocks before the waiting
    loop admits anyone. Here the next token of "rt" starts a block, so the
    waiting request, which needs the 4 free blocks and 6 more, preempts both
    requests of a lower priority and runs in the same step, rather than a block
    short until the next step."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    rt = _request("rt", 20 * BLOCK_SIZE - 3, priority=0, arrival_time=0)
    _run_until_decoding(
        scheduler,
        rt,
        _request("bg", 6 * BLOCK_SIZE - 12, priority=2, arrival_time=1),
        _request("bg2", 2 * BLOCK_SIZE - 12, priority=2, arrival_time=2),
    )
    for _ in range(4):
        if rt.num_tokens % BLOCK_SIZE == 1:
            break
        _step(scheduler)
    assert rt.num_tokens % BLOCK_SIZE == 1 and _num_free_blocks(scheduler) == 4

    scheduler.add_request(_request("hi", 10 * BLOCK_SIZE, priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg", "bg2"}
    assert "hi" in output.num_scheduled_tokens


def test_room_is_net_of_the_waiting_requests_prefix_cache_hit():
    """The waiting request shares an 18-block prefix that a priority 2 request
    holds: while that request holds them, those blocks cost it nothing, so it
    lacks only 3 of the 5 blocks of its own (2 free). The other request, the
    latest arrival, gives them, and the waiting request runs beside the one that
    holds its prefix."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, enable_prefix_caching=True
    )
    holder = _request(
        "holder", 18 * BLOCK_SIZE + 8, priority=2, arrival_time=1, prompt_token=5
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(5), priority=0, arrival_time=0, prompt_token=1),
        holder,
        _request("bg", _num_tokens(6), priority=2, arrival_time=2, prompt_token=2),
    )
    assert _num_free_blocks(scheduler) == 2

    hi = _request("hi", 18 * BLOCK_SIZE + 72, 0, arrival_time=3, prompt_token=5)
    scheduler.add_request(hi)
    output = _step(scheduler)

    assert output.preempted_req_ids == {"bg"}
    assert output.num_scheduled_tokens.get("hi") == 72
    assert holder.status == RequestStatus.RUNNING


def test_a_preempted_holder_of_the_waiting_requests_prefix_releases_none_of_it():
    """As above, but the request holding the 18-block prefix is the latest
    arrival and goes first: preempting it only puts those blocks in the free
    queue, which the waiting request takes them from as hits, a free block each.
    So it releases only its own block toward the 3 the waiting request lacks,
    and the other request goes too."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, enable_prefix_caching=True
    )
    holder = _request(
        "holder", 18 * BLOCK_SIZE + 8, priority=2, arrival_time=2, prompt_token=5
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(5), priority=0, arrival_time=0, prompt_token=1),
        _request("bg", _num_tokens(6), priority=2, arrival_time=1, prompt_token=2),
        holder,
    )
    assert _num_free_blocks(scheduler) == 2

    hi = _request("hi", 18 * BLOCK_SIZE + 72, 0, arrival_time=3, prompt_token=5)
    scheduler.add_request(hi)
    output = _step(scheduler)

    assert output.preempted_req_ids == {"holder", "bg"}
    assert output.num_scheduled_tokens.get("hi") == 72
    assert hi.num_computed_tokens == 18 * BLOCK_SIZE + 72


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


def test_the_inflight_gate_counts_a_waiting_requests_prefix_cache_hit():
    """While a priority 0 prefill is in flight, a priority 2 request is admitted
    only beside the blocks that prefill still needs. Its 8-block prefix, which a
    running request holds, costs it nothing, so with 15 free, the 8 the prefill
    still needs and its own 2 blocks fit."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        long_prefill_token_threshold=64,
        enable_prefix_caching=True,
    )
    _run_until_decoding(
        scheduler,
        _request("holder", 8 * BLOCK_SIZE + 8, 2, arrival_time=0, prompt_token=5),
    )
    prefill = _request("prefill", 16 * BLOCK_SIZE, 0, arrival_time=1, prompt_token=1)
    scheduler.add_request(prefill)
    _step(scheduler)
    assert prefill.is_prefill_chunk

    scheduler.add_request(
        _request("late", 8 * BLOCK_SIZE + 24, 2, arrival_time=2, prompt_token=5)
    )
    output = _step(scheduler)
    assert _num_free_blocks(scheduler) == 13
    assert output.num_scheduled_tokens.get("late") == 24


def test_released_blocks(monkeypatch):
    """Preempted requests release a block once every holder of it is one of them
    or a deferred free, unless the waiting request takes it as a prefix-cache
    hit; never the null block, nor a block of another pool (a host tier's)."""
    scheduler = _create_scheduler()
    block_pool = scheduler.kv_cache_manager.block_pool
    own_a, own_b, shared, hit_block, deferred = block_pool.get_new_blocks(5)
    block_pool.touch([shared])
    (other_pool_block,) = BlockPool(
        num_gpu_blocks=2, enable_caching=False, hash_block_size=BLOCK_SIZE
    ).get_new_blocks(1)
    held = {
        "a": [own_a, shared, hit_block, other_pool_block, block_pool.null_block],
        "b": [own_b, shared],
    }
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "get_blocks",
        lambda request_id: KVCacheBlocks((held[request_id],)),
    )
    scheduler.deferred_frees.append((0, [deferred]))
    a = _request("a", BLOCK_SIZE, priority=2, arrival_time=0)
    b = _request("b", BLOCK_SIZE, priority=2, arrival_time=1)
    hit = (([hit_block],), BLOCK_SIZE)

    assert scheduler._num_released_blocks([], hit) == 1
    assert scheduler._num_released_blocks([a], hit) == 2
    assert scheduler._num_released_blocks([a, b], hit) == 4
    assert scheduler._num_released_blocks([a, b], None) == 5


def test_no_blocks_are_released_without_victims_or_deferred_frees():
    """And the prefix-cache hit, as long as the request's prompt, is not read."""

    class Unread(list):
        def __iter__(self):
            raise AssertionError("read the hit's blocks")

    scheduler = _create_scheduler()
    assert scheduler._num_released_blocks([], ((Unread(),), BLOCK_SIZE)) == 0


def test_the_request_is_found_without_iterating_its_queue(monkeypatch):
    """Iterating a priority queue copies its heap: make-room takes the queue's
    first request when the waiting loop can schedule it, as it mostly can."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=4)
    _run_three_low_priority_requests(scheduler)

    def iterate(queue):
        raise AssertionError("iterated a waiting queue")

    monkeypatch.setattr(type(scheduler.waiting), "__iter__", iterate)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=4))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"other"}
    assert "hi" in output.num_scheduled_tokens


def test_no_prefix_cache_lookup_without_a_lower_priority_running(monkeypatch):
    """When nothing runs at a lower priority than the waiting request, there is
    nothing to preempt for it, and make-room spares its prefix cache lookup."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, enable_prefix_caching=True
    )
    _run_until_decoding(
        scheduler,
        _request("a", _num_tokens(15), priority=1, arrival_time=0, prompt_token=1),
        _request("b", _num_tokens(15), priority=1, arrival_time=1, prompt_token=2),
    )

    def local_prefix_hit(request: Request):
        raise AssertionError("looked the waiting request's prefix up")

    monkeypatch.setattr(scheduler, "_local_prefix_hit", local_prefix_hit)
    scheduler.add_request(
        _request("c", _num_tokens(8), priority=1, arrival_time=2, prompt_token=3)
    )
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "c" not in output.num_scheduled_tokens


def _create_encoder_decoder_scheduler(num_blocks: int) -> Scheduler:
    """A scheduler with a self-attention and a cross-attention KV cache group,
    patched to schedule as for an encoder-decoder model, as in
    test_scheduler.py."""
    model_config = ModelConfig(
        model="facebook/opt-125m", trust_remote_code=True, dtype="float16", seed=42
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=True,
        policy="priority",
        max_priority_preemptions_per_step=2,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, cache_dtype="auto", enable_prefix_caching=False
    )
    cache_config.num_gpu_blocks = num_blocks
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    spec = dict(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["self_attn_layer"], FullAttentionSpec(**spec)),
            KVCacheGroupSpec(["cross_attn_layer"], CrossAttentionSpec(**spec)),
        ],
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.is_encoder_decoder = True
    scheduler.max_num_encoder_input_tokens = 512
    scheduler.encoder_cache_manager = EncoderDecoderCacheManager(cache_size=512)
    return scheduler


def test_room_includes_the_cross_attention_blocks_of_an_encoder_decoder_model():
    """An encoder-decoder request is admitted with blocks for its encoder output
    too, which cross-attention reads: the waiting request needs 2 blocks for its
    prompt and 15 for its encoder output, 17 with 11 free. The latest arrival of
    a lower priority gives back its 4 and its 5, which is enough."""
    scheduler = _create_encoder_decoder_scheduler(num_blocks=65)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(4), 0, arrival_time=0, num_encoder_tokens=160),
        _request("bg", _num_tokens(10), 2, arrival_time=1, num_encoder_tokens=320),
        _request("bg2", _num_tokens(4), 2, arrival_time=2, num_encoder_tokens=80),
    )
    bg2_blocks = scheduler.kv_cache_manager.get_blocks("bg2").blocks
    assert [len(group) for group in bg2_blocks] == [4, 5]
    assert _num_free_blocks(scheduler) == 11

    scheduler.add_request(
        _request("hi", _num_tokens(2), 0, arrival_time=3, num_encoder_tokens=240)
    )
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg2"}
    assert "hi" in output.num_scheduled_tokens
    assert _num_free_blocks(scheduler) == 3


@pytest.mark.parametrize("drafts", [False, True])
def test_room_includes_the_decodes_draft_tokens(drafts):
    """With speculative decoding, a running request's step takes slots for its
    draft tokens too: three decodes one token short of filling their last block
    take a block each for two drafts. The waiting request's 60 tokens, with its
    own draft slots, need 4 blocks, as many as are free: it fits beside the
    decodes without drafts, and with them takes a preemption."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, num_speculative_tokens=2
    )
    running = [
        _request("p1", 10 * BLOCK_SIZE - 2, priority=1, arrival_time=0),
        _request("bg1", 9 * BLOCK_SIZE - 2, priority=2, arrival_time=1),
        _request("bg2", 9 * BLOCK_SIZE - 2, priority=2, arrival_time=2),
    ]
    _run_until_decoding(scheduler, *running)
    assert _num_free_blocks(scheduler) == 4
    if drafts:
        scheduler.update_draft_token_ids(
            DraftTokenIds([req.request_id for req in running], [[1, 2]] * 3)
        )

    scheduler.add_request(
        _request("hi", 4 * BLOCK_SIZE - 4, priority=0, arrival_time=3)
    )
    output = _step(scheduler)
    assert (output.preempted_req_ids or set()) == ({"bg2"} if drafts else set())
    assert "hi" in output.num_scheduled_tokens


def _create_sliding_window_scheduler() -> Scheduler:
    """A scheduler with a full-attention and a sliding-window KV cache group, as
    for a model that interleaves the two, a window of 4 blocks, 64-token steps
    and 64 blocks of 16 tokens."""
    model_config = ModelConfig(
        model="facebook/opt-125m", trust_remote_code=True, dtype="float16", seed=42
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=32 * BLOCK_SIZE,
        is_encoder_decoder=False,
        policy="priority",
        max_priority_preemptions_per_step=2,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE, cache_dtype="auto", enable_prefix_caching=False
    )
    cache_config.num_gpu_blocks = 65
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    spec = dict(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv_cache_config = KVCacheConfig(
        num_blocks=65,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full_attn_layer"], FullAttentionSpec(**spec)),
            KVCacheGroupSpec(
                ["sliding_attn_layer"],
                SlidingWindowSpec(**spec, sliding_window=4 * BLOCK_SIZE),
            ),
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


@pytest.mark.parametrize("num_blocks, victims", [(24, {"bg2"}), (26, {"bg", "bg2"})])
def test_room_counts_a_sliding_window_groups_blocks_in_its_window(num_blocks, victims):
    """A running request's sliding-window group holds only its window's blocks,
    and a waiting request's needs only as many as its window and a step span:
    bg2, the latest arrival, gives back its 10 full-attention blocks and 5
    sliding-window ones. With 18 free, a request of 24 blocks needs 24 + 9, which
    bg2 makes room for; one of 26 blocks needs bg too."""
    scheduler = _create_sliding_window_scheduler()
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(6), priority=0, arrival_time=0),
        _request("bg", _num_tokens(14), priority=2, arrival_time=1),
        _request("bg2", _num_tokens(10), priority=2, arrival_time=2),
    )
    for _ in range(3):
        _step(scheduler)
    bg2_blocks = scheduler.kv_cache_manager.get_blocks("bg2").blocks
    assert [sum(not b.is_null for b in group) for group in bg2_blocks] == [10, 5]
    assert _num_free_blocks(scheduler) == 18

    scheduler.add_request(
        _request("hi", _num_tokens(num_blocks), priority=0, arrival_time=3)
    )
    output = _step(scheduler)
    assert output.preempted_req_ids == victims
    assert "hi" in output.num_scheduled_tokens


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


# Deferred block frees


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
    assert output.preempted_req_ids == {"bg2"}
    assert "hi" not in output.num_scheduled_tokens
    # Before the step in flight is processed.
    output = schedule()
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens
    process()
    process()
    output = schedule()
    assert not output.preempted_req_ids
    assert {"hi", "bg"} <= output.num_scheduled_tokens.keys()


def test_with_deferred_frees_the_room_counts_two_steps_of_decode_growth():
    """With deferred frees the waiting request is admitted a step after its
    victims are preempted, so the room counts the running decodes' blocks for
    both steps. Here the next block of "rt" starts in the second: the 6 blocks
    of the latest arrival cover the 6 the waiting request lacks now but not the
    7 it lacks then, so it takes both requests of a lower priority at once and
    runs in the next step, rather than a block short then."""
    scheduler = _create_deferring_scheduler(max_priority_preemptions_per_step=2)
    schedule, process = _in_flight_steps(scheduler)
    rt = _request("rt", 20 * BLOCK_SIZE - 8, priority=0, arrival_time=0, max_tokens=400)
    for request in (
        rt,
        _request("bg", 6 * BLOCK_SIZE - 15, 2, arrival_time=2, max_tokens=400),
        _request("bg2", 2 * BLOCK_SIZE - 15, 2, arrival_time=1, max_tokens=400),
    ):
        scheduler.add_request(request)
    schedule()
    # Until the position rt's next token takes is a block's last: the one after
    # it starts a block.
    for _ in range(10):
        if (rt.num_tokens + rt.num_output_placeholders) % BLOCK_SIZE == 0:
            break
        schedule()
        process()
    assert (rt.num_tokens + rt.num_output_placeholders) % BLOCK_SIZE == 0
    assert _num_free_blocks(scheduler) == 4

    scheduler.add_request(_request("hi", 10 * BLOCK_SIZE, 0, arrival_time=3))
    output = schedule()
    assert output.preempted_req_ids == {"bg", "bg2"}
    process()
    output = schedule()
    assert not output.preempted_req_ids
    assert "hi" in output.num_scheduled_tokens


def test_with_deferred_frees_a_request_that_fits_now_preempts_nothing():
    """The room counts the decodes' growth until the victims' frees land only
    for a waiting request that lacks blocks now: one that fits in this step,
    after the running loop, runs at once, though the next block of "rt" starts
    in the next step and it would then lack one."""
    scheduler = _create_deferring_scheduler(max_priority_preemptions_per_step=2)
    schedule, process = _in_flight_steps(scheduler)
    rt = _request("rt", 20 * BLOCK_SIZE - 8, priority=0, arrival_time=0, max_tokens=400)
    for request in (
        rt,
        _request("bg", 6 * BLOCK_SIZE - 15, 2, arrival_time=1, max_tokens=400),
        _request("bg2", 2 * BLOCK_SIZE - 15, 2, arrival_time=2, max_tokens=400),
    ):
        scheduler.add_request(request)
    schedule()
    for _ in range(10):
        if (rt.num_tokens + rt.num_output_placeholders) % BLOCK_SIZE == 0:
            break
        schedule()
        process()
    assert (rt.num_tokens + rt.num_output_placeholders) % BLOCK_SIZE == 0
    assert _num_free_blocks(scheduler) == 4

    scheduler.add_request(_request("hi", 4 * BLOCK_SIZE, 0, arrival_time=3))
    output = schedule()
    assert not output.preempted_req_ids
    assert "hi" in output.num_scheduled_tokens


# The waiting request room is made for


def _start_loading(scheduler: Scheduler, monkeypatch) -> Request:
    """A priority 0 request of 19 blocks whose KV connector loads its first 16
    asynchronously, after "rt" (4 blocks) and "bg" (priority 2, 8 blocks): the
    waiting loop admits it to load them, so 4 blocks stay free and it still
    needs 3."""
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(4), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda request, num_computed_tokens: (
            (16 * BLOCK_SIZE, True) if request.request_id == "load" else (0, False)
        ),
    )
    load = _request("load", 300, priority=0, arrival_time=2)
    scheduler.add_request(load)
    _step(scheduler)
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _num_free_blocks(scheduler) == 4
    return load


def test_a_loading_request_does_not_hide_the_requests_behind_it(monkeypatch):
    """A priority 0 request whose KV load is in flight waits ahead of a later
    priority 0 request. The waiting loop cannot schedule it until the load is
    done, so room is made for the request behind it, though the loading request
    would fit."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    load = _start_loading(scheduler, monkeypatch)

    scheduler.add_request(_request("hi", _num_tokens(7), priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert "hi" in output.num_scheduled_tokens
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_loading_request_counts_the_blocks_it_still_needs_once(monkeypatch):
    """A request whose KV load is in flight reserves the blocks it still needs
    among the in-flight prefills' reservations, so its own shortfall counts them
    once."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    load = _start_loading(scheduler, monkeypatch)
    assert scheduler._inflight_prefill_reserved_blocks() == 3
    assert scheduler._num_blocks_short(load, None) == 3 - 4


def test_a_request_compiling_its_grammar_does_not_hide_the_requests_behind_it():
    """A priority 0 request whose structured-output grammar is still compiling
    waits ahead of a later priority 0 request. The waiting loop cannot schedule
    it yet, so room is made for the request behind it, though the first would
    fit."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    compiling = _request("compiling", BLOCK_SIZE, priority=0, arrival_time=2)
    compiling.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    scheduler.add_request(compiling)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=3))
    output = _step(scheduler)

    assert output.preempted_req_ids == {"bg"}
    assert "hi" in output.num_scheduled_tokens
    assert compiling.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


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
    like a new request, preempting the latest arrival of a lower priority."""
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
    assert output.preempted_req_ids == {"bg2"}
    assert output.num_scheduled_tokens["session"] == 100


def test_a_session_awaiting_input_does_not_hide_the_requests_behind_it():
    """A priority 0 session awaiting its next input chunk waits ahead of a
    priority 0 request that arrives later. The waiting loop cannot schedule the
    session, so room is made for the request behind it, as without the
    session."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_session_first_chunk(scheduler, priority=0)
    _run_until_decoding(
        scheduler, _request("bg", _num_tokens(20), priority=2, arrival_time=1)
    )
    assert _num_free_blocks(scheduler) == 8

    scheduler.add_request(_request("hi", _num_tokens(10), priority=0, arrival_time=2))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert "hi" in output.num_scheduled_tokens


def test_no_room_for_a_request_awaiting_output_from_before_its_preemption():
    """With several steps in flight (pipeline parallelism), a preempted
    request's output can still be in flight steps after its preemption, and the
    waiting loop skips it until that output is delivered: no room is made for it
    meanwhile. The test sets that state as _preempt_request does."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    hi = _request("hi", 100, priority=0, arrival_time=2)
    hi.num_stale_output_tokens, hi.drop_stale_output = 1, False
    scheduler.add_request(hi)
    for _ in range(2):
        output = _step(scheduler)
        assert not output.preempted_req_ids
        assert "hi" not in output.num_scheduled_tokens

    hi.num_stale_output_tokens = 0  # Delivered.
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert "hi" in output.num_scheduled_tokens


def test_no_room_for_a_request_whose_lora_adapter_cannot_join_the_running_ones():
    """Under max_loras 2, with two priority 0 requests running on adapters 1 and
    2, the waiting loop skips a priority 0 request on adapter 3. No room is made
    for it: preempting the priority 2 request would only see it resume in the
    next step and be preempted again, for as long as the adapters stay taken. It
    keeps its tokens, as with make-room off."""

    def bg_tokens(max_preemptions: int) -> int:
        scheduler = _create_scheduler(
            max_priority_preemptions_per_step=max_preemptions,
            lora_config=LoRAConfig(max_loras=2),
            max_model_len=32 * BLOCK_SIZE,
        )
        requests = [
            _request("rt1", _num_tokens(4), 0, arrival_time=0, max_tokens=400),
            _request("rt2", _num_tokens(4), 0, arrival_time=1, max_tokens=400),
            _request("bg", _num_tokens(20), 2, arrival_time=2, max_tokens=400),
        ]
        requests[0].lora_request = LoRARequest("lora-1", 1, "/path/1")
        requests[1].lora_request = LoRARequest("lora-2", 2, "/path/2")
        _run_until_decoding(scheduler, *requests)
        assert _num_free_blocks(scheduler) == 4
        hi = _request("hi", _num_tokens(6), priority=0, arrival_time=3)
        hi.lora_request = LoRARequest("lora-3", 3, "/path/3")
        scheduler.add_request(hi)
        outputs = [_step(scheduler) for _ in range(12)]
        assert not any(output.preempted_req_ids for output in outputs)
        assert not any("hi" in output.num_scheduled_tokens for output in outputs)
        return sum(output.num_scheduled_tokens["bg"] for output in outputs)

    assert bg_tokens(2) == bg_tokens(0) == 12


def test_no_second_round_for_a_request_the_waiting_loop_skipped():
    """Room is made for the waiting request, and the waiting loop then skips it:
    here its KV connector cannot yet tell its matched tokens, for 8 steps. The
    preempted request resumes in the next step, and no room is made for the
    waiting request again until the waiting loop has tried its allocation: so
    the request of a lower priority is preempted once while the waiting request
    is skipped, rather than every other step. Once tried and short, room is
    made again."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(
            matched_tokens=0, is_async=False, num_defers_before_matching=8
        ),
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(10), 0, arrival_time=0, max_tokens=400),
        _request("big", _num_tokens(14), 2, arrival_time=1, max_tokens=400),
        _request("bg", _num_tokens(6), 2, arrival_time=2, max_tokens=400),
    )
    assert _num_free_blocks(scheduler) == 2

    scheduler.add_request(_request("hi", _num_tokens(7), priority=0, arrival_time=3))
    outputs = [_step(scheduler) for _ in range(8)]
    assert [output.preempted_req_ids or set() for output in outputs] == [{"bg"}] + [
        set()
    ] * 7
    assert not any("hi" in output.num_scheduled_tokens for output in outputs)
    # It resumed in the next step.
    assert all("bg" in output.num_scheduled_tokens for output in outputs[1:])

    output = _step(scheduler)  # Tried, and short.
    assert not output.preempted_req_ids
    assert "hi" not in output.num_scheduled_tokens
    output = _step(scheduler)
    assert "bg" in output.preempted_req_ids
    assert "hi" in output.num_scheduled_tokens


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


# The budget cap


@pytest.mark.parametrize("long_prefill_token_threshold", [0, 256])
def test_budget_cap_gives_a_waiting_request_that_fits_the_step(
    long_prefill_token_threshold,
):
    """A waiting priority 0 request that fits takes its first chunk (its 300
    tokens, or the long-prefill threshold) in a step that a running priority 2
    prefill sits out; the priority 2 decode still gets its token. With the cap
    off, the prefill gets the budget first."""
    first_chunk = long_prefill_token_threshold or 300

    def hi_step(priority_prefill_budget_cap: bool) -> dict[str, int]:
        scheduler = _create_scheduler(
            priority_prefill_budget_cap=priority_prefill_budget_cap,
            long_prefill_token_threshold=long_prefill_token_threshold,
            max_model_len=2048,
            num_blocks=400,
        )
        _run_until_decoding(
            scheduler, _request("decode", 40, priority=2, arrival_time=0)
        )
        scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
        _step(scheduler)
        scheduler.add_request(_request("hi", 300, priority=0, arrival_time=2))
        return _step(scheduler).num_scheduled_tokens

    off = hi_step(False)
    assert off["decode"] == 1 and off.get("hi", 0) < first_chunk
    assert hi_step(True) == {"decode": 1, "hi": first_chunk}


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


def test_budget_cap_never_cuts_a_decode():
    """A priority 2 decode scheduled after the prefill that sits out the step
    keeps its token."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=0))
    _step(scheduler)
    # It runs after the prefill.
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=1))
    scheduler.add_request(_request("hi", 300, priority=0, arrival_time=2))
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "hi": 256}


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
    room, the latest arrival, are free once the step in flight is processed, and
    the waiting request runs a step later. It fits meanwhile, as the deferred
    blocks count as free, so the priority 2 prefill sits out these steps too."""
    scheduler = _create_deferring_scheduler(
        max_priority_preemptions_per_step=2,
        priority_prefill_budget_cap=True,
        long_prefill_token_threshold=64,
    )
    schedule, process = _in_flight_steps(scheduler)
    scheduler.add_request(_request("bg", _num_tokens(16), priority=2, arrival_time=1))
    schedule()
    for _ in range(3):  # One step stays in flight, as in the engine.
        schedule()
        process()
    scheduler.add_request(
        _request("prefill", 12 * BLOCK_SIZE, priority=2, arrival_time=0)
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


def test_no_room_while_the_budget_cap_holds_the_waiting_request_back():
    """While the first 4 tokens of a priority 0 request are due, the budget cap
    holds back the waiting priority 1 request, which lacks 2 blocks: no room is
    made for it meanwhile, as it would not be admitted. Then the priority 2
    request is preempted for it, and it runs."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
    )
    _run_until_decoding(
        scheduler, _request("bg", _num_tokens(20), priority=2, arrival_time=0)
    )
    rt = _request("rt", _num_tokens(4), priority=0, arrival_time=1)
    scheduler.add_request(rt)
    scheduler.add_request(_request("mid", _num_tokens(10), priority=1, arrival_time=2))
    outputs = [_step(scheduler) for _ in range(4)]
    assert not any(output.preempted_req_ids for output in outputs)
    assert not any("mid" in output.num_scheduled_tokens for output in outputs)
    assert rt.num_output_tokens == 4

    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert "mid" in output.num_scheduled_tokens


def test_a_lower_priority_request_holding_kv_sits_out_a_requests_first_tokens():
    """The waiting loop drains first the requests that hold KV blocks, as one
    does whose KV load is done. A priority 2 one sits out the first tokens of a
    priority 0 request like any lower-priority prefill, but it is skipped rather
    than waited on, so the loop goes on to the priority 0 request behind it,
    which runs in the same step; after its first 4 tokens, the priority 2 one
    prefills."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    late = _request("late", 2000, priority=2, arrival_time=1)
    # 256 tokens of its KV loaded.
    assert scheduler.kv_cache_manager.allocate_slots(late, 256) is not None
    late.num_computed_tokens = 256
    scheduler.add_request(late)
    assert late in scheduler.kv_holding_waiting
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))

    steps = [_step(scheduler).num_scheduled_tokens for _ in range(5)]
    assert steps[:4] == [{"decode": 1, "hi": 100}] + [{"decode": 1, "hi": 1}] * 3
    assert steps[4] == {"decode": 1, "hi": 1, "late": 256}


def _match_tokens(scheduler: Scheduler, monkeypatch, matched: dict[str, int]):
    """The KV connector finds `matched[request_id]` tokens of a request outside
    the GPU and loads them asynchronously; none of the others'."""
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda request, num_computed_tokens: (
            (matched[request.request_id], True)
            if request.request_id in matched
            else (0, False)
        ),
    )


def _run_until_hi_loads(
    monkeypatch, priority_prefill_budget_cap: bool = True, **kwargs
) -> tuple[Scheduler, Request]:
    """A priority 2 decode and prefill run when a priority 0 request arrives
    whose KV connector loads its first 256 tokens asynchronously: the waiting
    loop admits it to load them."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=priority_prefill_budget_cap,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=128,
        max_model_len=2048,
        num_blocks=400,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        **kwargs,
    )
    _match_tokens(scheduler, monkeypatch, {"hi": 256})
    _run_until_decoding(
        scheduler, _request("decode", 40, 2, arrival_time=0, max_tokens=400)
    )
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    _step(scheduler)
    hi = _request("hi", 300, priority=0, arrival_time=2)
    scheduler.add_request(hi)
    _step(scheduler)
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return scheduler, hi


@pytest.mark.parametrize("priority_prefill_budget_cap", [True, False])
def test_a_load_holds_lower_priority_prefill_back_for_its_first_steps(
    monkeypatch, priority_prefill_budget_cap
):
    """While its KV connector loads a request's KV, lower-priority prefill sits
    out the load's first `priority_prefill_budget_cap_tokens` steps, here 4: the
    priority 2 prefill gets no chunk, and a new priority 2 request is not
    admitted. With the cap off, the prefill runs throughout."""
    scheduler, hi = _run_until_hi_loads(monkeypatch, priority_prefill_budget_cap)
    scheduler.add_request(_request("late", 300, priority=2, arrival_time=3))
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(4)]
    if priority_prefill_budget_cap:
        assert steps == [{"decode": 1}] * 4
    else:
        assert all(tokens["prefill"] == 128 for tokens in steps)
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_stuck_load_holds_nothing_after_its_first_steps(monkeypatch):
    """A load that never completes holds lower-priority prefill back for its
    first 4 steps only: then the priority 2 prefill gets its chunk and a new
    priority 2 request is admitted, as with the cap off."""
    scheduler, hi = _run_until_hi_loads(monkeypatch)
    scheduler.add_request(_request("late", 300, priority=2, arrival_time=3))
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(12)]
    assert steps[:4] == [{"decode": 1}] * 4
    assert all(tokens["prefill"] == 128 for tokens in steps[4:])
    assert any("late" in tokens for tokens in steps[4:])
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize("priority_prefill_budget_cap", [True, False])
def test_a_load_holds_again_from_the_step_it_completes(
    monkeypatch, priority_prefill_budget_cap
):
    """A load past its first 4 steps holds nothing, but once the connector
    reports it done, the waiting loop promotes the request in the next step, on
    budget the running loop leaves: from that step the priority 2 prefill sits
    out again, until the request's first 4 tokens. With the cap off, the prefill
    gets its chunk beside the request's."""
    scheduler, _ = _run_until_hi_loads(monkeypatch, priority_prefill_budget_cap)
    for _ in range(4):
        _step(scheduler)  # The load's first steps.
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "prefill": 128}
    _step(scheduler, finished_recving={"hi"})  # The load is reported done.
    num_scheduled_tokens = _step(scheduler).num_scheduled_tokens

    if priority_prefill_budget_cap:
        # The 44 tokens it did not load.
        assert num_scheduled_tokens == {"decode": 1, "hi": 44}
        steps = [_step(scheduler).num_scheduled_tokens for _ in range(3)]
        assert steps == [{"decode": 1, "hi": 1}] * 3
        assert _step(scheduler).num_scheduled_tokens["prefill"] == 128
    else:
        assert num_scheduled_tokens == {"decode": 1, "prefill": 128, "hi": 44}


def test_a_completed_load_holds_prefill_back_while_a_request_that_does_not_fit_waits(
    monkeypatch,
):
    """A request whose load is done holds lower-priority prefill back from the
    step the waiting loop promotes it, whichever waiting request is served
    first: here a priority 0 request of 38 blocks, with fewer free, waits ahead
    of the priority 1 one whose load is done, past its first 4 steps, and the
    priority 2 prefill sits out the promotion's step again."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=4 * BLOCK_SIZE,
        max_model_len=40 * BLOCK_SIZE,
        num_blocks=60,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(scheduler, monkeypatch, {"load": 256})
    _run_until_decoding(
        scheduler, _request("decode", 40, 2, arrival_time=0, max_tokens=400)
    )
    scheduler.add_request(_request("prefill", 30 * BLOCK_SIZE, 2, arrival_time=1))
    _step(scheduler)
    load = _request("load", 300, priority=1, arrival_time=2)
    scheduler.add_request(load)
    _step(scheduler)
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    big = _request("big", 38 * BLOCK_SIZE, priority=0, arrival_time=3)
    scheduler.add_request(big)
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(5)]
    assert steps == [{"decode": 1}] * 4 + [{"decode": 1, "prefill": 4 * BLOCK_SIZE}]
    _step(scheduler, finished_recving={"load"})  # The load is reported done.

    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "load": 44}
    assert big.status == RequestStatus.WAITING and _num_free_blocks(scheduler) < 38


def test_with_every_running_slot_taken_a_completed_load_holds_nothing_back(
    monkeypatch,
):
    """A request whose load is done needs a running slot to be promoted. With
    both slots taken by priority 2 requests, one admitted past the load's first
    4 steps, it holds nothing back: otherwise the running prefill would sit out
    every step until the decode finished, and with no decode running, forever."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
        max_num_seqs=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(scheduler, monkeypatch, {"hi": 256})
    _run_until_decoding(
        scheduler, _request("decode", 40, 2, arrival_time=0, max_tokens=400)
    )
    hi = _request("hi", 300, priority=0, arrival_time=1)
    scheduler.add_request(hi)
    _step(scheduler)
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=2))
    for _ in range(4):
        _step(scheduler)  # The load's first steps: the prefill waits.
    _step(scheduler, finished_recving={"hi"})
    assert len(scheduler.running) == 2
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS


# Metrics


def test_the_scheduler_stats_count_the_requests_preempted_to_make_room():
    """Per step, apart from the running loop's preemptions."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=4)
    _run_three_low_priority_requests(scheduler)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=4))

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler, scheduler_output)
    )
    assert scheduler_output.preempted_req_ids == {"other"}
    assert outputs[0].scheduler_stats.num_priority_preemptions == 1

    scheduler_output = scheduler.schedule()
    outputs = scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler, scheduler_output)
    )
    assert outputs[0].scheduler_stats.num_priority_preemptions == 0


@pytest.mark.parametrize("max_preemptions", [0, 2])
def test_prometheus_counts_the_requests_preempted_to_make_room(max_preemptions):
    """As vllm:num_priority_preemptions, only with the option on."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m"),
        scheduler_config=SchedulerConfig.default_factory(
            policy="priority", max_priority_preemptions_per_step=max_preemptions
        ),
    )
    try:
        logger = PrometheusStatLogger(vllm_config)
        logger.record(SchedulerStats(num_priority_preemptions=3), None)
        logger.record(SchedulerStats(num_priority_preemptions=2), None)
        value = prometheus_client.REGISTRY.get_sample_value(
            "vllm:num_priority_preemptions_total",
            {"model_name": "facebook/opt-125m", "engine": "0"},
        )
        assert value == (5 if max_preemptions else None)
    finally:
        unregister_vllm_metrics()


# Off is stock


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
