# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `max_priority_preemptions_per_step`: under the priority policy, the
waiting request of the highest priority preempts running requests of a lower
priority for KV cache blocks.

Unless a test says otherwise, the KV cache holds 32 blocks of 16 tokens (33
with the null block) and prefix caching is off, so requests share no block.
"""

from collections import deque

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
)
from vllm.v1.outputs import ModelRunnerOutput
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
    scheduler: Scheduler, scheduler_output: SchedulerOutput
) -> ModelRunnerOutput:
    """A sampled token for each scheduled request but partial prefills."""
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
    )


def _step(scheduler: Scheduler) -> SchedulerOutput:
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, _model_runner_output(scheduler, scheduler_output)
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


def test_make_room_requires_the_priority_policy():
    with pytest.raises(ValueError, match="the priority scheduling policy"):
        SchedulerConfig.default_factory(
            policy="fcfs", max_priority_preemptions_per_step=1
        )


def test_max_priority_preemptions_per_step_is_not_negative():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        SchedulerConfig.default_factory(
            policy="priority", max_priority_preemptions_per_step=-1
        )


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
    """At one priority (the default 0) make-room never acts, and the schedules
    match those with it off, through preemptions in the running loop and chunked
    prefills. Across priorities, the same workload is scheduled differently."""
    options = dict(
        num_blocks=48, max_num_batched_tokens=128, long_prefill_token_threshold=64
    )
    priorities = [i % 3 if mixed_priorities else 0 for i in range(24)]
    trace = _trace(_create_scheduler(**options), priorities)
    assert any(preempted for _, preempted, _ in trace)

    trace_with_make_room = _trace(
        _create_scheduler(**options, max_priority_preemptions_per_step=4),
        priorities,
    )
    assert (trace_with_make_room != trace) == mixed_priorities
