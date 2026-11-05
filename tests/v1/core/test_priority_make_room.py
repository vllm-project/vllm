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
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
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

from .test_scheduler import EOS_TOKEN_ID, create_requests_with_priority
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
    encoder_offset: int = 0,
) -> Request:
    """A request whose prompt is `num_tokens` tokens of `prompt_token`, with an
    encoder input of `num_encoder_tokens` at `encoder_offset` if any."""
    encoder_inputs = {}
    if num_encoder_tokens:
        encoder_inputs = dict(
            mm_hashes_list=[[f"encoder-{request_id}"]],
            mm_positions=[
                [PlaceholderRange(offset=encoder_offset, length=num_encoder_tokens)]
            ],
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
    single block and preempting both frees 9, so with 1 free, the waiting
    request of 6 blocks takes both."""
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
        _request("hi", _num_tokens(6), priority=0, arrival_time=3, prompt_token=2)
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


def test_room_counts_only_the_prefix_cache_hit_the_waiting_loop_takes():
    """As in test_room_is_net_of_the_waiting_requests_prefix_cache_hit, but the
    waiting request scores its prompt from token 16, so it may read only its
    first block from the cache and needs 22 of its own: both priority 2 requests
    make room for it."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, enable_prefix_caching=True
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(5), priority=0, arrival_time=0, prompt_token=1),
        _request("holder", 18 * BLOCK_SIZE + 8, 2, arrival_time=1, prompt_token=5),
        _request("bg", _num_tokens(6), priority=2, arrival_time=2, prompt_token=2),
    )
    assert _num_free_blocks(scheduler) == 2

    sampling_params = SamplingParams(
        max_tokens=64,
        prompt_logprobs=1,
        prompt_logprob_start=16,
        skip_reading_prefix_cache=False,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    hi = Request(
        request_id="hi",
        prompt_token_ids=[5] * (18 * BLOCK_SIZE + 72),
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=3,
        priority=0,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(hi)
    output = _step(scheduler)

    assert output.preempted_req_ids == {"bg", "holder"}
    assert "hi" in output.num_scheduled_tokens


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


def _start_loading(scheduler: Scheduler, monkeypatch, priority: int = 0) -> Request:
    """A request of 19 blocks (priority 0 by default) whose KV connector loads its
    first 16 asynchronously, after "rt" (4 blocks) and "bg" (priority 2, 8
    blocks): the waiting loop admits it to load them, so 4 blocks stay free and
    it still needs 3."""
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
    load = _request("load", 300, priority=priority, arrival_time=2)
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


def test_a_kv_load_in_flight_keeps_no_lower_priority_request_out(monkeypatch):
    """A priority 0 request of 6 blocks loads its first, and reserves the other 5
    with 8 free. The priority 2 request of 4 blocks fits in those 8, though not
    beside the reservation: it is admitted as without the option, and finishes,
    however long the load takes."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        num_blocks=10,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(scheduler, monkeypatch, {"load": BLOCK_SIZE})
    load = _request("load", 6 * BLOCK_SIZE, 0, arrival_time=0, prompt_token=1)
    low = _request(
        "low", 4 * BLOCK_SIZE, 2, arrival_time=1, max_tokens=2, prompt_token=2
    )
    scheduler.add_request(load)
    scheduler.add_request(low)
    assert _step(scheduler).num_scheduled_tokens == {"low": 4 * BLOCK_SIZE}
    assert _num_free_blocks(scheduler) == 4
    for _ in range(3):
        _step(scheduler)
    assert low.is_finished()
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_victim_may_return_during_the_load_and_room_is_made_again(monkeypatch):
    """Room is made for a priority 0 request that then loads its first block: the
    victim returns meanwhile, as it fits in the pool, and when the load is done it
    is preempted again for the blocks the request still needs. Both finish."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        num_blocks=10,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(scheduler, monkeypatch, {"hi": BLOCK_SIZE})
    victim = _request("victim", 88, 2, arrival_time=0, max_tokens=8, prompt_token=2)
    scheduler.add_request(victim)
    _step(scheduler)
    _step(scheduler)
    hi = _request("hi", 4 * BLOCK_SIZE, 0, arrival_time=1, max_tokens=2, prompt_token=1)
    scheduler.add_request(hi)
    assert _step(scheduler).preempted_req_ids == {"victim"}
    assert hi.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    _step(scheduler)
    assert victim.status == RequestStatus.RUNNING

    _step(scheduler, finished_recving={"hi"})
    output = _step(scheduler)
    assert output.preempted_req_ids == {"victim"}
    assert "hi" in output.num_scheduled_tokens
    for _ in range(12):
        _step(scheduler)
    assert hi.is_finished()
    assert victim.is_finished()


def test_a_prefill_scheduled_nothing_keeps_no_lower_priority_request_out():
    """With --disable-chunked-mm-input, a priority 0 prefill that reaches its
    image, longer than the long prefill threshold, is scheduled nothing while it is
    not the only request. The priority 2 request, which fits in the pool but not
    beside the prefill's reservation, is admitted as without the option, and its end
    lets the prefill go on."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        num_blocks=10,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=BLOCK_SIZE,
        disable_chunked_mm_input=True,
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    hi = _request(
        "hi",
        4 * BLOCK_SIZE,
        0,
        arrival_time=0,
        max_tokens=2,
        prompt_token=1,
        num_encoder_tokens=2 * BLOCK_SIZE,
        encoder_offset=BLOCK_SIZE,
    )
    low = _request(
        "low", 6 * BLOCK_SIZE, 2, arrival_time=1, max_tokens=2, prompt_token=2
    )
    scheduler.add_request(hi)
    scheduler.add_request(low)
    for _ in range(12):
        _step(scheduler)
    assert hi.is_finished()
    assert low.is_finished()


def test_a_request_whose_load_is_done_fits_beside_its_own_reservation(monkeypatch):
    """Once its KV load is done, the loading request is the one room would be
    made for. The 3 blocks it still needs are among the in-flight prefills'
    reservations already, so it fits in the 4 free blocks and preempts nothing."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _start_loading(scheduler, monkeypatch)
    _step(scheduler, finished_recving={"load"})
    output = _step(scheduler)
    assert not output.preempted_req_ids
    assert "load" in output.num_scheduled_tokens


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


@pytest.mark.parametrize(
    "rt_blocks, victim_blocks, chunk_tokens", [(14, 10, 100), (18, 6, 200)]
)
def test_a_lower_priority_session_does_not_take_the_room(
    rt_blocks, victim_blocks, chunk_tokens
):
    """Room is made for a priority 0 request while a priority 2 session's next
    chunk, which holds 4 blocks, waits with the requests that hold KV blocks,
    which the waiting loop drains first. The chunk waits a step, rather than
    take the room (6 more blocks) or, as it does not fit (12), stop the loop
    before the priority 0 request."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2, max_model_len=32 * BLOCK_SIZE
    )
    _run_session_first_chunk(scheduler, priority=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(rt_blocks), 0, arrival_time=1, max_tokens=200),
        _request("v", _num_tokens(victim_blocks), 1, arrival_time=2, max_tokens=200),
    )
    assert _num_free_blocks(scheduler) == 4

    scheduler.add_request(_session_request(chunk_tokens, max_tokens=16, priority=2))
    scheduler.add_request(
        _request("hi", _num_tokens(9), priority=0, arrival_time=3, prompt_token=7)
    )
    output = _step(scheduler)
    assert output.preempted_req_ids == {"v"}
    assert "hi" in output.num_scheduled_tokens
    assert "session" not in output.num_scheduled_tokens


def test_a_lower_priority_request_with_reserved_blocks_is_admitted(monkeypatch):
    """Room is made for a priority 0 request in the step after a priority 1
    request's KV load is done. The 3 blocks the loaded request still needs are
    reserved already, so it does not take the room: it is admitted beside the
    priority 0 request."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _start_loading(scheduler, monkeypatch, priority=1)
    _step(scheduler, finished_recving={"load"})
    scheduler.add_request(_request("hi", _num_tokens(5), priority=0, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert {"hi", "load"} <= output.num_scheduled_tokens.keys()


def test_a_lower_priority_request_behind_the_room_is_admitted():
    """In the step room is made for a priority 0 request, a priority 1 request
    behind it in the waiting queue still takes what is left."""
    scheduler = _create_scheduler(max_priority_preemptions_per_step=2)
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(22), priority=0, arrival_time=0),
        _request("bg", _num_tokens(8), priority=2, arrival_time=1),
    )
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    scheduler.add_request(_request("small", BLOCK_SIZE, priority=1, arrival_time=3))
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert {"hi", "small"} <= output.num_scheduled_tokens.keys()


# Requests that wait to start an async KV load (--max-num-inflight-kv-loads)


def _waits_to_load(scheduler: Scheduler, monkeypatch, matched: dict[str, int]):
    """With --max-num-inflight-kv-loads, the requests in `matched` wait in the
    queue of requests the KV connector may load asynchronously, which loads
    `matched[request_id]` of their tokens (none: they prefill as others)."""
    monkeypatch.setattr(
        scheduler.connector,
        "may_load_async",
        lambda request: request.request_id in matched,
    )
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda request, num_computed_tokens: (
            (matched[request.request_id], True)
            if matched.get(request.request_id)
            else (0, False)
        ),
    )


def test_makes_room_for_a_request_waiting_to_start_its_kv_load(monkeypatch):
    """A priority 0 request of 19 blocks waits to start loading 16 of them, with
    12 free: room is made for it as for any waiting request, and its load
    starts."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        max_num_inflight_kv_loads=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _run_until_decoding(
        scheduler,
        _request("rt", _num_tokens(4), priority=0, arrival_time=0),
        _request("bg", _num_tokens(16), priority=2, arrival_time=1),
    )
    _waits_to_load(scheduler, monkeypatch, {"remote": 16 * BLOCK_SIZE})
    remote = _request("remote", 300, priority=0, arrival_time=2)
    scheduler.add_request(remote)
    assert list(scheduler.async_kv_waiting) == [remote]
    output = _step(scheduler)
    assert output.preempted_req_ids == {"bg"}
    assert remote.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_request_waiting_with_the_kv_loads_is_admitted_beside_inflight_prefills(
    monkeypatch,
):
    """A priority 2 request waits with the requests the KV connector may load
    asynchronously, but the connector finds none of its tokens: while a
    priority 0 prefill is in flight, it is admitted only if its 19 blocks fit
    beside the 12 that prefill still needs, as any lower-priority admission."""
    scheduler = _create_scheduler(
        max_priority_preemptions_per_step=2,
        max_num_inflight_kv_loads=2,
        long_prefill_token_threshold=64,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _run_until_decoding(
        scheduler, _request("bg", _num_tokens(4), priority=2, arrival_time=0)
    )
    prefill = _request("prefill", 16 * BLOCK_SIZE, 0, arrival_time=1)
    scheduler.add_request(prefill)
    _step(scheduler)
    assert prefill.is_prefill_chunk
    _waits_to_load(scheduler, monkeypatch, {"late": 0})
    late = _request("late", 300, priority=2, arrival_time=2)
    scheduler.add_request(late)
    assert list(scheduler.async_kv_waiting) == [late]
    output = _step(scheduler)
    assert output.num_scheduled_tokens == {"bg": 1, "prefill": 64}
    assert late.status == RequestStatus.WAITING


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


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_the_first_tokens_hold_as_many_steps_under_async_scheduling(async_scheduling):
    """Under async scheduling a step is scheduled before the output of the step
    before it is processed. The priority 0 request's first tokens count those in
    flight, so the running priority 2 prefill sits out the steps of its first 4
    tokens and goes on in the step of its fifth, as without async scheduling."""
    scheduler = _create_scheduler(
        async_scheduling=async_scheduling,
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "prefill": 256}
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    schedule, process = _in_flight_steps(scheduler)
    steps = [schedule().num_scheduled_tokens]
    for _ in range(4):
        # As in the engine: with async scheduling, one step stays in flight.
        if not async_scheduling:
            process()
        steps.append(schedule().num_scheduled_tokens)
        if async_scheduling:
            process()
    assert steps[0] == {"decode": 1, "hi": 100}
    assert steps[1:4] == [{"decode": 1, "hi": 1}] * 3
    assert steps[4] == {"decode": 1, "hi": 1, "prefill": 256}


def test_the_first_tokens_count_no_draft_tokens_under_async_scheduling():
    """With 3 draft tokens per step, a step in flight is sure to add only one
    token to the priority 0 request's output, as its drafts may be rejected:
    here all are, and the priority 2 prefill sits out the steps of its first 8
    tokens, as without drafts."""
    scheduler = _create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=8,
        long_prefill_token_threshold=128,
        max_model_len=2048,
        num_blocks=400,
    )
    schedule, process = _in_flight_steps(scheduler)
    scheduler.add_request(_request("prefill", 2000, 2, arrival_time=0, max_tokens=4))
    schedule()
    scheduler.add_request(
        _request("hi", 64, priority=0, arrival_time=1, max_tokens=200, prompt_token=9)
    )
    steps = []
    for _ in range(9):
        # As in the engine: with async scheduling, one step stays in flight.
        steps.append(schedule().num_scheduled_tokens)
        process()
    assert steps == [{"hi": 64}] + [{"hi": 4}] * 7 + [{"prefill": 128, "hi": 4}]


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


def test_budget_cap_never_holds_an_admission_of_one_token():
    """A priority 2 request whose prompt is cached but for its last token
    samples its first token like a decode: it is admitted in the step of the
    priority 0 request's prefill."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        enable_prefix_caching=True,
        max_model_len=2048,
        num_blocks=400,
    )
    warm = _request("warm", 257, priority=2, arrival_time=0, max_tokens=1)
    scheduler.add_request(warm)
    while not warm.is_finished():
        _step(scheduler)

    scheduler.add_request(
        _request("hi", 100, priority=0, arrival_time=1, prompt_token=5)
    )
    scheduler.add_request(_request("again", 257, priority=2, arrival_time=2))
    assert _step(scheduler).num_scheduled_tokens == {"hi": 100, "again": 1}


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


@pytest.mark.parametrize("hi_offset", [0, 8])
def test_budget_cap_never_holds_a_prefill_whose_encoder_output_is_cached(hi_offset):
    """The encoder cache has room for one image, held by the priority 2 prefill,
    which has computed 8 of its 16 tokens. The priority 0 request needs the
    space for its own image, at once or after 8 tokens of text. Only the
    prefill's progress frees the space, so it goes on, and then sits out the
    priority 0 request's prompt."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True, max_num_batched_tokens=8, max_num_seqs=2
    )
    scheduler.max_num_encoder_input_tokens = 16
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    scheduler.add_request(
        _request("prefill", 24, priority=2, arrival_time=0, num_encoder_tokens=16)
    )
    assert _step(scheduler).num_scheduled_tokens == {"prefill": 8}

    hi = _request(
        "hi",
        hi_offset + 24,
        priority=0,
        arrival_time=1,
        num_encoder_tokens=16,
        encoder_offset=hi_offset,
    )
    scheduler.add_request(hi)
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(hi_offset // 8 + 4)]
    assert steps == [{"prefill": 8}] + [{"hi": 8}] * (hi_offset // 8 + 3)
    assert hi.num_output_tokens == 1


def _create_encoder_cache_scheduler(**kwargs) -> Scheduler:
    """A scheduler whose encoder cache and budget hold 2048 embeddings, with a
    priority 2 decode running."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=128,
        max_model_len=2048,
        num_blocks=600,
        **kwargs,
    )
    scheduler.max_num_encoder_input_tokens = 2048
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=2048)
    _run_until_decoding(scheduler, _request("decode", 40, 2, 0, max_tokens=200))
    return scheduler


def test_budget_cap_holds_an_encoder_prefill_for_first_tokens_of_text():
    """A priority 2 prefill midway through a video keeps its encoder output
    cached, but the priority 0 request needs no encoder cache: the prefill sits
    out its first tokens like any other."""
    scheduler = _create_encoder_cache_scheduler()
    scheduler.add_request(
        _request("doc", 1600, 2, 1, max_tokens=4, num_encoder_tokens=1536)
    )
    assert _step(scheduler).num_scheduled_tokens == {"decode": 1, "doc": 128}

    scheduler.add_request(
        _request("hi", 100, priority=0, arrival_time=2, prompt_token=3)
    )
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(4)]
    assert steps == [{"decode": 1, "hi": 100}] + [{"decode": 1, "hi": 1}] * 3


def test_budget_cap_holds_an_encoder_prefill_whose_input_is_consumed_in_flight():
    """Under async scheduling, the image that a priority 2 prefill's step in
    flight consumes stays cached until that step's output is processed, which
    frees it whatever the prefill does next: the prefill sits out the prefill
    step of a priority 0 request that brings an image of its own."""
    scheduler = _create_encoder_cache_scheduler(async_scheduling=True)
    doc = create_requests_with_priority(
        num_requests=1,
        priorities=[2],
        arrival_times=[1],
        num_tokens=1024,
        max_tokens=4,
        req_ids=["doc"],
        mm_hashes_list=[[f"image-{k}" for k in range(8)]],
        mm_positions=[
            [PlaceholderRange(offset=128 * k + 32, length=64) for k in range(8)]
        ],
    )[0]
    scheduler.add_request(doc)
    schedule, process = _in_flight_steps(scheduler)
    assert schedule().num_scheduled_tokens == {"decode": 1, "doc": 128}

    scheduler.add_request(
        _request("hi", 100, 0, arrival_time=2, prompt_token=3, num_encoder_tokens=32)
    )
    assert schedule().num_scheduled_tokens == {"decode": 1, "hi": 100}


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


@pytest.mark.parametrize(
    ("num_loaded", "steps"),
    [
        (
            256,
            [{"decode": 1, "hi": 100}]
            + [{"decode": 1, "hi": 1}] * 3
            + [{"decode": 1, "hi": 1, "late": 256}],
        ),
        (
            1999,
            [{"decode": 1, "late": 1, "hi": 100}]
            + [{"decode": 1, "hi": 1, "late": 1}] * 4,
        ),
    ],
    ids=["partly_loaded", "one_token_left"],
)
def test_a_lower_priority_request_holding_kv_sits_out_a_requests_first_tokens(
    num_loaded: int, steps: list[dict[str, int]]
):
    """The waiting loop drains first the requests that hold KV blocks, as one
    does whose KV load is done. A priority 2 one sits out the first tokens of a
    priority 0 request like any lower-priority prefill, but it is skipped rather
    than waited on, so the loop goes on to the priority 0 request behind it,
    which runs in the same step; after its first 4 tokens, the priority 2 one
    prefills. With one token left to compute, it is not held."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    late = _request("late", 2000, priority=2, arrival_time=1)
    # Its KV loaded.
    assert scheduler.kv_cache_manager.allocate_slots(late, num_loaded) is not None
    late.num_computed_tokens = num_loaded
    scheduler.add_request(late)
    assert late in scheduler.kv_holding_waiting
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))

    assert [_step(scheduler).num_scheduled_tokens for _ in range(5)] == steps


def test_a_lower_priority_request_holding_kv_is_admitted_while_none_of_its_runs():
    """As above, but no priority 2 request runs. While it waits, preemption
    cannot reach the KV blocks of the request whose load is done, so the
    priority 0 request could preempt only itself for blocks: it is admitted,
    and the running loop holds its next chunks."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=256,
        max_model_len=2048,
        num_blocks=400,
    )
    late = _request("late", 2000, priority=2, arrival_time=1)
    # 256 tokens of its KV loaded.
    assert scheduler.kv_cache_manager.allocate_slots(late, 256) is not None
    late.num_computed_tokens = 256
    scheduler.add_request(late)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))

    steps = [_step(scheduler).num_scheduled_tokens for _ in range(5)]
    assert steps[:4] == [{"late": 256, "hi": 100}] + [{"hi": 1}] * 3
    assert steps[4] == {"hi": 1, "late": 256}


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


def test_a_held_request_starts_its_kv_load(monkeypatch):
    """A lower-priority request sits out a request's first tokens after its
    prefix cache and KV connector lookups, so a load of its KV, which takes no
    compute, starts meanwhile."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(scheduler, monkeypatch, {"late": 64})
    _run_until_decoding(scheduler, _request("hi", 32, priority=0, arrival_time=0))
    late = _request("late", 128, priority=2, arrival_time=1)
    scheduler.add_request(late)
    assert _step(scheduler).num_scheduled_tokens == {"hi": 1}
    assert late.status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_a_request_waiting_with_the_kv_loads_sits_out_a_requests_first_tokens(
    monkeypatch,
):
    """A priority 2 request waits with the requests the KV connector may load
    asynchronously, but the connector finds none of its tokens: it sits out the
    first tokens of the running priority 0 request, as any lower-priority
    prefill."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        max_num_inflight_kv_loads=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _waits_to_load(scheduler, monkeypatch, {"late": 0})
    _run_until_decoding(scheduler, _request("hi", 32, priority=0, arrival_time=0))
    late = _request("late", 128, priority=2, arrival_time=1)
    scheduler.add_request(late)
    assert list(scheduler.async_kv_waiting) == [late]
    assert _step(scheduler).num_scheduled_tokens == {"hi": 1}
    assert late.status == RequestStatus.WAITING


def test_a_waiting_request_holds_back_a_request_waiting_with_the_kv_loads(
    monkeypatch,
):
    """With no request running, the priority 0 request that fits holds back the
    prefill of a priority 2 request that waits with the requests the KV
    connector may load asynchronously, the connector finding none of its
    tokens, as it holds back any lower-priority prefill in its step."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        max_num_inflight_kv_loads=2,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _waits_to_load(scheduler, monkeypatch, {"late": 0})
    late = _request("late", 128, priority=2, arrival_time=0)
    scheduler.add_request(late)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=1))
    assert _step(scheduler).num_scheduled_tokens == {"hi": 100}
    assert late.status == RequestStatus.WAITING


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


def test_a_waiting_request_never_admitted_holds_for_its_first_steps_only(
    monkeypatch,
):
    """A priority 0 request fits, but the waiting loop does not admit it, here
    as its KV connector lookup stays pending: it holds lower-priority prefill
    back for its first 4 steps only, like a stuck load."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=128,
        max_model_len=2048,
        num_blocks=400,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    monkeypatch.setattr(
        scheduler.connector,
        "get_num_new_matched_tokens",
        lambda request, num_computed_tokens: (
            (None, False) if request.request_id == "hi" else (0, False)
        ),
    )
    _run_until_decoding(
        scheduler, _request("decode", 40, 2, arrival_time=0, max_tokens=400)
    )
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    _step(scheduler)
    scheduler.add_request(_request("hi", 300, priority=0, arrival_time=2))

    steps = [_step(scheduler).num_scheduled_tokens for _ in range(8)]
    assert steps[:4] == [{"decode": 1}] * 4
    assert all(tokens["prefill"] == 128 for tokens in steps[4:])


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


@pytest.mark.parametrize("low_runs", [False, True])
@pytest.mark.parametrize("window", [1, 16])
def test_a_running_request_scheduled_nothing_stops_holding(window, low_runs):
    """With --disable-chunked-mm-input, a priority 0 prefill that reaches its
    image, longer than the long prefill threshold, is scheduled nothing while it is
    not the only request. So it waits for the priority 2 request, which it holds
    back, whether that one runs or waits: it holds for `window` such steps, and then
    both finish."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=window,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=BLOCK_SIZE,
        disable_chunked_mm_input=True,
    )
    scheduler.max_num_encoder_input_tokens = 64
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=64)
    hi = _request(
        "hi",
        4 * BLOCK_SIZE,
        0,
        arrival_time=0,
        max_tokens=2,
        prompt_token=1,
        num_encoder_tokens=2 * BLOCK_SIZE,
        encoder_offset=BLOCK_SIZE,
    )
    low = _request(
        "low", 6 * BLOCK_SIZE, 2, arrival_time=1, max_tokens=2, prompt_token=2
    )
    scheduler.add_request(low if low_runs else hi)
    if low_runs:
        _step(scheduler)
    scheduler.add_request(hi if low_runs else low)
    assert "hi" in _step(scheduler).num_scheduled_tokens
    assert hi.num_computed_tokens == BLOCK_SIZE
    assert [_step(scheduler).num_scheduled_tokens for _ in range(window)] == [
        {}
    ] * window
    assert "low" in _step(scheduler).num_scheduled_tokens
    for _ in range(12):
        _step(scheduler)
    assert hi.is_finished()
    assert low.is_finished()


def test_a_waiting_request_held_back_keeps_its_window():
    """A priority 1 request that fits waits behind a priority 0 request's first
    tokens. Those steps do not count against its own window, though they outlast it:
    it then holds the priority 2 prefill back in the step it is admitted, and that
    one has had no chunk since the priority 0 request arrived."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        max_num_batched_tokens=BLOCK_SIZE,
        num_blocks=64,
    )
    low = _request(
        "low", 10 * BLOCK_SIZE, 2, arrival_time=0, max_tokens=2, prompt_token=3
    )
    scheduler.add_request(low)
    _step(scheduler)
    scheduler.add_request(
        _request("hi", 2 * BLOCK_SIZE, 0, arrival_time=1, max_tokens=4, prompt_token=1)
    )
    scheduler.add_request(
        _request("mid", 8, 1, arrival_time=2, max_tokens=4, prompt_token=2)
    )
    steps = [_step(scheduler).num_scheduled_tokens for _ in range(6)]
    assert steps == [
        {"hi": BLOCK_SIZE},
        {"hi": BLOCK_SIZE},
        {"hi": 1},
        {"hi": 1},
        {"hi": 1},
        {"mid": 8},
    ]
    assert low.num_computed_tokens == BLOCK_SIZE


def test_budget_cap_never_holds_a_running_request_with_one_token_left():
    """A priority 2 request has one token left to compute when a priority 0
    request arrives: it samples like a decode and is not held."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        max_num_batched_tokens=BLOCK_SIZE,
    )
    scheduler.add_request(
        _request("low", BLOCK_SIZE + 1, 2, arrival_time=0, prompt_token=2)
    )
    _step(scheduler)
    scheduler.add_request(_request("hi", 8, 0, arrival_time=1, prompt_token=1))
    assert _step(scheduler).num_scheduled_tokens == {"low": 1, "hi": 8}


def test_each_chunk_of_a_session_holds_as_its_first_tokens_are_due():
    """A session's chunks of one output token are admitted and finished within a
    step each. The steps of one chunk do not count against the next: each holds the
    priority 2 prefill back in the step it is admitted, as many chunks as there are."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=2,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=BLOCK_SIZE,
        max_model_len=64 * BLOCK_SIZE,
        num_blocks=80,
    )
    scheduler.add_request(
        _request(
            "low", 40 * BLOCK_SIZE, 2, arrival_time=0, max_tokens=1, prompt_token=2
        )
    )
    _step(scheduler)
    for _ in range(4):
        scheduler.add_request(_session_request(8, max_tokens=1, priority=0))
        assert _step(scheduler).num_scheduled_tokens == {"session": 8}
        assert "low" in _step(scheduler).num_scheduled_tokens


@pytest.mark.parametrize("window", [1, 4, 16])
def test_a_completed_load_the_waiting_loop_cannot_reach_stops_holding(
    monkeypatch, window
):
    """A priority 0 and a priority 1 request each load 4 blocks, and a priority 2
    prefill has computed 5 of its 6 when both loads are done. With one block free,
    the priority 0 request, which needs two, fails its allocation and ends the
    waiting loop, so the priority 1 request is not promoted though its load is done.
    The prefill it holds back is what would free the blocks: it sits out the first
    `window` steps only, and then all three finish."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=window,
        max_num_batched_tokens=BLOCK_SIZE,
        num_blocks=15,
        max_num_seqs=3,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    _match_tokens(
        scheduler, monkeypatch, {"hi": 4 * BLOCK_SIZE, "load": 4 * BLOCK_SIZE}
    )
    requests = (
        _request("hi", 6 * BLOCK_SIZE, 0, arrival_time=0, max_tokens=2, prompt_token=1),
        _request("load", 65, 1, arrival_time=1, max_tokens=2, prompt_token=2),
        _request(
            "prefill", 6 * BLOCK_SIZE, 2, arrival_time=2, max_tokens=1, prompt_token=3
        ),
    )
    hi, load, prefill = requests
    for request in requests:
        scheduler.add_request(request)
    for _ in range(window + 5):
        _step(scheduler)
    assert prefill.num_computed_tokens == 4 * BLOCK_SIZE
    _step(scheduler, finished_recving={"hi", "load"})
    assert _num_free_blocks(scheduler) == 1

    assert [_step(scheduler).num_scheduled_tokens for _ in range(window)] == (
        [{}] * window
    )
    assert hi.status == RequestStatus.WAITING
    assert load.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert _step(scheduler).num_scheduled_tokens == {"prefill": BLOCK_SIZE}
    for _ in range(10):
        _step(scheduler)
    assert all(request.is_finished() for request in requests)
    assert not scheduler._hold_progress


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


@pytest.mark.parametrize("policy", [SchedulingPolicy.PRIORITY, SchedulingPolicy.FCFS])
def test_a_queue_tells_whether_it_holds_a_lower_priority(policy):
    queue = create_request_queue(policy)
    assert not queue.has_lower_priority_than(0)
    for i, priority in enumerate([1, 0, 2, 1, 0, 1, 2, 0]):
        queue.add_request(_request(f"r{i}", BLOCK_SIZE, priority, arrival_time=i))
    assert queue.has_lower_priority_than(1)
    assert not queue.has_lower_priority_than(2)


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


@pytest.mark.parametrize(
    "long_prefill_token_threshold, held_tokens", [(256, 256), (0, 511)]
)
def test_the_scheduler_stats_count_the_budget_caps_holds(
    long_prefill_token_threshold, held_tokens
):
    """The steps in which the cap holds lower-priority prefill back, once each,
    and the prefill tokens it holds back in them: a running chunk's, as the
    running loop would have scheduled it, up to the long prefill threshold, or
    else the 511 tokens of the step's budget the decode leaves. A held admission
    counts its step."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    scheduler.add_request(_request("prefill", 2000, priority=2, arrival_time=1))
    _step(scheduler)
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=2))
    held = []
    for step in range(5):
        if step == 1:
            scheduler.add_request(_request("late", 300, priority=2, arrival_time=3))
        scheduler_output = scheduler.schedule()
        stats = scheduler.update_from_output(
            scheduler_output, _model_runner_output(scheduler, scheduler_output)
        )[0].scheduler_stats
        held.append((stats.num_prefill_held_steps, stats.num_prefill_held_tokens))
    assert held == [(1, held_tokens)] * 4 + [(0, 0)]


@pytest.mark.parametrize("holds_kv", [False, True])
def test_the_scheduler_stats_count_a_step_that_holds_only_an_admission(holds_kv):
    """With no lower-priority chunk to hold, a held admission counts its step,
    with no tokens, whether the held request holds KV blocks or not."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=1))
    late = _request("late", 300, priority=2, arrival_time=2)
    if holds_kv:
        # 256 tokens of its KV loaded.
        assert scheduler.kv_cache_manager.allocate_slots(late, 256) is not None
        late.num_computed_tokens = 256
    scheduler.add_request(late)
    held = []
    for _ in range(5):
        scheduler_output = scheduler.schedule()
        stats = scheduler.update_from_output(
            scheduler_output, _model_runner_output(scheduler, scheduler_output)
        )[0].scheduler_stats
        held.append((stats.num_prefill_held_steps, stats.num_prefill_held_tokens))
    assert held == [(1, 0)] * 4 + [(0, 0)]


@pytest.mark.parametrize(
    "long_prefill_token_threshold, held_tokens", [(128, 2 * 128), (384, 511)]
)
def test_the_held_tokens_of_a_step_are_at_most_what_its_budget_leaves(
    long_prefill_token_threshold, held_tokens
):
    """Two lower-priority prefill chunks held in one step count what the running
    loop would have scheduled for them: a held chunk takes nothing from the
    step's budget, so each counts against what the chunks held before it leave.
    Both chunks' 128 fit in the 511 tokens the decode leaves; of 384 each, the
    second chunk would have had 127."""
    scheduler = _create_scheduler(
        priority_prefill_budget_cap=True,
        priority_prefill_budget_cap_tokens=4,
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_model_len=2048,
        num_blocks=400,
    )
    _run_until_decoding(scheduler, _request("decode", 40, priority=2, arrival_time=0))
    scheduler.add_request(_request("a", 2000, priority=2, arrival_time=1))
    scheduler.add_request(_request("b", 2000, priority=2, arrival_time=2))
    assert _step(scheduler).num_scheduled_tokens.keys() == {"decode", "a", "b"}
    scheduler.add_request(_request("hi", 100, priority=0, arrival_time=3))
    held = []
    for _ in range(5):
        scheduler_output = scheduler.schedule()
        stats = scheduler.update_from_output(
            scheduler_output, _model_runner_output(scheduler, scheduler_output)
        )[0].scheduler_stats
        held.append((stats.num_prefill_held_steps, stats.num_prefill_held_tokens))
    assert held == [(1, held_tokens)] * 4 + [(0, 0)]


@pytest.mark.parametrize("budget_cap", [False, True])
def test_prometheus_counts_the_budget_caps_holds(budget_cap):
    """As vllm:priority_prefill_held_steps and _tokens, only with the cap on."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m"),
        scheduler_config=SchedulerConfig.default_factory(
            policy="priority", priority_prefill_budget_cap=budget_cap
        ),
    )
    try:
        logger = PrometheusStatLogger(vllm_config)
        logger.record(
            SchedulerStats(num_prefill_held_steps=2, num_prefill_held_tokens=512), None
        )
        labels = {"model_name": "facebook/opt-125m", "engine": "0"}
        held = [
            prometheus_client.REGISTRY.get_sample_value(f"{name}_total", labels)
            for name in (
                "vllm:priority_prefill_held_steps",
                "vllm:priority_prefill_held_tokens",
            )
        ]
        assert held == ([2, 512] if budget_cap else [None, None])
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
