# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict, deque
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import ConstantList

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[i] for i in range(len(req_ids))],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 5])
def test_stop_by_max_tokens(max_tokens: int):
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=2, max_tokens=max_tokens)
    req0, req1 = requests

    expected_total_num_scheduled_tokens = 0
    sched_outputs: deque[SchedulerOutput] = deque()
    scheduler.add_request(req0)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req0.num_prompt_tokens + max_tokens - 1

    scheduler.add_request(req1)
    sched_outputs.append(scheduler.schedule())
    expected_total_num_scheduled_tokens += req1.num_prompt_tokens + max_tokens - 1

    total_num_scheduled_tokens = 0
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        total_num_scheduled_tokens += sched_output.total_num_scheduled_tokens
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert req0.num_output_tokens == max_tokens
    assert req1.num_output_tokens == max_tokens
    # Ensure we aren't scheduling more tokens than necessary.
    assert total_num_scheduled_tokens == expected_total_num_scheduled_tokens


def test_abort():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_preempt():
    scheduler = create_scheduler(async_scheduling=True)
    requests = create_requests(num_requests=10, max_tokens=20)

    for req in requests:
        scheduler.add_request(req)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    abort_order = [0, 8, 3, 1, 6, 4, 2, 5, 7, 9]
    abort_order_copy = abort_order.copy()

    def abort_request():
        if not abort_order:
            return
        req = requests[abort_order.pop(0)]
        scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)

    while sched_outputs:
        # Abort a scheduled request.
        abort_request()
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)

        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)

    for i, req in enumerate(requests):
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.num_output_tokens == abort_order_copy.index(i)


def test_prefix_caching_for_prefill_dedup():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=3,
        same_prompt=True,
        block_size=BLOCK_SIZE,
    )

    # Two requests with the same prompt.
    req0 = requests.pop(0)
    req1 = requests.pop(0)
    scheduler.add_request(req0)
    scheduler.add_request(req1)

    sched_outputs: deque[SchedulerOutput] = deque()
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)
    # Make sure prefix caching de-duplicates the prompts in the same step,
    # so all the blocks except the last are shared between the two requests.
    assert len(sched_output.num_scheduled_tokens) == 2
    assert sched_output.num_scheduled_tokens[req0.request_id] == num_prompt_tokens
    assert (
        sched_output.num_scheduled_tokens[req1.request_id]
        == num_prompt_tokens % BLOCK_SIZE
    )

    sched_outputs.append(scheduler.schedule())
    while sched_outputs:
        added_req = None
        if requests:
            added_req = requests.pop(0)
            scheduler.add_request(added_req)
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
            if added_req:
                assert (
                    sched_output.num_scheduled_tokens[added_req.request_id]
                    == num_prompt_tokens % BLOCK_SIZE
                )

    assert scheduler.get_num_unfinished_requests() == 0


def test_prefix_caching_for_multi_turn():
    CHUNK_SIZE = 1000
    BLOCK_SIZE = 16
    num_prompt_tokens = 100
    num_output_tokens = 200
    scheduler = create_scheduler(
        async_scheduling=True,
        max_num_batched_tokens=CHUNK_SIZE,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
    )
    requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )

    for req in requests:
        scheduler.add_request(req)
    sched_outputs: deque[SchedulerOutput] = deque()
    sched_outputs.append(scheduler.schedule())
    sched_outputs.append(scheduler.schedule())

    # Process the requests.
    while sched_outputs:
        sched_output = sched_outputs.popleft()
        model_runner_output = _make_model_runner_output(sched_output)
        scheduler.update_from_output(sched_output, model_runner_output)
        sched_output = scheduler.schedule()
        if sched_output.num_scheduled_tokens:
            sched_outputs.append(sched_output)
    assert scheduler.get_num_unfinished_requests() == 0

    # Create next-turn requests whose prompts are the full output of the
    # previous turn.
    next_turn_requests = create_requests(
        num_requests=5,
        num_tokens=num_prompt_tokens + num_output_tokens,
        max_tokens=num_output_tokens,
        block_size=BLOCK_SIZE,
    )
    for i, req in enumerate(next_turn_requests):
        req.prompt_token_ids = requests[i].prompt_token_ids + list(
            requests[i].output_token_ids
        )
        req._all_token_ids = req.prompt_token_ids.copy()
        req.all_token_ids = ConstantList(req._all_token_ids)
        req.block_hashes = []
        req.update_block_hashes()

    # Schedule the next-turn requests.
    for req in next_turn_requests:
        scheduler.add_request(req)
    sched_output = scheduler.schedule()
    sched_outputs.append(sched_output)

    # Make sure the next-turn requests get prefix cache hit by the previous
    # requests.
    for req in next_turn_requests:
        assert sched_output.num_scheduled_tokens[req.request_id] == (
            req.num_prompt_tokens % BLOCK_SIZE
        )


def test_abort_request_when_structured_output_fsm_cannot_advance():
    scheduler = object.__new__(AsyncScheduler)
    request = create_requests(num_requests=1, num_tokens=1)[0]
    request.structured_output_request = Mock()
    request.structured_output_request.grammar = Mock()
    request.structured_output_request.grammar.accept_tokens.return_value = False
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.should_advance.return_value = True
    scheduler.structured_output_manager.trim_reasoning_for_advance.side_effect = (
        lambda request, new_token_ids: new_token_ids
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.vllm_config = Mock()
    scheduler.vllm_config.model_config.enable_return_routed_experts = False
    scheduler.enable_return_routed_experts = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.make_stats = Mock(return_value=None)
    scheduler.max_model_len = 128

    def free_request(req, delay_free_blocks=False):
        scheduler.finished_req_ids.add(req.request_id)
        scheduler.requests.pop(req.request_id, None)
        return None, None

    scheduler._free_request = Mock(side_effect=free_request)

    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
        scheduled_spec_decode_tokens={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[[123]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

    scheduler.update_from_output(output, model_runner_output)

    assert request.resumable is False
    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert not scheduler.running


class PipelinedEngine:
    """Drive a real AsyncScheduler like EngineCore.step_with_batch_queue:
    schedule until the batch queue is full, then process the oldest step's
    output. Async PP runs pp_size+1 concurrent batches, so up to pp_size
    steps are in flight at each schedule() call -- the window in which
    preemption must handle output that has not yet returned. (Single-GPU e2e
    tests can never create this window: at PP=1, exactly one step is in
    flight and it is processed before a preempted request can resume.)

    The model runner is emulated with the V2 runner's own bookkeeping, from
    only what the scheduler serializes to it: slots flushed on
    preempted_req_ids, resumed requests re-added from the NewRequestData
    snapshot, sampling when a step reaches the end of the runner's own view
    of the sequence. This makes preemption races observable: a stale token
    delivered after a resume is scheduled extends the scheduler's sequence
    but not the runner's.

    Every sample emits a globally unique token tagged with its sampled
    position, so tests can assert exact delivery.
    """

    def __init__(
        self,
        scheduler: AsyncScheduler,
        queue_size: int,
        accept_drafts: Callable[[int, str, int], int] | None = None,
    ):
        self.scheduler = scheduler
        self.queue_size = queue_size
        self.accept_drafts = accept_drafts
        # In-flight steps: (scheduler_output, new_reqs snapshot) in FIFO order.
        self.queue: deque[tuple[SchedulerOutput, list[tuple[str, int, int]]]] = deque()
        # Runner-side request state: req_id -> [seq_len, num_computed] as the
        # runner sees them (its own sampled tokens, not the scheduler's).
        self.runner_view: dict[str, list[int]] = {}
        # All tokens the fake runner ever sampled, per request, in order.
        self.emitted: dict[str, list[int]] = defaultdict(list)
        # Sequence position each (globally unique) token was sampled for.
        self.emitted_position: dict[int, int] = {}
        self.step_idx = 0
        self._next_token = 1000

    def _schedule(self) -> bool:
        scheduler_output = self.scheduler.schedule()
        self.step_idx += 1
        # Snapshot what NewRequestData serializes at schedule time (both new
        # and resumed requests for the V2 runner).
        new_reqs = [
            (r.req_id, len(r.prefill_token_ids), r.num_computed_tokens)
            for r in scheduler_output.scheduled_new_reqs
        ]
        # Enqueue empty steps too (the engine executes them), so the runner
        # still observes their preempted/finished request ids in step order.
        self.queue.appendleft((scheduler_output, new_reqs))
        return True

    def _process_oldest_step(self) -> None:
        scheduler_output, new_reqs = self.queue.pop()
        # Worker-side state updates, in step order: flush preempted/finished
        # slots, then (re-)add new/resumed requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id in scheduler_output.finished_req_ids or ():
            self.runner_view.pop(req_id, None)
        for req_id, seq_len, num_computed in new_reqs:
            self.runner_view[req_id] = [seq_len, num_computed]

        req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        sampled_token_ids: list[list[int]] = []
        for req_id in req_ids:
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            view = self.runner_view.get(req_id)
            if view is None:
                # Slot already flushed (request finished/aborted mid-flight).
                sampled_token_ids.append([])
                continue
            seq_len, num_computed = view
            end = num_computed + num_scheduled
            if end < seq_len:
                # Partial prefill by the runner's own bookkeeping: no sample.
                view[1] = end
                sampled_token_ids.append([])
                continue
            drafts = scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            num_accepted = (
                min(self.accept_drafts(self.step_idx, req_id, len(drafts)), len(drafts))
                if drafts and self.accept_drafts
                else 0
            )
            num_rejected = len(drafts) - num_accepted
            tokens = list(range(self._next_token, self._next_token + 1 + num_accepted))
            self._next_token += 1 + num_accepted
            self.emitted[req_id].extend(tokens)
            sampled_token_ids.append(tokens)
            # Rejected drafts roll back computed; the sampled tokens extend
            # the runner's sequence.
            view[1] = end - num_rejected
            view[0] = view[1] + 1
            for offset, token in enumerate(tokens):
                self.emitted_position[token] = view[0] - len(tokens) + offset
        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def run(
        self,
        max_steps: int = 2000,
        before_step: Callable[[int, "PipelinedEngine"], None] | None = None,
    ) -> None:
        for i in range(max_steps):
            if not self.scheduler.has_requests() and not self.queue:
                return
            if before_step is not None:
                before_step(i, self)
            scheduled = (
                self.scheduler.has_requests()
                and len(self.queue) < self.queue_size
                and self._schedule()
            )
            if scheduled and len(self.queue) < self.queue_size:
                # Queue not yet full: the engine returns without blocking.
                continue
            if self.queue:
                self._process_oldest_step()
        raise AssertionError("engine loop did not converge")


def _create_async_pp_scheduler(
    num_spec: int, pp_size: int = 3, num_blocks: int = 5
) -> AsyncScheduler:
    scheduler = create_scheduler(
        async_scheduling=True,
        num_speculative_tokens=num_spec or None,
        speculative_method="ngram_gpu" if num_spec else None,
        use_v2_model_runner=True,
        num_blocks=num_blocks,
        block_size=16,
        max_num_batched_tokens=512,
    )
    # Emulate PP at the scheduler level; constructing with
    # pipeline_parallel_size>1 requires that many visible GPUs. Drive with
    # queue_size=pp_size+1 (V2 async PP runs pp_size+1 concurrent batches).
    scheduler.pp_size = pp_size
    scheduler.use_pp = pp_size > 1
    return scheduler


def _assert_ordered_subset(delivered: list[int], emitted: list[int]) -> None:
    """Delivered tokens must be an order-preserving subset of the emitted
    tokens with no duplicates (tokens are globally unique)."""
    it = iter(emitted)
    for token in delivered:
        assert token in it, f"token {token} delivered out of order or twice"


def _assert_positions_consistent(req, engine: PipelinedEngine) -> None:
    """The i-th delivered output token must be one the runner sampled for
    exactly sequence position prompt_len + i: catches a preempted request's
    stale output landing on a position the resumed request resampled (or
    vice versa), which token-stream equality alone cannot see."""
    for i, token in enumerate(req.output_token_ids):
        expected = req.num_prompt_tokens + i
        actual = engine.emitted_position[token]
        assert actual == expected, (
            f"output {i} of {req.request_id}: token sampled for position "
            f"{actual}, delivered as position {expected}"
        )


@pytest.mark.parametrize("num_spec", [0, 3])
def test_kv_pressure_preemption_with_inflight_output(num_spec: int):
    """KV-pressure preemption of requests with in-flight async output.

    PP=3 + async scheduling (batch queue of 4), a block pool small enough
    that decodes contend and preempt mid-flight, and staggered arrivals so
    the batch queue actually pipelines. A preempted request's in-flight steps
    still return: their tokens must be delivered exactly once, their stale
    spec-rejection counts must not corrupt the rolled-back counters, and the
    resume must not resample a position that output later delivers.

    Regression for the num_output_placeholders underflow EngineCore crash:
    with the fix reverted, the num_spec=3 variant fails with exactly
    ``assert request.num_output_placeholders >= 0`` when a stale spec output
    returns after the preempted request was resumed and sampled.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe that the scenario under test actually occurs.
    preempts_with_inflight_output = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal preempts_with_inflight_output
        if request.num_in_flight_tokens > 0:
            preempts_with_inflight_output += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    def add_requests(step: int, engine: PipelinedEngine):
        if pending:
            scheduler.add_request(pending.pop(0))

    engine = PipelinedEngine(
        scheduler,
        queue_size=4,
        # Deterministically vary spec acceptance so stale outputs carry
        # nonzero rejection counts.
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=add_requests)

    assert preempts_with_inflight_output > 0, (
        "test did not exercise preemption with in-flight output"
    )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Lossless: delivered tokens are exactly the sampled tokens, in order
        # (the excluded tail was emitted after the request finished).
        emitted = engine.emitted[req.request_id]
        assert list(req.output_token_ids) == emitted[:max_tokens]
        _assert_positions_consistent(req, engine)


@pytest.mark.parametrize("pp_size", [1, 3])
def test_reset_prefix_cache_with_inflight_output_under_kv_pressure(pp_size: int):
    """reset_prefix_cache(reset_running_requests=True) resumes requests in
    the same step it preempts them, so in-flight output must be dropped (the
    resume resamples those positions).

    pp_size=1: regression for the frame-based discard this fix replaces,
    which with spec decode drained one *token* count per output frame and
    over-discarded, corrupting the fresh frames after the resume.
    pp_size=3: back-to-back resets, so the second re-preempts requests whose
    dropped stale share is still in flight -- it must be recorded once (not
    accumulated) and stay dropped.
    """
    max_tokens = 24
    scheduler = _create_async_pp_scheduler(num_spec=3, pp_size=pp_size)
    requests = create_requests(
        num_requests=8, num_tokens=8, max_tokens=max_tokens, ignore_eos=True
    )
    pending = list(requests)
    for _ in range(2):
        scheduler.add_request(pending.pop(0))

    # Observe re-preemptions with an undrained stale share (the
    # double-count hazard).
    repreempts_with_stale = 0
    orig_preempt = scheduler._preempt_request

    def counting_preempt(request, timestamp, **kwargs):
        nonlocal repreempts_with_stale
        if getattr(request, "num_stale_output_tokens", 0) > 0:
            repreempts_with_stale += 1
        return orig_preempt(request, timestamp, **kwargs)

    scheduler._preempt_request = counting_preempt

    resets = 0
    reset_steps = {6, 14} if pp_size == 1 else {6, 7, 18, 19}

    def before_step(step: int, engine: PipelinedEngine):
        nonlocal resets
        if pending:
            scheduler.add_request(pending.pop(0))
        if step in reset_steps and (engine.queue or scheduler.running):
            scheduler.reset_prefix_cache(reset_running_requests=True)
            resets += 1

    engine = PipelinedEngine(
        scheduler,
        queue_size=pp_size + 1,
        accept_drafts=lambda step, req_id, n: (step + int(req_id)) % (n + 1),
    )
    engine.run(before_step=before_step)

    assert resets > 0, "test did not exercise reset_prefix_cache"
    if pp_size > 1:
        # The re-preempt-while-stale-pending window needs pipeline depth.
        assert repreempts_with_stale > 0, (
            "test did not exercise re-preemption with an undrained stale share"
        )
    for req in requests:
        assert req.is_finished()
        assert req.num_output_tokens == max_tokens
        # Dropped tokens are never delivered; order must be preserved with
        # no duplicates.
        _assert_ordered_subset(
            list(req.output_token_ids), engine.emitted[req.request_id]
        )
        _assert_positions_consistent(req, engine)
        # All stale shares fully drained by the end.
        assert getattr(req, "num_stale_output_tokens", 0) == 0
