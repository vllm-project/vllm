# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.distributed.aux_output_connector.test_store import (
    _input_batch,
    _make_worker,
)
from tests.distributed.aux_output_connector.utils import fake_store
from vllm.config import AuxOutputConfig, KVTransferConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputSchedulerConnector,
)
from vllm.distributed.aux_output_connector.object_store import (
    AuxSampleKeys,
    aux_session_pattern,
    read_aux_rows,
    release_aux_keys,
)
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
    salt_aux_session,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.engine import CheckpointState
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.kv_offload.base import OffloadingSpec, get_offload_block_hash
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputGrammar
from vllm.v1.utils import ConstantList
from vllm.v1.worker.worker_base import WorkerBase

from .utils import (
    EOS_TOKEN_ID,
    create_requests,
    create_scheduler,
    mock_kv,
    new_kv_cache_group,
)

pytestmark = pytest.mark.cpu_test


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
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


def test_no_spec_decode_padding_up_to_max_model_len():
    """Uniform spec-decode padding must leave room for the sampled token.

    Padding a single-token decode up to exactly max_model_len makes the
    running-loop max_model_len cap negative on the next step (async
    scheduling schedules it before the output that would stop the request),
    and a negative num_scheduled_tokens crashes the model runner. The
    request is scheduled un-padded instead.

    Modelled on a disaggregated decode worker, where every request arrives
    with all but the last prompt token already computed.
    """
    max_model_len = 32
    num_spec = 1
    scheduler = create_scheduler(
        async_scheduling=True,
        max_model_len=max_model_len,
        num_speculative_tokens=num_spec,
        speculative_method="ngram_gpu",
    )

    # A running decode, so padding's "something already scheduled and no
    # prefill this step" precondition holds for the request below.
    (filler,) = create_requests(num_requests=1, num_tokens=4, req_ids=["filler"])
    scheduler.add_request(filler)
    sched_output = scheduler.schedule()
    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    (request,) = create_requests(
        num_requests=1, num_tokens=max_model_len - num_spec, req_ids=["boundary"]
    )
    request.num_computed_tokens = request.num_tokens - 1
    scheduler.add_request(request)

    sched_output = scheduler.schedule()
    assert sched_output.num_scheduled_tokens["boundary"] == 1
    assert "boundary" not in sched_output.scheduled_spec_decode_tokens
    assert request.num_computed_tokens < max_model_len

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


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


def test_connector_metadata_precedes_async_placeholder_advance(monkeypatch):
    """Mirroring must see earlier unresolved outputs, excluding the current step."""
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    (request,) = create_requests(num_requests=1, num_tokens=4, max_tokens=4)
    scheduler.add_request(request)

    assert scheduler.connector is not None
    build_connector_meta = scheduler.connector.build_connector_meta
    observed_placeholders = []

    def record_placeholders(scheduler_output):
        observed_placeholders.append(request.num_output_placeholders)
        return build_connector_meta(scheduler_output)

    monkeypatch.setattr(
        scheduler.connector, "build_connector_meta", record_placeholders
    )
    scheduler.schedule()
    scheduler.schedule()

    assert observed_placeholders == [0, 1]


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
    request.structured_output_request.grammar = Mock(spec=StructuredOutputGrammar)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = request.num_tokens
    request.num_output_placeholders = 1

    scheduler.perf_metrics = None
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.structured_output_manager = Mock()
    scheduler.structured_output_manager.accept_tokens.return_value = False
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = Mock()
    scheduler.kv_cache_manager = Mock()
    scheduler.kv_cache_manager.take_events.return_value = None
    scheduler.kv_cache_manager.estimate_cached_tokens.return_value = 0
    scheduler.kv_event_publisher = Mock()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.grammar_compile_error_reqs = {}
    scheduler.encoder_cache_mismatch_reqs = set()
    scheduler.vllm_config = Mock()
    scheduler.aux_output_connector = None
    scheduler.return_sampling_mask = False
    scheduler.recompute_kv_load_failures = False
    scheduler.defer_block_free = False
    scheduler.num_spec_tokens = 0
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
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    model_runner_output = ModelRunnerOutput(
        req_ids=[request.request_id],
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
        num_blocks=num_blocks,
        block_size=16,
        max_num_batched_tokens=512,
    )
    # Emulate PP at the scheduler level; constructing with
    # pipeline_parallel_size>1 requires that many visible GPUs. Drive with
    # queue_size=pp_size+1 (V2 async PP runs pp_size+1 concurrent batches).
    scheduler.decode_stagger = pp_size
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
    `assert request.num_output_placeholders >= 0` when a stale spec output
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


def test_requires_kv_delivery_defaults_to_producer_role():
    # No connector: nothing is handed off, so keep the lossless deliver-stale
    # path on preemption.
    assert create_scheduler(async_scheduling=True).requires_kv_delivery is False
    # Only a producer hands KV off when a request completes.
    for role, expected in (
        ("kv_producer", True),
        ("kv_both", True),
        ("kv_consumer", False),
    ):
        scheduler = create_scheduler(
            async_scheduling=True, use_kv_connector=True, kv_role=role
        )
        assert scheduler.requires_kv_delivery is expected, role


@pytest.mark.parametrize(
    ("kv_role", "defer_free"),
    [
        ("kv_producer", False),
        ("kv_consumer", False),
        ("kv_consumer", True),
    ],
)
def test_kv_pressure_preempt_mid_handoff(kv_role: str, defer_free: bool):
    """P/D race: KV pressure hits while the output of a request's final
    prefill chunk -- the hand-off token that would finish it -- is in flight.

    When the victim's blocks free immediately, the request is preempted. On a
    producer, that output must be dropped so the request recomputes; delivering
    it would finish the request and hand off blocks the preemption already
    freed, so the consumer pulls garbage. A consumer hands nothing off, so it
    keeps the lossless deliver-stale path.

    A consumer with overlapping batches (async scheduling or PP) instead fences
    the victim's free behind its in-flight output, so the allocation retry
    stops instead of preempting; the request then finishes from that output
    once it lands. The gate depends on the platform (async scheduling is
    force-disabled on CPU), so force the flag to cover both paths everywhere.
    """
    is_producer = kv_role == "kv_producer"
    scheduler = create_scheduler(
        async_scheduling=True,
        use_kv_connector=True,
        kv_role=kv_role,
        num_blocks=5,
        block_size=16,
        max_num_batched_tokens=512,
    )
    assert scheduler.requires_kv_delivery is is_producer
    # The production gate requires overlapping batches, which async
    # scheduling only provides off-CPU.
    scheduler.defer_block_free = defer_free

    # 32-token prompts fill 2 blocks each, exhausting the usable pool, so the
    # next decode allocation targets the tail of the running queue (the
    # handoff request) while its prefill output is still in flight.
    decoder = create_requests(
        num_requests=1, num_tokens=32, max_tokens=8, req_ids=["decoder"]
    )[0]
    handoff = create_requests(
        num_requests=1, num_tokens=32, max_tokens=1, req_ids=["handoff"]
    )[0]
    scheduler.add_request(decoder)
    scheduler.add_request(handoff)
    sched_output = scheduler.schedule()
    assert handoff.status == RequestStatus.RUNNING
    assert handoff.num_output_placeholders == 1

    scheduler.schedule()
    if defer_free:
        # The victim's blocks are fenced behind its in-flight output, so
        # preempting it could not satisfy the allocation; the retry stops
        # instead of preempting.
        assert handoff.status == RequestStatus.RUNNING
        assert handoff.num_stale_output_tokens == 0
    else:
        # Blocks free immediately, so the handoff request is preempted.
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_stale_output_tokens == handoff.num_prompt_tokens
        assert handoff.drop_stale_output is is_producer

    scheduler.update_from_output(sched_output, _make_model_runner_output(sched_output))

    assert handoff.num_stale_output_tokens == 0
    if is_producer:
        # Dropped: recomputed from the waiting queue, so the hand-off happens
        # against real KV.
        assert not handoff.is_finished()
        assert handoff.status == RequestStatus.PREEMPTED
        assert handoff.num_output_tokens == 0
        assert handoff.request_id in scheduler.requests
    else:
        assert handoff.is_finished()
        assert handoff.num_output_tokens == 1


def test_resumable_request_handoff():
    """New streaming input resumes with the correct context and stale results
    are ignored.
    """
    scheduler = create_scheduler(async_scheduling=True)
    engine = PipelinedEngine(scheduler, queue_size=3)
    engine._next_token = EOS_TOKEN_ID

    old_prompt_len = 10
    continuation_len = 12
    request_id = "resumable-race"

    request = create_requests(
        1, num_tokens=old_prompt_len, max_tokens=4, req_ids=[request_id]
    )[0]
    request.resumable = True
    continuation = create_requests(
        1, num_tokens=continuation_len, max_tokens=4, req_ids=[request_id]
    )[0]
    continuation.resumable = True

    scheduler.add_request(request)

    for _ in range(3):
        assert engine._schedule()

    engine._process_oldest_step()
    scheduler.add_request(continuation)
    assert engine._schedule()
    new_step, _ = engine.queue[0]
    assert new_step.scheduled_new_reqs[0].num_computed_tokens == old_prompt_len

    all_token_ids_before_stale = list(request.all_token_ids)
    for _ in range(2):
        engine._process_oldest_step()
    assert list(request.output_token_ids) == []
    assert list(request.all_token_ids) == all_token_ids_before_stale

    engine._process_oldest_step()
    assert len(request.output_token_ids) == 1


class SessionEngine:
    """A real scheduler and auxiliary output worker over an in-memory store;
    rows carry their position and weight version, and `close` checks every
    key against the KV each request used. It also plays the KV connector
    workers."""

    def __init__(self, scheduler, block_size=16, store=None, hash_block_size=16):
        self.scheduler = scheduler
        self.block_size = block_size
        self.connector = scheduler.aux_output_connector or AuxOutputSchedulerConnector(
            scheduler.kv_cache_manager, 0, key_prefix="job"
        )
        scheduler.aux_output_connector = self.connector
        self.client, self.store = store or fake_store()
        self.hash_block_size = hash_block_size
        self.worker = _make_worker(16, 256, block_size=hash_block_size)
        self.worker._object_writer = SessionObjectWriter(
            self.store,
            key_prefix="job",
            max_pending_batches=4,
            fields=self.worker._object_fields,
        )
        self.capturer = self.worker._routed_experts_capturer = Mock()
        self.set_version(0)
        self.requests = {}
        self.keys = {}
        self.segments = {}
        self.num_hits = 0
        self._kv_versions = {}
        self._group_sizes = [
            manager.block_size
            for manager in scheduler.kv_cache_manager.coordinator.single_type_managers
        ]
        self._hit_versions = {}
        self._row_versions = {}
        self._in_flight = {}
        self._stopped = []
        self.num_loads = 0
        self.num_stores = 0
        self._tier = {}
        self._store_jobs = {}
        self._kv_outputs = {}
        self._schedule = scheduler.schedule
        self._update_from_output = scheduler.update_from_output
        scheduler.schedule = self.schedule
        scheduler.update_from_output = self.update_from_output

    def set_version(self, version):
        self.version = version
        self.connector.set_weight_version(f"v{version}")

    def pause(self):
        self.worker.drain(self.connector.drain_metadata())

    def schedule(self):
        scheduler_output = self._schedule()
        metadata = scheduler_output.aux_output_connector_metadata
        self.worker.begin_step(metadata)
        size = self.block_size
        kv_metadata = scheduler_output.kv_connector_metadata
        if isinstance(kv_metadata, HashStoreMetadata):
            self._transfer_hash_store(kv_metadata)
        elif kv_metadata is not None:
            self._transfer_kv(scheduler_output)
        block_ids = {}
        request_groups = {}
        starts, counts, rows = [], [], []
        for request_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.scheduler.requests[request_id]
            self.requests[request_id] = request
            start = request.num_computed_tokens - num_tokens
            groups = self.scheduler.kv_cache_manager.get_blocks(request_id).blocks
            blocks = groups[self.connector._group_id]
            block_ids[request_id] = [block.block_id for block in blocks]
            request_groups[request_id] = groups
            if request_id in metadata.sessions:
                self._row_versions[request_id] = {}
                self.num_hits += start > 0
            self._row_versions.setdefault(request_id, {})
            step_rows = np.zeros((num_tokens, 3, 2), dtype=np.uint8)
            step_rows[:, 0, 0] = np.arange(start, start + num_tokens)
            step_rows[:, 0, 1] = self.version
            for position in range(start, start + num_tokens):
                for group, group_size in zip(groups, self._group_sizes):
                    block_id = group[position // group_size].block_id
                    versions = self._kv_versions.get(block_id)
                    if versions is None or len(versions) != group_size:
                        versions = self._kv_versions[block_id] = [None] * group_size
                    versions[position % group_size] = self.version
                self._row_versions[request_id][position] = self.version
            starts.append(start)
            counts.append(num_tokens)
            rows.append(step_rows)
        for request_id in metadata.sessions:
            num_hit_blocks = min(self._row_versions[request_id]) // size
            hit_ids = block_ids[request_id][:num_hit_blocks]
            self._hit_versions[request_id] = [
                version
                for block_id in hit_ids
                for version in self._kv_versions[block_id]
            ]
            num_hit = num_hit_blocks * size
            for group, group_size in zip(request_groups[request_id], self._group_sizes):
                for index, block in enumerate(group[: num_hit // group_size]):
                    positions = range(index * group_size, (index + 1) * group_size)
                    assert block.is_null or self._kv_versions[block.block_id] == [
                        self._kv_versions[hit_ids[p // size]][p % size]
                        for p in positions
                    ]
        step = None
        if rows:
            routing = np.concatenate(rows)
            self.capturer.snapshot_routing_data.return_value = torch.from_numpy(routing)
            step = self.worker.prepare_output(
                _input_batch(
                    list(block_ids), np.array(starts), np.cumsum([0] + counts)
                ),
                torch.from_numpy(routing[:, 0, :1].astype(np.float32)),
                _two_logits,
            )
        self._in_flight[id(scheduler_output)] = step
        return scheduler_output

    def _transfer_hash_store(self, metadata):
        store = self.scheduler.connector.store
        for key, block_id in metadata.stores:
            store[key] = list(self._kv_versions[block_id])
            self.num_stores += 1
        for key, block_id in metadata.loads:
            self._kv_versions[block_id] = list(store[key])
            self.num_loads += 1

    def _transfer_kv(self, scheduler_output):
        metadata = scheduler_output.kv_connector_metadata
        worker_meta = OffloadingWorkerMetadata()
        finished = set()
        for job_id, job in self._store_jobs.items():
            worker_meta.mark_completed(job_id, success=True)
            self.num_stores += 1
            for block_id, chunk_id in zip(
                job.src_spec.block_ids.tolist(), job.dst_spec.block_ids.tolist()
            ):
                self._tier[chunk_id] = list(self._kv_versions[block_id])
        self._store_jobs = metadata.store_jobs
        for job_id, job in metadata.load_jobs.items():
            worker_meta.mark_completed(job_id, success=True)
            if not job.sync:
                finished.add(job.req_id)
            self.num_loads += 1
            for chunk_id, block_id in zip(
                job.src_spec.block_ids.tolist(), job.dst_spec.block_ids.tolist()
            ):
                self._kv_versions[block_id] = list(self._tier[chunk_id])
        self._kv_outputs[id(scheduler_output)] = KVConnectorOutput(
            finished_recving=finished, kv_connector_worker_meta=worker_meta
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        kv_output = self._kv_outputs.pop(id(scheduler_output), None)
        if kv_output is not None:
            model_runner_output.kv_connector_output = kv_output
        step = self._in_flight.pop(id(scheduler_output))
        if step is not None:
            num_sampled = np.array(
                [len(tokens) for tokens in model_runner_output.sampled_token_ids]
            )
            drafts = scheduler_output.scheduled_spec_decode_tokens
            num_rejected = np.array(
                [
                    max(len(drafts.get(request_id, ())) + 1 - len(tokens), 0)
                    if tokens
                    else 0
                    for request_id, tokens in zip(
                        model_runner_output.req_ids,
                        model_runner_output.sampled_token_ids,
                    )
                ]
            )
            step.enqueue_cpu_copy(num_sampled, num_rejected)
            model_runner_output.aux_output_connector_output = step.process_output()
        outputs = self._update_from_output(scheduler_output, model_runner_output)
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                if output.aux_output_keys is None:
                    continue
                request_id = output.request_id
                row_versions = self._row_versions[request_id]
                num_rows = self.requests[request_id].num_tokens - 1
                versions = self._hit_versions[request_id]
                versions = versions + [
                    row_versions[position]
                    for position in range(len(versions), num_rows)
                ]
                self.keys[request_id] = output.aux_output_keys
                self.segments[request_id] = output.aux_version_segments
                self._stopped.append((request_id, output.aux_output_keys, versions))
        return outputs

    def step(self, scheduler_output, sampled):
        request_ids = list(scheduler_output.num_scheduled_tokens)
        outputs = self.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=request_ids,
                sampled_token_ids=[sampled.get(r, []) for r in request_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        return {
            output.request_id: output
            for client_outputs in outputs.values()
            for output in client_outputs.outputs
        }

    def run(self, before_step=None):
        outputs = {}
        while self.scheduler.has_requests():
            scheduler_output = self.schedule()
            requests = self.scheduler.requests
            sampled = {
                request_id: [0]
                for request_id in scheduler_output.num_scheduled_tokens
                if requests[request_id].num_computed_tokens
                == requests[request_id].num_tokens
            }
            outputs.update(self.step(scheduler_output, sampled))
            if before_step is not None:
                before_step()
        return outputs

    def close(self, stale_names=False, check_leaks=True):
        finished = not self.scheduler.has_unfinished_requests()
        if finished:
            self.run()
            self.pause()
        self.worker.close()
        size = self.block_size
        step = size // self.hash_block_size
        writes = Counter(key for batch in self.client.writes for key in batch)
        if not stale_names:
            assert all(
                payloads.count(payloads[0]) == len(payloads)
                for payloads in self.client.payloads.values()
            )
        for request_id, keys, versions in self._stopped:
            request = self.requests[request_id]
            session = request.sampling_params.aux_session
            num_full = sum("/tail/" not in key for key in keys)
            assert keys[:num_full] == [
                f"job/{session}/{block_hash.hex()}"
                for block_hash in request.block_hashes[step - 1 :: step][:num_full]
            ]
            assert all("/tail/" in key for key in keys[num_full:])
            assert len(keys) <= num_full + 1
            rows = read_aux_rows(self.store, keys, "routed_experts", timeout_s=0)
            assert rows[:, 0, 0].tolist() == list(range(request.num_tokens - 1))
            row_versions = rows[:, 0, 1].tolist()
            for index, key in enumerate(keys):
                end = None if "/tail/" in key else (index + 1) * size
                block = slice(index * size, end)
                if writes[key] == 1 or not stale_names:
                    assert row_versions[block] == versions[block]
        if finished and check_leaks:
            assert self.client.objects.keys() <= self.listed()

    def listed(self):
        return {key for _, keys, _ in self._stopped for key in keys}


def _session_requests(
    request_ids,
    num_tokens,
    max_tokens,
    lookahead=0,
    token=0,
    session="s",
    segments=None,
    stop_token_ids=None,
    hash_block_size=16,
    prompt_logprobs=None,
    **held,
):
    init_none_hash(sha256)
    params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
        stop_token_ids=stop_token_ids,
        aux_session=session,
        aux_version_segments=segments,
        prompt_logprobs=prompt_logprobs,
        **held,
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    salted = SimpleNamespace(
        request_id=request_ids[0],
        sampling_params=params,
        prompt_token_ids=[token] * num_tokens,
        prompt_embeds=None,
        cache_salt=None,
        cache_salt_segments=None,
    )
    config = SimpleNamespace(aux_output_config=SimpleNamespace(object_key_prefix="j"))
    salt_aux_session(config, salted, "v0", hash_block_size)
    return [
        Request(
            request_id=request_id,
            prompt_token_ids=[token] * num_tokens,
            sampling_params=params,
            pooling_params=None,
            cache_salt=salted.cache_salt,
            cache_salt_segments=list(salted.cache_salt_segments or ()),
            block_hasher=get_request_block_hasher(hash_block_size, sha256, lookahead),
        )
        for request_id in request_ids
    ]


@pytest.mark.parametrize(
    "async_scheduling,queue_size,lookahead",
    [(False, 1, 0), (True, 2, 0), (True, 2, 1)],
    ids=["sync", "async", "async-lookahead"],
)
def test_session_keys_resolve_to_the_rows_of_the_kv_each_request_used(
    async_scheduling: bool, queue_size: int, lookahead: int
):
    """Session requests under KV pressure with weight updates between steps.

    Siblings recompute the same blocks across weight updates, later turns hit
    them, preempted requests resume, requests stop exactly on an object
    boundary, and one request is aborted after it was scheduled.
    """
    scheduler = create_scheduler(
        async_scheduling=async_scheduling,
        enable_prefix_caching=True,
        num_blocks=20,
        max_num_batched_tokens=64,
    )
    engine = SessionEngine(scheduler)
    pending = [
        *_session_requests([f"0-{i}" for i in range(5)], 70, 27, lookahead),
        *_session_requests([f"1-{i}" for i in range(3)], 102, 27, lookahead),
    ]
    request_ids = {request.request_id for request in pending}
    aborted = "0-3"
    num_preemptions = 0
    preempt_request = scheduler._preempt_request

    def preempt(request, timestamp, **kwargs):
        nonlocal num_preemptions
        num_preemptions += 1
        return preempt_request(request, timestamp, **kwargs)

    scheduler._preempt_request = preempt

    def before_step(step: int, _: PipelinedEngine):
        engine.set_version(step // 4)
        if aborted in engine.requests and aborted in scheduler.requests:
            scheduler.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
        if step % 2 == 0 and pending:
            scheduler.add_request(pending.pop(0))

    scheduler.add_request(pending.pop(0))
    PipelinedEngine(scheduler, queue_size=queue_size).run(before_step=before_step)
    engine.close()

    assert engine.keys.keys() <= request_ids - {aborted}
    assert len(engine.keys) >= len(request_ids) - 2
    assert num_preemptions > 0 and engine.num_hits > 0
    assert any(len(segments) > 2 for segments in engine.segments.values())


@pytest.mark.parametrize("sliding_window", [None, 32], ids=["one-group", "two-groups"])
def test_session_siblings_share_no_objects(sliding_window):
    engine = _tier_engine(sliding_window=sliding_window)
    for request in _session_requests(["first", "second"], 40, 1):
        engine.scheduler.add_request(request)
    engine.run()
    engine.close()
    first, second = engine.keys["first"], engine.keys["second"]
    assert len(first) == len(second) == 3 and not set(first) & set(second)


def test_session_request_aborted_before_its_output_leaves_no_objects():
    scheduler = create_scheduler(enable_prefix_caching=True)
    engine = SessionEngine(scheduler)
    first, second = _session_requests(["first", "second"], 40, 1)
    scheduler.add_request(first)
    scheduler_output = engine.schedule()
    scheduler.finish_requests("first", RequestStatus.FINISHED_ABORTED)
    engine.step(scheduler_output, {"first": [1]})

    scheduler.add_request(second)
    scheduler_output = engine.schedule()
    assert scheduler_output.num_scheduled_tokens == {"second": 40}
    engine.step(scheduler_output, {"second": [2]})
    engine.close()
    assert "first" not in engine.keys and engine.client.removes
    assert set(engine.client.objects) == set(engine.keys["second"])


def _plan_rows(engine, plan):
    engine.pause()
    rows = read_aux_rows(engine.store, plan, "routed_experts", timeout_s=0)
    assert set(engine.client.objects) <= set(plan)
    return rows[:, 0, 0].tolist(), rows[:, 0, 1].tolist()


def test_session_turn_after_a_reset_keeps_the_products_first_produced():
    engine = _tier_engine(connector=None, hash_block_size=4)
    _run_turn(engine, 1, "first", 40, 5, hash_block_size=4, aux_held=0)
    held = engine.segments["first"][-1][0]
    first = engine.requests["first"]
    assert held == first.num_tokens - 1 and held % 16
    assert engine.scheduler.reset_prefix_cache()
    _run_turn(
        engine,
        2,
        "second",
        first.num_tokens + 8,
        3,
        segments=engine.segments["first"],
        hash_block_size=4,
        aux_held=held,
    )
    end = engine.segments["second"][-1][0]
    positions, versions = _plan_rows(
        engine, engine.keys["first"] + engine.keys["second"]
    )
    assert positions == list(range(end))
    assert set(versions[:held]) == {1} and set(versions[held:]) == {2}


def test_held_session_response_the_frontend_dropped_leaves_no_objects():
    engine = _tier_engine(connector=None, hash_block_size=4)
    _run_turn(engine, 1, "first", 40, 5, hash_block_size=4, aux_held=0)
    held = engine.segments["first"][-1][0]
    outputs = _run_turn(
        engine,
        2,
        "second",
        held + 1 + 30,
        3,
        segments=engine.segments["first"],
        hash_block_size=4,
        aux_held=held,
    )
    assert sum("/tail/" not in key for key in engine.keys["second"]) >= 2
    engine._stopped = [entry for entry in engine._stopped if entry[0] != "second"]
    processed = OutputProcessor(None, log_stats=False).process_outputs(
        [outputs["second"]]
    )
    assert not processed.request_outputs
    engine.connector.release_unreturned(processed.unreturned_aux_keys)
    engine.pause()
    assert set(engine.client.objects) == set(engine.keys["first"])


def test_session_retry_after_an_abort_keeps_the_products_of_the_aborted_attempt():
    engine = _tier_engine(connector=None, hash_block_size=4)
    scheduler = engine.scheduler
    engine.set_version(1)
    (request,) = _session_requests(["first"], 40, 20, hash_block_size=4, aux_held=0)
    scheduler.add_request(request)
    for _ in range(6):
        engine.step(engine.schedule(), {"first": [0]})
    scheduler.finish_requests("first", RequestStatus.FINISHED_ABORTED)
    keys, segments = engine.connector.pop_abort_output("first")
    held = segments[-1][0]
    assert keys and held % 16 == 0 and 0 < held < request.num_tokens - 1
    assert scheduler.reset_prefix_cache()
    _run_turn(
        engine,
        2,
        "retry",
        request.num_tokens,
        4,
        segments=segments,
        hash_block_size=4,
        aux_held=held,
    )
    end = engine.segments["retry"][-1][0]
    positions, versions = _plan_rows(engine, keys + engine.keys["retry"])
    assert positions == list(range(end))
    assert set(versions[:held]) == {1} and set(versions[held:]) == {2}


def test_preempted_session_request_that_stops_on_its_output_in_flight():
    scheduler = create_scheduler(
        async_scheduling=True, enable_prefix_caching=True, num_blocks=3
    )
    engine = SessionEngine(scheduler)
    (first,) = _session_requests(["first"], 15, 5, token=1)
    (second,) = _session_requests(["second"], 16, 1, lookahead=1)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [1]})

    scheduler.add_request(second)
    step = engine.schedule()
    assert step.num_scheduled_tokens == {"first": 1, "second": 16}
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(step, {"first": [2], "second": [3]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [4]})
    engine.close()
    assert outputs["second"].aux_output_keys is None


def test_preempted_session_request_whose_accepted_drafts_complete_an_object():
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        num_speculative_tokens=2,
        speculative_method="ngram_gpu",
        num_blocks=12,
    )
    engine = SessionEngine(scheduler)
    (first,) = _session_requests(["first"], 26, 30, token=1)
    (second,) = _session_requests(["second"], 46, 4)
    scheduler.add_request(first)
    engine.step(engine.schedule(), {"first": [1]})

    scheduler.add_request(second)
    prefill = engine.schedule()
    drafts = engine.schedule()
    assert drafts.num_scheduled_tokens == {"first": 3, "second": 3}
    engine.step(prefill, {"first": [2, 3, 4], "second": [5]})
    block_pool = scheduler.kv_cache_manager.block_pool
    block_pool.get_new_blocks(block_pool.get_num_free_blocks())
    preempting = engine.schedule()
    assert preempting.preempted_req_ids == {"second"}
    outputs = engine.step(drafts, {"first": [6, 7, 8], "second": [9, 10, 11]})
    assert outputs["second"].finish_reason is not None
    engine.step(preempting, {"first": [12]})
    engine.close()
    assert outputs["second"].aux_output_keys is None


def _two_logits(hidden_states):
    return torch.cat([hidden_states, -hidden_states], dim=1)


def test_session_fields_without_routed_experts_through_the_scheduler():
    """Session and plain requests share chunked-prefill and decode steps."""
    scheduler = create_scheduler(enable_prefix_caching=True, max_num_batched_tokens=48)
    scheduler.aux_output_connector = AuxOutputSchedulerConnector(
        scheduler.kv_cache_manager, 0, enable_routed_experts=False, key_prefix="job"
    )
    _, store = fake_store()
    worker = _make_worker(16, 256, enable_r3=False, shape_per_token=(1,), block_size=16)
    worker._return_hidden_states = True
    worker._topk_logprobs = 2
    worker._object_writer = SessionObjectWriter(
        store, key_prefix="job", max_pending_batches=4, fields=worker._object_fields
    )
    (first,) = _session_requests(["first"], 40, 3, token=1)
    (plain,) = create_requests(num_requests=1, num_tokens=30, max_tokens=3)
    plain.sampling_params.ignore_eos = True
    (second,) = _session_requests(["second"], 24, 3, token=3, session="t")
    offsets = {"first": 1, plain.request_id: 1000, "second": 2000}
    for request in (first, plain, second):
        scheduler.add_request(request)
    keys = {}
    while scheduler.has_unfinished_requests():
        scheduler_output = scheduler.schedule()
        worker.begin_step(scheduler_output.aux_output_connector_metadata)
        request_ids = list(scheduler_output.num_scheduled_tokens)
        counts = [scheduler_output.num_scheduled_tokens[r] for r in request_ids]
        ends = [scheduler.requests[r].num_computed_tokens for r in request_ids]
        starts = [end - count for end, count in zip(ends, counts)]
        hidden_states = torch.cat(
            [
                offsets[r] + torch.arange(start, end, dtype=torch.float32)
                for r, start, end in zip(request_ids, starts, ends)
            ]
        )[:, None]
        sampled = [
            [0] if end == scheduler.requests[r].num_tokens else []
            for r, end in zip(request_ids, ends)
        ]
        pending = worker.prepare_output(
            _input_batch(request_ids, np.array(starts), np.cumsum([0, *counts])),
            hidden_states,
            _two_logits,
        )
        aux_outputs = None
        if pending is not None:
            pending.enqueue_cpu_copy(
                np.array([len(tokens) for tokens in sampled]),
                np.zeros(len(request_ids), dtype=np.int64),
            )
            aux_outputs = pending.process_output()
        outputs = scheduler.update_from_output(
            scheduler_output,
            ModelRunnerOutput(
                req_ids=request_ids,
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                aux_output_connector_output=aux_outputs,
            ),
        )
        for client_outputs in outputs.values():
            for output in client_outputs.outputs:
                if output.finish_reason is not None:
                    keys[output.request_id] = output.aux_output_keys
    worker.begin_step(scheduler.schedule().aux_output_connector_metadata)
    worker.close()

    assert keys[plain.request_id] is None
    for request in (first, second):
        rows = offsets[request.request_id] + torch.arange(
            request.num_tokens - 1, dtype=torch.float32
        )
        expected = torch.log_softmax(_two_logits(rows[:, None]), dim=-1).topk(2)
        request_keys = keys[request.request_id]
        np.testing.assert_array_equal(
            read_aux_rows(store, request_keys, "hidden_states")[:, 0], rows.numpy()
        )
        np.testing.assert_array_equal(
            read_aux_rows(store, request_keys, "topk_token_ids"),
            expected.indices.numpy(),
        )
        np.testing.assert_allclose(
            read_aux_rows(store, request_keys, "topk_logprobs"),
            expected.values.numpy(),
        )


class HostTierSpec(OffloadingSpec):
    manager_cls = CPUOffloadingManager

    def get_manager(self):
        return self.manager_cls(num_chunks=self.extra_config["num_chunks"])

    def get_worker(self, kv_caches):
        raise NotImplementedError


_HASH_STORES: defaultdict[str, dict] = defaultdict(dict)


@dataclass
class HashStoreMetadata(KVConnectorMetadata):
    stores: list = field(default_factory=list)
    loads: list = field(default_factory=list)


class HashStoreConnector(KVConnectorBase_V1, SupportsHMA):
    """A KV store keyed by `(KV cache group, block hash)` that every replica
    configured with the same `store` shares; the session engine moves the KV."""

    _cache_hit_source = CacheHitSource.HOST

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.store = _HASH_STORES[extra_config["kv_store"]]
        self.num_groups = len(kv_cache_config.kv_cache_groups)
        self.local = {}
        self.metadata = HashStoreMetadata()

    def _stored(self, request, index):
        return all(
            (group, request.block_hashes[index]) in self.store
            for group in range(self.num_groups)
        )

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        first = num_computed_tokens // 16
        end = first
        while end < (request.num_tokens - 1) // 16 and (
            (0, request.block_hashes[end]) in self.store
        ):
            end += 1
        while end > first and not all(
            self._stored(request, index) for index in range(max(end - 2, 0), end)
        ):
            end -= 1
        self.local[request.request_id] = first
        return (end - first) * 16, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        first = self.local.pop(request.request_id, 0)
        for group, block_ids in enumerate(blocks.get_block_ids()):
            for index in range(first, first + num_external_tokens // 16):
                key = (group, request.block_hashes[index])
                if block_ids[index] and key in self.store:
                    self.metadata.loads.append((key, block_ids[index]))

    def build_connector_meta(self, scheduler_output):
        metadata, self.metadata = self.metadata, HashStoreMetadata()
        return metadata

    def request_finished_all_groups(self, request, block_ids):
        num_blocks = min(request.num_computed_tokens // 16, len(request.block_hashes))
        for group, group_block_ids in enumerate(block_ids):
            for index, block_id in enumerate(group_block_ids[:num_blocks]):
                if block_id:
                    key = (group, request.block_hashes[index])
                    self.metadata.stores.append((key, block_id))
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


@pytest.mark.parametrize(
    "session,segments,expected",
    [("s", False, 40), ("s", True, 8), ("t", True, 40), (None, False, 40)],
    ids=["same-session", "stated", "other-session", "no-session"],
)
def test_session_request_hits_no_blocks_of_another_request(session, segments, expected):
    engine = SessionEngine(create_scheduler(enable_prefix_caching=True))
    _run_turn(engine, 0, "first")
    segments = engine.segments["first"] if segments else None
    (second,) = _session_requests(["second"], 40, 1, session=session, segments=segments)
    engine.scheduler.add_request(second)
    assert engine.schedule().num_scheduled_tokens == {"second": expected}


def _tier_engine(
    spec_name="HostTierSpec",
    num_chunks=8,
    key_prefix="job",
    sliding_window=32,
    retention=0,
    connector="OffloadingConnector",
    store=None,
    block_size=16,
    hash_block_size=16,
    **extra_config,
):
    scheduler = create_scheduler(enable_prefix_caching=True)
    config = scheduler.vllm_config
    config.cache_config.prefix_cache_retention_interval = retention
    config.cache_config.prefix_match_unit = hash_block_size
    module_path = __name__
    if connector == "OffloadingConnector":
        module_path = None
        extra_config = {
            "spec_name": spec_name,
            "spec_module_path": __name__,
            "num_chunks": num_chunks,
            "offload_prompt_only": False,
            **extra_config,
        }
    config.kv_transfer_config = connector and KVTransferConfig(
        kv_connector=connector,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_load_failure_policy="recompute",
        kv_connector_extra_config=extra_config,
    )
    config.aux_output_config = AuxOutputConfig(
        enable_return_routed_experts=True, object_key_prefix=key_prefix
    )
    kv_cache_config = scheduler.kv_cache_config
    if sliding_window is not None:
        spec = AttentionSpec(
            block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        full_spec = AttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
        )
        window = SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
        kv_cache_config = KVCacheConfig(
            num_blocks=100,
            kv_cache_tensors=[],
            kv_cache_groups=[
                new_kv_cache_group(["full"], full_spec),
                new_kv_cache_group(["window"], window),
            ],
            prefix_cache_retention_interval=retention,
        )
    return SessionEngine(
        Scheduler(
            vllm_config=config,
            kv_cache_config=kv_cache_config,
            block_size=block_size,
            hash_block_size=hash_block_size,
            log_stats=True,
            structured_output_manager=scheduler.structured_output_manager,
        ),
        block_size,
        store,
        hash_block_size,
    )


def _run_turn(engine, version, request_id, num_tokens=40, max_tokens=1, **kwargs):
    engine.set_version(version)
    (request,) = _session_requests([request_id], num_tokens, max_tokens, **kwargs)
    engine.scheduler.add_request(request)
    return engine.run()


@pytest.mark.parametrize(
    "max_tokens,end",
    [(1, 32), (8, 32), (9, 48)],
    ids=["partial-block", "block-completed-by-the-last-token", "computed-block"],
)
def test_session_response_segments_end_at_the_last_computed_hash_block(max_tokens, end):
    engine = _tier_engine()
    _run_turn(engine, 1, "first", 40, max_tokens)
    engine.close()
    assert engine.segments["first"] == [(0, "first@v1"), (end, None)]


def test_session_tails_at_other_versions_have_other_keys():
    engine = _tier_engine()
    _run_turn(engine, 1, "first")
    _run_turn(engine, 2, "second", segments=engine.segments["first"])
    engine.close()
    first, second = engine.keys["first"], engine.keys["second"]
    assert first[:2] == second[:2] and first[2] != second[2]


def test_session_turn_computes_again_the_block_hashed_when_its_request_stopped():
    engine = _tier_engine()
    _run_turn(engine, 1, "first", 40, 9, lookahead=1)
    segments = engine.segments["first"]
    _run_turn(engine, 2, "second", 72, segments=segments, lookahead=1)
    engine.close()
    first, second = engine.keys["first"], engine.keys["second"]
    assert segments == [(0, "first@v1"), (32, None)]
    assert second[:2] == first[:2] and "/tail/" in first[2] and second[2] != first[2]


def test_session_objects_returned_before_a_prefix_cache_reset_are_written():
    engine = _tier_engine()
    engine.set_version(1)
    (request,) = _session_requests(["first"], 40, 1)
    engine.scheduler.add_request(request)
    engine.run(
        lambda: (
            "first" in engine.keys
            and engine.scheduler.reset_prefix_cache(reset_connector=False)
        )
    )
    engine.close()
    assert len(engine.keys["first"]) == 3


@pytest.mark.parametrize(
    "connector,reset_connector",
    [(None, False), ("OffloadingConnector", True)],
    ids=["no-connector", "connector-reset"],
)
def test_session_request_running_at_a_prefix_cache_reset_resumes(
    connector, reset_connector
):
    engine = _tier_engine(connector=connector)
    engine.set_version(1)
    (request,) = _session_requests(["r"], 40, 6)
    engine.scheduler.add_request(request)
    for _ in range(2):
        engine.step(engine.schedule(), {"r": [0]})
    assert engine.scheduler.reset_prefix_cache(
        reset_running_requests=True, reset_connector=reset_connector
    )
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (32, None)]


def test_session_request_resumed_after_a_reset_sends_hashes_from_its_hit():
    engine = _tier_engine()
    engine.set_version(1)
    (request,) = _session_requests(["r"], 40, 12)
    engine.scheduler.add_request(request)
    for _ in range(6):
        engine.step(engine.schedule(), {"r": [0]})
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = engine.schedule()
    while "r" not in resumed.num_scheduled_tokens:
        engine.step(resumed, {})
        resumed = engine.schedule()
    num_hit = request.num_computed_tokens - resumed.num_scheduled_tokens["r"]
    metadata = resumed.aux_output_connector_metadata
    sent = metadata.block_hashes.get("r", ())
    assert num_hit > 0
    assert metadata.session_hash_starts == {"r": num_hit // 16}
    assert (
        list(sent)
        == request.block_hashes[num_hit // 16 : request.num_computed_tokens // 16]
    )
    engine.step(resumed, {"r": [0]})
    engine.run()
    engine.close()


@pytest.mark.parametrize("fields", [False, True], ids=["rows", "fields"])
def test_reset_writes_the_objects_its_preempted_session_request_completed(fields):
    engine = _tier_engine(connector=None)
    engine.worker._return_hidden_states = fields
    engine.worker._topk_logprobs = 2 if fields else 0
    engine.set_version(1)
    (request,) = _session_requests(["r"], 40, 12, lookahead=1)
    engine.scheduler.add_request(request)
    for _ in range(9):
        engine.step(engine.schedule(), {"r": [0]})
    key = f"job/s/{request.block_hashes[2].hex()}"
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = engine.schedule()
    rows = read_aux_rows(engine.store, [key], "routed_experts", timeout_s=5)
    assert rows[:, 0, 0].tolist() == list(range(32, 48))
    if fields:
        hidden = read_aux_rows(engine.store, [key], "hidden_states", timeout_s=5)
        assert hidden[:, 0].tolist() == list(range(32, 48))
        expected = torch.log_softmax(_two_logits(torch.from_numpy(hidden)), dim=-1)
        expected = expected.topk(2)
        np.testing.assert_array_equal(
            read_aux_rows(engine.store, [key], "topk_token_ids", timeout_s=5),
            expected.indices.numpy(),
        )
        np.testing.assert_allclose(
            read_aux_rows(engine.store, [key], "topk_logprobs", timeout_s=5),
            expected.values.numpy(),
        )
    engine.step(resumed, {"r": [0]})
    engine.run()
    engine.close()


def test_released_samples_leave_no_objects():
    engine = _tier_engine(connector=None)
    _run_turn(engine, 1, "turn1", 48, lookahead=1)
    _run_turn(engine, 2, "turn2", 80, segments=engine.segments["turn1"])
    _run_turn(engine, 2, "parent", 40, token=1)
    for child in ("child1", "child2"):
        _run_turn(engine, 3, child, 72, token=1, segments=engine.segments["parent"])
    engine.close()
    sample = AuxSampleKeys()
    sample.add(engine.keys["turn1"])
    sample.add(engine.keys["turn2"])
    group = AuxSampleKeys()
    group.add(engine.keys["parent"])
    children = []
    for child in ("child1", "child2"):
        keys = AuxSampleKeys(group)
        keys.add(engine.keys[child])
        children.append(keys)
    assert set(engine.keys["parent"][:2]) <= set(engine.keys["child1"])
    owned = [sample.owned, group.owned] + [keys.owned for keys in children]
    assert (
        sum(map(len, owned)) == len(set().union(*owned)) == len(engine.client.objects)
    )
    for keys in [sample.owned] + [keys.owned for keys in children]:
        assert release_aux_keys(engine.store, keys, timeout_s=0) == []
    assert set(engine.client.objects) == set(group.owned)
    assert release_aux_keys(engine.store, group.owned, timeout_s=0) == []
    assert not engine.client.objects


@pytest.mark.parametrize("max_tokens", [1, 9], ids=["tail", "no-tail"])
def test_session_responses_the_frontend_dropped_leave_no_objects(max_tokens):
    engine = _tier_engine()
    _run_turn(engine, 1, "first")
    outputs = _run_turn(
        engine, 2, "second", 72, max_tokens, segments=engine.segments["first"]
    )
    engine._stopped = [entry for entry in engine._stopped if entry[0] != "second"]
    processed = OutputProcessor(None, log_stats=False).process_outputs(
        [outputs["second"]]
    )
    assert not processed.request_outputs
    engine.connector.release_unreturned(processed.unreturned_aux_keys)
    engine.close()
    assert set(engine.keys["first"][:2]) < set(engine.keys["second"])
    assert set(engine.client.objects) == set(engine.keys["first"])


def test_pause_writes_every_object_of_the_stopped_session_requests():
    engine = _tier_engine(connector=None)
    put_batch = engine.client.put_batch
    engine.client.put_batch = lambda *args: time.sleep(0.05) or put_batch(*args)
    _run_turn(engine, 1, "first", 40, 9)
    (repeat,) = _session_requests(["repeat"], 40, 4, prompt_logprobs=1)
    engine.scheduler.add_request(repeat)
    engine.step(engine.schedule(), {"repeat": [0]})
    engine.scheduler.finish_requests("repeat", RequestStatus.FINISHED_ABORTED)
    engine.pause()
    assert set(engine.keys["first"]) <= set(engine.client.objects)
    engine.client.remove_by_regex(aux_session_pattern("job", "s"), True)
    _run_turn(engine, 1, "other", 40, session="s2")
    engine.worker.close()
    assert engine.client.objects
    assert all(key.startswith("job/s2/") for key in engine.client.objects)


def test_engine_core_pause_writes_the_objects_of_the_requests_it_preempts():
    engine = SessionEngine(
        create_scheduler(async_scheduling=True, enable_prefix_caching=True)
    )
    put_batch = engine.client.put_batch
    engine.client.put_batch = lambda *args: time.sleep(0.05) or put_batch(*args)
    (request,) = _session_requests(["r"], 47, 8)
    engine.scheduler.add_request(request)
    prompt, decode = engine.schedule(), engine.schedule()
    engine.step(prompt, {"r": [0]})
    engine.step(decode, {"r": [0]})
    worker = object.__new__(WorkerBase)
    worker.model_runner = SimpleNamespace(aux_output_connector=engine.worker)
    core = object.__new__(EngineCore)
    core.scheduler = engine.scheduler
    core.model_executor = Mock()
    core.mm_receiver_cache = None
    core._kv_cache_reclaimable = False
    core._checkpoint_state = CheckpointState.RUNNING
    core.collective_rpc = lambda method, args=(): [getattr(worker, method)(*args)]
    core.pause_scheduler(mode="keep")
    assert set(engine.client.objects) == {
        f"job/s/{block_hash.hex()}" for block_hash in request.block_hashes[:3]
    }
    engine.worker.close()


def test_session_request_preempted_by_a_failed_pause_reset_resumes():
    engine = _tier_engine(connector=None)
    engine.set_version(1)
    (request,) = _session_requests(["r"], 40, 6)
    engine.scheduler.add_request(request)
    for _ in range(2):
        engine.step(engine.schedule(), {"r": [0]})
    block_pool = engine.scheduler.kv_cache_manager.block_pool
    held = block_pool.get_new_blocks(1)
    core = object.__new__(EngineCore)
    core.scheduler = engine.scheduler
    core.model_executor = Mock()
    core._checkpoint_state = CheckpointState.RUNNING
    core._kv_cache_reclaimable = False
    with pytest.raises(RuntimeError, match="Failed to reset KV cache"):
        core.pause_scheduler(mode="keep")
    assert request.status == RequestStatus.PREEMPTED
    block_pool.free_blocks(held)
    core.resume_scheduler()
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (32, None)]


def test_session_key_of_a_block_whose_hash_follows_the_stop_is_a_tail():
    engine = _tier_engine(connector=None)
    _run_turn(engine, 0, "first", 48, lookahead=1)
    _run_turn(engine, 0, "second", 72, segments=engine.segments["first"], lookahead=1)
    engine.close()
    first, second = engine.keys["first"], engine.keys["second"]
    assert len(first) == 3 and first[:2] == second[:2] and "/tail/" in first[2]
    assert second[2] not in first and engine.client.removes


def _preempt_at_an_object_boundary(engine, versions):
    scheduler = engine.scheduler
    engine.set_version(1)
    (request,) = _session_requests(["r"], 16, 20, hash_block_size=4)
    scheduler.add_request(request)
    for version in versions:
        engine.set_version(version)
        engine.step(engine.schedule(), {"r": [0]})
    hashes = list(request.block_hashes)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    scheduler.waiting.remove_request(request)
    engine.step(engine.schedule(), {})
    scheduler.waiting.prepend_request(request)
    return request, hashes


@pytest.mark.parametrize(
    "async_scheduling,block_size,chunk",
    [
        (False, 16, 24),
        (True, 16, 24),
        (False, 32, 16),
        (False, 32, 24),
        (False, 16, 40),
    ],
    ids=[
        "chunk-inside-a-block",
        "async",
        "object-of-two-hash-blocks",
        "chunk-inside-a-hash-block-of-an-object",
        "chunk-of-two-hash-blocks",
    ],
)
def test_weight_update_names_the_rest_of_a_running_prefill_by_the_new_version(
    async_scheduling, block_size, chunk
):
    if block_size == 16:
        engine = SessionEngine(
            create_scheduler(
                async_scheduling=async_scheduling, enable_prefix_caching=True
            )
        )
    else:
        engine = _tier_engine(connector=None, block_size=block_size)
    engine.scheduler.max_num_scheduled_tokens = chunk
    engine.set_version(1)
    (request,) = _session_requests(["r"], 72, 1)
    engine.scheduler.add_request(request)
    scheduler_output = engine.schedule()
    sent = scheduler_output.aux_output_connector_metadata.block_hashes["r"]
    assert list(sent) == [bytes(h) for h in request.block_hashes[: chunk // 16]]
    engine.step(scheduler_output, {})
    assert request.num_computed_tokens == chunk
    engine.set_version(2)
    engine.run()
    segments = [list(segment) for segment in engine.segments["r"]]
    (sibling,) = _session_requests(["s2"], 72, 1, segments=segments)
    engine.scheduler.add_request(sibling)
    scheduler_output = engine.schedule()
    metadata = scheduler_output.aux_output_connector_metadata
    assert metadata.session_hash_starts == {"s2": 4}
    assert scheduler_output.num_scheduled_tokens == {"s2": 8}
    engine.step(scheduler_output, {"s2": [0]})
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (chunk, "r@v2"), (64, None)]
    assert engine.segments["s2"] == engine.segments["r"]


def test_weight_updates_between_chunks_name_each_chunk_by_its_version():
    engine = SessionEngine(create_scheduler(enable_prefix_caching=True))
    engine.scheduler.max_num_scheduled_tokens = 24
    engine.set_version(1)
    (request,) = _session_requests(["r"], 72, 1)
    engine.scheduler.add_request(request)
    for version in (2, 3):
        engine.step(engine.schedule(), {})
        engine.set_version(version)
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (24, "r@v2"), (48, "r@v3"), (64, None)]
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=request.sampling_params,
        pooling_params=None,
        cache_salt=request.cache_salt,
        cache_salt_segments=[(0, "r@v1"), (24, "r@v2"), (48, "r@v3")],
        block_hasher=get_request_block_hasher(16, sha256),
    )
    assert fresh.block_hashes == request.block_hashes


def test_keep_pause_and_weight_update_through_the_engine_core_rename_a_prefill():
    engine = SessionEngine(create_scheduler(enable_prefix_caching=True))
    engine.scheduler.max_num_scheduled_tokens = 24
    core = object.__new__(EngineCore)
    core.scheduler = engine.scheduler
    core.model_executor = Mock()
    core._checkpoint_state = CheckpointState.RUNNING
    core._kv_cache_reclaimable = False
    core.collective_rpc = lambda method, args=(): engine.worker.drain(*args)
    engine.version = 1
    core.set_weight_version("v1")
    (request,) = _session_requests(["r"], 72, 1)
    engine.scheduler.add_request(request)
    engine.step(engine.schedule(), {})
    core.pause_scheduler(mode="keep", clear_cache=False)
    engine.step(engine.schedule(), {})
    engine.version = 2
    core.set_weight_version("v2")
    core.resume_scheduler()
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (24, "r@v2"), (64, None)]


def test_weight_update_names_a_block_after_rejected_drafts_by_the_new_version():
    scheduler = create_scheduler(
        async_scheduling=True,
        enable_prefix_caching=True,
        num_speculative_tokens=2,
        speculative_method="ngram_gpu",
    )
    engine = SessionEngine(scheduler)
    engine.set_version(1)
    (request,) = _session_requests(["r"], 28, 8)
    scheduler.add_request(request)
    engine.step(engine.schedule(), {"r": [1]})
    engine.step(engine.schedule(), {"r": [2, 3]})
    engine.step(engine.schedule(), {"r": [4]})
    computed = request.num_computed_tokens
    assert len(request.block_hashes) * 16 > computed
    engine.set_version(2)
    while not request.is_finished():
        engine.step(engine.schedule(), {"r": [0]})
    engine.close()
    assert engine.segments["r"][:2] == [(0, "r@v1"), (computed, "r@v2")]


def test_preempted_session_request_salts_the_object_it_computes_again():
    engine = _tier_engine(connector=None, hash_block_size=4, retention=16)
    request, _ = _preempt_at_an_object_boundary(engine, (1, 1, 1) + (2,) * 13)
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v1"), (16, "r@v2"), (32, None)]
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=request.sampling_params,
        pooling_params=None,
        cache_salt=request.cache_salt,
        cache_salt_segments=[(0, "r@v1"), (16, "r@v2")],
        block_hasher=get_request_block_hasher(4, sha256),
    )
    assert fresh.block_hashes == request.block_hashes


def test_session_request_preempted_by_a_reset_takes_the_next_version():
    engine = _tier_engine(hash_block_size=4, retention=16)
    engine.set_version(1)
    (request,) = _session_requests(["r"], 16, 20, hash_block_size=4)
    engine.scheduler.add_request(request)
    for _ in range(4):
        engine.step(engine.schedule(), {"r": [0]})
    assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    engine.set_version(2)
    engine.run()
    engine.close()
    assert engine.num_loads > 0
    assert engine.segments["r"] == [(0, "r@v1"), (16, "r@v2"), (32, None)]


def test_preempted_session_request_keeps_its_hashes_at_the_same_version():
    engine = _tier_engine(connector=None, hash_block_size=4, retention=16)
    request, hashes = _preempt_at_an_object_boundary(engine, (1,) * 16)
    engine.run()
    engine.close()
    resumed = request.block_hashes[: len(hashes)]
    assert all(a is b for a, b in zip(hashes, resumed, strict=True))


def test_session_request_without_the_kv_of_its_segments_names_them_again():
    engine = _tier_engine(connector=None)
    engine.set_version(2)
    (request,) = _session_requests(
        ["r"], 72, 1, segments=[(0, "v2"), (32, "v1"), (64, None)]
    )
    engine.scheduler.add_request(request)
    hashes = list(request.block_hashes)
    engine.run()
    engine.close()
    assert not set(hashes) & set(request.block_hashes)
    assert engine.segments["r"] == [(0, "r@v2"), (64, None)]


def test_preempted_session_request_offloads_under_its_new_hashes():
    engine = _tier_engine(hash_block_size=4)
    engine.scheduler.max_num_scheduled_tokens = 16
    manager = engine.scheduler.connector.connector_scheduler.manager
    stored = []
    prepare_store = manager.prepare_store

    def record(keys, *args, **kwargs):
        stored.extend(keys)
        return prepare_store(keys, *args, **kwargs)

    manager.prepare_store = record
    engine.set_version(1)
    (request,) = _session_requests(["r"], 48, 4, hash_block_size=4)
    engine.scheduler.add_request(request)
    engine.step(engine.schedule(), {})
    engine.set_version(2)
    engine.scheduler.running.remove(request)
    engine.scheduler._preempt_request(request, 0.0)
    engine.run()
    engine.close()
    assert stored
    assert {get_offload_block_hash(key) for key in stored} <= set(request.block_hashes)


@pytest.mark.parametrize("version", ["", None], ids=["empty", "none"])
@pytest.mark.parametrize("live", [False, True], ids=["idle", "live-session"])
def test_rejected_weight_version_keeps_the_current_one(live, version):
    engine = _tier_engine(connector=None)
    engine.set_version(1)
    (request,) = _session_requests(["r"], 40, 4)
    if live:
        engine.scheduler.add_request(request)
        engine.step(engine.schedule(), {"r": [0]})
    segments = list(request.cache_salt_segments)
    with pytest.raises(VLLMValidationError):
        engine.connector.set_weight_version(version)
    assert request.cache_salt_segments == segments
    (other,) = _session_requests(["other"], 40, 4)
    engine.scheduler.add_request(other)
    assert other.cache_salt_segments == [(0, "other@v1")]
    engine.close()


def test_session_response_segments_end_at_positions_no_step_in_flight_computes():
    scheduler = create_scheduler(async_scheduling=True, enable_prefix_caching=True)
    engine = SessionEngine(scheduler)
    (request,) = _session_requests(["r"], 46, 10, stop_token_ids=[7])
    scheduler.add_request(request)
    prefill = engine.schedule()
    first_decode = engine.schedule()
    engine.step(prefill, {"r": [1]})
    second_decode = engine.schedule()
    assert second_decode.num_scheduled_tokens == {"r": 1}
    outputs = engine.step(first_decode, {"r": [7]})
    engine.step(second_decode, {"r": [1]})
    engine.close()
    assert outputs["r"].aux_version_segments == [(0, "r@v0"), (32, None)]


def test_session_external_hit_inside_a_kv_block_is_refused():
    engine = _tier_engine()
    (request,) = _session_requests(["r"], 40, 1)
    request.sampling_params.routed_experts_prompt_start = 40
    assert engine.connector.cap_external_hit(request, 0, 8) == 0
    assert engine.connector.cap_external_hit(request, 0, 16) == 16


def test_session_segments_end_on_objects_when_hash_blocks_are_smaller():
    engine = _tier_engine(connector=None, hash_block_size=4, retention=16)
    engine.set_version(1)
    (first,) = _session_requests(["first"], 40, 14, hash_block_size=4)
    engine.scheduler.add_request(first)
    for _ in range(2):
        engine.step(engine.schedule(), {"first": [0]})
    engine.set_version(2)
    engine.run()
    segments = engine.segments["first"]
    _run_turn(engine, 3, "second", 72, segments=segments, hash_block_size=4)
    engine.close()
    second = engine.requests["second"]
    assert segments == [(0, "first@v1"), (41, "first@v2"), (48, None)]
    assert second.cache_salt_segments == [
        (0, "first@v1"),
        (41, "first@v2"),
        (48, "second@v3"),
    ]
    assert engine.keys["second"][:3] == engine.keys["first"][:3]
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=list(second.all_token_ids),
        sampling_params=second.sampling_params,
        pooling_params=None,
        cache_salt=second.cache_salt,
        cache_salt_segments=[(0, "first@v1"), (41, "first@v2"), (48, "second@v3")],
        block_hasher=get_request_block_hasher(4, sha256),
    )
    assert fresh.block_hashes == second.block_hashes


def test_session_admission_rounds_client_segments_down_to_an_object():
    engine = _tier_engine(connector=None, hash_block_size=4)
    engine.set_version(2)
    (request,) = _session_requests(
        ["r"], 46, 1, segments=[(0, "v1"), (48, None)], hash_block_size=4
    )
    engine.scheduler.add_request(request)
    assert request.cache_salt_segments == [(0, "v1"), (32, "r@v2")]
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=request.sampling_params,
        pooling_params=None,
        cache_salt=request.cache_salt,
        cache_salt_segments=[(0, "v1"), (32, "r@v2")],
        block_hasher=get_request_block_hasher(4, sha256),
    )
    assert fresh.block_hashes == request.block_hashes
    engine.run()
    engine.close()
    assert engine.segments["r"] == [(0, "r@v2"), (32, None)]


def test_session_object_of_two_hash_blocks_is_named_by_the_last():
    engine = _tier_engine(block_size=32)
    _run_turn(engine, 1, "first", 72)
    engine.close()
    assert len(engine.keys["first"]) == 3


@pytest.mark.parametrize(
    "with_segments,reset_connector,expected_loads",
    [(True, False, 1), (False, False, 0), (True, True, 0)],
    ids=["segments", "no-segments", "tier-reset"],
)
def test_session_turn_reuses_the_offload_tier_across_a_weight_update(
    with_segments, reset_connector, expected_loads
):
    engine = _tier_engine(num_chunks=64)
    _run_turn(engine, 1, "first")
    assert engine.segments["first"] == [(0, "first@v1"), (32, None)]
    assert engine.scheduler.reset_prefix_cache(reset_connector=reset_connector)
    segments = engine.segments["first"] if with_segments else None
    _run_turn(engine, 2, "second", 72, segments=segments)
    engine.close()
    reused = with_segments and not reset_connector
    assert engine.num_loads == expected_loads and engine.num_stores > 0
    assert (engine.keys["second"][:2] == engine.keys["first"][:2]) == reused
    expected = [(0, "first@v1"), (32, "second@v2")] if reused else [(0, "second@v2")]
    assert engine.segments["second"] == [*expected, (64, None)]


def test_session_turn_names_the_block_eagle_computes_again_by_its_version():
    engine = _tier_engine(connector=None, sliding_window=None)
    engine.scheduler.kv_cache_manager.coordinator.drop_eagle_group_ids = {0}
    _run_turn(engine, 1, "first", 40, 9)
    _run_turn(engine, 2, "second", 72, segments=engine.segments["first"])
    engine.close()
    assert engine.segments["first"] == [(0, "first@v1"), (48, None)]
    assert engine.segments["second"] == [(0, "first@v1"), (32, "second@v2"), (64, None)]


def test_session_turn_with_prompt_logprobs_names_its_prompt_by_its_version():
    engine = _tier_engine()
    _run_turn(engine, 1, "first")
    segments = engine.segments["first"]
    _run_turn(engine, 2, "second", 72, segments=segments, prompt_logprobs=1)
    engine.close()
    assert engine.segments["second"] == [(0, "second@v2"), (64, None)]


def test_session_request_loading_kv_across_a_weight_update_names_it_by_the_new():
    engine = _tier_engine(num_chunks=64, hash_block_size=4)
    _run_turn(engine, 1, "first", 24, hash_block_size=4)
    assert engine.segments["first"] == [(0, "first@v1"), (16, None)]
    assert engine.scheduler.reset_prefix_cache()
    (request,) = _session_requests(
        ["second"], 64, 1, segments=engine.segments["first"], hash_block_size=4
    )
    engine.scheduler.add_request(request)
    engine.step(engine.schedule(), {})
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    engine.set_version(2)
    engine.run()
    engine.close()
    assert engine.num_loads == 1
    assert engine.segments["second"] == [(0, "first@v1"), (16, "second@v2"), (64, None)]


@pytest.mark.parametrize("with_segments", [True, False], ids=["segments", "none"])
def test_session_turn_reuses_a_hash_keyed_store_of_another_replica(with_segments):
    store = fake_store()
    name = f"replicas-{with_segments}"
    engines = [
        _tier_engine(connector="HashStoreConnector", store=store, kv_store=name)
        for _ in range(2)
    ]
    _run_turn(engines[0], 1, "first")
    segments = engines[0].segments["first"] if with_segments else None
    for index in (1, 0):
        _run_turn(engines[index], 2, f"turn{index}", 72, segments=segments)
    for engine in engines:
        engine.close(check_leaks=False)
    first, second = engines
    assert store[0].objects.keys() <= first.listed() | second.listed()
    assert second.num_loads > 0 if with_segments else second.num_loads == 0
    assert first.num_loads == 0
    turns = first.keys["turn0"], second.keys["turn1"]
    assert all((keys[:2] == first.keys["first"][:2]) == with_segments for keys in turns)
    assert not set(turns[0][2:]) & set(turns[1][2:])


@pytest.mark.parametrize(
    "segments,num_tokens,expected",
    [
        (None, 40, [(0, "r@v2")]),
        ([(0, "a"), (32, None)], 40, [(0, "a"), (32, "r@v2")]),
        ([(0, "a"), (40, None)], 72, [(0, "a"), (32, "r@v2")]),
        ([(0, "a"), (64, None)], 40, [(0, "a"), (32, "r@v2")]),
        ([(0, "a"), (32, None)], 32, [(0, "a"), (16, "r@v2")]),
        ([(0, "a"), (16, "b"), (48, None)], 72, [(0, "a"), (16, "b"), (48, "r@v2")]),
    ],
    ids=[
        "none",
        "whole-blocks",
        "rounded-down",
        "clipped-to-prompt",
        "covered-to-the-prompt-end",
        "two-segments",
    ],
)
def test_session_request_salts_its_segments_then_its_own(
    segments, num_tokens, expected
):
    engine = _tier_engine()
    engine.set_version(2)
    (request,) = _session_requests(["r"], num_tokens, 1, segments=segments)
    engine.scheduler.add_request(request)
    assert request.cache_salt_segments == expected
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=[0] * num_tokens,
        sampling_params=request.sampling_params,
        pooling_params=None,
        cache_salt=request.cache_salt,
        cache_salt_segments=expected,
        block_hasher=get_request_block_hasher(16, sha256),
    )
    assert request.block_hashes == fresh.block_hashes


@pytest.mark.parametrize(
    "num_tokens,num_steps,preempt,later_versions,expected",
    [
        (40, 0, False, (), [(0, "r@v2"), (48, None)]),
        (40, 2, False, (), [(0, "r@v1"), (41, "r@v2"), (48, None)]),
        (47, 1, False, (), [(0, "r@v1"), (47, "r@v2"), (48, None)]),
        (40, 2, True, (), [(0, "r@v1"), (32, "r@v2"), (48, None)]),
        (40, 2, True, (3,), [(0, "r@v1"), (32, "r@v3"), (48, None)]),
        (40, 2, True, (1,), [(0, "r@v1"), (48, None)]),
        (47, 1, False, (3,), [(0, "r@v1"), (47, "r@v3"), (48, None)]),
        (47, 1, False, (1,), [(0, "r@v1"), (48, None)]),
    ],
    ids=[
        "waiting",
        "decoding",
        "sampled-token-completes-a-block",
        "preempted",
        "updated-after-preemption",
        "updated-back-after-preemption",
        "updated-twice-without-a-step-between",
        "updated-back-while-running",
    ],
)
def test_weight_update_salts_what_a_session_request_has_not_computed(
    num_tokens, num_steps, preempt, later_versions, expected
):
    engine = _tier_engine()
    engine.set_version(1)
    (request,) = _session_requests(["r"], num_tokens, 10)
    engine.scheduler.add_request(request)
    for _ in range(num_steps):
        engine.step(engine.schedule(), {"r": [0]})
    engine.set_version(2)
    if preempt:
        assert engine.scheduler.reset_prefix_cache(reset_running_requests=True)
    for version in later_versions:
        engine.set_version(version)
    engine.run()
    engine.close()
    assert engine.segments["r"] == expected
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=list(request.all_token_ids),
        sampling_params=request.sampling_params,
        pooling_params=None,
        cache_salt=request.cache_salt,
        cache_salt_segments=list(request.cache_salt_segments),
        block_hasher=get_request_block_hasher(16, sha256),
    )
    assert fresh.block_hashes == request.block_hashes
