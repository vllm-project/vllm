# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy M1: the SchedulerOutput carries what a worker needs to detect drift.

A reference worker replays the deltas exactly as a correct worker does
(drop finished/preempted, add new rows, then per cached request apply
revokes before appends) and must reproduce `expected_row_len` and
`row_digest` on every step; its rows must equal the scheduler's own rows.
"""

import pickle
import random
from collections import deque

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.grant_table import NULL_BLOCK_ID, NULL_GEN, row_digest
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


class RefWorker:
    """A worker's view of the block tables, built only from the messages."""

    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[list[int], list[int]]]] = {}
        self.msg_seq = 0
        self.num_revokes = 0

    def _check(self, req_id: str, lens, digests) -> None:
        rows = self.rows[req_id]
        assert len(lens) == len(digests) == len(rows)
        for (ids, gens), length, digest in zip(rows, lens, digests):
            assert len(ids) == length
            assert row_digest(range(len(ids)), ids, gens) == digest

    def apply(self, so: SchedulerOutput) -> None:
        assert so.msg_seq == self.msg_seq + 1, "msg_seq must be gap-free"
        self.msg_seq = so.msg_seq
        for req_id in so.finished_req_ids | (so.preempted_req_ids or set()):
            self.rows.pop(req_id, None)
        for new in so.scheduled_new_reqs:
            self.rows[new.req_id] = [
                (list(ids), list(gens))
                for ids, gens in zip(new.block_ids, new.block_gens, strict=True)
            ]
            self._check(new.req_id, new.expected_row_len, new.row_digest)
        cached = so.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            rows = self.rows[req_id]
            for group, col in cached.revoked[i] or ():
                ids, gens = rows[group]
                ids[col] = NULL_BLOCK_ID
                gens[col] = NULL_GEN
                self.num_revokes += 1
            if cached.new_block_ids[i] is not None:
                for (ids, gens), new_ids, new_gens in zip(
                    rows, cached.new_block_ids[i], cached.new_block_gens[i]
                ):
                    assert len(new_ids) == len(new_gens)
                    ids.extend(new_ids)
                    gens.extend(new_gens)
            self._check(req_id, cached.expected_row_len[i], cached.row_digest[i])


def _check_against_scheduler(scheduler, worker: RefWorker, so: SchedulerOutput):
    """Scheduled rows equal the scheduler's rows, carry current gens, and the
    write windows cover exactly the written columns with current gens."""
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    grant_table = scheduler.kv_cache_manager.grant_table
    builder = scheduler.percy_messages
    num_computed = {n.req_id: n.num_computed_tokens for n in so.scheduled_new_reqs}
    windows = {n.req_id: n.write_window for n in so.scheduled_new_reqs}
    cached = so.scheduled_cached_reqs
    num_computed.update(zip(cached.req_ids, cached.num_computed_tokens))
    windows.update(zip(cached.req_ids, cached.write_windows))
    for req_id, num_tokens in so.num_scheduled_tokens.items():
        for group, manager in enumerate(managers):
            ids, gens = worker.rows[req_id][group]
            assert ids == [b.block_id for b in manager.req_to_blocks[req_id]]
            assert gens == grant_table.gens_of(ids).tolist()
            window = windows[req_id][group]
            start = max(0, num_computed[req_id] - builder.num_position_slack)
            end = num_computed[req_id] + num_tokens + builder.num_lookahead_tokens
            first = start // manager.block_size
            last = min(len(ids), cdiv(end, manager.block_size))
            assert window[0] == first
            expected = [0 if b == NULL_BLOCK_ID else g for b, g in zip(ids, gens)]
            assert list(window[1:]) == expected[first:last]
            # Every column the scheduled tokens write is inside the window.
            written_last = (num_computed[req_id] + num_tokens - 1) // manager.block_size
            assert first <= num_computed[req_id] // manager.block_size
            assert written_last < first + len(window) - 1
    if so.new_block_ids_to_zero:
        assert so.new_block_gens_to_zero == (
            grant_table.gens_of(so.new_block_ids_to_zero).tolist()
        )
    if so.kv_cache_block_copies:
        assert len(so.kv_cache_block_copy_gens) == len(so.kv_cache_block_copies)


def _model_output(so: SchedulerOutput, rng: random.Random) -> ModelRunnerOutput:
    """One sampled token per request, plus a random number of accepted drafts."""
    req_ids = list(so.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        num_drafts = len(so.scheduled_spec_decode_tokens.get(req_id, ()))
        sampled.append([1] * (1 + rng.randint(0, num_drafts)))
    return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=sampled)


def _drive(
    scheduler,
    arrivals: dict[int, list],
    num_steps: int,
    *,
    depth: int = 1,
    num_drafts: int = 0,
    seed: int = 0,
) -> RefWorker:
    """Run the scheduler like the engine (a batch queue of `depth`), feeding
    every dispatched output to a reference worker and checking it."""
    rng = random.Random(seed)
    worker = RefWorker()
    in_flight: deque[SchedulerOutput] = deque()
    for step in range(num_steps):
        for request in arrivals.get(step, ()):
            scheduler.add_request(request)
        if scheduler.has_requests():
            so = scheduler.schedule()
            worker.apply(so)
            _check_against_scheduler(scheduler, worker, so)
            in_flight.append(so)
        while in_flight and (len(in_flight) >= depth or not scheduler.has_requests()):
            done = in_flight.popleft()
            out = _model_output(done, rng)
            scheduler.update_from_output(done, out)
            if num_drafts and depth == 1:
                running = [r.request_id for r in scheduler.running]
                scheduler.update_draft_token_ids(
                    DraftTokenIds(running, [[2] * num_drafts for _ in running])
                )
    return worker


def test_prefill_chunks_decode_and_prefix_hits():
    scheduler = create_scheduler(
        enable_prefix_caching=True, max_num_batched_tokens=48, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        4, num_tokens=100, max_tokens=12, same_prompt=True, block_size=BLOCK_SIZE
    )
    hits: list[int] = []
    orig_schedule = scheduler.schedule

    def schedule(*args, **kwargs):
        so = orig_schedule(*args, **kwargs)
        hits.extend(n.num_computed_tokens for n in so.scheduled_new_reqs)
        return so

    scheduler.schedule = schedule
    worker = _drive(scheduler, {0: reqs[:1], 6: reqs[1:]}, num_steps=40)
    assert worker.msg_seq > 10
    # The later requests hit the first one's published prefix.
    assert max(hits[1:]) >= 3 * BLOCK_SIZE


def test_preemption_and_resume():
    scheduler = create_scheduler(num_blocks=14, block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=60, max_tokens=40, block_size=BLOCK_SIZE)
    preempted: set[str] = set()
    orig_schedule = scheduler.schedule

    def schedule(*args, **kwargs):
        so = orig_schedule(*args, **kwargs)
        preempted.update(so.preempted_req_ids or ())
        return so

    scheduler.schedule = schedule
    _drive(scheduler, {0: reqs}, num_steps=80)
    assert preempted, "the scenario must preempt and resume"
    assert all(r.is_finished() for r in reqs)


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_sliding_window_column_frees_are_revoked(async_scheduling: bool):
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_layer_config=SlidingWindowManager.layer_config(
            AttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
            sliding_window=BLOCK_SIZE,
        ),
    )
    reqs = create_requests(2, num_tokens=100, max_tokens=40, block_size=BLOCK_SIZE)
    worker = _drive(
        scheduler, {0: reqs}, num_steps=60, depth=2 if async_scheduling else 1
    )
    assert worker.num_revokes >= 2 * (100 // BLOCK_SIZE - 1)


def test_spec_decode_with_rejections():
    scheduler = create_scheduler(num_speculative_tokens=3, block_size=BLOCK_SIZE)
    reqs = create_requests(3, num_tokens=30, max_tokens=60, block_size=BLOCK_SIZE)
    _drive(scheduler, {0: reqs}, num_steps=60, num_drafts=3, seed=1)
    assert all(r.is_finished() for r in reqs)


def test_async_scheduling():
    scheduler = create_scheduler(
        async_scheduling=True, enable_prefix_caching=True, block_size=BLOCK_SIZE
    )
    reqs = create_requests(
        3, num_tokens=70, max_tokens=20, same_prompt=True, block_size=BLOCK_SIZE
    )
    _drive(scheduler, {0: reqs[:1], 3: reqs[1:]}, num_steps=60, depth=2)
    assert all(r.is_finished() for r in reqs)


def test_async_spec_decode_window_slack():
    scheduler = create_scheduler(
        async_scheduling=True, num_speculative_tokens=2, block_size=BLOCK_SIZE
    )
    assert scheduler.percy_messages.num_position_slack == 2
    reqs = create_requests(2, num_tokens=20, max_tokens=30, block_size=BLOCK_SIZE)
    _drive(scheduler, {0: reqs}, num_steps=60, depth=2, seed=2)


def test_each_row_revoke_appears_in_next_output():
    """A sliding-window column freed during schedule() is revoked for that
    request in the same output, exactly once."""
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        kv_cache_layer_config=SlidingWindowManager.layer_config(
            AttentionSpec(
                block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
            sliding_window=BLOCK_SIZE,
        ),
    )
    (request,) = create_requests(1, num_tokens=100, max_tokens=4)
    scheduler.add_request(request)
    rng = random.Random(0)
    first = scheduler.schedule()
    scheduler.update_from_output(first, _model_output(first, rng))
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    before = [b.block_id for b in manager.req_to_blocks[request.request_id]]
    second = scheduler.schedule()
    after = [b.block_id for b in manager.req_to_blocks[request.request_id]]
    nulled = [c for c, (b, a) in enumerate(zip(before, after)) if b != a]
    assert nulled and all(after[c] == NULL_BLOCK_ID for c in nulled)
    (revoked,) = second.scheduled_cached_reqs.revoked
    assert sorted(col for _, col in revoked) == nulled
    scheduler.update_from_output(second, _model_output(second, rng))
    assert scheduler.schedule().scheduled_cached_reqs.revoked == [None]


def test_message_size_per_decode_request():
    """Percy fields cost a few bytes per decode request per group."""
    num_reqs = 64
    scheduler = create_scheduler(max_num_seqs=num_reqs, block_size=BLOCK_SIZE)
    reqs = create_requests(num_reqs, num_tokens=40, max_tokens=100)
    for r in reqs:
        scheduler.add_request(r)
    rng = random.Random(0)
    so = scheduler.schedule()
    scheduler.update_from_output(so, _model_output(so, rng))
    so = scheduler.schedule()  # a pure decode step, no new blocks
    assert len(so.scheduled_cached_reqs.req_ids) == num_reqs
    with_percy = len(pickle.dumps(so))
    cached = so.scheduled_cached_reqs
    cached.new_block_gens = []
    cached.revoked = []
    cached.expected_row_len = []
    cached.row_digest = []
    cached.write_windows = []
    so.msg_seq = 0
    without = len(pickle.dumps(so))
    per_request = (with_percy - without) / num_reqs
    print(f"Percy decode overhead: {per_request:.1f} bytes/request/group")
    assert per_request < 40
