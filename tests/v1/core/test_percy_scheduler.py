# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Percy A1: every scheduler free is a GrantTable release/revoke, revokes
reach the next SchedulerOutput, and planned writes are checked (I2)."""

import ast
from pathlib import Path

import pytest

import vllm.v1.core.sched as sched_pkg
from vllm.v1.core.grant_table import BlockState
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _output(scheduler_output, token: int = 0) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids, sampled_token_ids=[[token] for _ in req_ids]
    )


def _full_attention_manager(scheduler) -> FullAttentionManager:
    (manager,) = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert isinstance(manager, FullAttentionManager)
    return manager


def test_scheduler_frees_always_pass_step_seq():
    """Every KV free in the scheduler names the step that last used the
    blocks, so the GrantTable stamps `last_use` precisely (I5)."""
    offenders = []
    for path in Path(sched_pkg.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            func = node.func
            owner = ast.unparse(func.value)
            frees_kv = (func.attr == "free" and owner.endswith("kv_cache_manager")) or (
                func.attr == "free_blocks"
            )
            if frees_kv and not any(k.arg == "step_seq" for k in node.keywords):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_revoke_appears_in_next_scheduler_output():
    """Blocks freed between steps are revoked (gen + 1) and the new gens are
    carried by the next SchedulerOutput, once."""
    scheduler = create_scheduler(enable_prefix_caching=False, block_size=BLOCK_SIZE)
    (request,) = create_requests(1, num_tokens=40, max_tokens=1, block_size=BLOCK_SIZE)
    scheduler.add_request(request)
    first = scheduler.schedule()
    block_ids = list(first.scheduled_new_reqs[0].block_ids[0])
    assert first.gen_updates is None
    grant_table = scheduler.kv_cache_manager.grant_table
    old_gens = grant_table.gens_of(block_ids).tolist()

    scheduler.update_from_output(first, _output(first))
    assert request.is_finished()

    second = scheduler.schedule()
    assert second.gen_updates is not None
    updated = dict(zip(*second.gen_updates))
    assert set(block_ids) <= set(updated)
    assert [updated[b] for b in block_ids] == [g + 1 for g in old_gens]
    assert all(grant_table.state(b) == BlockState.FREE for b in block_ids)
    assert scheduler.schedule().gen_updates is None


def test_abort_revoke_appears_in_next_scheduler_output():
    scheduler = create_scheduler(enable_prefix_caching=False, block_size=BLOCK_SIZE)
    (request,) = create_requests(1, num_tokens=20, block_size=BLOCK_SIZE)
    scheduler.add_request(request)
    first = scheduler.schedule()
    block_ids = list(first.scheduled_new_reqs[0].block_ids[0])
    scheduler.update_from_output(first, _output(first))
    scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    updated = dict(zip(*scheduler.schedule().gen_updates))
    assert set(block_ids) <= set(updated)


def test_injected_write_into_other_requests_block_is_refused():
    """A planning bug that points request B's written column at request A's
    block is refused before the message is built; B is skipped for the step
    while A proceeds, and B runs again once the bug is gone."""
    scheduler = create_scheduler(enable_prefix_caching=False, block_size=BLOCK_SIZE)
    req_a, req_b = create_requests(2, num_tokens=10, block_size=BLOCK_SIZE)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _output(prefill))

    manager = _full_attention_manager(scheduler)
    b_row = manager.req_to_blocks[req_b.request_id]
    b_block = b_row[0]
    b_row[0] = manager.req_to_blocks[req_a.request_id][0]  # the injected bug
    try:
        decode = scheduler.schedule()
    finally:
        b_row[0] = b_block
    assert req_a.request_id in decode.num_scheduled_tokens
    assert req_b.request_id not in decode.num_scheduled_tokens
    assert req_b.status == RequestStatus.RUNNING

    scheduler.update_from_output(decode, _output(decode))
    assert req_b.request_id in scheduler.schedule().num_scheduled_tokens


def test_refused_waiting_request_rolls_back_and_stays_waiting(monkeypatch):
    """A refused new request gives back what its allocation took and stays
    waiting; the other new request is scheduled."""
    scheduler = create_scheduler(enable_prefix_caching=False, block_size=BLOCK_SIZE)
    req_a, req_b = create_requests(2, num_tokens=40, block_size=BLOCK_SIZE)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()

    manager = _full_attention_manager(scheduler)
    real = manager.write_columns
    foreign: list = []

    def buggy(request_id, start, end):
        columns = real(request_id, start, end)
        if request_id == req_b.request_id and foreign:
            return columns + foreign
        return columns

    monkeypatch.setattr(manager, "write_columns", buggy)
    orig_allocate = scheduler.kv_cache_manager.allocate_slots

    def allocate(request, *args, **kwargs):
        result = orig_allocate(request, *args, **kwargs)
        if request is req_a:
            foreign.append((0, manager.req_to_blocks[req_a.request_id][0]))
        return result

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate)
    output = scheduler.schedule()
    assert list(output.num_scheduled_tokens) == [req_a.request_id]
    assert req_b.status == RequestStatus.WAITING
    assert manager.req_to_blocks.get(req_b.request_id, []) == []
    num_a_blocks = len(manager.req_to_blocks[req_a.request_id])
    assert pool.get_num_free_blocks() == free_before - num_a_blocks


def test_short_make_room_fails_allocation_instead_of_raising():
    """Cached-idle blocks an in-flight step may still read count as free
    but cannot be evicted: allocation returns None (as if out of blocks)
    rather than failing inside get_new_blocks, and succeeds once drained."""
    stop = 42
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        num_blocks=6,
        block_size=BLOCK_SIZE,
        async_scheduling=True,
    )
    kv_cache_manager = scheduler.kv_cache_manager
    pool = kv_cache_manager.block_pool

    (cached,) = create_requests(
        1,
        num_tokens=4 * BLOCK_SIZE,
        max_tokens=5,
        stop_token_ids=[stop],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(cached)
    prefill = scheduler.schedule()
    decode = scheduler.schedule()  # over-scheduled, in flight
    assert cached.request_id in decode.num_scheduled_tokens

    # It stops on the prefill output while the decode (which may still read
    # its blocks) is in flight: 4 published blocks become cached-idle but not
    # evictable, the decode's unpublished block is DRAINING.
    scheduler.update_from_output(prefill, _output(prefill, token=stop))
    assert cached.is_finished()
    assert pool.get_num_free_blocks() == 4
    assert pool.get_num_allocatable_blocks() == 0

    # The second request's prompt differs from `cached`'s: no prefix hit.
    _, other = create_requests(
        2, num_tokens=3 * BLOCK_SIZE, block_size=BLOCK_SIZE, req_ids=["x", "other"]
    )
    assert kv_cache_manager.allocate_slots(other, other.num_tokens) is None

    scheduler.update_from_output(decode, _output(decode))
    assert pool.get_num_allocatable_blocks() == 5
    assert kv_cache_manager.allocate_slots(other, other.num_tokens) is not None


def test_write_into_foreign_published_block_is_refused():
    """Recomputing confirmed slots of the request's own published prefix is
    exempt; the same write aimed at another prefix's published, fully
    confirmed block is refused (no content-identity proof)."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    req_a, req_b = create_requests(2, num_tokens=2 * BLOCK_SIZE + 4, block_size=16)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _output(prefill))

    manager = _full_attention_manager(scheduler)
    grant_table = scheduler.kv_cache_manager.grant_table
    a_block = manager.req_to_blocks[req_a.request_id][0]
    assert grant_table.state(a_block.block_id) == BlockState.SHARED
    assert grant_table.valid(a_block.block_id) == BLOCK_SIZE

    b_row = manager.req_to_blocks[req_b.request_id]
    b_tail = b_row[2]
    b_row[2] = a_block  # the injected bug: B's written column -> A's prefix
    try:
        decode = scheduler.schedule()
    finally:
        b_row[2] = b_tail
    assert req_a.request_id in decode.num_scheduled_tokens
    assert req_b.request_id not in decode.num_scheduled_tokens
