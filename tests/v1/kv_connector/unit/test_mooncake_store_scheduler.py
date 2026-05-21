# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)


def _make_bare_scheduler() -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.original_block_size = 16
    scheduler._block_size = 16
    scheduler._discard_partial_chunks = True
    scheduler.load_specs = {}
    scheduler._preempted_req_ids = set()
    scheduler._unfinished_request_ids = {"req-0"}
    scheduler._unfinished_requests = {}
    scheduler._request_trackers = {}
    return scheduler


def _make_scheduler_output(*, scheduled_spec_tokens: list[int] | None):
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([2],)],
            num_computed_tokens=[44],
        ),
        num_scheduled_tokens={"req-0": 4},
        scheduled_spec_decode_tokens=(
            {"req-0": scheduled_spec_tokens} if scheduled_spec_tokens else {}
        ),
    )


def _add_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    token_ids: list[int],
    block_hashes: list[bytes],
    prefill_end_tokens: int,
) -> None:
    request = SimpleNamespace(
        all_token_ids=token_ids,
        block_hashes=block_hashes,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=44,
        allocated_block_ids=([0, 1],),
        num_saved_tokens=32,
        token_ids=token_ids[:44],
        prefill_end_tokens=prefill_end_tokens,
    )


def test_cached_request_with_spec_decode_does_not_save_scheduled_drafts():
    # Drafts in scheduled_spec_decode_tokens are not appended to all_token_ids
    # yet, so the tracker's token_len does not advance and num_tokens_to_save
    # stays below chunk_boundary — the save is naturally skipped.
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(44)),
        block_hashes=[b"h0", b"h1"],
        prefill_end_tokens=48,
    )

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=[101, 102, 103])
    )

    assert meta.requests == []
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 44
    assert tracker.num_saved_tokens == 32
    assert tracker.allocated_block_ids == ([0, 1, 2],)


def test_cached_request_without_spec_decode_keeps_current_step_save_overlap():
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=48,
    )

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=None)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.token_len_chunk == 48
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 48
    assert tracker.num_saved_tokens == 48


def _make_pending_load_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    num_tokens: int,
    block_hashes: list[bytes],
    block_ids: tuple[list[int], ...] = ([0, 1, 2],),
) -> None:
    request = SimpleNamespace(
        num_tokens=num_tokens,
        block_hashes=block_hashes,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, block_ids)


def _make_pending_load_scheduler_output() -> SimpleNamespace:
    """scheduler_output for a step where req-0 is parked on a pending load
    (not in scheduled_new_reqs or scheduled_cached_reqs)."""
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
    )


def test_pending_load_does_not_co_queue_save():
    # Regression: a cache-hit request waiting on an async load must not also
    # enqueue a save in the same scheduling step. Co-queuing both produces a
    # recv+send pair for the same req_id, and the scheduler's
    # _update_from_kv_xfer_finished then trips `assert req_id in self.requests`
    # when both completions land for the delay-freed request.
    scheduler = _make_bare_scheduler()
    _make_pending_load_unfinished_request(
        scheduler,
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(_make_pending_load_scheduler_output())

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    # Save must be off so the worker does not call add_stored_request.
    assert req_meta.can_save is False
    # Load is still issued as planned.
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.can_load is True
    # And the tracker's saved-tokens watermark stays at 0 so request_finished
    # later sees `num_saved_tokens <= 0` and frees immediately rather than
    # waiting for a finished_sending that will never come.
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0


def _make_resumed_unfinished_request(
    scheduler: MooncakeStoreScheduler,
    *,
    token_ids: list[int],
    block_hashes: list[bytes],
    num_computed_tokens: int,
) -> None:
    request = SimpleNamespace(
        all_token_ids=token_ids,
        block_hashes=block_hashes,
        num_computed_tokens=num_computed_tokens,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))


def _make_resumed_scheduler_output(*, num_scheduled_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([2],)],
            num_computed_tokens=[0],
        ),
        num_scheduled_tokens={"req-0": num_scheduled_tokens},
        scheduled_spec_decode_tokens={},
    )


def test_resumed_from_preemption_with_load_skips_save():
    # On resume-from-preemption with a cache hit, the same co-queueing race
    # applies: the resumed-from-preemption branch in build_connector_meta also
    # passes load_spec.can_load=True. Skip save in this step; subsequent
    # cached_reqs steps will save new tokens normally.
    scheduler = _make_bare_scheduler()
    scheduler._preempted_req_ids = {"req-0"}
    _make_resumed_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_computed_tokens=0,
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(
        _make_resumed_scheduler_output(num_scheduled_tokens=48)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is False
    assert req_meta.load_spec is not None
    assert req_meta.load_spec.can_load is True
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0


def test_resumed_from_preemption_without_load_still_saves():
    # No load_spec → behavior is unchanged: save proceeds.
    scheduler = _make_bare_scheduler()
    scheduler._preempted_req_ids = {"req-0"}
    _make_resumed_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_computed_tokens=0,
    )

    meta = scheduler.build_connector_meta(
        _make_resumed_scheduler_output(num_scheduled_tokens=48)
    )

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.load_spec is None
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 48
