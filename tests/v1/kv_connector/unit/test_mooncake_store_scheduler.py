# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    LoadSpec,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)


class _FakeBlockPool:
    def __init__(self, num_blocks: int = 32):
        self.blocks = [SimpleNamespace(block_id=i) for i in range(num_blocks)]
        self.touched: list[SimpleNamespace] = []
        self.freed: list[SimpleNamespace] = []

    def touch(self, blocks):
        self.touched.extend(blocks)

    def free_blocks(self, blocks):
        self.freed.extend(blocks)


def _make_bare_scheduler(
    *,
    hash_block_size: int = 16,
    enable_partial_hash_hits: bool = False,
    max_inflight_handoffs: int = 8,
) -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = "kv_both"
    scheduler.lookup_async = False
    scheduler.enable_lookup = True
    scheduler._block_size = 16
    scheduler._hash_block_size = hash_block_size
    scheduler.enable_partial_hash_hits = enable_partial_hash_hits
    scheduler.load_specs = {}
    scheduler._unfinished_request_ids = {"req-0"}
    scheduler._unfinished_requests = {}
    scheduler._request_trackers = {}
    scheduler._gpu_block_pool = _FakeBlockPool()
    scheduler._boundary_handoff_ids = itertools.count(1)
    scheduler._inflight_handoffs = {}
    scheduler._max_inflight_handoffs = max_inflight_handoffs
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
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req-0": 4},
        scheduled_spec_decode_tokens=(
            {"req-0": scheduled_spec_tokens} if scheduled_spec_tokens else {}
        ),
    )


def _make_preemption_scheduler_output():
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req-0"},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
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


def test_preemption_resets_tracker_before_request_finished():
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(44)),
        block_hashes=[b"h0", b"h1"],
        prefill_end_tokens=48,
    )
    scheduler._request_trackers["req-0"].has_pending_offload = True

    scheduler.build_connector_meta(_make_preemption_scheduler_output())

    tracker = scheduler._request_trackers["req-0"]
    assert tracker.token_len == 0
    assert tracker.allocated_block_ids == ()
    assert tracker.num_saved_tokens == 0
    assert tracker.token_ids is None
    assert tracker.has_pending_offload is False
    assert tracker.prefill_end_tokens == 0
    request = SimpleNamespace(request_id="req-0")
    assert scheduler.request_finished(request, ([0, 1],)) == (False, None)


def test_preemption_clears_stale_load_state():
    scheduler = _make_bare_scheduler()
    _make_pending_load_unfinished_request(
        scheduler,
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
        block_ids=([10, 11, 12],),
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=True,
    )

    meta = scheduler.build_connector_meta(_make_preemption_scheduler_output())

    assert meta.requests == []
    assert "req-0" not in scheduler.load_specs
    assert "req-0" not in scheduler._unfinished_requests


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
            resumed_req_ids=set(),
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
    # A resumed-from-preemption step: the scheduler lists the request in
    # resumed_req_ids and sends the FULL block table (replace semantics).
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([0, 1, 2],)],
            num_computed_tokens=[0],
            resumed_req_ids={"req-0"},
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


def test_running_request_not_in_resumed_req_ids_appends_blocks():
    """Regression: the replace-vs-append choice must follow the scheduler's
    cached_reqs.resumed_req_ids, NOT connector-local preemption history.

    A running request that is not resumed this step carries a *delta*
    new_block_ids and must be APPENDED to the tracker's existing blocks.
    Treating it as resumed would replace allocated_block_ids with just the
    delta while token_len stays at the full computed length, so the store
    path's block_ids[start // block_size] runs off the end (the
    "list index out of range" / token_len >> len(block_ids) bug).
    """
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=48,
    )

    out = _make_scheduler_output(scheduled_spec_tokens=None)
    assert "req-0" not in out.scheduled_cached_reqs.resumed_req_ids

    meta = scheduler.build_connector_meta(out)

    tracker = scheduler._request_trackers["req-0"]
    # Delta [2] appended to existing [0, 1] (decode path), not replaced by [2].
    assert tracker.allocated_block_ids == ([0, 1, 2],)
    # token_len stays covered by the block table: no store-path under-count.
    blocks_held = sum(len(g) for g in tracker.allocated_block_ids)
    assert tracker.token_len // scheduler._block_size <= blocks_held
    assert len(meta.requests) == 1
    assert meta.requests[0].token_len_chunk == 48


def test_resumed_request_in_resumed_req_ids_replaces_blocks():
    """A request the scheduler marks resumed gets the FULL block table in
    new_block_ids and must REPLACE the tracker's blocks (not append), even if
    a stale tracker from before preemption is still present."""
    scheduler = _make_bare_scheduler()
    _make_resumed_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_computed_tokens=0,
    )
    # Stale pre-preemption tracker that must be overwritten, not appended to.
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=99,
        allocated_block_ids=([7, 8, 9],),
        num_saved_tokens=0,
    )

    scheduler.build_connector_meta(
        _make_resumed_scheduler_output(num_scheduled_tokens=48)
    )

    tracker = scheduler._request_trackers["req-0"]
    # Replaced with the full table from new_block_ids, not appended to [7,8,9].
    assert tracker.allocated_block_ids == ([0, 1, 2],)
    assert tracker.token_len == 48
    blocks_held = sum(len(g) for g in tracker.allocated_block_ids)
    assert tracker.token_len // scheduler._block_size <= blocks_held


# Focused tests for ReqMeta.from_request_tracker — the centralized guard that
# enforces "a ReqMeta never carries both a save and a load".


def test_from_request_tracker_load_overrides_caller_skip_save():
    # Caller asks for skip_save=False, but load_spec.can_load=True. The
    # function must force skip_save=True to avoid producing a ReqMeta the
    # worker would enqueue on both kv_send_thread and kv_recv_thread.
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )
    load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=48, can_load=True)

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=load_spec,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is False
    assert req_meta.load_spec is load_spec
    assert tracker.num_saved_tokens == 0


def test_from_request_tracker_load_with_can_load_false_still_saves():
    # A LoadSpec with can_load=False (e.g., no external tokens to load after
    # update_state_after_alloc) must not suppress the save.
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )
    load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=48, can_load=False)

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=load_spec,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is True
    # from_request_tracker clears load_spec when can_load is False.
    assert req_meta.load_spec is None
    assert tracker.num_saved_tokens == 48


def test_from_request_tracker_no_load_saves_normally():
    tracker = RequestTracker(
        req_id="req-0",
        token_len=48,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
    )

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=None,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    assert req_meta is not None
    assert req_meta.can_save is True
    assert req_meta.load_spec is None
    assert tracker.num_saved_tokens == 48


class _StubLookupClient:
    def __init__(self, hit_tokens: int) -> None:
        self._hit_tokens = hit_tokens
        self.num_tokens: list[int] = []

    def lookup(
        self,
        req_id: str,
        num_tokens: int,
        block_hashes: list[bytes],
        non_block: bool = False,
    ) -> int:
        self.num_tokens.append(num_tokens)
        return self._hit_tokens


def test_full_external_hit_keeps_kvpool_cached_tokens_block_aligned():
    # The worker re-derives a full external hit below the request end on an
    # existing boundary, so the scheduler receives the usable aligned hit.
    scheduler = _make_bare_scheduler()
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=32)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=16
    )

    # 47 // 16 * 16 == 32 tokens left in external store after reserving the
    # sub-block tail for sampling. 32 - 16 (local) == 16 to load.
    assert need_to_allocate == 16
    assert load_async is True
    load_spec = scheduler.load_specs["req-0"]
    assert scheduler.client.num_tokens == [48]
    assert load_spec.vllm_cached_tokens == 16
    assert load_spec.kvpool_cached_tokens == 32
    assert load_spec.kvpool_cached_tokens % 16 == 0


def test_full_external_hit_with_full_local_hit_skips_load():
    # When local prefix cache already covers the block-aligned external hit,
    # there is nothing for the connector to load. The pre-fix behavior would
    # have scheduled a 15-token load that the recv thread couldn't translate
    # into any block-aligned key.
    scheduler = _make_bare_scheduler()
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=32)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=32
    )

    assert need_to_allocate == 0
    assert load_async is False
    assert "req-0" not in scheduler.load_specs


def test_partial_hash_hit_block_aligned_local_loads_partial_tail():
    # Fine-grained on (hash=4, block=16): a block-aligned local hit can pull a
    # sub-block remote hit (24 = a hash boundary inside block 1). Loads [16, 24).
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=24)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=32,
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4", b"h5", b"h6", b"h7"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=16
    )

    assert need_to_allocate == 8
    assert load_async is True
    load_spec = scheduler.load_specs["req-0"]
    assert load_spec.vllm_cached_tokens == 16
    assert load_spec.kvpool_cached_tokens == 24


def test_partial_hash_hit_no_remote_gain_skips_load():
    # Core always presents a block-aligned local hit (it floors a sub-block
    # tail before calling the connector). When the remote hit does not exceed
    # that block-aligned local hit, nothing is loaded.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=16)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=32,
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4", b"h5", b"h6", b"h7"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=16
    )

    assert need_to_allocate == 0
    assert load_async is False
    assert "req-0" not in scheduler.load_specs


def test_sub_block_prompt_looks_up_with_fine_grained():
    # A prompt smaller than one block (12 < block 16). With fine-grained partial
    # hits the sub-block prefix is worth looking up (floor is the hash unit 4,
    # not a full block), so a remote partial hit is loaded. Pre-change the
    # block-size floor returned (0, False) for such prompts.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler.load_async = True
    scheduler.client = _StubLookupClient(hit_tokens=8)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=12,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert need_to_allocate == 8
    assert load_async is True
    assert scheduler.load_specs["req-0"].kvpool_cached_tokens == 8


def test_sub_block_prompt_not_looked_up_without_fine_grained():
    # Without fine-grained partial hits, sub-block prompts still skip the lookup
    # (there is no full block, and no sub-block key granularity).
    scheduler = _make_bare_scheduler()
    scheduler.client = _StubLookupClient(hit_tokens=8)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=12,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert need_to_allocate == 0
    assert load_async is False
    assert "req-0" not in scheduler.load_specs


def test_disabled_lookup_reports_no_hit_without_querying_client():
    # With enable_lookup=False the connector reports no external hit without
    # consulting the lookup client, so admission is never deferred on a store
    # lookup. Used by instances that only contribute store capacity.
    scheduler = _make_bare_scheduler()
    scheduler.enable_lookup = False
    scheduler.client = _StubLookupClient(hit_tokens=32)

    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
    )

    need_to_allocate, load_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert need_to_allocate == 0
    assert load_async is False
    assert scheduler.client.num_tokens == []
    assert scheduler.load_specs == {}


def test_pending_boundary_state_emits_offload_only_reqmeta():
    # A sub-block prompt never produces a block-aligned save, so the boundary-
    # state offload arriving this step is emitted as an offload-only ReqMeta
    # (can_save=True so it takes the normal enqueue path, token_len_chunk=0 so
    # the worker skips the normal save). Pending-offload state delays the free
    # without advancing the normal-save watermark before the put succeeds.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    request = SimpleNamespace(
        all_token_ids=list(range(12)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_output_placeholders=0,
        num_prompt_tokens=12,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=12,
        allocated_block_ids=([0],),
        num_saved_tokens=0,
        token_ids=list(range(12)),
        prefill_end_tokens=12,
    )

    out = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        boundary_state_offloads={"req-0": [(1, 7, 12)]},
    )

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.token_len_chunk == 0
    assert req_meta.boundary_state_offloads == [(1, 1, 7, 12)]
    assert req_meta.num_prompt_tokens == 12
    assert req_meta.block_ids == ([0],)
    assert [block.block_id for block in scheduler._gpu_block_pool.touched] == [7]
    assert set(scheduler._inflight_handoffs) == {1}
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0
    assert tracker.has_pending_offload is True
    request = SimpleNamespace(request_id="req-0")
    assert scheduler.request_finished(request, ([0],)) == (True, None)


def test_decode_boundary_state_offload_dropped_unclaimed():
    # A hand-off past the prefill end can never complete a joint hybrid hit
    # (every other group stops saving there), so it is neither transferred nor
    # claimed — leaving the core free to release the block immediately.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    request = SimpleNamespace(
        all_token_ids=list(range(24)),
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4", b"h5"],
        num_output_placeholders=0,
        num_prompt_tokens=12,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=12,
        allocated_block_ids=([0],),
        num_saved_tokens=12,
        token_ids=list(range(12)),
        prefill_end_tokens=12,
    )

    out = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        # Boundary 16 is a decode boundary for a 12-token prefill.
        boundary_state_offloads={"req-0": [(1, 7, 16)]},
    )

    meta = scheduler.build_connector_meta(out)

    assert meta.requests == []
    assert scheduler._gpu_block_pool.touched == []
    assert scheduler._request_trackers["req-0"].has_pending_offload is False


def _register_offload_request(scheduler, *, prefill_end_tokens, num_prompt_tokens):
    request = SimpleNamespace(
        all_token_ids=list(range(64)),
        block_hashes=[bytes([i]) for i in range(16)],
        num_output_placeholders=0,
        num_prompt_tokens=num_prompt_tokens,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=prefill_end_tokens,
        allocated_block_ids=([0],),
        num_saved_tokens=prefill_end_tokens,
        token_ids=list(range(prefill_end_tokens)),
        prefill_end_tokens=prefill_end_tokens,
    )


def _make_offload_only_output(entries):
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        boundary_state_offloads={"req-0": entries},
    )


def test_resumed_prefill_claims_boundaries_past_prompt_length():
    # A resumed request re-prefills its previously generated tokens, so its
    # save window (`prefill_end_tokens`) extends past `num_prompt_tokens`.
    # Boundaries in that range must still be claimed, or the mamba key would be
    # missing for boundaries full attention does store and no joint hit could
    # complete there.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=20, num_prompt_tokens=12)
    out = _make_offload_only_output([(1, 7, 16), (1, 9, 20), (1, 11, 24)])

    meta = scheduler.build_connector_meta(out)

    # 16 and 20 are inside the resumed prefill; 24 is past it.
    assert meta.requests[0].boundary_state_offloads == [(1, 1, 7, 16), (2, 1, 9, 20)]
    assert [block.block_id for block in scheduler._gpu_block_pool.touched] == [7, 9]
    assert set(scheduler._inflight_handoffs) == {1, 2}


def test_boundary_state_claims_stop_at_inflight_limit():
    # Each claim pins a mamba state block, so past the cap the remaining
    # hand-offs are simply not claimed: a lost cache-save opportunity, never an
    # early release of a block whose transfer is still queued.
    scheduler = _make_bare_scheduler(
        hash_block_size=4, enable_partial_hash_hits=True, max_inflight_handoffs=2
    )
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    out = _make_offload_only_output([(1, 7, 16), (1, 8, 32), (1, 9, 48)])

    meta = scheduler.build_connector_meta(out)

    assert meta.requests[0].boundary_state_offloads == [(1, 1, 7, 16), (2, 1, 8, 32)]
    assert [block.block_id for block in scheduler._gpu_block_pool.touched] == [7, 8]
    assert set(scheduler._inflight_handoffs) == {1, 2}

    # Still saturated on the next step, so nothing new is claimed...
    out = _make_offload_only_output([(1, 10, 48)])
    assert scheduler.build_connector_meta(out).requests == []
    assert [block.block_id for block in scheduler._gpu_block_pool.touched] == [7, 8]

    # ...until the worker reports both sends done, which releases both pins.
    scheduler.update_boundary_handoff_watermark(2)
    assert [block.block_id for block in scheduler._gpu_block_pool.freed] == [7, 8]
    out = _make_offload_only_output([(1, 10, 48)])
    meta = scheduler.build_connector_meta(out)
    assert meta.requests[0].boundary_state_offloads == [(3, 1, 10, 48)]
    assert set(scheduler._inflight_handoffs) == {3}


def test_boundary_state_release_is_per_entry():
    # The watermark releases exactly the entries the worker has finished with;
    # a later, still-queued hand-off keeps its pin.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    scheduler.build_connector_meta(_make_offload_only_output([(1, 7, 16), (1, 8, 32)]))

    scheduler.update_boundary_handoff_watermark(1)
    assert [block.block_id for block in scheduler._gpu_block_pool.freed] == [7]
    assert set(scheduler._inflight_handoffs) == {2}


def test_preemption_and_request_id_reuse_do_not_release_inflight_pin():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    scheduler.build_connector_meta(_make_offload_only_output([(1, 7, 16)]))

    scheduler.build_connector_meta(_make_preemption_scheduler_output())
    assert scheduler._gpu_block_pool.freed == []
    assert set(scheduler._inflight_handoffs) == {1}

    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    meta = scheduler.build_connector_meta(_make_offload_only_output([(1, 8, 32)]))
    assert meta.requests[0].boundary_state_offloads == [(2, 1, 8, 32)]

    scheduler.update_boundary_handoff_watermark(1)
    assert [block.block_id for block in scheduler._gpu_block_pool.freed] == [7]
    assert set(scheduler._inflight_handoffs) == {2}


def test_boundary_state_never_claimed_without_a_send_thread():
    # kv_consumer has no send thread to acknowledge a hand-off, so claiming one
    # would pin a state block until the request's blocks are freed.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler.kv_role = "kv_consumer"
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    out = _make_offload_only_output([(1, 7, 16)])

    assert scheduler.build_connector_meta(out).requests == []
    assert scheduler._gpu_block_pool.touched == []
    assert scheduler._inflight_handoffs == {}


def test_resumed_partial_tail_uses_handoff_boundary():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    request = SimpleNamespace(
        all_token_ids=list(range(20)),
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4"],
        num_output_placeholders=0,
        num_prompt_tokens=12,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=20,
        allocated_block_ids=([0, 1],),
        num_saved_tokens=0,
        token_ids=list(range(20)),
        # Resumption replays prompt + previously generated tokens.
        prefill_end_tokens=20,
    )

    out = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={},
        scheduled_spec_decode_tokens={},
        boundary_state_offloads={"req-0": [(1, 7, 12)]},
    )

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    assert meta.requests[0].boundary_state_offloads == [(1, 1, 7, 12)]
    # Ordinary metadata retains the full resumed prefill range.
    assert meta.requests[0].num_prompt_tokens == 20
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0
    assert tracker.has_pending_offload is True


def test_resumed_partial_tail_attached_to_save_keeps_handoff_boundary():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    request = SimpleNamespace(
        all_token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_output_placeholders=0,
        num_prompt_tokens=36,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([0, 1],))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=44,
        allocated_block_ids=([0, 1],),
        num_saved_tokens=32,
        token_ids=list(range(44)),
        prefill_end_tokens=48,
    )
    out = _make_scheduler_output(scheduled_spec_tokens=None)
    out.boundary_state_offloads = {"req-0": [(0, 7, 36)]}

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    assert meta.requests[0].can_save is True
    assert meta.requests[0].boundary_state_offloads == [(1, 0, 7, 36)]
    assert meta.requests[0].num_prompt_tokens == 48
    # Ordinary saving still covers the full resumed prefill range.
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 48
    assert tracker.has_pending_offload is True
