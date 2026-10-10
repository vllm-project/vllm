# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    LoadSpec,
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
    RequestTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
    _partial_tail_non_mamba_puts,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def _make_bare_scheduler(
    *,
    hash_block_size: int = 16,
    enable_partial_hash_hits: bool = False,
    kv_role: str = "kv_both",
    save_decode_cache: bool = False,
) -> MooncakeStoreScheduler:
    scheduler = object.__new__(MooncakeStoreScheduler)
    scheduler.kv_role = kv_role
    scheduler.save_decode_cache = save_decode_cache
    scheduler.enable_kv_events = False
    scheduler.lookup_async = False
    scheduler.enable_lookup = True
    scheduler._block_size = 16
    scheduler._hash_block_size = hash_block_size
    scheduler.enable_partial_hash_hits = enable_partial_hash_hits
    scheduler.kv_cache_config = SimpleNamespace()
    scheduler._store_group_ids = (0,)
    scheduler._store_group_id_by_kv_cache_group_id = {0: 0, 1: 1}
    scheduler.load_specs = {}
    scheduler._unfinished_request_ids = {"req-0"}
    scheduler._unfinished_requests = {}
    scheduler._allocated_req_ids = set()
    scheduler._request_trackers = {}
    scheduler._finished_partial_tail_metas = {}
    scheduler._gpu_block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=hash_block_size
    )
    scheduler._num_workers = 1
    scheduler._next_store_job_id = 0
    scheduler._pinned_saves = {}
    scheduler._boundary_state_group_ids = frozenset({1})
    scheduler._store_coord = MooncakeStoreCoordinator(
        [
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=max(16, hash_block_size),
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=max(16, hash_block_size),
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
        max(16, hash_block_size),
        hash_block_size,
        enable_partial_hash_hits=enable_partial_hash_hits,
    )
    return scheduler


def _make_connector_block_state(
    block_ids: tuple[list[int], ...] | None = None,
    offloads: list[tuple[int, int, int]] | None = None,
) -> KVConnectorBlockState:
    tables = {} if block_ids is None else {"req-0": block_ids}
    return KVConnectorBlockState(
        req_ids=set(tables),
        resolve_block_ids=tables.__getitem__,
        boundary_state_offloads=({} if offloads is None else {"req-0": offloads}),
    )


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
        kv_connector_block_state=_make_connector_block_state(block_ids=([0, 1, 2],)),
    )


def _make_decode_scheduler_output(
    *, num_computed_tokens: int, num_scheduled_tokens: int = 1
) -> SimpleNamespace:
    return SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            # The block that becomes full was allocated on an earlier step.
            new_block_ids=[()],
            num_computed_tokens=[num_computed_tokens],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req-0": num_scheduled_tokens},
        scheduled_spec_decode_tokens={},
        kv_connector_block_state=_make_connector_block_state(block_ids=([0, 1, 2],)),
    )


def _make_new_scheduler_output() -> SimpleNamespace:
    request = SimpleNamespace(
        req_id="req-0",
        num_computed_tokens=0,
        prompt_token_ids=list(range(32)),
        num_prompt_tokens=32,
        prefill_token_ids=None,
        block_ids=([0, 1],),
        block_hashes=[b"h0", b"h1"],
    )
    return SimpleNamespace(
        request=request,
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req-0": 32},
        scheduled_spec_decode_tokens={},
        kv_connector_block_state=_make_connector_block_state(block_ids=([0, 1],)),
    )


def test_scheduler_only_tracks_token_ids_for_kv_events():
    for enable_kv_events in (False, True):
        scheduler = _make_bare_scheduler()
        scheduler.enable_kv_events = enable_kv_events
        scheduler._unfinished_requests["req-0"] = (
            _make_new_scheduler_output().request,
            ([0, 1],),
        )

        meta = scheduler.build_connector_meta(_make_new_scheduler_output())

        tracker = scheduler._request_trackers["req-0"]
        req_meta = meta.requests[0]
        if enable_kv_events:
            assert tracker.token_ids == list(range(32))
            assert req_meta.token_ids == list(range(32))
            assert req_meta.token_ids_start == 0
            tracker.token_ids.append(99)
            assert req_meta.token_ids == list(range(32))
        else:
            assert tracker.token_ids is None
            assert req_meta.token_ids is None


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


def _make_worker_output(completed_saves: dict[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_worker_meta=MooncakeStoreWorkerMetadata(
            completed_saves=completed_saves
        )
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
        num_prompt_tokens=prefill_end_tokens,
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


def test_pending_load_for_non_chosen_connector_is_dropped():
    """A MultiConnector loser must not turn its proposed load into a save."""
    scheduler = _make_bare_scheduler()
    request = SimpleNamespace(
        request_id="req-0",
        block_hashes=[b"h0", b"h1", b"h2"],
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2], [9]))
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=False,
    )

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=0)
    meta = scheduler.build_connector_meta(_make_pending_load_scheduler_output())

    # MultiConnector exposes the real allocation to every child, but only the
    # chosen child receives external tokens. The losing store connector must
    # neither retain those blocks for a pending load nor enqueue a save from
    # the rejected speculative LoadSpec.
    assert scheduler._unfinished_requests["req-0"][1] == ()
    assert meta.requests == []
    assert "req-0" not in scheduler.load_specs
    assert "req-0" not in scheduler._request_trackers


def _make_qsa_hybrid_cache_config():
    full = FullAttentionSpec(block_size=800, num_kv_heads=8, head_size=64, dtype=None)
    circular = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        head_size_v=0,
        dtype=torch.float16,
    )
    mamba = MambaSpec(
        block_size=800,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["qsa"], circular),
            KVCacheGroupSpec(["mamba"], mamba),
        ],
    )


def test_scheduler_projects_nonprefix_groups_and_mamba_ids():
    vllm_config = SimpleNamespace(
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both", kv_connector_extra_config={}
        ),
        kv_events_config=None,
        cache_config=SimpleNamespace(
            block_size=800, enable_prefix_caching=True, prefix_match_unit=None
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, world_size=1),
    )

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store."
        "scheduler.LookupKeyClient"
    ):
        scheduler = MooncakeStoreScheduler(vllm_config, _make_qsa_hybrid_cache_config())

    assert scheduler._store_group_ids == (0, 2)
    assert scheduler._boundary_state_group_ids == frozenset({1})


def test_current_save_block_ids_use_store_group_projection():
    scheduler = _make_bare_scheduler()
    scheduler.kv_cache_config = _make_qsa_hybrid_cache_config()
    scheduler._store_group_ids = (0, 2)
    meta = MooncakeStoreConnectorMetadata(set(), set())
    req_meta = ReqMeta(
        req_id="req-0",
        token_len_chunk=800,
        block_ids=(),
        block_hashes=[b"h0"],
        can_save=True,
    )
    meta.add_request(req_meta)
    output = SimpleNamespace(
        kv_connector_block_state=_make_connector_block_state(([10], [80], [30]))
    )

    scheduler._apply_current_save_block_ids(meta, output)

    assert req_meta.block_ids == ([10], [30])


def test_update_state_excludes_nontransfer_groups():
    """Store metadata must match the worker's registered cache groups."""
    scheduler = _make_bare_scheduler()
    scheduler._store_group_ids = (0,)
    request = SimpleNamespace(request_id="req-1")
    blocks = KVCacheBlocks(([KVCacheBlock(1), KVCacheBlock(2)], [KVCacheBlock(9)]))

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=32)

    assert scheduler._unfinished_requests["req-1"][1] == ([1, 2],)


def _setup_decode_request(
    *,
    kv_role: str = "kv_consumer",
    save_decode_cache: bool = False,
    token_len: int = 47,
) -> tuple[MooncakeStoreScheduler, RequestTracker]:
    scheduler = _make_bare_scheduler(
        kv_role=kv_role, save_decode_cache=save_decode_cache
    )
    token_ids = list(range(token_len + 1))
    _add_unfinished_request(
        scheduler,
        token_ids=token_ids,
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=32,
    )
    tracker = scheduler._request_trackers["req-0"]
    tracker.token_len = token_len
    tracker.allocated_block_ids = ([0, 1, 2],)
    tracker.token_ids = token_ids[:token_len]
    return scheduler, tracker


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


@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both"])
def test_decode_tracking_is_skipped_by_default(kv_role):
    scheduler, tracker = _setup_decode_request(kv_role=kv_role)

    meta = scheduler.build_connector_meta(
        _make_decode_scheduler_output(num_computed_tokens=47)
    )

    assert meta.requests == []
    assert tracker.token_len == 47
    assert tracker.num_saved_tokens == 32
    assert tracker.allocated_block_ids == ([0, 1, 2],)
    assert tracker.token_ids == list(range(47))


def test_fresh_consumer_first_decode_save_can_backfill_missing_prompt():
    scheduler = _make_bare_scheduler(
        kv_role="kv_consumer",
        save_decode_cache=True,
    )
    scheduler.enable_kv_events = True
    new_output = _make_new_scheduler_output()
    request = new_output.request
    request.block_ids = ([0, 1, 2],)
    request.block_hashes = [b"h0", b"h1", b"h2"]
    request.all_token_ids = list(range(48))
    request.num_output_placeholders = 0
    scheduler._unfinished_requests["req-0"] = (request, request.block_ids)

    # The consumer does not save during prefill, so its first decode save must
    # still cover the prompt. The worker deduplicates prompt blocks already in
    # the Store and fills any that are missing, preserving a complete prefix.
    assert scheduler.build_connector_meta(new_output).requests == []
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0

    # The first decode step checks/saves the block-aligned prompt. The worker's
    # Store dedup turns this into a no-op when the producer already saved it.
    [prompt_meta] = scheduler.build_connector_meta(
        _make_decode_scheduler_output(num_computed_tokens=32)
    ).requests
    assert prompt_meta.token_len_chunk == 32
    assert prompt_meta.token_ids_start == 0
    assert prompt_meta.token_ids == list(range(32))

    for num_computed_tokens in range(33, 48):
        meta = scheduler.build_connector_meta(
            _make_decode_scheduler_output(
                num_computed_tokens=num_computed_tokens,
            )
        )
        if num_computed_tokens < 47:
            assert meta.requests == []

    [req_meta] = meta.requests
    assert req_meta.token_len_chunk == 48
    assert req_meta.token_ids_start == 32
    assert req_meta.token_ids == list(range(32, 48))
    assert tracker.num_saved_tokens == 48


@pytest.mark.parametrize("token_len, saved_tokens", [(46, 32), (47, 48)])
def test_consumer_saves_only_full_decode_blocks(token_len, saved_tokens):
    scheduler, tracker = _setup_decode_request(
        save_decode_cache=True, token_len=token_len
    )

    meta = scheduler.build_connector_meta(
        _make_decode_scheduler_output(num_computed_tokens=token_len)
    )

    assert tracker.token_len == token_len + 1
    assert tracker.num_saved_tokens == saved_tokens
    if saved_tokens == 32:
        assert meta.requests == []
    else:
        [req_meta] = meta.requests
        assert req_meta.can_save is True
        assert req_meta.token_len_chunk == 48
        assert req_meta.block_ids == ([0, 1, 2],)
        assert req_meta.token_ids == list(range(32, 48))
        assert req_meta.token_ids_start == 32
        tracker.token_ids.append(999)
        assert req_meta.token_ids == list(range(32, 48))


def test_preemption_resets_tracker():
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
        num_prompt_tokens=num_tokens,
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
    # when a completion lands for a request it has already dropped.
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
    # And the save watermark does not advance for a save that was never queued.
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
        num_prompt_tokens=32,
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
        kv_connector_block_state=_make_connector_block_state(block_ids=([0, 1, 2],)),
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


@pytest.mark.parametrize(
    "make_output",
    [
        pytest.param(_make_new_scheduler_output, id="new"),
        pytest.param(
            partial(_make_resumed_scheduler_output, num_scheduled_tokens=48),
            id="resumed",
        ),
    ],
)
def test_preempted_request_readmitted_in_same_step(make_output):
    scheduler = _make_bare_scheduler()
    request = SimpleNamespace(
        request_id="req-0",
        all_token_ids=list(range(48)),
        num_computed_tokens=0,
        block_hashes=[b"h0", b"h1", b"h2"],
        num_output_placeholders=0,
    )
    scheduler.update_state_after_alloc(request, SimpleNamespace(), 0)
    out = make_output()
    out.preempted_req_ids = {"req-0"}

    meta = scheduler.build_connector_meta(out)

    assert [req_meta.req_id for req_meta in meta.requests] == ["req-0"]
    assert "req-0" in scheduler._unfinished_requests


def test_preempted_request_readmitted_with_pending_load_in_same_step():
    scheduler = _make_bare_scheduler()
    request = SimpleNamespace(
        request_id="req-0",
        num_tokens=48,
        block_hashes=[b"h0", b"h1", b"h2"],
        num_output_placeholders=0,
    )
    scheduler.load_specs["req-0"] = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=48,
        can_load=False,
    )
    blocks = SimpleNamespace(get_block_ids=lambda group_ids: ([10, 11, 12],))
    scheduler.update_state_after_alloc(request, blocks, 48)
    out = _make_pending_load_scheduler_output()
    out.preempted_req_ids = {"req-0"}

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    load_spec = meta.requests[0].load_spec
    assert load_spec is not None and load_spec.can_load


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


@pytest.mark.parametrize("token_len,save_partial_tail", [(48, False), (13, True)])
def test_from_request_tracker_no_load_saves_normally(token_len, save_partial_tail):
    tracker = RequestTracker(
        req_id="req-0",
        token_len=token_len,
        allocated_block_ids=([0, 1, 2],),
        num_saved_tokens=0,
        prefill_end_tokens=token_len,
    )
    if save_partial_tail:
        tracker.token_len = 8
        assert (
            ReqMeta.from_request_tracker(
                tracker, 16, save_partial_tail=True, num_prompt_tokens=token_len
            )
            is None
        )
        tracker.token_len = token_len

    req_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=16,
        load_spec=None,
        skip_save=False,
        block_hashes=[b"h0", b"h1", b"h2"],
        save_partial_tail=save_partial_tail,
        num_prompt_tokens=token_len,
    )

    assert req_meta is not None
    assert req_meta.can_save is True
    assert req_meta.load_spec is None
    assert req_meta.token_len_chunk == token_len // 16 * 16
    assert req_meta.completed_token_len == token_len
    assert req_meta.boundary_puts is None
    assert tracker.num_saved_tokens == token_len // 16 * 16
    assert (
        ReqMeta.from_request_tracker(
            tracker,
            16,
            save_partial_tail=save_partial_tail,
            num_prompt_tokens=token_len,
        )
        is None
    )


def test_partial_tail_is_resolved_once_on_the_prompt_completing_save():
    scheduler = _make_bare_scheduler(
        hash_block_size=4, enable_partial_hash_hits=True, save_decode_cache=True
    )
    scheduler._store_group_ids = (0, 1)
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[bytes([i]) for i in range(12)],
        prefill_end_tokens=45,
    )
    block_ids = ([0, 1, 2], [50])
    scheduler._request_trackers["req-0"].allocated_block_ids = block_ids

    def step(num_computed_tokens: int, num_scheduled_tokens: int):
        return scheduler.build_connector_meta(
            SimpleNamespace(
                finished_req_ids=set(),
                preempted_req_ids=set(),
                scheduled_new_reqs=[],
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=["req-0"],
                    new_block_ids=[()],
                    num_computed_tokens=[num_computed_tokens],
                    resumed_req_ids=set(),
                ),
                num_scheduled_tokens={"req-0": num_scheduled_tokens},
                scheduled_spec_decode_tokens={},
                kv_connector_block_state=_make_connector_block_state(block_ids),
            )
        ).requests

    # The save that completes the prompt carries the tail for the worker.
    (req_meta,) = step(44, 1)
    assert req_meta.publish_partial_tail
    assert req_meta.boundary_puts == [(0, 2, 44)]

    # A later save does not recompute or resend it.
    (req_meta,) = step(45, 3)
    assert req_meta.token_len_chunk == 48
    assert not req_meta.publish_partial_tail
    assert req_meta.boundary_puts is None


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
    ) -> MooncakeLookupResult:
        self.num_tokens.append(num_tokens)
        return MooncakeLookupResult(self._hit_tokens)


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


def _add_pending_partial_tail_request(
    scheduler: MooncakeStoreScheduler,
    *,
    num_tokens: int,
    block_hashes: list[bytes],
    block_ids: tuple[list[int], ...],
) -> SimpleNamespace:
    """Register a sub-block request and return the step that offloads its tail.

    The CoW block holding the tail is block 7, which the core deliberately keeps
    out of the request's block table.
    """
    request = SimpleNamespace(
        all_token_ids=list(range(num_tokens)),
        block_hashes=block_hashes,
        num_output_placeholders=0,
        num_prompt_tokens=13,
    )
    scheduler._unfinished_requests["req-0"] = (request, block_ids)
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=num_tokens,
        allocated_block_ids=block_ids,
        num_saved_tokens=0,
        token_ids=list(range(num_tokens)),
        prefill_end_tokens=num_tokens,
    )
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
        kv_connector_block_state=_make_connector_block_state(
            block_ids=block_ids,
            offloads=[(1, 7, 12)],
        ),
    )


def test_pending_partial_tail_emits_offload_only_reqmeta():
    # A sub-block prompt never produces a block-aligned save, so the partial-
    # tail offload arriving this step is emitted as an offload-only ReqMeta
    # (can_save=True so it takes the normal enqueue path, token_len_chunk=0 so
    # the worker skips the normal save), without advancing the normal-save
    # watermark before the put succeeds.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    out = _add_pending_partial_tail_request(
        scheduler,
        num_tokens=13,
        block_hashes=[b"h0", b"h1", b"h2"],
        block_ids=([0],),
    )

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.can_save is True
    assert req_meta.token_len_chunk == 0
    assert req_meta.boundary_puts == [(1, 7, 12)]
    assert req_meta.num_prompt_tokens == 13
    assert req_meta.block_ids == ([0],)
    store_job_id = req_meta.store_job_id
    assert scheduler._pinned_saves[store_job_id][0] == [7]
    assert scheduler._gpu_block_pool.blocks[0].ref_cnt == 0
    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 1
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0
    assert tracker.has_pending_offload is True


@pytest.mark.parametrize("attention_block_size", [4, 8, 16])
@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("num_workers", [1, 2])
def test_finished_partial_tail_is_pre_pinned_as_store_job(
    attention_block_size, use_eagle, num_workers
):
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler._store_group_ids = (0, 1)
    scheduler.client = SimpleNamespace(discard=lambda *_: None)
    scheduler._num_workers = num_workers
    groups = list(scheduler._store_coord.kv_cache_groups)
    groups[0] = KVCacheGroupSpec(
        ["attention"],
        FullAttentionSpec(
            block_size=attention_block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        is_eagle_group=use_eagle,
    )
    scheduler._store_coord = MooncakeStoreCoordinator(
        groups, 16, 4, use_eagle, enable_partial_hash_hits=True
    )
    prompt_tokens = 49 if use_eagle else 45
    attention_ids = list(
        range(1, (prompt_tokens + attention_block_size - 1) // attention_block_size + 1)
    )
    block_ids = (attention_ids, [50])
    request = SimpleNamespace(
        request_id="req-0",
        block_hashes=[bytes([i]) for i in range(prompt_tokens // 4)],
        num_computed_tokens=prompt_tokens,
        num_prompt_tokens=prompt_tokens,
    )
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=prompt_tokens,
        allocated_block_ids=block_ids,
        token_ids=list(range(prompt_tokens)),
        prefill_end_tokens=prompt_tokens,
    )
    pool = scheduler._gpu_block_pool
    owned_ids = attention_ids + [50]
    pool.touch([pool.blocks[i] for i in owned_ids])

    delay_free = scheduler.register_finished_partial_tail(
        request,
        block_ids,
        [(1, 50, 44)],
    )

    # The exact source is pinned immediately, so the request can be freed
    # before the next connector metadata build.
    assert delay_free is False
    proof_end = 48 if use_eagle else 44
    expected = attention_ids[
        32 // attention_block_size : (proof_end + attention_block_size - 1)
        // attention_block_size
    ]
    pool.free_blocks([pool.blocks[i] for i in owned_ids])
    assert all(
        pool.blocks[i].ref_cnt == (1 if i in expected else 0) for i in attention_ids
    )
    assert pool.blocks[50].ref_cnt == 1

    out = SimpleNamespace(
        finished_req_ids={"req-0"},
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
        kv_connector_block_state=_make_connector_block_state(),
    )
    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    req_meta = meta.requests[0]
    assert req_meta.req_id == "req-0"
    assert req_meta.token_len_chunk == 0
    assert req_meta.block_ids == block_ids
    assert req_meta.block_hashes == request.block_hashes
    assert req_meta.boundary_puts == [
        (0, attention_ids[idx], min((idx + 1) * attention_block_size, proof_end))
        for idx in range(
            32 // attention_block_size,
            (proof_end + attention_block_size - 1) // attention_block_size,
        )
    ] + [(1, 50, 44)]
    assert set(scheduler._pinned_saves[req_meta.store_job_id][0]) == {50, *expected}
    assert all(pool.blocks[i].ref_cnt == 1 for i in expected + [50])
    assert scheduler._finished_partial_tail_metas == {}

    for _ in range(num_workers - 1):
        scheduler.update_connector_output(
            _make_worker_output({req_meta.store_job_id: 1})
        )
        assert all(pool.blocks[i].ref_cnt == 1 for i in expected + [50])
    scheduler.update_connector_output(_make_worker_output({req_meta.store_job_id: 1}))
    assert all(pool.blocks[i].ref_cnt == 0 for i in owned_ids)


def test_partial_tail_boundary_follows_eagle_block_drop_not_group_flags():
    """Core drops the Mamba checkpoint whenever EAGLE block drop is on, even if
    only the Mamba group carries the eagle flag."""
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            is_eagle_group=True,
        ),
    ]
    coord = MooncakeStoreCoordinator(groups, 16, 4, use_eagle=True)
    # Core's checkpoint for a 49-token prompt: 48 minus one EAGLE hash unit.
    req_meta = ReqMeta(
        req_id="req-0",
        token_len_chunk=0,
        block_ids=([1, 2, 3, 4], [50]),
        block_hashes=[bytes([i]) for i in range(12)],
        num_prompt_tokens=49,
        completed_token_len=49,
        boundary_puts=[BoundaryPut(1, 50, 44)],
    )

    assert _partial_tail_non_mamba_puts(coord, req_meta, [16, 16]) == [
        BoundaryPut(0, 3, 44)
    ]


@pytest.mark.parametrize(
    "completed, mamba_tails, publish, expected",
    [
        # The chunk ending at the junction (40) hands off its Mamba state. Its
        # attention KV comes from the normal saves, so nothing is added here.
        (40, [40], False, []),
        # The prompt-completing save: blocks from the LCM floor 64 up to the
        # checkpoint 68 plus the EAGLE proof unit, whether or not the junction
        # hand-off is in the same save.
        (75, [68], True, [BoundaryPut(0, 17, 68), BoundaryPut(0, 18, 72)]),
        (75, [40, 68], True, [BoundaryPut(0, 17, 68), BoundaryPut(0, 18, 72)]),
    ],
)
def test_shared_prefix_junction_handoff_adds_no_attention_tail(
    completed, mamba_tails, publish, expected
):
    """``--enable-mamba-shared-prefix-checkpoint`` also hands off a sub-block
    Mamba state at the shared-prefix junction, below the prompt checkpoint."""
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
            is_eagle_group=True,
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=16,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]
    coord = MooncakeStoreCoordinator(
        groups, 16, 4, use_eagle=True, enable_partial_hash_hits=True
    )
    # 75-token prompt: checkpoint 72 - 4 = 68.
    req_meta = ReqMeta(
        req_id="req-0",
        token_len_chunk=0,
        block_ids=(list(range(1, 20)), [50, 51, 52, 53, 54]),
        block_hashes=[bytes([i]) for i in range(18)],
        num_prompt_tokens=75,
        completed_token_len=completed,
        boundary_puts=[BoundaryPut(1, 60 + i, n) for i, n in enumerate(mamba_tails)],
        publish_partial_tail=publish,
    )

    assert _partial_tail_non_mamba_puts(coord, req_meta, [4, 16]) == expected


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
        kv_connector_block_state=_make_connector_block_state(offloads=[(1, 7, 16)]),
    )

    meta = scheduler.build_connector_meta(out)

    assert meta.requests == []
    assert scheduler._pinned_saves == {}
    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 0
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


def _make_offload_only_output(entries, block_ids=([0],)):
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
        kv_connector_block_state=_make_connector_block_state(
            block_ids=block_ids,
            offloads=entries,
        ),
    )


def test_boundary_state_group_ids_are_remapped_to_store_projection():
    scheduler = _make_bare_scheduler(hash_block_size=800, enable_partial_hash_hits=True)
    scheduler.kv_cache_config = _make_qsa_hybrid_cache_config()
    scheduler._store_group_ids = (0, 2)
    scheduler._store_group_id_by_kv_cache_group_id = {0: 0, 2: 1}
    scheduler._boundary_state_group_ids = frozenset({1})
    _register_offload_request(scheduler, prefill_end_tokens=800, num_prompt_tokens=800)
    meta = MooncakeStoreConnectorMetadata(set(), set())

    scheduler._handle_boundary_state_offloads({"req-0": [(2, 7, 800)]}, meta)

    assert meta.requests[0].boundary_puts == [(1, 7, 800)]


def test_resumed_prefill_claims_boundaries_past_prompt_length():
    # A resumed request re-prefills its previously generated tokens, so its
    # save window (`prefill_end_tokens`) extends past `num_prompt_tokens`.
    # Boundaries in that range must still be claimed, or the mamba key would be
    # missing for boundaries full attention does store and no joint hit could
    # complete there.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=37, num_prompt_tokens=13)
    out = _make_offload_only_output([(1, 7, 16), (1, 9, 32), (1, 11, 48)])

    meta = scheduler.build_connector_meta(out)

    # Aligned states at 16 and 32 are inside the resumed prefill; 48 is past it.
    assert meta.requests[0].boundary_puts == [(1, 7, 16), (1, 9, 32)]
    store_job_id = meta.requests[0].store_job_id
    assert scheduler._pinned_saves[store_job_id][0] == [7, 9]


def test_boundary_state_job_pins_exact_blocks_once():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    scheduler._request_trackers["req-0"].allocated_block_ids = ([7],)
    out = _make_offload_only_output(
        [(1, 7, 16), (1, 8, 32), (1, 9, 48)], block_ids=([7],)
    )

    meta = scheduler.build_connector_meta(out)

    req_meta = meta.requests[0]
    assert req_meta.boundary_puts == [
        (1, 7, 16),
        (1, 8, 32),
        (1, 9, 48),
    ]
    store_job_id = req_meta.store_job_id
    assert scheduler._pinned_saves[store_job_id][0] == [7, 8, 9]
    assert [scheduler._gpu_block_pool.blocks[i].ref_cnt for i in (7, 8, 9)] == [
        1,
        1,
        1,
    ]

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))
    assert [scheduler._gpu_block_pool.blocks[i].ref_cnt for i in (7, 8, 9)] == [
        0,
        0,
        0,
    ]


def test_store_job_pins_current_non_null_non_mamba_blocks():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler._store_group_ids = (0, 1)
    request = SimpleNamespace(
        all_token_ids=list(range(48)),
        block_hashes=[bytes([i]) for i in range(12)],
        num_prompt_tokens=48,
        num_output_placeholders=0,
    )
    scheduler._unfinished_requests["req-0"] = (request, ([7, 2, 0], [21, 0, 22]))
    scheduler._request_trackers["req-0"] = RequestTracker(
        req_id="req-0",
        token_len=44,
        allocated_block_ids=([7, 2, 0], [21, 0, 22]),
        num_saved_tokens=32,
        token_ids=list(range(44)),
        prefill_end_tokens=48,
    )
    stale_block_ids = ([7, 2, 0, 5], [21, 0, 22, 23])
    current_block_ids = ([7, 2, 0, 6], [24, 0, 25, 26])
    out = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-0"],
            new_block_ids=[([5], [23])],
            num_computed_tokens=[44],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req-0": 4},
        scheduled_spec_decode_tokens={},
        kv_connector_block_state=_make_connector_block_state(
            block_ids=current_block_ids,
            offloads=[(1, 8, 16)],
        ),
    )
    pool = scheduler._gpu_block_pool

    meta = scheduler.build_connector_meta(out)

    assert scheduler._request_trackers["req-0"].allocated_block_ids == stale_block_ids
    req_meta = meta.requests[0]
    assert req_meta.block_ids == current_block_ids
    store_job_id = req_meta.store_job_id
    assert scheduler._pinned_saves[store_job_id][0] == [8, 7, 2, 6]
    assert pool.blocks[0].ref_cnt == 0
    assert pool.blocks[5].ref_cnt == 0
    assert [pool.blocks[i].ref_cnt for i in (21, 22, 23, 24, 25, 26)] == [
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert [pool.blocks[i].ref_cnt for i in (8, 7, 2, 6)] == [1, 1, 1, 1]

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))

    assert [pool.blocks[i].ref_cnt for i in (8, 7, 2, 6)] == [0, 0, 0, 0]


def test_boundary_state_release_is_per_store_job():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    first = scheduler.build_connector_meta(_make_offload_only_output([(1, 7, 16)]))
    second = scheduler.build_connector_meta(_make_offload_only_output([(1, 8, 32)]))
    first_job_id = first.requests[0].store_job_id
    second_job_id = second.requests[0].store_job_id

    scheduler.update_connector_output(_make_worker_output({first_job_id: 1}))

    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 0
    assert scheduler._gpu_block_pool.blocks[8].ref_cnt == 1
    assert first_job_id not in scheduler._pinned_saves
    assert second_job_id in scheduler._pinned_saves


def test_preemption_and_request_id_reuse_do_not_release_inflight_job():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    first = scheduler.build_connector_meta(_make_offload_only_output([(1, 7, 16)]))
    first_job_id = first.requests[0].store_job_id

    scheduler.build_connector_meta(_make_preemption_scheduler_output())
    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 1
    assert first_job_id in scheduler._pinned_saves

    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    second = scheduler.build_connector_meta(_make_offload_only_output([(1, 8, 32)]))
    second_job_id = second.requests[0].store_job_id
    assert second.requests[0].boundary_puts == [(1, 8, 32)]

    scheduler.update_connector_output(_make_worker_output({first_job_id: 1}))
    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 0
    assert scheduler._gpu_block_pool.blocks[8].ref_cnt == 1
    assert second_job_id in scheduler._pinned_saves


def test_boundary_state_never_claimed_without_a_send_thread():
    # A kv_consumer has no store job that can acknowledge the block reference.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler.kv_role = "kv_consumer"
    _register_offload_request(scheduler, prefill_end_tokens=64, num_prompt_tokens=64)
    out = _make_offload_only_output([(1, 7, 16)])

    assert scheduler.build_connector_meta(out).requests == []
    assert scheduler._pinned_saves == {}
    assert scheduler._gpu_block_pool.blocks[7].ref_cnt == 0


def test_resumed_partial_tail_uses_exact_boundary():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    # Resumption replays prompt + previously generated tokens.
    out = _add_pending_partial_tail_request(
        scheduler,
        num_tokens=20,
        block_hashes=[b"h0", b"h1", b"h2", b"h3", b"h4"],
        block_ids=([0, 1],),
    )

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    assert meta.requests[0].boundary_puts == [(1, 7, 12)]
    assert meta.requests[0].num_prompt_tokens == 13
    assert meta.requests[0].prefill_end_tokens == 20
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 0
    assert tracker.has_pending_offload is True


def test_resumed_partial_tail_attached_to_save_keeps_exact_boundary():
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    scheduler._boundary_state_group_ids = frozenset({0})
    request = SimpleNamespace(
        all_token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        num_output_placeholders=0,
        num_prompt_tokens=37,
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
    out.kv_connector_block_state.boundary_state_offloads = {"req-0": [(0, 7, 36)]}

    meta = scheduler.build_connector_meta(out)

    assert len(meta.requests) == 1
    assert meta.requests[0].can_save is True
    assert meta.requests[0].boundary_puts == [(0, 7, 36)]
    assert meta.requests[0].num_prompt_tokens == 37
    assert meta.requests[0].prefill_end_tokens == 48
    # Ordinary saving still covers the full resumed prefill range.
    tracker = scheduler._request_trackers["req-0"]
    assert tracker.num_saved_tokens == 48
    assert tracker.has_pending_offload is True


def test_partial_tail_cow_block_is_referenced_for_the_job():
    # The CoW block a partial-tail offload reads is deliberately kept out of the
    # request block table, so it is absent from ReqMeta.block_ids. The worker
    # DMAs out of it just as asynchronously, so it needs its own reference.
    scheduler = _make_bare_scheduler(hash_block_size=4, enable_partial_hash_hits=True)
    out = _add_pending_partial_tail_request(
        scheduler,
        num_tokens=13,
        block_hashes=[b"h0", b"h1", b"h2"],
        block_ids=([0],),
    )
    pool = scheduler._gpu_block_pool

    meta = scheduler.build_connector_meta(out)

    store_job_id = meta.requests[0].store_job_id
    # It leads the list, as in `pop_blocks_for_free`, so that the reversed free
    # puts it last in eviction priority.
    assert scheduler._pinned_saves[store_job_id][0] == [7]
    assert pool.blocks[0].ref_cnt == 0
    assert pool.blocks[7].ref_cnt == 1

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))
    assert pool.blocks[7].ref_cnt == 0


def test_store_job_blocks_are_released_once_every_rank_reports():
    # Every rank DMAs the job's blocks on its own, so the reference can only be
    # dropped once the last of them reports. Until then the engine has to keep
    # stepping: a completion only reaches the scheduler as worker metadata
    # attached to a step, and a finishing request no longer defers its own free.
    scheduler = _make_bare_scheduler()
    scheduler._num_workers = 2
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(48)),
        block_hashes=[b"h0", b"h1", b"h2"],
        prefill_end_tokens=48,
    )
    pool = scheduler._gpu_block_pool
    assert scheduler.has_pending_push_work() is False

    meta = scheduler.build_connector_meta(
        _make_scheduler_output(scheduled_spec_tokens=None)
    )
    store_job_id = meta.requests[0].store_job_id
    assert pool.blocks[2].ref_cnt == 1
    assert scheduler.has_pending_push_work() is True

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))
    assert pool.blocks[2].ref_cnt == 1
    assert scheduler.has_pending_push_work() is True

    scheduler.update_connector_output(_make_worker_output({store_job_id: 1}))
    assert pool.blocks[2].ref_cnt == 0
    assert scheduler.has_pending_push_work() is False


def test_worker_metadata_aggregates_completions_across_ranks():
    # The engine merges each rank's metadata before the scheduler sees it, so a
    # job that every rank finished in one step arrives as a single count.
    merged = MooncakeStoreWorkerMetadata(completed_saves={1: 1}).aggregate(
        MooncakeStoreWorkerMetadata(completed_saves={1: 1, 2: 1})
    )
    assert merged.completed_saves == {1: 2, 2: 1}
