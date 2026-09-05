# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for streaming re-prefill at position-range threshold.

Two test surfaces:

  - `should_trigger_reprefill` (pure check, no side effects). Tested
    directly with a synthetic Request.

  - `Scheduler._reprefill_streaming_session` (does the clearing +
    re-queue). Tested with a real `KVCacheManager` + `EncoderCacheManager`
    plus a minimal scheduler-stub that exposes the attributes the
    helper reads (`kv_cache_manager`, `encoder_cache_manager`,
    `waiting`).

The helper is called as an unbound method, mirroring how
`test_eviction.py` calls `Scheduler._record_streaming_segment(None, ...)`.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.streaming.reprefill import should_trigger_reprefill
from vllm.v1.streaming.retention import (
    HistorySegment,
    StreamingRetentionParams,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_managers(block_size: int = 16, num_blocks: int = 64):
    kv = KVCacheManager(
        KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        ),
        max_model_len=1024,
        enable_caching=False,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    enc = EncoderCacheManager(cache_size=4096)
    return kv, enc


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    block_size: int = 16,
    hash_fn: Callable = sha256,
) -> Request:
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )
    return req


class _SchedulerStub:
    """Minimum-viable surface for unbound calls to
    `Scheduler._reprefill_streaming_session`. The helper reads
    `kv_cache_manager`, `encoder_cache_manager`, `waiting`, and
    `prev_step_scheduled_req_ids`."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        encoder_cache_manager: EncoderCacheManager,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.encoder_cache_manager = encoder_cache_manager
        self.waiting = FCFSRequestQueue()
        # `_reprefill_streaming_session` clears the request's prior-step
        # scheduling bookkeeping so the resumed PREEMPTED request
        # doesn't trip the `assert not scheduled_in_prev_step` in
        # `_make_cached_request_data`. The stub must expose this
        # attribute or the helper crashes with AttributeError.
        self.prev_step_scheduled_req_ids: set[str] = set()
        # `_free_encoder_inputs` consults `self.is_encoder_decoder` to
        # decide between the VLM and enc-dec release paths. Tests of the
        # gate-disabled fallback path reach this code; set False so they
        # exercise the VLM branch.
        self.is_encoder_decoder = False
        self.log_stats = False
        # Drifted upstream attrs read by _free_request / encoder paths.
        self.num_prefill_lookahead = 0
        self._inflight_prefills: set = set()


def _bind_reprefill_helper(stub: _SchedulerStub):
    """Return a callable invoking `Scheduler._reprefill_streaming_session`
    as a bound method on the stub. The helper doesn't reference any
    attribute outside what the stub exposes.

    `_reprefill_streaming_session` now calls
    `self._reset_streaming_position_state(request)` (scheduler.py refactor:
    the position/mRoPE reset shared with preemption-resume was extracted
    into its own method). Bind the REAL method onto the stub so the reset
    genuinely clears `_mrope_positions`, `max_cached_position`, and
    `pending_evicted_token_ranges` exactly as production does (rather than a
    hand-written no-op that wouldn't satisfy the clears-position-state
    assertions)."""
    import types

    from vllm.v1.core.sched.scheduler import Scheduler

    stub._reset_streaming_position_state = types.MethodType(
        Scheduler._reset_streaming_position_state, stub
    )

    def _call(request: Request, discard_next_sample: bool = False) -> None:
        # discard_next_sample=False models the folded-chunk case (the
        # prompt ends with an unanswered chunk, so the first sample is a
        # genuine caption token); these tests exercise the reset
        # mechanics, which are identical for both values.
        Scheduler._reprefill_streaming_session(
            stub, request, discard_next_sample=discard_next_sample
        )

    return _call


# ---------------------------------------------------------------------------
# should_trigger_reprefill (pure check)
# ---------------------------------------------------------------------------


def test_trigger_fires_above_threshold_not_below():
    """`should_trigger_reprefill` returns True only when
    `max_cached_position` exceeds `reprefill_threshold * model_max_position`."""
    req = _make_request("r-trigger", [9] * 8)
    retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    model_max_position = 1000

    req.max_cached_position = 699  # just below threshold (700)
    assert should_trigger_reprefill(req, retention, model_max_position) is False

    req.max_cached_position = 700  # exactly at threshold — strict >
    assert should_trigger_reprefill(req, retention, model_max_position) is False

    req.max_cached_position = 701  # just above threshold
    assert should_trigger_reprefill(req, retention, model_max_position) is True


def test_trigger_disabled_when_threshold_at_or_above_one():
    """A threshold of 1.0 (or greater) means "disable re-prefill"."""
    req = _make_request("r-disabled", [9] * 8)
    retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=1.0,
    )
    req.max_cached_position = 10_000_000  # arbitrarily large
    assert should_trigger_reprefill(req, retention, model_max_position=1_000) is False


# ---------------------------------------------------------------------------
# _reprefill_streaming_session (the side-effecting helper)
# ---------------------------------------------------------------------------


def _build_request_with_kv_and_history(
    request_id: str, total_tokens: int = 80, block_size: int = 16
):
    """Build a request with KV blocks allocated, mRoPE positions
    populated, and a non-trivial session_history + mm_features. Returns
    `(request, scheduler_stub, _call_reprefill)`."""
    kv, enc = _make_managers(block_size=block_size, num_blocks=64)
    req = _make_request(request_id, [9] * total_tokens, block_size=block_size)
    req._mrope_positions = [(i, i, i) for i in range(total_tokens)]
    req.max_cached_position = total_tokens - 1
    req.num_prompt_tokens = total_tokens
    req.num_computed_tokens = total_tokens

    # Populate session_history with one pinned anchor, one video segment,
    # and one assistant_text caption. Mirrors what the streaming branch
    # would produce after several chunks. age_chunks > 0 on every segment:
    # the eviction phases only pick victims with age_chunks > 0 (finding
    # #0 — the just-appended, not-yet-computed segment is never evicted),
    # and this fixture models already-aged prior chunks, so the mvs=0
    # forced-eviction tests can actually drop the video segment. Ages
    # descend oldest-highest to preserve oldest-first eviction order.
    req.session_history = [
        HistorySegment(
            segment_type="user_text",
            token_range=(0, 16),
            pinned=True,
            age_chunks=4,
        ),
        HistorySegment(
            segment_type="assistant_text",
            token_range=(16, 32),
            age_chunks=3,
        ),
        HistorySegment(
            segment_type="video",
            token_range=(32, 64),
            mm_item_id="v0",
            age_chunks=2,
        ),
        HistorySegment(
            segment_type="assistant_text",
            token_range=(64, 80),
            age_chunks=1,
        ),
    ]
    req.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="v0",
            mm_position=PlaceholderRange(offset=32, length=32),
        )
    ]
    manager_blocks, _, _ = kv.get_computed_blocks(req)
    kv.allocate_slots(req, total_tokens, 0, manager_blocks)

    # Simulate an in-flight pending_evicted_token_ranges entry that hasn't
    # yet been drained to the worker (so we can assert it's cleared).
    req.pending_evicted_token_ranges.append((40, 56))

    # Make sure the request looks like a streaming session.
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    req.reprefill_count = 0

    stub = _SchedulerStub(kv, enc)
    return req, stub, _bind_reprefill_helper(stub)


def test_reprefill_preserves_content_clears_position_state():
    """`_reprefill_streaming_session` must clear K cache, mRoPE state,
    num_computed_tokens, and pending evictions — but leave the content
    (token ids, session_history, mm_features) untouched."""
    req, stub, call_reprefill = _build_request_with_kv_and_history("r-content-survives")

    # Snapshot content-bearing state before re-prefill.
    all_token_ids_before = list(req._all_token_ids)
    session_history_before = list(req.session_history)
    mm_features_before = list(req.mm_features)
    retention_before = req.streaming_retention

    call_reprefill(req)

    # Cleared fields.
    assert req._mrope_positions == []
    assert req.max_cached_position == -1
    assert req.num_computed_tokens == 0
    assert req.pending_evicted_token_ranges == []
    assert req.status == RequestStatus.WAITING

    # Surviving content.
    assert req._all_token_ids == all_token_ids_before
    assert req.session_history == session_history_before
    assert req.mm_features == mm_features_before
    assert req.streaming_retention is retention_before


def test_reprefill_prepends_request_to_waiting_queue():
    """After re-prefill, the request must be at the HEAD of the
    waiting queue (so the next scheduler step picks it up first)."""
    req, stub, call_reprefill = _build_request_with_kv_and_history("r-queued-head")

    # Pre-populate the waiting queue with another fake request so we
    # can verify "prepend" puts ours at the head.
    other = _make_request("r-other", [1] * 8)
    stub.waiting.add_request(other)
    assert list(stub.waiting) == [other]

    call_reprefill(req)

    queued = list(stub.waiting)
    assert queued[0] is req, [r.request_id for r in queued]
    assert queued[1] is other


def test_reprefill_increments_count():
    """`reprefill_count` increments by one each call."""
    req, stub, call_reprefill = _build_request_with_kv_and_history("r-counter")
    assert req.reprefill_count == 0

    call_reprefill(req)
    assert req.reprefill_count == 1

    # Set up state for a second re-prefill (positions repopulated, etc.).
    req.max_cached_position = 100
    req.num_computed_tokens = 100
    call_reprefill(req)
    assert req.reprefill_count == 2


def test_reprefill_frees_kv_blocks_and_preserves_encoder_cache():
    """The helper must release the request's KV blocks but MUST NOT
    touch the encoder cache. Under the persistent-encoder-cache scheme
    (Request.uses_persistent_encoder_cache), entries created during
    chunk-N's prefill must survive re-prefill so subsequent re-prefill
    prefill chunks reuse them without re-running the vision encoder.

    The earlier force-evict behavior was defensive against an
    mm_features reassignment bug (now fixed via in-place mutation in
    `eviction.py`); this test verifies the defensive logic is gone and
    entries persist as designed.
    """
    req, stub, call_reprefill = _build_request_with_kv_and_history("r-cache-persists")
    kv = stub.kv_cache_manager
    enc = stub.encoder_cache_manager
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)
    # Pre-state: entry is in `cached` with this request as the only
    # referencer.
    assert enc.cached.get("v0") == {req.request_id}
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == cache_size - 32

    free_q = kv.block_pool.free_block_queue
    free_before = free_q.num_free_blocks

    call_reprefill(req)

    # KV blocks returned to the pool (strict: a zero-free regression must fail).
    free_after = free_q.num_free_blocks
    assert free_after > free_before, (free_before, free_after)
    # Encoder cache entry survives: still in `cached`, still referenced
    # by this request, slot accounting unchanged. Worker-side
    # encoder_cache[mm_hash] is similarly untouched (verified
    # indirectly by the absence of an entry in `freed`).
    assert enc.cached.get("v0") == {req.request_id}
    assert "v0" not in enc.freeable
    assert "v0" not in enc.freed
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == cache_size - 32


def test_reprefill_persists_freeable_encoder_entry():
    """If the encoder entry happens to already be on `freeable` at
    re-prefill time (e.g., a prior code path explicitly called
    `free_encoder_input`), re-prefill leaves it alone — the entry is
    still recoverable via `check_and_update_cache` on the next
    scheduling pass, which is exactly what re-prefill prefill needs.
    """
    req, stub, call_reprefill = _build_request_with_kv_and_history(
        "r-cache-freeable-survives"
    )
    enc = stub.encoder_cache_manager
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)
    enc.free_encoder_input(req, input_id=0)
    # Pre-state: cached key still present with empty refs; freeable
    # holds the entry; num_freeable_slots restored.
    assert "v0" in enc.freeable
    assert enc.cached["v0"] == set()
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == cache_size

    call_reprefill(req)

    # Entry survives in both `cached` and `freeable`. Slot accounting
    # unchanged from the post-free_encoder_input state.
    assert enc.cached.get("v0") == set()
    assert "v0" in enc.freeable
    assert "v0" not in enc.freed
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == cache_size


def test_reprefill_works_with_status_running_or_waiting():
    """The helper must NOT assert status == RUNNING (the way
    `_preempt_request` does). Verify it accepts WAITING and
    WAITING_FOR_STREAMING_REQ as starting states — those are the
    states the streaming path actually fires from."""
    for starting_status in (
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
        RequestStatus.RUNNING,
    ):
        req, stub, call_reprefill = _build_request_with_kv_and_history(
            f"r-status-{starting_status.name.lower()}"
        )
        req.status = starting_status
        call_reprefill(req)
        # Always ends in WAITING regardless of where it started. The
        # request goes back through the scheduled_new_reqs path so the
        # worker re-runs `_update_streaming_request` (refreshing
        # `num_prompt_tokens`/`output_token_ids` for the discard mask).
        assert req.status == RequestStatus.WAITING


# ---------------------------------------------------------------------------
# Persistent encoder cache (encoder-cache-lifetime extension)
# ---------------------------------------------------------------------------


def test_uses_persistent_encoder_cache_predicate():
    """Request.uses_persistent_encoder_cache() returns True for
    streaming sessions with re-prefill enabled, False otherwise. This
    predicate gates `_free_encoder_inputs` and is the single source of
    truth for the encoder-cache-lifetime extension."""
    req = _make_request("r-predicate", [9] * 8)
    # No streaming retention → False.
    assert req.uses_persistent_encoder_cache() is False

    # Streaming retention with threshold < 1.0 → True.
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    assert req.uses_persistent_encoder_cache() is True

    # Streaming retention with threshold == 1.0 (re-prefill disabled)
    # → False (falls back to old per-chunk freeing).
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=1.0,
    )
    assert req.uses_persistent_encoder_cache() is False


def test_encoder_cache_persists_across_chunks_for_streaming():
    """Encoder cache entries for a streaming request with persistent
    encoder cache enabled must NOT move to `freeable` after the chunk
    that scattered them completes. Verifies that `_free_encoder_inputs`
    is gated for these sessions and that `evict_segment` (called from
    retention) is the sole release path.

    We model this by calling `_free_encoder_inputs` directly on a
    request whose mm_feature has been "consumed" by the chunk's
    forward pass (i.e., `num_computed_tokens` exceeds the placeholder's
    end position). With the gate active the entry stays referenced;
    with the gate inactive it would transition to `freeable`.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    req, stub, _ = _build_request_with_kv_and_history(
        "r-cache-persists-chunk", total_tokens=80
    )
    enc = stub.encoder_cache_manager
    enc.allocate(req, input_id=0)
    # mm_feature is at offset=32, length=32 → ends at 64. Set
    # num_computed_tokens past that to simulate "chunk done with this
    # mm_feature".
    req.num_computed_tokens = 80

    # Streaming + re-prefill enabled (from fixture).
    assert req.uses_persistent_encoder_cache() is True

    Scheduler._free_encoder_inputs(stub, req)

    # Gate active: entry remains in cached with our ref, NOT in
    # freeable, NOT in freed.
    assert enc.cached.get("v0") == {req.request_id}
    assert "v0" not in enc.freeable
    assert "v0" not in enc.freed


def test_encoder_cache_freeing_falls_back_when_reprefill_disabled():
    """If `reprefill_threshold == 1.0` (re-prefill disabled), the gate
    in `_free_encoder_inputs` falls through to the original behavior:
    once a chunk has consumed its mm_feature, the encoder cache entry
    is released to `freeable`."""
    from vllm.v1.core.sched.scheduler import Scheduler

    req, stub, _ = _build_request_with_kv_and_history(
        "r-cache-fallback", total_tokens=80
    )
    enc = stub.encoder_cache_manager
    enc.allocate(req, input_id=0)
    req.num_computed_tokens = 80
    # Disable re-prefill — gate should fall through to old behavior.
    assert req.streaming_retention is not None
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=req.streaming_retention.max_video_segments,
        max_session_tokens=req.streaming_retention.max_session_tokens,
        reprefill_threshold=1.0,
    )
    assert req.uses_persistent_encoder_cache() is False

    Scheduler._free_encoder_inputs(stub, req)

    # Gate inactive: entry moved to freeable; ref dropped.
    assert enc.cached.get("v0") == set()
    assert "v0" in enc.freeable


def test_segment_eviction_releases_encoder_cache():
    """Under the persistent-encoder-cache scheme, the intra-session
    release path for streaming entries is `evict_segment` in
    `vllm/v1/streaming/eviction.py`. Since the multi-session OOM fix it
    must PHYSICALLY evict a zero-ref entry, not just park it in
    `freeable`: the entry leaves `cached` and `freeable`, its mm_hash
    reaches the worker exactly once via `get_freed_mm_hashes` (so the
    GPU tensor is dropped), and slot accounting is fully restored."""
    from vllm.v1.streaming.eviction import evict_segment

    req, stub, _ = _build_request_with_kv_and_history(
        "r-segment-evict", total_tokens=80
    )
    enc = stub.encoder_cache_manager
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)
    assert enc.cached.get("v0") == {req.request_id}
    assert enc.num_free_slots == cache_size - 32

    # Find the video segment.
    video_seg = next(s for s in req.session_history if s.mm_item_id == "v0")

    evict_segment(req, video_seg, stub.kv_cache_manager, enc)

    # Physically evicted: gone from both `cached` and `freeable`.
    assert "v0" not in enc.cached
    assert "v0" not in enc.freeable
    # Slot accounting restored to the pre-allocate value; with
    # `freeable` empty the invariant num_freeable_slots ==
    # num_free_slots + sum(freeable.values()) collapses to equality.
    assert enc.num_free_slots == cache_size
    assert enc.num_freeable_slots == cache_size
    # The worker is told to drop the GPU tensor exactly once; the freed
    # list has drain semantics (second call returns nothing).
    assert enc.get_freed_mm_hashes() == ["v0"]
    assert enc.get_freed_mm_hashes() == []
    # Idempotence: re-evicting the same hash is a pop-miss no-op and
    # must not double-credit num_free_slots.
    assert enc.evict_unreferenced("v0") is False
    assert enc.num_free_slots == cache_size
    assert enc.num_freeable_slots == cache_size


def test_free_and_evict_releases_persistent_entries():
    """`free_and_evict` (the session close/abort path dispatched by
    `Scheduler._free_request` for persistent-encoder-cache requests)
    must drop the request's ref AND physically evict the now
    unreferenced entry in one call: it leaves `cached`/`freeable`,
    lands in `freed`, and the slots are restored."""
    req, stub, _ = _build_request_with_kv_and_history(
        "r-free-and-evict", total_tokens=80
    )
    enc = stub.encoder_cache_manager
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)
    assert req.uses_persistent_encoder_cache() is True

    enc.free_and_evict(req)

    assert "v0" not in enc.cached
    assert "v0" not in enc.freeable
    assert enc.get_freed_mm_hashes() == ["v0"]
    assert enc.get_freed_mm_hashes() == []
    assert enc.num_free_slots == cache_size
    assert enc.num_freeable_slots == cache_size


def test_evict_segment_multi_ref_entry_survives():
    """If another request still references the mm_hash, `evict_segment`
    must only drop THIS session's ref: the entry stays in `cached` with
    the survivor's request_id and is neither parked in `freeable` nor
    physically freed (`evict_unreferenced` early-returns on the
    `freeable.pop` miss)."""
    from vllm.v1.streaming.eviction import evict_segment

    req, stub, _ = _build_request_with_kv_and_history(
        "r-multi-ref-evict", total_tokens=80
    )
    enc = stub.encoder_cache_manager
    cache_size = enc.cache_size
    enc.allocate(req, input_id=0)

    # A second request referencing the same cached entry.
    other = _make_request("r-multi-ref-survivor", [9] * 8)
    other.mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem.dummy(),
            modality="video",
            identifier="v0",
            mm_position=PlaceholderRange(offset=0, length=32),
        )
    ]
    assert enc.check_and_update_cache(other, 0) is True
    assert enc.cached["v0"] == {req.request_id, other.request_id}

    video_seg = next(s for s in req.session_history if s.mm_item_id == "v0")
    evict_segment(req, video_seg, stub.kv_cache_manager, enc)

    # Survivor keeps the entry: still cached, not freeable, not freed.
    assert enc.cached.get("v0") == {other.request_id}
    assert not enc.freeable
    assert enc.get_freed_mm_hashes() == []
    # Slots stay charged for the still-live entry.
    assert enc.num_free_slots == cache_size - 32
    assert enc.num_freeable_slots == cache_size - 32


def test_free_request_dispatches_free_and_evict():
    """`Scheduler._free_request` must route persistent-encoder-cache
    sessions (close/abort) to `free_and_evict` and everything else to
    plain `free`. Uses the unbound-method-on-stub pattern with a mocked
    encoder cache manager so only the dispatch is under test."""
    import types
    from unittest.mock import Mock

    from vllm.v1.core.sched.scheduler import Scheduler

    for persistent in (True, False):
        req, stub, _ = _build_request_with_kv_and_history(
            f"r-free-dispatch-{persistent}"
        )
        if not persistent:
            req.streaming_retention = None
        assert req.uses_persistent_encoder_cache() is persistent
        req.status = RequestStatus.FINISHED_ABORTED

        # Minimal extra surface `_free_request` reads beyond the stub.
        stub.encoder_cache_manager = Mock()
        stub._connector_finished = lambda request: (False, None)
        stub._free_blocks = types.MethodType(Scheduler._free_blocks, stub)
        stub.finished_req_ids = set()
        stub.finished_req_ids_dict = None
        stub.requests = {req.request_id: req}
        # Drifted upstream surface reached by _free_request.
        stub.ec_connector = None
        stub.defer_block_free = False
        stub._pause_state = None
        stub._free_request_blocks = types.MethodType(
            Scheduler._free_request_blocks, stub
        )

        Scheduler._free_request(stub, req)

        if persistent:
            stub.encoder_cache_manager.free_and_evict.assert_called_once_with(req)
            stub.encoder_cache_manager.free.assert_not_called()
        else:
            stub.encoder_cache_manager.free.assert_called_once_with(req)
            stub.encoder_cache_manager.free_and_evict.assert_not_called()
        # Blocks freed and the request dropped from the table either way.
        assert req.request_id in stub.finished_req_ids
        assert req.request_id not in stub.requests


def test_encoder_cache_size_auto_sized_from_mm_limits():
    """compute_mm_encoder_budget should raise encoder_cache_size when
    the per-modality concurrent-items cap implies a larger working
    set than the default (= max_num_batched_tokens). This is the
    auto-sizing path that keeps persistent encoder cache feasible
    without forcing users to manually bump max_num_batched_tokens."""
    from types import SimpleNamespace

    from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget

    scheduler_config = SimpleNamespace(
        disable_chunked_mm_input=False,
        max_num_batched_tokens=2048,
        max_num_encoder_input_tokens=2048,
        encoder_cache_size=2048,
    )
    mm_max_toks_per_item = {"video": 413}

    # Without limits — falls back to default.
    _, cache_size_default = compute_mm_encoder_budget(
        scheduler_config, mm_max_toks_per_item
    )
    assert cache_size_default == 2048

    # With limit_mm_per_prompt {video: 60} (sports-config default of
    # max_video_segments * 2), required floor = 60 * 413 * 1.3 = 32214.
    # Must be >= that.
    _, cache_size_streaming = compute_mm_encoder_budget(
        scheduler_config, mm_max_toks_per_item, {"video": 60}
    )
    assert cache_size_streaming >= 60 * 413, (
        f"cache_size={cache_size_streaming} should accommodate 60 frames "
        f"× 413 tokens = {60 * 413}"
    )
    # Has the 30% headroom.
    assert cache_size_streaming >= (60 * 413 * 13) // 10


# ---------------------------------------------------------------------------
# is_reprefill IPC signal: distinguishes re-prefill from eviction
# ---------------------------------------------------------------------------


def test_eviction_alone_does_not_set_reprefill_flag():
    """Intra-session eviction must NOT set `pending_reprefill`. Eviction
    is a structured shrink (the worker slices mRoPE preserving surviving
    positions' values); re-prefill is a full reset (positions recomputed
    from 0 because the K cache was freed). Conflating them broke RoPE
    consistency on surviving tokens once the sliding window started
    dropping segments."""
    from vllm.v1.streaming.eviction import maybe_evict_old_segments

    req, stub, _ = _build_request_with_kv_and_history(
        "r-evict-not-reprefill", total_tokens=80
    )
    # Force eviction by setting max_video_segments to 0 — the one
    # video segment in session_history will be dropped. The constructor
    # guard now rejects max_video_segments < 1, so build a valid config
    # and then degrade it by direct attribute assignment (the dataclass
    # is mutable) to keep exercising the mvs=0 eviction path.
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    req.streaming_retention.max_video_segments = 0
    assert req.pending_reprefill is False

    maybe_evict_old_segments(
        req,
        req.streaming_retention,
        stub.kv_cache_manager,
        stub.encoder_cache_manager,
    )

    # Eviction populated `pending_evicted_token_ranges` but DID NOT
    # touch `pending_reprefill`. The worker will see evicted ranges
    # in the next NewRequestData and slice mRoPE accordingly; the
    # reset path stays untouched.
    assert req.pending_evicted_token_ranges, "expected eviction to populate ranges"
    assert req.pending_reprefill is False


def test_new_request_data_drains_reprefill_flag():
    """`NewRequestData.from_request` drains `request.pending_reprefill`
    onto `is_reprefill`, then clears the request-side flag so subsequent
    re-schedules of the same chunk don't re-fire the worker reset."""
    from vllm.v1.core.sched.output import NewRequestData

    req = _make_request("r-drain-flag", [9] * 8)
    req.pending_reprefill = True

    data = NewRequestData.from_request(req, block_ids=([],))
    assert data.is_reprefill is True
    assert req.pending_reprefill is False

    # Second drain returns False — flag stays cleared until the next
    # `_reprefill_streaming_session` sets it again.
    data2 = NewRequestData.from_request(req, block_ids=([],))
    assert data2.is_reprefill is False


def test_eviction_then_drain_carries_ranges_not_reprefill():
    """End-to-end check: after eviction, `NewRequestData` carries the
    evicted ranges but `is_reprefill` stays False. This is the contract
    the worker relies on to slice (not reset) mRoPE positions."""
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.streaming.eviction import maybe_evict_old_segments

    req, stub, _ = _build_request_with_kv_and_history(
        "r-evict-then-drain", total_tokens=80
    )
    # `_build_request_with_kv_and_history` seeds `pending_evicted_token_ranges`
    # with (40, 56) to test re-prefill's clearing behavior — drain that
    # here so the test only sees ranges produced by THIS test's eviction.
    req.pending_evicted_token_ranges.clear()
    # Constructor guard rejects max_video_segments < 1; build a valid
    # config and degrade it post-construction to keep exercising the
    # mvs=0 forced-eviction path (the dataclass is mutable).
    req.streaming_retention = StreamingRetentionParams(
        max_video_segments=30,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    req.streaming_retention.max_video_segments = 0

    maybe_evict_old_segments(
        req,
        req.streaming_retention,
        stub.kv_cache_manager,
        stub.encoder_cache_manager,
    )
    assert req.pending_evicted_token_ranges

    data = NewRequestData.from_request(req, block_ids=([],))
    assert data.evicted_token_ranges, "expected ranges to propagate"
    assert data.is_reprefill is False, (
        "eviction must not set is_reprefill — that would cause the "
        "worker to discard surviving mRoPE positions and recompute "
        "them from 0, breaking RoPE consistency with cached K's"
    )
    # Drained on the request side too.
    assert req.pending_evicted_token_ranges == []


def test_reprefill_clears_pending_ranges_and_sets_flag():
    """Combined re-prefill scenario: when eviction populated pending
    ranges and then re-prefill fires in the same handle-stopped pass,
    the helper clears the ranges (re-prefill subsumes them: all KV is
    freed) and sets the explicit re-prefill flag. The worker then sees
    is_reprefill=True with empty evicted_token_ranges and resets mRoPE."""
    from vllm.v1.core.sched.output import NewRequestData

    req, _, call_reprefill = _build_request_with_kv_and_history(
        "r-reprefill-clears-ranges"
    )
    # Fixture pre-seeds a range; verify it's actually there before re-prefill.
    assert req.pending_evicted_token_ranges == [(40, 56)]

    call_reprefill(req)

    assert req.pending_evicted_token_ranges == []
    assert req.pending_reprefill is True

    data = NewRequestData.from_request(req, block_ids=([],))
    assert data.is_reprefill is True
    assert not data.evicted_token_ranges  # None or empty list
