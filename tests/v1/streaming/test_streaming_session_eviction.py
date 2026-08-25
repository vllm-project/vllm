# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end streaming-session eviction tests (no padding).

Segments are left block-unaligned (no synthetic pad tokens). These tests
drive the real `Scheduler._update_request_as_session` (via a stub) across
many chunk arrivals and verify strict eviction keeps the session leak-free:
the only prompt-vs-owned gap is bounded forward-absorption residue
(<= 2*(block_size-1)), never unbounded orphan drift.
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
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.streaming.eviction import maybe_evict_old_segments
from vllm.v1.streaming.retention import StreamingRetentionParams

pytestmark = pytest.mark.cpu_test

BLOCK = 16


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_managers(block_size: int = BLOCK, num_blocks: int = 64):
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
        max_model_len=4096,
        enable_caching=False,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )
    enc = EncoderCacheManager(cache_size=4096)
    return kv, enc


def _make_streaming_session(
    request_id: str,
    initial_prompt_tokens: list[int],
    retention: StreamingRetentionParams,
    block_size: int = BLOCK,
    hash_fn: Callable = sha256,
) -> Request:
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id=request_id,
        prompt_token_ids=initial_prompt_tokens,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, hash_fn),
    )
    req.streaming_retention = retention
    req.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    return req


class _SchedulerStub:
    """Minimum surface for unbound calls to
    `Scheduler._update_request_as_session` / `_record_streaming_segment`."""

    def __init__(
        self,
        kv_cache_manager: KVCacheManager,
        encoder_cache_manager: EncoderCacheManager,
        block_size: int = BLOCK,
    ):
        self.kv_cache_manager = kv_cache_manager
        self.encoder_cache_manager = encoder_cache_manager
        self.block_size = block_size
        self.log_stats = False
        # The C29 cumulative context-length guard reads this; 4096 matches
        # the file's engine config and stays inert for these sessions.
        self.max_model_len = 4096
        self.num_waiting_for_streaming_input = 1
        self.waiting = FCFSRequestQueue()
        self.prev_step_scheduled_req_ids: set[str] = set()
        self.is_encoder_decoder = False
        self.mm_receiver_cache = None
        # Drifted upstream attrs read by _free_encoder_inputs/_free_request
        # paths reachable from the session-update call.
        self.num_prefill_lookahead = 0
        self._inflight_prefills: set = set()
        from vllm.v1.core.sched.scheduler import Scheduler

        self._record_streaming_segment = (
            lambda session, update, prior_response_start, segment_start_idx: (
                Scheduler._record_streaming_segment(
                    self, session, update, prior_response_start, segment_start_idx
                )
            )
        )


def _apply_session_update(
    stub: _SchedulerStub, session: Request, update: StreamingUpdate
) -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    Scheduler._update_request_as_session(stub, session, update)


def _make_update(
    new_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec] | None = None,
) -> StreamingUpdate:
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    return StreamingUpdate(
        mm_features=mm_features or [],
        prompt_token_ids=new_tokens,
        max_tokens=16,
        arrival_time=0.0,
        sampling_params=sp,
    )


def _retention(**kw) -> StreamingRetentionParams:
    """Build a StreamingRetentionParams for the session-eviction tests.

    `test_unpadded_session_eviction_no_unbounded_orphan_leak` forces
    aggressive eviction with a tiny `max_session_tokens` (16) that
    `StreamingRetentionParams.__post_init__` now rejects (it must be >= the
    minimum; and max_text_tokens must not exceed max_session_tokens). To
    preserve that tiny-budget coverage we build a VALID config and then
    degrade the token budgets via direct attribute assignment after
    construction (the dataclass is mutable, so this bypasses __post_init__).
    """
    base = dict(
        max_video_segments=30,
        max_text_tokens=4000,
        max_session_tokens=4000,
        reprefill_threshold=0.7,
    )
    # Pull out any caller-supplied token budgets (which may be illegally
    # small) and apply them after the valid construction.
    requested_mst = kw.pop("max_session_tokens", base["max_session_tokens"])
    requested_mtt = kw.pop("max_text_tokens", base["max_text_tokens"])
    base.update(kw)
    base["max_session_tokens"] = 4000
    base["max_text_tokens"] = 4000
    retention = StreamingRetentionParams(**base)
    retention.max_session_tokens = requested_mst
    retention.max_text_tokens = requested_mtt
    return retention


def _drive_chunks(stub, session, real_sizes, out_len: int = 2) -> None:
    """Apply a sequence of chunks; between chunks append a short 'output'
    so the next chunk turns it into an assistant_text segment."""
    for chunk_idx, n_real in enumerate(real_sizes):
        update = _make_update(
            new_tokens=[20 + chunk_idx] * n_real,
            mm_features=[
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.dummy(),
                    modality="video",
                    identifier=f"v-{chunk_idx}",
                    mm_position=PlaceholderRange(offset=0, length=n_real),
                )
            ],
        )
        _apply_session_update(stub, session, update)
        session._all_token_ids.extend([99] * out_len)
        session.num_computed_tokens = session.num_prompt_tokens + out_len


def _owned(session: Request) -> int:
    return sum(e - s for s, e in (seg.token_range for seg in session.session_history))


def test_unpadded_session_is_gapless_and_unaligned():
    """With no padding, the per-chunk segments tile [0, num_prompt_tokens)
    with no gaps and no synthetic tokens — and at least one segment is
    block-UNALIGNED, proving nothing was padded to a block boundary."""
    kv, enc = _make_managers()
    stub = _SchedulerStub(kv, enc)
    session = _make_streaming_session(
        "r-unpadded", initial_prompt_tokens=[1] * 13, retention=_retention()
    )
    session.num_computed_tokens = 13
    _drive_chunks(stub, session, [5, 11, 7, 13, 9])

    ranges = [s.token_range for s in session.session_history]
    assert ranges[0][0] == 0
    for (s0, e0), (s1, e1) in zip(ranges, ranges[1:]):
        assert e0 == s1, f"gap/overlap between {(s0, e0)} and {(s1, e1)}"
    assert ranges[-1][1] == session.num_prompt_tokens
    assert _owned(session) == session.num_prompt_tokens, "no orphans during build"

    lengths = [e - s for s, e in ranges]
    assert any(length % BLOCK != 0 for length in lengths), (
        f"expected unpadded (block-unaligned) segments, got lengths {lengths}"
    )


def test_unpadded_session_eviction_no_unbounded_orphan_leak():
    """Build an unpadded session over several chunks, then force aggressive
    eviction. Strict eviction reduces the session toward budget WITHOUT
    PATH C — the only prompt-vs-owned gap is bounded forward-absorption
    residue (<= 2*(block_size-1)), never unbounded orphan drift."""
    kv, enc = _make_managers()
    stub = _SchedulerStub(kv, enc)
    session = _make_streaming_session(
        "r-evict", initial_prompt_tokens=[1] * 13, retention=_retention()
    )
    session.num_computed_tokens = 13
    _drive_chunks(stub, session, [5, 11, 7, 13, 9, 6, 8])

    manager_blocks, _, _ = kv.get_computed_blocks(session)
    kv.allocate_slots(session, session.num_prompt_tokens, 0, manager_blocks)

    before = session.num_prompt_tokens
    tight = _retention(max_video_segments=2, max_session_tokens=16)
    n = maybe_evict_old_segments(session, tight, kv, enc)

    assert n > 0, "expected eviction to fire"
    assert session.num_prompt_tokens < before, "expected progress toward budget"
    gap = session.num_prompt_tokens - _owned(session)
    assert 0 <= gap <= 2 * (BLOCK - 1), (
        f"unbounded orphan leak: num_prompt_tokens="
        f"{session.num_prompt_tokens} owned={_owned(session)} gap={gap}"
    )
    assert len(session.prompt_token_ids) == session.num_prompt_tokens
