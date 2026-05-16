# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
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
        prefill_end_tokens=44,
    )


def test_cached_request_with_spec_decode_does_not_save_scheduled_drafts():
    scheduler = _make_bare_scheduler()
    _add_unfinished_request(
        scheduler,
        token_ids=list(range(44)),
        block_hashes=[b"h0", b"h1"],
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
