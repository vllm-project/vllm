# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NixlPushConnector (scheduler + worker).

These tests cover the end-to-end mechanics of the push design without
requiring a real NIXL agent or network:

* Scheduler stages D registrations on ``update_state_after_alloc`` and
  P finished blocks on ``request_finished``.
* ``build_connector_meta`` drains them onto
  ``meta.push_registrations`` / ``meta.push_finished_blocks``.
* ``has_pending_push_work`` reports True/False over the lifecycle.
* ``update_connector_output`` clears state on ``finished_sending`` and
  ``finished_recving``.
* The worker matches D registrations against P finished blocks (both
  scenario directions) and forwards non-PUSH_REG NIXL notifs to the main
  thread's ``_get_new_notifs``.
* ``get_finished`` enqueues evictions for the writer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import make_nixl_push_scheduler

# ----------------------------------------------------------------- #
#  Helpers / fakes                                                   #
# ----------------------------------------------------------------- #


def _make_request(
    *,
    request_id: str,
    is_d_side: bool = True,
    remote_engine_id: str = "prefill-engine",
    remote_request_id: str | None = None,
    remote_host: str = "10.0.0.1",
    remote_port: int = 5601,
    tp_size: int = 1,
    finished: bool = True,
) -> MagicMock:
    """Build a minimal Request mock used by request_finished."""
    from vllm.v1.request import RequestStatus

    req = MagicMock()
    req.request_id = request_id
    req.num_computed_tokens = 64

    if is_d_side:
        # D-side request: do_remote_prefill=True -> prefill on a remote P.
        params: dict[str, Any] = {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": remote_request_id or f"prefill-{request_id}",
            "remote_host": remote_host,
            "remote_port": remote_port,
            "tp_size": tp_size,
        }
    else:
        # P-side request: do_remote_decode=True (we are the prefiller).
        params = {
            "do_remote_prefill": False,
            "do_remote_decode": True,
        }
    req.kv_transfer_params = params
    req.status = (
        RequestStatus.FINISHED_LENGTH_CAPPED if finished else RequestStatus.RUNNING
    )
    return req


class _BlocksMock:
    """Minimal stand-in for ``KVCacheBlocks`` used in update_state_after_alloc."""

    def __init__(self, block_ids: tuple[list[int], ...]):
        self._block_ids = block_ids

    def get_unhashed_block_ids_all_groups(self) -> tuple[list[int], ...]:
        return self._block_ids


def _stub_sw_clipping(scheduler) -> None:
    """Make ``get_sw_clipped_blocks`` a passthrough so tests don't need
    the full sliding-window machinery."""
    scheduler.get_sw_clipped_blocks = lambda block_ids: block_ids


# ----------------------------------------------------------------- #
#  Scheduler-side tests                                              #
# ----------------------------------------------------------------- #


class TestPushScheduler:
    def test_d_side_update_state_after_alloc_stages_registration(self):
        """D scheduler stashes registration data + arms watchdog deadline."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = _make_request(request_id="req-d-1")
        blocks = _BlocksMock(block_ids=([10, 11, 12],))

        sched.update_state_after_alloc(request, blocks, num_external_tokens=48)

        assert request.request_id in sched._push_pending_registrations
        reg = sched._push_pending_registrations[request.request_id]
        # ``request_id`` is D's own vLLM request id; plus our own (D) coords.
        assert reg["request_id"] == request.request_id
        assert reg["decode_engine_id"] == sched.engine_id
        assert reg["decode_host"] == sched.side_channel_host
        assert reg["decode_port"] == sched.side_channel_port
        assert reg["local_block_ids"] == ([10, 11, 12],)
        assert reg["remote_engine_id"] == "prefill-engine"

        # Watchdog deadline set in the future.
        deadline = sched._push_registration_deadlines[request.request_id]
        assert deadline > time.perf_counter()
        # do_remote_prefill flipped off so the request isn't reprocessed.
        assert request.kv_transfer_params["do_remote_prefill"] is False
        # Tracked as awaiting a recv.
        assert request.request_id in sched._reqs_need_recv

    def test_p_side_request_finished_stages_blocks(self):
        """P scheduler pushes blocks into both _finished_request_blocks (lease)
        and _newly_finished_push_blocks (metadata for next step)."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = _make_request(request_id="req-p-1", is_d_side=False)
        block_ids = ([20, 21, 22, 23],)

        delay, ret_params = sched.request_finished(request, block_ids)

        assert delay is True
        assert ret_params is not None
        assert ret_params["do_remote_prefill"] is True
        assert ret_params["do_remote_decode"] is False
        assert request.request_id in sched._finished_request_blocks
        assert request.request_id in sched._newly_finished_push_blocks
        assert request.request_id in sched._reqs_need_send  # lease armed

    def test_build_connector_meta_drains_both_sides(self):
        """meta.push_registrations and meta.push_finished_blocks are filled
        from the staging dicts and the staging dicts are cleared."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        # Stage one D registration and one P finished entry.
        d_req = _make_request(request_id="req-d-9")
        sched.update_state_after_alloc(
            d_req, _BlocksMock(([1, 2, 3],)), num_external_tokens=48
        )
        p_req = _make_request(request_id="req-p-9", is_d_side=False)
        sched.request_finished(p_req, ([4, 5, 6],))

        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock(
            req_ids=[], resumed_req_ids=set()
        )

        # Patch parent build_connector_meta so we don't have to set up
        # all the base scheduler plumbing.
        with patch.object(
            sched.__class__.__mro__[1],
            "build_connector_meta",
            return_value=NixlConnectorMetadata(),
        ):
            meta = sched.build_connector_meta(scheduler_output)

        assert isinstance(meta, NixlConnectorMetadata)
        assert "req-d-9" in meta.push_registrations
        assert "req-p-9" in meta.push_finished_blocks
        # Staging dicts cleared.
        assert sched._push_pending_registrations == {}
        assert sched._newly_finished_push_blocks == {}
        # Lease bookkeeping kept until the WRITE completes.
        assert "req-p-9" in sched._finished_request_blocks

    def test_has_pending_push_work_lifecycle(self):
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        assert sched.has_pending_push_work() is False

        # P finished blocks waiting for WRITE completion.
        p_req = _make_request(request_id="req-p-7", is_d_side=False)
        sched.request_finished(p_req, ([0, 1],))
        assert sched.has_pending_push_work() is True

        # Drain via build_connector_meta - lease still pending until WRITE.
        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock(
            req_ids=[], resumed_req_ids=set()
        )
        with patch.object(
            sched.__class__.__mro__[1],
            "build_connector_meta",
            return_value=NixlConnectorMetadata(),
        ):
            sched.build_connector_meta(scheduler_output)
        # Lease is pending until WRITE completes -> still True.
        assert sched.has_pending_push_work() is True

        # Simulate WRITE completion via update_connector_output.
        sched.update_connector_output(
            KVConnectorOutput(
                finished_sending={"req-p-7"},
                finished_recving=set(),
                invalid_block_ids=set(),
            )
        )
        assert sched.has_pending_push_work() is False

    def test_update_connector_output_clears_lease_and_watchdog(self):
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        d_req = _make_request(request_id="req-d-x")
        sched.update_state_after_alloc(
            d_req, _BlocksMock(([1, 2],)), num_external_tokens=32
        )
        p_req = _make_request(request_id="req-p-x", is_d_side=False)
        sched.request_finished(p_req, ([3, 4],))

        sched.update_connector_output(
            KVConnectorOutput(
                finished_sending={"req-p-x"},
                finished_recving={"req-d-x"},
                invalid_block_ids=set(),
            )
        )
        assert "req-p-x" not in sched._finished_request_blocks
        assert "req-d-x" not in sched._push_registration_deadlines

    def test_registration_watchdog_expires(self, caplog):
        """Stale D registrations whose deadline has passed are dropped at
        ``build_connector_meta`` time."""
        # Watchdog logs a WARNING when it drops the stale entry; that's
        # what this test is verifying, so silence it in the test report.
        caplog.set_level(
            logging.CRITICAL,
            logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler"),
        )
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        d_req = _make_request(request_id="req-d-stale")
        sched.update_state_after_alloc(
            d_req, _BlocksMock(([7, 8],)), num_external_tokens=32
        )
        # Force the deadline into the past.
        sched._push_registration_deadlines[d_req.request_id] = time.perf_counter() - 1.0

        scheduler_output = MagicMock()
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_cached_reqs = MagicMock(
            req_ids=[], resumed_req_ids=set()
        )
        with patch.object(
            sched.__class__.__mro__[1],
            "build_connector_meta",
            return_value=NixlConnectorMetadata(),
        ):
            meta = sched.build_connector_meta(scheduler_output)

        assert d_req.request_id not in sched._push_registration_deadlines
        assert d_req.request_id not in sched._push_pending_registrations
        assert d_req.request_id not in meta.push_registrations


# ----------------------------------------------------------------- #
#  Worker-side tests                                                 #
# ----------------------------------------------------------------- #


class _StubWriterWorker(NixlPushConnectorWorker):
    """Construct a worker without invoking ``__init__`` so we can drive
    the matching/notif logic without bringing up NIXL or torch."""

    @classmethod
    def fresh(cls) -> _StubWriterWorker:
        w = object.__new__(cls)

        # Push-specific state managed by NixlPushConnectorWorker.
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            ReqId,
            TransferHandle,
        )

        w._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        w._sending_transfers_lock = threading.Lock()
        w._push_finished_blocks = {}
        w._pending_d_registrations = {}
        w._reg_send_inbox = queue.Queue()
        w._finished_blocks_inbox = queue.Queue()
        w._pending_completion_notifs = queue.Queue()
        w._evict_finished_inbox = queue.Queue()
        w._push_writer_wake = threading.Event()
        w._push_writer_stop = threading.Event()
        w._push_writer_thread = None

        # Base worker fields touched by start_load_kv / _get_new_notifs.
        w._recving_metadata = {}
        w._recving_transfers = defaultdict(list)
        w._reqs_to_process = set()
        w._reqs_to_send = {}
        w.consumer_notification_counts_by_req = defaultdict(int)
        w.tp_rank = 0
        w.world_size = 1
        w.engine_id = "test-decode-engine"
        w._remote_agents = {}

        # Track _do_start_push_kv invocations.
        calls: list[tuple[str, Any, dict[str, Any]]] = []
        w.start_push_calls = calls
        return w

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids,
        registration_data: dict[str, Any],
    ) -> None:  # pragma: no cover - exercised through tests
        # Track the call instead of issuing real WRITEs.
        self.start_push_calls.append((request_id, local_block_ids, registration_data))


def _registration_data(
    request_id: str,
    *,
    decode_engine_id: str = "decode-engine",
    decode_host: str = "10.0.0.2",
    decode_port: int = 5602,
    decode_tp_size: int = 1,
    local_block_ids=((100, 101, 102),),
    remote_engine_id: str = "prefill-engine",
    remote_host: str = "10.0.0.1",
    remote_port: int = 5601,
    remote_tp_size: int = 1,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "decode_engine_id": decode_engine_id,
        "decode_host": decode_host,
        "decode_port": decode_port,
        "decode_tp_size": decode_tp_size,
        "local_block_ids": local_block_ids,
        "remote_engine_id": remote_engine_id,
        "remote_host": remote_host,
        "remote_port": remote_port,
        "remote_tp_size": remote_tp_size,
    }


class TestPushWriterMatching:
    def test_handle_push_reg_matches_existing_finished_blocks(self):
        """PUSH_REG arrives second (P finished first): match + fire."""
        w = _StubWriterWorker.fresh()
        # P had already finished; its blocks were stashed via metadata.
        w._push_finished_blocks["req-A"] = ([200, 201, 202],)

        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(
            _registration_data("req-A")
        )
        w._handle_push_reg_notif(notif)

        assert len(w.start_push_calls) == 1
        rid, blocks, reg = w.start_push_calls[0]
        assert rid == "req-A"
        assert blocks == ([200, 201, 202],)
        assert reg["decode_engine_id"] == "decode-engine"
        # Finished blocks consumed.
        assert "req-A" not in w._push_finished_blocks
        assert w._pending_d_registrations == {}

    def test_handle_push_reg_stashes_when_no_finished_blocks_yet(self):
        """PUSH_REG arrives first (D registered first): stash, no fire."""
        w = _StubWriterWorker.fresh()

        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(
            _registration_data("req-B")
        )
        w._handle_push_reg_notif(notif)

        assert len(w.start_push_calls) == 0
        assert "req-B" in w._pending_d_registrations

    def test_handle_push_reg_matches_after_stripping_random_suffix(self):
        """P and D assign the same logical request the same
        ``cmpl-<uuid>-<index>`` but different per-engine random suffixes;
        the writer should still match P's finished blocks via the
        suffix-stripping fallback in ``_pop_matching_finished_blocks``.
        """
        w = _StubWriterWorker.fresh()
        # Same base id + completion index; differ only in the trailing
        # ``-<8 hex>`` randomization suffix.
        p_id = "cmpl-12345678-aaaa-bbbb-cccc-1234567890ab-0-aaaaaaaa"
        d_id = "cmpl-12345678-aaaa-bbbb-cccc-1234567890ab-0-bbbbbbbb"
        # Sanity: same base id under the helper used by the connector.
        assert get_base_request_id(p_id) == get_base_request_id(d_id)

        w._push_finished_blocks[p_id] = ([1, 2, 3],)
        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(_registration_data(d_id))
        w._handle_push_reg_notif(notif)

        # Suffix-stripped fallback matched and fired.
        assert len(w.start_push_calls) == 1
        assert w.start_push_calls[0][0] == p_id
        assert p_id not in w._push_finished_blocks

    def test_handle_push_reg_drops_malformed(self, caplog):
        # The writer logs WARNING/ERROR when it sees these bad payloads;
        # that's the desired behavior, so suppress the noise from test
        # output rather than letting it look like a failure.
        caplog.set_level(
            logging.CRITICAL,
            logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker"),
        )
        w = _StubWriterWorker.fresh()
        # Missing request_id -> should drop without raising.
        bad = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode({"decode_engine_id": "x"})
        w._handle_push_reg_notif(bad)
        assert w._pending_d_registrations == {}
        assert w.start_push_calls == []

        # Undecodable payload also dropped.
        w._handle_push_reg_notif(PUSH_REG_NOTIF_PREFIX + b"\xff\xff\xff")
        assert w.start_push_calls == []


class TestPushWriterStartLoadKv:
    def test_finished_blocks_inbox_matches_stashed_registration(self):
        """Run the writer-loop's finished-blocks drain against a
        pre-populated _pending_d_registrations entry."""
        w = _StubWriterWorker.fresh()
        w._pending_d_registrations["req-C"] = _registration_data("req-C")

        # Simulate start_load_kv enqueuing finished blocks.
        w._finished_blocks_inbox.put(("req-C", ([10, 11, 12],)))

        # Drain like the writer loop does.
        while True:
            try:
                rid, blocks = w._finished_blocks_inbox.get_nowait()
            except queue.Empty:
                break
            matched = w._pop_matching_registration(rid)
            if matched is not None:
                w._do_start_push_kv(rid, blocks, matched)
            else:
                w._push_finished_blocks[rid] = blocks

        assert len(w.start_push_calls) == 1
        assert w.start_push_calls[0][0] == "req-C"
        assert "req-C" not in w._pending_d_registrations

    def test_start_load_kv_enqueues_to_writer(self):
        """``start_load_kv`` should hand registrations + finished blocks
        to the writer queues without doing matching itself."""
        w = _StubWriterWorker.fresh()
        # Stub heartbeats to a no-op; tests don't exercise the heartbeat
        # path here.
        w._send_heartbeats = lambda metadata: None
        # Stub logical-to-kernel mapping used by reqs_to_recv.
        w._logical_to_kernel_block_ids = lambda x: x

        meta = NixlConnectorMetadata()
        meta.push_registrations = {
            "req-D": _registration_data("req-D"),
        }
        meta.push_finished_blocks = {
            "req-E": ([5, 6, 7],),
        }

        w.start_load_kv(meta)

        # Things are queued for the writer; nothing fires yet.
        assert w._reg_send_inbox.qsize() == 1
        assert w._finished_blocks_inbox.qsize() == 1
        assert w._push_writer_wake.is_set()
        assert w.start_push_calls == []


class TestPushWriterNotifs:
    def test_get_new_notifs_processes_forwarded_completion_notif(self):
        """Non-PUSH_REG notifs forwarded by the writer thread are drained
        on the engine main thread inside ``_get_new_notifs``."""
        w = _StubWriterWorker.fresh()
        # Pretend the writer thread already forwarded a completion notif
        # for a request whose KV is being received.
        request_id = "req-recv-1"
        w._recving_metadata[request_id] = MagicMock(pp_size=1)
        # Compose the standard completion notif: req_id:tp_size.
        notif_msg = f"{request_id}:1".encode()
        w._pending_completion_notifs.put(notif_msg)

        # transfer_topo is consulted only for the producer-side path; we
        # make it a MagicMock because the D-side branch returns early.
        w.transfer_topo = MagicMock()

        notified = w._get_new_notifs()

        # Notif consumed; D-side just touches _recving_transfers.
        assert notified == set()
        assert request_id in w._recving_transfers

    def test_get_finished_evicts_completed_state(self):
        """``get_finished`` should enqueue evictions and wake the writer."""
        w = _StubWriterWorker.fresh()

        # Stub the base ``get_finished`` to return one done_sending entry.
        # Patch via the MRO's parent class.
        with patch.object(
            NixlPushConnectorWorker.__mro__[1],
            "get_finished",
            return_value=({"req-done"}, set()),
        ):
            done_sending, done_recving = w.get_finished()

        assert "req-done" in done_sending
        assert done_recving == set()
        # Eviction enqueued for the writer.
        evicted = []
        while True:
            try:
                evicted.append(w._evict_finished_inbox.get_nowait())
            except queue.Empty:
                break
        assert evicted == ["req-done"]
        assert w._push_writer_wake.is_set()


# ----------------------------------------------------------------- #
#  Negative / error-path tests                                       #
# ----------------------------------------------------------------- #


class TestPushSchedulerNegative:
    """Failure / no-op paths on the scheduler side."""

    def test_update_state_after_alloc_no_kv_transfer_params_is_noop(self):
        """Requests without kv_transfer_params must not register anything."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = MagicMock()
        request.request_id = "req-no-params"
        request.kv_transfer_params = None

        sched.update_state_after_alloc(
            request, _BlocksMock(([1, 2, 3],)), num_external_tokens=64
        )

        assert sched._push_pending_registrations == {}
        assert sched._push_registration_deadlines == {}
        assert sched._reqs_need_recv == {}

    def test_update_state_after_alloc_zero_external_tokens_does_not_register(self):
        """num_external_tokens=0 should not stage a D registration."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = _make_request(request_id="req-zero-ext")
        sched.update_state_after_alloc(
            request, _BlocksMock(([1, 2, 3],)), num_external_tokens=0
        )

        assert sched._push_pending_registrations == {}
        assert sched._push_registration_deadlines == {}

    def test_request_finished_unfinished_status_does_not_stage(self):
        """If a request is still RUNNING, request_finished must not stash
        blocks for the worker (no push needed)."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = _make_request(
            request_id="req-running", is_d_side=False, finished=False
        )

        delay, ret = sched.request_finished(request, ([1, 2, 3],))

        assert delay is False
        assert ret is None
        assert sched._finished_request_blocks == {}
        assert sched._newly_finished_push_blocks == {}

    def test_request_finished_empty_blocks_does_not_arm_lease(self):
        """Empty block-id groups should still complete cleanly without
        arming the lease/finished maps."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        request = _make_request(request_id="req-empty", is_d_side=False)
        delay, ret = sched.request_finished(request, ((),))

        assert delay is False
        assert ret is not None
        assert "req-empty" not in sched._finished_request_blocks
        assert "req-empty" not in sched._newly_finished_push_blocks
        assert "req-empty" not in sched._reqs_need_send

    def test_update_connector_output_unknown_request_is_noop(self):
        """Idempotent cleanup: clearing a request that was never staged
        must not raise or mutate other state."""
        sched = make_nixl_push_scheduler()
        _stub_sw_clipping(sched)

        # Stage one real request to ensure it's NOT touched.
        live = _make_request(request_id="req-live", is_d_side=False)
        sched.request_finished(live, ([1],))

        sched.update_connector_output(
            KVConnectorOutput(
                finished_sending={"unknown-1"},
                finished_recving={"unknown-2"},
                invalid_block_ids=set(),
            )
        )

        # Live entry untouched.
        assert "req-live" in sched._finished_request_blocks


class TestPushWriterNegative:
    """Failure / drop / idempotence paths in the writer thread."""

    def test_pop_matching_registration_returns_none_when_empty(self):
        w = _StubWriterWorker.fresh()
        assert w._pop_matching_registration("nope") is None

    def test_pop_matching_finished_blocks_returns_none_when_empty(self):
        w = _StubWriterWorker.fresh()
        assert w._pop_matching_finished_blocks("nope") is None

    def test_pop_matching_registration_no_match_when_base_ids_differ(self):
        """A registration whose base id (after stripping the random suffix)
        does NOT match the lookup request_id must not be popped."""
        w = _StubWriterWorker.fresh()
        # Two unrelated requests: different base UUIDs, so stripping the
        # trailing ``-<8 hex>`` suffix still yields different base ids.
        unrelated_d = "cmpl-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa-0-11111111"
        lookup = "cmpl-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb-0-22222222"
        assert get_base_request_id(unrelated_d) != get_base_request_id(lookup)

        w._pending_d_registrations[unrelated_d] = _registration_data(unrelated_d)
        result = w._pop_matching_registration(lookup)
        assert result is None
        # Original entry untouched.
        assert unrelated_d in w._pending_d_registrations

    def test_handle_push_reg_with_non_dict_payload_is_dropped(self, caplog):
        """msgpack-encoded non-dict payload (e.g. a list) should be
        dropped without raising."""
        caplog.set_level(
            logging.CRITICAL,
            logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker"),
        )
        w = _StubWriterWorker.fresh()
        bad = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode([1, 2, 3])
        w._handle_push_reg_notif(bad)
        assert w._pending_d_registrations == {}
        assert w.start_push_calls == []

    def test_handle_push_reg_with_non_string_request_id_is_dropped(self, caplog):
        """request_id must be a str; integers, None, etc. must drop."""
        caplog.set_level(
            logging.CRITICAL,
            logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker"),
        )
        w = _StubWriterWorker.fresh()
        for bogus_rid in (123, None, 4.5, b"bytes-not-str"):
            payload = _registration_data("placeholder")
            payload["request_id"] = bogus_rid  # type: ignore[assignment]
            notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(payload)
            w._handle_push_reg_notif(notif)
        assert w._pending_d_registrations == {}
        assert w.start_push_calls == []

    def test_handle_push_reg_idempotent_for_same_request_id(self):
        """Receiving the same PUSH_REG twice (e.g. P retries after a
        flake) keeps the entry staged exactly once and never fires."""
        w = _StubWriterWorker.fresh()
        notif = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(
            _registration_data("req-dup")
        )
        w._handle_push_reg_notif(notif)
        w._handle_push_reg_notif(notif)
        assert "req-dup" in w._pending_d_registrations
        assert len(w._pending_d_registrations) == 1
        assert w.start_push_calls == []

    def test_get_finished_enqueues_eviction_for_each_done_request(self):
        """``get_finished`` must enqueue an eviction for every request
        in ``done_sending`` so the writer can drop stale matching state.
        Unlike the happy-path test, this verifies the *cardinality*: N
        completed requests -> N evictions, in order."""
        w = _StubWriterWorker.fresh()
        with patch.object(
            NixlPushConnectorWorker.__mro__[1],
            "get_finished",
            return_value=({"req-1", "req-2", "req-3"}, set()),
        ):
            done_sending, _ = w.get_finished()
        assert done_sending == {"req-1", "req-2", "req-3"}

        evicted: list[str] = []
        while True:
            try:
                evicted.append(w._evict_finished_inbox.get_nowait())
            except queue.Empty:
                break
        assert sorted(evicted) == ["req-1", "req-2", "req-3"]

    def test_get_finished_with_no_completions_does_not_enqueue_eviction(self):
        """If there's nothing newly done, no eviction should be enqueued.
        The wake event IS still set because ``get_finished`` always wakes
        the writer to drain notifs."""
        w = _StubWriterWorker.fresh()
        with patch.object(
            NixlPushConnectorWorker.__mro__[1],
            "get_finished",
            return_value=(set(), set()),
        ):
            done_sending, done_recving = w.get_finished()
        assert done_sending == set()
        assert done_recving == set()
        assert w._evict_finished_inbox.qsize() == 0
        # Wake set so the writer drains NIXL notifs even when idle.
        assert w._push_writer_wake.is_set()

    def test_get_new_notifs_unknown_request_is_logged_and_skipped(self, caplog):
        """A completion notif for a request the worker doesn't know
        about should be logged but not crash."""
        caplog.set_level(
            logging.CRITICAL,
            logger=("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker"),
        )
        w = _StubWriterWorker.fresh()
        w.transfer_topo = MagicMock()
        # Forward a completion notif for an unknown request_id.
        w._pending_completion_notifs.put(b"never-heard-of-you:1")

        notified = w._get_new_notifs()
        assert notified == set()
        # Did not register anywhere.
        assert "never-heard-of-you" not in w._recving_transfers

    def test_start_load_kv_with_empty_metadata_is_noop(self):
        """Empty metadata must not wake the writer or enqueue anything."""
        w = _StubWriterWorker.fresh()
        w._send_heartbeats = lambda metadata: None
        w._logical_to_kernel_block_ids = lambda x: x

        meta = NixlConnectorMetadata()
        w.start_load_kv(meta)

        assert w._reg_send_inbox.qsize() == 0
        assert w._finished_blocks_inbox.qsize() == 0
        # Wake should NOT be set if there was nothing to push.
        assert not w._push_writer_wake.is_set()

    def test_get_new_notifs_extends_lease_on_heartbeat(self):
        """``HB:`` notifs forwarded by the writer thread must extend the
        leases of tracked P-side requests on the engine main thread, and
        ignore request IDs that aren't being tracked."""
        w = _StubWriterWorker.fresh()
        w.transfer_topo = MagicMock()
        # _handle_heartbeat reads ``self._lease_extension`` (set in the
        # real ``__init__``).
        w._lease_extension = 10

        # Tracked P-side requests with a lease about to expire.
        old_expiry = time.perf_counter() - 5.0
        w._reqs_to_send["req-a"] = old_expiry
        w._reqs_to_send["req-b"] = old_expiry

        # Forwarded heartbeat covers a tracked request, an unknown one,
        # and another tracked one.
        w._pending_completion_notifs.put(b"HB:req-a,req-unknown,req-b")

        notified = w._get_new_notifs()
        assert notified == set()

        # Tracked leases were renewed strictly forward in time.
        now = time.perf_counter()
        for rid in ("req-a", "req-b"):
            assert w._reqs_to_send[rid] > old_expiry
            # New expiry must be roughly now + _lease_extension.
            assert w._reqs_to_send[rid] >= now
        # Unknown request must not be inserted by the heartbeat path.
        assert "req-unknown" not in w._reqs_to_send


# ----------------------------------------------------------------- #
#  Pipeline-parallel producer (push-mode PP-disagg)                  #
# ----------------------------------------------------------------- #


class TestPushPipelineParallel:
    """PP-sharded producer: per-stage completion counting, pp_size plumbing,
    and remote-region slicing."""

    def test_completion_waits_for_one_notif_per_pp_stage(self):
        """Each PP stage WRITEs its own layers and sends one notif; D must
        collect pp_size notifs before reporting the recv done."""
        w = _StubWriterWorker.fresh()
        w.transfer_topo = MagicMock()
        request_id = "req-pp-2"
        w._recving_metadata[request_id] = MagicMock(pp_size=2)
        notif = f"{request_id}:1".encode()

        # First stage: counted, not yet done.
        w._pending_completion_notifs.put(notif)
        assert w._get_new_notifs() == set()
        assert request_id not in w._recving_transfers
        assert w.consumer_notification_counts_by_req[request_id] == 1

        # Second (final) stage: now reported done.
        w._pending_completion_notifs.put(notif)
        assert w._get_new_notifs() == set()
        assert request_id in w._recving_transfers
        assert request_id not in w.consumer_notification_counts_by_req

    def test_req_meta_reads_pp_size_from_kv_transfer_params(self):
        """D learns the producer's pp_size from kv_transfer_params (forwarded
        by the proxy) and defaults to 1 when absent."""
        metadata = NixlConnectorMetadata()
        params = {
            "remote_block_ids": ([0],),
            "remote_engine_id": "p-engine",
            "remote_request_id": "p-req",
            "remote_host": "localhost",
            "remote_port": 1234,
            "tp_size": 1,
            "pp_size": 2,
        }
        metadata.add_new_req_to_recv("req", ([0],), params)
        assert metadata.reqs_to_recv["req"].pp_size == 2

        params.pop("pp_size")
        metadata.add_new_req_to_recv("req-default", ([0],), params)
        assert metadata.reqs_to_recv["req-default"].pp_size == 1

    def test_add_remote_agent_slices_remote_regions_to_local_pp_window(self):
        """With PP>1 the producer registered regions for the full model, but
        this worker holds only a contiguous layer slice; add_remote_agent
        trims the remote region list to [offset : offset + num_local_regions]
        before building descriptors. We stop right after the slice via a
        sentinel on the next collaborator call."""
        block_len = 4096 * 16
        w = _StubWriterWorker.fresh()  # seeds writer-thread state for teardown
        w.pp_size = 2
        w._remote_region_offset = 2  # this worker owns layers [2, 4)
        w.block_len_per_layer = [block_len, block_len]  # 2 local layers
        w.nixl_wrapper = MagicMock()

        class _StopAfterSlice(RuntimeError):
            pass

        w.transfer_topo = MagicMock()
        w.transfer_topo.register_remote_engine.side_effect = _StopAfterSlice()

        meta = NixlAgentMetadata(
            engine_id="p-engine",
            agent_metadata=b"agent",
            kv_caches_base_addr=[10, 11, 12, 13],  # full 4-layer model
            device_id=0,
            num_blocks=4,
            block_lens=[block_len] * 4,
            kv_cache_layout="HND",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="FLASH_ATTN",
            physical_blocks_per_logical_kv_block=1,
        )

        with pytest.raises(_StopAfterSlice):
            w.add_remote_agent(meta, remote_tp_rank=0, remote_tp_size=1)

        # Sliced down to this worker's layer window (regions [2:4]).
        assert meta.kv_caches_base_addr == [12, 13]
        assert meta.block_lens == [block_len, block_len]


class TestPushWriterMlaReplication:
    """MLA latent KV is replicated across D's TP ranks, so when D_TP > P_TP
    (``tp_ratio < 0``) the producer must *WRITE* the latent into every D rank
    it handshook -- not write one and merely notify the rest, which would
    leave the un-written ranks decoding against stale KV."""

    @staticmethod
    def _mla_worker_writing_to(d_ranks):
        from types import SimpleNamespace

        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
            TPMapping,
        )

        engine_id = "decode-engine"
        w = _StubWriterWorker.fresh()
        w.use_mla = True
        w.nixl_wrapper = MagicMock()
        w.transfer_topo = MagicMock()
        w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
            remote_tp_size=len(d_ranks),
            remote_block_size=16,
            remote_physical_blocks_per_logical=1,
        )
        # D_TP > P_TP => negative ratio (one P rank feeds |ratio| D ranks).
        w.transfer_topo.tp_ratio.return_value = -len(d_ranks)
        # The tp-mapping collapses MLA to a single source rank (correct for
        # the pull/read direction); the push path must fan it back out.
        w.tp_mappings = {
            engine_id: TPMapping(
                source_ranks_per_group=((0,),),
                all_source_ranks=(0,),
                rank_to_attention_slot={0: 0},
                rank_offset_factor=0,
            )
        }
        w._logical_to_remote_kernel_block_ids = lambda block_ids, ratio: block_ids
        w.dst_xfer_side_handles = {engine_id: {r: 1000 + r for r in d_ranks}}
        w.src_xfer_handles_by_block_size = {16: 2000}
        w._remote_agents = {engine_id: {(0, r): f"agent-{r}" for r in d_ranks}}
        return w, engine_id

    def test_mla_hetero_tp_writes_every_d_rank(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            RemoteMeta,
            ReqMeta,
        )

        w, engine_id = self._mla_worker_writing_to(d_ranks=(0, 1))

        written: list[int] = []

        def _fake_xfer(**kw):
            rank = kw["read_spec"].remote_rank
            written.append(rank)
            return 1000 + rank  # in-flight handle, as the real method returns

        w._xfer_blocks = _fake_xfer

        meta = ReqMeta(
            local_block_ids=([100, 101],),
            local_physical_block_ids=([100, 101],),
            tp_size=1,
            remote=RemoteMeta(
                block_ids=([200, 201],),
                host="",
                port=0,
                engine_id=engine_id,
                request_id="d-req",
            ),
        )

        w._xfer_blocks_for_req(req_id="p-req", meta=meta)

        # Every handshook D rank must receive a real WRITE of the latent...
        assert sorted(written) == [0, 1]
        # ...and no rank may be fobbed off with a bare completion notif.
        assert w.nixl_wrapper.send_notif.call_count == 0
        # All of the request's WRITE handles must be tracked together, so the
        # engine thread never sees a partial set and double-frees the request.
        assert sorted(w._sending_transfers["p-req"]) == [1000, 1001]
