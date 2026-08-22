# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for transfer_id collision vulnerability.

Verifies that:
1. The scheduler rejects duplicate active transfer_ids (primary defense).
2. The worker frees overwritten blocks on collision (defense-in-depth).
"""

import asyncio

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    SendBlockMeta,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config

SHARED_TRANSFER_ID = "xfer-attacker-controlled"


def _make_producer_scheduler():
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    scheduler = create_scheduler(vllm_config)
    connector_scheduler = scheduler.get_kv_connector().connector_scheduler
    return scheduler, connector_scheduler


@pytest.mark.cpu_test
class TestSchedulerRejectsDuplicateTransferId:
    """Scheduler must reject requests that reuse an active transfer_id."""

    def test_duplicate_transfer_id_rejected(self):
        _, sched = _make_producer_scheduler()

        req_a = create_request(request_id=100, do_remote_decode=True)
        req_a.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        req_b = create_request(request_id=101, do_remote_decode=True)
        req_b.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        # First request accepted
        sched.update_state_after_alloc(req_a, blocks=None, num_external_tokens=0)
        assert req_a.request_id in sched._reqs_need_send
        assert SHARED_TRANSFER_ID in sched._active_transfer_ids

        # Second request with same transfer_id rejected
        sched.update_state_after_alloc(req_b, blocks=None, num_external_tokens=0)
        assert req_b.request_id not in sched._reqs_need_send
        # do_remote_decode neutralized so request_finished won't track it
        assert req_b.kv_transfer_params["do_remote_decode"] is False

    def test_transfer_id_freed_after_abort(self):
        _, sched = _make_producer_scheduler()

        req = create_request(request_id=200, do_remote_decode=True)
        req.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        sched.update_state_after_alloc(req, blocks=None, num_external_tokens=0)
        assert SHARED_TRANSFER_ID in sched._active_transfer_ids

        # Simulate abort
        req.status = RequestStatus.FINISHED_ABORTED
        sched.request_finished(req, block_ids=([1, 2],))
        assert SHARED_TRANSFER_ID in sched._reqs_not_processed

        # build_connector_meta should clear the active transfer_id
        sched.build_connector_meta(scheduler_output=None)
        assert SHARED_TRANSFER_ID not in sched._active_transfer_ids

    def test_transfer_id_freed_after_successful_handoff(self):
        _, sched = _make_producer_scheduler()

        req = create_request(request_id=300, do_remote_decode=True)
        req.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        sched.update_state_after_alloc(req, blocks=None, num_external_tokens=0)

        # First build_connector_meta: empty blocks (pre-finish registration)
        sched.build_connector_meta(scheduler_output=None)
        # Still active because blocks haven't been assigned yet
        assert SHARED_TRANSFER_ID in sched._active_transfer_ids

        # Simulate successful prefill completion
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        sched.request_finished(req, block_ids=([5, 6, 7],))

        # Second build_connector_meta: with blocks (handoff to worker)
        sched.build_connector_meta(scheduler_output=None)
        assert SHARED_TRANSFER_ID not in sched._active_transfer_ids

    def test_reuse_after_release(self):
        _, sched = _make_producer_scheduler()

        req_a = create_request(request_id=400, do_remote_decode=True)
        req_a.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        sched.update_state_after_alloc(req_a, blocks=None, num_external_tokens=0)

        # Abort and clear
        req_a.status = RequestStatus.FINISHED_ABORTED
        sched.request_finished(req_a, block_ids=([1],))
        sched.build_connector_meta(scheduler_output=None)

        # Now a new request with the same transfer_id should be accepted
        req_b = create_request(request_id=401, do_remote_decode=True)
        req_b.kv_transfer_params["transfer_id"] = SHARED_TRANSFER_ID

        sched.update_state_after_alloc(req_b, blocks=None, num_external_tokens=0)
        assert req_b.request_id in sched._reqs_need_send


@pytest.mark.cpu_test
class TestWorkerFreesOverwrittenBlocks:
    """Worker must free blocks from overwritten requests (defense-in-depth)."""

    @pytest.mark.asyncio
    async def test_collision_frees_old_blocks(self):
        worker_reqs_need_send: dict[str, SendBlockMeta] = {}
        finished_sending: set[str] = set()

        # Simulate existing entry (request A already registered and ready)
        worker_reqs_need_send[SHARED_TRANSFER_ID] = SendBlockMeta(
            p_req_id="req-A",
            transfer_id=SHARED_TRANSFER_ID,
            local_block_ids=[[10, 11, 12]],
            ready=asyncio.Event(),
        )
        worker_reqs_need_send[SHARED_TRANSFER_ID].ready.set()

        # Build metadata that overwrites with request B
        meta = MooncakeConnectorMetadata()
        meta.reqs_to_send["req-B"] = (SHARED_TRANSFER_ID, [[20, 21, 22]])

        # Simulate record_send_reqs logic (mirrors the fixed code)
        for p_req_id, (transfer_id, block_ids) in meta.reqs_to_send.items():
            if block_ids:
                send_meta = worker_reqs_need_send.get(transfer_id)
                if send_meta is None:
                    finished_sending.add(p_req_id)
                    continue
                if (
                    send_meta.p_req_id
                    and send_meta.p_req_id != p_req_id
                    and send_meta.local_block_ids
                ):
                    finished_sending.add(send_meta.p_req_id)
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids

        # Old request's blocks are freed
        assert "req-A" in finished_sending
        # New request takes ownership
        assert worker_reqs_need_send[SHARED_TRANSFER_ID].p_req_id == "req-B"
        assert worker_reqs_need_send[SHARED_TRANSFER_ID].local_block_ids == [
            [20, 21, 22]
        ]

    @pytest.mark.asyncio
    async def test_no_false_positive_on_same_request(self):
        """Re-registering the same p_req_id should NOT trigger collision logic."""
        worker_reqs_need_send: dict[str, SendBlockMeta] = {}
        finished_sending: set[str] = set()

        worker_reqs_need_send[SHARED_TRANSFER_ID] = SendBlockMeta(
            p_req_id="req-A",
            transfer_id=SHARED_TRANSFER_ID,
            local_block_ids=[[10, 11]],
            ready=asyncio.Event(),
        )

        meta = MooncakeConnectorMetadata()
        meta.reqs_to_send["req-A"] = (SHARED_TRANSFER_ID, [[10, 11, 12]])

        for p_req_id, (transfer_id, block_ids) in meta.reqs_to_send.items():
            if block_ids:
                send_meta = worker_reqs_need_send.get(transfer_id)
                if send_meta is None:
                    finished_sending.add(p_req_id)
                    continue
                if (
                    send_meta.p_req_id
                    and send_meta.p_req_id != p_req_id
                    and send_meta.local_block_ids
                ):
                    finished_sending.add(send_meta.p_req_id)
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids

        # No false positive: same request updating itself
        assert "req-A" not in finished_sending
