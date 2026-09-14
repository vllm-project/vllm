# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections import deque
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import KVTransferConfig
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_role": "kv_producer"},
        {"kv_role": "kv_consumer"},
        {"kv_connector_extra_config": {"bidirectional_kv_xfer": True}},
    ],
)
def test_pd_role_requires_both_capabilities_without_bidirectional_transfer(overrides):
    config = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "pd_role": "prefill",
        **overrides,
    }
    with pytest.raises(ValueError, match="pd_role requires NixlConnector"):
        KVTransferConfig(**config)


def test_pd_role_keeps_both_transfer_capabilities():
    config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_both", pd_role="decode"
    )
    assert config.is_kv_producer and config.is_kv_consumer


def _pd_role_engine_core() -> EngineCoreProc:
    core = object.__new__(EngineCoreProc)
    core.batch_queue = None
    core.scheduler = create_scheduler(use_kv_connector=True)
    core.vllm_config = core.scheduler.vllm_config
    core._pd_role = "prefill"
    core._pd_role_epoch = 0
    core._pending_pd_role = None
    return core


def test_pd_role_commit_waits_for_queued_requests_and_kv_release():
    core = _pd_role_engine_core()
    scheduler = core.scheduler
    queued, retained = create_requests(num_requests=2)
    scheduler.add_request(queued)
    retained.status = RequestStatus.FINISHED_STOPPED
    scheduler.requests[retained.request_id] = retained

    status = core.prepare_pd_role("decode", expected_epoch=0)
    assert not status["drained"]
    assert status["waiting"] == 1
    assert scheduler.pause_state == PauseState.UNPAUSED
    with pytest.raises(ValueError, match="still draining"):
        core.commit_pd_role("decode", expected_epoch=0)

    scheduler.finish_requests(queued.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler_output = scheduler.schedule()
    assert not scheduler.has_unfinished_requests()
    assert not core.get_pd_role_status()["drained"]
    with pytest.raises(ValueError, match="still draining"):
        core.commit_pd_role("decode", expected_epoch=0)

    scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_connector_output=KVConnectorOutput(
                finished_sending={retained.request_id}
            ),
        ),
    )
    assert core.get_pd_role_status()["drained"]
    core.batch_queue = deque([(Future(), scheduler_output, Future())])
    with pytest.raises(ValueError, match="still draining"):
        core.commit_pd_role("decode", expected_epoch=0)
    core.batch_queue.clear()
    status = core.commit_pd_role("decode", expected_epoch=0)
    assert status["role"] == "decode"
    assert status["epoch"] == 1
    assert core.commit_pd_role("decode", expected_epoch=0) == status
    with pytest.raises(ValueError, match="committed or stale"):
        core.cancel_pd_role(expected_epoch=0)
    with pytest.raises(ValueError, match="Stale"):
        core.prepare_pd_role("prefill", expected_epoch=0)


def test_pd_role_cancel_preserves_active_role_and_admission():
    core = _pd_role_engine_core()
    request = create_requests(num_requests=1)[0]
    request.kv_transfer_params = {"do_remote_decode": True}
    assert not core._rejects_pd_role_request(request)
    core.prepare_pd_role("decode", expected_epoch=0)
    assert core._rejects_pd_role_request(request)
    status = core.cancel_pd_role(expected_epoch=0)
    assert status["role"] == "prefill" and status["epoch"] == 0
    assert not core._rejects_pd_role_request(request)


@pytest.mark.parametrize(
    "params,pending",
    [
        ({"do_remote_prefill": True}, False),
        ({"do_remote_prefill": True, "do_remote_decode": True}, False),
        ({"do_remote_decode": True}, True),
    ],
    ids=["wrong-role", "ambiguous-role", "pending-transition"],
)
def test_pd_role_rejection_releases_kv_without_duplicate_output(params, pending):
    core = _pd_role_engine_core()
    core._reject_add_in_shutdown = MagicMock(return_value=False)
    core._send_error_outputs_to_client = MagicMock()
    core._send_abort_outputs_to_client = MagicMock()
    connector = core.scheduler.connector
    connector.request_finished = MagicMock(return_value=(False, None))
    request = create_requests(num_requests=1)[0]
    request.kv_transfer_params = params
    if pending:
        core.prepare_pd_role("decode", expected_epoch=0)
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))
    core._send_error_outputs_to_client.assert_called_once_with(
        [request.request_id], request.client_index
    )
    core._send_abort_outputs_to_client.assert_not_called()
    assert request.status == RequestStatus.FINISHED_ABORTED
    connector.request_finished.assert_called_once_with(request, [])
    assert core.scheduler.get_request_counts() == (0, 0)

    cleanup = create_requests(num_requests=1, req_ids=["cleanup"])[0]
    cleanup.kv_transfer_params = params
    cleanup.abort_immediately = True
    core._handle_client_request(EngineCoreRequestType.ADD, (cleanup, 0))
    assert cleanup.status == RequestStatus.FINISHED_ABORTED
    assert connector.request_finished.call_count == 2
    assert core._send_error_outputs_to_client.call_count == 1


def _pd_role_client(num_engines: int = 3) -> DPLBAsyncMPClient:
    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = [bytes([i, 0]) for i in range(num_engines)]
    return client


@pytest.mark.asyncio
async def test_dplb_utility_all_returns_every_rank():
    client = _pd_role_client()

    async def utility(method, *args, engine):
        return {"rank": int.from_bytes(engine, "little"), "epoch": args[0]}

    client._call_utility_async = utility
    statuses = await client.call_utility_all_async("get_pd_role_status", 7)
    assert statuses == [{"rank": rank, "epoch": 7} for rank in range(3)]


@pytest.mark.asyncio
async def test_dplb_utility_all_waits_for_other_ranks_on_error():
    client = _pd_role_client(num_engines=2)
    failed = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def utility(method, *args, engine):
        if engine == client.core_engines[0]:
            failed.set()
            raise ValueError("Stale P/D role epoch")
        await release.wait()
        completed.append(engine)

    client._call_utility_async = utility
    task = asyncio.create_task(
        client.call_utility_all_async("prepare_pd_role", "decode", 0)
    )
    await failed.wait()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(ValueError, match="Stale P/D role epoch"):
        await task
    assert completed == [client.core_engines[1]]
