# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer pod hosts from the serve config when the router sends none."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
    fill_remote_pod_from_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)

from ..utils import create_request

pytestmark = pytest.mark.cpu_test

_EXTRA = {"remote_hosts": ["10.0.0.1", "10.0.0.2"], "remote_dp_size_local": 8}


@pytest.mark.parametrize(
    ("params", "is_producer", "leader"),
    [
        ({"remote_dp_rank": 3, "remote_dp_size": 16}, False, True),
        ({"remote_dp_size": 16}, False, None),
        ({"remote_dp_size": 1}, False, True),
        ({"remote_dp_rank": 3}, True, None),
        ({"remote_dp_rank": 3, "is_request_leader": False}, False, False),
    ],
    ids=["pinned", "unpinned", "single-rank-peer", "producer", "preset"],
)
def test_fill_marks_the_leader_only_when_unambiguous(params, is_producer, leader):
    fill_remote_pod_from_config(params, _EXTRA, is_producer)
    assert params["remote_hosts"] == _EXTRA["remote_hosts"]
    assert params["remote_dp_size_local"] == 8
    assert params.get("is_request_leader") == leader


def test_router_hosts_win_over_serve_config():
    params = {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 3}
    fill_remote_pod_from_config(params, _EXTRA, is_producer=False)
    assert params == {"remote_hosts": ["10.9.9.9"], "remote_dp_rank": 3}


@pytest.mark.parametrize("finished_before_alloc", [True, False])
@pytest.mark.parametrize(
    ("dp_rank", "host", "port"), [(3, "10.0.0.1", 7003), (9, "10.0.0.2", 7001)]
)
def test_release_reaches_the_peer_pod(finished_before_alloc, dp_rank, host, port):
    """The router sent only the master's address; global DP rank 9 is local
    rank 1 on the second pod. Either path releases every prefill TP rank."""
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.is_producer = False
    scheduler.kv_transfer_config = SimpleNamespace(kv_connector_extra_config=_EXTRA)
    scheduler.tp_size = 2
    scheduler._has_mamba = False
    scheduler._is_kv_master = False
    scheduler._reqs_need_recv = {}
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    released = []
    scheduler._send_transfer_release = lambda transfer_id, host, port: (  # type: ignore[method-assign]
        released.append((transfer_id, host, port))
    )
    request = create_request(request_id=7, do_remote_prefill=True)
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "transfer_id": "xfer-7",
        "remote_dp_rank": dp_rank,
        "remote_dp_size": 16,
        "remote_host": "10.0.0.1",
        "remote_notify_port": 7000,
    }

    if finished_before_alloc:
        scheduler.request_finished(request, block_ids=[])
    else:
        # A full local prefix hit leaves nothing to write, so the leader releases.
        scheduler.update_state_after_alloc(request, MagicMock(), 0)

    assert released == [("xfer-7", host, port), ("xfer-7", host, port + 1)]
