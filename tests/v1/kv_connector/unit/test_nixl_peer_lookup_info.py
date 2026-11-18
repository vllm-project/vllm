# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL peer lookup: the record a rank reports on `/kv_event_sources`."""

import threading
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NIXL_CONNECTOR_VERSION,
    NixlHandshakePayload,
)
from vllm.v1.engine import PeerLookupInfo

from .test_nixl_peer_pins import (
    COMPAT,
    _init_hash,  # noqa: F401  (autouse fixture)
    make_wired_connector,
    make_wired_scheduler,
)

pytestmark = pytest.mark.cpu_test


def make_reporting_scheduler(**overrides):
    sched = make_wired_scheduler(None, **overrides)
    sched.engine_id = "engine-a"
    sched.side_channel_host = "10.0.0.7"
    sched.side_channel_port = 5602
    sched.transfer_tp_size = 4
    sched.vllm_config.parallel_config = SimpleNamespace(
        decode_context_parallel_size=2, pipeline_parallel_size=3
    )
    sched.vllm_config.kv_transfer_config = SimpleNamespace(kv_connector="NixlConnector")
    sched._handshake_data = {}
    # Do not bind a real listener socket: the thread only signals that it is ready.
    sched._nixl_handshake_listener_t = None
    sched._stop_event = threading.Event()
    sched._nixl_handshake_listener = lambda _data, ready, *_: ready.set()
    sched.set_xfer_handshake_metadata({(0, 0): NixlHandshakePayload(COMPAT, b"x")})
    return sched


def test_peer_lookup_info_values():
    sched = make_reporting_scheduler()
    assert sched.get_peer_lookup_info() == PeerLookupInfo(
        enabled=True,
        engine_id="engine-a",
        side_channel_host="10.0.0.7",
        side_channel_port=5602,
        # The transfer TP size, as in the kv_transfer_params a producer exports.
        tp_size=4,
        dcp_size=2,
        pp_size=3,
        connector="NixlConnector",
        connector_version=str(NIXL_CONNECTOR_VERSION),
        layout_fingerprint=COMPAT,
    )


def test_peer_lookup_info_none_when_lookup_not_served():
    # Lookup off, or push mode (the channel is only enabled in pull mode).
    assert make_reporting_scheduler(peer_lookup=False).get_peer_lookup_info() is None
    sched = make_reporting_scheduler()
    sched._peer_lookup_enabled = True
    sched._lookup.enabled = False
    assert sched.get_peer_lookup_info() is None


def test_peer_lookup_info_none_until_hash_known():
    sched = make_reporting_scheduler()
    sched._compatibility_hash = None
    assert sched.get_peer_lookup_info() is None
    sched._compatibility_hash = ""
    assert sched.get_peer_lookup_info() is None


def test_connector_delegates_to_scheduler():
    sched = make_reporting_scheduler()
    connector = make_wired_connector(sched)
    assert connector.get_peer_lookup_info() == sched.get_peer_lookup_info()
    assert connector.get_peer_lookup_info() is not None

    no_scheduler = object.__new__(NixlPullConnector)
    no_scheduler.connector_scheduler = None
    assert no_scheduler.get_peer_lookup_info() is None


def test_base_connector_reports_nothing():
    assert KVConnectorBase_V1.get_peer_lookup_info(object()) is None
