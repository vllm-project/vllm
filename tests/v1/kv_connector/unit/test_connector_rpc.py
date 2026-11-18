# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generic connector control RPC (RFC #51639).

Covers the `bytes -> bytes` control channel at the layers below the HTTP
transport:

* `KVConnectorBase_V1.on_rpc` default (not-implemented sentinel).
* `MultiConnector.on_rpc` first-non-None dispatch, empty-bytes success, and
  exception propagation.
* `Scheduler.invoke_kv_connector` guarding on `self.connector`.
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.core.sched.scheduler import Scheduler


class _StubConnector(KVConnectorBase_V1):
    """Minimal concrete connector that skips the config-bound __init__.

    Only `on_rpc` behavior is under test, so the abstract methods are
    stubbed and no vllm_config is required.
    """

    def __init__(self):
        pass

    def start_load_kv(self, *args, **kwargs):
        pass

    def wait_for_layer_load(self, *args, **kwargs):
        pass

    def save_kv_layer(self, *args, **kwargs):
        pass

    def wait_for_save(self, *args, **kwargs):
        pass

    def get_num_new_matched_tokens(self, *args, **kwargs):
        return (0, False)

    def update_state_after_alloc(self, *args, **kwargs):
        pass

    def build_connector_meta(self, *args, **kwargs):
        return None

    def request_finished_all_groups(self, *args, **kwargs):
        return (False, None)

    def aggregate(self, *args, **kwargs):
        return None


class _EchoConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        return b"echo:" + payload


class _EmptyAckConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        return b""


class _RaisingConnector(_StubConnector):
    def on_rpc(self, payload: bytes) -> bytes:
        raise RuntimeError("boom")


def test_base_on_rpc_returns_none():
    # The base-class default signals "not implemented".
    assert _StubConnector().on_rpc(b"anything") is None


def test_on_rpc_override_returns_bytes():
    assert _EchoConnector().on_rpc(b"ping") == b"echo:ping"


def _make_multi(connectors) -> MultiConnector:
    mc = object.__new__(MultiConnector)
    mc._connectors = connectors
    return mc


def test_multi_connector_returns_first_non_none():
    mc = _make_multi([_StubConnector(), _EchoConnector()])
    assert mc.on_rpc(b"x") == b"echo:x"


def test_multi_connector_empty_bytes_is_success_not_decline():
    # An empty b"" from an earlier child is a valid reply and must win over a
    # later child that would have returned non-empty bytes.
    mc = _make_multi([_EmptyAckConnector(), _EchoConnector()])
    assert mc.on_rpc(b"x") == b""


def test_multi_connector_all_decline_returns_none():
    mc = _make_multi([_StubConnector(), _StubConnector()])
    assert mc.on_rpc(b"x") is None


def test_multi_connector_exception_propagates_without_fallthrough():
    # The raising child comes first; the exception must propagate rather than
    # falling through to the echo child.
    mc = _make_multi([_RaisingConnector(), _EchoConnector()])
    with pytest.raises(RuntimeError, match="boom"):
        mc.on_rpc(b"x")


def _make_scheduler(connector) -> Scheduler:
    sched = object.__new__(Scheduler)
    sched.connector = connector
    return sched


def test_scheduler_invoke_no_connector_returns_none():
    assert _make_scheduler(None).invoke_kv_connector(b"x") is None


def test_scheduler_invoke_delegates_to_connector():
    assert (
        _make_scheduler(_EchoConnector()).invoke_kv_connector(b"ping") == b"echo:ping"
    )


def test_scheduler_invoke_propagates_connector_exception():
    with pytest.raises(RuntimeError, match="boom"):
        _make_scheduler(_RaisingConnector()).invoke_kv_connector(b"x")
