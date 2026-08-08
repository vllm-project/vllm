# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generic asynchronous connector-init readiness handshake."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    ConnectorInitState,
    KVConnectorBase_V1,
    KVConnectorRankInitStatus,
    KVConnectorRole,
)

pytestmark = pytest.mark.cpu_test


class _FakeConnector(KVConnectorBase_V1):
    """Minimal concrete connector: only the init-state hook is meaningful."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state: ConnectorInitState | None = None

    def get_connector_init_state(self) -> ConnectorInitState | None:
        return self.state

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass


def _make_connector(
    role: KVConnectorRole, rank: int = 0, world_size: int = 2
) -> _FakeConnector:
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=True),
        parallel_config=SimpleNamespace(rank=rank, world_size=world_size),
    )
    return _FakeConnector(vllm_config, role, SimpleNamespace())


# ---------------------------------------------------------------------------
# KVConnectorRankInitStatus.aggregate — pure union semantics
# ---------------------------------------------------------------------------


def test_rank_init_status_aggregate_unions_ready_ranks():
    """aggregate() must union ready_ranks from both statuses."""
    a = KVConnectorRankInitStatus(ready_ranks={0})
    b = KVConnectorRankInitStatus(ready_ranks={1})
    assert a.aggregate(b).ready_ranks == {0, 1}


# ---------------------------------------------------------------------------
# is_connector_ready
# ---------------------------------------------------------------------------


def test_is_connector_ready_true_when_async_init_disabled():
    """Without enable_async_init(), the connector is ready immediately."""
    connector = _make_connector(KVConnectorRole.WORKER)
    assert connector.is_connector_ready() is True


def test_worker_is_connector_ready_tracks_own_init_state():
    """A worker's readiness must follow get_connector_init_state() directly."""
    connector = _make_connector(KVConnectorRole.WORKER)
    connector.enable_async_init()
    connector.state = ConnectorInitState.INITIALIZING
    assert connector.is_connector_ready() is False
    connector.state = ConnectorInitState.READY
    assert connector.is_connector_ready() is True


# ---------------------------------------------------------------------------
# build_connector_init_status — worker side
# ---------------------------------------------------------------------------


def test_build_connector_init_status_reports_own_rank_when_ready():
    connector = _make_connector(KVConnectorRole.WORKER, rank=1)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    status = connector.build_connector_init_status()
    assert isinstance(status, KVConnectorRankInitStatus)
    assert status.ready_ranks == {1}


def test_build_connector_init_status_empty_while_initializing():
    connector = _make_connector(KVConnectorRole.WORKER, rank=1)
    connector.enable_async_init()
    connector.state = ConnectorInitState.INITIALIZING
    status = connector.build_connector_init_status()
    assert status.ready_ranks == set()


def test_build_connector_init_status_none_once_scheduler_acknowledged():
    """Binding real metadata marks the report acknowledged; no further
    status reports should be built afterwards."""
    connector = _make_connector(KVConnectorRole.WORKER)
    connector.enable_async_init()
    connector.state = ConnectorInitState.READY
    connector.bind_connector_metadata(SimpleNamespace())
    assert connector.build_connector_init_status() is None


# ---------------------------------------------------------------------------
# update_connector_init_status — scheduler side
# ---------------------------------------------------------------------------


def test_update_connector_init_status_ready_once_all_ranks_report():
    connector = _make_connector(KVConnectorRole.SCHEDULER, world_size=2)
    connector.enable_async_init()
    assert connector.is_connector_ready() is False

    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={0}))
    assert connector.is_connector_ready() is False

    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={1}))
    assert connector.is_connector_ready() is True


def test_update_connector_init_status_noop_for_worker_role():
    """This method only applies scheduler-side; a worker's readiness must
    keep tracking its own init state instead."""
    connector = _make_connector(KVConnectorRole.WORKER, world_size=1)
    connector.enable_async_init()
    connector.update_connector_init_status(KVConnectorRankInitStatus(ready_ranks={0}))
    assert connector.is_connector_ready() is False
