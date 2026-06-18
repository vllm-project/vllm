# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DataTransport ABC and NixlDataTransport."""

from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    serialize_mem_descriptor,
)

_NIXL_PATH = "vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl"


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_nixl_mock(prep_side_effect=None) -> MagicMock:
    nixl = MagicMock()
    nixl.get_agent_metadata.return_value = b"agent-meta"
    nixl.get_reg_descs.return_value = MagicMock()
    nixl.get_xfer_descs.return_value = MagicMock()
    nixl.prep_xfer_dlist.side_effect = prep_side_effect or [77, 88]
    nixl.add_remote_agent.return_value = "remote-agent-1"
    nixl.get_new_notifs.return_value = {}
    nixl.check_xfer_state.return_value = "PROC"
    return nixl


def _make_transport(nixl_mock=None):
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl import (
        NixlDataTransport,
    )

    mock = nixl_mock or _make_nixl_mock()
    with (
        patch(f"{_NIXL_PATH}.NixlWrapper", return_value=mock),
        patch(f"{_NIXL_PATH}.nixl_agent_config", return_value=MagicMock()),
    ):
        t = NixlDataTransport("agent", 0, 4, 64, 256)
    t._nixl = mock
    return t


# ── ABC ───────────────────────────────────────────────────────────────────────


def test_data_transport_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataTransport()  # type: ignore[abstract]


# ── NixlDataTransport construction ───────────────────────────────────────────


def test_nixl_data_transport_registers_memory_on_init():
    nixl = _make_nixl_mock()
    _make_transport(nixl)
    nixl.register_memory.assert_called_once()


def test_nixl_data_transport_raises_if_nixl_unavailable():
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.data.nixl import (
        NixlDataTransport,
    )

    with (
        patch(f"{_NIXL_PATH}.NixlWrapper", None),
        patch(f"{_NIXL_PATH}.nixl_agent_config", None),
        pytest.raises(RuntimeError, match="requires NIXL"),
    ):
        NixlDataTransport("a", 0, 1, 64, 64)


# ── public accessors ──────────────────────────────────────────────────────────


def test_get_agent_metadata_returns_nixl_blob():
    assert _make_transport().get_agent_metadata() == b"agent-meta"


def test_get_mem_descriptor_roundtrips_block_descs():
    t = _make_transport()
    raw = t.get_mem_descriptor()
    assert isinstance(raw, bytes) and len(raw) > 0


# ── add_remote_peer ───────────────────────────────────────────────────────────


def test_add_remote_peer_returns_agent_name_and_stores_handle_internally():
    nixl = _make_nixl_mock(prep_side_effect=[77, 88])
    t = _make_transport(nixl)
    mem_desc = serialize_mem_descriptor([(0, 64, 0)])
    agent_name = t.add_remote_peer(b"meta", mem_desc)
    assert agent_name == "remote-agent-1"
    # Dlist handle stored internally; accessible via _peer_handles.
    assert t._peer_handles["remote-agent-1"] == 88


def test_add_remote_peer_registers_remote_agent():
    nixl = _make_nixl_mock(prep_side_effect=[77, 88])
    t = _make_transport(nixl)
    mem_desc = serialize_mem_descriptor([(0, 64, 0)])
    t.add_remote_peer(b"fresh-meta", mem_desc)
    nixl.add_remote_agent.assert_called_once_with(b"fresh-meta")


# ── remove_remote_peer ────────────────────────────────────────────────────────


def test_remove_remote_peer_delegates_to_nixl():
    t = _make_transport()
    t.remove_remote_peer("agent-x")
    t._nixl.remove_remote_agent.assert_called_once_with("agent-x")


def test_remove_remote_peer_swallows_exceptions():
    t = _make_transport()
    t._nixl.remove_remote_agent.side_effect = RuntimeError("gone")
    t.remove_remote_peer("agent-x")  # must not raise


# ── post_read ─────────────────────────────────────────────────────────────────


def test_post_read_raises_on_index_count_mismatch():
    nixl = _make_nixl_mock(prep_side_effect=[77, 88])
    t = _make_transport(nixl)
    mem_desc = serialize_mem_descriptor([(0, 64, 0)])
    agent = t.add_remote_peer(b"meta", mem_desc)
    with pytest.raises(ValueError, match="mismatch"):
        t.post_read([0, 1], agent, [5], notif_msg=b"sid")


def test_post_read_calls_make_prepped_xfer_and_transfer():
    nixl = _make_nixl_mock(prep_side_effect=[77, 88])
    t = _make_transport(nixl)
    mem_desc = serialize_mem_descriptor([(0, 64, 0)])
    agent = t.add_remote_peer(b"meta", mem_desc)
    nixl.make_prepped_xfer.return_value = 999
    handle = t.post_read([0, 1], agent, [5, 6], notif_msg=b"sid")
    nixl.make_prepped_xfer.assert_called_once()
    nixl.transfer.assert_called_once_with(999)
    assert handle == 999


# ── release_xfer_handle / deregister ─────────────────────────────────────────


def test_release_xfer_handle_swallows_exceptions():
    t = _make_transport()
    t._nixl.release_xfer_handle.side_effect = RuntimeError("stale")
    t.release_xfer_handle(42)  # must not raise


def test_deregister_swallows_exceptions():
    t = _make_transport()
    t._nixl.deregister_memory.side_effect = RuntimeError("already gone")
    t.deregister()  # must not raise
