# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DataTransport ABC, NixlDataTransport and MoRIIODataTransport."""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.distributed.ec_transfer.ec_connector.cpu.data.moriio as moriio_mod
from vllm.distributed.ec_transfer.ec_connector.cpu.data.base import (
    DataTransport,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.data.moriio import (
    MoRIIODataTransport,
    _merge_segments,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.utils import (
    serialize_mem_descriptor,
)
from vllm.utils.network_utils import get_open_port

_NIXL_PATH = "vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl"


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
    from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
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


def test_nixl_data_transport_unwinds_registration_if_init_fails():
    """A failure after register_memory must not leave the region pinned.

    __init__ propagates, so the caller never gets a transport to deregister.
    """
    nixl = _make_nixl_mock(prep_side_effect=RuntimeError("no dlist"))
    from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
        NixlDataTransport,
    )

    with (
        patch(f"{_NIXL_PATH}.NixlWrapper", return_value=nixl),
        patch(f"{_NIXL_PATH}.nixl_agent_config", return_value=MagicMock()),
        pytest.raises(RuntimeError, match="no dlist"),
    ):
        NixlDataTransport("agent", 0, 4, 64, 256)

    nixl.deregister_memory.assert_called_once_with(nixl.get_reg_descs.return_value)


def test_nixl_data_transport_raises_if_nixl_unavailable():
    from vllm.distributed.ec_transfer.ec_connector.cpu.data.nixl import (
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


# ── MoRIIODataTransport ───────────────────────────────────────────────────────
# MoRI-IO is replaced by an in-process fake whose sessions copy bytes between
# registered tensors, so the tests check the bytes a READ moves, the block
# offsets it uses and the completion notification the producer receives.

_BS = 16
_N = 8


class _Status:
    def __init__(self, done: bool = True) -> None:
        self.done = done

    def Succeeded(self) -> bool:
        return self.done

    def Failed(self) -> bool:
        return False

    def InProgress(self) -> bool:
        return not self.done


class _MemoryDesc:
    def __init__(self, key: str, tensor: torch.Tensor) -> None:
        self.engine_key = key
        self.data = tensor.data_ptr()

    def pack(self) -> bytes:
        return f"{self.engine_key}|{self.data}".encode()

    @staticmethod
    def unpack(raw: bytes) -> "_MemoryDesc":
        key, data = raw.decode().split("|")
        desc = _MemoryDesc.__new__(_MemoryDesc)
        desc.engine_key, desc.data = key, int(data)
        return desc


class _EngineDesc:
    def __init__(self, key: str) -> None:
        self.key = key

    def pack(self) -> bytes:
        return self.key.encode()

    @staticmethod
    def unpack(raw: bytes) -> "_EngineDesc":
        return _EngineDesc(raw.decode())


class _Session:
    def __init__(self, local: torch.Tensor, remote: torch.Tensor, pending: bool):
        self.local = local.view(-1)
        self.remote = remote.view(-1)
        self.pending = pending
        self.calls: list[tuple[list[int], list[int], list[int]]] = []

    def batch_read(self, local_offsets, remote_offsets, sizes, uid):
        self.calls.append((list(local_offsets), list(remote_offsets), list(sizes)))
        for lo, ro, n in zip(local_offsets, remote_offsets, sizes):
            self.local[lo : lo + n] = self.remote[ro : ro + n]
        return _Status(done=not self.pending)


class _FakeMori:
    """Registry shared by every fake engine in a test."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.remote_engines: dict[str, set[str]] = {}
        self.pending = False
        self.sessions: list[_Session] = []
        fake = self

        class IOEngine:
            def __init__(self, key, config):
                self.key = key
                fake.remote_engines[key] = set()

            def create_backend(self, backend_type, config):
                pass

            def get_engine_desc(self):
                return _EngineDesc(self.key)

            def register_torch_tensor(self, tensor):
                fake.tensors[self.key] = tensor
                return _MemoryDesc(self.key, tensor)

            def deregister_memory(self, desc):
                fake.tensors.pop(desc.engine_key, None)

            def register_remote_engine(self, desc):
                fake.remote_engines[self.key].add(desc.key)

            def deregister_remote_engine(self, desc):
                fake.remote_engines[self.key].discard(desc.key)

            def create_session(self, local, remote):
                session = _Session(
                    fake.tensors[local.engine_key],
                    fake.tensors[remote.engine_key],
                    fake.pending,
                )
                fake.sessions.append(session)
                return session

            def allocate_transfer_uid(self):
                return 0

        self.IOEngine = IOEngine


@pytest.fixture
def fake_mori(monkeypatch) -> _FakeMori:
    fake = _FakeMori()
    monkeypatch.setattr(moriio_mod, "IOEngine", fake.IOEngine)
    monkeypatch.setattr(moriio_mod, "IOEngineConfig", lambda host, port: None)
    monkeypatch.setattr(moriio_mod, "BackendType", type("B", (), {"RDMA": 2}))
    monkeypatch.setattr(moriio_mod, "PollCqMode", type("P", (), {"POLLING": 0}))
    monkeypatch.setattr(
        moriio_mod, "RdmaBackendConfig", lambda *args, **kwargs: (args, kwargs)
    )
    monkeypatch.setattr(moriio_mod, "EngineDesc", _EngineDesc)
    monkeypatch.setattr(moriio_mod, "MemoryDesc", _MemoryDesc)
    return fake


def _transport(name: str, fill: int) -> MoRIIODataTransport:
    region = torch.full((_N, _BS), fill, dtype=torch.int8)
    return MoRIIODataTransport(
        agent_name=name,
        region=region,
        block_size_bytes=_BS,
        host="127.0.0.1",
        port=0,
        notify_port=get_open_port(),
    )


def _drain(producer: MoRIIODataTransport, timeout_s: float = 2.0) -> list[bytes]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        notifs = producer.get_new_notifs()
        if notifs:
            return [m for msgs in notifs.values() for m in msgs]
        time.sleep(0.01)
    return []


def test_merge_segments_coalesces_runs_contiguous_on_both_sides():
    assert _merge_segments([0, 16, 32, 80], [64, 80, 96, 0], 16) == (
        [0, 80],
        [64, 0],
        [48, 16],
    )


def test_merge_segments_splits_when_only_one_side_is_contiguous():
    assert _merge_segments([0, 16], [32, 0], 16) == ([0, 16], [32, 0], [16, 16])


def test_read_moves_requested_blocks_and_notifies_producer(fake_mori):
    producer = _transport("enc", 0)
    consumer = _transport("pf", 0)
    try:
        src = fake_mori.tensors[producer._engine.key]
        for i in range(_N):
            src[i] = i + 1
        agent = consumer.add_remote_peer(
            producer.get_agent_metadata(), producer.get_mem_descriptor()
        )

        handle = consumer.post_read([0, 1, 5], agent, [3, 4, 7], notif_msg=b"s:h1")

        dst = fake_mori.tensors[consumer._engine.key]
        assert dst[0].eq(4).all() and dst[1].eq(5).all() and dst[5].eq(8).all()
        assert dst[2].eq(0).all()
        assert fake_mori.sessions[0].calls == [([0, 80], [48, 112], [32, 16])]
        assert consumer.check_xfer_state(handle) == "DONE"
        assert consumer.check_xfer_state(handle) == "DONE"
        assert _drain(producer) == [b"s:h1"]
        assert _drain(producer, timeout_s=0.2) == []
    finally:
        consumer.deregister()
        producer.deregister()


def test_released_in_flight_handle_is_kept_until_it_finishes(fake_mori):
    fake_mori.pending = True
    producer = _transport("enc", 1)
    consumer = _transport("pf", 0)
    try:
        agent = consumer.add_remote_peer(
            producer.get_agent_metadata(), producer.get_mem_descriptor()
        )
        handle = consumer.post_read([0], agent, [0], notif_msg=b"s:h1")
        assert consumer.check_xfer_state(handle) == "PROC"

        consumer.release_xfer_handle(handle)
        assert consumer._parked == [handle]

        handle.status.done = True
        consumer.get_new_notifs()
        assert consumer._parked == []
    finally:
        consumer.deregister()
        producer.deregister()


def test_remove_remote_peer_deregisters_engine_and_rejects_reads(fake_mori):
    producer = _transport("enc", 1)
    consumer = _transport("pf", 0)
    try:
        agent = consumer.add_remote_peer(
            producer.get_agent_metadata(), producer.get_mem_descriptor()
        )
        consumer.remove_remote_peer(agent)
        assert agent not in fake_mori.remote_engines[consumer._engine.key]
        with pytest.raises(KeyError):
            consumer.post_read([0], agent, [0], notif_msg=b"s:h1")
    finally:
        consumer.deregister()
        producer.deregister()


def test_scheduler_builds_moriio_transport_from_extra_config(fake_mori, monkeypatch):
    """ec_transport=moriio must reach the scheduler's data plane."""
    import uuid

    import vllm.distributed.ec_transfer.ec_connector.cpu.scheduler as sched_mod
    from tests.v1.ec_connector.unit.utils import create_ec_vllm_config
    from vllm.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
        ECSharedRegion,
    )
    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import (
        ECCPUScheduler,
    )

    region = ECSharedRegion(
        engine_id="eng-" + str(uuid.uuid4()), num_blocks=_N, block_size_bytes=_BS
    )
    monkeypatch.setattr(sched_mod, "create_ec_shared_region", lambda cfg: region)
    monkeypatch.setenv("VLLM_EC_SIDE_CHANNEL_HOST", "127.0.0.1")
    cfg = create_ec_vllm_config(ec_role="ec_consumer")
    cfg.model_config.hf_config = None
    cfg.model_config.get_inputs_embeds_size.return_value = 8
    cfg.model_config.model = "test-model"
    cfg.ec_transfer_config.ec_connector_extra_config.update(
        ec_enable_nixl=True,
        ec_transport="moriio",
        moriio_port=0,
        moriio_notify_port=get_open_port(),
    )

    s = ECCPUScheduler(cfg)
    try:
        assert isinstance(s._data, MoRIIODataTransport)
        assert fake_mori.tensors[s._data._engine.key].data_ptr() == (
            region.blocks.data_ptr()
        )
    finally:
        s.shutdown()


def test_moriio_transport_requires_mori(monkeypatch):
    monkeypatch.setattr(moriio_mod, "IOEngine", None)
    with pytest.raises(RuntimeError, match="requires MoRI-IO"):
        _transport("pf", 0)
