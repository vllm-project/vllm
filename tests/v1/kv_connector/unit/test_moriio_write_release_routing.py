# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WRITE-mode prefill-block release routing under Wide-EP.

``_release_write_prefill_blocks`` must address the producer the same way the
notify path does: pod-local rank for the port offset, owning pod's IP for the
host. Using the raw global rank targets a port no pod binds, so the release is
dropped and the blocks leak. Only multi-pod is affected.

Following test_moriio_routing.py, we bind the unbound method to a lightweight
stand-in rather than constructing a full scheduler.
"""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_connector
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)

_release = MoRIIOConnectorScheduler._release_write_prefill_blocks

MASTER_HOST = "10.0.0.10"
CHILD_HOST = "10.0.0.11"
NOTIFY_PORT = 61005


def _scheduler(tp_size: int = 1):
    """Producer-side stand-in capturing what the release would be sent to."""
    s = SimpleNamespace(tp_size=tp_size, sent=[])
    s._send_transfer_release = lambda transfer_id, host, port: s.sent.append(
        (transfer_id, host, port)
    )
    s._release_write_prefill_blocks = _release.__get__(s)
    return s


def _params(remote_dp_rank: int, dp_local: int, hosts: list[str] | None = None):
    """kv_transfer_params as the router/decode leg populates them."""
    p = {
        "transfer_id": "tx-1",
        "remote_dp_rank": remote_dp_rank,
        "remote_host": MASTER_HOST,
        "remote_notify_port": NOTIFY_PORT,
        "remote_dp_size_local": dp_local,
    }
    if hosts is not None:
        p["remote_hosts"] = hosts
    return p


# --------------------------------------------------------------------------
# Routing parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", range(8))
def test_master_pod_ranks_target_master_host(rank):
    """Ranks 0-7 live on the master pod: host unchanged, offset == rank."""
    s = _scheduler()
    s._release_write_prefill_blocks(
        "req", _params(rank, dp_local=8, hosts=[MASTER_HOST, CHILD_HOST])
    )
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + rank)]


@pytest.mark.parametrize("rank", range(8, 16))
def test_child_pod_ranks_target_child_host_with_folded_port(rank):
    """REGRESSION: ranks 8-15 must reach the child pod at a folded offset.

    Before the fix this sent to MASTER_HOST at NOTIFY_PORT + rank (61013-61020),
    which no pod binds, so the producer never freed the blocks.
    """
    s = _scheduler()
    s._release_write_prefill_blocks(
        "req", _params(rank, dp_local=8, hosts=[MASTER_HOST, CHILD_HOST])
    )
    assert s.sent == [("tx-1", CHILD_HOST, NOTIFY_PORT + (rank - 8))]


@pytest.mark.parametrize("rank", range(16))
def test_release_port_is_always_one_a_pod_binds(rank):
    """Every release lands inside the port range a pod actually binds."""
    dp_local = 8
    bound = {NOTIFY_PORT + r for r in range(dp_local)}
    s = _scheduler()
    s._release_write_prefill_blocks(
        "req", _params(rank, dp_local=dp_local, hosts=[MASTER_HOST, CHILD_HOST])
    )
    _, _, port = s.sent[0]
    assert port in bound


@pytest.mark.parametrize("rank", range(8))
def test_single_pod_is_unaffected(rank):
    """1P1D: one host, dp_local == dp_size, so the fold is a no-op."""
    s = _scheduler()
    s._release_write_prefill_blocks(
        "req", _params(rank, dp_local=8, hosts=[MASTER_HOST])
    )
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + rank)]


@pytest.mark.parametrize("rank", [0, 5, 12])
def test_external_dp_sentinel_does_not_fold(rank):
    """dp_local == 0 is the external-DP sentinel: no folding, single pod."""
    s = _scheduler()
    s._release_write_prefill_blocks("req", _params(rank, dp_local=0))
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + rank)]


def test_missing_remote_hosts_falls_back_to_remote_host():
    """Without a host list we cannot resolve the pod, but still fold the port.

    Folding is still correct on its own: the offset a pod binds is local.
    """
    s = _scheduler()
    s._release_write_prefill_blocks("req", _params(12, dp_local=8))
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + 4)]


def test_pod_index_out_of_range_falls_back_to_remote_host():
    """A short remote_hosts list must not raise or index out of range."""
    s = _scheduler()
    s._release_write_prefill_blocks("req", _params(12, dp_local=8, hosts=[MASTER_HOST]))
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + 4)]


def test_malformed_dp_local_is_tolerated():
    """A non-numeric remote_dp_size_local degrades to the sentinel."""
    s = _scheduler()
    params = _params(3, dp_local=8)
    params["remote_dp_size_local"] = "not-a-number"
    s._release_write_prefill_blocks("req", params)
    assert s.sent == [("tx-1", MASTER_HOST, NOTIFY_PORT + 3)]


def test_release_fans_out_over_tp_ranks():
    """Every TP rank of the owning pod is released, at folded offsets."""
    s = _scheduler(tp_size=2)
    s._release_write_prefill_blocks(
        "req", _params(9, dp_local=8, hosts=[MASTER_HOST, CHILD_HOST])
    )
    assert s.sent == [
        ("tx-1", CHILD_HOST, NOTIFY_PORT + 1),
        ("tx-1", CHILD_HOST, NOTIFY_PORT + 2),
    ]


def test_release_matches_notify_addressing_for_every_rank():
    """The property the fix restores: release and notify agree everywhere."""
    dp_local, hosts = 8, [MASTER_HOST, CHILD_HOST]

    def notify_target(rank):
        # Mirrors update_state_after_alloc's per-pod resolution.
        local = moriio_connector.fold_local_rank(rank, dp_local)
        pod = moriio_connector.pod_index(rank, dp_local)
        host = hosts[pod] if 0 <= pod < len(hosts) else MASTER_HOST
        return host, NOTIFY_PORT + moriio_connector.get_port_offset(local, 0)

    for rank in range(16):
        s = _scheduler()
        s._release_write_prefill_blocks("req", _params(rank, dp_local, hosts))
        _, host, port = s.sent[0]
        assert (host, port) == notify_target(rank), f"rank {rank} diverges"


# --------------------------------------------------------------------------
# Leak guards -- the release must never fail silently
# --------------------------------------------------------------------------


def test_missing_transfer_id_logs_error_and_sends_nothing(caplog):
    s = _scheduler()
    with caplog.at_level("ERROR"):
        s._release_write_prefill_blocks("req-no-tid", {"remote_host": MASTER_HOST})
    assert s.sent == []
    assert "Leaking WRITE prefill blocks" in caplog.text
    assert "req-no-tid" in caplog.text


def test_unresolvable_address_logs_error_and_sends_nothing(caplog, monkeypatch):
    monkeypatch.setattr(
        moriio_connector, "get_peer_zmq_from_request_id", lambda *a, **k: None
    )
    s = _scheduler()
    with caplog.at_level("ERROR"):
        s._release_write_prefill_blocks(
            "req-no-addr", {"transfer_id": "tx-9", "remote_dp_rank": 11}
        )
    assert s.sent == []
    # The leak must be attributable: transfer id and rank are both reported.
    assert "Leaking WRITE prefill blocks" in caplog.text
    assert "tx-9" in caplog.text
    assert "11" in caplog.text


def test_address_recovered_from_request_id_still_releases(monkeypatch):
    """When the address is recoverable, the blocks are freed normally."""
    monkeypatch.setattr(
        moriio_connector,
        "get_peer_zmq_from_request_id",
        lambda *a, **k: "tcp://10.0.0.11:6301:61005",
    )
    monkeypatch.setattr(
        moriio_connector,
        "parse_moriio_zmq_address",
        lambda _addr: (CHILD_HOST, 6301, NOTIFY_PORT),
    )
    s = _scheduler()
    s._release_write_prefill_blocks("req", {"transfer_id": "tx-2", "remote_dp_rank": 0})
    assert s.sent == [("tx-2", CHILD_HOST, NOTIFY_PORT)]
