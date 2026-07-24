# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common


def _capture_warnings(monkeypatch):
    warnings: list[str] = []

    def warning_once(msg: str, *args):
        warnings.append(msg % args if args else msg)

    monkeypatch.setattr(moriio_common.logger, "warning_once", warning_once)
    return warnings


def test_resolve_host_ip_uses_explicit_real_host_ip(monkeypatch):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "eth0")

    assert moriio_common.resolve_host_ip({"host_ip": "10.0.0.7"}) == "10.0.0.7"
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth0"


@pytest.mark.parametrize("backend", ["rdma", "xgmi"])
def test_resolve_host_ip_allows_explicit_loopback(monkeypatch, backend):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "lo")
    warnings = _capture_warnings(monkeypatch)

    assert (
        moriio_common.resolve_host_ip(
            {"host_ip": "127.0.0.1", "proxy_ip": "127.0.0.1", "backend": backend}
        )
        == "127.0.0.1"
    )
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "lo"
    assert any("loopback" in warning for warning in warnings)


def test_resolve_host_ip_infers_from_proxy_route(monkeypatch):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(
        moriio_common,
        "_infer_local_ip_for_peer",
        lambda peer_ip: "165.245.143.170",
    )
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "eth0")

    assert moriio_common.resolve_host_ip({"proxy_ip": "165.245.143.170"}) == (
        "165.245.143.170"
    )
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth0"


def test_resolve_host_ip_infers_multinode_rdma_without_touching_rdma_env(
    monkeypatch,
):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)
    monkeypatch.setenv("MORI_RDMA_DEVICES", "ionic_0,ionic_1")
    monkeypatch.setenv("MORI_IB_GID_INDEX", "3")
    monkeypatch.setattr(
        moriio_common,
        "_infer_local_ip_for_peer",
        lambda peer_ip: "10.194.30.12",
    )
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "eth1")

    assert moriio_common.resolve_host_ip({"proxy_ip": "10.194.30.29"}) == (
        "10.194.30.12"
    )
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth1"
    assert moriio_common.os.environ["MORI_RDMA_DEVICES"] == "ionic_0,ionic_1"
    assert moriio_common.os.environ["MORI_IB_GID_INDEX"] == "3"


@pytest.mark.parametrize("proxy_ip", ["localhost", "127.0.0.1"])
def test_resolve_host_ip_falls_back_from_inferred_loopback(monkeypatch, proxy_ip):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(
        moriio_common,
        "_infer_local_ip_for_peer",
        lambda peer_ip: "127.0.0.1",
    )
    monkeypatch.setattr(moriio_common, "get_ip", lambda: "192.168.4.5")
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "eth0")
    warnings = _capture_warnings(monkeypatch)

    assert moriio_common.resolve_host_ip({"proxy_ip": proxy_ip}) == "192.168.4.5"
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth0"
    assert any("loopback" in warning for warning in warnings)


def test_resolve_host_ip_requires_proxy_when_host_ip_is_missing():
    with pytest.raises(ValueError, match="proxy_ip"):
        moriio_common.resolve_host_ip({})


def test_resolve_host_ip_falls_back_to_get_ip_on_route_failure(monkeypatch):
    monkeypatch.delenv("MORI_SOCKET_IFNAME", raising=False)

    def fail_route(peer_ip: str) -> str:
        raise OSError("no route to host")

    monkeypatch.setattr(moriio_common, "_infer_local_ip_for_peer", fail_route)
    monkeypatch.setattr(moriio_common, "get_ip", lambda: "192.168.4.5")
    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", lambda ip: "eth0")

    assert moriio_common.resolve_host_ip({"proxy_ip": "10.0.0.1"}) == "192.168.4.5"
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth0"


@pytest.mark.parametrize("fallback_ip", ["0.0.0.0", "127.0.0.1"])
def test_resolve_host_ip_rejects_bad_auto_fallback(monkeypatch, fallback_ip):
    def fail_route(peer_ip: str) -> str:
        raise OSError("no route to host")

    monkeypatch.setattr(moriio_common, "_infer_local_ip_for_peer", fail_route)
    monkeypatch.setattr(moriio_common, "get_ip", lambda: fallback_ip)

    with pytest.raises(ValueError, match="non-advertisable"):
        moriio_common.resolve_host_ip({"proxy_ip": "10.0.0.1"})


def test_resolve_host_ip_preserves_user_socket_ifname(monkeypatch):
    monkeypatch.setenv("MORI_SOCKET_IFNAME", "eth9")
    monkeypatch.setattr(
        moriio_common,
        "_infer_local_ip_for_peer",
        lambda peer_ip: "10.194.30.12",
    )

    def fail_interface_lookup(ip: str) -> str:
        raise AssertionError("explicit MORI_SOCKET_IFNAME should be preserved")

    monkeypatch.setattr(moriio_common, "_interface_for_ipv4", fail_interface_lookup)

    assert moriio_common.resolve_host_ip({"proxy_ip": "10.194.30.29"}) == (
        "10.194.30.12"
    )
    assert moriio_common.os.environ["MORI_SOCKET_IFNAME"] == "eth9"
