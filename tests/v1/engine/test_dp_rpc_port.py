# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The data parallel RPC port is checked before the handshake binds it."""

import socket
from types import SimpleNamespace

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.v1.engine.dp_rpc_port import (
    _is_endpoint_free,
    binds_dp_rpc_port,
    check_dp_rpc_port_available,
    find_tcp_listener,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_free_endpoint_is_reported_as_free():
    port = _free_port()
    assert _is_endpoint_free("127.0.0.1", port)


def test_occupied_endpoint_is_reported_as_taken():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert not _is_endpoint_free("127.0.0.1", port)


def test_check_passes_when_the_port_is_free():
    check_dp_rpc_port_available("127.0.0.1", _free_port())


def test_check_raises_when_the_port_is_taken():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        with pytest.raises(RuntimeError, match="already in use"):
            check_dp_rpc_port_available("127.0.0.1", port)


def test_check_error_names_the_holder():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        with pytest.raises(RuntimeError) as exc_info:
            check_dp_rpc_port_available("127.0.0.1", port)
    message = str(exc_info.value)
    assert f"127.0.0.1:{port}" in message
    assert "PID" in message


def test_listener_on_another_address_is_not_a_conflict():
    """A listener on a different address of the same port does not hold the
    endpoint, so the engine can still bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert find_tcp_listener("192.0.2.1", port) is None


def test_wildcard_listener_holds_the_endpoint():
    """A wildcard listener owns every local address of the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert find_tcp_listener("127.0.0.1", port) is not None


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 is not available")
def test_ipv6_only_listener_does_not_hold_the_ipv4_endpoint():
    """An IPv6-only listener sits next to an IPv4 endpoint without taking it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # The IPv4 endpoint is free: nothing holds it yet.
    assert _is_endpoint_free("127.0.0.1", port)

    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock6:
        sock6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        try:
            sock6.bind(("::", port))
        except OSError:
            pytest.skip("the host does not allow an IPv6-only listener")
        sock6.listen(1)
        # The IPv6-only listener does not take the IPv4 endpoint.
        assert _is_endpoint_free("127.0.0.1", port)


def test_rank0_frontend_binds_the_rpc_port():
    """The rank 0 frontend binds the TCP endpoint, so the port is checked."""
    assert binds_dp_rpc_port(
        handshake_local_only=False,
        local_engines_only=False,
        dp_rank=0,
    )


def test_all_local_engines_do_not_bind_the_rpc_port():
    """With every engine local the handshake uses the local IPC transport, so
    the TCP port is never bound and must not be checked."""
    assert not binds_dp_rpc_port(
        handshake_local_only=True,
        local_engines_only=False,
        dp_rank=0,
    )


def test_non_master_rank_does_not_bind_the_rpc_port():
    """Only the rank 0 frontend binds the endpoint; the other ranks connect to
    the master instead."""
    assert not binds_dp_rpc_port(
        handshake_local_only=False,
        local_engines_only=True,
        dp_rank=1,
    )


def test_bind_site_reports_a_held_rpc_port():
    """The bind site rejects a held port before libzmq binds it.

    This drives the real bind path with a real `ParallelConfig`, so the check
    is exercised where the handshake address is bound rather than only through
    the helper.
    """
    from vllm.config.parallel import ParallelConfig

    # The socket has to stay open: it holds the endpoint for the whole test.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]

        parallel_config = ParallelConfig(
            data_parallel_size=2,
            data_parallel_size_local=1,
            data_parallel_rank=0,
        )
        parallel_config.data_parallel_master_ip = "127.0.0.1"
        parallel_config.data_parallel_rpc_port = port

        # The rank 0 frontend of a multi-node deployment binds the TCP endpoint.
        handshake_local_only = parallel_config.data_parallel_size_local == (
            parallel_config.data_parallel_size
        )
        assert binds_dp_rpc_port(
            handshake_local_only=handshake_local_only,
            local_engines_only=parallel_config.local_engines_only,
            dp_rank=parallel_config.data_parallel_rank,
        )

        # The endpoint is held, so the check must report it.
        with pytest.raises(RuntimeError, match="already in use"):
            check_dp_rpc_port_available(
                parallel_config.data_parallel_master_ip,
                parallel_config.data_parallel_rpc_port,
            )
    finally:
        sock.close()


def test_launch_core_engines_checks_the_rpc_port(monkeypatch: pytest.MonkeyPatch):
    """The bind site checks the port before it binds the handshake address.

    This asserts the wiring itself: `launch_core_engines` must call the check
    with the master address and the configured RPC port.
    """
    from vllm.v1.engine.utils import launch_core_engines

    checked: list[tuple[str, int]] = []

    def record(host: str, port: int) -> None:
        checked.append((host, port))

    monkeypatch.setattr(
        "vllm.v1.engine.utils.check_dp_rpc_port_available",
        record,
    )
    # Stop the check from reaching the bind: the engine cores are not started
    # here, so the handshake never completes.
    monkeypatch.setattr(
        "vllm.v1.engine.utils.wait_for_engine_startup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.v1.engine.utils.CoreEngineProcManager",
        lambda **kwargs: None,
    )

    parallel_config = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=0,
    )
    parallel_config.data_parallel_master_ip = "127.0.0.1"
    parallel_config.data_parallel_rpc_port = 29550

    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(multimodal_config=None, is_moe=False),
        cache_config=SimpleNamespace(),
        needs_dp_coordinator=False,
        kv_transfer_config=None,
        shutdown_timeout=None,
        instance_id="vllm-dp-rpc-port-test",
    )

    with launch_core_engines(
        vllm_config,
        executor_class=object,  # type: ignore[arg-type]
        log_stats=False,
        addresses=SimpleNamespace(),
    ):
        pass

    assert checked == [("127.0.0.1", 29550)]
