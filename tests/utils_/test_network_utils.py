# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import socket
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
import zmq

from vllm.utils.network_utils import (
    get_open_port,
    get_open_ports_list,
    get_tcp_uri,
    join_host_port,
    make_zmq_path,
    make_zmq_socket,
    split_host_port,
    split_zmq_path,
)

NETWORK_UTILS_PATH = str(
    Path(__file__).parents[2] / "vllm" / "utils" / "network_utils.py"
)


def _get_unused_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _run_port_subprocess(
    base_port: int,
    count: int,
    call: str,
) -> subprocess.Popen[str]:
    """Spawn a subprocess that imports network_utils.py and calls a port
    allocation function (get_open_port or get_open_ports_list).

    Uses a lightweight module shim to avoid importing the full vllm stack.
    """
    code = r"""
import importlib.util, logging, os, sys, types

fake_vllm = types.ModuleType("vllm")
fake_envs = types.ModuleType("vllm.envs")
fake_envs.VLLM_PORT = int(os.environ["VLLM_PORT"])

fake_logger = types.ModuleType("vllm.logger")
fake_logger.init_logger = lambda _: logging.getLogger("test")

sys.modules["vllm"] = fake_vllm
sys.modules["vllm.envs"] = fake_envs
sys.modules["vllm.logger"] = fake_logger

spec = importlib.util.spec_from_file_location("net", os.environ["NETWORK_UTILS_PATH"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
count = int(os.environ["NETWORK_UTILS_COUNT"])
print(",".join(str(p) for p in mod.get_open_ports_list(count)))
"""
    env = os.environ.copy()
    env["VLLM_PORT"] = str(base_port)
    env["NETWORK_UTILS_PATH"] = NETWORK_UTILS_PATH
    env["NETWORK_UTILS_COUNT"] = str(count)
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        # make sure we can get multiple ports, even if the env var is set
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("localhost", get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(("localhost", get_open_port()))


def test_get_open_ports_list_with_vllm_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", "5678")
        ports = get_open_ports_list(5)
        assert len(ports) == 5
        assert len(set(ports)) == 5, "ports must be unique"

        # verify every port is actually bindable
        sockets = []
        try:
            for p in ports:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("localhost", p))
                sockets.append(s)
        finally:
            for s in sockets:
                s.close()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("ipc://some_path", ("ipc", "some_path", "")),
        ("tcp://127.0.0.1:5555", ("tcp", "127.0.0.1", "5555")),
        ("tcp://[::1]:5555", ("tcp", "::1", "5555")),  # IPv6 address
        ("inproc://some_identifier", ("inproc", "some_identifier", "")),
    ],
)
def test_split_zmq_path(path, expected):
    assert split_zmq_path(path) == expected


@pytest.mark.parametrize(
    "invalid_path",
    [
        "invalid_path",  # Missing scheme
        "tcp://127.0.0.1",  # Missing port
        "tcp://[::1]",  # Missing port for IPv6
        "tcp://:5555",  # Missing host
    ],
)
def test_split_zmq_path_invalid(invalid_path):
    with pytest.raises(ValueError):
        split_zmq_path(invalid_path)


def test_make_zmq_socket_ipv6():
    # Check if IPv6 is supported by trying to create an IPv6 socket
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.close()
    except OSError:
        pytest.skip("IPv6 is not supported on this system")

    ctx = zmq.Context()
    ipv6_path = "tcp://[::]:5555"  # IPv6 loopback address
    socket_type = zmq.REP  # Example socket type

    # Create the socket
    zsock: zmq.Socket = make_zmq_socket(ctx, ipv6_path, socket_type)

    # Verify that the IPV6 option is set
    assert zsock.getsockopt(zmq.IPV6) == 1, (
        "IPV6 option should be enabled for IPv6 addresses"
    )

    # Clean up
    zsock.close()
    ctx.term()


def test_make_zmq_path():
    assert make_zmq_path("tcp", "127.0.0.1", "5555") == "tcp://127.0.0.1:5555"
    assert make_zmq_path("tcp", "::1", "5555") == "tcp://[::1]:5555"


def test_get_tcp_uri():
    assert get_tcp_uri("127.0.0.1", 5555) == "tcp://127.0.0.1:5555"
    assert get_tcp_uri("::1", 5555) == "tcp://[::1]:5555"


def test_split_host_port():
    # valid ipv4
    assert split_host_port("127.0.0.1:5555") == ("127.0.0.1", 5555)
    # invalid ipv4
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("127.0.0.1::5555")
    with pytest.raises(ValueError):
        # tailing colon
        assert split_host_port("127.0.0.1:5555:")
    with pytest.raises(ValueError):
        # no colon
        assert split_host_port("127.0.0.15555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("127.0.0.1:5555a")

    # valid ipv6
    assert split_host_port("[::1]:5555") == ("::1", 5555)
    # invalid ipv6
    with pytest.raises(ValueError):
        # multi colon
        assert split_host_port("[::1]::5555")
    with pytest.raises(IndexError):
        # no colon
        assert split_host_port("[::1]5555")
    with pytest.raises(ValueError):
        # none int port
        assert split_host_port("[::1]:5555a")


def test_join_host_port():
    assert join_host_port("127.0.0.1", 5555) == "127.0.0.1:5555"
    assert join_host_port("::1", 5555) == "[::1]:5555"


def test_get_open_ports_list_no_collisions_across_processes():
    """Spawn 5 concurrent processes, each calling get_open_ports_list(5).

    Verifies that concurrent port allocation does not produce collisions,
    which was the TOCTOU race described in #28498.  Before the port=0
    atomic fix, all processes scanned from the same VLLM_PORT and
    produced near-total collisions.
    """
    base_port = _get_unused_port()
    procs = [_run_port_subprocess(base_port, count=5, call="list") for _ in range(5)]

    all_ports: list[int] = []
    for p in procs:
        out, err = p.communicate(timeout=30)
        assert p.returncode == 0, f"Subprocess failed: {err}"
        ports = [int(x) for x in out.strip().split(",")]
        assert len(ports) == 5, f"Expected 5 ports, got {ports}"
        all_ports.extend(ports)

    assert len(all_ports) == len(set(all_ports)), (
        f"Port collisions detected: {len(all_ports)} total, "
        f"{len(set(all_ports))} unique"
    )
