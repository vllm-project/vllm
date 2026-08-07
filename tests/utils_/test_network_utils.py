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


def _get_unused_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _make_port_subprocess(base_port: int, rpc_base_path: Path, call: str):
    code = r"""
import importlib.util
import logging
import os
import sys
import types

fake_vllm = types.ModuleType("vllm")
fake_envs = types.ModuleType("vllm.envs")
fake_envs.VLLM_DP_MASTER_PORT = 0
fake_envs.VLLM_PORT = int(os.environ["VLLM_PORT"])
fake_envs.VLLM_RPC_BASE_PATH = os.environ["VLLM_RPC_BASE_PATH"]

fake_logger = types.ModuleType("vllm.logger")
fake_logger.init_logger = lambda _: logging.getLogger("network_utils_under_test")

sys.modules["vllm"] = fake_vllm
sys.modules["vllm.envs"] = fake_envs
sys.modules["vllm.logger"] = fake_logger

spec = importlib.util.spec_from_file_location(
    "network_utils_under_test", os.environ["NETWORK_UTILS_PATH"]
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if os.environ["NETWORK_UTILS_CALL"] == "one":
    print(module.get_open_port())
else:
    print(",".join(str(port) for port in module.get_open_ports_list(5)))
"""
    env = os.environ.copy()
    env.update(
        {
            "NETWORK_UTILS_CALL": call,
            "NETWORK_UTILS_PATH": str(
                Path(__file__).parents[2] / "vllm" / "utils" / "network_utils.py"
            ),
            "VLLM_PORT": str(base_port),
            "VLLM_RPC_BASE_PATH": str(rpc_base_path),
        }
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )


def _collect_port_subprocesses(procs: list[subprocess.Popen[str]]) -> list[str]:
    results: list[str] = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0, stderr
        results.append(stdout.strip())
    return results


def test_get_open_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(_get_unused_port()))
        # make sure we can get multiple ports, even if the env var is set
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
            s1.bind(("localhost", get_open_port()))
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                s2.bind(("localhost", get_open_port()))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s3:
                    s3.bind(("localhost", get_open_port()))


def test_get_open_port_with_vllm_port_is_unique_across_processes(tmp_path: Path):
    base_port = _get_unused_port()
    procs = [_make_port_subprocess(base_port, tmp_path, "one") for _ in range(8)]
    ports = [int(port) for port in _collect_port_subprocesses(procs)]

    assert len(ports) == len(set(ports)), ports


def test_get_open_ports_list_with_vllm_port_is_unique_across_processes(
    tmp_path: Path,
):
    base_port = _get_unused_port()
    procs = [_make_port_subprocess(base_port, tmp_path, "list") for _ in range(2)]
    port_lists = [
        [int(port) for port in output.split(",")]
        for output in _collect_port_subprocesses(procs)
    ]
    ports = [port for port_list in port_lists for port in port_list]

    assert len(ports) == len(set(ports)), port_lists


def test_get_open_ports_list_with_vllm_port(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_PORT", str(_get_unused_port()))
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
