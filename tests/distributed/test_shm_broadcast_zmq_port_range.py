# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import random
import socket

import pytest

from vllm import envs
from vllm.distributed.device_communicators.shm_broadcast import (
    MessageQueue,
    _parse_port_range,
)


def _port_is_free(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def _find_free_port_range(size: int = 20) -> tuple[int, int]:
    # Use an unprivileged range and keep the range small so that
    # `bind_to_random_port(..., max_tries=range_size)` stays fast.
    lo = 20_000
    hi = 65_535 - size
    for _ in range(200):
        start = random.randint(lo, hi)
        end = start + size - 1
        if all(_port_is_free(p) for p in range(start, end + 1)):
            return start, end
    pytest.skip("Could not find a free contiguous TCP port range on localhost.")
    raise RuntimeError("unreachable")


@pytest.mark.parametrize(
    "port_range,expected",
    [
        ("1-1", (1, 1)),
        ("  2-3  ", (2, 3)),
        ("4:5", (4, 5)),
    ],
)
def test_parse_port_range_valid(port_range: str, expected: tuple[int, int]) -> None:
    assert _parse_port_range(port_range) == expected


@pytest.mark.parametrize(
    "port_range",
    [
        "",
        "abc",
        "1",
        "2-1",
        "0-1",
        "1-65536",
        "1-2-3",
    ],
)
def test_parse_port_range_invalid(port_range: str) -> None:
    with pytest.raises(ValueError):
        _parse_port_range(port_range)


def test_message_queue_binds_within_zmq_port_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_port, end_port = _find_free_port_range(size=30)
    monkeypatch.setenv("VLLM_ZMQ_PORT_RANGE", f"{start_port}-{end_port}")
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()

    mq = MessageQueue(
        n_reader=1,
        n_local_reader=0,
        connect_ip="127.0.0.1",
    )
    try:
        addr = mq.export_handle().remote_subscribe_addr
        assert addr is not None
        port = int(addr.rsplit(":", 1)[1])
        assert start_port <= port <= end_port
    finally:
        if mq.remote_socket is not None:
            mq.remote_socket.close(0)
        if hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
