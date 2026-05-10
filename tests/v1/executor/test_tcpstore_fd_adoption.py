# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TCPStore master_listen_fd adoption used by MultiprocExecutor.

Without this fix, MultiprocExecutor pre-picked a port via get_open_port()
(bind + getsockname + close), passed the bare integer to a worker, and
rank-0 re-bound the port inside libtorch's TCPStore. On a busy host
another process can grab the port between the two binds → EADDRINUSE,
which surfaced as CI flake b5cd7338.

The fix holds the listening socket open in the parent, hands the FD to
rank-0 via send_handle, and adopts it in TCPStore via master_listen_fd.
"""

import multiprocessing
import socket
from datetime import timedelta

import pytest

from vllm.distributed.utils import create_tcp_store
from vllm.v1.executor.multiproc_executor import _bind_local_listen_socket


def _client_sets_key(host: str, port: int, key: str, value: str) -> None:
    """Run in a child process: connect to the master TCPStore and set a key."""
    import torch.distributed as dist

    client = dist.TCPStore(
        host_name=host,
        port=port,
        world_size=2,
        is_master=False,
        timeout=timedelta(seconds=15),
        wait_for_workers=False,
        use_libuv=False,
    )
    client.set(key, value)


def test_bind_local_listen_socket_returns_listening_socket():
    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    try:
        host, port = s.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        s.close()


def test_held_listen_socket_blocks_intruders():
    """While the parent holds the bound listener, no other process can
    grab the same port. This is the property that closes the TOCTOU
    window the fix targets."""
    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    try:
        port = s.getsockname()[1]
        intruder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                intruder.bind(("127.0.0.1", port))
        finally:
            intruder.close()
    finally:
        s.close()


def test_tcpstore_adopts_pre_bound_fd_end_to_end():
    """Master TCPStore created via create_tcp_store(listen_socket=...)
    serves a real client connecting to the same host:port."""
    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    port = s.getsockname()[1]

    # Master adopts the FD via master_listen_fd; no rebind happens.
    store = create_tcp_store(
        host="127.0.0.1",
        port=port,
        listen_socket=s,
        world_size=2,
        is_master=True,
        timeout=timedelta(seconds=20),
        wait_for_workers=False,
        use_libuv=False,
    )

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_client_sets_key,
        args=("127.0.0.1", port, "hello", "world"),
    )
    proc.start()
    try:
        # Block until the client sets the key. Bounded by the store's
        # timeout above; if adoption is broken this hangs and fails.
        value = store.get("hello")
        assert value == b"world"
    finally:
        proc.join(timeout=20)
        assert proc.exitcode == 0, f"client exit={proc.exitcode}"
        # Drop the store so its listener FD is released before next test.
        del store


def _send_fd_child(fd_pipe, host: str, port: int, world_size: int, result_q):
    """Run in a child: receive a listening FD, build a master TCPStore."""
    from multiprocessing.reduction import recv_handle

    try:
        fd = recv_handle(fd_pipe)
        listen_sock = socket.socket(fileno=fd)
        store = create_tcp_store(
            host=host,
            port=port,
            listen_socket=listen_sock,
            world_size=world_size,
            is_master=True,
            timeout=timedelta(seconds=20),
            wait_for_workers=False,
            use_libuv=False,
        )
        # Block on a key set by the parent's client to confirm two-way
        # liveness through the adopted listener.
        value = store.get("ping")
        result_q.put(("ok", bytes(value)))
        del store
    except Exception as e:
        result_q.put(("err", repr(e)))


def test_url_fallback_works_after_listener_closed():
    """If FD adoption fails (mocked send_handle failure), the worker must
    be able to bind the same host:port via the URL path. This requires
    the parent to release its listener before the worker tries to bind —
    otherwise EADDRINUSE poisons the fallback."""
    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    port = s.getsockname()[1]
    # Simulate the post-fix lifecycle: parent attempts send_handle, it
    # fails, parent immediately closes its copy of the listener.
    s.close()

    # The fallback URL path now binds a fresh master TCPStore. With the
    # parent's listener gone, this must succeed (no EADDRINUSE).
    import torch.distributed as dist

    master = dist.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=2,
        is_master=True,
        timeout=timedelta(seconds=20),
        wait_for_workers=False,
        use_libuv=False,
    )

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_client_sets_key,
        args=("127.0.0.1", port, "fallback", "ok"),
    )
    proc.start()
    try:
        value = master.get("fallback")
        assert value == b"ok"
    finally:
        proc.join(timeout=20)
        assert proc.exitcode == 0
        del master


def test_listener_held_after_close_releases_port():
    """Confirms the kernel actually releases the port after the parent's
    close — defensive against any lingering-FD surprise (e.g. another
    dup somewhere)."""
    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    port = s.getsockname()[1]
    s.close()
    # Re-bind succeeds: kernel released the port.
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s2.bind(("127.0.0.1", port))
    finally:
        s2.close()


def test_send_handle_to_child_then_adopt():
    """Mirror what MultiprocExecutor does: parent binds + listens, sends
    the FD to a child via send_handle, child constructs a master
    TCPStore via FD adoption. Parent acts as a client."""
    from multiprocessing.reduction import send_handle

    s = _bind_local_listen_socket("127.0.0.1")
    assert s is not None
    port = s.getsockname()[1]

    ctx = multiprocessing.get_context("spawn")
    # Must be duplex=True so the connection is socketpair-backed (AF_UNIX);
    # send_handle uses SCM_RIGHTS which is unsupported on os.pipe()-backed
    # connections.
    parent_end, child_end = ctx.Pipe(duplex=True)
    result_q: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_send_fd_child,
        args=(child_end, "127.0.0.1", port, 2, result_q),
    )
    proc.start()
    try:
        # Hand off the FD across processes; parent then closes its copy
        # so the child becomes the sole owner of the listener.
        send_handle(parent_end, s.fileno(), proc.pid)
        parent_end.close()
        child_end.close()
        s.close()

        # Connect as a client and set the key the child is waiting on.
        import torch.distributed as dist

        client = dist.TCPStore(
            host_name="127.0.0.1",
            port=port,
            world_size=2,
            is_master=False,
            timeout=timedelta(seconds=20),
            wait_for_workers=False,
            use_libuv=False,
        )
        client.set("ping", "pong")
        del client

        proc.join(timeout=30)
        assert proc.exitcode == 0
        status, value = result_q.get(timeout=5)
        assert status == "ok", f"child error: {value!r}"
        assert value == b"pong"
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
