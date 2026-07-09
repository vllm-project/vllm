"""Standalone test to verify Windows multiprocessing spawn works with vLLM IPC.

Run with: python tests/test_win_spawn.py

This tests:
1. multiprocessing.get_context("spawn") works
2. A spawned subprocess can import vLLM core modules
3. get_open_zmq_ipc_path() returns a valid address
4. ZMQ socket bind/connect works across spawn boundary
5. Poller works with handshake socket (no sentinel in poller)
"""
import multiprocessing
import os
import sys
import time

import zmq


def _child_proc(handshake_addr: str, identity: bytes) -> None:
    """Child process: connect to parent handshake, send HELLO/READY."""
    import msgspec

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, identity)
    sock.connect(handshake_addr)

    # Send HELLO
    sock.send(msgspec.msgpack.encode({"status": "HELLO", "local": True, "headless": False}))
    # Wait for init
    poll = sock.poll(timeout=30_000)
    if not poll:
        print("CHILD: TIMEOUT waiting for init message", flush=True)
        return
    init = msgspec.msgpack.decode(sock.recv())
    print(f"CHILD: got init with addresses: {init.get('addresses')}", flush=True)

    # Send READY
    sock.send(msgspec.msgpack.encode({"status": "READY", "local": True, "headless": False}))
    print("CHILD: sent READY, done", flush=True)
    sock.close()
    ctx.term()


def test_spawn_import() -> None:
    """Test 1: basic spawn context."""
    ctx = multiprocessing.get_context("spawn")
    print(f"PASS: spawn context={ctx}")
    assert ctx is not None


def test_get_open_zmq_ipc_path() -> None:
    """Test 2: IPC path generation works on all platforms."""
    from vllm.utils.network_utils import get_open_zmq_ipc_path

    path = get_open_zmq_ipc_path()
    print(f"IPC path: {path}")
    if sys.platform == "win32":
        assert path.startswith("tcp://127.0.0.1:")
        # Should be tcp://127.0.0.1:0 (ZMQ auto-assign at bind)
        assert path == "tcp://127.0.0.1:0", f"Expected port 0 on Windows, got: {path}"
    else:
        assert path.startswith("ipc://"), f"Expected ipc://, got: {path}"
    print("PASS: get_open_zmq_ipc_path")


def test_zmq_bind_last_endpoint() -> None:
    """Test 3: binding to port 0 and resolving via LAST_ENDPOINT."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind("tcp://127.0.0.1:0")
    actual = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
    print(f"Bound to 127.0.0.1:0, got endpoint: {actual}")
    assert actual.startswith("tcp://127.0.0.1:")
    assert not actual.endswith(":0"), "Port should not be 0 after LAST_ENDPOINT"
    sock.close()
    ctx.term()
    print("PASS: zmq_bind_last_endpoint")


def test_spawn_with_zmq_handshake() -> None:
    """Test 4: full handshake across spawn boundary (no sentinel in poller)."""
    from vllm.utils.network_utils import zmq_socket_ctx

    handshake_addr = "tcp://127.0.0.1:0"
    ctx = multiprocessing.get_context("spawn")

    with zmq_socket_ctx(handshake_addr, zmq.ROUTER, bind=True) as sock:
        actual_addr = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        print(f"Handshake bound: {actual_addr}")

        # Spawn child with resolved address
        eng_identity = (0).to_bytes(2, "little")
        proc = ctx.Process(
            target=_child_proc,
            args=(actual_addr, eng_identity),
            name="TestEngine",
        )
        proc.start()
        print(f"Spawned child PID: {proc.pid}")

        # Wait for handshake -- DO NOT register sentinel with zmq.Poller
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        # Phase 1: wait for HELLO
        events = poller.poll(timeout=15_000)
        assert events, "TIMEOUT waiting for HELLO"
        identity, payload = sock.recv_multipart()
        import msgspec
        msg = msgspec.msgpack.decode(payload)
        assert msg["status"] == "HELLO", f"Expected HELLO, got {msg}"
        print("Got HELLO from child")

        # Send init
        init = msgspec.msgpack.encode({"addresses": None, "parallel_config": {}})
        sock.send_multipart((identity, init))

        # Phase 2: wait for READY
        events = poller.poll(timeout=15_000)
        assert events, "TIMEOUT waiting for READY"
        identity, payload = sock.recv_multipart()
        msg = msgspec.msgpack.decode(payload)
        assert msg["status"] == "READY", f"Expected READY, got {msg}"
        print("Got READY from child")

        # Check process exit
        proc.join(timeout=5)
        assert proc.exitcode == 0, f"Child failed with exit code: {proc.exitcode}"

    print("PASS: spawn_with_zmq_handshake")


def test_poller_no_sentinel() -> None:
    """Test 5: zmq.Poller works without registering sentinel FDs."""
    ctx = zmq.Context()
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.bind("tcp://127.0.0.1:0")
    actual = a.getsockopt(zmq.LAST_ENDPOINT).decode()
    b.connect(actual)

    poller = zmq.Poller()
    poller.register(a, zmq.POLLIN)

    # Send a message
    b.send(b"hello")
    events = poller.poll(timeout=5000)
    assert events, "TIMEOUT on PAIR socket"
    assert events[0][0] == a
    assert events[0][1] == zmq.POLLIN
    assert a.recv() == b"hello"

    a.close()
    b.close()
    ctx.term()
    print("PASS: poller_no_sentinel")


def main() -> None:
    print("=" * 60)
    print("vLLM Windows Spawn Test Suite")
    print("=" * 60)

    test_spawn_import()
    test_get_open_zmq_ipc_path()
    test_zmq_bind_last_endpoint()
    test_poller_no_sentinel()
    test_spawn_with_zmq_handshake()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
