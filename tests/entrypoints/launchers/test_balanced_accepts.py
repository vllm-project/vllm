# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NoSignalServer(peer_loads=...): API workers sharing one listening socket
accept each connection on the worker with the fewest open connections."""

import asyncio
import datetime
import multiprocessing
import os
import socket
import ssl
from collections import Counter

import pytest
import uvicorn
import uvloop

from vllm.entrypoints.launchers.launcher import NoSignalServer

TIMEOUT_S = 30


def _bound_socket() -> socket.socket:
    # Like setup_server: bound, not yet listening.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock


def _app(hold_s: float = 0.0):
    pid = str(os.getpid()).encode()

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        await asyncio.sleep(hold_s)
        headers = [(b"content-length", str(len(pid)).encode())]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": pid})

    return app


async def _get(port: int, requests: int = 1, ssl_ctx=None) -> list[bytes]:
    """``requests`` requests on one keep-alive connection; returns the bodies."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ssl_ctx)
    bodies = []
    for _ in range(requests):
        writer.write(b"GET / HTTP/1.1\r\nHost: t\r\n\r\n")
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), TIMEOUT_S)
        assert head.startswith(b"HTTP/1.1 200")
        length = int(head.split(b"content-length: ")[1].split(b"\r\n")[0])
        bodies.append(await asyncio.wait_for(reader.readexactly(length), TIMEOUT_S))
    writer.close()
    return bodies


async def _serve(sock, requests, loads=None, **config_kwargs):
    config = uvicorn.Config(_app(), lifespan="on", log_level="warning", **config_kwargs)
    loads = loads if loads is not None else [0]
    server = NoSignalServer(config, peer_loads=(loads, 0))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        assert not task.done(), task.result()
        await asyncio.sleep(0.01)
    try:
        return await requests(sock.getsockname()[1], server)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, TIMEOUT_S)
        # The listener is closed, nothing is left accepting, and peers no
        # longer see this worker as a target.
        assert sock.fileno() == -1
        assert not server._accept_tasks
        assert loads[0] == 2**31 - 1


def _run(coro, loop_impl):
    return uvloop.run(coro) if loop_impl == "uvloop" else asyncio.run(coro)


@pytest.mark.parametrize("loop_impl", ["asyncio", "uvloop"])
def test_requests_and_keep_alive_served(loop_impl):
    async def requests(port, server):
        return await _get(port), await _get(port, requests=3)

    one, three = _run(_serve(_bound_socket(), requests), loop_impl)
    assert one == [str(os.getpid()).encode()]
    assert len(three) == 3


def _self_signed(tmp_path):
    x509 = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .sign(key, hashes.SHA256())
    )
    keyfile, certfile = tmp_path / "key.pem", tmp_path / "cert.pem"
    keyfile.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(keyfile), str(certfile)


def test_tls_handshake_failure_keeps_serving(tmp_path):
    keyfile, certfile = _self_signed(tmp_path)
    client_ctx = ssl.create_default_context()
    client_ctx.check_hostname = False
    client_ctx.verify_mode = ssl.CERT_NONE

    async def requests(port, server):
        # A plain-text client fails the handshake; the next TLS client is served.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET / HTTP/1.1\r\nHost: t\r\n\r\n")
        await asyncio.wait_for(reader.read(), TIMEOUT_S)
        writer.close()
        return await _get(port, ssl_ctx=client_ctx)

    bodies = uvloop.run(
        _serve(_bound_socket(), requests, ssl_keyfile=keyfile, ssl_certfile=certfile)
    )
    assert bodies == [str(os.getpid()).encode()]


def test_busier_worker_still_serves():
    # This worker is index 0; peer 1 claims no connections but never accepts.
    loads = [0, 0]

    async def requests(port, server):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        bodies = await _get(port)
        assert loads[0] >= 1  # the held connection was published
        writer.close()
        return bodies

    assert uvloop.run(_serve(_bound_socket(), requests, loads)) == [
        str(os.getpid()).encode()
    ]


def _worker(sock, loads, index, ready, stop):
    async def main():
        config = uvicorn.Config(_app(hold_s=1), lifespan="on", log_level="warning")
        server = NoSignalServer(config, peer_loads=(loads, index))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            await asyncio.sleep(0.01)
        ready.release()
        await asyncio.get_running_loop().run_in_executor(None, stop.wait)
        server.should_exit = True
        await task

    # API workers run on uvloop, whose own server accepts a whole burst at once.
    uvloop.run(main())


@pytest.mark.skipif((os.cpu_count() or 1) < 8, reason="needs idle cores per worker")
@pytest.mark.parametrize("arrival_s", [0, 1])
def test_connections_balanced_over_workers(arrival_s):
    """Both a burst and a steady stream of connections end up balanced; with
    the event loop's own server one worker takes most of either."""
    workers, connections = 4, 400
    sock = _bound_socket()
    ctx = multiprocessing.get_context("spawn")
    ready, stop = ctx.Semaphore(0), ctx.Event()
    loads = ctx.RawArray("i", workers)
    procs = [
        ctx.Process(target=_worker, args=(sock, loads, i, ready, stop))
        for i in range(workers)
    ]
    for p in procs:
        p.start()
    try:
        for _ in range(workers):
            assert ready.acquire(timeout=120)

        async def get_after(delay):
            await asyncio.sleep(delay)
            return await _get(sock.getsockname()[1])

        async def arrive():
            delays = [arrival_s * i / connections for i in range(connections)]
            results = await asyncio.gather(*(get_after(d) for d in delays))
            return Counter(body for (body,) in results)

        counts = uvloop.run(arrive())
        assert sum(counts.values()) == connections
        # Fair share is 100.
        assert max(counts.values()) <= 5 * connections // (4 * workers), counts
    finally:
        stop.set()
        for p in procs:
            p.join(30)
            if p.is_alive():
                p.kill()
        sock.close()
