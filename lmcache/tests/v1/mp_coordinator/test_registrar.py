# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mp-side coordinator registration helpers.

Driven with an httpx MockTransport (no real I/O): register, the keep_registered
loop (heartbeat, re-register on 404, deregister on cancel), and resilience when
the coordinator is unreachable.
"""

# Standard
import asyncio
import contextlib

# Third Party
import httpx

# First Party
from lmcache.v1.mp_coordinator.registrar import keep_registered, register

_BASE = "http://coord:9300"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_register_returns_assigned_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/instances"
            return httpx.Response(
                200, json={"instance_id": "mp-xyz", "re_registered": False}
            )
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            assigned = await register(
                client, _BASE, http_port=8080, advertise_ip="127.0.0.1"
            )
            assert assigned == "mp-xyz"

    asyncio.run(run())


def test_register_forwards_p2p_and_mq_fields():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Standard
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"instance_id": "i1", "re_registered": False})

    async def run():
        async with _client(handler) as client:
            await register(
                client,
                _BASE,
                http_port=8080,
                advertise_ip="10.0.0.1",
                instance_id="i1",
                p2p_advertised_url="10.0.0.1:7600",
                mq_port=5555,
            )

    asyncio.run(run())
    # The MQ server is reachable at the same advertised ip, so no separate
    # mq_ip is sent -- only the port.
    assert captured["ip"] == "10.0.0.1"
    assert captured["p2p_advertised_url"] == "10.0.0.1:7600"
    assert captured["mq_port"] == 5555


def test_register_omits_mq_port_when_p2p_disabled():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Standard
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"instance_id": "i1", "re_registered": False})

    async def run():
        async with _client(handler) as client:
            await register(client, _BASE, http_port=8080, advertise_ip="10.0.0.1")

    asyncio.run(run())
    assert captured["mq_port"] == 0


async def _run_loop_briefly(client: httpx.AsyncClient, **kwargs) -> None:
    """Run keep_registered for a few ticks, then cancel it."""
    task = asyncio.create_task(
        keep_registered(
            client,
            _BASE,
            http_port=8080,
            advertise_ip="127.0.0.1",
            heartbeat_interval=0.03,
            **kwargs,
        )
    )
    await asyncio.sleep(0.12)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_keep_registered_heartbeats_then_deregisters():
    seen = {"heartbeats": 0, "delete_path": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            seen["heartbeats"] += 1
            return httpx.Response(200, json={"instance_id": "i1"})
        seen["delete_path"] = request.url.path  # DELETE on cancel
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    assert seen["heartbeats"] >= 1
    assert seen["delete_path"] == "/instances/i1"


def test_keep_registered_reregisters_on_404():
    calls = {"register": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            calls["register"] += 1
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(404, json={"error": "unknown"})
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    assert calls["register"] >= 2  # initial + at least one re-register after 404


def test_keep_registered_keeps_id_on_transient_heartbeat_failure():
    seen = {"register": 0, "delete_path": None}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            seen["register"] += 1
            return httpx.Response(
                200, json={"instance_id": "i1", "re_registered": False}
            )
        if request.method == "PUT":
            return httpx.Response(500, json={"error": "transient"})
        seen["delete_path"] = request.url.path
        return httpx.Response(204)

    async def run():
        async with _client(handler) as client:
            await _run_loop_briefly(client, instance_id="i1")

    asyncio.run(run())
    # Transient 5xx must NOT spawn a duplicate registration, and the id is
    # kept so shutdown can still deregister.
    assert seen["register"] == 1
    assert seen["delete_path"] == "/instances/i1"


def test_keep_registered_survives_malformed_response():
    def handler(request: httpx.Request) -> httpx.Response:
        # 200 with a schema-mismatched body -> RegisterResponse validation fails.
        return httpx.Response(200, json={"unexpected": "field"})

    async def run():
        async with _client(handler) as client:
            task = asyncio.create_task(
                keep_registered(
                    client,
                    _BASE,
                    http_port=8080,
                    advertise_ip="127.0.0.1",
                    heartbeat_interval=0.03,
                )
            )
            await asyncio.sleep(0.1)
            assert not task.done()  # validation error did not kill the task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_keep_registered_survives_unreachable_coordinator():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("coordinator down")

    async def run():
        async with _client(handler) as client:
            task = asyncio.create_task(
                keep_registered(
                    client,
                    _BASE,
                    http_port=8080,
                    advertise_ip="127.0.0.1",
                    heartbeat_interval=0.03,
                )
            )
            await asyncio.sleep(0.1)
            assert not task.done()  # kept retrying, did not crash
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())
