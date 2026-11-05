# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq

from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import CircuitBreaker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E501
    MooncakeLookupResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (  # noqa: E501
    encode_lookup_response,
)

pytestmark = pytest.mark.cpu_test

LOOKUP_TIMEOUT_S = 0.5


@pytest.fixture
def lookup_service(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[tuple[worker.LookupKeyClient, zmq.Socket]]:
    monkeypatch.setattr(worker, "LOOKUP_BREAKER_COOLDOWN_S", 0.5)
    monkeypatch.setattr(
        worker, "LOOKUP_BREAKER_FAILURE_THRESHOLD", getattr(request, "param", 1)
    )
    # Keep Unix socket paths below the platform limit.
    with tempfile.TemporaryDirectory(prefix="mc-", dir="/tmp") as rpc_dir:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", rpc_dir)
        config = VllmConfig(
            device_config=DeviceConfig("cpu"),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lookup_prefetch_limit": 1,
                    "lookup_timeout_s": LOOKUP_TIMEOUT_S,
                },
            ),
        )
        context = zmq.Context()
        peer = context.socket(zmq.REP)
        peer.setsockopt(zmq.RCVTIMEO, 2000)
        peer.bind(worker.get_zmq_rpc_path_lookup(config))
        client = worker.LookupKeyClient(config)
        try:
            yield client, peer
        finally:
            client.executor.shutdown(wait=True, cancel_futures=True)
            client.close()
            client.ctx.term()
            peer.close(linger=0)
            context.term()


def _resolve_lookup(
    client: worker.LookupKeyClient, req_id: str, mode: str
) -> MooncakeLookupResult:
    if mode == "prefetch":
        assert client.prefetch(req_id, 128, [])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.lookup(req_id, 128, [], non_block=mode != "blocking")
        if result is not None:
            return result
        threading.Event().wait(0.001)
    raise AssertionError("lookup remained pending")


@pytest.mark.parametrize("mode", ["blocking", "async", "prefetch"])
def test_slow_lookup_returns_miss_and_recovers_without_stale_reply(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket], mode: str
) -> None:
    client, peer = lookup_service
    release_reply = threading.Event()
    errors: list[Exception] = []

    def serve() -> None:
        try:
            peer.recv_multipart()
            assert release_reply.wait(10)
            peer.send(encode_lookup_response(MooncakeLookupResult(32)))
            peer.recv_multipart()
            peer.send(encode_lookup_response(MooncakeLookupResult(16)))
        except Exception as exc:
            errors.append(exc)

    server = threading.Thread(target=serve)
    server.start()
    with ThreadPoolExecutor(max_workers=1) as requests:
        try:
            # The outer timeout also makes this test fail promptly on the base.
            result = requests.submit(_resolve_lookup, client, "slow", mode).result(5)
            assert result.hit_length == 0
            assert client.circuit_breaker.is_open
            assert client.lookup("skip", 128, []).hit_length == 0

            # Recovery must ignore the abandoned request's 32-token reply.
            release_reply.set()
            deadline = time.monotonic() + 5
            while client.circuit_breaker.is_open:
                assert time.monotonic() < deadline
                threading.Event().wait(0.001)
            assert client.lookup("recovered", 128, []).hit_length == 16
            assert not client.circuit_breaker.is_open
        finally:
            release_reply.set()
            server.join(10)
    assert not server.is_alive()
    assert not errors


@pytest.mark.parametrize("mode", ["blocking", "async", "prefetch"])
def test_lookup_expiring_in_queue_returns_miss_without_failure(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket], mode: str
) -> None:
    client, _ = lookup_service
    release_executor = threading.Event()
    client.executor.submit(release_executor.wait)
    with ThreadPoolExecutor(max_workers=1) as requests:
        try:
            result = requests.submit(_resolve_lookup, client, "queued", mode).result(5)
            assert result.hit_length == 0
            # The lookup never reached the store, so the breaker stays closed.
            assert not client.circuit_breaker.is_open
        finally:
            client.discard("queued")
            release_executor.set()


def test_circuit_breaker_reopens_after_failed_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    breaker = CircuitBreaker("Test", 30, failure_threshold=3, probe_window_s=5)
    assert not breaker.is_open
    for _ in range(2):
        breaker.record_failure()
        assert not breaker.is_open
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure()
        assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open
    assert "3 consecutive failures; skipping requests for 30 seconds" in caplog.text
    clock[0] = 130
    assert not breaker.is_open
    # After the cooldown a single probe is admitted; others keep skipping.
    assert breaker.allow()
    assert breaker.is_open
    assert not breaker.allow()
    breaker.record_failure()
    clock[0] = 159
    assert breaker.is_open
    clock[0] = 160
    assert breaker.allow()
    # A probe that never reports back frees the slot after the probe window.
    clock[0] = 164
    assert not breaker.allow()
    clock[0] = 165
    assert breaker.allow()
    breaker.record_success()
    assert not breaker.is_open
    assert breaker.allow()
    assert breaker.allow()


@pytest.mark.parametrize("lookup_service", [3], indirect=True)
def test_failed_lookup_counts_once_toward_threshold(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, _ = lookup_service
    # The bound peer sends no replies; each request times out once.
    for index in range(3):
        assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
        assert client.circuit_breaker.is_open == (index == 2)


@pytest.mark.parametrize("lookup_service", [2], indirect=True)
def test_lookup_started_after_deadline_does_not_update_breaker(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, _ = lookup_service
    client.circuit_breaker.record_failure()
    release_executor = threading.Event()
    client.executor.submit(release_executor.wait)
    try:
        assert client.lookup("queued", 128, [], non_block=True) is None
        # Skip polling past the deadline so the lookup starts late, not cancelled.
        threading.Event().wait(LOOKUP_TIMEOUT_S + 0.1)
    finally:
        release_executor.set()
    client.executor.submit(lambda: None).result(timeout=5)
    assert client.lookup("queued", 128, [], non_block=True).hit_length == 0
    assert not client.circuit_breaker.is_open
    # Neither a failure nor a success: the earlier failure still counts.
    assert client.lookup("failed", 128, []).hit_length == 0
    assert client.circuit_breaker.is_open


@pytest.mark.parametrize("lookup_service", [3], indirect=True)
def test_cached_hit_does_not_reset_newer_failures(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, peer = lookup_service

    def serve() -> None:
        peer.recv_multipart()
        peer.send(encode_lookup_response(MooncakeLookupResult(16)))

    server = threading.Thread(target=serve)
    server.start()
    try:
        assert _resolve_lookup(client, "cached", "async").hit_length == 16
        for index in range(3):
            assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
            assert client.circuit_breaker.is_open == (index == 2)
            if index < 2:
                assert client.lookup("cached", 128, [], non_block=True).hit_length == 16
    finally:
        server.join(10)
    assert not server.is_alive()
