# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import CircuitBreaker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import (
    protocol,
    worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E501
    MooncakeLookupResult,
)

from .mooncake_lookup_peer import LookupPeer, hit

pytestmark = pytest.mark.cpu_test

LOOKUP_TIMEOUT_S = 0.5


def _make_config(**extra_config) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lookup_prefetch_limit": 1,
                "lookup_timeout_s": LOOKUP_TIMEOUT_S,
                **extra_config,
            },
        ),
    )


@pytest.fixture
def rpc_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    # Keep Unix socket paths below the platform limit.
    with tempfile.TemporaryDirectory(prefix="mc-", dir="/tmp") as path:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", path)
        yield path


@pytest.fixture
def lookup_service(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, rpc_dir: str
) -> Iterator[tuple[worker.LookupKeyClient, LookupPeer]]:
    monkeypatch.setattr(worker, "LOOKUP_BREAKER_COOLDOWN_S", 0.5)
    monkeypatch.setattr(
        worker, "LOOKUP_BREAKER_FAILURE_THRESHOLD", getattr(request, "param", 1)
    )
    config = _make_config()
    peer = LookupPeer(config)
    client = worker.LookupKeyClient(config)
    try:
        yield client, peer
    finally:
        client.close()
        client.ctx.term()
        peer.close()


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
    lookup_service: tuple[worker.LookupKeyClient, LookupPeer], mode: str
) -> None:
    client, peer = lookup_service
    with ThreadPoolExecutor(max_workers=1) as requests:
        # The outer timeout also makes this test fail promptly on a hang.
        result = requests.submit(_resolve_lookup, client, "slow", mode).result(5)
    assert result.hit_length == 0
    assert client.circuit_breaker.is_open
    assert client.lookup("skip", 128, []).hit_length == 0
    # The open breaker skips the lookup without sending it.
    [slow] = peer.wait_for_requests(1)

    # Recovery must ignore the abandoned request's 32-token reply.
    peer.reply_hit(slow, 32)
    deadline = time.monotonic() + 5
    while client.circuit_breaker.is_open:
        assert time.monotonic() < deadline
        threading.Event().wait(0.001)
    peer.responder = lambda _: hit(16)
    assert client.lookup("recovered", 128, []).hit_length == 16
    assert not client.circuit_breaker.is_open
    assert len(peer.requests) == 2


def test_lookups_are_in_flight_together_and_matched_by_id(
    lookup_service: tuple[worker.LookupKeyClient, LookupPeer],
) -> None:
    client, peer = lookup_service
    for req_id in ("a", "b", "c"):
        assert client.lookup(req_id, 128, [], non_block=True) is None
    # All three reach the server before any reply.
    a, b, c = peer.wait_for_requests(3)
    for request, hit_length in ((c, 48), (a, 16), (b, 32)):
        peer.reply_hit(request, hit_length)
    assert _resolve_lookup(client, "a", "async").hit_length == 16
    assert _resolve_lookup(client, "b", "async").hit_length == 32
    assert _resolve_lookup(client, "c", "async").hit_length == 48
    assert not client.circuit_breaker.is_open


@pytest.mark.parametrize("status", [protocol.RESP_ERR, protocol.RESP_EXPIRED])
def test_server_error_or_expiry_returns_miss_and_counts_as_failure(
    lookup_service: tuple[worker.LookupKeyClient, LookupPeer], status: bytes
) -> None:
    client, peer = lookup_service
    peer.responder = lambda _: (status, b"")
    assert client.lookup("req", 128, []).hit_length == 0
    assert client.circuit_breaker.is_open


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
    lookup_service: tuple[worker.LookupKeyClient, LookupPeer],
) -> None:
    client, _ = lookup_service
    # The peer sends no replies; each request times out once.
    for index in range(3):
        assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
        assert client.circuit_breaker.is_open == (index == 2)


@pytest.mark.parametrize("lookup_service", [3], indirect=True)
def test_cached_hit_does_not_reset_newer_failures(
    lookup_service: tuple[worker.LookupKeyClient, LookupPeer],
) -> None:
    client, peer = lookup_service
    peer.responder = lambda request: hit(16) if len(peer.requests) == 1 else None
    assert _resolve_lookup(client, "cached", "async").hit_length == 16
    for index in range(3):
        assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
        assert client.circuit_breaker.is_open == (index == 2)
        if index < 2:
            assert client.lookup("cached", 128, [], non_block=True).hit_length == 16


# ------------------------------------------------------------------
# LookupKeyServer
# ------------------------------------------------------------------


@pytest.fixture
def lookup_server(
    request: pytest.FixtureRequest, rpc_dir: str
) -> Iterator[tuple[worker.LookupKeyServer, worker.LookupKeyClient, MagicMock]]:
    config = _make_config(
        lookup_server_threads=getattr(request, "param", 4),
        lookup_prefetch_limit=0,
    )
    store_worker = MagicMock()
    store_worker.kv_send_thread = None
    server = worker.LookupKeyServer(store_worker, config)
    client = worker.LookupKeyClient(config)
    try:
        yield server, client, store_worker
    finally:
        client.close()
        server.close()


@pytest.mark.parametrize("lookup_server", [3], indirect=True)
def test_server_runs_lookups_concurrently(
    lookup_server: tuple[worker.LookupKeyServer, worker.LookupKeyClient, MagicMock],
) -> None:
    _, client, store_worker = lookup_server
    # Each lookup returns only once all three are running at the same time.
    all_running = threading.Barrier(3, timeout=5)

    def lookup(num_tokens, block_hashes):
        all_running.wait()
        return MooncakeLookupResult(num_tokens // 2)

    store_worker.lookup.side_effect = lookup
    for index in range(3):
        assert client.lookup(f"req-{index}", 32 * (index + 1), [], True) is None
    for index in range(3):
        result = _resolve_lookup_tokens(client, f"req-{index}", 32 * (index + 1))
        assert result.hit_length == 16 * (index + 1)


def _resolve_lookup_tokens(
    client: worker.LookupKeyClient, req_id: str, num_tokens: int
) -> MooncakeLookupResult:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.lookup(req_id, num_tokens, [], non_block=True)
        if result is not None:
            return result
        threading.Event().wait(0.001)
    raise AssertionError("lookup remained pending")


@pytest.mark.parametrize("lookup_server", [1], indirect=True)
def test_server_skips_lookup_queued_past_its_deadline(
    lookup_server: tuple[worker.LookupKeyServer, worker.LookupKeyClient, MagicMock],
) -> None:
    server, client, store_worker = lookup_server
    release = threading.Event()

    def lookup(num_tokens, block_hashes):
        assert release.wait(5)
        return MooncakeLookupResult(16)

    store_worker.lookup.side_effect = lookup
    assert client.lookup("busy", 128, [], non_block=True) is None
    # Queued on the server behind "busy" until after the client gives up.
    assert client.lookup("queued", 128, []).hit_length == 0
    # The server's budget starts when the request arrives, a bit after the
    # client's; release only once that later deadline has passed too.
    threading.Event().wait(0.2)
    release.set()
    # The pool has one worker, so this runs after "queued" was handled.
    server._lookup_pool.submit(lambda: None).result(timeout=5)
    assert store_worker.lookup.call_count == 1


def test_server_answers_after_a_failed_lookup(
    lookup_server: tuple[worker.LookupKeyServer, worker.LookupKeyClient, MagicMock],
) -> None:
    _, client, store_worker = lookup_server
    store_worker.lookup.side_effect = [
        ValueError("bad lookup"),
        MooncakeLookupResult(8),
    ]
    assert client.lookup("bad", 128, []).hit_length == 0
    client.circuit_breaker.record_success()
    assert client.lookup("good", 128, []).hit_length == 8


def test_reset_waits_for_running_lookups(
    lookup_server: tuple[worker.LookupKeyServer, worker.LookupKeyClient, MagicMock],
) -> None:
    _, client, store_worker = lookup_server
    lookup_started, release = threading.Event(), threading.Event()

    def lookup(num_tokens, block_hashes):
        lookup_started.set()
        assert release.wait(5)
        return MooncakeLookupResult(16)

    store_worker.lookup.side_effect = lookup
    assert client.lookup("running", 128, [], non_block=True) is None
    assert lookup_started.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(client.reset)
        threading.Event().wait(0.2)
        store_worker.store.remove_all.assert_not_called()
        release.set()
        assert reset.result(timeout=5) is True
    store_worker.store.remove_all.assert_called_once_with(force=True)
