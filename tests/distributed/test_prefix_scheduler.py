# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import http.server
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import aiohttp
import msgspec
import pytest
import zmq
from fastapi import HTTPException

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    HttpPrefixCacheEventUploader,
    KVEventBatch,
    NullEventPublisher,
    PrefixCacheEventUploaderFactory,
    _build_prefix_cache_opener,
    _NoRedirectHandler,
)
from vllm.distributed.prefix_scheduler import (
    GlobalPrefixScheduler,
    NodePrefixCacheState,
    PrefixCacheSnapshot,
)
from vllm.entrypoints.openai.prefix_routing import (
    PrefixRoutingConfig,
    PrefixRoutingMiddleware,
    PrefixRoutingNode,
    PrefixRoutingProxy,
    _forward_request,
    _parse_prefix_routing_config,
    ingest_prefix_routing_kv_events,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)

pytestmark = pytest.mark.skip_global_cleanup


def _hash(value: int) -> BlockHash:
    return BlockHash(bytes([value]) * 32)


def _external_hash(value: BlockHash) -> ExternalBlockHash:
    return maybe_convert_block_hash(value)


def test_node_prefix_cache_state_applies_events_and_matches_longest_prefix():
    state = NodePrefixCacheState(node_id="node-a", hash_block_size=16)
    block_hashes = [_hash(1), _hash(2), _hash(3)]

    state.apply_events(
        [
            BlockStored(
                block_hashes=[
                    _external_hash(block_hashes[0]),
                    _external_hash(block_hashes[1]),
                ],
                parent_block_hash=None,
                token_ids=[],
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
                group_idx=0,
            )
        ]
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 32

    state.apply_events(
        [
            BlockRemoved(
                block_hashes=[_external_hash(block_hashes[1])],
                medium="GPU",
                group_idx=0,
            )
        ]
    )
    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 16

    state.apply_events([AllBlocksCleared()])
    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=64) == 0


def test_global_prefix_scheduler_routes_to_longest_match():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1), _hash(2), _hash(3)]

    scheduler.update_snapshot(
        PrefixCacheSnapshot(
            node_id="node-a",
            data_parallel_rank=0,
            hash_block_size=16,
            group_block_sizes={0: 16},
            group_hashes={0: {_external_hash(block_hashes[0])}},
        )
    )
    scheduler.update_snapshot(
        PrefixCacheSnapshot(
            node_id="node-b",
            data_parallel_rank=1,
            hash_block_size=16,
            group_block_sizes={0: 16},
            group_hashes={
                0: {_external_hash(block_hashes[0]), _external_hash(block_hashes[1])}
            },
        )
    )

    decision = scheduler.choose_node(block_hashes, prompt_num_tokens=64)

    assert decision is not None
    assert decision.node_id == "node-b"
    assert decision.data_parallel_rank == 1
    assert decision.matched_tokens == 32


def test_global_prefix_scheduler_round_robins_ties():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1)]
    for node_id in ("node-a", "node-b"):
        scheduler.update_snapshot(
            PrefixCacheSnapshot(
                node_id=node_id,
                data_parallel_rank=None,
                hash_block_size=16,
                group_block_sizes={0: 16},
                group_hashes={0: {_external_hash(block_hashes[0])}},
            )
        )

    decisions = [
        scheduler.choose_node(block_hashes, prompt_num_tokens=32).node_id
        for _ in range(4)
    ]

    assert decisions == ["node-a", "node-b", "node-a", "node-b"]


def test_global_prefix_scheduler_applies_event_batch_rank():
    scheduler = GlobalPrefixScheduler()
    block_hashes = [_hash(1)]
    scheduler.register_node("node-a", hash_block_size=16)

    scheduler.apply_event_batch(
        "node-a",
        KVEventBatch(
            ts=1.0,
            data_parallel_rank=3,
            events=[
                BlockStored(
                    block_hashes=[_external_hash(block_hashes[0])],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                    group_idx=0,
                )
            ],
        ),
    )

    decision = scheduler.choose_node(block_hashes, prompt_num_tokens=32)

    assert decision is not None
    assert decision.data_parallel_rank == 3
    assert decision.matched_tokens == 16


def test_global_prefix_scheduler_isolates_data_parallel_rank_state():
    scheduler = GlobalPrefixScheduler()
    rank_zero_hash = _hash(1)
    rank_one_hash = _hash(2)
    scheduler.register_node("node-a", hash_block_size=16)

    scheduler.apply_event_batch(
        "node-a",
        KVEventBatch(
            ts=1.0,
            data_parallel_rank=0,
            events=[_stored_event(rank_zero_hash)],
        ),
    )
    scheduler.apply_event_batch(
        "node-a",
        KVEventBatch(
            ts=2.0,
            data_parallel_rank=1,
            events=[_stored_event(rank_one_hash)],
        ),
    )

    rank_zero = scheduler.choose_node([rank_zero_hash], prompt_num_tokens=32)
    rank_one = scheduler.choose_node([rank_one_hash], prompt_num_tokens=32)

    assert rank_zero is not None
    assert rank_zero.data_parallel_rank == 0
    assert rank_zero.matched_tokens == 16
    assert rank_one is not None
    assert rank_one.data_parallel_rank == 1
    assert rank_one.matched_tokens == 16


def test_fixed_rank_node_rejects_mismatched_batches_and_snapshots():
    scheduler = GlobalPrefixScheduler()
    accepted_hash = _hash(1)
    rejected_hash = _hash(2)
    scheduler.register_node(
        "node-a",
        hash_block_size=16,
        data_parallel_rank=1,
        group_block_sizes={0: 16},
    )
    scheduler.apply_event_batch(
        "node-a",
        KVEventBatch(
            ts=1.0,
            data_parallel_rank=1,
            events=[_stored_event(accepted_hash)],
        ),
    )

    with pytest.raises(ValueError, match="configured.*rank 1.*reported rank 2"):
        scheduler.apply_event_batch(
            "node-a",
            KVEventBatch(
                ts=2.0,
                data_parallel_rank=2,
                events=[_stored_event(rejected_hash)],
            ),
        )
    with pytest.raises(ValueError, match="configured.*rank 1.*reported rank 2"):
        scheduler.update_snapshot(
            PrefixCacheSnapshot(
                node_id="node-a",
                data_parallel_rank=2,
                hash_block_size=16,
                group_block_sizes={0: 16},
                group_hashes={0: {_external_hash(rejected_hash)}},
            )
        )

    accepted = scheduler.choose_node([accepted_hash], prompt_num_tokens=32)
    rejected = scheduler.choose_node([rejected_hash], prompt_num_tokens=32)
    assert accepted is not None
    assert accepted.data_parallel_rank == 1
    assert accepted.matched_tokens == 16
    assert rejected is not None
    assert rejected.matched_tokens == 0


def test_node_prefix_cache_state_matches_larger_group_block_size():
    block_hashes = [_hash(1), _hash(2), _hash(3), _hash(4)]
    state = NodePrefixCacheState(
        node_id="node-a",
        hash_block_size=16,
        group_block_sizes={0: 32},
        group_hashes={0: {_external_hash(block_hashes[1])}},
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=80) == 32


def test_node_prefix_cache_state_requires_a_hit_in_every_cache_group():
    block_hashes = [_hash(1), _hash(2)]
    state = NodePrefixCacheState(
        node_id="node-a",
        hash_block_size=16,
        group_block_sizes={0: 16, 1: 16},
        group_hashes={0: {_external_hash(block_hashes[0])}},
    )

    assert state.longest_prefix_match(block_hashes, prompt_num_tokens=32) == 0


def test_prefix_cache_upload_is_independent_from_event_publisher():
    config = KVEventsConfig(
        enable_kv_cache_events=False,
        publisher="null",
        prefix_cache_upload_endpoint="http://127.0.0.1:9/prefix_routing",
        prefix_cache_upload_token="shared-secret",
    )

    publisher = EventPublisherFactory.create(config)
    uploader = PrefixCacheEventUploaderFactory.create(config)

    try:
        assert isinstance(publisher, NullEventPublisher)
        assert isinstance(uploader, HttpPrefixCacheEventUploader)
        assert uploader._headers["Authorization"] == "Bearer shared-secret"
    finally:
        uploader.shutdown()


def _stored_event(block_hash, *, group_idx=0):
    return BlockStored(
        block_hashes=[_external_hash(block_hash)],
        parent_block_hash=None,
        token_ids=[],
        block_size=16,
        lora_id=None,
        medium="GPU",
        lora_name=None,
        group_idx=group_idx,
    )


def test_prefix_cache_upload_queue_overflow_reconciles_latest_state():
    uploader = HttpPrefixCacheEventUploader(
        data_parallel_rank=0,
        endpoint="http://127.0.0.1:9/prefix_routing",
        max_queue_size=1,
        token="shared-secret",
        _start_thread=False,
    )
    first_hash = _hash(1)
    second_hash = _hash(2)

    uploader.publish(KVEventBatch(ts=1.0, events=[_stored_event(first_hash)]))
    uploader.publish(KVEventBatch(ts=2.0, events=[_stored_event(second_hash)]))

    assert uploader._needs_snapshot.is_set()
    uploader._discard_queued_batches()
    snapshot, _ = uploader._build_snapshot_batch()
    state = NodePrefixCacheState(node_id="node-a", hash_block_size=16)
    state.apply_events(snapshot.events)
    # Reapplying a reconciliation snapshot is idempotent.
    state.apply_events(snapshot.events)
    # A later reconciliation also repairs an out-of-order stale delta.
    state.apply_events(
        [
            BlockRemoved(
                block_hashes=[_external_hash(first_hash)],
                medium="GPU",
                group_idx=0,
            )
        ]
    )
    state.apply_events(snapshot.events)
    assert state.group_hashes[0] == {
        _external_hash(first_hash),
        _external_hash(second_hash),
    }
    uploader.shutdown()


def test_prefix_cache_upload_recovers_after_disconnect():
    attempts = []

    class Response:
        status = 204
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Opener:
        def open(self, request, timeout):
            attempts.append((request.data, timeout))
            if len(attempts) == 1:
                raise urllib.error.URLError("disconnected")
            return Response()

    uploader = HttpPrefixCacheEventUploader(
        data_parallel_rank=0,
        endpoint="http://127.0.0.1:9/prefix_routing",
        token="shared-secret",
        initial_snapshot=PrefixCacheSnapshot(
            node_id="node-a",
            hash_block_size=16,
            group_block_sizes={0: 16},
            group_hashes={0: {_external_hash(_hash(1))}},
        ),
        retry_interval=0.001,
        max_retry_interval=0.002,
        _opener=Opener(),
    )
    deadline = time.monotonic() + 2
    while uploader._needs_snapshot.is_set() and time.monotonic() < deadline:
        time.sleep(0.005)

    try:
        assert len(attempts) >= 2
        assert not uploader._needs_snapshot.is_set()
        recovered = msgspec.msgpack.decode(attempts[-1][0], type=KVEventBatch)
        assert isinstance(recovered.events[0], AllBlocksCleared)
    finally:
        uploader.shutdown()


def test_prefix_cache_upload_ignores_proxy_and_rejects_redirect(monkeypatch):
    source_requests = []
    redirected_requests = []

    class RedirectTarget(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            redirected_requests.append(dict(self.headers))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectTarget)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    target_url = f"http://127.0.0.1:{target.server_port}/stolen"

    class RedirectSource(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            source_requests.append(dict(self.headers))
            self.send_response(307)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    source = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectSource)
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()

    # A process-level proxy must not receive authenticated cache events. Using
    # an unreachable endpoint also proves the uploader is not silently falling
    # back to environment proxy handling.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    uploader = HttpPrefixCacheEventUploader(
        data_parallel_rank=0,
        endpoint=f"http://127.0.0.1:{source.server_port}/events",
        token="shared-secret",
        retry_interval=0.01,
        max_retry_interval=0.01,
    )
    uploader.publish(KVEventBatch(ts=1.0, events=[]))
    deadline = time.monotonic() + 2
    while (
        not source_requests or not uploader._needs_snapshot.is_set()
    ) and time.monotonic() < deadline:
        time.sleep(0.005)

    try:
        assert source_requests
        assert source_requests[0]["Authorization"] == "Bearer shared-secret"
        assert not redirected_requests
        assert uploader._needs_snapshot.is_set()
        assert not any(
            isinstance(handler, urllib.request.ProxyHandler)
            for handler in uploader._opener.handlers
        )
        assert any(
            isinstance(handler, _NoRedirectHandler)
            for handler in uploader._opener.handlers
        )
    finally:
        uploader.shutdown()
        source.shutdown()
        target.shutdown()
        source.server_close()
        target.server_close()
        source_thread.join(timeout=1)
        target_thread.join(timeout=1)


def test_prefix_cache_opener_has_no_environment_proxy():
    opener = _build_prefix_cache_opener()

    assert not any(
        isinstance(handler, urllib.request.ProxyHandler) for handler in opener.handlers
    )


def _event_ingest_request(
    token: str | None,
    configured_token: str | None,
    *,
    body: bytes | None = None,
    max_event_ingest_body_size: int = 16 * 1024 * 1024,
):
    class Scheduler:
        batch = None

        def apply_event_batch(self, node_id, batch):
            self.batch = (node_id, batch)

    class Request:
        headers = {} if token is None else {"authorization": f"Bearer {token}"}

        def __init__(self):
            self.scheduler = Scheduler()
            proxy = SimpleNamespace(
                config=PrefixRoutingConfig(
                    nodes=[],
                    hash_block_size=16,
                    event_ingest_token=configured_token,
                    max_event_ingest_body_size=max_event_ingest_body_size,
                ),
                nodes={"node-a": object()},
                scheduler=self.scheduler,
            )
            self.app = SimpleNamespace(
                state=SimpleNamespace(prefix_routing_proxy=proxy)
            )

        async def stream(self):
            yield (
                body
                if body is not None
                else msgspec.msgpack.encode(
                    KVEventBatch(ts=1.0, data_parallel_rank=0, events=[])
                )
            )

    return Request()


def test_prefix_routing_http_event_ingest_is_disabled_without_token():
    request = _event_ingest_request(token=None, configured_token=None)

    with pytest.raises(HTTPException, match="ingestion is disabled") as exc_info:
        asyncio.run(ingest_prefix_routing_kv_events("node-a", request))

    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("token", [None, "wrong-secret"])
def test_prefix_routing_http_event_ingest_rejects_invalid_token(token):
    request = _event_ingest_request(token=token, configured_token="shared-secret")

    with pytest.raises(HTTPException, match="Invalid.*credential") as exc_info:
        asyncio.run(ingest_prefix_routing_kv_events("node-a", request))

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def test_prefix_routing_http_event_ingest_accepts_shared_token():
    request = _event_ingest_request(
        token="shared-secret", configured_token="shared-secret"
    )

    response = asyncio.run(ingest_prefix_routing_kv_events("node-a", request))

    assert response.status_code == 204
    assert request.scheduler.batch is not None
    assert request.scheduler.batch[0] == "node-a"


def test_prefix_routing_http_event_ingest_rejects_oversized_batch():
    request = _event_ingest_request(
        token="shared-secret",
        configured_token="shared-secret",
        body=b"x" * 9,
        max_event_ingest_body_size=8,
    )

    with pytest.raises(HTTPException, match="batch is too large") as exc_info:
        asyncio.run(ingest_prefix_routing_kv_events("node-a", request))

    assert exc_info.value.status_code == 413
    assert request.scheduler.batch is None


def test_prefix_routing_renders_completion_engine_inputs():
    class Renderer:
        async def render_completion(self, request):
            return [
                {"prompt_token_ids": [1, 2], "cache_salt": "salt"},
                {"prompt_token_ids": [3]},
            ]

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.app_state = SimpleNamespace(online_renderer=Renderer())

    rendered = asyncio.run(
        proxy._render_request(
            "/v1/completions",
            {"model": "test-model", "prompt": ["first", "second"]},
        )
    )

    assert rendered == [([1, 2], "salt"), ([3], None)]


def test_prefix_routing_renders_chat_engine_inputs():
    class Renderer:
        async def render_chat(self, request):
            return [], [{"prompt_token_ids": [4, 5], "cache_salt": "chat-salt"}]

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.app_state = SimpleNamespace(online_renderer=Renderer())

    rendered = asyncio.run(
        proxy._render_request(
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )

    assert rendered == [([4, 5], "chat-salt")]


def test_prefix_routing_falls_back_locally_when_upstream_cannot_start():
    request_body = b'{"model":"test-model","prompt":"hello"}'
    local_calls = []
    sent = []

    class FailedRequest:
        async def __aenter__(self):
            raise aiohttp.ClientConnectionError("unreachable")

        async def __aexit__(self, *args):
            return None

    class ClientSession:
        def request(self, **kwargs):
            return FailedRequest()

    class Proxy:
        config = SimpleNamespace(
            routing_token="incoming-secret", max_request_body_size=1024
        )
        session = ClientSession()
        nodes = {
            "node-a": PrefixRoutingNode(
                node_id="node-a",
                url="http://node-a:8000",
                routing_token="outgoing-secret",
            )
        }

        async def choose_node_for_request(self, path, payload):
            return SimpleNamespace(
                node_id="node-a", matched_tokens=16, data_parallel_rank=2
            )

    async def app(scope, receive, send):
        local_calls.append((scope, await receive()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"local"})

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"router")],
        "server": ("router", 8000),
        "app": SimpleNamespace(state=SimpleNamespace(prefix_routing_proxy=Proxy())),
    }
    received = False

    async def receive():
        nonlocal received
        assert not received
        received = True
        return {"type": "http.request", "body": request_body}

    async def send(message):
        sent.append(message)

    asyncio.run(PrefixRoutingMiddleware(app)(scope, receive, send))

    assert len(local_calls) == 1
    routed_scope, replayed = local_calls[0]
    assert all(
        key.lower() != b"x-data-parallel-rank" for key, _ in routed_scope["headers"]
    )
    assert replayed["body"] == request_body
    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    assert sent[0]["status"] == 200


def test_prefix_routing_stream_failure_does_not_restart_response():
    sent = []
    request_headers = []

    class Content:
        async def iter_chunked(self, size):
            yield b"partial"
            raise aiohttp.ClientPayloadError("stream interrupted")

    class Response:
        status = 200
        headers = {"content-type": "text/event-stream"}
        content = Content()

    class RequestContext:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            return None

    class ClientSession:
        def request(self, **kwargs):
            assert kwargs["allow_redirects"] is False
            request_headers.extend(kwargs["headers"])
            return RequestContext()

    async def send(message):
        sent.append(message)

    scope = {
        "method": "POST",
        "path": "/v1/completions",
        "query_string": b"",
        "headers": [
            (b"x-vllm-prefix-routing", b"forged"),
            (b"x-data-parallel-rank", b"99"),
        ],
    }
    node = PrefixRoutingNode(
        node_id="node-a",
        url="http://node-a:8000",
        routing_token="outgoing-secret",
    )

    with pytest.raises(aiohttp.ClientPayloadError, match="stream interrupted"):
        asyncio.run(
            _forward_request(
                scope,
                b"{}",
                send,
                node,
                ClientSession(),
                data_parallel_rank=3,
            )
        )

    assert ("x-data-parallel-rank", "3") in request_headers
    assert ("x-vllm-prefix-routing", "outgoing-secret") in request_headers
    assert ("x-vllm-prefix-routing", "forged") not in request_headers
    assert ("x-data-parallel-rank", "99") not in request_headers
    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    assert sum(message["type"] == "http.response.start" for message in sent) == 1


def test_prefix_routing_rejects_forged_bypass_and_external_rank():
    request_body = b'{"model":"test-model","prompt":"hello"}'
    routed_scopes = []
    choose_calls = 0

    class Proxy:
        config = SimpleNamespace(
            routing_token="shared-secret", max_request_body_size=1024
        )
        nodes = {"node-a": PrefixRoutingNode(node_id="node-a", url=None, local=True)}

        async def choose_node_for_request(self, path, payload):
            nonlocal choose_calls
            choose_calls += 1
            return SimpleNamespace(
                node_id="node-a", matched_tokens=16, data_parallel_rank=2
            )

    async def app(scope, receive, send):
        routed_scopes.append(scope)
        assert (await receive())["body"] == request_body

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"x-vllm-prefix-routing", b"bypass"),
            (b"x-data-parallel-rank", b"99"),
        ],
        "server": ("router", 8000),
        "app": SimpleNamespace(state=SimpleNamespace(prefix_routing_proxy=Proxy())),
    }

    async def receive():
        return {"type": "http.request", "body": request_body}

    async def send(message):
        raise AssertionError(f"unexpected response: {message}")

    asyncio.run(PrefixRoutingMiddleware(app)(scope, receive, send))

    assert choose_calls == 1
    assert len(routed_scopes) == 1
    assert (b"x-data-parallel-rank", b"2") in routed_scopes[0]["headers"]
    assert all(
        value != b"99"
        for key, value in routed_scopes[0]["headers"]
        if key.lower() == b"x-data-parallel-rank"
    )
    assert all(
        key.lower() != b"x-vllm-prefix-routing"
        for key, _ in routed_scopes[0]["headers"]
    )


def test_prefix_routing_accepts_authenticated_internal_bypass():
    app_scopes = []

    class Proxy:
        config = SimpleNamespace(routing_token="shared-secret")

        async def choose_node_for_request(self, path, payload):
            raise AssertionError("authenticated bypass must not be routed again")

    async def app(scope, receive, send):
        app_scopes.append(scope)
        await receive()

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"x-vllm-prefix-routing", b"shared-secret"),
            (b"x-data-parallel-rank", b"4"),
        ],
        "server": ("router", 8000),
        "app": SimpleNamespace(state=SimpleNamespace(prefix_routing_proxy=Proxy())),
    }

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    async def send(message):
        raise AssertionError(f"unexpected response: {message}")

    asyncio.run(PrefixRoutingMiddleware(app)(scope, receive, send))

    assert len(app_scopes) == 1
    assert (b"x-data-parallel-rank", b"4") in app_scopes[0]["headers"]
    assert all(
        key.lower() != b"x-vllm-prefix-routing" for key, _ in app_scopes[0]["headers"]
    )


def test_prefix_routing_rejects_oversized_request_body():
    sent = []
    choose_called = False
    chunks = iter(
        [
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"56", "more_body": False},
        ]
    )

    class Proxy:
        config = SimpleNamespace(routing_token=None, max_request_body_size=5)

        async def choose_node_for_request(self, path, payload):
            nonlocal choose_called
            choose_called = True

    async def app(scope, receive, send):
        raise AssertionError("oversized request must not reach the local app")

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "server": ("router", 8000),
        "app": SimpleNamespace(state=SimpleNamespace(prefix_routing_proxy=Proxy())),
    }

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    asyncio.run(PrefixRoutingMiddleware(app)(scope, receive, send))

    assert not choose_called
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert b"too large" in sent[1]["body"]


def test_prefix_routing_proxy_reuses_and_closes_http_session(monkeypatch):
    sessions = []

    class ClientSession:
        def __init__(self, **kwargs):
            self.closed = False
            sessions.append(self)

        async def close(self):
            self.closed = True

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.config = SimpleNamespace(request_timeout=1.0)
    proxy.nodes = {
        "node-a": PrefixRoutingNode(
            node_id="node-a",
            url="http://node-a:8000",
            routing_token="outgoing-secret",
        )
    }
    proxy._tasks = []
    proxy._session = None
    monkeypatch.setattr(
        "vllm.entrypoints.openai.prefix_routing.aiohttp.ClientSession", ClientSession
    )

    async def exercise_lifecycle():
        proxy.start()
        first_session = proxy.session
        proxy.start()
        assert proxy.session is first_session
        await proxy.shutdown()
        return first_session

    session = asyncio.run(exercise_lifecycle())

    assert sessions == [session]
    assert session.closed
    assert proxy._session is None


def test_prefix_routing_zmq_subscriber_isolates_invalid_messages(monkeypatch):
    valid_hash = _hash(7)
    mismatched = msgspec.msgpack.encode(
        KVEventBatch(
            ts=1.0,
            data_parallel_rank=2,
            events=[_stored_event(_hash(6))],
        )
    )
    valid = msgspec.msgpack.encode(
        KVEventBatch(
            ts=2.0,
            data_parallel_rank=1,
            events=[_stored_event(valid_hash)],
        )
    )

    class Socket:
        def __init__(self):
            self.messages = iter(
                [
                    [b"too", b"short"],
                    [b"", b"0", b"\xc1"],
                    [b"", b"1", mismatched],
                    [b"", b"2", valid],
                ]
            )
            self.closed = False

        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            assert endpoint == "inproc://node-a"

        async def recv_multipart(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise asyncio.CancelledError from None

        def close(self, *, linger):
            assert linger == 0
            self.closed = True

    socket = Socket()
    context = SimpleNamespace(socket=lambda socket_type: socket)
    monkeypatch.setattr(zmq.asyncio.Context, "instance", lambda: context)

    proxy = object.__new__(PrefixRoutingProxy)
    proxy.config = SimpleNamespace(event_topic="")
    proxy.scheduler = GlobalPrefixScheduler()
    proxy.scheduler.register_node(
        "node-a",
        hash_block_size=16,
        data_parallel_rank=1,
        group_block_sizes={0: 16},
    )
    node = PrefixRoutingNode(
        node_id="node-a",
        url=None,
        event_endpoint="inproc://node-a",
        data_parallel_rank=1,
        local=True,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(proxy._subscribe_node_events(node))

    decision = proxy.scheduler.choose_node([valid_hash], prompt_num_tokens=32)
    assert decision is not None
    assert decision.node_id == "node-a"
    assert decision.data_parallel_rank == 1
    assert decision.matched_tokens == 16
    assert socket.closed


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"max_queue_size": 0}, "max_queue_size must be a positive integer"),
        ({"max_queue_size": True}, "max_queue_size must be a positive integer"),
        ({"request_timeout": 0}, "request_timeout must be positive"),
        ({"request_timeout": True}, "request_timeout must be positive"),
    ],
)
def test_prefix_cache_uploader_rejects_invalid_limits(kwargs, error):
    with pytest.raises(ValueError, match=error):
        HttpPrefixCacheEventUploader(
            data_parallel_rank=0,
            endpoint="http://127.0.0.1:9/events",
            token="shared-secret",
            _start_thread=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (
            {"prefix_cache_upload_max_queue_size": 0},
            "prefix_cache_upload_max_queue_size must be a positive integer",
        ),
        (
            {"prefix_cache_upload_timeout": 0},
            "prefix_cache_upload_timeout must be positive",
        ),
    ],
)
def test_prefix_cache_config_rejects_invalid_upload_limits(kwargs, error):
    with pytest.raises(ValueError, match=error):
        KVEventsConfig(
            prefix_cache_upload_endpoint="http://127.0.0.1:9/events",
            prefix_cache_upload_token="shared-secret",
            **kwargs,
        )


@pytest.mark.parametrize(
    "config, error",
    [
        (
            {
                "nodes": [
                    {"id": "node-a", "url": "local"},
                    {"id": "node-a", "url": "local"},
                ]
            },
            "duplicate prefix routing node id",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "ftp://node-a"}]},
            r"requires an HTTP\(S\) URL",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "local"}], "hash_block_size": 0},
            "hash_block_size must be a positive integer",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "local"}], "request_timeout": 0},
            "request_timeout must be positive",
        ),
        (
            {"nodes": [{"id": "node-a", "url": "http://node-a:8000"}]},
            "requires a non-empty routing_token",
        ),
        (
            {
                "nodes": [{"id": "node-a", "url": "local"}],
                "routing_token": "",
            },
            "routing_token must be a non-empty string",
        ),
        (
            {
                "nodes": [{"id": "node-a", "url": "local"}],
                "max_request_body_size": 0,
            },
            "max_request_body_size must be a positive integer",
        ),
        (
            {
                "nodes": [{"id": "node-a", "url": "local"}],
                "max_event_ingest_body_size": 0,
            },
            "max_event_ingest_body_size must be a positive integer",
        ),
        (
            {
                "nodes": [{"id": "node-a", "url": "local"}],
                "max_event_ingest_body_size": True,
            },
            "max_event_ingest_body_size must be a positive integer",
        ),
    ],
)
def test_prefix_routing_config_rejects_invalid_values(config, error):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(hash_block_size=16, block_size=16)
    )

    with pytest.raises(ValueError, match=error):
        _parse_prefix_routing_config(config, vllm_config)


def test_api_app_wires_prefix_routing_middleware_route_and_shutdown(monkeypatch):
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.platforms import current_platform
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    parser = FlexibleArgumentParser()
    make_arg_parser(parser)
    args = parser.parse_args([])
    args.enable_prefix_routing = True
    app = build_app(args, ())

    assert any(
        middleware.cls is PrefixRoutingMiddleware for middleware in app.user_middleware
    )
    assert any(
        getattr(route, "path", None) == "/prefix_routing/kv_events/{node_id}"
        for route in app.routes
    )

    class Proxy:
        def __init__(self):
            self.shutdown_called = False

        async def shutdown(self):
            self.shutdown_called = True

    proxy = Proxy()
    app.state.log_stats = False
    app.state.prefix_routing_proxy = proxy

    async def exercise_lifespan():
        async with app.router.lifespan_context(app):
            assert not proxy.shutdown_called

    asyncio.run(exercise_lifespan())

    assert proxy.shutdown_called
