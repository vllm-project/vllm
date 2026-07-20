# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
import urllib.error
from types import SimpleNamespace

import aiohttp
import msgspec
import pytest
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


def test_prefix_cache_upload_recovers_after_disconnect(monkeypatch):
    attempts = []

    class Response:
        status = 204
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def urlopen(request, timeout):
        attempts.append((request.data, timeout))
        if len(attempts) == 1:
            raise urllib.error.URLError("disconnected")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
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


def _event_ingest_request(token: str | None, configured_token: str | None):
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
                ),
                nodes={"node-a": object()},
                scheduler=self.scheduler,
            )
            self.app = SimpleNamespace(
                state=SimpleNamespace(prefix_routing_proxy=proxy)
            )

        async def body(self):
            return msgspec.msgpack.encode(
                KVEventBatch(ts=1.0, data_parallel_rank=0, events=[])
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


def test_prefix_routing_falls_back_locally_when_upstream_cannot_start(
    monkeypatch,
):
    request_body = b'{"model":"test-model","prompt":"hello"}'
    local_calls = []
    sent = []

    class FailedRequest:
        async def __aenter__(self):
            raise aiohttp.ClientConnectionError("unreachable")

        async def __aexit__(self, *args):
            return None

    class ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def request(self, **kwargs):
            return FailedRequest()

    class Proxy:
        config = SimpleNamespace(request_timeout=1.0)
        nodes = {
            "node-a": PrefixRoutingNode(node_id="node-a", url="http://node-a:8000")
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

    monkeypatch.setattr(
        "vllm.entrypoints.openai.prefix_routing.aiohttp.ClientSession",
        lambda **kwargs: ClientSession(),
    )

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


def test_prefix_routing_stream_failure_does_not_restart_response(monkeypatch):
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
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def request(self, **kwargs):
            request_headers.extend(kwargs["headers"])
            return RequestContext()

    async def send(message):
        sent.append(message)

    monkeypatch.setattr(
        "vllm.entrypoints.openai.prefix_routing.aiohttp.ClientSession",
        lambda **kwargs: ClientSession(),
    )
    scope = {
        "method": "POST",
        "path": "/v1/completions",
        "query_string": b"",
        "headers": [],
    }
    node = PrefixRoutingNode(node_id="node-a", url="http://node-a:8000")

    with pytest.raises(aiohttp.ClientPayloadError, match="stream interrupted"):
        asyncio.run(
            _forward_request(
                scope,
                b"{}",
                send,
                node,
                request_timeout=1.0,
                data_parallel_rank=3,
            )
        )

    assert ("x-data-parallel-rank", "3") in request_headers
    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    assert sum(message["type"] == "http.response.start" for message in sent) == 1


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
    ],
)
def test_prefix_routing_config_rejects_invalid_values(config, error):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(hash_block_size=16, block_size=16)
    )

    with pytest.raises(ValueError, match=error):
        _parse_prefix_routing_config(config, vllm_config)
