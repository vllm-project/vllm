# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI API prefix-routing proxy integration."""

import asyncio
import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
import msgspec
import zmq
import zmq.asyncio
from fastapi import APIRouter, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import Response
from starlette.datastructures import URL, Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.prefix_scheduler import (
    GlobalPrefixScheduler,
    PrefixRouteDecision,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request as VllmRequest

logger = init_logger(__name__)

PREFIX_ROUTING_BYPASS_HEADER = "x-vllm-prefix-routing"
DATA_PARALLEL_RANK_HEADER = "x-data-parallel-rank"
DEFAULT_MAX_REQUEST_BODY_SIZE = 16 * 1024 * 1024

router = APIRouter()


@router.post("/prefix_routing/kv_events/{node_id}")
async def ingest_prefix_routing_kv_events(
    node_id: str, raw_request: FastAPIRequest
) -> Response:
    """Receive worker KV-cache events and update the global prefix index."""
    proxy = getattr(raw_request.app.state, "prefix_routing_proxy", None)
    if proxy is None:
        raise HTTPException(status_code=503, detail="Prefix routing is not enabled")

    expected_token = proxy.config.event_ingest_token
    if expected_token is None:
        raise HTTPException(
            status_code=503, detail="Prefix routing HTTP event ingestion is disabled"
        )
    authorization = raw_request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    token_matches = scheme.lower() == "bearer" and secrets.compare_digest(
        hashlib.sha256(token.encode("utf-8")).digest(),
        hashlib.sha256(expected_token.encode("utf-8")).digest(),
    )
    if not token_matches:
        raise HTTPException(
            status_code=401,
            detail="Invalid prefix routing event credential",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if node_id not in proxy.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown prefix node {node_id!r}")

    body = await raw_request.body()
    try:
        batch = msgspec.msgpack.decode(body, type=KVEventBatch)
    except msgspec.DecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid KV event batch") from exc

    try:
        proxy.scheduler.apply_event_batch(node_id, batch)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="KV event rank does not match the configured prefix node",
        ) from exc
    return Response(status_code=204)


def attach_prefix_routing_router(app: Any) -> None:
    app.include_router(router)


@dataclass(frozen=True)
class PrefixRoutingNode:
    node_id: str
    url: str | None
    event_endpoint: str | None = None
    data_parallel_rank: int | None = None
    routing_token: str | None = None
    local: bool = False


@dataclass(frozen=True)
class PrefixRoutingConfig:
    nodes: list[PrefixRoutingNode]
    hash_block_size: int
    event_topic: str = ""
    request_timeout: float = 6 * 60 * 60
    event_ingest_token: str | None = None
    routing_token: str | None = None
    max_request_body_size: int = DEFAULT_MAX_REQUEST_BODY_SIZE


class PrefixRoutingProxy:
    def __init__(
        self,
        *,
        config: PrefixRoutingConfig,
        app_state: Any,
    ) -> None:
        self.config = config
        self.app_state = app_state
        self.scheduler = GlobalPrefixScheduler()
        self.nodes = {node.node_id: node for node in config.nodes}
        self._tasks: list[asyncio.Task] = []
        self._session: aiohttp.ClientSession | None = None

        vllm_config = app_state.vllm_config
        caching_hash_fn = get_hash_fn_by_name(
            vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)
        self._block_hasher = get_request_block_hasher(
            config.hash_block_size, caching_hash_fn
        )

        group_block_sizes = {0: config.hash_block_size}
        for node in config.nodes:
            self.scheduler.register_node(
                node.node_id,
                hash_block_size=config.hash_block_size,
                data_parallel_rank=node.data_parallel_rank,
                group_block_sizes=group_block_sizes,
            )

    def start(self) -> None:
        if self._session is None and any(
            not node.local and node.url is not None for node in self.nodes.values()
        ):
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )
        for node in self.nodes.values():
            if node.event_endpoint is None:
                continue
            task = asyncio.create_task(self._subscribe_node_events(node))
            self._tasks.append(task)

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("prefix routing HTTP session is not initialized")
        return self._session

    async def choose_node_for_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> PrefixRouteDecision | None:
        rendered_requests = await self._render_request(path, payload)
        if not rendered_requests:
            return None

        best_decision: PrefixRouteDecision | None = None
        for token_ids, cache_salt in rendered_requests:
            block_hashes = self._make_block_hashes(token_ids, cache_salt)
            decision = self.scheduler.choose_node(
                block_hashes=block_hashes,
                prompt_num_tokens=len(token_ids),
            )
            if decision is None:
                continue
            if (
                best_decision is None
                or decision.matched_tokens > best_decision.matched_tokens
            ):
                best_decision = decision
        return best_decision

    async def _render_request(
        self, path: str, payload: Mapping[str, Any]
    ) -> list[tuple[list[int], str | None]]:
        renderer = self.app_state.online_renderer
        if path == "/v1/completions":
            completion_request = CompletionRequest.model_validate(payload)
            result = await renderer.render_completion(completion_request)
            if isinstance(result, ErrorResponse):
                return []
            return [
                (list(engine_input["prompt_token_ids"]), engine_input.get("cache_salt"))
                for engine_input in result
            ]
        if path == "/v1/chat/completions":
            chat_request = ChatCompletionRequest.model_validate(payload)
            result = await renderer.render_chat(chat_request)
            if isinstance(result, ErrorResponse):
                return []
            _, engine_inputs = result
            return [
                (list(engine_input["prompt_token_ids"]), engine_input.get("cache_salt"))
                for engine_input in engine_inputs
            ]
        return []

    def _make_block_hashes(
        self, token_ids: list[int], cache_salt: str | None
    ) -> list[BlockHash]:
        request = VllmRequest(
            request_id="prefix-routing",
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            cache_salt=cache_salt,
            block_hasher=self._block_hasher,
        )
        return request.block_hashes

    async def _subscribe_node_events(self, node: PrefixRoutingNode) -> None:
        assert node.event_endpoint is not None
        ctx = zmq.asyncio.Context.instance()
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        topic = self.config.event_topic.encode("utf-8")
        socket = ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.connect(node.event_endpoint)
        logger.info(
            "Prefix routing subscribed to KV events from %s at %s",
            node.node_id,
            node.event_endpoint,
        )
        try:
            while True:
                frames = await socket.recv_multipart()
                if len(frames) != 3:
                    logger.warning("Ignoring malformed KV event frames: %s", frames)
                    continue
                _, _, payload = frames
                batch = decoder.decode(payload)
                self.scheduler.apply_event_batch(node.node_id, batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Prefix routing event subscriber failed for %s", node.node_id
            )
        finally:
            socket.close(linger=0)


class PrefixRoutingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        path = URL(scope=scope).path.removeprefix(root_path)
        method = scope.get("method")
        headers = Headers(scope=scope)
        if method != "POST" or path not in ("/v1/completions", "/v1/chat/completions"):
            await self.app(scope, receive, send)
            return

        proxy = getattr(scope["app"].state, "prefix_routing_proxy", None)
        if proxy is None:
            await self.app(scope, receive, send)
            return

        if _has_authenticated_bypass(headers, proxy.config.routing_token):
            await self.app(_without_bypass_header(scope), receive, send)
            return

        scope = _without_external_routing_headers(scope)
        try:
            body = await _receive_body(receive, proxy.config.max_request_body_size)
        except RequestBodyTooLarge:
            response = Response(
                content="Prefix routing request body is too large",
                status_code=413,
                media_type="text/plain",
            )
            await response(scope, receive, send)
            return
        try:
            payload = json.loads(body)
            decision = await proxy.choose_node_for_request(path, payload)
            logger.info(
                "Prefix routing chose node=%s matched_tokens=%s",
                decision.node_id if decision else None,
                decision.matched_tokens if decision else None,
            )
        except Exception:
            logger.exception("Prefix routing failed; falling back to local handling")
            await self.app(scope, _replay_body(body), send)
            return

        if decision is None:
            await self.app(scope, _replay_body(body), send)
            return

        node = proxy.nodes.get(decision.node_id)
        if node is None or node.local or node.url is None:
            await self.app(
                _with_data_parallel_rank(scope, decision.data_parallel_rank),
                _replay_body(body),
                send,
            )
            return

        forwarded = await _forward_request(
            scope,
            body,
            send,
            node,
            proxy.session,
            decision.data_parallel_rank,
        )
        if not forwarded:
            logger.warning(
                "Prefix routing upstream %s failed before responding; "
                "falling back to local handling",
                node.node_id,
            )
            await self.app(
                scope,
                _replay_body(body),
                send,
            )


async def init_prefix_routing(app_state: Any, args: Any) -> None:
    raw_config = getattr(args, "prefix_routing_config", None)
    if not getattr(args, "enable_prefix_routing", False):
        return
    config = _parse_prefix_routing_config(raw_config, app_state.vllm_config)
    if os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "Prefix routing is enabled but PYTHONHASHSEED is not set. "
            "Set the same PYTHONHASHSEED on every vLLM node and the routing "
            "master so prefix-cache block hashes are comparable across nodes."
        )
    proxy = PrefixRoutingProxy(config=config, app_state=app_state)
    app_state.prefix_routing_proxy = proxy
    proxy.start()
    logger.info("Prefix routing enabled with %d nodes", len(config.nodes))


def _parse_prefix_routing_config(
    raw_config: Mapping[str, Any] | str | None, vllm_config: Any
) -> PrefixRoutingConfig:
    if raw_config is None:
        raise ValueError("--enable-prefix-routing requires --prefix-routing-config")
    if isinstance(raw_config, str):
        raw_config = json.loads(raw_config)
    if not isinstance(raw_config, Mapping):
        raise ValueError("prefix routing config must be a JSON object")

    raw_nodes = raw_config.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise ValueError("prefix routing config 'nodes' must be a list")

    nodes: list[PrefixRoutingNode] = []
    node_ids: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("prefix routing node config must be a JSON object")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("prefix routing node 'id' must be a non-empty string")
        if node_id in node_ids:
            raise ValueError(f"duplicate prefix routing node id: {node_id!r}")
        node_ids.add(node_id)

        node_url = node.get("url")
        local = bool(node.get("local", node_url == "local"))
        if local:
            if node_url not in (None, "local"):
                raise ValueError(
                    f"local prefix routing node {node_id!r} cannot have a remote URL"
                )
            node_url = None
        elif not isinstance(node_url, str) or not node_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"remote prefix routing node {node_id!r} requires an HTTP(S) URL"
            )

        event_endpoint = node.get("event_endpoint")
        if event_endpoint is not None and (
            not isinstance(event_endpoint, str) or not event_endpoint
        ):
            raise ValueError(
                f"prefix routing node {node_id!r} event_endpoint must be a string"
            )

        data_parallel_rank = node.get("data_parallel_rank")
        if data_parallel_rank is not None and (
            not isinstance(data_parallel_rank, int)
            or isinstance(data_parallel_rank, bool)
            or data_parallel_rank < 0
        ):
            raise ValueError(
                f"prefix routing node {node_id!r} data_parallel_rank "
                "must be a non-negative integer"
            )

        node_routing_token = node.get("routing_token")
        if not local and (
            not isinstance(node_routing_token, str) or not node_routing_token
        ):
            raise ValueError(
                f"remote prefix routing node {node_id!r} requires a non-empty "
                "routing_token"
            )
        if local:
            node_routing_token = None

        nodes.append(
            PrefixRoutingNode(
                node_id=node_id,
                url=node_url,
                event_endpoint=event_endpoint,
                data_parallel_rank=data_parallel_rank,
                routing_token=node_routing_token,
                local=local,
            )
        )
    if not nodes:
        raise ValueError("prefix routing config must include at least one node")

    cache_config = vllm_config.cache_config
    hash_block_size = raw_config.get("hash_block_size")
    if hash_block_size is None:
        hash_block_size = cache_config.hash_block_size or cache_config.block_size
    if (
        not isinstance(hash_block_size, int)
        or isinstance(hash_block_size, bool)
        or hash_block_size <= 0
    ):
        raise ValueError("prefix routing hash_block_size must be a positive integer")

    request_timeout = raw_config.get("request_timeout", 6 * 60 * 60)
    if (
        not isinstance(request_timeout, int | float)
        or isinstance(request_timeout, bool)
        or request_timeout <= 0
    ):
        raise ValueError("prefix routing request_timeout must be positive")

    event_ingest_token = raw_config.get("event_ingest_token")
    if event_ingest_token is not None and (
        not isinstance(event_ingest_token, str) or not event_ingest_token
    ):
        raise ValueError("prefix routing event_ingest_token must be a non-empty string")

    routing_token = raw_config.get("routing_token")
    if routing_token is not None and (
        not isinstance(routing_token, str) or not routing_token
    ):
        raise ValueError("prefix routing routing_token must be a non-empty string")

    max_request_body_size = raw_config.get(
        "max_request_body_size", DEFAULT_MAX_REQUEST_BODY_SIZE
    )
    if (
        not isinstance(max_request_body_size, int)
        or isinstance(max_request_body_size, bool)
        or max_request_body_size <= 0
    ):
        raise ValueError(
            "prefix routing max_request_body_size must be a positive integer"
        )

    return PrefixRoutingConfig(
        nodes=nodes,
        hash_block_size=hash_block_size,
        event_topic=str(raw_config.get("event_topic", "")),
        request_timeout=float(request_timeout),
        event_ingest_token=event_ingest_token,
        routing_token=routing_token,
        max_request_body_size=max_request_body_size,
    )


class RequestBodyTooLarge(Exception):
    pass


async def _receive_body(receive: Receive, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    body_size = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        if message["type"] != "http.request":
            continue
        chunk = message.get("body", b"")
        body_size += len(chunk)
        if body_size > max_bytes:
            raise RequestBodyTooLarge
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _replay_body(body: bytes) -> Receive:
    sent = False
    wait_forever = asyncio.Event()

    async def receive() -> Message:
        nonlocal sent
        if sent:
            await wait_forever.wait()
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _with_data_parallel_rank(scope: Scope, rank: int | None) -> Scope:
    if rank is None:
        return scope
    routed_scope = dict(scope)
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() != DATA_PARALLEL_RANK_HEADER.encode()
    ] + [(DATA_PARALLEL_RANK_HEADER.encode(), str(rank).encode("ascii"))]
    return routed_scope


def _without_external_routing_headers(scope: Scope) -> Scope:
    routed_scope = dict(scope)
    stripped_headers = {
        PREFIX_ROUTING_BYPASS_HEADER.encode(),
        DATA_PARALLEL_RANK_HEADER.encode(),
    }
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() not in stripped_headers
    ]
    return routed_scope


def _without_bypass_header(scope: Scope) -> Scope:
    routed_scope = dict(scope)
    routed_scope["headers"] = [
        (key, value)
        for key, value in scope["headers"]
        if key.lower() != PREFIX_ROUTING_BYPASS_HEADER.encode()
    ]
    return routed_scope


def _has_authenticated_bypass(headers: Headers, expected_token: str | None) -> bool:
    if expected_token is None:
        return False
    token = headers.get(PREFIX_ROUTING_BYPASS_HEADER)
    if token is None:
        return False
    return secrets.compare_digest(
        hashlib.sha256(token.encode("utf-8")).digest(),
        hashlib.sha256(expected_token.encode("utf-8")).digest(),
    )


async def _forward_request(
    scope: Scope,
    body: bytes,
    send: Send,
    node: PrefixRoutingNode,
    session: aiohttp.ClientSession,
    data_parallel_rank: int | None = None,
) -> bool:
    if node.url is None or node.routing_token is None:
        raise RuntimeError("remote prefix routing node is missing forwarding config")
    url = f"{node.url.rstrip('/')}{scope['path']}"
    if scope.get("query_string"):
        url += "?" + scope["query_string"].decode("latin-1")

    headers = [
        (key.decode("latin-1"), value.decode("latin-1"))
        for key, value in scope["headers"]
        if key.lower() not in (b"host", b"content-length")
    ]
    headers = [
        (key, value)
        for key, value in headers
        if key.lower() not in (PREFIX_ROUTING_BYPASS_HEADER, DATA_PARALLEL_RANK_HEADER)
    ]
    headers.append((PREFIX_ROUTING_BYPASS_HEADER, node.routing_token))
    if data_parallel_rank is not None:
        headers.append((DATA_PARALLEL_RANK_HEADER, str(data_parallel_rank)))

    response_started = False
    try:
        async with session.request(
            method=scope["method"],
            url=url,
            data=body,
            headers=headers,
            allow_redirects=False,
        ) as response:
            response_headers = [
                (key.encode("latin-1"), value.encode("latin-1"))
                for key, value in response.headers.items()
                if key.lower()
                not in ("transfer-encoding", "content-encoding", "content-length")
            ]
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status,
                    "headers": response_headers,
                }
            )
            response_started = True
            async for chunk in response.content.iter_chunked(1024):
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b""})
            return True
    except (aiohttp.ClientError, TimeoutError):
        if response_started:
            raise
        return False
