# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Prefill/Decode Proxy with Bidirectional KV Transfer

This proxy sits between clients and a vLLM Prefill/Decode (P/D) deployment,
routing multi-turn chat requests so that each turn reuses KV cache blocks
from the previous turn's Decode node via bidirectional KV transfer.

Architecture:
    Client  ──►  Proxy  ──►  Prefill (P)  ──►  Decode (D)
                   │              │                 │
                   │   kv_transfer_params flow:     │
                   │   D finish ──► proxy caches    │
                   │   next turn ──► proxy sends    │
                   │   cached D blocks to P ──►     │
                   │   P reads D blocks (bidir)     │
                   │   P sends its blocks to D      │

Per-request flow:
    1. Client sends chat/completions request to proxy.
    2. Proxy looks up cached D block info from the previous turn
       (keyed by conversation_id).
    3. If cache hit, proxy attaches D's block info to the request
       so P can read D's KV blocks instead of recomputing.
    4. Proxy sends request to P (max_tokens=1, non-streaming).
    5. P returns kv_transfer_params with its own block info.
    6. Proxy forwards request + P's block info to D (streaming).
    7. D streams the response. The final chunk includes D's
       kv_transfer_params, which the proxy caches for the next turn.
    8. Proxy returns D's response to the client.

Conversation isolation:
    Each request must include a ``conversation_id`` field (top-level in
    the JSON body) to scope the KV cache across turns. Without it, the
    proxy cannot link turns and falls back to no-cache behavior.

Usage:
    python disagg_proxy_multiturn.py \\
        --host 0.0.0.0 --port 8000 \\
        --prefiller-host 10.0.0.1 --prefiller-port 8100 \\
        --decoder-host 10.0.0.2 --decoder-port 8200

Dependencies:
    pip install fastapi uvicorn httpx
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("disagg_proxy")


# Data structures
@dataclass
class CachedKVEntry:
    """KV transfer parameters cached from D's response for one turn."""

    kv_transfer_params: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ConversationKVCache:
    """Per-conversation KV block cache.

    Each conversation is identified by a ``conversation_id`` supplied by
    the client. After D finishes a turn, its ``kv_transfer_params`` are
    stored here. On the next turn, the proxy retrieves them so P can
    read D's blocks via bidirectional KV transfer.
    """

    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self._store: dict[str, CachedKVEntry] = {}
        self._ttl = ttl_seconds

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve and consume cached KV params for a conversation.

        Returns a *copy* of the kv_transfer_params dict, or None.
        The entry is removed after retrieval (single-use).
        """
        entry = self._store.pop(conversation_id, None)
        if entry is None:
            return None
        age = time.time() - entry.timestamp
        if age > self._ttl:
            logger.info(
                "conv=%s: stale cache entry (age=%.1fs > ttl=%.1fs), discarding",
                conversation_id,
                age,
                self._ttl,
            )
            return None
        logger.info(
            "conv=%s: cache HIT (age=%.1fs)",
            conversation_id,
            age,
        )
        return dict(entry.kv_transfer_params)

    def put(self, conversation_id: str, kv_params: dict[str, Any]) -> None:
        """Store D's kv_transfer_params for a conversation."""
        self._store[conversation_id] = CachedKVEntry(
            kv_transfer_params=dict(kv_params),  # defensive copy
        )
        logger.info(
            "conv=%s: cached D blocks (remote_request_id=%s, blocks=%d)",
            conversation_id,
            kv_params.get("remote_request_id", "?"),
            len(kv_params.get("remote_block_ids", [[]])[0])
            if kv_params.get("remote_block_ids")
            else 0,
        )

    def evict_stale(self) -> int:
        """Remove entries older than TTL. Returns count of evicted entries."""
        now = time.time()
        stale = [
            cid
            for cid, entry in self._store.items()
            if now - entry.timestamp > self._ttl
        ]
        for cid in stale:
            del self._store[cid]
        return len(stale)

    @property
    def size(self) -> int:
        return len(self._store)


# Global state
kv_cache = ConversationKVCache(
    ttl_seconds=450.0
)  # Must be < VLLM_NIXL_ABORT_REQUEST_TIMEOUT (480s)


# Service client helpers
@dataclass
class ServiceClient:
    """Wrapper around an httpx.AsyncClient for a P or D instance."""

    client: httpx.AsyncClient
    host: str
    port: int
    id: int


def _make_headers(request_id: str) -> dict[str, str]:
    """Build HTTP headers for upstream requests."""
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _send_to_prefill(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Send a non-streaming prefill request (max_tokens=1).

    Returns the JSON response from P, which includes kv_transfer_params.
    """
    payload = req_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload.pop("max_completion_tokens", None)
    payload.pop("min_tokens", None)
    payload.pop("stream_options", None)

    resp = await client.client.post(
        endpoint,
        json=payload,
        headers=_make_headers(request_id),
    )
    resp.raise_for_status()
    return resp.json()


async def _stream_from_decode(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
    conversation_id: str,
) -> tuple[str, str | None, dict[str, Any] | None, str, str | None, int | None]:
    """Stream response from D, capturing text and kv_transfer_params.

    Returns (collected_text, finish_reason, kv_params, response_id, created).
    Also stores kv_params in the conversation cache.
    """
    payload = req_data.copy()
    payload["stream"] = True

    collected_text = ""
    finish_reason: str | None = None
    response_id: str | None = None
    model_name: str | None = None
    created: int | None = None
    captured_kv: dict[str, Any] | None = None

    async with client.client.stream(
        "POST",
        endpoint,
        json=payload,
        headers=_make_headers(request_id),
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if response_id is None:
                response_id = chunk.get("id")
                model_name = chunk.get("model")
                created = chunk.get("created")

            for choice in chunk.get("choices", []):
                collected_text += choice.get("text", "")
                delta = choice.get("delta", {})
                collected_text += delta.get("content", "")
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            kv_params = chunk.get("kv_transfer_params")
            if kv_params:
                kv_params["remote_host"] = client.host
                captured_kv = kv_params
                if conversation_id:
                    kv_cache.put(conversation_id, kv_params)

    return (
        collected_text,
        finish_reason,
        captured_kv,
        response_id or request_id,
        model_name,
        created,
    )


async def _stream_from_decode_sse(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
    conversation_id: str,
):
    """Yield SSE chunks from D to the client, capturing kv_transfer_params."""
    payload = req_data.copy()
    payload["stream"] = True

    async with client.client.stream(
        "POST",
        endpoint,
        json=payload,
        headers=_make_headers(request_id),
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                yield "\n"
                continue

            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = json.loads(line[6:])
                    kv_params = chunk.get("kv_transfer_params")
                    if kv_params and conversation_id:
                        kv_params["remote_host"] = client.host
                        kv_cache.put(conversation_id, kv_params)
                except json.JSONDecodeError:
                    pass

            yield line + "\n"


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize HTTP clients for P and D instances."""
    app.state.prefill_clients: list[ServiceClient] = []
    app.state.decode_clients: list[ServiceClient] = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            ServiceClient(
                client=httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                ),
                host=host,
                port=port,
                id=i,
            )
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            ServiceClient(
                client=httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                ),
                host=host,
                port=port,
                id=i,
            )
        )

    app.state.prefill_iter = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iter = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Ready: %d prefill, %d decode instances",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )
    yield

    for sc in app.state.prefill_clients + app.state.decode_clients:
        await sc.client.aclose()


app = FastAPI(title="Disaggregated P/D Proxy (Multi-turn)", lifespan=lifespan)


def _next_client(app_state, role: str) -> ServiceClient:
    if role == "prefill":
        return app_state.prefill_clients[next(app_state.prefill_iter)]
    return app_state.decode_clients[next(app_state.decode_iter)]


# Request handler
async def _handle_request(api_path: str, request: Request):
    """Core request handler for both /v1/chat/completions and /v1/completions."""
    req_data = await request.json()
    request_id = str(uuid.uuid4())
    conversation_id: str = req_data.pop("conversation_id", "")
    client_wants_stream = req_data.get("stream", False)

    if not conversation_id:
        logger.warning(
            "[%s] No conversation_id provided — KV cache reuse disabled "
            "for this request. Add a 'conversation_id' field to enable "
            "cross-turn KV sharing.",
            request_id,
        )

    # Step 1: Look up cached D blocks from the previous turn
    cached_kv = kv_cache.get(conversation_id) if conversation_id else None

    if cached_kv:
        # Tell P to read D's blocks (bidirectional transfer)
        cached_kv["do_remote_decode"] = True
        cached_kv["do_remote_prefill"] = False
        req_data["kv_transfer_params"] = cached_kv
        logger.info(
            "[%s] conv=%s: sending D's cached blocks to P (remote_request_id=%s)",
            request_id,
            conversation_id,
            cached_kv.get("remote_request_id"),
        )
    else:
        # No cached blocks — P recomputes from scratch
        req_data["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        logger.info("[%s] conv=%s: cache MISS", request_id, conversation_id)

    # Step 2: Send to Prefill node (non-streaming, max_tokens=1)
    prefill_client = _next_client(request.app.state, "prefill")
    t0 = time.time()
    prefill_resp = await _send_to_prefill(
        prefill_client,
        api_path,
        req_data,
        request_id,
    )
    logger.info(
        "[%s] Prefill done in %.0fms",
        request_id,
        (time.time() - t0) * 1000,
    )

    # Attach P's kv_transfer_params for D to read P's blocks
    p_kv_params = prefill_resp.get("kv_transfer_params", {})
    if p_kv_params:
        p_kv_params["remote_host"] = prefill_client.host
        req_data["kv_transfer_params"] = p_kv_params

    # Step 3: Stream from Decode node, capturing kv_transfer_params
    decode_client = _next_client(request.app.state, "decode")

    if client_wants_stream:
        return StreamingResponse(
            _stream_from_decode_sse(
                decode_client,
                api_path,
                req_data,
                request_id,
                conversation_id,
            ),
            media_type="text/event-stream",
        )

    text, finish_reason, _, resp_id, model, created = await _stream_from_decode(
        decode_client,
        api_path,
        req_data,
        request_id,
        conversation_id,
    )

    # Build OpenAI-compatible response
    is_chat = "messages" in req_data
    if is_chat:
        body = {
            "id": resp_id,
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": model or req_data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": None,
        }
    else:
        body = {
            "id": resp_id,
            "object": "text_completion",
            "created": created or int(time.time()),
            "model": model or req_data.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": None,
        }
    return JSONResponse(content=body)


# Routes
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_request("/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_request("/completions", request)


@app.get("/health")
async def health():
    evicted = kv_cache.evict_stale()
    return {
        "status": "ok",
        "cached_conversations": kv_cache.size,
        "evicted_stale": evicted,
    }


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Disaggregated P/D proxy with bidirectional KV transfer",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--prefiller-host",
        "--prefiller-hosts",
        dest="prefiller_hosts",
        nargs="+",
        default=["localhost"],
    )
    p.add_argument(
        "--prefiller-port",
        "--prefiller-ports",
        dest="prefiller_ports",
        type=int,
        nargs="+",
        default=[8100],
    )
    p.add_argument(
        "--decoder-host",
        "--decoder-hosts",
        dest="decoder_hosts",
        nargs="+",
        default=["localhost"],
    )
    p.add_argument(
        "--decoder-port",
        "--decoder-ports",
        dest="decoder_ports",
        type=int,
        nargs="+",
        default=[8200],
    )
    args = p.parse_args()

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        p.error("Number of prefiller hosts must match ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        p.error("Number of decoder hosts must match ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
