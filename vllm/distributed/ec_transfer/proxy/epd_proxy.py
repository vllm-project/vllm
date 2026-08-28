# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proxy that fans a multimodal request out to encoders, then prefill/decode.

The proxy starts with an empty roster: encode, prefill and decode instances
register themselves once they are serving, and the proxy owns liveness from
then on. Nothing about the topology is known at launch, so an instance can
join or leave without restarting anything else.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiohttp
import pybase64 as base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from vllm.distributed.ec_transfer.proxy.registry import (
    InstanceRecord,
    InstanceRegistry,
    InstanceRole,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

MM_TYPES = {"image_url", "audio_url", "input_audio"}


@dataclass
class EPDProxyConfig:
    """Attributes:
    no_rewrite: Forward the media to the decoder unchanged instead of
        replacing it with a reference to the encoder's embedding. Only
        useful for A/B timing the rewrite itself.
    """

    no_rewrite: bool = False
    probe_interval: float = 5.0
    probe_timeout: float = 2.0
    fail_threshold: int = 3
    evicted_ttl: float = 900.0


def content_uuid(item: dict) -> str:
    """Cache key for a multimodal item, derived from its content.

    Must be content-derived, not request-derived: the EC cache is keyed by
    this value, so a per-request key would make every request a miss and
    throw away cross-request reuse of already-encoded media.
    """
    url = (item.get("image_url") or item.get("audio_url") or {}).get("url") or ""
    payload = url or json.dumps(item, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _b64_tensor(values: list) -> str:
    import torch

    buf = io.BytesIO()
    grid = torch.tensor(values, dtype=torch.long)
    # Downstream stacks per item, so hand over a flat (t, h, w).
    torch.save(grid.reshape(-1)[:3], buf)
    return base64.b64encode(buf.getvalue()).decode()


def extract_mm_items(request_data: dict) -> list[dict]:
    """Return every image/audio item appearing anywhere in ``messages``."""
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        items.extend(item for item in content if item.get("type") in MM_TYPES)
    return items


def rewrite_for_decode(req_data: dict, item_meta: dict[int, dict]) -> dict:
    """Replace each media item with a metadata-only reference.

    The decoder does not need the pixels: the encoder already produced the
    embedding and published it under the same uuid. `item_meta` holds what
    the encoder reported, so the grid is never re-derived here -- a second
    derivation could disagree with the encoder's.
    """
    rewritten = 0
    idx = 0
    new_messages = []
    for msg in req_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            new_messages.append(msg)
            continue
        new_content = []
        for item in content:
            if item.get("type") not in MM_TYPES:
                new_content.append(item)
                continue
            meta = dict(item_meta.get(idx) or {})
            idx += 1
            item_uuid = meta.pop("mm_hash", None)
            metadata = {key: _b64_tensor(value) for key, value in meta.items()}
            if not metadata or not item_uuid:
                # The encoder reported no metadata (the item came from its
                # processor cache); let the decoder process the media itself.
                new_content.append(item)
                continue
            new_content.append(
                {
                    "type": "image_embeds",
                    "image_embeds": metadata,
                    "uuid": item_uuid,
                }
            )
            rewritten += 1
        new_messages.append({**msg, "content": new_content})

    if not rewritten:
        return req_data
    logger.info("Rewrote %d media item(s) as metadata references", rewritten)
    return {**req_data, "messages": new_messages}


@dataclass
class _Route:
    """The instances one request was assigned to."""

    encoders: list[InstanceRecord]
    prefill: InstanceRecord | None
    decode: InstanceRecord
    consumer_zmq: str | None


class EPDProxy:
    def __init__(self, config: EPDProxyConfig, registry: InstanceRegistry):
        self.config = config
        self.registry = registry
        self.session: aiohttp.ClientSession | None = None

    @property
    def http(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError("The EPD proxy was used before it started")
        return self.session

    # ---------------------------------------------------------------- #
    # Routing                                                          #
    # ---------------------------------------------------------------- #
    def route(self, num_items: int) -> _Route:
        decode = self.registry.pick(InstanceRole.DECODE)
        if decode is None:
            raise HTTPException(
                status_code=503, detail="No decode instance is registered"
            )
        prefill = self.registry.pick(InstanceRole.PREFILL)
        encoders = self.registry.pick_many(InstanceRole.ENCODE, num_items)
        if num_items and not encoders:
            raise HTTPException(
                status_code=503, detail="No encode instance is registered"
            )
        return _Route(
            encoders=encoders,
            prefill=prefill,
            decode=decode,
            consumer_zmq=self._consumer_zmq(prefill, decode),
        )

    @staticmethod
    def _consumer_zmq(
        prefill: InstanceRecord | None, decode: InstanceRecord
    ) -> str | None:
        """Address the encoder should push this request's embedding to.

        Which stage consumes the embedding depends on the topology -- the
        prefill instance when prefill is split out, the decode instance
        otherwise -- so the consumer is whichever one registered a receive
        address rather than whichever list it came from. Connectors that
        publish to shared storage register none, and the encoder is then
        told nothing.
        """
        for candidate in (prefill, decode):
            if candidate is not None and candidate.ec_zmq_addrs:
                return candidate.ec_zmq_addrs[0]
        return None

    # ---------------------------------------------------------------- #
    # Stages                                                           #
    # ---------------------------------------------------------------- #
    async def encode(
        self, req_data: dict, route: _Route, req_id: str
    ) -> dict[int, dict]:
        """Send one text-free request per media item and collect the metadata."""
        mm_items = extract_mm_items(req_data)
        if not mm_items:
            return {}

        logger.info("[%s] Encoding %d media item(s)", req_id, len(mm_items))
        item_uuids: dict[int, str] = {}
        tasks = []
        for idx, (item, encoder) in enumerate(zip(mm_items, route.encoders)):
            child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
            # With no_rewrite the decoder receives the raw media and derives
            # the key by hashing it, so the encoder must do the same: passing
            # a uuid here would make the two disagree and silently defeat the
            # transfer.
            item_uuid = None if self.config.no_rewrite else content_uuid(item)
            if item_uuid is not None:
                item_uuids[idx] = item_uuid
            encoder_req: dict = {
                "model": req_data.get("model"),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            item if item_uuid is None else {**item, "uuid": item_uuid}
                        ],
                    }
                ],
                # No max_tokens cap: the encoder never samples, it finishes
                # once the prompt is encoded and its embeddings are published.
                "stream": False,
            }
            if route.consumer_zmq is not None and item_uuid is not None:
                encoder_req["ec_transfer_params"] = {
                    "consumer_zmq": route.consumer_zmq,
                    "ec_items": [{"mm_hash": item_uuid}],
                }
            tasks.append(
                self.http.post(
                    f"{encoder.url}/v1/chat/completions",
                    json=encoder_req,
                    headers={"x-request-id": child_req_id},
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return await self._collect_encoder_metadata(results, item_uuids, req_id)

    async def _collect_encoder_metadata(
        self, results: list, item_uuids: dict[int, str], req_id: str
    ) -> dict[int, dict]:
        item_meta: dict[int, dict] = {}
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "[%s] Encoder request #%d raised", req_id, idx, exc_info=result
                )
                raise HTTPException(
                    status_code=502, detail=f"Encoder request failed: {result}"
                )
            if result.status != 200:
                detail = await result.text()
                logger.error(
                    "[%s] Encoder request #%d returned %s: %s",
                    req_id,
                    idx,
                    result.status,
                    detail,
                )
                raise HTTPException(
                    status_code=result.status,
                    detail=f"Encoder request failed: {detail}",
                )
            try:
                params = (await result.json()).get("ec_transfer_params") or {}
                reported = params.get("ec_items") or []
            except Exception:
                logger.warning("[%s] Unreadable encoder metadata #%d", req_id, idx)
                reported = []
            if reported and idx in item_uuids:
                # One item per encoder request, so the first entry is this one's.
                item_meta[idx] = {**reported[0], "mm_hash": item_uuids[idx]}
        return item_meta

    async def prefill(self, req_data: dict, route: _Route, req_id: str) -> dict:
        """Run the prefill stage and carry its transfer params to the decoder."""
        if route.prefill is None:
            return req_data

        prefill_request = req_data.copy()
        prefill_request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        prefill_request["stream"] = False
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        prefill_request.pop("stream_options", None)

        logger.info("[%s] Prefilling on %s", req_id, route.prefill.url)
        async with self.http.post(
            f"{route.prefill.url}/v1/chat/completions",
            json=prefill_request,
            headers={"x-request-id": req_id},
        ) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise HTTPException(
                    status_code=resp.status,
                    detail=f"Prefill request failed: {detail}",
                )
            body = await resp.json()

        # The prefill instance reports where its KV lives; the decoder pulls
        # from there. The proxy only relays it and needs to know nothing
        # about the transport.
        kv_transfer_params = body.get("kv_transfer_params") or {}
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        return req_data

    async def _through_encode_and_prefill(
        self, req_data: dict, route: _Route, req_id: str
    ) -> dict:
        item_meta = await self.encode(req_data, route, req_id)
        if not self.config.no_rewrite:
            req_data = rewrite_for_decode(req_data, item_meta)
        return await self.prefill(req_data, route, req_id)

    async def forward(self, req_data: dict, route: _Route, req_id: str) -> dict:
        req_data = await self._through_encode_and_prefill(req_data, route, req_id)
        logger.info("[%s] Decoding on %s", req_id, route.decode.url)
        async with self.http.post(
            f"{route.decode.url}/v1/chat/completions",
            json=req_data,
            headers={"x-request-id": req_id},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def forward_stream(
        self, req_data: dict, route: _Route, req_id: str
    ) -> AsyncIterator[str]:
        req_data = await self._through_encode_and_prefill(req_data, route, req_id)
        logger.info("[%s] Streaming from %s", req_id, route.decode.url)
        async with self.http.post(
            f"{route.decode.url}/v1/chat/completions",
            json=req_data,
            headers={"x-request-id": req_id},
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(1024):
                if chunk:
                    yield chunk.decode("utf-8", errors="ignore")


class RegisterRequest(BaseModel):
    """What an instance reports when it joins.

    Attributes:
        role: Which stage this instance serves.
        url: Base OpenAI-compatible URL other components should reach it at.
        ec_zmq_addrs: Encoder-cache receive addresses, one per rank. Only an
            EC consumer has these and only it knows them; reporting them here
            is what lets the proxy name a push target without operators
            hand-aligning a list against the instance URLs.
        dp_size: Data-parallel replicas behind `url`.
    """

    role: InstanceRole
    url: str
    ec_zmq_addrs: list[str] = Field(default_factory=list)
    dp_size: int = 1
    metadata: dict = Field(default_factory=dict)


def build_app(config: EPDProxyConfig | None = None) -> FastAPI:
    config = config or EPDProxyConfig()
    registry = InstanceRegistry(
        probe_interval=config.probe_interval,
        probe_timeout=config.probe_timeout,
        fail_threshold=config.fail_threshold,
        evicted_ttl=config.evicted_ttl,
    )
    proxy = EPDProxy(config, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One session for every stage: which stages exist is not known until
        # instances register, so none of them can be set up conditionally.
        timeout = aiohttp.ClientTimeout(total=100_000)
        connector = aiohttp.TCPConnector(limit=0, force_close=False)
        proxy.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        registry.start_probing()
        try:
            yield
        finally:
            await registry.stop_probing()
            await proxy.session.close()

    app = FastAPI(lifespan=lifespan)
    app.state.proxy = proxy
    app.state.registry = registry

    @app.post("/instances")
    async def register_instance(body: RegisterRequest):
        registry.register(
            InstanceRecord(
                role=body.role,
                url=body.url.rstrip("/"),
                ec_zmq_addrs=body.ec_zmq_addrs,
                dp_size=body.dp_size,
                metadata=body.metadata,
            )
        )
        return {"registered": body.url, "role": body.role.value}

    @app.delete("/instances")
    async def unregister_instance(body: RegisterRequest):
        removed = registry.unregister(body.url.rstrip("/"))
        return {"unregistered": body.url, "found": removed}

    @app.get("/instances")
    async def list_instances():
        return registry.status()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        route = proxy.route(len(extract_mm_items(req_data)))
        if req_data.get("stream", False):
            return StreamingResponse(
                proxy.forward_stream(req_data, route, req_id),
                media_type="text/event-stream",
            )
        return JSONResponse(content=await proxy.forward(req_data, route, req_id))

    @app.get("/v1/models")
    async def list_models():
        decode = registry.pick(InstanceRole.DECODE)
        if decode is None:
            raise HTTPException(
                status_code=503, detail="No decode instance is registered"
            )
        async with proxy.http.get(f"{decode.url}/v1/models") as resp:
            resp.raise_for_status()
            return await resp.json()

    @app.get("/health")
    async def health():
        status = registry.status()
        # An empty roster is not unhealthy: the proxy is meant to come up
        # before anything registers with it.
        return JSONResponse({"proxy": "healthy", "instances": status})

    @app.post("/start_profile")
    async def start_profile(request: Request):
        return await _profile(proxy, registry, "start", await request.json())

    @app.post("/stop_profile")
    async def stop_profile(request: Request):
        return await _profile(proxy, registry, "stop", await request.json())

    return app


async def _post_if_available(
    session: aiohttp.ClientSession, url: str, payload: dict, headers: dict
) -> dict | None:
    """POST `payload` to `url`, treating a missing route as "not enabled"."""
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise


async def _profile(
    proxy: EPDProxy, registry: InstanceRegistry, cmd: str, payload: dict
) -> dict:
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
    targets = {role.value: registry.pick(role) for role in InstanceRole}
    results = await asyncio.gather(
        *(
            _post_if_available(
                proxy.http, f"{record.url}/{cmd}_profile", payload, headers
            )
            for record in targets.values()
            if record is not None
        )
    )
    reachable = [result for result in results if result is not None]
    if not reachable:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on every instance",
        )
    live = [name for name, record in targets.items() if record is not None]
    return dict(zip(live, results))
