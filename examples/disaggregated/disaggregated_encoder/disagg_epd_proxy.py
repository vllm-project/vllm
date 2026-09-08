#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
disagg_encoder_proxy.py

Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)

For MM input we:
    1. Extract *every* image/audio/video item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import itertools
import json
import logging
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pybase64 as base64
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("proxy")

app = FastAPI()
encode_session: aiohttp.ClientSession | None = None
prefill_session: aiohttp.ClientSession | None = None
decode_session: aiohttp.ClientSession | None = None

# Cursor for round-robin encoder assignment, shared across requests so the
# fan-out doesn't restart from e_urls[0] every time.
encoder_rr_idx = 0
encoder_rr_lock = asyncio.Lock()

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio", "video_url"}

# The embeds content type each MM item is rewritten to once the encoder has
# published its embedding out of band.
EMBEDS_TYPES = {
    "image_url": "image_embeds",
    "audio_url": "audio_embeds",
    "input_audio": "audio_embeds",
    "video_url": "video_embeds",
}


def encoder_rr_assignment(
    e_urls: list[str], start: int, count: int
) -> tuple[list[str], int]:
    """Assign `count` items to encoder URLs starting from cursor `start`.

    Returns the per-item URL list and the cursor value the next call should
    start from, so the assignment is contiguous across calls instead of
    restarting at e_urls[0] every time.
    """
    urls = [e_urls[(start + i) % len(e_urls)] for i in range(count)]
    next_start = (start + count) % len(e_urls)
    return urls, next_start


def validate_ec_consumer_routing(
    prefill_urls: list[str], consumer_addrs: list[str]
) -> None:
    """Reject the topology whose EC destination cannot be routed safely."""
    if prefill_urls and consumer_addrs:
        raise ValueError(
            "Mooncake EC consumer routing supports E+PD only; disable independent "
            "prefill or omit --ec-consumer-zmq-addrs."
        )


# Diagnostic switch: forward the original request to the decoder so the
# only difference from the rewrite path is the rewrite itself.
NO_REWRITE = False
# Decode-side retries for a retryable internal error (`finish_reason="error"`,
# e.g. an encoder embedding the connector could not deliver). Re-issuing runs
# the encode again, which produces a fresh transfer.
DECODE_RETRIES = 1


# Grid metadata reported by the encoder instance, keyed by item index.
# Empty when the encoder did not report any (then nothing is rewritten).
def content_uuid(item: dict) -> str:
    """Cache key for a multimodal item, derived from its content.

    Must be content-derived, not request-derived: the EC cache is keyed by this
    value, so a per-request key (a request id, say) would make every request a
    miss and throw away cross-request reuse of already-encoded media -- while
    the unmodified path, which hashes the content, would keep it. That asymmetry
    silently biases any comparison between the two.
    """
    url = (
        item.get("image_url") or item.get("audio_url") or item.get("video_url") or {}
    ).get("url") or ""
    payload = url or json.dumps(item, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _b64_tensor(values: list) -> str:
    import torch

    buf = io.BytesIO()
    flat = [v for item in values for v in (item if isinstance(item, list) else [item])]
    # Floats stay float64 so timestamp strings format exactly as the
    # encoder computed them.
    dtype = torch.float64 if any(isinstance(v, float) for v in flat) else None
    # Downstream stacks per item, so hand over a flat vector.
    torch.save(torch.tensor(flat, dtype=dtype), buf)
    return base64.b64encode(buf.getvalue()).decode()


def rewrite_for_decode(req_data: dict, item_meta: dict[int, dict]) -> dict:
    """Replace each media item with a metadata-only reference for the decoder.

    The decoder does not need the pixels: the encoder instance already produced
    the embedding and published it through the EC connector under the same uuid.
    Sending only the grid lets the decoder size the placeholder range without
    re-running the media transform.

    `item_meta` holds what the encoder reported for each item (its cache key and
    the grid its processor actually produced), so the grid is never re-derived
    here -- a second derivation could disagree with the encoder's.
    """
    rewritten = 0
    transfer_items = []
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
            ec_mm_hash = meta.pop("ec_mm_hash", None) or item_uuid
            transfer_id = meta.pop("transfer_id", None)
            # Whatever keys the encoder reported are the metadata its model
            # declared as needed to size the placeholder range; the proxy does
            # not need to know their names.
            metadata = {k: _b64_tensor(v) for k, v in meta.items()}
            if not metadata or not item_uuid:
                # Nothing to size the placeholder range with. A processor cache
                # hit is not a cause on its own: with the default `lru` type the
                # engine restores the item before the scheduler reports it. It
                # goes missing when the encode request failed, or under
                # `--mm-processor-cache-type shm`, where a hit replaces the item
                # with its shared-memory address and only the worker restores
                # it. Send the media so the decoder can derive the grid itself.
                new_content.append(item)
                continue
            embeds_type = EMBEDS_TYPES[item["type"]]
            new_content.append(
                {"type": embeds_type, embeds_type: metadata, "uuid": item_uuid}
            )
            if transfer_id is not None:
                transfer_items.append(
                    {"mm_hash": ec_mm_hash, "transfer_id": transfer_id}
                )
            rewritten += 1
        new_messages.append({**msg, "content": new_content})

    if not rewritten:
        return req_data
    logger.info("Rewrote %d media item(s) as metadata references", rewritten)
    rewritten_request = {**req_data, "messages": new_messages}
    if transfer_items:
        ec_transfer_params = dict(req_data.get("ec_transfer_params") or {})
        ec_transfer_params["ec_items"] = transfer_items
        rewritten_request["ec_transfer_params"] = ec_transfer_params
    return rewritten_request


def extract_mm_items(request_data: dict) -> list[dict]:
    """
    Return *all* image/audio/video items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: list[dict] = []
    for msg in request_data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in MM_TYPES:
                items.append(item)
    return items


async def fanout_encoder_primer(
    orig_request: dict,
    e_urls: list[str],
    req_id: str,
    consumer_zmq: str | None = None,
) -> tuple[dict[int, dict], dict[str, Any]]:
    """
    1. Build one request *per MM item* with all text removed.
    2. Send them concurrently to the encode cluster.
    3. Raise if any of them fails.

    Returns, per item index, the metadata the encoder reported in
    `ec_transfer_params`: its EC cache key and the grid its processor produced.
    The proxy still supplies the uuid so both sides key the cache the same way;
    the grid can only come from the encoder, which is the side that computed it.

    Also returns the connector handles to put on the decode body, as a fresh
    mapping. `orig_request` is left untouched so a retry re-encodes from the
    original request instead of carrying the previous attempt's handles.
    """
    logger.info("[%s] Processing multimodal items...", req_id)

    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        logger.info("[%s] No multimodal items, skipping encoder", req_id)
        return {}, {}  # nothing to do

    logger.info("[%s] got %d multimodal items...", req_id, len(mm_items))

    tasks = []
    item_uuids: dict[int, str] = {}
    item_transfer_ids: dict[int, str] = {}
    item_meta: dict[int, dict] = {}
    ec_params: dict[str, Any] = {}

    # Round-robin over encode servers to distribute load a bit. The cursor
    # persists across requests so fan-out doesn't restart at e_urls[0] every
    # time (which would hot-spot the first encoder for single-item requests).
    global encoder_rr_idx
    async with encoder_rr_lock:
        url_cycle, encoder_rr_idx = encoder_rr_assignment(
            e_urls, encoder_rr_idx, len(mm_items)
        )

    for idx, (item, target_url) in enumerate(zip(mm_items, url_cycle)):
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{req_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}

        # With --no-rewrite the decoder still receives the raw image and derives
        # the cache key by hashing it, so the encoder must do the same -- passing
        # a uuid here would make the two disagree and silently defeat the EC
        # transfer, leaving the decoder to encode the image itself.
        item_uuid = None if NO_REWRITE else content_uuid(item)
        if item_uuid is not None:
            item_uuids[idx] = item_uuid
        transfer_id = uuid.uuid4().hex
        item_transfer_ids[idx] = transfer_id

        encoder_req = {
            # You *may* need to keep additional fields
            "model": orig_request.get("model"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        item if item_uuid is None else {**item, "uuid": item_uuid}
                    ],
                },
            ],
            # No max_tokens cap: the encoder instance never samples, it finishes
            # once the prompt is encoded and its embeddings are published.
            "stream": False,
        }
        if consumer_zmq is not None:
            # No mm_hash here on purpose. The encoder's own
            # `mm_features[i].identifier` is derived from the uuid *and* the
            # engine's media_io_kwargs / mm_processor_kwargs, so this proxy
            # cannot know it before the encoder runs. Sending the bare uuid
            # would fail the connector's hash match and make the producer
            # invent its own transfer id, which the consumer could then never
            # cancel. Omitting it lets the connector match by position, which
            # is exact: one encoder request carries exactly one item.
            encoder_req["ec_transfer_params"] = {
                "consumer_zmq": consumer_zmq,
                "ec_items": [{"transfer_id": transfer_id}],
            }
        tasks.append(
            encode_session.post(
                f"{target_url}/v1/chat/completions",
                json=encoder_req,
                headers=headers,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Fail fast if any sub-request failed
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(
                "[%s] Encoder request #%d raised exception: %s",
                req_id,
                idx,
                r,
                exc_info=r,
            )
            raise HTTPException(
                status_code=502, detail=f"Encoder request failed: {str(r)}"
            )
        if r.status != 200:
            try:
                detail = await r.text()
            except Exception:
                detail = "<unable to read body>"
            logger.error(
                "[%s] Encoder request #%d returned status %s: %s",
                req_id,
                idx,
                r.status,
                detail,
            )
            raise HTTPException(
                status_code=r.status,
                detail=f"Encoder request failed: {detail}",
            )

        # The encoder reports each mm_hash's metadata (e.g. the grid) here,
        # keyed by the same uuid this proxy assigned above.
        try:
            params = (await r.json()).get("ec_transfer_params") or {}
        except Exception:
            logger.warning("[%s] Could not read encoder metadata #%d", req_id, idx)
            params = {}
        if params:
            # One encoder request carries exactly one item, so there is a
            # single reported entry. Do not key it by this proxy's uuid: when
            # media_io_kwargs or mm_processor_kwargs are set the engine
            # re-hashes the uuid together with them, so the encoder's own
            # `mm_features[i].identifier` is a derived value this proxy cannot
            # predict. Fall back to the sole entry, and carry the key the
            # encoder actually used through as `ec_mm_hash`.
            ec_mm_hash = item_uuids.get(idx)
            reported = params.get(ec_mm_hash)
            if reported is None and len(params) == 1:
                ((ec_mm_hash, reported),) = params.items()
            if reported:
                metadata = reported.get("metadata") or {}
                if metadata:
                    item_meta[idx] = {
                        **metadata,
                        "mm_hash": item_uuids.get(idx, ec_mm_hash),
                        "ec_mm_hash": ec_mm_hash,
                    }
                    if idx in item_transfer_ids:
                        item_meta[idx]["transfer_id"] = item_transfer_ids[idx]
                # Whatever the encoder reported alongside `metadata` is the
                # connector's own handle on the published embedding (for NIXL,
                # peer_host/peer_port/size_bytes). The decoder's connector
                # looks it up by mm_hash on the request, so carry it through.
                ec_params[item_uuids.get(idx, ec_mm_hash)] = reported
                if NO_REWRITE and consumer_zmq is not None:
                    ec_params.setdefault("ec_items", []).append(
                        {"mm_hash": ec_mm_hash, "transfer_id": item_transfer_ids[idx]}
                    )

    logger.info(
        "[%s] All %d encoder requests completed successfully", req_id, len(mm_items)
    )
    return item_meta, ec_params


async def maybe_prefill(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """
    - Do prefill-only task if p_url exist;
    - Return a new body carrying kv transfer params (for nixl connector)
    - Else, skip and return the original request data for decode

    `req_data` is never mutated: a decode retry re-enters this function with the
    same body, and one attempt's `remote_block_ids` must not reach the next.
    """
    if p_url:
        logger.info("[%s] Processing through prefill: %s", req_id, p_url)

        prefill_response = await process_prefill_stage(req_data, p_url, req_id)
        # for nixl connector to facilitate kv transfer...
        prefill_response_json = await prefill_response.json()
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            return {**req_data, "kv_transfer_params": kv_transfer_params}

    return req_data


async def process_prefill_stage(
    req_data: dict,
    p_url: str,
    req_id: str,
) -> dict:
    """Process request through Prefill stage and return kv_transfer_params"""
    logger.info("[%s] Sending prefill request to: %s", req_id, p_url)

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
    if "stream_options" in prefill_request:
        del prefill_request["stream_options"]

    headers = {"x-request-id": req_id}
    try:
        prefill_response = await prefill_session.post(
            f"{p_url}/v1/chat/completions", json=prefill_request, headers=headers
        )
        prefill_response.raise_for_status()

        if prefill_response.status != 200:
            error_text = await prefill_response.text()
            logger.error(
                "[%s] Prefill request failed with status %d: %s",
                req_id,
                prefill_response.status,
                error_text,
            )
            raise HTTPException(
                status_code=prefill_response.status,
                detail={"error": "Prefill request failed", "message": error_text},
            )
        logger.info("[%s] Prefill request completed successfully", req_id)

        return prefill_response

    except Exception as e:
        logger.error("Prefill processing failed: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": "Prefill processing error", "message": str(e)},
        ) from e


###############################################################################
# Middleware for request/response logging
###############################################################################


async def log_requests(request: Request, call_next):
    """Middleware to log all incoming requests and responses"""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Log incoming request
    logger.info(
        ">>> [%s] %s %s from %s",
        req_id,
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            "<<< [%s] %s %s completed with status %d",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
        )

        return response
    except Exception as e:
        # Log errors
        logger.exception(
            "!!! [%s] %s %s failed with error: %s",
            req_id,
            request.method,
            request.url.path,
            str(e),
        )
        raise


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, prefill_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    # vLLM closes an idle keep-alive connection after
    # VLLM_HTTP_TIMEOUT_KEEP_ALIVE seconds (5 by default), while aiohttp keeps
    # pooling it for 15. Reusing one it has already closed fails the request
    # with ServerDisconnectedError, and the server logs nothing at all: it
    # closed the socket before the request arrived. Retire ours first.
    server_keep_alive = float(os.getenv("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "5"))
    connector = aiohttp.TCPConnector(
        limit=0,
        **(
            {"keepalive_timeout": server_keep_alive / 2}
            if server_keep_alive > 0
            else {"force_close": True}
        ),
    )
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    if app.state.p_urls:
        # only setup if prefill instance(s) exist
        prefill_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def prepare_for_decode(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    consumer_zmq: str | None,
) -> tuple[dict, float, float]:
    """Encode, rewrite and prefill, returning the body to send to decode.

    `req_data` is left untouched so a retry starts from the original media
    rather than from a body whose images are already metadata references.
    """
    _t0 = time.perf_counter()
    item_meta, ec_params = await fanout_encoder_primer(
        req_data, e_urls, req_id, consumer_zmq
    )
    _t1 = time.perf_counter()
    prepared = req_data if NO_REWRITE else rewrite_for_decode(req_data, item_meta)
    if ec_params:
        # A fresh body every time: `rewrite_for_decode` hands back `req_data`
        # itself when it rewrote nothing, and this attempt's handles must not
        # outlive it into a retry.
        handles = dict(prepared.get("ec_transfer_params") or {})
        handles.update(ec_params)
        prepared = {**prepared, "ec_transfer_params": handles}
    _t2 = time.perf_counter()
    prepared = await maybe_prefill(prepared, p_url, req_id)
    return prepared, _t1 - _t0, _t2 - _t1


async def forward_non_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    consumer_zmq: str | None,
    dp_rank: int | None = None,
) -> dict:
    try:
        for attempt in range(DECODE_RETRIES + 1):
            _t0 = time.perf_counter()
            prepared, encode_s, rewrite_s = await prepare_for_decode(
                req_data, req_id, e_urls, p_url, consumer_zmq
            )
            _t2 = time.perf_counter()

            logger.info("[%s] Forwarding to decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id}
            if dp_rank is not None:
                headers["X-data-parallel-rank"] = str(dp_rank)

            async with decode_session.post(
                f"{d_url}/v1/chat/completions", json=prepared, headers=headers
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    # 500 is the decoder's retryable internal error, which
                    # includes an encoder embedding it could not obtain. Redoing
                    # the encode publishes the item again.
                    if resp.status == 500 and attempt < DECODE_RETRIES:
                        logger.warning(
                            "[%s] Decode returned 500, re-encoding and retrying "
                            "(attempt %d/%d): %s",
                            req_id,
                            attempt + 1,
                            DECODE_RETRIES,
                            detail[:200],
                        )
                        continue
                    logger.error(
                        "[%s] Decode request returned status %s: %s",
                        req_id,
                        resp.status,
                        detail,
                    )
                    raise HTTPException(status_code=resp.status, detail=detail)
                out = await resp.json()
                _t3 = time.perf_counter()
                logger.info(
                    "STAGE %s encode=%.1f rewrite=%.1f decode=%.1f total=%.1f "
                    "attempt=%d",
                    "no-rewrite" if NO_REWRITE else "rewrite",
                    encode_s * 1e3,
                    rewrite_s * 1e3,
                    (_t3 - _t2) * 1e3,
                    (_t3 - _t0) * 1e3,
                    attempt,
                )
                return out
        raise HTTPException(status_code=500, detail="Decode failed after re-encoding")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_non_stream: %s", req_id, str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}") from e


async def forward_stream(
    req_data: dict,
    req_id: str,
    e_urls: list[str],
    p_url: str,
    d_url: str,
    consumer_zmq: str | None,
    dp_rank: int | None = None,
) -> AsyncIterator[str]:
    try:
        for attempt in range(DECODE_RETRIES + 1):
            _t0 = time.perf_counter()
            prepared, encode_s, rewrite_s = await prepare_for_decode(
                req_data, req_id, e_urls, p_url, consumer_zmq
            )
            _t2 = time.perf_counter()

            logger.info("[%s] Starting streaming from decode: %s", req_id, d_url)
            headers = {"x-request-id": req_id}
            if dp_rank is not None:
                headers["X-data-parallel-rank"] = str(dp_rank)

            _first = None
            async with decode_session.post(
                f"{d_url}/v1/chat/completions",
                json=prepared,
                headers=headers,
            ) as resp:
                # Retry only before the first chunk: once anything reached the
                # client the response cannot be replaced.
                if resp.status == 500 and attempt < DECODE_RETRIES:
                    detail = await resp.text()
                    logger.warning(
                        "[%s] Decode returned 500 before streaming, re-encoding "
                        "and retrying (attempt %d/%d): %s",
                        req_id,
                        attempt + 1,
                        DECODE_RETRIES,
                        detail[:200],
                    )
                    continue
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(1024):
                    if chunk:
                        if _first is None:
                            _first = time.perf_counter()
                        yield chunk.decode("utf-8", errors="ignore")
            _t3 = time.perf_counter()

            logger.info(
                "STAGE %s encode=%.1f rewrite=%.2f decode_ttfb=%.1f "
                "decode_total=%.1f attempt=%d",
                "no-rewrite" if NO_REWRITE else "rewrite",
                encode_s * 1e3,
                rewrite_s * 1e3,
                ((_first or _t3) - _t2) * 1e3,
                (_t3 - _t2) * 1e3,
                attempt,
            )
            logger.info("[%s] Streaming completed", req_id)
            return

    except HTTPException:
        logger.exception("[%s] HTTPException in forward_stream", req_id)
        raise
    except Exception as e:
        logger.exception("[%s] Error in forward_stream: %s", req_id, str(e))
        raise HTTPException(
            status_code=500, detail=f"Proxy streaming error: {str(e)}"
        ) from e


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        e_urls = app.state.e_urls  # we want the full list for fan-out
        p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
        decode_index = random.randrange(len(app.state.d_urls))
        d_url = app.state.d_urls[decode_index]
        dp_size = app.state.ec_consumer_dp_size
        # Round-robin the replica, then name it to both halves: the decoder
        # honours the rank header instead of its own balancer, and the encoder
        # pushes to that replica's control channel. Choosing once here means a
        # decode retry re-encodes to the same replica.
        dp_rank = next(app.state.replica_counter) % dp_size if dp_size > 1 else None
        ec_index = decode_index * dp_size + (dp_rank or 0)
        consumer_zmq = app.state.d_ec_urls[ec_index] if app.state.d_ec_urls else None

        is_streaming = req_data.get("stream", False)

        if is_streaming:
            return StreamingResponse(
                forward_stream(
                    req_data, req_id, e_urls, p_url, d_url, consumer_zmq, dp_rank
                ),
                media_type="text/event-stream",
            )
        result = await forward_non_stream(
            req_data, req_id, e_urls, p_url, d_url, consumer_zmq, dp_rank
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in chat_completions endpoint: %s", str(e))
        raise HTTPException(
            status_code=500, detail=f"Request processing error: {str(e)}"
        ) from e


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health_check():
    async def healthy(urls):
        if not urls:
            return "empty"
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return "unhealthy"
        return "healthy"

    e_status, p_status, d_status = await asyncio.gather(
        healthy(app.state.e_urls), healthy(app.state.p_urls), healthy(app.state.d_urls)
    )

    overall_healthy = all(
        status != "unhealthy" for status in (e_status, p_status, d_status)
    )

    status_code = 200 if overall_healthy else 503

    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": e_status,
            "prefill_cluster": p_status,
            "decode_cluster": d_status,
        },
        status_code=status_code,
    )


###############################################################################
# Simple profiler fan-out (unchanged except for sessions)
###############################################################################


async def _post_if_available(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    headers: dict,
) -> dict | None:
    """
    POST `payload` to `url`.

    Returns
    -------
    • The decoded JSON body on success (2xx)
    • None if the endpoint does not exist (404)
    • Raises for anything else.
    """
    try:
        resp = await session.post(url, json=payload, headers=headers)
        if resp.status == 404:  # profiling disabled on that server
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        resp.raise_for_status()
        return await resp.json(content_type=None)
    except aiohttp.ClientResponseError as exc:
        # Pass 404 through the branch above, re-raise everything else
        if exc.status == 404:
            logger.warning("Profiling endpoint missing on %s", url)
            return None
        raise
    except Exception:
        # Network errors etc.: propagate
        raise


async def _profile_cmd(cmd: str, payload: dict, e_url: str, p_url: str, d_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    prefill_task = (
        _post_if_available(prefill_session, f"{p_url}/{cmd}_profile", payload, headers)
        if p_url is not None
        else asyncio.sleep(0)
    )
    decode_task = _post_if_available(
        decode_session, f"{d_url}/{cmd}_profile", payload, headers
    )

    encode_res, prefill_res, decode_res = await asyncio.gather(
        encode_task, prefill_task, decode_task
    )

    # If *all* clusters said “I don’t have that route”, surface an error
    if encode_res is prefill_res is decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on all clusters",
        )

    return {
        "encode": encode_res,  # may be None
        "prefill": prefill_res,  # may be None
        "decode": decode_res,  # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("start", body, e_url, p_url, d_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    # TODO: handle multi urls properly
    e_url = random.choice(app.state.e_urls)
    p_url = random.choice(app.state.p_urls) if app.state.p_urls else None
    d_url = random.choice(app.state.d_urls)
    return await _profile_cmd("stop", body, e_url, p_url, d_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-requests",
        action="store_true",
        help=(
            "Log every request in and out, and raise the log level to DEBUG. "
            "Off by default: the proxy is on the request path."
        ),
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Forward images to the decoder unchanged (for stage-timing A/B).",
    )
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-servers-urls",
        required=True,
        help=(
            'Comma-separated prefill URLs ("http://p1:8003,http://p2:8004") '
            'to enable E->P->D, set "disable" or "none" to enable E->PD'
        ),
    )
    parser.add_argument(
        "--decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8005,http://d2:8006")',
    )
    parser.add_argument(
        "--decode-retries",
        type=int,
        default=1,
        help=(
            "Re-encode and re-send when decode returns 500, which is its "
            "retryable internal error (an undeliverable encoder embedding "
            "among them). 0 disables."
        ),
    )
    parser.add_argument(
        "--ec-consumer-zmq-addrs",
        default="",
        help=(
            "Comma-separated Mooncake EC consumer control addresses, aligned "
            "with --decode-servers-urls. Required for Mooncake EC consumers and "
            "supported only in E+PD mode. With --ec-consumer-dp-size > 1, list "
            "each server's replicas consecutively: s0r0,s0r1,s1r0,s1r1."
        ),
    )
    parser.add_argument(
        "--ec-consumer-dp-size",
        type=int,
        default=1,
        help=(
            "Data-parallel replicas per EC consumer. The proxy picks a replica "
            "round-robin and names it to both halves of the request, because an "
            "encoder push has to land where the request will run."
        ),
    )

    args = parser.parse_args()
    if args.log_requests:
        logging.getLogger().setLevel(logging.DEBUG)
        app.middleware("http")(log_requests)
    NO_REWRITE = args.no_rewrite
    DECODE_RETRIES = max(0, args.decode_retries)
    app.state.e_urls = [
        u.strip() for u in args.encode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_urls = [
        u.strip() for u in args.decode_servers_urls.split(",") if u.strip()
    ]
    app.state.d_ec_urls = [
        u.strip() for u in args.ec_consumer_zmq_addrs.split(",") if u.strip()
    ]
    if args.ec_consumer_dp_size < 1:
        parser.error("--ec-consumer-dp-size must be at least 1")
    app.state.ec_consumer_dp_size = args.ec_consumer_dp_size
    app.state.replica_counter = itertools.count()
    expected = len(app.state.d_urls) * args.ec_consumer_dp_size
    if app.state.d_ec_urls and len(app.state.d_ec_urls) != expected:
        parser.error(
            "--ec-consumer-zmq-addrs must contain one address per consumer "
            f"replica: expected {expected} "
            f"({len(app.state.d_urls)} servers x {args.ec_consumer_dp_size} replicas), "
            f"got {len(app.state.d_ec_urls)}"
        )
    # handle prefill instances
    if args.prefill_servers_urls.lower() in ("disable", "none", ""):
        app.state.p_urls = []
        logger.info(
            "Disaggregated prefill phase explicitly disabled by user. Running E + PD..."
        )
    else:
        app.state.p_urls = [
            u.strip() for u in args.prefill_servers_urls.split(",") if u.strip()
        ]
        logger.info("Disaggregated prefill phase is enabled. Running E + P + D...")
    try:
        validate_ec_consumer_routing(app.state.p_urls, app.state.d_ec_urls)
    except ValueError as exc:
        parser.error(str(exc))

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Prefill instances %s", app.state.p_urls)
    logger.info("Decode servers: %s", app.state.d_urls)
    if app.state.ec_consumer_dp_size > 1:
        logger.info(
            "EC consumer replicas per server: %d (control addresses: %s)",
            app.state.ec_consumer_dp_size,
            app.state.d_ec_urls,
        )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=True,
    )
