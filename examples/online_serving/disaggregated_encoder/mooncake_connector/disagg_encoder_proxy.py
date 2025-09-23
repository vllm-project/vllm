#!/usr/bin/env python3
"""
disagg_encoder_proxy.py

Proxy that routes OpenAI-compatible “/v1/chat/completions” requests to two
clusters:
  • encode  (multimodal feature extraction)
  • decode  (language-model inference)

For MM input we:
    1. Extract *every* image/audio item.
    2. Fire N concurrent requests to the encoder cluster
       (one request per item, with **all text removed**).
    3. Wait for all of them to succeed.
    4. Forward the *original* request to a decode server.

Usage
$ python disagg_encoder_proxy.py \
      --encode-servers-urls "http://e1:8001,http://e2:8001" \
      --prefill-decode-servers-urls "http://d1:8003,http://d2:8003"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import uuid
from copy import deepcopy
from typing import Any, AsyncIterator, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

###############################################################################
# FastAPI app & global state
###############################################################################

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("proxy")

app = FastAPI()
encode_session: Optional[aiohttp.ClientSession] = None
decode_session: Optional[aiohttp.ClientSession] = None

###############################################################################
# Utils
###############################################################################


MM_TYPES = {"image_url", "audio_url", "input_audio"}


def extract_mm_items(request_data: dict) -> List[dict]:
    """
    Return *all* image/audio items that appear anywhere in `messages`.

    Each returned dict looks like:
        { "type": "image_url", "image_url": {...} }
    """
    items: List[dict] = []
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
    e_urls: List[str],
    request_id: str,
) -> None:
    """
    1. Build one request *per MM item* with all text removed.
    2. Send them concurrently to the encode cluster.
    3. Raise if any of them fails.
    """
    mm_items = extract_mm_items(orig_request)
    if not mm_items:
        return  # nothing to do

    tasks = []

    # Round-robin over encode servers to distribute load a bit
    url_cycle = (e_urls[i % len(e_urls)] for i in range(len(mm_items)))

    for idx, (item, target_url) in enumerate(zip(mm_items, url_cycle)):
        # Derive a *child* request id:  <parent>:<index>:<random-short>
        child_req_id = f"{request_id}:{idx}:{uuid.uuid4().hex[:6]}"
        headers = {"x-request-id": child_req_id}

        encoder_req = {
            # You *may* need to keep additional fields
            "model": orig_request.get("model"),
            "messages": [
                {"role": "user", "content": [item]},
            ],
            # Only need 1 token so the server actually runs the encoder path
            "max_tokens": 1,
            "stream": False,
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
    for r in results:
        if isinstance(r, Exception):
            logger.error("Encoder request raised: %s", r)
            raise HTTPException(status_code=502, detail=str(r))
        if r.status != 200:
            try:
                detail = await r.text()
            except Exception:
                detail = "<unable to read body>"
            logger.error("Encoder request returned %s: %s", r.status, detail)
            raise HTTPException(
                status_code=r.status,
                detail=f"Encoder request failed: {detail}",
            )


###############################################################################
# FastAPI lifecycle
###############################################################################


@app.on_event("startup")
async def on_startup() -> None:
    global encode_session, decode_session
    timeout = aiohttp.ClientTimeout(total=100_000)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    encode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    decode_session = aiohttp.ClientSession(timeout=timeout, connector=connector)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await encode_session.close()
    await decode_session.close()


###############################################################################
# Core forwarding
###############################################################################


async def forward_non_stream(
    req_data: dict, req_id: str, e_urls: List[str], pd_url: str
) -> dict:
    await fanout_encoder_primer(req_data, e_urls, req_id)

    headers = {"x-request-id": req_id}
    async with decode_session.post(
        f"{pd_url}/v1/chat/completions", json=req_data, headers=headers
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def forward_stream(
    req_data: dict, req_id: str, e_urls: List[str], pd_url: str
) -> AsyncIterator[str]:
    await fanout_encoder_primer(req_data, e_urls, req_id)

    headers = {"x-request-id": req_id}
    async with decode_session.post(
        f"{pd_url}/v1/chat/completions",
        json=req_data,
        headers=headers,
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_chunked(1024):
            if chunk:
                yield chunk.decode("utf-8", errors="ignore")


###############################################################################
# Public routes
###############################################################################


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_data = await request.json()
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    pd_url = random.choice(app.state.pd_urls)
    e_urls = app.state.e_urls  # we want the full list for fan-out

    if req_data.get("stream", False):
        return StreamingResponse(
            forward_stream(req_data, req_id, e_urls, pd_url),
            media_type="text/event-stream",
        )
    result = await forward_non_stream(req_data, req_id, e_urls, pd_url)
    return JSONResponse(content=result)


@app.get("/v1/models")
async def list_models():
    async with decode_session.get(f"{app.state.pd_urls[0]}/v1/models") as resp:
        resp.raise_for_status()
        return await resp.json()


@app.get("/health")
async def health():
    async def healthy(urls):
        for u in urls:
            try:
                async with encode_session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            except Exception:
                return False
        return True

    e_ok, pd_ok = await asyncio.gather(
        healthy(app.state.e_urls), healthy(app.state.pd_urls)
    )

    status_code = 200 if e_ok and pd_ok else 503
    return JSONResponse(
        {
            "proxy": "healthy",
            "encode_cluster": "healthy" if e_ok else "unhealthy",
            "decode_cluster": "healthy" if pd_ok else "unhealthy",
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
) -> Optional[dict]:
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
        if resp.status == 404:           # profiling disabled on that server
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


async def _profile_cmd(cmd: str, payload: dict, e_url: str, pd_url: str):
    """
    Fire & forget to both clusters, tolerate 404.
    """
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}

    encode_task = _post_if_available(
        encode_session, f"{e_url}/{cmd}_profile", payload, headers
    )
    decode_task = _post_if_available(
        decode_session, f"{pd_url}/{cmd}_profile", payload, headers
    )

    encode_res, decode_res = await asyncio.gather(encode_task, decode_task)

    # If *both* clusters said “I don’t have that route”, surface an error
    if encode_res is None and decode_res is None:
        raise HTTPException(
            status_code=503,
            detail="Profiling endpoints are disabled on both clusters",
        )

    return {
        "encode": encode_res,   # may be None
        "decode": decode_res,   # may be None
    }


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    e_url = random.choice(app.state.e_urls)
    pd_url = random.choice(app.state.pd_urls)
    return await _profile_cmd("start", body, e_url, pd_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    e_url = random.choice(app.state.e_urls)
    pd_url = random.choice(app.state.pd_urls)
    return await _profile_cmd("stop", body, e_url, pd_url)


###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated encode URLs ("http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-decode-servers-urls",
        required=True,
        help='Comma-separated decode URLs ("http://d1:8003,http://d2:8003")',
    )
    args = parser.parse_args()

    app.state.e_urls = [u.strip() for u in args.encode_servers_urls.split(",") if u.strip()]
    app.state.pd_urls = [u.strip() for u in args.prefill_decode_servers_urls.split(",") if u.strip()]

    logger.info("Proxy listening on %s:%s", args.host, args.port)
    logger.info("Encode servers: %s", app.state.e_urls)
    logger.info("Decode servers: %s", app.state.pd_urls)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="uvloop",
        access_log=False,
    )
