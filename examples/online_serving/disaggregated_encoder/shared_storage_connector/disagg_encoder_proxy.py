# disagg_encoder_proxy.py
"""
FastAPI-based proxy that routes “/v1/chat/completions” requests to a
pair of back-end clusters:
  • encode (multimodal feature extraction)
  • prefill/decode (language model inference)

It supports both streaming and non-streaming Chat Completions, health
checks, and an ad-hoc profiling API that can be fanned out to all
back-ends.

Run:

$ python disagg_encoder_proxy.py \
    --encode-servers-urls "http://localhost:8001" \
    --prefill-decode-servers-urls "http://localhost:8003"
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

encode_session: Optional[aiohttp.ClientSession] = None
decode_session: Optional[aiohttp.ClientSession] = None


# ---------------------------------------------------------------------------
# FastAPI life-cycle hooks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    global encode_session, decode_session

    encode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=100_000),
    )
    decode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=100_000),
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global encode_session, decode_session

    if encode_session:
        await encode_session.close()
    if decode_session:
        await decode_session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_mm_input(request_data: dict) -> bool:
    """Return True if the chat request contains image or audio content."""
    if "messages" not in request_data:
        return False

    for message in request_data["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if item.get("type") in {"image_url", "audio_url", "input_audio"}:
                return True
    return False


# ---------------------------------------------------------------------------
# Core forwarding logic
# ---------------------------------------------------------------------------


async def forward_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    pd_server_url: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks from the decode server, optionally priming the encoder."""
    headers = {"x-request-id": request_id}

    # 1. Kick off a 1-token request to the encoder if MM input is present.
    if has_mm_input(request_data):
        encoder_req = copy.deepcopy(request_data)
        encoder_req["max_tokens"] = 1
        if "max_completion_tokens" in encoder_req:
            encoder_req["max_completion_tokens"] = 1

        try:
            resp = await encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=encoder_req,
                headers=headers,
            )
            if resp.status != 200:
                raise HTTPException(
                    status_code=resp.status,
                    detail={"error": "Encoder request failed", "message": await resp.text()},
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 2. Stream from the decode server.
    try:
        async with decode_session.post(
            f"{pd_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(128):
                if chunk:
                    yield chunk.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Error in streaming: %s", exc)
        raise


async def forward_non_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    pd_server_url: str,
) -> dict:
    """Return full JSON response from decode server, with optional encoder prime."""
    headers = {"x-request-id": request_id}

    # 1. Optional encoder prime.
    if has_mm_input(request_data):
        encoder_req = copy.deepcopy(request_data)
        encoder_req["max_tokens"] = 1
        if "max_completion_tokens" in encoder_req:
            encoder_req["max_completion_tokens"] = 1

        try:
            resp = await encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=encoder_req,
                headers=headers,
            )
            if resp.status != 200:
                raise HTTPException(
                    status_code=resp.status,
                    detail={"error": "Encoder request failed", "message": await resp.text()},
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 2. Full decode request.
    try:
        async with decode_session.post(
            f"{pd_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as exc:
        logger.error("Error in non-streaming: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Public API routes
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle OpenAI-compatible Chat Completion requests."""
    try:
        e_server_url = random.choice(app.state.e_urls)
        pd_server_url = random.choice(app.state.pd_urls)

        req_data = await request.json()
        req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        is_stream = req_data.get("stream", False)

        if is_stream:
            return StreamingResponse(
                forward_streaming_request(req_data, req_id, e_server_url, pd_server_url),
                media_type="text/event-stream",
            )
        result = await forward_non_streaming_request(
            req_data, req_id, e_server_url, pd_server_url
        )
        return JSONResponse(content=result)

    except Exception as exc:
        logger.error("Error processing request: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/v1/models")
async def list_models():
    """Proxy the model list from the first decode server."""
    try:
        async with decode_session.get(f"{app.state.pd_urls[0]}/v1/models") as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as exc:
        logger.error("Error fetching models: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    """Aggregate health of proxy, encode, and decode servers."""
    async def check(urls, session):
        try:
            for u in urls:
                async with session.get(f"{u}/health") as resp:
                    resp.raise_for_status()
            return True
        except Exception:
            return False

    encode_ok, decode_ok = await asyncio.gather(
        check(app.state.e_urls, encode_session),
        check(app.state.pd_urls, encode_session),
    )

    status = {
        "proxy": "healthy",
        "encode_servers": "healthy" if encode_ok else "unhealthy",
        "prefill_decode_servers": "healthy" if decode_ok else "unhealthy",
    }
    return JSONResponse(content=status, status_code=200 if encode_ok and decode_ok else 503)


# ---------------------------------------------------------------------------
# Profiling fan-out helpers
# ---------------------------------------------------------------------------


async def send_profile_cmd(
    request: Request,
    req_data: dict,
    cmd: str,
    e_server_url: str,
    pd_server_url: str,
):
    """Fan out a profiler start/stop command to all back-ends."""
    assert cmd in {"start", "stop"}

    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

    tasks = [
        encode_session.post(f"{e_server_url}/{cmd}_profile", json=req_data, headers=headers),
        decode_session.post(f"{pd_server_url}/{cmd}_profile", json=req_data, headers=headers),
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for r in responses:
        if isinstance(r, Exception):
            raise r
        r.raise_for_status()

    return await responses[0].json(content_type=None)


@app.post("/start_profile")
async def start_profile(request: Request):
    """Start profiling on all back-ends."""
    e_server_url = random.choice(app.state.e_urls)
    pd_server_url = random.choice(app.state.pd_urls)
    data = await request.json()
    return await send_profile_cmd(request, data, "start", e_server_url, pd_server_url)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    """Stop profiling on all back-ends."""
    e_server_url = random.choice(app.state.e_urls)
    pd_server_url = random.choice(app.state.pd_urls)
    data = await request.json()
    return await send_profile_cmd(request, data, "stop", e_server_url, pd_server_url)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Proxy for distributed vLLM servers")
    parser.add_argument("--host", default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")
    parser.add_argument(
        "--encode-servers-urls",
        required=True,
        help='Comma-separated list of encode server URLs (e.g. "http://e1:8001,http://e2:8001")',
    )
    parser.add_argument(
        "--prefill-decode-servers-urls",
        required=True,
        help='Comma-separated list of prefill/decode server URLs (e.g. "http://d1:8003,http://d2:8003")',
    )

    args = parser.parse_args()
    app.state.e_urls = args.encode_servers_urls.split(",")
    app.state.pd_urls = args.prefill_decode_servers_urls.split(",")

    logger.info(
        "Starting API proxy on %s:%s (encode=%s, decode=%s)",
        args.host,
        args.port,
        app.state.e_urls,
        app.state.pd_urls,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop",
    )