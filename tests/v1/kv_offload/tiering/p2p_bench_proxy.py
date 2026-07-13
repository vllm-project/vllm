# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-target P2P injector proxy for the P2P-vs-baseline benchmark.

Unlike ``p2p_connector_proxy.py`` (which splits each request into a prefill on
one instance and a decode on another, PD-style), this proxy has a single
upstream: the P2P *consumer*. It injects the pre-warmed *producer*'s P2P
coordinates into every request so the consumer pulls the prompt's KV blocks from
the producer's CPU cache instead of computing prefill locally.

The injected shape matches what ``P2PSecondaryTierManager`` reads via
``_p2p_params`` (``vllm/v1/kv_offload/tiering/p2p/manager.py``):

    kv_transfer_params = {
        "p2p": {
            "kv_request_id": "<unique-per-request>",
            "remote_host": "<producer-ip>",
            "remote_port": <producer-p2p-port>,
        }
    }

Usage:
    python p2p_bench_proxy.py \
        --port 8192 --host 127.0.0.1 \
        --target-host 127.0.0.1 --target-port 8200 \
        --producer-p2p-host 10.0.0.5 --producer-p2p-port 5710
"""

import argparse
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        timeout=None,
        base_url=f"http://{global_args.target_host}:{global_args.target_port}/v1",
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
    print(
        f"P2P bench proxy ready: consumer="
        f"{global_args.target_host}:{global_args.target_port}, "
        f"producer P2P="
        f"{global_args.producer_p2p_host}:{global_args.producer_p2p_port}"
    )
    yield
    await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8192)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument(
        "--target-host",
        type=str,
        default="127.0.0.1",
        help="HTTP host of the P2P consumer vLLM instance",
    )
    p.add_argument(
        "--target-port",
        type=int,
        default=8200,
        help="HTTP port of the P2P consumer vLLM instance",
    )
    p.add_argument(
        "--producer-p2p-host",
        type=str,
        required=True,
        help="Host of the producer's P2PConnector ZMQ side-channel socket",
    )
    p.add_argument(
        "--producer-p2p-port",
        type=int,
        default=int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710")),
        help="Port of the producer's P2PConnector ZMQ side-channel socket "
        "(default: $VLLM_P2P_SIDE_CHANNEL_PORT or 5710)",
    )
    return p.parse_args()


def _auth_headers(request_id: str) -> dict:
    headers: dict = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _handle(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Tell the consumer to pull this prompt's KV from the producer peer.
        req_data["kv_transfer_params"] = {
            "p2p": {
                "kv_request_id": request_id,
                "remote_host": global_args.producer_p2p_host,
                "remote_port": global_args.producer_p2p_port,
            },
        }

        headers = _auth_headers(request_id)

        async def generate():
            async with request.app.state.client.stream(
                "POST", api, json=req_data, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "target": f"{global_args.target_host}:{global_args.target_port}",
        "producer_p2p": (
            f"{global_args.producer_p2p_host}:{global_args.producer_p2p_port}"
        ),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
