# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-robin proxy across independent, non-disaggregated vLLM replicas.

Unlike toy_proxy_server.py (which splits a single request across a
prefill instance and a decode instance via NixlConnector), this proxy
sends each incoming request in full to exactly one backend replica.
Each replica performs its own prefill *and* decode locally. This is the
"just add more GPUs, no disaggregation" arm used to benchmark whether
prefill/decode disaggregation is actually worth its transfer overhead,
at equal GPU budget.
"""

import argparse
import itertools
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize one client per replica.
    app.state.clients = []
    for i, (host, port) in enumerate(global_args.replica_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.iterator = itertools.cycle(range(len(app.state.clients)))

    print(f"Initialized {len(app.state.clients)} replica clients.")

    yield

    # Shutdown: close all clients.
    for client_info in app.state.clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Round-robin proxy across independent, non-disaggregated "
            "vLLM replicas."
        )
    )

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--replica-hosts",
        "--replica-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--replica-ports", "--replica-port", type=int, nargs="+", default=[8100]
    )

    args = parser.parse_args()

    if len(args.replica_hosts) != len(args.replica_ports):
        raise ValueError("Number of replica hosts must match number of replica ports")

    args.replica_instances = list(zip(args.replica_hosts, args.replica_ports))

    return args


def get_next_client(app):
    """Get the next replica client in round-robin fashion."""
    client_idx = next(app.state.iterator)
    return app.state.clients[client_idx]


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """Asynchronously stream response from a replica using a pooled client."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        client_info = get_next_client(request.app)

        async def generate_stream():
            async for chunk in stream_service_response(
                client_info, api, req_data, request_id=request_id
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        exc_info = sys.exc_info()
        print(f"Error occurred in round-robin proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "replica_instances": len(app.state.clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
