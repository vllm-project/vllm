# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import ipaddress
import itertools
import logging
import os
import urllib
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (url, bootstrap_port) in enumerate(global_args.prefill):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": hostname,
                "port": parsed_url.port,
                "bootstrap_port": bootstrap_port or 8998,
                "id": i,
            }
        )

    # Create decode clients
    for i, url in enumerate(global_args.decode):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": hostname,
                "port": parsed_url.port,
                "id": i,
            }
        )

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Initialized {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "bootstrap_port"),
        help=(
            "Prefill server URL and optional bootstrap port. "
            "Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, "
            "'none', or omitted (defaults to none)."
        ),
    )

    # For decoder instances
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar=("URL",),
        help="Decode server URL. Can be specified multiple times.",
    )

    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)

    return args


# From sglang router_args.py
def _parse_prefill_urls(prefill_list):
    """Parse prefill URLs from --prefill arguments.

    Format: --prefill URL [BOOTSTRAP_PORT]
    Example:
        --prefill http://prefill1:8080 9000  # With bootstrap port
        --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
        --prefill http://prefill3:8080       # Defaults to no bootstrap port
    """
    if not prefill_list:
        return []

    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]

        # Handle optional bootstrap port
        if len(prefill_args) >= 2:
            bootstrap_port_str = prefill_args[1]
            # Handle 'none' as None
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"  # noqa: E501
                    ) from e
        else:
            # No bootstrap port specified, default to None
            bootstrap_port = None

        prefill_urls.append((url, bootstrap_port))

    return prefill_urls


def _parse_decode_urls(decode_list):
    """Parse decode URLs from --decode arguments.

    Format: --decode URL
    Example: --decode http://decode1:8081 --decode http://decode2:8081
    """
    if not decode_list:
        return []

    # decode_list is a list of single-element lists due to nargs=1
    return [url[0] for url in decode_list]


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # CRITICAL: Release connection back to pool
    await response.aclose()


async def stream_service_response(
    prefill_client_info: dict,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    req_data["kv_transfer_params"] = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "bootstrap_server_host": prefill_client_info["host"],
        "bootstrap_server_port": prefill_client_info["bootstrap_port"],
    }

    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")

        # Send request to prefill service
        asyncio.create_task(
            send_request_to_service(prefill_client_info, api, req_data, request_id)
        )

        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(
                prefill_client_info,
                decode_client_info,
                api,
                req_data,
                request_id=request_id,
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
