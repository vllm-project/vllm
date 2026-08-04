# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional
import argparse
import asyncio
import itertools
import os
import threading
import time
import uuid

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Global dictionary to store asyncio.Events indexed by request ID.
# This is shared between the main proxy app and the telemetry app.
pending_requests: dict[str, asyncio.Event] = {}
pending_requests_lock = threading.Lock()

# Reference to the main event loop (set at startup)
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    # Startup: Initialize clients
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    def pair_hosts_and_ports(hosts, ports, count=None):
        """
        Flexible host-port pairing with expansion strategies.
        """
        if not isinstance(hosts, list):
            hosts = [hosts]
        if not isinstance(ports, list):
            ports = [ports]
        if len(hosts) == 1 and len(ports) == 1:
            if count is None or count <= 1:
                return [(hosts[0], ports[0])]
            else:
                return [(hosts[0], ports[0] + i) for i in range(count)]
        if len(hosts) == 1:
            return [(hosts[0], p) for p in ports]
        if len(ports) == 1:
            return [(h, ports[0]) for h in hosts]
        if len(hosts) != len(ports):
            raise ValueError(
                "Length mismatch between hosts and ports lists for pairing"
            )
        return list(zip(hosts, ports, strict=False))

    prefill_pairs = pair_hosts_and_ports(
        pref_hosts, pref_ports, global_args.num_prefillers
    )
    for host, port in prefill_pairs:
        prefiller_base_url = f"http://{host}:{int(port)}"
        prefill_client = httpx.AsyncClient(timeout=None, base_url=prefiller_base_url)
        app.state.prefill_clients.append(ClientInfo(prefill_client))

    # Build decoder clients
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    decoder_pairs = pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)

    for host, port in decoder_pairs:
        decoder_base_url = f"http://{host}:{int(port)}"
        decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
        app.state.decode_clients.append(ClientInfo(decode_client))

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.aclose()
    for client in app.state.decode_clients:
        await client.aclose()


# Main proxy FastAPI app
app = FastAPI(lifespan=lifespan)


def csv_ints(s):
    return [int(x) for x in s.split(",")]


def csv_strs(s):
    return [x.strip() for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--prefiller-port", type=csv_ints, default=[8100])
    parser.add_argument("--num-prefillers", type=int, default=1)
    parser.add_argument("--decoder-host", type=csv_strs, default=["localhost"])
    parser.add_argument("--decoder-port", type=csv_ints, default=[8200])
    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--telemetry-port", type=int, default=5768)

    args = parser.parse_args()
    return args


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: Optional[str] = None

    async def aclose(self):
        await self.client.aclose()


# Initialize state
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []


def round_robin_pick_client(clients, idx):
    return clients[idx % len(clients)]


round_robin_counter = itertools.count()


def round_robin_pick_clients() -> tuple[ClientInfo, ClientInfo]:
    idx = next(round_robin_counter)
    prefill_client = round_robin_pick_client(app.state.prefill_clients, idx)
    decode_client = round_robin_pick_client(app.state.decode_clients, idx)
    return prefill_client, decode_client


def _build_headers(**extra: str) -> dict[str, str]:
    """Build common HTTP headers, including auth if OPENAI_API_KEY is set."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(extra)
    return headers


async def send_request_to_prefiller(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
):
    """
    Send a request to prefiller with request-id header.
    """
    headers = _build_headers(**{"X-Request-Id": request_id})
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def send_request_to_decoder(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
):
    """
    Send a request to decoder service.
    """
    headers = _build_headers()
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Asynchronously stream the response from a service.
    """
    headers = _build_headers()
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:16]


def create_pending_request(request_id: str) -> asyncio.Event:
    """Create an asyncio.Event for a request and store it."""
    event = asyncio.Event()
    with pending_requests_lock:
        pending_requests[request_id] = event
    return event


def remove_pending_request(request_id: str):
    """Remove a pending request from the dictionary."""
    with pending_requests_lock:
        pending_requests.pop(request_id, None)


def notify_request(chatcmpl_request_id: str) -> bool:
    """
    Notify a pending request that the KV store is complete.
    Returns True if the request was found and notified.
    """
    with pending_requests_lock:
        # vLLM wraps the request ID as "chatcmpl-{uuid}-{suffix}",
        # so strip the first and last segments to recover the original UUID.
        request_id = "-".join(chatcmpl_request_id.split("-")[1:])
        request_id = request_id[:16]
        event = pending_requests.get(request_id, None)
        if event:
            # Schedule the event.set() on the main event loop
            if main_event_loop is not None:
                main_event_loop.call_soon_threadsafe(event.set)
            return True
    return False


async def _handle_disagg_request(request: Request, endpoint: str):
    """
    Common handler for disaggregated prefill/decode requests.

    Works for both /v1/completions and /v1/chat/completions — the only
    difference is the *endpoint* path forwarded to the prefiller and decoder.
    """
    try:
        req_data = await request.json()

        # Generate a random request ID
        request_id = generate_request_id()
        logger.info(f"Received {endpoint} request with generated ID: {request_id}")

        # Pick prefill and decode clients
        prefill_client, decode_client = round_robin_pick_clients()

        # Create event for this request (signaled when KV store finishes)
        event = create_pending_request(request_id)

        try:
            # Modify request for prefiller: set max_tokens=1
            prefill_req_data = req_data.copy()
            prefill_req_data["max_tokens"] = 1
            if "max_completion_tokens" in prefill_req_data:
                prefill_req_data["max_completion_tokens"] = 1
            prefill_req_data["stream"] = False
            prefill_req_data.pop("stream_options", None)

            # Send to prefiller with X-Request-Id header (ignore output)
            prefill_send_time = time.monotonic()
            await send_request_to_prefiller(
                prefill_client.client,
                endpoint,
                prefill_req_data,
                request_id,
            )
            prefill_first_response_time = time.monotonic()
            prefill_duration = prefill_first_response_time - prefill_send_time
            logger.info(
                f"Request {request_id}: prefill request"
                f" duration = {prefill_duration:.4f}s"
            )

            # Wait for the event to be signaled (KV store finished)
            await event.wait()
            notify_time = time.monotonic()
            notify_wait_duration = notify_time - prefill_first_response_time
            logger.info(
                f"Request {request_id}: finished saving KV caches after prefill"
                f" response = {notify_wait_duration * 1000:.2f}ms"
            )
            logger.debug(f"Event signaled for {request_id}, forwarding to decoder")

        finally:
            # Clean up the pending request
            remove_pending_request(request_id)

        # Forward original request to decoder
        is_stream = req_data.get("stream", False)

        if is_stream:

            async def generate_stream():
                first_chunk = True
                async for chunk in stream_service_response(
                    decode_client.client, endpoint, req_data
                ):
                    if first_chunk:
                        decode_first_response_time = time.monotonic()
                        latency = (
                            decode_first_response_time - prefill_first_response_time
                        )
                        logger.info(
                            f"Request {request_id}: latency between prefill first "
                            f"response and decode first response = "
                            f"{latency * 1000:.2f}ms"
                        )
                        first_chunk = False
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="application/json")
        else:
            response = await send_request_to_decoder(
                decode_client.client, endpoint, req_data
            )
            return JSONResponse(content=response.json())

    except Exception as e:
        # Standard
        import sys
        import traceback

        exc_info = sys.exc_info()
        logger.error(f"Error in {endpoint} endpoint")
        logger.error(str(e))
        logger.error("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    """Handle /v1/completions requests."""
    return await _handle_disagg_request(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    """Handle /v1/chat/completions requests."""
    return await _handle_disagg_request(request, "/v1/chat/completions")


@app.get("/v1/models")
async def handle_models():
    """Handle /v1/models requests by forwarding to the first prefiller."""
    try:
        prefill_client = app.state.prefill_clients[0]
        headers = _build_headers()
        response = await prefill_client.client.get("/v1/models", headers=headers)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except Exception as e:
        logger.error(f"Error in /v1/models endpoint: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )


# ============================================================================
# Telemetry FastAPI app (runs on separate port)
# ============================================================================

telemetry_app = FastAPI()


@telemetry_app.post("/api/v1/telemetry")
async def handle_telemetry(request: Request):
    """
    Handle telemetry POST requests.

    Expected payload format (from FastAPIRequestTelemetry):
    {
        "event": "request_store_finished",
        "request_ids_set": ["id1", "id2", ...],
        "model_name": "...",
        "world_size": N,
        "kv_rank": K,
    }

    For each request ID in request_ids_set, signal the corresponding
    event if it exists.
    """
    try:
        payload = await request.json()

        event_type = payload.get("event")
        request_ids = payload.get("request_ids_set", [])

        for request_id in request_ids:
            logger.info(
                f"Received telemetry event: {event_type} for request: {request_id}"
            )

        notified_count = 0
        for request_id in request_ids:
            if notify_request(request_id):
                notified_count += 1
                logger.info(f"Notified request: {request_id}")

        return JSONResponse(
            content={
                "status": "ok",
                "notified": notified_count,
                "total": len(request_ids),
            }
        )

    except Exception as e:
        logger.error(f"Error processing telemetry: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500,
        )


def run_telemetry_server(host: str, port: int):
    """Run the telemetry server in a separate thread."""
    # Third Party
    import uvicorn

    uvicorn.run(telemetry_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    # Third Party
    import uvicorn

    # Start telemetry server in a background thread
    telemetry_thread = threading.Thread(
        target=run_telemetry_server,
        args=(global_args.host, global_args.telemetry_port),
        daemon=True,
    )
    telemetry_thread.start()
    logger.info(
        f"Telemetry server started on {global_args.host}:{global_args.telemetry_port}"
    )

    # Run main proxy server
    uvicorn.run(app, host=global_args.host, port=global_args.port)
