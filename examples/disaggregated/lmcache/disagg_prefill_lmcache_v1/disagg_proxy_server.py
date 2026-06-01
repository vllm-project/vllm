# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    prefiller_base_url = (
        f"http://{global_args.prefiller_host}:{global_args.prefiller_port}/v1"
    )
    decoder_base_url = (
        f"http://{global_args.decoder_host}:{global_args.decoder_port}/v1"
    )

    app.state.prefill_client = httpx.AsyncClient(
        timeout=None,
        base_url=prefiller_base_url,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )
    app.state.decode_client = httpx.AsyncClient(
        timeout=None,
        base_url=decoder_base_url,
        limits=httpx.Limits(
            max_connections=None,
            max_keepalive_connections=None,
        ),
    )

    yield

    # Shutdown: Close clients
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


class StatsCalculator:
    def __init__(self):
        self._stats = []
        self._last_log_time = time.time()

    def add(self, value):
        self._stats.append(value)
        if time.time() - self._last_log_time > 5:
            self._log_stats()
            self._last_log_time = time.time()

    def _log_stats(self):
        # Print average, median, and 99th percentile
        np_arr = np.array(self._stats)
        output_str = (
            f"\nNum requests: {len(self._stats)}"
            "\nPrefill node TTFT stats:"
            f"\n - Average (ms): {np.mean(np_arr)}"
            f"\n - Median (ms): {np.median(np_arr)}"
            f"\n - 99th Percentile (ms): {np.percentile(np_arr, 99)}\n"
        )
        print(
            "===============================",
            output_str,
            "===============================",
        )


stats_calculator = StatsCalculator()
counter = 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-host", type=str, default="localhost")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", type=str, default="localhost")
    parser.add_argument("--decoder-port", type=int, default=8200)
    args = parser.parse_args()
    return args


# Initialize variables to hold the persistent clients
app.state.prefill_client = None
app.state.decode_client = None


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Send a request to a service using a persistent client.
    """
    req_data = req_data.copy()
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    return response


async def stream_service_response(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Asynchronously stream the response from a service using a persistent client.
    """
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1

    st = time.time()
    try:
        req_data = await request.json()

        # Send request to prefill service, ignore the response
        await send_request_to_service(
            app.state.prefill_client, "/completions", req_data
        )

        et = time.time()
        stats_calculator.add(et - st)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(
                app.state.decode_client, "/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    global counter, stats_calculator
    counter += 1

    st = time.time()
    try:
        req_data = await request.json()

        # Send request to prefill service, ignore the response
        await send_request_to_service(
            app.state.prefill_client, "/chat/completions", req_data
        )

        et = time.time()
        stats_calculator.add(et - st)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(
                app.state.decode_client, "/chat/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(
            "Error occurred in disagg prefill proxy server  - chat completions endpoint"
        )
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
