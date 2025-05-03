# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients
    prefiller_base_url = f'http://{global_args.prefiller_host}:{global_args.prefiller_port}/v1'
    decoder_base_url = f'http://{global_args.decoder_host}:{global_args.decoder_port}/v1'

    app.state.prefill_client = httpx.AsyncClient(timeout=None,
                                                 base_url=prefiller_base_url)
    app.state.decode_client = httpx.AsyncClient(timeout=None,
                                                base_url=decoder_base_url)

    yield

    # Shutdown: Close clients
    await app.state.prefill_client.aclose()
    await app.state.decode_client.aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


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


async def send_request_to_service(client: httpx.AsyncClient, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a persistent client.
    """
    req_data = req_data.copy()
    req_data['do_remote_decode'] = True
    req_data["stream"] = False
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client: httpx.AsyncClient, endpoint: str,
                                  req_data: dict, remote_block_ids: list[int],
                                  remote_engine_id: str, remote_host: str,
                                  remote_port: int, request_id: str):
    """
    Asynchronously stream the response from a service using a persistent client.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }
    req_data['do_remote_prefill'] = True
    req_data["remote_block_ids"] = remote_block_ids
    req_data['remote_engine_id'] = remote_engine_id
    req_data["remote_host"] = remote_host
    req_data["remote_port"] = remote_port

    async with client.stream("POST", endpoint, json=req_data,
                             headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


@app.post("/v1/completions")
async def handle_completions(request: Request):
    try:
        req_data = await request.json()

        request_id = str(uuid.uuid4())

        # Send request to prefill service
        response = await send_request_to_service(app.state.prefill_client,
                                                 "/completions", req_data,
                                                 request_id)

        # Extract the needed fields
        response_json = response.json()
        remote_block_ids = response_json.get('remote_block_ids', [])
        remote_engine_id = response_json.get('remote_engine_id', '')
        remote_host = response_json.get('remote_host', '')
        remote_port = response_json.get('remote_port', 0)

        # Stream response from decode service
        async def generate_stream():
            async for chunk in stream_service_response(
                    app.state.decode_client,
                    "/completions",
                    req_data,
                    remote_block_ids=remote_block_ids,
                    remote_engine_id=remote_engine_id,
                    remote_host=remote_host,
                    remote_port=remote_port,
                    request_id=request_id):
                yield chunk

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              " - completions endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
