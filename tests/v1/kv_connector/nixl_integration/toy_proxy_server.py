# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import itertools
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
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

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
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
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


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
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    # Ask the prefiller to return the tokenized prompt and the sampled token
    # ids so the decoder can be fed pre-tokenized input and skip re-tokenizing
    # the (potentially large) prompt on its critical path.
    req_data["return_token_ids"] = True
    # These args are not supported for P
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

    # read/consume the response body to release the connection
    # otherwise, it would http.ReadError
    await response.aread()

    # Add back the min_tokens and min_completion_tokens so D can use them
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response


async def stream_service_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
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


async def _handle_single_prompt_completions(
    api: str,
    request: Request,
    req_data: dict,
    request_id: str,
) -> tuple[dict, dict]:
    """Run prefill + build the decode request for a single-prompt request.

    Args:
        api: Downstream endpoint path (e.g. ``/completions``).
        request: Incoming FastAPI request (used to pick clients).
        req_data: The (single-prompt) request payload.
        request_id: Unique id propagated to prefiller and decoder.

    Returns:
        A ``(prefill_response_json, decode_req_data)`` tuple. The caller
        forwards the prefiller's one-token output to the client and streams
        the decoder response built from ``decode_req_data``.
    """
    prefill_client_info = get_next_client(request.app, "prefill")

    response = await send_request_to_service(
        prefill_client_info, api, req_data, request_id
    )
    response_json = response.json()
    await response.aclose()  # CRITICAL: Release connection back to pool

    # Build the decode request inheriting all fields from the prefill request.
    decode_req_data = req_data.copy()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        decode_req_data["kv_transfer_params"] = kv_transfer_params

    choice = response_json["choices"][0]
    prompt_token_ids = choice.get("prompt_token_ids")
    output_token_ids = choice.get("token_ids")

    if prompt_token_ids is not None and output_token_ids is not None:
        # Fast path: feed the decoder pre-tokenized ids (full prompt plus the
        # one token already sampled by the prefiller). This avoids re-tokenizing
        # the whole prompt on the decoder, which otherwise dominates TTFT for
        # long prompts.
        decode_req_data["prompt"] = list(prompt_token_ids) + list(output_token_ids)
    elif "prompt" in decode_req_data and "text" in choice:
        # Fallback: append the one prefilled token as text so the decoder
        # continues from there (requires re-tokenization on the decoder).
        decode_req_data["prompt"] = decode_req_data["prompt"] + choice["text"]

    # The decoder does not need to echo token ids back.
    decode_req_data.pop("return_token_ids", None)

    # Prefill generated one token already; decrement the remaining budget.
    if "max_tokens" in decode_req_data:
        decode_req_data["max_tokens"] -= 1

    # Avoid forwarding the large token-id arrays back to the client; restore the
    # original (null) shape of the prefill response chunk.
    choice["prompt_token_ids"] = None
    choice["token_ids"] = None

    return response_json, decode_req_data


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        prompts = req_data.get("prompt")

        if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
            # Split into individual single-prompt requests so each gets its own
            # kv_transfer_params from the prefiller. A shared multi-prompt
            # request would give every sub-request the same remote_block_ids,
            # causing later sub-requests to fail the "block marked busy" assert
            # in start_load_kv after the first sub-request clears the flag.
            single_reqs = []
            for prompt in prompts:
                single_req = req_data.copy()
                single_req["prompt"] = prompt
                single_reqs.append(single_req)
            request_ids = [str(uuid.uuid4()) for _ in single_reqs]

            # Run all prefills concurrently.
            prefill_results = await asyncio.gather(
                *[
                    _handle_single_prompt_completions(api, request, r, rid)
                    for r, rid in zip(single_reqs, request_ids)
                ]
            )

            decode_client_info = get_next_client(request.app, "decode")

            async def generate_stream_multi():
                # Forward each prefiller's one-token output first.
                for (prefill_response_json, _), rid in zip(
                    prefill_results, request_ids
                ):
                    yield b"data: " + json.dumps(prefill_response_json).encode()

                async def _decode_one(dreq, rid):
                    chunks = []
                    async for chunk in stream_service_response(
                        decode_client_info, api, dreq, request_id=rid
                    ):
                        chunks.append(chunk)
                    return chunks

                decode_chunk_lists = await asyncio.gather(
                    *[
                        _decode_one(dreq, rid)
                        for (_, dreq), rid in zip(prefill_results, request_ids)
                    ]
                )
                for chunks in decode_chunk_lists:
                    for chunk in chunks:
                        yield chunk

            return StreamingResponse(
                generate_stream_multi(), media_type="application/json"
            )

        # Single-prompt fast path.
        request_id = str(uuid.uuid4())
        (
            prefill_response_json,
            decode_req_data,
        ) = await _handle_single_prompt_completions(
            api, request, req_data.copy(), request_id
        )

        decode_client_info = get_next_client(request.app, "decode")
        logger.debug("Using decode client %s", decode_client_info)

        async def generate_stream():
            # Emit the one-token prefill output first, then stream the decoder.
            yield b"data: " + json.dumps(prefill_response_json).encode()
            async for chunk in stream_service_response(
                decode_client_info, api, decode_req_data, request_id=request_id
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
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


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
