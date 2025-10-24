# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
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
                "client": httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
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
                "client": httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
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


async def send_request_to_prefill(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send request to prefill service. Preserves echo and logprobs settings.
    Prefill should only generate 1 token to establish KV cache.
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
    # Prefill should only generate 1 token
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

    return response


async def send_request_to_decode(
    client_info: dict, endpoint: str, req_data: dict, request_id: str, kv_transfer_params: dict
):
    """
    Send request to decode service with KV transfer params from prefill.
    Decode gets the full max_tokens from the original request.
    """
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = kv_transfer_params
    req_data["stream"] = False
    # Keep max_tokens from original request (decode generates the full completion)
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

    return response


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """
    Send a request to a service using a client from the pool.
    Legacy function for compatibility.
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
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=headers
    )
    response.raise_for_status()

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


def merge_prompt_logprobs(prefill_json: dict, decode_json: dict) -> dict:
    """
    Merge prompt_logprobs from prefill response into decode response.
    Handles both Completions API (prompt_logprobs in choices) and
    Chat Completions API (prompt_logprobs at top level).

    For Completions API with echo=true and logprobs, we need to merge:
    1. choices[].prompt_logprobs - top-level per-choice field
    2. choices[].logprobs.token_logprobs - flattened array of all logprobs
    3. choices[].logprobs.tokens - token strings
    4. choices[].logprobs.text_offset - text offsets
    5. choices[].logprobs.top_logprobs - alternative tokens with logprobs
    """
    # For Completions API: prompt_logprobs is inside each choice
    if "choices" in decode_json and "choices" in prefill_json:
        decode_choices = decode_json.get("choices", [])
        prefill_choices = prefill_json.get("choices", [])

        for decode_choice, prefill_choice in zip(decode_choices, prefill_choices):
            # 1. Merge top-level prompt_logprobs field
            if "prompt_logprobs" in prefill_choice:
                logger.debug(f"[LOGPROBS MERGE] Merging prompt_logprobs from prefill choice: {len(prefill_choice['prompt_logprobs'])} items")
                decode_choice["prompt_logprobs"] = prefill_choice["prompt_logprobs"]
            else:
                logger.debug("[LOGPROBS MERGE] No prompt_logprobs found in prefill choice")

            # 2. Merge logprobs object (token_logprobs, tokens, text_offset, top_logprobs)
            if "logprobs" in prefill_choice and "logprobs" in decode_choice:
                prefill_logprobs = prefill_choice["logprobs"]
                decode_logprobs = decode_choice["logprobs"]

                # Determine how many prompt tokens there are from prompt_logprobs
                # Prefill generates with max_tokens=1, so it has [prompt_tokens] + [1 output token]
                # We only want the prompt tokens, not prefill's output token
                num_prompt_tokens = len(prefill_choice.get("prompt_logprobs", []))

                # Merge token_logprobs: [prefill_PROMPT_logprobs_only] + [decode_ALL_logprobs]
                if "token_logprobs" in prefill_logprobs and "token_logprobs" in decode_logprobs:
                    prefill_token_logprobs = prefill_logprobs["token_logprobs"]
                    decode_token_logprobs = decode_logprobs["token_logprobs"]

                    # Extract only prompt logprobs from prefill (exclude the 1 output token)
                    prefill_prompt_only = prefill_token_logprobs[:num_prompt_tokens]

                    merged_token_logprobs = prefill_prompt_only + decode_token_logprobs
                    decode_logprobs["token_logprobs"] = merged_token_logprobs
                    logger.debug(f"[LOGPROBS MERGE] Merged token_logprobs: {len(prefill_prompt_only)} prompt (from prefill) + {len(decode_token_logprobs)} all (from decode) = {len(merged_token_logprobs)} total")

                # Merge tokens: [prefill_PROMPT_tokens_only] + [decode_ALL_tokens]
                if "tokens" in prefill_logprobs and "tokens" in decode_logprobs:
                    prefill_tokens = prefill_logprobs["tokens"]
                    decode_tokens = decode_logprobs["tokens"]

                    # Extract only prompt tokens from prefill (exclude the 1 output token)
                    prefill_prompt_tokens_only = prefill_tokens[:num_prompt_tokens]

                    decode_logprobs["tokens"] = prefill_prompt_tokens_only + decode_tokens
                    logger.debug(f"[LOGPROBS MERGE] Merged tokens: {len(prefill_prompt_tokens_only)} prompt + {len(decode_tokens)} all")

                # Merge text_offset: [prefill_PROMPT_offsets_only] + [decode_ALL_offsets_adjusted]
                if "text_offset" in prefill_logprobs and "text_offset" in decode_logprobs:
                    prefill_offsets = prefill_logprobs["text_offset"]
                    decode_offsets = decode_logprobs["text_offset"]

                    # Extract only prompt offsets from prefill (exclude the 1 output token)
                    prefill_prompt_offsets_only = prefill_offsets[:num_prompt_tokens]

                    # Decode offsets need to be adjusted by the last prefill prompt offset
                    if prefill_prompt_offsets_only:
                        last_prefill_offset = prefill_prompt_offsets_only[-1]
                        # Get the length of the last prefill prompt token to compute the base offset
                        if "tokens" in prefill_logprobs and len(prefill_logprobs["tokens"]) >= num_prompt_tokens:
                            last_token_len = len(prefill_logprobs["tokens"][num_prompt_tokens - 1])
                            base_offset = last_prefill_offset + last_token_len
                        else:
                            base_offset = last_prefill_offset
                        # Adjust decode offsets by adding base_offset
                        adjusted_decode_offsets = [offset + base_offset for offset in decode_offsets]
                        decode_logprobs["text_offset"] = prefill_prompt_offsets_only + adjusted_decode_offsets
                        logger.debug(f"[LOGPROBS MERGE] Merged text_offset: {len(prefill_prompt_offsets_only)} prompt + {len(decode_offsets)} all (adjusted by {base_offset})")
                    else:
                        decode_logprobs["text_offset"] = prefill_prompt_offsets_only + decode_offsets

                # Merge top_logprobs: [prefill_PROMPT_top_logprobs_only] + [decode_ALL_top_logprobs]
                if "top_logprobs" in prefill_logprobs and "top_logprobs" in decode_logprobs:
                    prefill_top_logprobs = prefill_logprobs["top_logprobs"]
                    decode_top_logprobs = decode_logprobs["top_logprobs"]

                    # Extract only prompt top_logprobs from prefill (exclude the 1 output token)
                    prefill_prompt_top_only = prefill_top_logprobs[:num_prompt_tokens]

                    decode_logprobs["top_logprobs"] = prefill_prompt_top_only + decode_top_logprobs
                    logger.debug(f"[LOGPROBS MERGE] Merged top_logprobs: {len(prefill_prompt_top_only)} prompt + {len(decode_top_logprobs)} all")

    # For Chat Completions API: prompt_logprobs at top level
    if "prompt_logprobs" in prefill_json:
        logger.debug(f"[LOGPROBS MERGE] Merging top-level prompt_logprobs: {len(prefill_json['prompt_logprobs'])} items")
        decode_json["prompt_logprobs"] = prefill_json["prompt_logprobs"]

    return decode_json


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Check if logprobs are requested
        needs_logprobs = req_data.get("logprobs") is not None or req_data.get("echo", False)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")

        # Send request to prefill service (preserves echo and logprobs)
        prefill_response = await send_request_to_prefill(
            prefill_client_info, api, req_data, request_id
        )

        # Extract the needed fields
        prefill_json = prefill_response.json()
        kv_transfer_params = prefill_json.get("kv_transfer_params", {})
        if not kv_transfer_params:
            logger.error("No kv_transfer_params in prefill response!")
            raise ValueError("Prefill response missing kv_transfer_params")

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using prefill=%s decode=%s", prefill_client_info, decode_client_info)

        # For non-streaming with logprobs, we need to merge responses
        if needs_logprobs and not req_data.get("stream", False):
            decode_response = await send_request_to_decode(
                decode_client_info, api, req_data, request_id, kv_transfer_params
            )
            decode_json = decode_response.json()

            # Merge prompt_logprobs from prefill into decode response
            merged_json = merge_prompt_logprobs(prefill_json, decode_json)

            import json
            return StreamingResponse(
                iter([json.dumps(merged_json).encode()]),
                media_type="application/json"
            )

        # Stream response from decode service (original behavior for streaming or no logprobs)
        req_data["kv_transfer_params"] = kv_transfer_params
        async def generate_stream():
            async for chunk in stream_service_response(
                decode_client_info, api, req_data, request_id=request_id
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
