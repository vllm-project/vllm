# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import itertools
import json
import logging
import os
import uuid
from contextlib import aclosing, asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
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

    parser.add_argument(
        "--super-prefill",
        action="store_true",
        help="On /v1/completions, keep generating on the prefiller while the "
        "decoder pulls the KV cache, then let the decoder continue the answer. "
        "Needs prefix caching on both instances. Off by default.",
    )
    parser.add_argument(
        "--super-prefill-max-tokens",
        type=int,
        default=0,
        help="With --super-prefill, the most tokens the prefiller generates "
        "before it hands the request to the decoder (0: no limit).",
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
    """Get the next client in round-robin fashion.

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


def _headers(request_id: str) -> dict:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


async def send_request_to_service(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """Send a request to a service using a client from the pool."""
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
    # These args are not supported for P
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)

    response = await client_info["client"].post(
        endpoint, json=req_data, headers=_headers(request_id)
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
    """Asynchronously stream response from a service using a client from the pool."""
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=_headers(request_id)
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


# Requests with these fields stay on the serial path: their state (logprobs,
# stop strings, penalties, grammars) does not carry across the switch to D.
_SUPER_PREFILL_UNSUPPORTED = (
    "echo",
    "suffix",
    "stop",
    "min_tokens",
    "use_beam_search",
    "response_format",
    "structured_outputs",
    "prompt_embeds",
    "return_token_ids",
    "frequency_penalty",
    "presence_penalty",
)


def _super_prefill_eligible(req_data: dict) -> bool:
    prompt = req_data.get("prompt")
    single_prompt = isinstance(prompt, str) or (
        isinstance(prompt, list)
        and len(prompt) > 0
        and all(isinstance(t, int) for t in prompt)
    )
    max_tokens = req_data.get("max_tokens", 16)
    return (
        single_prompt
        and isinstance(max_tokens, int)
        and max_tokens > 0
        and req_data.get("n") in (None, 1)
        and req_data.get("logprobs") is None
        and req_data.get("prompt_logprobs") is None
        and req_data.get("repetition_penalty") in (None, 1, 1.0)
        and all(
            req_data.get(key) in (None, False, 0, "", [], {})
            for key in _SUPER_PREFILL_UNSUPPORTED
        )
    )


async def _stream_choices(client_info: dict, body: dict, request_id: str):
    """Yield the first choice of every chunk of a streamed completion."""
    async with client_info["client"].stream(
        "POST", "/completions", json=body, headers=_headers(request_id)
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            data = line.removeprefix("data:").strip()
            if line.startswith("data:") and data != "[DONE]":
                for choice in json.loads(data)["choices"]:
                    yield choice


_pending_pulls: set[asyncio.Task] = set()


def _pull_done(task: asyncio.Task) -> None:
    _pending_pulls.discard(task)
    if task.cancelled():
        return
    error = task.exception() or (task.result().is_error and task.result().status_code)
    if error:
        # D then recomputes the prompt itself: slower, still correct.
        logger.warning("Super Prefill KV pull failed: %s", error)


async def _super_prefill(app, req_data: dict, request_id: str, meta: dict):
    """Yield (text, num_tokens, finish_reason) of a Super Prefill completion.

    1. P runs the prompt (max_tokens=1), as on the serial path, and also
       returns the prompt token ids and the first token id.
    2. P keeps generating from those ids, streaming to the client, while
    3. D pulls the prompt's KV cache with a max_tokens=1 request.
    4. Once the pull returns, P's stream is closed and D continues from the
       prompt ids plus the tokens P produced whose text was already sent.
    Step 2 reuses step 1's blocks on P and step 4 reuses step 3's blocks on D
    through prefix caching; D recomputes only the tokens P produced.
    """
    max_tokens = req_data.get("max_tokens", 16)
    cap = global_args.super_prefill_max_tokens
    prefill_budget = min(max_tokens, cap) if cap > 0 else max_tokens
    common = {
        k: v
        for k, v in req_data.items()
        if k not in ("prompt", "max_tokens", "stream", "stream_options")
        and k not in ("kv_transfer_params", "truncate_prompt_tokens")
    }
    prefill_client = get_next_client(app, "prefill")
    decode_client = get_next_client(app, "decode")

    response = await send_request_to_service(
        prefill_client,
        "/completions",
        {**req_data, "return_token_ids": True},
        request_id,
    )
    response_json = response.json()
    await response.aclose()
    first = response_json["choices"][0]
    prompt_ids = first["prompt_token_ids"]
    meta.update(
        id=response_json["id"],
        created=response_json["created"],
        model=response_json["model"],
        prompt_tokens=len(prompt_ids),
    )
    # `sent` counts the tokens whose text reached the client. A token that ends
    # inside a multi-byte character has its text held back by the detokenizer,
    # so D regenerates it rather than continuing after it.
    generated, sent = list(first["token_ids"]), 0
    if first["finish_reason"] == "stop" or len(generated) >= max_tokens:
        yield first["text"], len(generated), first["finish_reason"]
        return
    if first["text"]:
        yield first["text"], len(generated), None
        sent = len(generated)

    pull = asyncio.create_task(
        decode_client["client"].post(
            "/completions",
            json={
                **common,
                "prompt": prompt_ids,
                "max_tokens": 1,
                "stream": False,
                "kv_transfer_params": response_json.get("kv_transfer_params") or {},
            },
            headers=_headers(request_id),
        )
    )
    _pending_pulls.add(pull)
    pull.add_done_callback(_pull_done)

    final = False
    if prefill_budget > len(generated):
        body = {
            **common,
            "prompt": prompt_ids + generated,
            "max_tokens": prefill_budget - len(generated),
            "stream": True,
            "return_token_ids": True,
        }
        async with aclosing(
            _stream_choices(prefill_client, body, f"{request_id}-p")
        ) as choices:
            while not pull.done():
                next_choice = asyncio.ensure_future(anext(choices))
                await asyncio.wait(
                    {next_choice, pull}, return_when=asyncio.FIRST_COMPLETED
                )
                if not next_choice.done():
                    next_choice.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await next_choice
                    break
                try:
                    choice = next_choice.result()
                except StopAsyncIteration:
                    break
                generated += choice.get("token_ids") or []
                finish = choice["finish_reason"]
                final = finish is not None and (
                    finish != "length" or len(generated) >= max_tokens
                )
                if choice["text"] or final:
                    yield (
                        choice["text"],
                        len(generated) - sent,
                        finish if final else None,
                    )
                    sent = len(generated)
                if finish is not None:
                    break
        if final:
            return

    with suppress(httpx.HTTPError):
        await pull
    logger.debug("Super Prefill %s: %d tokens on P", request_id, sent)
    body = {
        **common,
        "prompt": prompt_ids + generated[:sent],
        "max_tokens": max_tokens - sent,
        "stream": True,
        "return_token_ids": True,
    }
    pending = 0
    async with aclosing(
        _stream_choices(decode_client, body, f"{request_id}-d")
    ) as choices:
        async for choice in choices:
            pending += len(choice.get("token_ids") or [])
            if choice["text"] or choice["finish_reason"]:
                yield choice["text"], pending, choice["finish_reason"]
                pending = 0


def _completion(meta: dict, choices: list[dict], num_tokens: int | None) -> dict:
    out = {
        "id": meta["id"],
        "object": "text_completion",
        "created": meta["created"],
        "model": meta["model"],
        "choices": choices,
    }
    if num_tokens is not None:
        out["usage"] = {
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": num_tokens,
            "total_tokens": meta["prompt_tokens"] + num_tokens,
        }
    return out


async def _super_prefill_response(app, req_data: dict, request_id: str):
    meta: dict = {}
    pieces = _super_prefill(app, req_data, request_id, meta)

    if not req_data.get("stream"):
        texts, num_tokens, finish = [], 0, None
        async with aclosing(pieces):
            async for text, n, finish in pieces:
                texts.append(text)
                num_tokens += n
        choice = {
            "index": 0,
            "text": "".join(texts),
            "logprobs": None,
            "finish_reason": finish,
        }
        return JSONResponse(_completion(meta, [choice], num_tokens))

    async def generate_sse():
        num_tokens = 0
        async with aclosing(pieces):
            async for text, n, finish in pieces:
                num_tokens += n
                choice = {
                    "index": 0,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish,
                }
                yield f"data: {json.dumps(_completion(meta, [choice], None))}\n\n"
        if (req_data.get("stream_options") or {}).get("include_usage"):
            yield f"data: {json.dumps(_completion(meta, [], num_tokens))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_sse(), media_type="text/event-stream")


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        if (
            global_args.super_prefill
            and api == "/completions"
            and _super_prefill_eligible(req_data)
        ):
            return await _super_prefill_response(request.app, req_data, request_id)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, "prefill")

        # Send request to prefill service
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )

        # Extract the needed fields
        response_json = response.json()
        await response.aclose()  # CRITICAL: Release connection back to pool
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, "decode")

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
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
