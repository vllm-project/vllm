# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PConnector proxy server for OffloadingConnector + TieringOffloadingSpec.

Unlike NixlConnector (which returns remote_host/remote_port in the prefill
response), OffloadingConnector does not embed connector coordinates in its
response. This proxy injects the prefiller's P2PConnector address into
kv_transfer_params before forwarding the decode request so the decoder knows
where to pull KV blocks from.

Usage:
    .venv/bin/python p2p_connector_proxy.py \
        --port 8192 \
        --prefiller-host 127.0.0.1 --prefiller-port 8100 \
        --decoder-host   127.0.0.1 --decoder-port  8200 \
        --p2p-connector-host 127.0.0.1 --p2p-connector-port 5710
"""

import argparse
import asyncio
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
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    mode = "decoder-first" if global_args.decoder_first else "prefiller-first"
    pd_host = global_args.p2p_connector_host
    pd_port = global_args.p2p_connector_port
    print(
        f"Proxy ready [{mode}]: "
        f"{len(app.state.prefill_clients)} prefiller(s), "
        f"{len(app.state.decode_clients)} decoder(s). "
        f"P2PConnector at {pd_host}:{pd_port}"
    )
    yield

    for ci in app.state.prefill_clients:
        await ci["client"].aclose()
    for ci in app.state.decode_clients:
        await ci["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8192)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--prefiller-hosts", type=str, nargs="+", default=["127.0.0.1"])
    p.add_argument("--prefiller-ports", type=int, nargs="+", default=[8100])
    p.add_argument("--decoder-hosts", type=str, nargs="+", default=["127.0.0.1"])
    p.add_argument("--decoder-ports", type=int, nargs="+", default=[8200])
    # P2PConnector coordinates of the prefiller — injected into decode requests.
    p.add_argument(
        "--p2p-connector-host",
        type=str,
        default="127.0.0.1",
        help="Host of the prefiller's P2PConnector ZMQ socket",
    )
    p.add_argument(
        "--p2p-connector-port",
        type=int,
        default=int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710")),
        help="Port of the prefiller's P2PConnector ZMQ socket "
        "(default: $VLLM_P2P_SIDE_CHANNEL_PORT or 5710)",
    )
    # P2PConnector coordinates of the decoder — injected into prefill requests
    # so the prefiller's submit_store can resolve the peer to push KV to.
    p.add_argument(
        "--decoder-p2p-connector-host",
        type=str,
        default="127.0.0.1",
        help="Host of the decoder's P2PConnector ZMQ socket",
    )
    p.add_argument(
        "--decoder-p2p-connector-port",
        type=int,
        default=int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710")) + 1,
        help="Port of the decoder's P2PConnector ZMQ socket "
        "(default: $VLLM_P2P_SIDE_CHANNEL_PORT + 1 or 5711)",
    )
    p.add_argument(
        "--decoder-first",
        action="store_true",
        help="Send decode request before prefill so decoder is already "
        "waiting when KV blocks arrive (decoder-first mode)",
    )
    args = p.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Prefiller host/port count mismatch")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Decoder host/port count mismatch")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


def _get_next(app, service: str):
    if service == "prefill":
        return app.state.prefill_clients[next(app.state.prefill_iterator)]
    return app.state.decode_clients[next(app.state.decode_iterator)]


def _auth_headers(request_id: str) -> dict:
    headers: dict = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _prefill(client_info, endpoint, req_data, request_id):
    """Send a prefill-only request (max_tokens=1) to the prefiller."""
    data = req_data.copy()
    data["kv_transfer_params"] = {
        "decode": {
            "kv_request_id": request_id,
        },
    }
    data["stream"] = False
    data["max_tokens"] = 1
    data.pop("max_completion_tokens", None)
    data.pop("stream_options", None)
    data.pop("min_tokens", None)
    data.pop("min_completion_tokens", None)

    headers = _auth_headers(request_id)
    resp = await client_info["client"].post(endpoint, json=data, headers=headers)
    resp.raise_for_status()
    await resp.aread()
    return resp


async def _stream_decode(client_info, endpoint, req_data, request_id):
    headers = _auth_headers(request_id)
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        prefill_client = _get_next(request.app, "prefill")
        await _prefill(prefill_client, api, req_data, request_id)

        # Inject the prefiller's P2PConnector address so the decoder can pull
        # KV blocks from it via the P2PConnector transport.
        req_data["kv_transfer_params"] = {
            "prefill": {
                "kv_request_id": request_id,
                "remote_host": global_args.p2p_connector_host,
                "remote_port": global_args.p2p_connector_port,
            },
        }

        decode_client = _get_next(request.app, "decode")
        logger.debug("prefill=%s decode=%s", prefill_client, decode_client)

        async def generate():
            async for chunk in _stream_decode(decode_client, api, req_data, request_id):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


async def _handle_completions_decoder_first(api: str, request: Request):
    """Decoder-first mode: send decode request before prefill.

    The decoder establishes its request and starts polling for KV blocks
    immediately. The prefill is then sent so the prefiller computes and
    pushes blocks to the already-waiting decoder.
    """
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        prefill_client = _get_next(request.app, "prefill")
        decode_client = _get_next(request.app, "decode")

        decode_data = req_data.copy()
        decode_data["kv_transfer_params"] = {
            "prefill": {
                "kv_request_id": request_id,
                "remote_host": global_args.p2p_connector_host,
                "remote_port": global_args.p2p_connector_port,
            },
        }

        async def generate():
            queue: asyncio.Queue = asyncio.Queue()

            async def _run_decode():
                try:
                    async for chunk in _stream_decode(
                        decode_client, api, decode_data, request_id
                    ):
                        await queue.put(("data", chunk))
                except Exception as exc:
                    await queue.put(("error", exc))
                finally:
                    await queue.put(("done", None))

            # 1. Start decode request — decoder is now waiting for KV blocks
            asyncio.create_task(_run_decode())

            # 2. Send prefill — blocks are computed and pushed to the decoder
            try:
                await _prefill(prefill_client, api, req_data, request_id)
            except Exception as exc:
                logger.warning("decoder-first: prefill failed: %s", exc)

            logger.debug(
                "decoder-first: prefill done, streaming decode prefill=%s decode=%s",
                prefill_client,
                decode_client,
            )

            # 3. Stream the decode response
            while True:
                kind, value = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise value  # type: ignore[misc]
                yield value

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import sys
        import traceback

        print(f"Proxy error on {api}: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


def _route_handler(api: str):
    if global_args.decoder_first:
        return lambda req: _handle_completions_decoder_first(api, req)
    return lambda req: _handle_completions(api, req)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _route_handler("/completions")(request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _route_handler("/chat/completions")(request)


@app.get("/healthcheck")
async def healthcheck():
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
