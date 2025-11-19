# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import msgspec
import numpy as np
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.pd_backend import (
    PDMsg,
)

logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize clients

    # Build prefill clients with CSV-based broadcast pairing
    pref_hosts = global_args.prefiller_host
    pref_ports = global_args.prefiller_port

    def pair_hosts_and_ports(hosts, ports, count=None):
        """
        Flexible host-port pairing with expansion strategies.

        Multiple pairing strategies:
        1. Single host + single port + count: Generate incremental ports on same host
        2. Single host + multiple ports: Pair the host with each port
        3. Multiple hosts + single port: Pair each host with the same port
        4. Multiple hosts + multiple ports: Strict one-to-one pairing
           (must have same length)
        """
        # Ensure lists
        if not isinstance(hosts, list):
            hosts = [hosts]
        if not isinstance(ports, list):
            ports = [ports]
        # Single host/port with count -> incremental ports
        if len(hosts) == 1 and len(ports) == 1:
            if count is None or count <= 1:
                return [(hosts[0], ports[0])]
            else:
                return [(hosts[0], ports[0] + i) for i in range(count)]
        # Expand single host to multiple ports
        if len(hosts) == 1:
            return [(hosts[0], p) for p in ports]
        # Expand single port to multiple hosts
        if len(ports) == 1:
            return [(h, ports[0]) for h in hosts]
        # Strict one-to-one pairing when both lists are provided
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
        app.state.prefill_clients.append(
            ClientInfo(
                prefill_client,
            )
        )

    # Build decoder clients with CSV-based broadcast pairing
    dec_hosts = global_args.decoder_host
    dec_ports = global_args.decoder_port

    decoder_pairs = pair_hosts_and_ports(dec_hosts, dec_ports, global_args.num_decoders)

    # Whether the ports increase per instances
    # (only when using single host/port with num_decoders > 1)
    incremental_mode = (
        len(dec_hosts) == 1 and len(dec_ports) == 1 and global_args.num_decoders > 1
    )

    for i, (host, port) in enumerate(decoder_pairs):
        decoder_base_url = f"http://{host}:{int(port)}"
        decode_client = httpx.AsyncClient(timeout=None, base_url=decoder_base_url)
        if incremental_mode:
            init_ports = [p + i for p in global_args.decoder_init_port]
            alloc_ports = [p + i for p in global_args.decoder_alloc_port]
        else:
            # Use the provided ports as-is
            # (suitable when different hosts can reuse same port numbers)
            init_ports = list(global_args.decoder_init_port)
            alloc_ports = list(global_args.decoder_alloc_port)

        app.state.decode_clients.append(
            ClientInfo(
                decode_client,
                host,
                init_ports,
                alloc_ports,
            )
        )

    app.state.total_clients = app.state.prefill_clients + app.state.decode_clients

    app.state.zmq_task = asyncio.create_task(zmq_pull_server())

    yield

    # Shutdown: Close clients
    for client in app.state.prefill_clients:
        await client.aclose()
    for client in app.state.decode_clients:
        await client.aclose()

    global run_proxy
    run_proxy = False
    await app.state.zmq_task  # Wait for background task to finish


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
        np_arr = np.array(self._stats) * 1000
        output_str = (
            f"\nNum requests: {len(self._stats)}"
            + "\nPrefill node TTFT stats:"
            + f"\n - Average (ms): {np.mean(np_arr)}"
            + f"\n - Median (ms): {np.median(np_arr)}"
            + f"\n - 99th Percentile (ms): {np.percentile(np_arr, 99)}\n"
        )
        print(
            "===============================",
            output_str,
            "===============================",
        )


stats_calculator = StatsCalculator()
counter = 0


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
    parser.add_argument("--decoder-init-port", type=csv_ints, default=[8300])
    parser.add_argument("--decoder-alloc-port", type=csv_ints, default=[8400])

    parser.add_argument("--num-decoders", type=int, default=1)
    parser.add_argument("--proxy-host", type=str, default="localhost")
    parser.add_argument("--proxy-port", type=int, default=8500)

    args = parser.parse_args()
    return args


@dataclass
class ClientInfo:
    client: httpx.AsyncClient
    host: str | None = None
    init_port: list[int] | None = None
    alloc_port: list[int] | None = None


# Initialize variables to hold the persistent clients
app.state.prefill_clients = []
app.state.decode_clients = []
app.state.total_clients = []

# Keep finished reqs
app.state.finished_reqs = defaultdict(int)


zmq_ctx = zmq.asyncio.Context()
run_proxy = True  # Shutdown flag


async def zmq_pull_server():
    socket = zmq_ctx.socket(zmq.PULL)
    proxy_url = f"{global_args.proxy_host}:{global_args.proxy_port}"
    socket.bind(f"tcp://{proxy_url}")
    logger.info("ZMQ proxy server started on %s", proxy_url)

    while run_proxy:
        try:
            msg_bytes = await socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=PDMsg)
            req_id = msg.req_id
            app.state.finished_reqs[req_id] += 1
            logger.debug("Prefill of req %s done.", req_id)
        except zmq.Again:
            await asyncio.sleep(0.01)  # Avoid busy loop
        except Exception as e:
            print("ZMQ Error:", e)
            break

    socket.close()
    logger.info("ZMQ PULL server stopped.")


async def send_request_to_service(
    client: httpx.AsyncClient, endpoint: str, req_data: dict
):
    """
    Send a request to a service using a persistent client.
    """

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
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


def round_robin_pick_client(clients, idx):
    return clients[idx % len(clients)]


async def wait_decode_kv_ready(req_id: str, num_tp_rank: int):
    while app.state.finished_reqs[req_id] < num_tp_rank:
        await asyncio.sleep(0.0001)  # sleep for 0.1 ms
    logger.debug("Prefill node signaled kv ready for req %s", req_id)
    app.state.finished_reqs.pop(req_id)


@app.post("/v1/completions")
async def handle_completions(request: Request):
    global counter, stats_calculator
    counter += 1
    req_id = str(counter)  # we use counter as req_id

    st = time.time()
    try:
        req_data = await request.json()

        tokenization_client = round_robin_pick_client(app.state.total_clients, counter)

        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"prompt": req_data["prompt"]}
        )
        tokenize_output = tokenize_output.json()

        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1

        # Pick decode client
        decode_client = round_robin_pick_client(app.state.decode_clients, counter)

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }
        num_tp_rank = len(decode_client.init_port)

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Send request to prefill service round robin, ignore the response
        prefill_client = round_robin_pick_client(app.state.prefill_clients, counter)
        prefill_output = await send_request_to_service(
            prefill_client.client, "/v1/completions", req_data
        )

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens - 1
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])
        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        # Stream response from decode service
        async def generate_stream():
            head_chunk = {
                "id": prefill_output["id"],
                "object": "text_completion",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "text": prefill_output["choices"][0]["text"],
                        "logprobs": None,
                        "finish_reason": None,
                        "stop_reason": None,
                    }
                ],
                "usage": None,
            }
            yield (
                "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            # Wait until decode node signals that kv is ready
            await wait_decode_kv_ready(req_id, num_tp_rank)

            async for chunk in stream_service_response(
                decode_client.client, "/v1/completions", req_data
            ):
                yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        # Standard
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
    req_id = str(counter)

    st = time.time()
    try:
        req_data = await request.json()

        tokenization_client = round_robin_pick_client(app.state.total_clients, counter)

        # For chat completions, we need to tokenize the messages
        tokenize_output = await send_request_to_service(
            tokenization_client.client, "/tokenize", {"messages": req_data["messages"]}
        )
        tokenize_output = tokenize_output.json()

        org_max_tokens = req_data["max_tokens"]
        req_data["prompt"] = tokenize_output["tokens"]
        req_data["max_tokens"] = 1

        org_max_completion_tokens = None
        if "max_completion_tokens" in req_data:
            org_max_completion_tokens = req_data["max_completion_tokens"]
            req_data["max_completion_tokens"] = 1

        # Pick decode client
        decode_client = round_robin_pick_client(app.state.decode_clients, counter)

        disagg_spec = {
            "req_id": req_id,
            "receiver_host": decode_client.host,
            "receiver_init_port": decode_client.init_port,
            "receiver_alloc_port": decode_client.alloc_port,
        }

        num_tp_rank = len(decode_client.init_port)

        req_data["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": disagg_spec,
        }

        req_data["stream"] = False
        stream_options = req_data.pop("stream_options", None)

        # Send request to prefill service round robin, get the response
        prefill_client = round_robin_pick_client(app.state.prefill_clients, counter)
        prefill_output = await send_request_to_service(
            prefill_client.client, "/v1/completions", req_data
        )

        prefill_output = prefill_output.json()

        et = time.time()
        stats_calculator.add(et - st)

        req_data["max_tokens"] = org_max_tokens - 1
        if org_max_completion_tokens is not None:
            req_data["max_completion_tokens"] = org_max_completion_tokens - 1

        # Add the first token from prefill to the tokenized messages for decode
        req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])

        req_data.pop("kv_transfer_params")
        req_data["stream"] = True
        if stream_options is not None:
            req_data["stream_options"] = stream_options

        # Stream response from decode service
        async def generate_stream():
            initial_chunk = {
                "id": prefill_output["id"],
                "object": "chat.completion.chunk",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield (
                "data: " + json.dumps(initial_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            head_chunk = {
                "id": prefill_output["id"],
                "object": "chat.completion.chunk",
                "created": prefill_output["created"],
                "model": prefill_output["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": prefill_output["choices"][0]["text"]},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            yield (
                "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
            ).encode()

            await wait_decode_kv_ready(req_id, num_tp_rank)

            # Stream and convert completion format chunks to chat completion format
            async for chunk in stream_service_response(
                decode_client.client, "/v1/completions", req_data
            ):
                chunk_str = chunk.decode("utf-8")
                if chunk_str.startswith("data: ") and not chunk_str.startswith(
                    "data: [DONE]"
                ):
                    try:
                        json_str = chunk_str[6:].strip()  # Remove 'data: ' prefix
                        if json_str:
                            completion_data = json.loads(json_str)
                            chat_completion_data = {
                                "id": completion_data["id"],
                                "object": "chat.completion.chunk",
                                "created": completion_data["created"],
                                "model": completion_data["model"],
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": completion_data["choices"][0][
                                                "text"
                                            ]
                                        },
                                        "logprobs": completion_data["choices"][0].get(
                                            "logprobs"
                                        ),
                                        "finish_reason": completion_data["choices"][
                                            0
                                        ].get("finish_reason"),
                                    }
                                ],
                            }
                            converted_chunk = (
                                "data: "
                                + json.dumps(
                                    chat_completion_data, separators=(",", ":")
                                )
                                + "\n\n"
                            ).encode()
                            yield converted_chunk
                    except (json.JSONDecodeError, KeyError):
                        yield chunk
                else:
                    yield chunk

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        # Standard
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

    # Third Party
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
