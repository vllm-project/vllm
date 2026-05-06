# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import copy
import logging
import os
import socket
import threading
import uuid

import aiohttp
import msgpack
import zmq
from quart import Quart, Request, make_response, request

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConstants,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
prefill_instances: list[dict] = []
decode_instances: list[dict] = []
request_nums = 0
app = Quart(__name__)


TRANSFER_TYPE = None


_list_lock = threading.RLock()


def _listen_for_register(hostname, port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    global prefill_instances
    global decode_instances

    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_addr, msg = router_socket.recv_multipart()
            data = msgpack.loads(msg)
            if data.get("type") == "HELLO":
                pass
            elif data.get("type") in ("P", "D"):
                role = data["type"]
                required_keys = {
                    "http_address",
                    "zmq_address",
                    "dp_size",
                    "tp_size",
                    "transfer_mode",
                }
                missing = required_keys - data.keys()
                if missing:
                    logger.error(
                        "Registration message missing required keys %s; skipping",
                        missing,
                    )
                    continue
                # Derive request_address from http_address
                # api path suffix is appended at request time
                instance = {
                    "role": role,
                    "request_address": f"http://{data['http_address']}/v1",
                    "http_address": data["http_address"],
                    "zmq_address": data["zmq_address"],
                    "dp_size": data["dp_size"],
                    "tp_size": data["tp_size"],
                    "transfer_mode": data["transfer_mode"],
                }
                # zmq_address format: "host:IP,handshake:PORT,notify:PORT"
                # Stored verbatim; embedded into the request_id by handle_request.

                global TRANSFER_TYPE
                transfer_mode = instance["transfer_mode"]
                target_list = prefill_instances if role == "P" else decode_instances
                with _list_lock:
                    if TRANSFER_TYPE is None:
                        TRANSFER_TYPE = transfer_mode
                        logger.info("SET TRANSFER TYPE TO %s", TRANSFER_TYPE)
                    elif transfer_mode != TRANSFER_TYPE:
                        logger.error(
                            "Mismatched transfer mode: expected %s, got %s;"
                            " skipping registration of %s",
                            TRANSFER_TYPE,
                            transfer_mode,
                            data["http_address"],
                        )
                        continue
                    existing_idx = next(
                        (
                            idx
                            for idx, i in enumerate(target_list)
                            if i.get("http_address") == data["http_address"]
                        ),
                        None,
                    )
                    if existing_idx is not None:
                        target_list[existing_idx] = instance
                        logger.info(
                            "Updated existing %s instance: %s",
                            "Prefill" if role == "P" else "Decode",
                            instance,
                        )
                    else:
                        target_list.append(instance)
                        logger.info(
                            "Registered %s instance: %s",
                            "Prefill" if role == "P" else "Decode",
                            instance,
                        )
            else:
                logger.warning(
                    "Received message with unrecognized type %r; ignoring",
                    data.get("type"),
                )


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=(hostname, port), daemon=True
    )
    _listener_thread.start()
    return _listener_thread


async def send_request_to_prefill(
    endpoint, req_data, request_id, selected_prefill_dp_rank
):
    req_data_copy = req_data

    req_data_copy["kv_transfer_params"].update(
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
        }
    )
    req_data_copy["stream"] = False
    req_data_copy["max_tokens"] = 1
    if "max_completion_tokens" in req_data_copy:
        req_data_copy["max_completion_tokens"] = 1
    if "stream_options" in req_data_copy:
        del req_data_copy["stream_options"]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    ) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        if selected_prefill_dp_rank is not None:
            headers["X-data-parallel-rank"] = str(selected_prefill_dp_rank)
        async with session.post(
            url=endpoint, json=req_data_copy, headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()

            else:
                error_message = (
                    f"send_request_to_prefill response ={response},"
                    f"reason={response.reason}, status={response.status},"
                    f"method={response.method}, url={response.url},"
                    f"real_url={response.real_url}"
                )
                raise RuntimeError(error_message)


async def start_decode_request(endpoint, req_data, request_id):
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    )
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    response = await session.post(url=endpoint, json=req_data, headers=headers)
    return session, response


async def stream_decode_response(session, response, request_id):
    try:
        if response.status == 200:
            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes
        else:
            error_message = (
                f"stream_decode_response response ={response},"
                f"reason={response.reason}, status={response.status},"
                f"method={response.method}, url={response.url},"
                f"real_url={response.real_url}"
            )
            raise RuntimeError(error_message)
    finally:
        await session.close()


def example_round_robin_dp_loader(request_number, dp_size):
    return request_nums % dp_size


@app.route("/v1/completions", methods=["POST"])
async def handle_completions_request():
    return await handle_request("/completions", request)


@app.route("/v1/chat/completions", methods=["POST"])
async def handle_chat_completions_request():
    return await handle_request("/chat/completions", request)


async def handle_request(api: str, request: Request):
    try:
        with _list_lock:
            global request_nums
            request_nums += 1

        req_data = await request.get_json()

        prefill_instance_endpoint = None
        decode_instance_endpoint = None
        error_msg = (
            "Service Unavailable: No prefill or decode instances are registered."
        )
        if not prefill_instances or not decode_instances:
            return await make_response(
                (
                    error_msg,
                    503,
                )
            )
        pid = request_nums % len(prefill_instances)
        did = request_nums % len(decode_instances)
        prefill_instance_endpoint = prefill_instances[pid]
        decode_instance_endpoint = decode_instances[did]

        selected_prefill_dp_rank = None
        if prefill_instance_endpoint["dp_size"] > 1:
            selected_prefill_dp_rank = example_round_robin_dp_loader(
                request_nums // len(prefill_instance_endpoint),
                prefill_instance_endpoint["dp_size"],
            )

        # Embed both zmq_addresses in the request_id so the connector can parse
        # the peer's host/ports from it, similar to P2P-NCCL
        uid = str(uuid.uuid4()).replace("-", "")
        request_id = (
            f"___prefill_addr_{prefill_instance_endpoint['zmq_address']}"
            f"___decode_addr_{decode_instance_endpoint['zmq_address']}"
            f"_{uid}"
        )

        transfer_id = f"{MoRIIOConstants.TRANSFER_PREFIX}-{str(uuid.uuid4())}"

        req_data_to_prefill = copy.deepcopy(req_data)
        req_data_to_prefill["kv_transfer_params"] = {}
        req_data["kv_transfer_params"] = {}
        req_data_to_prefill["kv_transfer_params"]["remote_dp_size"] = (
            decode_instance_endpoint["dp_size"]
        )
        req_data_to_prefill["kv_transfer_params"]["remote_tp_size"] = (
            decode_instance_endpoint["tp_size"]
        )
        req_data_to_prefill["kv_transfer_params"]["transfer_id"] = transfer_id

        prefill_request_url = prefill_instance_endpoint["request_address"] + api
        send_prefill_task = asyncio.create_task(
            send_request_to_prefill(
                prefill_request_url,
                req_data_to_prefill,
                request_id,
                selected_prefill_dp_rank,
            )
        )

        req_data["max_tokens"] -= 1

        req_data["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "transfer_id": transfer_id,
        }
        if TRANSFER_TYPE == "READ":
            # In read mode, prefill and decode are executed serially.
            prefill_response = await send_prefill_task
            prefill_kv = prefill_response["kv_transfer_params"]
            req_data["kv_transfer_params"]["remote_engine_id"] = prefill_kv[
                "remote_engine_id"
            ]
            req_data["kv_transfer_params"]["remote_block_ids"] = prefill_kv[
                "remote_block_ids"
            ]
            req_data["kv_transfer_params"]["transfer_id"] = prefill_kv["transfer_id"]

        req_data["kv_transfer_params"]["remote_dp_size"] = prefill_instance_endpoint[
            "dp_size"
        ]
        req_data["kv_transfer_params"]["remote_tp_size"] = prefill_instance_endpoint[
            "tp_size"
        ]

        if selected_prefill_dp_rank is not None:
            req_data["kv_transfer_params"]["remote_dp_rank"] = selected_prefill_dp_rank

        decode_request_url = decode_instance_endpoint["request_address"] + api
        decode_request_task = asyncio.create_task(
            start_decode_request(decode_request_url, req_data, request_id)
        )

        session, decode_response = await decode_request_task
        stream_generator = stream_decode_response(session, decode_response, request_id)
        response = await make_response(stream_generator)
        return response
    except Exception as e:
        logger.exception("An error occurred while handling the request: %s", e)
        return await make_response(
            (
                f"Internal Server Error: {e!s}",
                500,
            )
        )


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 36367)
    app.debug = True
    app.config["BODY_TIMEOUT"] = 360000
    app.config["RESPONSE_TIMEOUT"] = 360000

    app.run(host="0.0.0.0", port=10001)
    t.join()
