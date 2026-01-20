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
import regex as re
import zmq
from quart import Quart, make_response, request

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
prefill_instances: list[dict] = []
decode_instances: list[dict] = []
request_nums = 0
app = Quart(__name__)

IP_PORT_PATTERN = re.compile(r"//(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)")


TRANSFER_TYPE = None


def _append_whole_dict_unique(target_list, data_dict):
    new_filtered = {k: v for k, v in data_dict.items() if k != "index"}
    for existed in target_list:
        existed_filtered = {k: v for k, v in existed.items() if k != "index"}
        if existed_filtered == new_filtered:
            return False
    print("!!APPEND!!", data_dict)
    target_list.append(data_dict)
    transfer_mode = data_dict.get("transfer_mode", "unknown")
    global TRANSFER_TYPE

    if TRANSFER_TYPE is None:
        TRANSFER_TYPE = transfer_mode
        logger.info("SET TRANSFER TYPE TO %s", TRANSFER_TYPE)
    elif transfer_mode != TRANSFER_TYPE:
        raise ValueError(f"mismatched transfer mode {TRANSFER_TYPE} vs {transfer_mode}")

    return True


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
            if data["type"] == "HELLO":
                pass
            elif (
                data["type"] == "register"
                and data["role"] == "P"
                and data["request_address"] not in prefill_instances
            ):
                with _list_lock:
                    _append_whole_dict_unique(prefill_instances, data)

            elif (
                data["type"] == "register"
                and data["role"] == "D"
                and data["request_address"] not in decode_instances
            ):
                with _list_lock:
                    _append_whole_dict_unique(decode_instances, data)


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
    endpoint, req_data, request_id, d_endpoint, dip, dport, selected_prefill_dp_rank
):
    req_data_copy = req_data

    req_data_copy["kv_transfer_params"].update(
        {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_handshake_port": d_endpoint["handshake_port"],
            "remote_notify_port": d_endpoint["notify_port"],
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": dip,
            "remote_port": dport,
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
                raise RuntimeError(
                    "send_request_to_prefill response.status != 200response.status = ",
                    response.status,
                )


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
            raise RuntimeError(
                f"decode response.status != 200, status = {response.status}"
            )
    finally:
        await session.close()


async def send_request_to_decode(endpoint, req_data, request_id):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=6 * 6000 * 6000)
    ) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(
            url=endpoint, json=req_data, headers=headers
        ) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes
            else:
                raise RuntimeError(
                    "send_request_to_decode response.status != 200,response.statuus = ",
                    response.status,
                )


def example_round_robin_dp_loader(request_number, dp_size):
    return request_nums % dp_size


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        with _list_lock:
            global request_nums
            request_nums += 1

        def extract_ip_port_fast(url):
            match = IP_PORT_PATTERN.search(url)
            if not match:
                raise ValueError(f"Invalid URL format: {url}")
            return match.groups()

        req_data = await request.get_json()
        request_id = str(uuid.uuid4())

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

        dip, dport = extract_ip_port_fast(decode_instance_endpoint["request_address"])
        ip, port = extract_ip_port_fast(prefill_instance_endpoint["request_address"])

        req_data_to_prefill = copy.deepcopy(req_data)
        req_data_to_prefill["kv_transfer_params"] = {}
        req_data["kv_transfer_params"] = {}
        req_data_to_prefill["kv_transfer_params"]["remote_dp_size"] = (
            decode_instance_endpoint["dp_size"]
        )
        req_data_to_prefill["kv_transfer_params"]["remote_tp_size"] = (
            decode_instance_endpoint["tp_size"]
        )

        send_prefill_task = asyncio.create_task(
            send_request_to_prefill(
                prefill_instance_endpoint["request_address"],
                req_data_to_prefill,
                request_id,
                decode_instance_endpoint,
                dip,
                dport,
                selected_prefill_dp_rank,
            )
        )
        ip, port = extract_ip_port_fast(prefill_instance_endpoint["request_address"])

        req_data["max_tokens"] -= 1

        req_data["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_handshake_port": prefill_instance_endpoint["handshake_port"],
            "remote_notify_port": prefill_instance_endpoint["notify_port"],
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": ip,
            "remote_port": port,
        }
        if TRANSFER_TYPE == "READ":
            # In read mode, prefill and decode are executed serially.
            prefill_response = await send_prefill_task
            req_data["kv_transfer_params"]["remote_engine_id"] = prefill_response[
                "kv_transfer_params"
            ]["remote_engine_id"]
            req_data["kv_transfer_params"]["remote_block_ids"] = prefill_response[
                "kv_transfer_params"
            ]["remote_block_ids"]

        req_data["kv_transfer_params"]["remote_dp_size"] = prefill_instance_endpoint[
            "dp_size"
        ]
        req_data["kv_transfer_params"]["remote_tp_size"] = prefill_instance_endpoint[
            "tp_size"
        ]

        if selected_prefill_dp_rank is not None:
            req_data["kv_transfer_params"]["remote_dp_rank"] = selected_prefill_dp_rank

        decode_request_task = asyncio.create_task(
            start_decode_request(
                decode_instance_endpoint["request_address"], req_data, request_id
            )
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
