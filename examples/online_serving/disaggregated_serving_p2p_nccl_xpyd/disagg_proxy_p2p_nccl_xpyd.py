# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )
                return

            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


async def forward_profiling_request(url):
    """Forward start_profile or stop_profile to a vLLM instance."""
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        async with session.post(url=url, headers=headers) as response:
            return response.status == 200


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # finish prefill
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        # return decode
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback

        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


@app.route("/start_profile", methods=["POST"])
async def start_profile():
    """Start profiling on both prefill and decode instances."""
    try:
        success_count = 0
        total_count = 0

        # Forward to all prefill instances
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())

        for prefill_addr, _ in prefill_list:
            total_count += 1
            try:
                success = await forward_profiling_request(
                    f"http://{prefill_addr}/start_profile"
                )
                if success:
                    success_count += 1
                    print(f"✅ Started profiling on prefill: {prefill_addr}")
                else:
                    print(f"❌ Failed to start profiling on prefill: {prefill_addr}")
            except Exception as e:
                print(f"❌ Error starting profiling on prefill {prefill_addr}: {e}")

        # Forward to all decode instances
        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())

        for decode_addr, _ in decode_list:
            total_count += 1
            try:
                success = await forward_profiling_request(
                    f"http://{decode_addr}/start_profile"
                )
                if success:
                    success_count += 1
                    print(f"✅ Started profiling on decode: {decode_addr}")
                else:
                    print(f"❌ Failed to start profiling on decode: {decode_addr}")
            except Exception as e:
                print(f"❌ Error starting profiling on decode {decode_addr}: {e}")

        if success_count == total_count:
            return {
                "status": "success",
                "message": f"Profiling started on all {total_count} instances",
            }
        elif success_count > 0:
            return {
                "status": "partial",
                "message": f"Profiling started on {success_count}/{total_count} "
                f"instances",
            }
        else:
            return {
                "status": "error",
                "message": "Failed to start profiling on any instances",
            }, 500

    except Exception as e:
        print(f"❌ Error in start_profile: {e}")
        return {"status": "error", "message": str(e)}, 500


@app.route("/stop_profile", methods=["POST"])
async def stop_profile():
    """Stop profiling on both prefill and decode instances."""
    try:
        success_count = 0
        total_count = 0

        # Forward to all prefill instances
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())

        for prefill_addr, _ in prefill_list:
            total_count += 1
            try:
                success = await forward_profiling_request(
                    f"http://{prefill_addr}/stop_profile"
                )
                if success:
                    success_count += 1
                    print(f"✅ Stopped profiling on prefill: {prefill_addr}")
                else:
                    print(f"❌ Failed to stop profiling on prefill: {prefill_addr}")
            except Exception as e:
                print(f"❌ Error stopping profiling on prefill {prefill_addr}: {e}")

        # Forward to all decode instances
        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())

        for decode_addr, _ in decode_list:
            total_count += 1
            try:
                success = await forward_profiling_request(
                    f"http://{decode_addr}/stop_profile"
                )
                if success:
                    success_count += 1
                    print(f"✅ Stopped profiling on decode: {decode_addr}")
                else:
                    print(f"❌ Failed to stop profiling on decode: {decode_addr}")
            except Exception as e:
                print(f"❌ Error stopping profiling on decode {decode_addr}: {e}")

        if success_count == total_count:
            return {
                "status": "success",
                "message": f"Profiling stopped on all {total_count} instances",
            }
        elif success_count > 0:
            return {
                "status": "partial",
                "message": f"Profiling stopped on {success_count}/{total_count} "
                f"instances",
            }
        else:
            return {
                "status": "error",
                "message": "Failed to stop profiling on any instances",
            }, 500

    except Exception as e:
        print(f"❌ Error in stop_profile: {e}")
        return {"status": "error", "message": str(e)}, 500


if __name__ == "__main__":
    t = start_service_discovery("0.0.0.0", 30001)
    app.run(host="0.0.0.0", port=8000)
    t.join()
