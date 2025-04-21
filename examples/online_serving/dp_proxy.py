# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import ipaddress
import itertools
import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager, suppress
from typing import Callable, Optional

import aiohttp
import requests
import uvicorn
from fastapi import (APIRouter, Depends, FastAPI, Header, HTTPException,
                     Request, status)
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class SchedulingPolicy(ABC):

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


def deep_hash_dict(d: dict) -> int:
    """Hashes a dict deeply by serializing it to a stable JSON string."""
    json_str = json.dumps(d, sort_keys=True)
    return hash(json_str)


class HashableQueue:

    def __init__(self):
        self.queue = deque()
        self.items_set = set()

    def put(self, item):
        item_hash = deep_hash_dict(item)
        if item_hash not in self.items_set:
            self.queue.append(item)
            self.items_set.add(item_hash)

    def get(self):
        item = self.queue.popleft()
        item_hash = deep_hash_dict(item)
        self.items_set.remove(item_hash)
        return item

    def contains(self, item):
        item_hash = deep_hash_dict(item)
        return item_hash in self.items_set

    def qsize(self):
        return len(self.items_set)


class Proxy:

    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        custom_create_completion: Optional[Callable[[Request],
                                                    StreamingResponse]] = None,
        custom_create_chat_completion: Optional[Callable[
            [Request], StreamingResponse]] = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.send_requests_list = HashableQueue()
        self.router = APIRouter()
        logger.info("proxy server is starting..., decode_instances: %s",
                    decode_instances)
        self.setup_routes()

    def setup_routes(self):
        self.router.post(
            "/v1/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_completion if self.
               custom_create_completion else self.create_completion)
        self.router.post(
            "/v1/chat/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_chat_completion if self.
               custom_create_chat_completion else self.create_chat_completion)
        self.router.get("/status",
                        response_class=JSONResponse)(self.get_status)
        self.router.post("/instances/add",
                         dependencies=[Depends(self.api_key_authenticate)
                                       ])(self.add_instance_endpoint)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail=
                "Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s",
                           x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(
                    timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.",
                                            instance)
                                return True
                            else:
                                logger.warning("Mismatch model %s : %s != %s",
                                               instance, model_cur, self.model)
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400,
                                        detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance address.") from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(status_code=400,
                                    detail="Instance validation failed.")

            if instance_type == "prefill":
                with self.scheduling_policy.lock:
                    if instance not in self.prefill_instances:
                        self.prefill_instances.append(instance)
                        self.prefill_cycler = itertools.cycle(
                            self.prefill_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")
            else:
                with self.scheduling_policy.lock:
                    if instance not in self.decode_instances:
                        self.decode_instances.append(instance)
                        self.decode_cycler = itertools.cycle(
                            self.decode_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")

            return JSONResponse(content={
                "message":
                f"Added {instance} to {instance_type}_instances."
            })
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:  # noqa: E501
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(  # noqa: E501
                                    1024):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error("Request failed with status %s: %s",
                                     response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=
                            f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail=
                    "Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    async def fire_and_forget_post(self, url, payload):

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }
            try:
                async with session.post(url, headers=headers,
                                        json=payload) as response:
                    return await response.text()
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    async def process_queued_requests(self):
        n_workers = len(self.decode_instances)
        waiting_time = 0
        while True:
            if self.send_requests_list.qsize() == 0:
                await asyncio.sleep(0.1)
                continue

            if self.send_requests_list.qsize() < n_workers:
                # If there are fewer than n_workers requests, wait for a bit
                waiting_time += 0.1
                if waiting_time < 5:
                    logger.info("Waiting for more requests.")
                    await asyncio.sleep(0.1)
                    continue
                else:
                    logger.info(
                        "Still can't collect %s requests, will try padding.",
                        str(n_workers))

            waiting_time = 0
            # Pull up to n_workers requests

            batch = []
            fetch_num_requests = min(n_workers,
                                     self.send_requests_list.qsize())
            for _ in range(fetch_num_requests):
                batch.append(self.send_requests_list.get())
            logger.info("Getting %s requests from queue...", str(len(batch)))

            # If fewer than n_workers, duplicate to reach n_workers
            actual_len = len(batch)
            padding_len = n_workers - actual_len
            if padding_len == 0:
                continue

            request = batch[0]

            logger.info("padding requests length is %s", str(len(batch)))
            tasks = []
            for _ in range(padding_len):
                decode_instance = self.schedule(self.decode_cycler)
                logger.info(
                    "Forwarding padding request to decode instance: %s",
                    str(decode_instance))
                try:
                    tasks.append(
                        self.fire_and_forget_post(
                            f"http://{decode_instance}/v1/completions",
                            request))
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("decode", decode_instance)
                    logger.error("Decode instance error: {}", http_exc)
            await asyncio.gather(*tasks)

    async def get_pass_approval(self, request):
        self.send_requests_list.put(request)
        # wait until request is offlist
        while True:
            if self.send_requests_list.contains(request):
                await asyncio.sleep(0.1)
                continue
            else:
                break

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            await self.get_pass_approval(request)

            if len(self.prefill_instances) > 0:
                kv_prepare_request = request.copy()
                kv_prepare_request["max_tokens"] = 1

                prefill_instance = self.schedule(self.prefill_cycler)
                try:
                    async for _ in self.forward_request(
                            f"http://{prefill_instance}/v1/completions",
                            kv_prepare_request):
                        continue
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("prefill", prefill_instance)
                    raise http_exc

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)
            logger.info("Forwarding actual request to decode instance: %s",
                        str(decode_instance))

            try:
                generator = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            response = StreamingResponse(generator)
            return response
        except Exception:
            import sys
            import traceback
            logger.error("Error in create_completion: %s", str(sys.exc_info()))
            traceback.print_exc()

    async def create_chat_completion(self, raw_request: Request):
        logger.info("Received request for completion.")
        try:
            request = await raw_request.json()

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1

            # prefill stage
            prefill_instance = self.schedule(self.prefill_cycler)
            try:
                async for _ in self.forward_request(
                        f"http://{prefill_instance}/v1/chat/completions",
                        kv_prepare_request):
                    continue
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc
            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)

            try:
                generator = self.forward_request(
                    "http://" + decode_instance + "/v1/chat/completions",
                    request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            response = StreamingResponse(content=generator)
            return response
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            logger.error("Error occurred in disagg proxy server, %s",
                         str(exc_info))
            return StreamingResponse(content=iter(error_messages),
                                     media_type="text/event-stream")

    def remove_instance_endpoint(self, instance_type, instance):
        with self.scheduling_policy.lock:
            if (instance_type == "decode"
                    and instance in self.decode_instances):
                self.decode_instances.remove(instance)
                self.decode_cycler = itertools.cycle(self.decode_instances)
            if (instance_type == "prefill"
                    and instance in self.decode_instances):
                self.prefill_instances.remove(instance)
                self.prefill_cycler = itertools.cycle(self.decode_instances)


class RoundRobinSchedulingPolicy(SchedulingPolicy):

    def __init__(self):
        super().__init__()

    def safe_next(self, cycler: itertools.cycle):
        with self.lock:
            return next(cycler)

    def schedule(self, cycler: itertools.cycle) -> str:
        return self.safe_next(cycler)


class ProxyServer:

    def __init__(
        self,
        args: argparse.Namespace,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        create_completion: Optional[Callable[[Request],
                                             StreamingResponse]] = None,
        create_chat_completion: Optional[Callable[[Request],
                                                  StreamingResponse]] = None,
    ):
        self.validate_parsed_serve_args(args)
        self.port = args.port
        self.proxy_instance = Proxy(
            prefill_instances=[] if args.prefill is None else args.prefill,
            decode_instances=[] if args.decode is None else args.decode,
            model=args.model,
            scheduling_policy=(scheduling_policy if scheduling_policy
                               is not None else RoundRobinSchedulingPolicy()),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
        )

    def validate_parsed_serve_args(self, args: argparse.Namespace):
        # if not args.prefill:
        #     raise ValueError("Please specify at least one prefill node.")
        if not args.decode:
            raise ValueError("Please specify at least one decode node.")
        if args.prefill:
            self.validate_instances(args.prefill)
            self.verify_model_config(args.prefill, args.model)
        self.validate_instances(args.decode)
        self.verify_model_config(args.decode, args.model)

    def validate_instances(self, instances: list):
        for instance in instances:
            if len(instance.split(":")) != 2:
                raise ValueError(f"Invalid instance format: {instance}")
            host, port = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port)
                if not (0 < port < 65536):
                    raise ValueError(
                        f"Invalid port number in instance: {instance}")
            except Exception as e:
                raise ValueError(
                    f"Invalid instance {instance}: {str(e)}") from e

    def verify_model_config(self, instances: list, model: str) -> None:
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models")
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    if model_cur != model:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}")
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}") from e

    def run_server(self):
        app = FastAPI()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Start your background task
            background_task = asyncio.create_task(
                self.proxy_instance.process_queued_requests())

            yield  # app runs during this

            # On shutdown: clean up
            background_task.cancel()
            with suppress(asyncio.CancelledError):
                await background_task

        app.include_router(self.proxy_instance.router)
        app.router.lifespan_context = lifespan
        config = uvicorn.Config(app,
                                host="0.0.0.0",
                                port=self.port,
                                loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


if __name__ == "__main__":
    # Todo: allow more config
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )
    args = parser.parse_args()
    proxy_server = ProxyServer(args=args)
    proxy_server.run_server()
