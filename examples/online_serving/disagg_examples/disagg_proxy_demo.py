# SPDX-License-Identifier: Apache-2.0
import argparse
import ipaddress
import itertools
import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional
from datetime import datetime

import aiohttp
import requests
import uvicorn
from fastapi import (APIRouter, Depends, FastAPI, Header, HTTPException,
                     Request, status)
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)



async def P_first_token_generator(generator_p, generator_d, callback_owner=None, prefill_instance:str=None, decode_instance:str=None):
    first_decode = True
    async for chunk in generator_p:
        yield chunk
    print(f"P->[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] prefill completed: ", prefill_instance)
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(prefill_instance=prefill_instance)

    async for chunk in generator_d:
        if first_decode:
            first_decode = False
            continue
        yield chunk
    print(f"P->[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] decode completed: ", decode_instance)
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(decode_instance=decode_instance)

async def D_first_token_generator(generator_p, generator_d, callback_owner=None, prefill_instance:str=None, decode_instance:str=None):
    async for _ in generator_p:
        continue
    print(f"D->[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] prefill completed: ", prefill_instance)
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(prefill_instance=prefill_instance)

    async for chunk in generator_d:
        yield chunk
    print(f"D->[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] decode completed: ", decode_instance)
    if callback_owner and hasattr(callback_owner, "on_done"):
        callback_owner.on_done(decode_instance=decode_instance)

class SchedulingPolicy(ABC):

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError("Scheduling Proxy is not set.")


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
        generator_on_p_node:bool = False
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()
        self.generator = P_first_token_generator if generator_on_p_node else D_first_token_generator

    def on_done(self, prefill_instance:str=None, decode_instance:str=None):
        self.schedule_completion(prefill_instance, decode_instance)

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

    def schedule(self, cycler: itertools.cycle, request_len: Optional[int] = None) -> str:
        return self.scheduling_policy.schedule(cycler, request_len)

    def schedule_completion(self, prefill_instance:str=None, decode_instance:str=None):
        self.scheduling_policy.schedule_completion(prefill_instance=prefill_instance, decode_instance=decode_instance)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()
            
            if len(self.prefill_instances) > 0:
                kv_prepare_request = request.copy()
                kv_prepare_request["max_tokens"] = 1
                
                print("create_completion, request_len=", len(kv_prepare_request['prompt']))
                prefill_instance = self.schedule(self.prefill_cycler, request_len=len(kv_prepare_request['prompt']))
                value = b''
                try:
                    async for chunk in self.forward_request(
                            f"http://{prefill_instance}/v1/completions",
                            kv_prepare_request):
                        value += chunk
                except HTTPException as http_exc:
                    self.remove_instance_endpoint("prefill", prefill_instance)
                    raise http_exc

            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)
            value = value.strip().decode("utf-8").removesuffix("data: [DONE]").encode("utf-8")
            async def streaming_response(value):
                if value:
                    yield value
                else:
                    yield b""
            generator_p = streaming_response(value)
            try:
                generator_d = self.forward_request(
                    f"http://{decode_instance}/v1/completions", request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            final_generator = self.generator(generator_p, generator_d, self, prefill_instance, decode_instance)    
            response = StreamingResponse(final_generator)
            return response
        except Exception:
            import sys

            exc_info = sys.exc_info()
            print("Error occurred in disagg proxy server")
            print(exc_info)

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            # add params to request
            kv_prepare_request = request.copy()
            kv_prepare_request["max_tokens"] = 1

            # prefill stage
            total_length = sum(len(msg['content']) for msg in kv_prepare_request['messages'])
            print("Total content length:", total_length)
            prefill_instance = self.schedule(self.prefill_cycler, request_len=total_length)

            value = b''
            try:
                async for chunk in self.forward_request(
                        f"http://{prefill_instance}/v1/chat/completions",
                        kv_prepare_request):
                    value += chunk
            except HTTPException as http_exc:
                self.remove_instance_endpoint("prefill", prefill_instance)
                raise http_exc
            # Perform kv recv and decoding stage
            decode_instance = self.schedule(self.decode_cycler)
            value = value.strip().decode("utf-8").removesuffix("data: [DONE]").encode("utf-8")
            async def streaming_response(value):
                if value:
                    yield value
                else:
                    yield b""
            generator_p = streaming_response(value)
            try:
                generator_d = self.forward_request(
                    "http://" + decode_instance + "/v1/chat/completions",
                    request)
            except HTTPException as http_exc:
                self.remove_instance_endpoint("decode", decode_instance)
                raise http_exc
            final_generator = self.generator(generator_p, generator_d, self, prefill_instance, decode_instance)
            response = StreamingResponse(final_generator)
            return response
        except Exception:
            exc_info = sys.exc_info()
            error_messages = [str(e) for e in exc_info if e]
            print("Error occurred in disagg proxy server")
            print(error_messages)
            return StreamingResponse(content=iter(error_messages),
                                     media_type="text/event-stream")

    def remove_instance_endpoint(self, instance_type, instance):
        with self.scheduling_policy.lock:
            if (instance_type == "decode"
                    and instance in self.decode_instances):
                self.decode_instances.remove(instance)
                self.decode_cycler = itertools.cycle(self.decode_instances)
            if (instance_type == "prefill"
                    and instance in self.prefill_instances):
                self.prefill_instances.remove(instance)
                self.prefill_cycler = itertools.cycle(self.decode_instances)


class RoundRobinSchedulingPolicy(SchedulingPolicy):

    def __init__(self):
        print("RoundRobinSchedulingPolicy")
        super().__init__()

    def safe_next(self, cycler: itertools.cycle):
        with self.lock:
            return next(cycler)

    def schedule(self, cycler: itertools.cycle, request: Optional[dict[str, any]] = None) -> str:
        return self.safe_next(cycler)

class LoadBalancedScheduler(SchedulingPolicy):

    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str]
    ):
        self.prefill_utils_counter = [0] * len(prefill_instances)
        self.prefill_bs_counter = [0] * len(prefill_instances)
        self.decode_bs_counter = [0] * len(decode_instances)
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        print(" LoadBalancedScheduler, prefill/decode instance is = ", len(self.prefill_bs_counter), len(self.decode_bs_counter))
        print(" LoadBalancedScheduler, self.prefill_instances =", self.prefill_instances)
        print(" LoadBalancedScheduler, self.decode_instances =", self.decode_instances)
        super().__init__()

    def schedule(self, cycler: itertools.cycle, request_len: Optional[int] = None) -> str:
        with self.lock:
            if request_len:
                min_value = min(self.prefill_utils_counter)
                min_index = self.prefill_utils_counter.index(min_value)
                self.prefill_bs_counter[min_index] += 1
                self.prefill_utils_counter[min_index] += request_len
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] schedule prefill! scheduling prefill instance... min_value={min_value}, min_index={min_index}")
                return self.prefill_instances[min_index]
            else:
                min_value = min(self.decode_bs_counter)
                min_index = self.decode_bs_counter.index(min_value)
                self.decode_bs_counter[min_index] += 1
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] schedule decode! scheduling decode instance... min_value={min_value}, min_index={min_index}")
                return self.decode_instances[min_index]

    def schedule_completion(self, prefill_instance:str=None, decode_instance:str=None):
        with self.lock:
            if prefill_instance:
                print(" LoadBalancedScheduler->schedule_completion prefill_instance =", prefill_instance)
                index = self.prefill_instances.index(prefill_instance)
                self.prefill_bs_counter[index] -= 1
                all_zero = True
                for index, _ in enumerate(self.prefill_instances):
                    if self.prefill_bs_counter[index] != 0:
                        all_zero = False
                        break
                if all_zero:
                    print("all bs is 0, clear entire entries")
                    for index, _ in enumerate(self.prefill_instances):
                        self.prefill_utils_counter[index] = 0

            if decode_instance:
                print(" LoadBalancedScheduler->schedule_completion decode_instance =", decode_instance)
                index = self.decode_instances.index(decode_instance)
                self.decode_bs_counter[index] -= 1


        
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
            scheduling_policy=(scheduling_policy(args.prefill, args.decode) if scheduling_policy
                               is not None else RoundRobinSchedulingPolicy()),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
            generator_on_p_node=args.generator_on_p_node,
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
        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, loop="uvloop")
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
    
    parser.add_argument(
        "--generator_on_p_node",
        action="store_true",
        help="generate first token on P node or D node",
    )

    parser.add_argument(
        "--roundrobin",
        action="store_true",
        help="Use Round Robin scheduling for load balancing",
    )
    args = parser.parse_args()
    if args.roundrobin:
        proxy_server = ProxyServer(args=args)
    else:
        proxy_server = ProxyServer(args=args, scheduling_policy=LoadBalancedScheduler)
    proxy_server.run_server()
