# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import signal
import traceback
from argparse import Namespace

import zmq
import zmq.asyncio
from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.logger import init_logger
from vllm.utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger('vllm.entrypoints.openai.zmq_server')

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_ERROR = "error"
CONTENT_TYPE_STREAM = "text/event-stream"

openai_serving_completion: OpenAIServingCompletion
openai_serving_models: OpenAIServingModels


async def log_stats(running_requests: set[asyncio.Task]):
    while True:
        logger.info("Running requests: %d", len(running_requests))
        await asyncio.sleep(10)


def _cleanup_ipc_path(server_addr: str):
    socket_path = server_addr.replace("ipc://", "")
    logger.info("cleaning up local IPC socket file %s", socket_path)
    if os.path.exists(socket_path):
        os.remove(socket_path)


async def serve_zmq(arg) -> None:
    """Server routine"""
    logger.info("zmq Server start arg: %s, zmq_server_addr: %s", arg,
                arg.zmq_server_addr)
    # different zmq context can't communicate use inproc
    server_addr = f"ipc://{arg.zmq_server_addr}"
    try:
        # Prepare our context and sockets
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.ROUTER)
        # unlimited HWM
        hwm_limit = 0

        socket.bind(server_addr)
        socket.setsockopt(zmq.SNDHWM, hwm_limit)
        socket.setsockopt(zmq.RCVHWM, hwm_limit)

        running_requests: set[asyncio.Task] = set()
        logger.info("zmq Server started at %s", server_addr)
        asyncio.create_task(log_stats(running_requests))

        while True:
            try:
                logger.debug("zmq Server waiting for request")
                # get new request from the client
                identity, request_id, body = await socket.recv_multipart()
                # launch request handler coroutine
                task = asyncio.create_task(
                    worker_routine(identity, request_id, body, socket))
                running_requests.add(task)
                task.add_done_callback(running_requests.discard)
            except zmq.ZMQError as e:
                logger.error("ZMQError: %s", e)
                break
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting")
    finally:
        # Clean up resources
        for task in running_requests:
            task.cancel()
        await asyncio.gather(*running_requests, return_exceptions=True)
        socket.close()
        context.destroy(linger=0)
        _cleanup_ipc_path(server_addr)


async def run_zmq_server(args) -> None:
    logger.info("vLLM zmq server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:

        model_config = await engine_client.get_model_config()
        await init_state(engine_client, model_config, args)
        logger.info("init_state successful")
        await serve_zmq(args)


async def init_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]

    global openai_serving_models
    openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
    )
    await openai_serving_models.init_static_loras()

    global openai_serving_completion
    openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        openai_serving_models,
        request_logger=None,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )


async def worker_routine(identity: bytes, request_id: bytes, body: bytes,
                         socket: zmq.asyncio.Socket):
    """Worker routine"""
    try:
        body_json = json.loads(body.decode())
        request_id_str = request_id.decode()
        logger.debug("receive request: %s from %s, request_id: %s", body_json,
                     identity.decode(), request_id_str)

        completionRequest = CompletionRequest(**body_json)
        generator = await create_completion(completionRequest, None)
        content_type_json = CONTENT_TYPE_JSON.encode('utf-8')
        content_type_stream = CONTENT_TYPE_STREAM.encode('utf-8')
        if isinstance(generator, ErrorResponse):
            content = json.loads(generator.model_dump_json())
            content.update({"status_code": generator.code})
            logger.debug("send ErrorResponse %s", json.dumps(content))
            await socket.send_multipart([
                identity, request_id, content_type_json,
                json.dumps(content).encode('utf-8')
            ])
        elif isinstance(generator, CompletionResponse):
            logger.debug("send CompletionResponse %s",
                         json.dumps(generator.model_dump()))
            await socket.send_multipart([
                identity, request_id, content_type_json,
                json.dumps(generator.model_dump()).encode('utf-8')
            ])
        else:
            async for chunk in generator:
                logger.debug(
                    "send chunk identity: %s, request_id: %s, chunk: %s",
                    identity.decode(), request_id.decode(), chunk)
                await socket.send_multipart([
                    identity, request_id, content_type_stream,
                    chunk.encode('utf-8')
                ])
    except Exception as e:
        logger.error("Error in worker routine: %s request_id: %s", e,
                     request_id_str)
        logger.error(traceback.format_exc())
        content_type_stream = CONTENT_TYPE_STREAM.encode('utf-8')
        logger.debug("send ErrorResponse %s", str(e))
        await socket.send_multipart([
            identity, request_id,
            CONTENT_TYPE_ERROR.encode('utf-8'),
            str(e).encode('utf-8')
        ])


async def create_completion(request: CompletionRequest, raw_request: Request):
    logger.debug("zmq request post: %s", request)
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    logger.debug("zmq request end post")
    return generator
