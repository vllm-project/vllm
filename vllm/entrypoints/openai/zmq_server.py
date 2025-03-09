# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import signal
import traceback
from argparse import Namespace
from http import HTTPStatus

import zmq
import zmq.asyncio

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse, ZmqMsgRequest,
                                              ZmqMsgResponse)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.logger import init_logger
from vllm.utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger('vllm.entrypoints.openai.zmq_server')

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
                message_parts = await socket.recv_multipart()
                logger.debug("received request: %s", message_parts)
                logger.debug("received len: %d", len(message_parts))
                identity, body = message_parts[0], message_parts[1]
                zmq_msg_request = ZmqMsgRequest.model_validate_json(body)
                # launch request handler coroutine
                task = asyncio.create_task(
                    worker_routine(identity, zmq_msg_request, socket))
                running_requests.add(task)
                task.add_done_callback(running_requests.discard)
            except zmq.ZMQError as e:
                logger.error(traceback.format_exc())
                logger.error("ZMQError: %s", e)
                break
            except Exception as e:
                logger.error(traceback.format_exc())
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


async def worker_routine(identity: bytes, zmq_msg_request: ZmqMsgRequest,
                         socket: zmq.asyncio.Socket):
    """Worker routine"""
    try:
        request_id = zmq_msg_request.request_id
        logger.debug("receive request: %s from %s, request_id: %s",
                     zmq_msg_request.model_dump_json(), identity.decode(),
                     request_id)
        if isinstance(zmq_msg_request.body, CompletionRequest):
            await create_completion(identity, zmq_msg_request, socket)
        else:
            logger.error("Error in worker routine: %s request_id: %s",
                         "unsupported request type", request_id)
            raise Exception("unsupported request type")

    except Exception as e:
        logger.error("Error in worker routine: %s request_id: %s", e,
                     request_id)
        logger.error(traceback.format_exc())
        logger.debug("send ErrorResponse %s", str(e))
        await socket.send_multipart([
            identity,
            ZmqMsgResponse(request_id=request_id,
                           type=zmq_msg_request.type,
                           body={
                               "content": "unsupported request type",
                               "status_code": HTTPStatus.INTERNAL_SERVER_ERROR
                           }).model_dump_json().encode(),
        ])


async def create_completion(identity: bytes, zmq_msg_request: ZmqMsgRequest,
                            socket: zmq.asyncio.Socket):
    request: CompletionRequest = zmq_msg_request.body
    logger.debug("zmq request post: %s", request.model_dump_json())
    generator = await openai_serving_completion.create_completion(request)
    logger.debug("zmq request end post")
    request_id = zmq_msg_request.request_id
    if isinstance(generator, (ErrorResponse, CompletionResponse)):
        logger.debug("send response %s", generator.model_dump_json())
        zmq_msg_response = ZmqMsgResponse(request_id=request_id,
                                          type=zmq_msg_request.type,
                                          body_type="response")
        if isinstance(generator, ErrorResponse):
            zmq_msg_response.body = {
                "content": generator.model_dump(),
                "status_code": generator.code
            }
        elif isinstance(generator, CompletionResponse):
            zmq_msg_response.body = {"content": generator.model_dump()}

        await socket.send_multipart(
            [identity, zmq_msg_response.model_dump_json().encode()])
    else:
        async for chunk in generator:
            zmq_msg_response = ZmqMsgResponse(request_id=request_id,
                                              type=zmq_msg_request.type,
                                              body=chunk)
            if "data: [DONE]" not in chunk:
                zmq_msg_response.stop = False
            logger.debug("send chunk identity: %s, request_id: %s, chunk: %s",
                         identity.decode(), request_id, chunk)
            await socket.send_multipart(
                [identity,
                 zmq_msg_response.model_dump_json().encode()])
