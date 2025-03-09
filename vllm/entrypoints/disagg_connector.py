# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
import sys
import traceback
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Union

import uvicorn
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.protocol import (CompletionRequest, ZmqMsgRequest,
                                              ZmqMsgResponse)
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.disagg_connector')

TIME_OUT = 5
X_REQUEST_ID_KEY = "X-Request-Id"
CONTENT_TYPE_STREAM = "text/event-stream"

# communication between output handlers and execute_task_async
request_queues: dict[str, asyncio.Queue]


async def log_stats(request_queues: dict[str, asyncio.Queue]):
    while True:
        logger.info("Running requests: %d", len(request_queues))
        await asyncio.sleep(10)


# create async socket use ZMQ_DEALER
async def create_socket(url: str,
                        zmqctx: zmq.asyncio.Context) -> zmq.asyncio.Socket:
    sock = zmqctx.socket(zmq.DEALER)
    identity = f"connector-{uuid.uuid4()}"
    sock.setsockopt(zmq.IDENTITY, identity.encode())
    sock.connect(url)
    logger.info("%s started at %s", identity, url)
    return sock


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create socket pool with prefill and decode
    logger.info("start connect zmq server")
    app.state.zmqctx = zmq.asyncio.Context()
    app.state.prefill_socket = await create_socket(app.state.prefill_addr,
                                                   zmqctx=app.state.zmqctx)
    logger.info("success create_socke sockets_prefill")
    app.state.decode_socket = await create_socket(app.state.decode_addr,
                                                  zmqctx=app.state.zmqctx)
    logger.info("success create_socket sockets_decode")
    global request_queues
    request_queues = {}
    asyncio.create_task(prefill_handler(app.state.prefill_socket))
    asyncio.create_task(decode_handler(app.state.decode_socket))
    asyncio.create_task(log_stats(request_queues))
    yield
    ## close zmq context
    logger.info("shutdown disagg connector")
    logger.info("term zmqctx")
    app.state.zmqctx.destroy(linger=0)


app = FastAPI(lifespan=lifespan)


@app.post('/v1/completions')
async def completions(request: CompletionRequest, raw_request: Request,
                      background_tasks: BackgroundTasks):
    try:
        # Add the X-Request-Id header to the raw headers list
        header = dict(raw_request.headers)
        request_id = header.get(X_REQUEST_ID_KEY)
        queue: asyncio.Queue[ZmqMsgResponse] = asyncio.Queue()
        if request_id is None:
            request_id = str(uuid.uuid4())
            logger.debug("add X-Request-Id: %s", request_id)
        logger.debug("X-Request-Id is: %s", request_id)
        request_queues[request_id] = queue
        zmq_msg_request = ZmqMsgRequest(request_id=request_id,
                                        type="completions",
                                        body=request)
        logger.info("Received request_id: %s, request: %s, header: %s",
                    request_id, zmq_msg_request.model_dump_json(), header)
        original_max_tokens = request.max_tokens
        # change max_tokens = 1 to let it only do prefill
        request.max_tokens = 1
        # finish prefill
        try:
            prefill_response = await prefill(zmq_msg_request)
            if isinstance(prefill_response, JSONResponse
                          ) and prefill_response.status_code != HTTPStatus.OK:
                return prefill_response
            logger.debug("finish prefill start decode")
            request.max_tokens = original_max_tokens
            response = await decode(zmq_msg_request)
            logger.debug("finish decode")
        except Exception as e:
            logger.error("Error occurred in disagg prefill proxy server, %s",
                         e)
            response = JSONResponse(
                {"error": {
                    "message": str(e)
                }},
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        return response

    except Exception as e:
        exc_info = sys.exc_info()
        logger.error("Error occurred in disagg prefill proxy server")
        logger.error(e)
        logger.error("".join(traceback.format_exception(*exc_info)))
        response = JSONResponse({"error": {
            "message": str(e)
        }}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return response
    finally:
        if request_id is not None:
            background_tasks.add_task(cleanup_request_id, request_id)


async def socket_recv_handler(socket: zmq.asyncio.Socket, scene: str):
    while True:
        try:
            [body] = await socket.recv_multipart()
            response = ZmqMsgResponse.model_validate_json(body)
            request_id = response.request_id
            logger.debug("%s socket received result: %s", scene,
                         response.model_dump_json())
            if request_id in request_queues:
                request_queues[request_id].put_nowait(response)
            else:
                logger.debug(
                    "%s socket received but request_id not found discard: %s",
                    scene, request_id)
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("%s handler error: %s", scene, e)


# prefill handler
async def prefill_handler(prefill_socket: zmq.asyncio.Socket):
    await socket_recv_handler(prefill_socket, "prefill")


# decode handler
async def decode_handler(decode_socket: zmq.asyncio.Socket):
    await socket_recv_handler(decode_socket, "decode")


# select a socket and execute task
async def execute_task_async(zmq_msg_request: ZmqMsgRequest,
                             socket: zmq.asyncio.Socket):
    try:
        request_id = zmq_msg_request.request_id
        requestBody = zmq_msg_request.model_dump_json()
        logger.debug("Sending requestBody: %s", requestBody)
        socket.send_multipart([requestBody.encode()])
        logger.debug("Sent end")
        queue = request_queues[request_id]
        while True:
            logger.debug("Waiting for reply")
            zmq_msg_response: ZmqMsgResponse = await asyncio.wait_for(
                queue.get(), TIME_OUT)
            logger.debug("Received result: %s",
                         zmq_msg_response.model_dump_json())
            yield zmq_msg_response
            if zmq_msg_response.stop:
                logger.debug("Received stop: %s", zmq_msg_response.stop)
                break
    except asyncio.TimeoutError:
        logger.error(traceback.format_exc())
        yield JSONResponse("timeout", HTTPStatus.REQUEST_TIMEOUT)
    finally:
        logger.debug("request_id: %s, execute_task_async end", request_id)


async def prefill(zmq_msg_request: ZmqMsgRequest) -> Union[JSONResponse, bool]:
    logger.debug("start prefill")
    generator = execute_task_async(zmq_msg_request, app.state.prefill_socket)
    async for res in generator:
        logger.debug("res: %s", res)
        if res.body_type == "response":
            return JSONResponse(res.body)
    return True


async def generate_stream_response(
        fisrt_reply: str, generator: AsyncGenerator[ZmqMsgResponse]
) -> AsyncGenerator[dict, str]:
    yield fisrt_reply
    async for reply in generator:
        yield reply.body


async def decode(
        zmq_msg_request: ZmqMsgRequest
) -> Union[JSONResponse, StreamingResponse]:
    logger.debug("start decode")
    generator = execute_task_async(zmq_msg_request, app.state.decode_socket)

    async for res in generator:
        logger.debug("res: %s", res)
        if res.body_type == "response":
            return JSONResponse(res.body)
        else:
            return StreamingResponse(generate_stream_response(
                res.body, generator),
                                     media_type=CONTENT_TYPE_STREAM)

    # If the generator is empty, return a default error response
    logger.error("No response received from generator")
    return JSONResponse({"error": "No response received from generator"},
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR)


def cleanup_request_id(request_id: str):
    if request_id in request_queues:
        logger.info("del request_id: %s, decode finished", request_id)
        del request_queues[request_id]


async def run_disagg_connector(args, **uvicorn_kwargs):
    logger.info("vLLM Disaggregate Connector start %s %s", args,
                uvicorn_kwargs)
    logger.info(args.prefill_addr)
    app.state.port = args.port
    app.state.prefill_addr = f"ipc://{args.prefill_addr}"
    app.state.decode_addr = f"ipc://{args.decode_addr}"
    logger.info(
        "start connect prefill_addr: %s decode_addr: %s "
        "zmq server fastapi port: %s", app.state.prefill_addr,
        app.state.decode_addr, app.state.port)

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)
    # init uvicorn server
    config = uvicorn.Config(app, host="0.0.0.0", port=app.state.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be sync with vllm/entrypoints/cli/connect.py for CLI
    # entrypoints.
    parser = FlexibleArgumentParser(description="vLLM disagg connect server.")
    parser.add_argument("--port",
                        type=int,
                        default=8001,
                        help="The fastapi server port default 8001")
    # security concern only support ipc now
    parser.add_argument("--prefill-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc prefill address")
    parser.add_argument("--decode-addr",
                        type=str,
                        required=True,
                        help="The zmq ipc decode address")

    args = parser.parse_args()

    uvloop.run(run_disagg_connector(args))
