# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import signal
import sys
import traceback
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Union

import uvicorn
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.zmq_server import (CONTENT_TYPE_ERROR,
                                                CONTENT_TYPE_JSON,
                                                CONTENT_TYPE_STREAM)
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.disagg_connector')

TIME_OUT = 5
X_REQUEST_ID_KEY = "X-Request-Id"

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
async def completions(request: Request, background_tasks: BackgroundTasks):
    try:
        # Add the X-Request-Id header to the raw headers list
        header = dict(request.headers)
        request_id = header.get(X_REQUEST_ID_KEY)
        queue = asyncio.Queue()
        if request_id is None:
            request_id = str(uuid.uuid4())
            logger.debug("add X-Request-Id: %s", request_id)
            header[X_REQUEST_ID_KEY] = request_id
        logger.debug("X-Request-Id is: %s", request_id)
        request_queues[request_id] = queue
        request_data = await request.json()
        logger.info("Received request_id: %s, request: %s, header: %s",
                    request_id, request_data, header)
        original_max_tokens = request_data['max_tokens']
        # change max_tokens = 1 to let it only do prefill
        request_data['max_tokens'] = 1
        # finish prefill
        try:
            prefill_response = await prefill(header, request_data)
            if isinstance(prefill_response, JSONResponse):
                return prefill_response
            logger.debug("finish prefill start decode")
            request_data['max_tokens'] = original_max_tokens
            response = await decode(header, request_data)
            logger.debug("finish decode")
        except Exception as e:
            logger.error("Error occurred in disagg prefill proxy server, %s",
                         e)
            response = JSONResponse({"error": {
                "message": str(e)
            }},
                                    status_code=500)
        return response

    except Exception as e:
        exc_info = sys.exc_info()
        logger.error("Error occurred in disagg prefill proxy server")
        logger.error(e)
        logger.error("".join(traceback.format_exception(*exc_info)))
        response = JSONResponse({"error": {
            "message": str(e)
        }},
                                status_code=500)
        return response
    finally:
        background_tasks.add_task(cleanup_request_id, request_id)


async def socket_recv_handler(socket: zmq.asyncio.Socket, scene: str):
    while True:
        try:
            [request_id, contentType, reply] = await socket.recv_multipart()
            contentType_str = contentType.decode()
            reply_str = reply.decode()
            request_id_str = request_id.decode()
            logger.debug(
                "%s socket received result contentType: %s, "
                "request_id: %s, reply: %s", scene, contentType_str,
                request_id_str, reply_str)
            if request_id_str in request_queues:
                request_queues[request_id_str].put_nowait(
                    (contentType_str, reply_str))
                if "[DONE]" in reply_str:
                    logger.debug(
                        "%s socket received stop signal request_id: %s", scene,
                        request_id_str)
                    request_queues[request_id_str].put_nowait(
                        (contentType_str, None))
            else:
                logger.debug(
                    "%s socket received but request_id not found discard: %s",
                    scene, request_id_str)
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
async def execute_task_async(headers: dict, request: dict,
                             socket: zmq.asyncio.Socket):
    try:
        request_id = headers.get(X_REQUEST_ID_KEY)
        requestBody = json.dumps(request)
        logger.info("Sending requestBody: %s", requestBody)
        socket.send_multipart([request_id.encode(), requestBody.encode()])
        logger.debug("Sent end")
        queue = request_queues[request_id]
        while True:
            logger.debug("Waiting for reply")
            (contentType,
             reply) = await asyncio.wait_for(queue.get(), TIME_OUT)
            logger.debug("Received result: %s, %s", contentType, reply)
            if reply is None:
                logger.debug("Received stop signal, request_id: %s",
                             request_id)
                break
            yield (contentType, reply)
            if contentType == CONTENT_TYPE_JSON:
                logger.debug("Received %s message, request_id: %s",
                             contentType, request_id)
                break

    except asyncio.TimeoutError:
        logger.error(traceback.format_exc())
        yield (CONTENT_TYPE_ERROR, "System Error")
    finally:
        logger.debug("request_id: %s, execute_task_async end", request_id)


async def prefill(header: dict,
                  original_request_data: dict) -> Union[JSONResponse, bool]:
    logger.debug("start prefill")
    generator = execute_task_async(header, original_request_data,
                                   app.state.prefill_socket)
    async for contentType, reply in generator:
        logger.debug("contentType: %s, reply: %s", contentType, reply)
        if contentType == CONTENT_TYPE_ERROR:
            response = JSONResponse({"error": reply})
            response.status_code = 500
            return response
    return True


async def generate_stream_response(fisrt_reply: str,
                                   generator: AsyncGenerator):
    yield fisrt_reply
    async for _, reply in generator:
        yield reply


async def decode(
        header: dict,
        original_request_data: dict) -> Union[JSONResponse, StreamingResponse]:
    logger.info("start decode")
    generator = execute_task_async(header, original_request_data,
                                   app.state.decode_socket)

    async for contentType, reply in generator:
        logger.debug("contentType: %s, reply: %s", contentType, reply)
        if contentType == CONTENT_TYPE_ERROR:
            response = JSONResponse({"error": reply})
            response.status_code = 500
            return response
        elif contentType == CONTENT_TYPE_JSON:
            return JSONResponse(reply)
        else:
            return StreamingResponse(generate_stream_response(
                reply, generator),
                                     media_type=CONTENT_TYPE_STREAM)


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
