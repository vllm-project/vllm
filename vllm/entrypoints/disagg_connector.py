# SPDX-License-Identifier: Apache-2.0

import json
import signal
import uuid
# from fastapi.lifespan import Lifespan
from asyncio import Queue
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger

# default prefill and decode addr
fastapi_port = 8001
prefill_addr = "ipc://localhost:7010"
socket_prefill_num = 5
decode_addr = "ipc://localhost:7020"
socket_decode_num = 5
context_type_json = "application/json"

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.connect')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create socket pool with prefill and decode
    logger.info("start create_socket_pool")
    app.state.zmqctx = zmq.asyncio.Context()
    app.state.sockets_prefill = await create_socket_pool(
        app.state.prefill_addr, socket_prefill_num, zmqctx=app.state.zmqctx)
    logger.info("success create_socket_pool sockets_prefill")
    app.state.sockets_decode = await create_socket_pool(
        app.state.decode_addr, socket_decode_num, zmqctx=app.state.zmqctx)
    logger.info("success create_socket_pool sockets_decode")
    yield
    ## close zmq context
    logger.info("term zmqctx")
    app.state.zmqctx.destroy(linger=0)


app = FastAPI(lifespan=lifespan)


# create async socket pool with num_sockets use ZMQ_DEALER
async def create_socket_pool(url: str, num_sockets: int,
                             zmqctx: zmq.asyncio.Context) -> Queue:
    sockets: Queue = Queue()
    for i in range(num_sockets):
        sock = zmqctx.socket(zmq.DEALER)
        identity = f"worker-{i}-{uuid.uuid4()}"
        sock.setsockopt(zmq.IDENTITY, identity.encode())
        sock.connect(url)
        logger.info("%s started at %s with queue size %s", identity, url,
                    sockets.qsize())
        await sockets.put(sock)
    return sockets


# select a socket and execute task
async def execute_task_async(route: str, headers: dict, request: dict,
                             sockets: Queue):
    sock = await sockets.get()
    try:
        requestBody = json.dumps(request)
        headersJson = json.dumps(headers)
        logger.info("Sending requestBody: %s to %s with headers: %s",
                    requestBody, route, headersJson)
        await sock.send_multipart(
            [route.encode(),
             headersJson.encode(),
             requestBody.encode()])
        logger.info("Sent end")
        while True:
            logger.info("Waiting for reply")
            [contentType, reply] = await sock.recv_multipart()
            contentType_str = contentType.decode()
            reply_str = reply.decode()
            logger.info("Received result: %s, %s", contentType_str, reply_str)
            yield (contentType_str, reply_str)
            if context_type_json == contentType_str:
                logger.info("Received %s message, return socket",
                            contentType_str)
                break
            if "[DONE]" in reply_str:
                logger.info("Received stop signal, return socket")
                break
    finally:
        await sockets.put(sock)


async def generate_stream_response(fisrt_reply: str,
                                   generator: AsyncGenerator):
    yield fisrt_reply
    async for _, reply in generator:
        yield reply


async def decode(route: str, header: dict, original_request_data: dict):
    logger.info("start decode")
    generator = execute_task_async(route, header, original_request_data,
                                   app.state.sockets_decode)
    logger.info("finish decode")

    async for contentType, reply in generator:
        logger.info("contentType: %s, reply: %s", contentType, reply)
        if context_type_json == contentType:
            return JSONResponse(reply)
        else:
            return StreamingResponse(generate_stream_response(
                reply, generator),
                                     media_type="text/event-stream")


@app.post('/v1/connect/completions')
async def chat_completions(request: Request):
    try:
        # Add the X-Request-Id header to the raw headers list
        x_request_id = str(uuid.uuid4())
        header = dict(request.headers)
        if header.get("X-Request-Id") is None:
            logger.info("add X-Request-Id: %s", x_request_id)
            header["X-Request-Id"] = x_request_id
        original_request_data = await request.json()
        logger.info("Received request: %s header: %s", original_request_data,
                    header)
        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1
        route = "/v1/completions"
        # finish prefill
        async for _ in execute_task_async(route, header, prefill_request,
                                          app.state.sockets_prefill):
            continue

        logger.info("finish prefill start decode")
        response = await decode(route, header, original_request_data)
        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        logger.error("Error occurred in disagg prefill proxy server")
        logger.error(e)
        logger.error("".join(traceback.format_exception(*exc_info)))


async def run_disagg_connector(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM Disaggregate Connector start %s %s", args,
                uvicorn_kwargs)
    logger.info(args.prefill_addr)
    app.state.port = args.port if args.port is not None else fastapi_port
    app.state.prefill_addr = (f"ipc://{args.prefill_addr}" if args.prefill_addr
                              is not None else decode_addr)
    app.state.decode_addr = (f"ipc://{args.decode_addr}"
                             if args.decode_addr is not None else decode_addr)
    logger.info(
        "start connect prefill_addr: %s decode_addr: %s zmq server port: %s",
        app.state.prefill_addr, app.state.decode_addr, app.state.port)

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)
    # init uvicorn server
    config = uvicorn.Config(app, host="0.0.0.0", port=app.state.port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # url = 'tcp://127.0.0.1:5555'
    uvicorn.run(app, host="0.0.0.0", port=fastapi_port)
