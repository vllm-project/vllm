# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import signal
import traceback
import uuid
# from fastapi.lifespan import Lifespan
from asyncio import Queue
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
import uvloop
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser

# default prefill and decode addr
time_out = 180
fastapi_port = 8000
prefill_addr = "ipc://localhost:7010"
socket_prefill_num = 100
decode_addr = "ipc://localhost:7020"
socket_decode_num = 100
context_type_json = "application/json"
context_type_error = "error"

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.disagg_connector')


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
    logger.info("shutdown disagg connector")
    logger.info("term zmqctx")
    app.state.zmqctx.destroy(linger=0)


app = FastAPI(lifespan=lifespan)


# create async socket pool with num_sockets use ZMQ_DEALER
async def create_socket_pool(url: str, num_sockets: int,
                             zmqctx: zmq.asyncio.Context) -> Queue:
    sockets: Queue[zmq.Socket] = Queue()
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
    sock: zmq.Socket = await sockets.get()
    try:
        requestBody = json.dumps(request)
        headersJson = json.dumps(headers)
        logger.info("Sending requestBody: %s to %s with headers: %s",
                    requestBody, route, headersJson)
        await asyncio.wait_for(sock.send_multipart(
            [route.encode(),
             headersJson.encode(),
             requestBody.encode()]),
                               timeout=time_out)
        logger.debug("Sent end")
        while True:
            logger.debug("Waiting for reply")
            [contentType,
             reply] = await asyncio.wait_for(sock.recv_multipart(),
                                             timeout=time_out)
            contentType_str = contentType.decode()
            reply_str = reply.decode()
            logger.debug("Received result: %s, %s", contentType_str, reply_str)
            yield (contentType_str, reply_str)
            if context_type_json == contentType_str:
                logger.debug("Received %s message, return socket",
                            contentType_str)
                break
            if "[DONE]" in reply_str:
                logger.debug("Received stop signal, return socket")
                break
    except asyncio.TimeoutError:
        logger.error(traceback.format_exc())
        logger.error("Timeout, return socket: %s",
                     sock.getsockopt(zmq.IDENTITY))
        yield (context_type_error, "System Error")
    finally:
        await sockets.put(sock)


async def generate_stream_response(fisrt_reply: str,
                                   generator: AsyncGenerator):
    yield fisrt_reply
    async for _, reply in generator:
        yield reply


async def prefill(route: str, header: dict, original_request_data: dict):
    logger.info("start prefill")
    generator = execute_task_async(route, header, original_request_data,
                                   app.state.sockets_prefill)
    async for contentType, reply in generator:
        logger.debug("contentType: %s, reply: %s", contentType, reply)
        if context_type_error == contentType:
            response = JSONResponse({"error": reply})
            response.status_code = 500
            return response
    return True


async def decode(route: str, header: dict, original_request_data: dict):
    logger.info("start decode")
    generator = execute_task_async(route, header, original_request_data,
                                   app.state.sockets_decode)

    async for contentType, reply in generator:
        logger.debug("contentType: %s, reply: %s", contentType, reply)
        if context_type_error == contentType:
            response = JSONResponse({"error": reply})
            response.status_code = 500
            return response
        elif context_type_json == contentType:
            return JSONResponse(reply)
        else:
            return StreamingResponse(generate_stream_response(
                reply, generator),
                                     media_type="text/event-stream")


@app.post('/v1/completions')
async def chat_completions(request: Request):
    try:
        # Add the X-Request-Id header to the raw headers list
        x_request_id = str(uuid.uuid4())
        header = dict(request.headers)
        if header.get("X-Request-Id") is None:
            logger.info("add X-Request-Id: %s", x_request_id)
            header["X-Request-Id"] = x_request_id
        request_data = await request.json()
        logger.info("Received request: %s header: %s", request_data,
                    header)
        original_max_tokens = request_data['max_tokens']
        # change max_tokens = 1 to let it only do prefill
        request_data['max_tokens'] = 1
        route = "/v1/completions"
        # finish prefill
        try:
            prefill_response = await prefill(route, header, request_data)
            if isinstance(prefill_response, JSONResponse):
                return prefill_response
            logger.info("finish prefill start decode")
            request_data['max_tokens'] = original_max_tokens
            response = await decode(route, header, request_data)
            logger.info("finish decode")
        except Exception as e:
            logger.error("Error occurred in disagg prefill proxy server, %s",
                         e)
            response = JSONResponse({"error": {"message": str(e)}})
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
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(description="vLLM disagg zmq server.")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="The fastapi server port")
    parser.add_argument("--prefill-addr",
                        type=str,
                        required=True,
                        help="The prefill address IP:PORT")
    parser.add_argument("--decode-addr",
                        type=str,
                        required=True,
                        help="The decode address IP:PORT")

    args = parser.parse_args()

    uvloop.run(run_disagg_connector(args))