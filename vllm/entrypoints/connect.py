import json
import uvicorn
import zmq
import zmq.asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
# from fastapi.lifespan import Lifespan
from asyncio import Queue
import uuid
import signal
from vllm.logger import init_logger

# default prefill and decode url
url_prefill = "tcp://localhost:8110"
socket_prefill_num = 5
url_decode = "tcp://localhost:8220"
socket_decode_num = 5

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.connect')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create scoket pool with prefill and decode
    logger.info("start create_socket_pool")
    app.state.zmqctx = zmq.asyncio.Context()
    app.state.sockets_prefill = await create_socket_pool(app.state.prefill_addr, socket_prefill_num, zmqctx=app.state.zmqctx)
    logger.info("success create_socket_pool sockets_prefill")
    app.state.sockets_decode = await create_socket_pool(app.state.decode_addr, socket_decode_num, zmqctx=app.state.zmqctx)
    logger.info("success create_socket_pool sockets_decode")
    yield
    ## close zmq context
    logger.info("term zmqctx")
    await app.state.zmqctx.term()

app = FastAPI(lifespan=lifespan)

# create async socket pool with num_sockets use ZMQ_DEALER
async def create_socket_pool(url: str, num_sockets: int, zmqctx: zmq.asyncio.Context):
    sockets = Queue()
    for i in range(num_sockets):
        sock = zmqctx.socket(zmq.DEALER)
        identity = f"worker-{i}-{uuid.uuid4()}"
        sock.setsockopt(zmq.IDENTITY, identity.encode())
        sock.connect(url)
        logger.info(f"{identity} started at {url} {sockets.qsize()}")
        await sockets.put(sock)
    return sockets

# select a scoket and execute task
async def execute_task_async(request: dict, sockets: list):
    sock = await sockets.get()
    try:
        requestBody = json.dumps(request)
        logger.info(f"Sending requestBody: {requestBody}")
        await sock.send(requestBody.encode())
        logger.info(f"Sent end")
        while True:
            logger.info(f"Waiting for reply")
            reply = await sock.recv_multipart()
            logger.info(f"Received result: {reply}")
            yield f"data: {reply[0].decode()}\n\n"
            if "finish_reason" in reply[0].decode() and "stop" in reply[0].decode():
                logger.info(f"Received stop signal, return socket")
                yield "data: [DONE]\n\n"
                break
    finally:
        await sockets.put(sock)

@app.post('/v1/connect/completions')
async def chat_completions(request: Request):
    try:
        original_request_data = await request.json()
        logger.info(f"Received request: {original_request_data}")
        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        # finish prefill
        async for x in execute_task_async(prefill_request, app.state.sockets_prefill):
            logger.info(f"{x}")
            continue

        # return decode
        return StreamingResponse(execute_task_async(original_request_data, app.state.sockets_decode), media_type="text/event-stream")
    
    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        logger.error("Error occurred in disagg prefill proxy server")
        logger.error(e)
        logger.error("".join(traceback.format_exception(*exc_info)))
        

async def run_connect(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM Connect start %s", args)
    logger.info(f"start connect {args} {uvicorn_kwargs}")
    logger.info(args.prefill_addr)

    app.state.prefill_addr = f"tcp://{args.prefill_addr}" if args.prefill_addr is not None else url_prefill
    app.state.decode_addr =  f"tcp://{args.decode_addr}" if args.decode_addr is not None else url_decode
    logger.info(f"start connect url_prefill: {app.state.prefill_addr} url_decode: {app.state.decode_addr}")
    
    
    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)
    # init uvicorn server
    config = uvicorn.Config(app, host="0.0.0.0", port=8001)
    server = uvicorn.Server(config)
    await server.serve()

   

if __name__ == "__main__":
    # url = 'tcp://127.0.0.1:5555'
    uvicorn.run(app, host="0.0.0.0", port=8001)