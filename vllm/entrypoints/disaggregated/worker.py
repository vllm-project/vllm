# SPDX-License-Identifier: Apache-2.0

import asyncio
import msgspec
import signal
import uvloop
from typing import Optional

import zmq
import zmq.asyncio

from vllm.inputs.data import TokensPrompt
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.disaggregated.types import PDRequest, PDResponse
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser, set_ulimit
from vllm.version import __version__ as VLLM_VERSION

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.disaggregated.worker')


async def handle_request(
    request: PDRequest,
    engine: EngineClient,                     
    socket: zmq.asyncio.Socket,
    encoder: msgspec.msgpack.Encoder,
) -> None:
    request_id = request.request_id
    try:
        # 1) Generate RequestOutputs.
        prompt: TokensPrompt = {
            "prompt_token_ids": request.prompt_token_ids}
        async for request_output in engine.generate(
            prompt=prompt,
            sampling_params=request.sampling_params,
            request_id=request_id):

            assert len(request_output.outputs) == 1, "Only support N=1 right now."
            out = request_output.outputs[0]

            # 2) Convert RequestOutput --> PDResponse.
            response = PDResponse(
                request_id=request_id,
                success=True,
                text=out.text,
                token_ids=out.token_ids,
                finish_reason=out.finish_reason,
                stop_reason=out.stop_reason,
            )
            response_bytes = encoder.encode(response)

            # 3) Send to Connector.
            logger.info("Sending: %s", request_id)
            await socket.send(response_bytes, copy=False)
            logger.info("Sent: %s", request_id)
            
    except Exception as e:
        # TODO: actual error handling.
        logger.error("Exception in Worker Routine: %s request_id: %s", e,
                     request_id)
        response = PDResponse(request_id=request_id, success=False)
        response_bytes = encoder.encode(response)
        await socket.send(response, copy=False)

async def run_server(args, engine: EngineClient):
    """Get Requests and Handle Them."""
    logger.info("P/D Worker is Ready To Recieve Requests.")

    running_requests: set[asyncio.Task] = set()
    decoder = msgspec.msgpack.Decoder(PDRequest)
    encoder = msgspec.msgpack.Encoder()
    
    ctx: Optional[zmq.asyncio.Context] = None
    try:
        # IPC Setup.
        ctx = zmq.asyncio.Context()
        from_connector = ctx.socket(zmq.constants.PULL)
        from_connector.connect(f"ipc://{args.worker_addr}")
        to_connector = ctx.socket(zmq.constants.PUSH)
        to_connector.connect(f"ipc://{args.connector_addr}")

        # Main Loop.
        while True:
            # 1) Get request from the Connector.
            pd_request_bytes = await from_connector.recv()
            pd_request = decoder.decode(pd_request_bytes)
            
            # 2) Launch a coroutine to handle the request.
            task = asyncio.create_task(handle_request(
                pd_request, engine, to_connector, encoder))
            running_requests.add(task)
            task.add_done_callback(running_requests.discard)

    except KeyboardInterrupt:
        logger.debug("Worker server loop interrupted.")

    finally:
        for task in running_requests:
            task.cancel()
        if ctx is not None:
            ctx.destroy(linger=0)


async def main(args) -> None:
    logger.info("vLLM P/D Worker Server %s", VLLM_VERSION)
    logger.info("args: %s", args)

    # Workaround to avoid footguns where uvicorn drops requests
    # with too many concurrent requests active due to ulimit.
    set_ulimit()

    # Interrupt on sigterm during initialization.
    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, signal_handler)

    args.disable_frontend_multiprocessing = False
    async with build_async_engine_client(args) as engine:
        await run_server(args, engine)

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument('--connector-addr',
                        type=str,
                        required=True,
                        help='The address of the connector.')
    parser.add_argument('--worker-addr',
                        type=str,
                        required=True,
                        help='The address of the worker.')
    AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    uvloop.run(main(args))
