import asyncio
import pickle
import signal
from typing import Any, Coroutine

import zmq
import zmq.asyncio
from typing_extensions import Never

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger('vllm.entrypoints.openai.rpc.server')


class RPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs,
                 usage_context: UsageContext, port: int):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(async_engine_args,
                                                      usage_context)

        # Initialize context.
        self.context = zmq.asyncio.Context()

        # Init socket for readiness state.
        self.socket = self.context.socket(zmq.constants.ROUTER)
        self.socket.bind(f"tcp://localhost:{port}")

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()

    async def _send_success_message(self, identity):
        """Send message to client indicating an action was successful."""
        await self.socket.send_multipart([
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR, pickle.HIGHEST_PROTOCOL),
        ])

    async def get_model_config(self, identity):
        """Send the ModelConfig """
        model_config = await self.engine.get_model_config()

        await self.socket.send_multipart(
            [identity,
             pickle.dumps(model_config, pickle.HIGHEST_PROTOCOL)])

    async def do_log_stats(self, identity):
        await self.engine.do_log_stats()

        await self.socket.send_multipart([
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR, pickle.HIGHEST_PROTOCOL),
        ])

    async def is_server_ready(self, identity):
        await self.socket.send_multipart([
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR, pickle.HIGHEST_PROTOCOL),
        ])

    async def abort(self, identity, request: RPCAbortRequest):
        # Abort the request in the llm engine.
        await self.engine.abort(request.request_id)

        # Send confirmation to the client.
        await self.socket.send_multipart([
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR, pickle.HIGHEST_PROTOCOL),
        ])

    async def generate(self, identity, generate_request: RPCGenerateRequest):
        try:
            results_generator = self.engine.generate(
                generate_request.inputs,
                sampling_params=generate_request.sampling_params,
                request_id=generate_request.request_id)

            async for request_output in results_generator:
                await self.socket.send_multipart([
                    identity,
                    pickle.dumps(request_output, pickle.HIGHEST_PROTOCOL)
                ])

        except Exception as e:
            ### Notify client of all failures
            await self.socket.send_multipart(
                [identity, pickle.dumps(e, pickle.HIGHEST_PROTOCOL)])

    def _make_handler_coro(self, identity,
                           message) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = pickle.loads(message)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(identity, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(identity, request)

        elif isinstance(request, RPCUtilityRequest):
            if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                return self.get_model_config(identity)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(identity)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(identity)
            else:
                raise ValueError(f"Unknown RPCUtilityRequest type: {request}")

        else:
            raise ValueError(f"Unknown RPCRequest type: {request}")

    async def run_server_loop(self):
        """Inner RPC Server Loop"""

        running_tasks = set()
        while True:
            # Wait for a request.
            identity, message = await self.socket.recv_multipart()

            # Process the request async.
            task = asyncio.create_task(
                self._make_handler_coro(identity, message))

            # We need to keep around a strong reference to the task,
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)


async def run_server(server: RPCServer):
    # Put the server task into the asyncio loop.
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.run_server_loop())

    # Interruption handling.
    def signal_handler() -> None:
        # Kill the server on interrupt / terminate
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("vLLM ZMQ RPC Server was interrupted.")
    finally:
        # Clean up all resources.
        server.cleanup()


def run_rpc_server(async_engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, port: int):
    server = RPCServer(async_engine_args, usage_context, port)
    asyncio.run(run_server(server))
