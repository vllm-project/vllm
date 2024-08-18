import asyncio
import signal
from typing import Any, Coroutine

import cloudpickle
import uvloop
import zmq
import zmq.asyncio
from typing_extensions import Never

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


class AsyncEngineRPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs,
                 usage_context: UsageContext, rpc_path: str):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(async_engine_args,
                                                      usage_context)

        # Initialize context.
        self.context = zmq.asyncio.Context()
        self.context.set(zmq.constants.MAX_SOCKETS,
                         self.context.get(zmq.constants.SOCKET_LIMIT))

        # Init socket for readiness state.
        self.socket = self.context.socket(zmq.constants.ROUTER)
        self.socket.bind(rpc_path)

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()
        self.engine.shutdown_background_loop()
        # Clear the engine reference so that it can be GC'ed.
        self.engine = None

    async def get_config(self, rpc_id, client_id, request):
        try:
            if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                config = await self.engine.get_model_config()
            elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
                config = await self.engine.get_decoding_config()
            elif request == RPCUtilityRequest.GET_LORA_CONFIG:
                config = await self.engine.get_lora_config()
            elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
                config = await self.engine.get_scheduler_config()
            elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
                config = await self.engine.get_parallel_config()
            else:
                raise ValueError("Unknown Config Request: %s", request)

            await self.socket.send_multipart(
                [rpc_id, client_id,
                 cloudpickle.dumps(config)])

        except Exception as e:
            ### Notify client of all failures
            await self.socket.send_multipart(
                [rpc_id, client_id, cloudpickle.dumps(e)])

    async def is_tracing_enabled(self, rpc_id, client_id):
        """Send the is_tracing_enabled flag"""
        tracing_flag = await self.engine.is_tracing_enabled()

        await self.socket.send_multipart(
            [rpc_id, client_id,
             cloudpickle.dumps(tracing_flag)])

    async def do_log_stats(self, rpc_id, client_id):
        """Log stats and confirm success."""
        await self.engine.do_log_stats()

        await self.socket.send_multipart(
            [rpc_id, client_id,
             cloudpickle.dumps(VLLM_RPC_SUCCESS_STR)])

    async def is_server_ready(self, rpc_id, client_id):
        """Notify the client that we are ready."""
        await self.socket.send_multipart(
            [rpc_id, client_id,
             cloudpickle.dumps(VLLM_RPC_SUCCESS_STR)])

    async def abort(self, rpc_id, client_id, request: RPCAbortRequest):
        """Abort request and notify the client of success."""
        try:
            # Abort the request in the llm engine.
            await self.engine.abort(request.request_id)
            await self.socket.send_multipart(
                [rpc_id, client_id,
                 cloudpickle.dumps(VLLM_RPC_SUCCESS_STR)])

        except Exception as e:
            await self.socket.send_multipart(
                [rpc_id, client_id, cloudpickle.dumps(e)])

    async def generate(self, rpc_id, client_id,
                       generate_request: RPCGenerateRequest):
        try:
            results_generator = self.engine.generate(
                generate_request.inputs,
                sampling_params=generate_request.sampling_params,
                request_id=generate_request.request_id,
                lora_request=generate_request.lora_request,
                trace_headers=generate_request.trace_headers,
                prompt_adapter_request=generate_request.prompt_adapter_request)

            async for request_output in results_generator:
                await self.socket.send_multipart(
                    [rpc_id, client_id,
                     cloudpickle.dumps(request_output)])

        except Exception as e:
            await self.socket.send_multipart(
                [rpc_id, client_id, cloudpickle.dumps(e)])

    async def check_health(self, rpc_id, client_id):
        try:
            await self.engine.check_health()
            await self.socket.send_multipart(
                [rpc_id, client_id,
                 cloudpickle.dumps(VLLM_RPC_SUCCESS_STR)])

        except Exception as e:
            await self.socket.send_multipart(
                [rpc_id, client_id, cloudpickle.dumps(e)])

    def _make_handler_coro(self, rpc_id, client_id,
                           message) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = cloudpickle.loads(message)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(rpc_id, client_id, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(rpc_id, client_id, request)

        elif isinstance(request, RPCUtilityRequest):
            if request in [
                    RPCUtilityRequest.GET_MODEL_CONFIG,
                    RPCUtilityRequest.GET_PARALLEL_CONFIG,
                    RPCUtilityRequest.GET_DECODING_CONFIG,
                    RPCUtilityRequest.GET_SCHEDULER_CONFIG,
                    RPCUtilityRequest.GET_LORA_CONFIG
            ]:
                return self.get_config(rpc_id, client_id, request)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(rpc_id, client_id)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(rpc_id, client_id)
            elif request == RPCUtilityRequest.IS_SERVER_HEALTHY:
                return self.check_health(rpc_id, client_id)
            elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                return self.is_tracing_enabled(rpc_id, client_id)
            else:
                raise ValueError(f"Unknown RPCUtilityRequest type: {request}")

        else:
            raise ValueError(f"Unknown RPCRequest type: {request}")

    async def run_server_loop(self):
        """Inner RPC Server Loop"""

        running_tasks = set()
        while True:
            # Wait for a request.
            # Identity of RPC Endpoint, Identity of Client, Message
            rpc_id, client_id, message = await self.socket.recv_multipart()

            # Process the request async.
            task = asyncio.create_task(
                self._make_handler_coro(rpc_id, client_id, message))

            # We need to keep around a strong reference to the task,
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)


async def run_server(server: AsyncEngineRPCServer):
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
                   usage_context: UsageContext, rpc_path: str):
    server = AsyncEngineRPCServer(async_engine_args, usage_context, rpc_path)
    uvloop.run(run_server(server))
