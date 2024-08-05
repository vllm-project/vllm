import asyncio
import signal
from typing import Any, Coroutine

import cloudpickle
import zmq
import zmq.asyncio
from typing_extensions import Never

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_RPC_HEALTHY_STR,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


class AsyncEngineRPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs,
                 usage_context: UsageContext, port: int):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(async_engine_args,
                                                      usage_context)

        # Initialize context.
        self.context = zmq.asyncio.Context()

        # Init socket for readiness state.
        self.socket = self.context.socket(zmq.constants.ROUTER)
        # Note numeric form of localhost should be used for zmq bind(),
        # see https://stackoverflow.com/a/8958414
        self.socket.bind(f"tcp://127.0.0.1:{port}")

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()

    async def get_model_config(self, identity):
        """Send the ModelConfig"""
        model_config = await self.engine.get_model_config()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(model_config)])

    async def get_decoding_config(self, identity):
        """Send the DecodingConfig"""
        decoding_config = await self.engine.get_decoding_config()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(decoding_config)])

    async def get_lora_config(self, identity):
        lora_config = await self.engine.get_lora_config()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(lora_config)])

    async def get_scheduler_config(self, identity):
        """Send the SchedulerConfig"""
        parallel_config = await self.engine.get_scheduler_config()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(parallel_config)])

    async def get_parallel_config(self, identity):
        """Send the ParallelConfig"""
        parallel_config = await self.engine.get_parallel_config()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(parallel_config)])

    async def is_tracing_enabled(self, identity):
        """Send the is_tracing_enabled flag"""
        tracing_flag = await self.engine.is_tracing_enabled()

        await self.socket.send_multipart(
            [identity, cloudpickle.dumps(tracing_flag)])

    async def do_log_stats(self, identity):
        """Log stats and confirm success."""
        await self.engine.do_log_stats()

        await self.socket.send_multipart([
            identity,
            cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
        ])

    async def is_server_ready(self, identity):
        """Notify the client that we are ready."""
        await self.socket.send_multipart([
            identity,
            cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
        ])

    async def abort(self, identity, request: RPCAbortRequest):
        """Abort request and notify the client of success."""
        # Abort the request in the llm engine.
        await self.engine.abort(request.request_id)

        # Send confirmation to the client.
        await self.socket.send_multipart([
            identity,
            cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
        ])

    async def generate(self, identity, generate_request: RPCGenerateRequest):
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
                    [identity, cloudpickle.dumps(request_output)])

        except Exception as e:
            ### Notify client of all failures
            await self.socket.send_multipart([identity, cloudpickle.dumps(e)])

    async def check_health(self, identity):
        try:
            await self.engine.check_health()
            await self.socket.send_multipart(
                [identity, cloudpickle.dumps(VLLM_RPC_HEALTHY_STR)])
        except Exception as e:
            await self.socket.send_multipart([identity, cloudpickle.dumps(e)])

    def _make_handler_coro(self, identity,
                           message) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = cloudpickle.loads(message)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(identity, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(identity, request)

        elif isinstance(request, RPCUtilityRequest):
            if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                return self.get_model_config(identity)
            elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
                return self.get_parallel_config(identity)
            elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
                return self.get_decoding_config(identity)
            elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
                return self.get_scheduler_config(identity)
            elif request == RPCUtilityRequest.GET_LORA_CONFIG:
                return self.get_lora_config(identity)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(identity)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(identity)
            elif request == RPCUtilityRequest.CHECK_HEALTH:
                return self.check_health(identity)
            elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                return self.is_tracing_enabled(identity)
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
                   usage_context: UsageContext, port: int):
    server = AsyncEngineRPCServer(async_engine_args, usage_context, port)
    asyncio.run(run_server(server))
