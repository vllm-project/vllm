import asyncio
import pickle
import signal
from typing import Any, Coroutine, Union

import cloudpickle
import uvloop
import zmq
import zmq.asyncio
from typing_extensions import Never
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.entrypoints.openai.rpc import (VLLM_RPC_SUCCESS_STR,
                                         VLLM_RPC_ZMQ_HWM, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)

CONFIG_TYPE = Union[ModelConfig, DecodingConfig, ParallelConfig,
                    SchedulerConfig, LoRAConfig]


class AsyncEngineRPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs,
                 usage_context: UsageContext, rpc_path: str):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(
            async_engine_args, usage_context=usage_context)

        # Initialize context.
        self.context = zmq.asyncio.Context()

        # Init socket.
        self.socket: Socket = self.context.socket(zmq.constants.DEALER)
        self.socket.set_hwm(VLLM_RPC_ZMQ_HWM)
        self.socket.connect(rpc_path)

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()
        self.engine.shutdown_background_loop()
        # Clear the engine reference so that it can be GC'ed.
        del self.engine

    async def get_config(self, identity, request):
        try:
            config: CONFIG_TYPE
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

            await self.socket.send_multipart((identity, pickle.dumps(config)),
                                             copy=False)

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    async def is_tracing_enabled(self, identity):
        """Send the is_tracing_enabled flag"""
        tracing_flag = await self.engine.is_tracing_enabled()

        await self.socket.send_multipart(
            (identity, pickle.dumps(tracing_flag)))

    async def do_log_stats(self, identity):
        """Log stats and confirm success."""
        await self.engine.do_log_stats()

        await self.socket.send_multipart(
            (identity, pickle.dumps(VLLM_RPC_SUCCESS_STR)))

    async def is_server_ready(self, identity):
        """Notify the client that we are ready."""
        await self.socket.send_multipart(
            (identity, pickle.dumps(VLLM_RPC_SUCCESS_STR)))

    async def abort(self, identity, request: RPCAbortRequest):
        """Abort request and notify the client of success."""
        try:
            # Abort the request in the llm engine.
            await self.engine.abort(request.request_id)
            result: Union[str, Exception] = VLLM_RPC_SUCCESS_STR
        except Exception as e:
            result = e
        await self.socket.send_multipart((identity, pickle.dumps(result)))

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
                    (identity, pickle.dumps(request_output)), copy=False)

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    async def check_health(self, identity):
        try:
            await self.engine.check_health()
            await self.socket.send_multipart(
                (identity, pickle.dumps(VLLM_RPC_SUCCESS_STR)))

        except Exception as e:
            await self.socket.send_multipart((identity, pickle.dumps(e)),
                                             copy=False)

    async def start_profile(self, identity):
        logger.info("Starting profiler...")
        await self.engine.start_profile()
        logger.info("Profiler started.")

        await self.socket.send_multipart((
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR),
        ))

    async def stop_profile(self, identity):
        logger.info("Stopping profiler...")
        await self.engine.stop_profile()
        logger.info("Profiler stopped.")

        await self.socket.send_multipart((
            identity,
            pickle.dumps(VLLM_RPC_SUCCESS_STR),
        ))

    def _make_handler_coro(self, identity,
                           message: Frame) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = cloudpickle.loads(message.buffer)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(identity, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(identity, request)

        elif isinstance(request, RPCUtilityRequest):
            if request in [
                    RPCUtilityRequest.GET_MODEL_CONFIG,
                    RPCUtilityRequest.GET_PARALLEL_CONFIG,
                    RPCUtilityRequest.GET_DECODING_CONFIG,
                    RPCUtilityRequest.GET_SCHEDULER_CONFIG,
                    RPCUtilityRequest.GET_LORA_CONFIG
            ]:
                return self.get_config(identity, request)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(identity)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(identity)
            elif request == RPCUtilityRequest.IS_SERVER_HEALTHY:
                return self.check_health(identity)
            elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                return self.is_tracing_enabled(identity)
            elif request == RPCUtilityRequest.START_PROFILE:
                return self.start_profile(identity)
            elif request == RPCUtilityRequest.STOP_PROFILE:
                return self.stop_profile(identity)
            else:
                raise ValueError(f"Unknown RPCUtilityRequest type: {request}")

        else:
            raise ValueError(f"Unknown RPCRequest type: {request}")

    async def run_server_loop(self):
        """Inner RPC Server Loop"""

        running_tasks = set()
        while True:
            # Wait for a request.
            identity, message = await self.socket.recv_multipart(copy=False)

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
                   usage_context: UsageContext, rpc_path: str):
    server = AsyncEngineRPCServer(async_engine_args, usage_context, rpc_path)
    uvloop.run(run_server(server))
