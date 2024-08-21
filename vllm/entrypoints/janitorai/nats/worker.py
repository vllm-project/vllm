import os

import asyncio
import signal
from typing import Any, Coroutine, Union

import cloudpickle
import uvloop
import zmq
import zmq.asyncio
from typing_extensions import Never

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.entrypoints.openai.rpc import (VLLM_RPC_SUCCESS_STR,
                                         VLLM_RPC_ZMQ_HWM, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

import nats
from nats.errors import TimeoutError, NoRespondersError

import requests


logger = init_logger(__name__)



class AsyncEngineNatsWorker:

    def __init__(self, async_engine: AsyncEngineArgs):
        # lock in engine so can examine queue length
        self.engine = async_engine

        # set the max queue depth
        self.max_queue_length = os.environ.get("MAX_QUEUE_LENGTH", 10)

        # grab nats servers directly from env vars
        self.servers = os.environ.get("NATS_URL", "nats://localhost:4222").split(",")


    def cleanup(self):
        """Cleanup all resources."""
        if self.nc:
            if self.nc.is_closed:
                return
        print("Disconnecting...")
        asyncio.create_task(self.nc.close()) # is this synchronous? 
        # Clear the engine reference so that it can be GC'ed.
        del self.engine

    async def generate_handler(self, message, generate_request):
        try:
            results_generator = requests.post("http://localhost/v1/chat/completions", json=generate_request)
            async for request_output in results_generator:
                await message.respond(request_output)

        except Exception as e:
            # DO SOMETHING I HAVE NO IDEA
            # maybe do nothing because exception should be grabbed by 
            # the REST server itself?
            pass

    async def run(self):
        """inner NATS worker loop"""

        # (does this need the 'await' to properly init?)
        self.nc = await nats.connect(servers=self.servers)

        jobs = await self.nc.subscribe('v1.chat.completions', 'PRODUCTION_WORKER_GROUP')

        running_tasks = set()
        while True:
            # blip if the queue length is too high
            if self.engine.get_queue_length() >= self.max_queue_length:
                asyncio.sleep(0.1)
                continue

            async for job in jobs:
                task = asyncio.create_task(
                    self.generate_handler(job))

                # We need to keep around a strong reference to the task,
                # to avoid the task disappearing mid-execution as running tasks
                # can be GC'ed. Below is a common "fire-and-forget" tasks
                # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
                running_tasks.add(task)
                task.add_done_callback(running_tasks.discard)

