import asyncio
import time
from typing import Dict, Optional

import ray

from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import ServerArgs
from cacheflow.server.llm_server import LLMServer
from cacheflow.server.ray_utils import initialize_cluster
from cacheflow.utils import random_uuid

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds


class AsyncLLMServer:

    def __init__(self, server_use_ray: bool, *args, **kwargs) -> None:
        if server_use_ray:
            remote_server_class = ray.remote(num_cpus=0)(LLMServer)
        else:
            remote_server_class = ray.remote(num_gpus=1)(LLMServer)
        self.server = remote_server_class.remote(*args, **kwargs)

        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_server_running = False

    async def server_step(self):
        self.is_server_running = True
        request_outputs = await self.server.step.remote()
        self.is_server_running = False
        # Notify the waiting coroutines that there are new outputs ready.
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    async def generate(self, prompt: str, sampling_params: SamplingParams,
                       request_id: Optional[str] = None) -> RequestOutput:
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # cacheflow server.
        if request_id is None:
            request_id = random_uuid()
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        # Add the request into the cacheflow server's waiting queue.
        await self.server.add_request.remote(
            request_id, prompt, sampling_params, arrival_time=arrival_time)

        # The cacheflow server does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the server to process the requests.
        while True:
            # Kick the server if the server is not running.
            if not self.is_server_running:
                await self.server_step()

            # Wait for new output. The group_event will be set in server_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]
            yield request_output

            # Once finished, release the resources of the sequence group.
            if request_output.finished():
                del self.request_outputs[request_id]
                del self.request_events[request_id]
                # Kick the server if the server is not running. This is to
                # prevent that there are still requests in server's waiting
                # queue to be executed.
                if not self.is_server_running:
                    await self.server_step()
                break

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "AsyncLLMServer":
        # Create the server configs.
        server_configs = server_args.create_server_configs()
        parallel_config = server_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(parallel_config)
        # Create the LLM server.
        server = cls(server_args.use_ray, *server_configs,
                     distributed_init_method, devices,
                     log_stats=not server_args.disable_log_stats)
        return server
