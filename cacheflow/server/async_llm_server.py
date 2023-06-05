import asyncio
import time
from typing import Dict, Optional

from cacheflow.logger import init_logger
from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import AsyncServerArgs
from cacheflow.server.llm_server import LLMServer
from cacheflow.server.ray_utils import ray, initialize_cluster

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds


class AsyncLLMServer:

    def __init__(self, worker_use_ray: bool, server_use_ray: bool,
                 *args, **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.server_use_ray = server_use_ray
        if not self.server_use_ray:
            server_class = LLMServer
        elif self.worker_use_ray:
            server_class = ray.remote(num_cpus=0)(LLMServer).remote
        else:
            server_class = ray.remote(num_gpus=1)(LLMServer).remote
        self.server = server_class(*args, **kwargs)
        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_server_running = False
        self.kicking_request_id: Optional[str] = None

    async def server_step(self, kicking_request_id: Optional[str] = None):
        self.is_server_running = True
        self.kicking_request_id = kicking_request_id
        if self.server_use_ray:
            request_outputs = await self.server.step.remote()
        else:
            # Yield to the event loop to allow other coroutines to run
            # while is_server_running is True. This let the server to add new
            # requests into the queue.
            await asyncio.sleep(0)
            request_outputs = self.server.step()
        self.is_server_running = False
        self.kicking_request_id = None

        # Notify the waiting coroutines that there are new outputs ready.
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    async def generate(self, prompt: str, sampling_params: SamplingParams,
                       request_id: str) -> RequestOutput:
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # cacheflow server.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        logger.info(f"Received request {request_id}: "
                    f"prompt: {prompt!r}, "
                    f"sampling params: {sampling_params}.")

        # Add the request into the cacheflow server's waiting queue.
        if self.server_use_ray:
            await self.server.add_request.remote(
                request_id, prompt, sampling_params, arrival_time=arrival_time)
        else:
            self.server.add_request(
                request_id, prompt, sampling_params, arrival_time=arrival_time)

        # The cacheflow server does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the server to process the requests.
        while True:
            # Kick the server if the server is not running.
            if not self.is_server_running:
                await self.server_step(request_id)

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
                logger.info(f"Finished request {request_id}.")

                del self.request_outputs[request_id]
                del self.request_events[request_id]
                # Kick the server if the server is not running. This is to
                # prevent that there are still requests in server's waiting
                # queue to be executed.
                if not self.is_server_running:
                    await self.server_step()
                break

    async def abort(self, request_id: str) -> None:
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        logger.info(f"Aborted request {request_id}.")

        if self.server_use_ray:
            await self.server.abort_request.remote(request_id)
        else:
            self.server.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        # To prevent deadlock when a request is aborted while the server is
        # running.
        if self.kicking_request_id == request_id:
            self.is_server_running = False
            self.kicking_request_id = None

    @classmethod
    def from_server_args(cls, server_args: AsyncServerArgs) -> "AsyncLLMServer":
        # Create the server configs.
        server_configs = server_args.create_server_configs()
        parallel_config = server_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(
            parallel_config, server_args.server_use_ray)
        # Create the LLM server.
        server = cls(server_args.worker_use_ray,
                     server_args.server_use_ray,
                     *server_configs,
                     distributed_init_method, devices,
                     log_stats=not server_args.disable_log_stats)
        return server
