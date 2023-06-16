import asyncio
import time
from typing import Dict, List, Optional

from cacheflow.logger import init_logger
from cacheflow.outputs import RequestOutput
from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import AsyncServerArgs
from cacheflow.server.llm_server import LLMEngine
from cacheflow.server.ray_utils import ray, initialize_cluster

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        server_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        *args, *kwargs: Arguments for LLMEngine.
    """
    def __init__(self, worker_use_ray: bool, server_use_ray: bool,
                 log_requests: bool = True, *args, **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.server_use_ray = server_use_ray
        self.log_requests = log_requests
        if not self.server_use_ray:
            server_class = LLMEngine
        elif self.worker_use_ray:
            server_class = ray.remote(num_cpus=0)(LLMEngine).remote
        else:
            server_class = ray.remote(num_gpus=1)(LLMEngine).remote
        self.server = server_class(*args, **kwargs)
        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_server_running = False
        self.kicking_request_id: Optional[str] = None

    async def server_step(self, kicking_request_id: Optional[str] = None):
        """Kick the server to process the waiting requests."""
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

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None
    ) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # cacheflow server.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        if self.log_requests:
            logger.info(f"Received request {request_id}: "
                        f"prompt: {prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {prompt_token_ids}.")

        # Add the request into the cacheflow server's waiting queue.
        if self.server_use_ray:
            await self.server.add_request.remote(
                request_id, prompt, sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.server.add_request(
                request_id, prompt, sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)

        # The cacheflow server does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the server to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

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
                if self.log_requests:
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
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        if self.log_requests:
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
    def from_server_args(cls, server_args: AsyncServerArgs) -> "AsyncLLMEngine":
        """Creates an async LLM server from the server arguments."""
        # Create the server configs.
        server_configs = server_args.create_server_configs()
        parallel_config = server_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(
            parallel_config, server_args.server_use_ray)
        # Create the LLM server.
        server = cls(server_args.worker_use_ray,
                     server_args.server_use_ray,
                     not server_args.disable_log_requests,
                     *server_configs,
                     distributed_init_method, devices,
                     log_stats=not server_args.disable_log_stats)
        return server
