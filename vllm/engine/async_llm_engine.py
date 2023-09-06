import asyncio
import time
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: RequestOutput) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        return result


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception as e:
        raise RuntimeError("Task finished unexpectedly.") from e
    raise RuntimeError("Task finished unexpectedly.")


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        (seq_group_metadata_list, scheduler_outputs,
         early_return) = self._schedule()
        if early_return is not None:
            return early_return

        # Execute the model.
        output = await self._run_workers_async(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )

        return self._process_model_outputs(output, scheduler_outputs)

    async def _run_workers_async(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            if self.parallel_config.worker_use_ray:
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = await asyncio.gather(*all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output


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
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        *args, *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 start_engine_loop: bool = False,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.engine = self._init_engine(*args, **kwargs)

        # Request id -> stream.
        self.request_streams: Dict[str, AsyncStream] = {}
        self.finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self.background_loop = None
        if start_engine_loop:
            self.start_background_loop()

    @property
    def is_running(self) -> bool:
        return self.background_loop is not None

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        self.background_loop = asyncio.get_event_loop().create_task(
            self.run_engine_loop())
        self.background_loop.add_done_callback(_raise_exception_on_finish)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            engine_class = ray.remote(num_gpus=1)(self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self):
        """Kick the engine to process the waiting requests."""
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            request_outputs = await self.engine.step_async()

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_streams[request_id].put(request_output)
            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Finished request {request_id}.")
                self.request_streams[request_id].finish()
                self.finished_requests.put_nowait(request_id)

        finished_request = set()
        while not self.finished_requests.empty():
            finished_request.add(self.finished_requests.get_nowait())
        await self._engine_abort(finished_request)
        for request_id in finished_request:
            del self.request_streams[request_id]

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        while True:
            await self.engine_step()
            await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> AsyncStream:
        if self.log_requests:
            logger.info(f"Received request {request_id}: "
                        f"prompt: {prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {prompt_token_ids}.")

        if request_id in self.request_streams:
            raise KeyError(f"Request {request_id} already exists.")
        stream = AsyncStream(request_id)
        self.request_streams[request_id] = stream

        # Add the request into the vLLM engine's waiting queue.
        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids=prompt_token_ids,
                                    arrival_time=arrival_time)

        return stream

    async def generate(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None) -> RequestOutput:
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

        try:
            stream = await self.add_request(request_id,
                                            prompt,
                                            sampling_params,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time)

            async for request_output in stream:
                yield request_output
        except Exception as e:
            # If there is an exception, abort the request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_streams or self.request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        if self.log_requests:
            logger.info(f"Aborted request {request_id}.")

        self.request_streams[request_id].finish()
        self.finished_requests.put_nowait(request_id)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs,
                         start_engine_loop: bool = False) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats,
                     start_engine_loop=start_engine_loop)
        return engine
