import asyncio
import time
from functools import partial
from typing import (AsyncIterator, Callable, Dict, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

from transformers import PreTrainedTokenizer

import vllm.envs as envs
from vllm.config import DecodingConfig, ModelConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.executor.ray_utils import initialize_ray_cluster, ray
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceStatus, ExecuteModelRequest, MultiModalData, SamplerOutput
from vllm.usage.usage_lib import UsageContext
from threading import Thread
from threading import Event, Semaphore
from queue import Queue
from functools import wraps
import traceback
import os
import time

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


class AsyncEngineDeadError(RuntimeError):
    pass

def exit_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            os._exit(1)

    return wrapper

def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs or EmbeddingRequestOutputs for a request
    that can be iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, EmbeddingRequestOutput,
                              Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> Union[RequestOutput, EmbeddingRequestOutput]:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()

    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               request_output: Union[RequestOutput,
                                                     EmbeddingRequestOutput],
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id

        self._request_streams[request_id].put(request_output)
        if request_output.finished:
            if verbose:
                logger.info("Finished request %s.", request_id)
            self.abort_request(request_id)

    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id)

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        return stream

    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)

        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler[
            virtual_engine].schedule()

        if not scheduler_outputs.is_empty():
            # Execute the model.
            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )
            output = await self.model_executor.execute_model_async(
                execute_model_req)
        else:
            output = []

        request_outputs = self._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.do_log_stats(scheduler_outputs, output)

        return request_outputs

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Stop the remote worker execution loop."""
        await self.model_executor.stop_remote_worker_execution_loop_async()

    async def encode_request_async(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(
                request_id=request_id,
                prompt=prompt,
                lora_request=lora_request)
        return prompt_token_ids

    async def add_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        params: Union[SamplingParams, PoolingParams],
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> None:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.encode_request_async(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            lora_request=lora_request)

        return self.add_request(request_id,
                                prompt=prompt,
                                params=params,
                                prompt_token_ids=prompt_token_ids,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                multi_modal_data=multi_modal_data)

    async def check_health_async(self) -> None:
        self.model_executor.check_health()


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
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

        self.mb_event = Event()
        self.sched_event = Event()
        self.sched_queue = Queue()
        self.mb_slot = Semaphore(2)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        distributed_executor_backend = (
            engine_config.parallel_config.distributed_executor_backend)

        if engine_config.device_config.device_type == "neuron":
            from vllm.executor.neuron_executor import NeuronExecutorAsync
            executor_class = NeuronExecutorAsync
        elif engine_config.device_config.device_type == "cpu":
            assert distributed_executor_backend is None, (
                "Distributed execution is not supported with the CPU backend.")
            from vllm.executor.cpu_executor import CPUExecutorAsync
            executor_class = CPUExecutorAsync
        elif distributed_executor_backend == "ray":
            initialize_ray_cluster(engine_config.parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        elif distributed_executor_backend == "mp":
            from vllm.executor.multiproc_gpu_executor import (
                MultiprocessingGPUExecutorAsync)
            executor_class = MultiprocessingGPUExecutorAsync
        else:
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        # Create the async LLM engine.
        engine = cls(
            distributed_executor_backend == "ray",
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )
        return engine

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and self._background_loop_unshielded is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None and
                                self._background_loop_unshielded is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_tokenizer(self) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.engine.get_tokenizer.remote()  # type: ignore
        else:
            return self.engine.get_tokenizer()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker()

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = kwargs["cache_config"]
            parallel_config = kwargs["parallel_config"]
            if (parallel_config.tensor_parallel_size == 1
                    and parallel_config.pipeline_parallel_size == 1):
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self, virtual_engine: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.engine.add_request.remote(  # type: ignore
                        **new_request)
                else:
                    await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if finished_requests:
            await self._engine_abort(finished_requests)

        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()  # type: ignore
        else:
            request_outputs = await self.engine.step_async(virtual_engine)

        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)

        return len(request_outputs) > 0

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)  # type: ignore
        else:
            self.engine.abort_request(request_ids)
    
    async def sched_timer_thread(self):
        print('starting timer thread')
        while True:
        #     await self._request_tracker.wait_for_new_requests()
            # print('waiting on timer event')
            time.sleep(0.01)
            # print('after on timer event')
            self.sched_event.set()
        # print('ahhh')
        # await asyncio.sleep(5)

    async def mb_watch_thread(self):
        print('starting mb watch')
        while True:
            self.mb_event.wait()
            self.mb_event.clear()

            # print('waiting on mb watch get_output')
            output = self.engine.model_executor._run_worker(0, 'get_output')
            # print('after mb watch get_output=====================')
            

            self.sched_event.set()

    async def sched_thread(self):
        virtual_engine = 0
        print('starting sched thread')
        while True:
            # print('waiting on sched event')
            self.sched_event.wait()
            self.sched_event.clear()
            # print('after watiing sched event=========')


            new_requests, finished_requests = (
                self._request_tracker.get_new_and_finished_requests())

            for new_request in new_requests:
                # Add the request into the vLLM engine's waiting queue.
                # TODO: Maybe add add_request_batch to reduce Ray overhead
                try:
                    if self.engine_use_ray:
                        await self.engine.add_request.remote(  # type: ignore
                            **new_request)
                    else:
                        await self.engine.add_request_async(**new_request)
                except ValueError as e:
                    # TODO: use a vLLM specific error for failed validation
                    self._request_tracker.process_exception(
                        new_request["request_id"],
                        e,
                        verbose=self.log_requests,
                    )

            if finished_requests:
                # print('after watiing sched event==========2=')
                await self._engine_abort(finished_requests)
                # print('after watiing sched event==========1=')
            # print('after watiing sched event==========2=')

                


            # print('before sched!!!!!!!!!!!!!!!')
            seq_group_metadata_list, scheduler_outputs = self.engine.scheduler[
                virtual_engine].schedule()
            # print('after sched!!!!!!!!!!!!!!!')

            if scheduler_outputs.has_no_output():
                # print('-------------------------has no output')
                continue
            
            for seq_group in scheduler_outputs.scheduled_seq_groups:
                for seq in seq_group.seq_group.get_seqs():
                    seq.status = SequenceStatus.PAUSED
            # if scheduler_outputs.is_empty():
            #     continue

            # print('sched thread after schedule()!!!!!!!!!!!!!')

            # self.mb_slot.acquire()
            # print('sched thread after put')
            if not scheduler_outputs.is_empty():
                # print('sched thread not empty')
                # Execute the model.
                execute_model_req = ExecuteModelRequest(
                    seq_group_metadata_list=seq_group_metadata_list,
                    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                    blocks_to_copy=scheduler_outputs.blocks_to_copy,
                    virtual_engine=virtual_engine,
                    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                    running_queue_size=scheduler_outputs.running_queue_size,
                )
                self.mb_event.set()
                # print('enqueue')
                # self.engine.model_executor._run_workers('enqueue', execute_model_req)
                # print('after watiing sched event==========4=')
                # import pdb; pdb.set_trace()
                # output = await self.engine.model_executor.execute_model_async(
                #     execute_model_req)
                self.engine.model_executor._run_workers('enqueue', execute_model_req)
                # print('after watiing sched event==========5=')
            self.sched_queue.put((seq_group_metadata_list, scheduler_outputs, virtual_engine))
            # else:
            #     print('empty')
            #     print(seq_group_metadata_list)
            #     print(scheduler_outputs)

                # print('after enqueue')
            # virtual_engine = (virtual_engine + 1) % 2

    async def output_loop(self):
        print('output_loop')
        while True:
            # print('sched wait output_loop')
            # await self.sched_event.wait()
            # self.sched_event.clear()
            seq_group_metadata_list, scheduler_outputs, virtual_engine = self.sched_queue.get()
            # print('|||||||||||sched wait output_loop')

            if not scheduler_outputs.is_empty():
                # print('|||||||||||| before get_output')
                output = self.engine.model_executor._run_worker(1, 'get_output')
            # if not scheduler_outputs.is_empty():
                # print('|||||||||||| after get_output')
                # print(output)
            else:
                output = []


            # import pdb; pdb.set_trace()
            # for seq_group in scheduler_outputs.scheduled_seq_groups:
            #     for seq in seq_group.seq_group.get_seqs():
            #         seq.status = SequenceStatus.RUNNING
            request_outputs = self.engine._process_model_outputs(
                output, scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
            
                
            if not scheduler_outputs.is_empty():
                self.engine.scheduler[virtual_engine].complete_mb()
            # print('|||||||||||| after process model') 

            # Log stats.
            await self.do_log_stats(scheduler_outputs, output)
            # print('|||||||||||| after log stats') 

            # return request_outputs


            # Put the outputs into the corresponding streams.
            for request_output in request_outputs:
                self._request_tracker.process_request_output(
                    request_output, verbose=self.log_requests)
            # print("DONEEEEEEEEEEEEEEEE")
            # self.mb_slot.release()
            self.sched_event.set()

            # return len(request_outputs) > 0
    
    @exit_on_error
    def run_sched(self, func):
        loop = asyncio.new_event_loop()
        t = loop.run_until_complete(func())
        assert False
        print('got running loop')
        #loop.run_forever()
        #await asyncio.gather(t)

    async def run_engine_loop(self):
        # loop = asyncio.get_event_loop()
        # import pdb; pdb.set_trace()
        # print('starting driver threads in loop')
        # await self.engine.model_executor._start_driver_execution_loop()
        # print('done starting driver threads in loop')
        sched_thread = Thread(target=self.run_sched, args=(self.sched_thread,), daemon=True)
        output_thread = Thread(target=self.run_sched, args=(self.output_loop,), daemon=True)
        timer_thread = Thread(target=self.run_sched, args=(self.sched_timer_thread,), daemon=True)
        mb_thread = Thread(target=self.run_sched, args=(self.mb_watch_thread,), daemon=True)
        sched_thread.start()
        output_thread.start()
        timer_thread.start()
        mb_thread.start()
        await asyncio.sleep(5000)
        #while True:
           # pass
        return

        # sched_task = asyncio.create_task(self.sched_thread())
        # output_task = asyncio.create_task(self.output_loop())
        # timer_task = asyncio.create_task(self.sched_timer_thread())
        # mb_task = asyncio.create_task(self.mb_watch_thread())
        sched_task = self.sched_thread()
        output_task = self.output_loop()
        timer_task = self.sched_timer_thread()
        mb_task = self.mb_watch_thread()
        # print('before gather')
        await asyncio.gather(sched_task, output_task, timer_task, mb_task)
        assert False
        print('after gather')
        while True:
            print('watig new req    ')
            await self._request_tracker.wait_for_new_requests()
            print('got watig new req    ')
            await asyncio.sleep(5)
            self.sched_event.set()
            # await 

        
        return 
        output_task = self.output_loop()
        print('running ofreve')
        # loop.run_forever()
        await asyncio.gather(sched_task, timer_task, output_task)
        # while True: # print('hi')
        #     pass
        print('wtf')
        return

        # if self.engine_use_ray:
        #     pipeline_parallel_size = 1  # type: ignore
        # else:
        #     pipeline_parallel_size = \
        #         self.engine.parallel_config.pipeline_parallel_size
        # has_requests_in_progress = [False] * pipeline_parallel_size
        # while True:
        #     if not any(has_requests_in_progress):
        #         logger.debug("Waiting for new requests...")
        #         # Stop the execute model loop in parallel workers until there
        #         # are more requests to process. This avoids waiting
        #         # indefinitely in torch.distributed ops which may otherwise
        #         # timeout, and unblocks the RPC thread in the workers so that
        #         # they can process any other queued control plane messages,
        #         # such as add/remove lora adapters.
        #         if self.engine_use_ray:
        #             await (self.engine.stop_remote_worker_execution_loop.
        #                    remote()  # type: ignore
        #                    )
        #         else:
        #             await self.engine.stop_remote_worker_execution_loop_async()
        #         await self._request_tracker.wait_for_new_requests()
        #         logger.debug("Got new requests!")
        #         requests_in_progress = [
        #             asyncio.create_task(
        #                 asyncio.wait_for(self.engine_step(ve),
        #                                  ENGINE_ITERATION_TIMEOUT_S))
        #             for ve in range(pipeline_parallel_size)
        #         ]
        #         has_requests_in_progress = [True] * pipeline_parallel_size

        #     # Abort if iteration takes too long due to unrecoverable errors
        #     # (eg. NCCL timeouts).
        #     try:
        #         done, _ = await asyncio.wait(
        #             requests_in_progress, return_when=asyncio.FIRST_COMPLETED)
        #         for task in done:
        #             result = task.result()
        #             virtual_engine = requests_in_progress.index(task)
        #             if self.engine_use_ray:
        #                 has_unfinished_requests = (
        #                     await (self.engine.
        #                            has_unfinished_requests_for_virtual_engine.
        #                            remote(  # type: ignore
        #                                virtual_engine)))
        #             else:
        #                 has_unfinished_requests = (
        #                     self.engine.
        #                     has_unfinished_requests_for_virtual_engine(
        #                         virtual_engine))
        #             if result or has_unfinished_requests:
        #                 requests_in_progress[
        #                     virtual_engine] = asyncio.create_task(
        #                         asyncio.wait_for(
        #                             self.engine_step(virtual_engine),
        #                             ENGINE_ITERATION_TIMEOUT_S))
        #                 has_requests_in_progress[virtual_engine] = True
        #             else:
        #                 has_requests_in_progress[virtual_engine] = False
        #     except asyncio.TimeoutError as exc:
        #         logger.error(
        #             "Engine iteration timed out. This should never happen!")
        #         self.set_errored(exc)
        #         raise
        #     await asyncio.sleep(0)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        params: Union[SamplingParams, PoolingParams],
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncStream:
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:self.
                                                              max_log_len]
            logger.info(
                "Received request %s: prompt: %r, "
                "params: %s, prompt_token_ids: %s, "
                "lora_request: %s.", request_id, shortened_prompt, params,
                shortened_token_ids, lora_request)

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if arrival_time is None:
            arrival_time = time.time()

        if self.engine_use_ray:
            prompt_token_ids = await (
                self.engine.encode_request_async.remote(  # type: ignore
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    lora_request=lora_request))
        else:
            prompt_token_ids = await self.engine.encode_request_async(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                lora_request=lora_request)

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            params=params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
        )

        return stream

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None
    ) -> AsyncIterator[RequestOutput]:
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
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data per request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in self.process_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids,
                lora_request,
                multi_modal_data,
        ):
            yield output

    async def encode(
        self,
        prompt: Optional[str],
        pooling_params: PoolingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None
    ) -> AsyncIterator[EmbeddingRequestOutput]:
        """Generate outputs for a request from an embedding model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data per request.

        Yields:
            The output `EmbeddingRequestOutput` objects from the LLMEngine
            for the request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "input": "What is LLM?",
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.encode(
            >>>    example_input["input"],
            >>>    PoolingParams(),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        async for output in self.process_request(
                request_id,
                prompt,
                pooling_params,
                prompt_token_ids,
                lora_request,
                multi_modal_data,
        ):
            yield output

    async def process_request(
        self,
        request_id: str,
        prompt: Optional[str],
        params: Union[SamplingParams, PoolingParams],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncIterator[Union[RequestOutput, EmbeddingRequestOutput]]:
        """Common logic to process requests with SamplingParams or
        PoolingParams."""
        arrival_time = time.time()

        stream = await self.add_request(
            request_id,
            prompt,
            params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
        )

        try:
            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()  # type: ignore
        else:
            return self.engine.get_model_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_decoding_config.remote(  # type: ignore
            )
        else:
            return self.engine.get_decoding_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote(  # type: ignore
                scheduler_outputs, model_output)
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()  # type: ignore
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)
