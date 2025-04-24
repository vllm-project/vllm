# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import time
import weakref
from functools import partial
from typing import (Any, AsyncGenerator, Callable, Coroutine, Dict, Iterable,
                    List, Mapping, Optional, Set, Tuple, Type, Union, overload)
from weakref import ReferenceType

from typing_extensions import deprecated

import vllm.envs as envs
from vllm.config import (DecodingConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.core.scheduler import SchedulerOutputs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_timeout import asyncio_timeout
from vllm.engine.llm_engine import LLMEngine, SchedulerOutputState
from vllm.engine.metrics_types import StatLoggerBase
from vllm.engine.protocol import EngineClient
from vllm.executor.executor_base import ExecutorBase
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding import (
    get_guided_decoding_logits_processor)
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import ExecuteModelRequest
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, deprecate_kwargs, weak_bind

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S


class AsyncEngineDeadError(RuntimeError):
    pass


def _log_task_completion(task: asyncio.Task,
                         error_callback: Callable[[Exception], None]) -> None:
    """This function is only intended for the `engine.run_engine_loop()` task.

    In particular, that task runs a `while True` loop that can only exit if
    there is an exception.
    """

    exception = None
    try:
        return_value = task.result()
        raise AssertionError(
            f"The engine background task should never finish without an "
            f"exception. {return_value}")
    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        logger.info("Engine is gracefully shutting down.")
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            "Task finished unexpectedly. This should never happen! "
            "Please open an issue on GitHub. See stack trace above for the "
            "actual cause.") from e


STOP_ITERATION = Exception()  # Sentinel


class AsyncStream:
    """A stream of RequestOutputs or PoolingRequestOutputs for a request
    that can be iterated over asynchronously via an async generator."""

    def __init__(self, request_id: str, cancel: Callable[[str], None]) -> None:
        self.request_id = request_id
        self._cancel = cancel
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, PoolingRequestOutput,
                              Exception]) -> None:
        if not self._finished:
            self._queue.put_nowait(item)

    def finish(
        self,
        exception: Optional[Union[BaseException, Type[BaseException]]] = None,
    ) -> None:
        if not self._finished:
            self._finished = True
            self._queue.put_nowait(
                exception if self._is_raisable(exception) else STOP_ITERATION)

    @property
    def finished(self) -> bool:
        return self._finished

    async def generator(
        self
    ) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        try:
            while True:
                result = await self._queue.get()
                if self._is_raisable(result):
                    if result == STOP_ITERATION:
                        return
                    raise result
                yield result
        except GeneratorExit:
            self._cancel(self.request_id)
            raise asyncio.CancelledError from None

    @staticmethod
    def _is_raisable(value: Any):
        return isinstance(value, BaseException) or \
                (isinstance(value, type) and \
                 issubclass(value, BaseException))


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._aborted_requests: asyncio.Queue[str] = asyncio.Queue()
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
            self.abort_request(request_id, exception=exc)
        else:
            # NB: tuple() used here because self.abort_request pops the stream
            # out of self._request_streams, so we can't iterate on it directly
            for rid in tuple(self._request_streams.keys()):
                self.abort_request(rid, exception=exc)

    def process_request_output(self,
                               request_output: Union[RequestOutput,
                                                     PoolingRequestOutput],
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        finished = request_output.finished

        if finished:
            stream = self._request_streams.pop(request_id, None)
        else:
            stream = self._request_streams.get(request_id)
        # Guard against a KeyError which can occur if the request was aborted
        # while the output was generated
        if stream is not None:
            stream.put(request_output)
            if finished:
                stream.finish()

        if verbose and finished:
            logger.info("Finished request %s.", request_id)

    def process_exception(self,
                          request_id: str,
                          exception: BaseException,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        if verbose:
            logger.info("Finished request %s.", request_id)
        self.abort_request(request_id, exception=exception)

    def add_request(self,
                    request_id: str,
                    *,
                    verbose: bool = False,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")

        abort_request = partial(self.abort_request, verbose=verbose)
        stream = AsyncStream(request_id, abort_request)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))

        self.new_requests_event.set()

        if verbose:
            logger.info("Added request %s.", request_id)

        return stream

    def abort_request(self,
                      request_id: str,
                      *,
                      exception: Optional[Union[BaseException,
                                                Type[BaseException]]] = None,
                      verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info("Aborted request %s.", request_id)

        self._aborted_requests.put_nowait(request_id)

        stream = self._request_streams.pop(request_id, None)
        if stream is not None:
            stream.finish(exception=exception)

    def get_new_and_aborted_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._aborted_requests.empty():
            request_id = self._aborted_requests.get_nowait()
            finished_requests.add(request_id)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            request_id = stream.request_id
            if request_id in finished_requests:
                # The request has already been aborted.
                stream.finish(asyncio.CancelledError)
                finished_requests.discard(request_id)
            else:
                self._request_streams[request_id] = stream
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def step_async(
        self, virtual_engine: int
    ) -> List[Union[RequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        # these are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):

            # Schedule iteration
            (seq_group_metadata_list, scheduler_outputs,
             allow_async_output_proc
             ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            if not scheduler_outputs.is_empty():
                # this will cause mamba_cache/minimax_cache failed
                # to release finished_requests_ids of the last steps
                finished_requests_ids = self.scheduler[
                    virtual_engine].get_and_reset_finished_requests_ids()

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step
                    and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine, seq_group_metadata_list, scheduler_outputs,
                    allow_async_output_proc)
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():

            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = \
                self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                virtual_engine=virtual_engine,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            # Execute the model.
            outputs = await self.model_executor.execute_model_async(
                execute_model_req)

            # we need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # Clear the cache if we have finished all the steps
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[
                    virtual_engine] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = False if not seq_group_metadata_list \
                else seq_group_metadata_list[0].state.num_steps == 1

            ctx.append_output(outputs=outputs,
                              seq_group_metadata_list=seq_group_metadata_list,
                              scheduler_outputs=scheduler_outputs,
                              is_async=allow_async_output_proc,
                              is_last_step=True,
                              is_first_step_output=is_first_step_output)

            if outputs and allow_async_output_proc:
                assert len(
                    outputs
                ) == 1, "Async postprocessor expects only a single output set"
                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)

        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

        return ctx.request_outputs

    async def stop_remote_worker_execution_loop_async(self) -> None:
        """Stop the remote worker execution loop."""
        await self.model_executor.stop_remote_worker_execution_loop_async()

    async def get_tokenizer_async(self,
                                  lora_request: Optional[LoRARequest] = None
                                  ) -> AnyTokenizer:
        return await (
            self.get_tokenizer_group().get_lora_tokenizer_async(lora_request))

    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    async def add_request_async(
        self,
        request_id: str,
        *,
        inputs: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @overload
    async def add_request_async(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    async def add_request_async(
            self,
            request_id: str,
            prompt: Optional[PromptType] = None,
            params: Optional[Union[SamplingParams, PoolingParams]] = None,
            arrival_time: Optional[float] = None,
            lora_request: Optional[LoRARequest] = None,
            trace_headers: Optional[Mapping[str, str]] = None,
            prompt_adapter_request: Optional[PromptAdapterRequest] = None,
            priority: int = 0,
            *,
            inputs: Optional[PromptType] = None,  # DEPRECATED
    ) -> None:
        """Async version of :meth:`add_request`."""
        if inputs is not None:
            prompt = inputs
        assert prompt is not None and params is not None

        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                             "not enabled!")
        if priority != 0 and not self.scheduler_config.policy == "priority":
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")
        if arrival_time is None:
            arrival_time = time.time()

        if self.tokenizer is not None:
            tokenizer = await self.get_tokenizer_async(lora_request)
            self._validate_token_prompt(prompt, tokenizer=tokenizer)

        preprocessed_inputs = await self.input_preprocessor.preprocess_async(
            prompt,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )
        processed_inputs = self.input_processor(preprocessed_inputs)

        if isinstance(params, SamplingParams) and \
            params.guided_decoding is not None:
            # Guided decoding has an async implementation for building logits
            # processors in a separate threadpool.
            # We want to invoke that here instead of using the blocking
            # implementation in the LLMEngine
            params = await build_guided_decoding_logits_processor_async(
                sampling_params=params,
                tokenizer=await self.get_tokenizer_async(lora_request),
                default_guided_backend=self.decoding_config.
                guided_decoding_backend,
                reasoning_backend=self.decoding_config.reasoning_backend,
                model_config=self.model_config)

        self._add_processed_request(
            request_id=request_id,
            processed_inputs=processed_inputs,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            trace_headers=trace_headers,
            priority=priority,
        )

    async def check_health_async(self) -> None:
        self.model_executor.check_health()

    async def collective_rpc_async(self,
                                   method: str,
                                   timeout: Optional[float] = None,
                                   args: tuple = (),
                                   kwargs: Optional[dict] = None):
        raise NotImplementedError


async def build_guided_decoding_logits_processor_async(
        sampling_params: SamplingParams, tokenizer: AnyTokenizer,
        default_guided_backend: str, reasoning_backend: Optional[str],
        model_config: ModelConfig) -> SamplingParams:
    """Constructs logits processors based on the guided_decoding,
    logits_bias, and allowed_token_ids fields in sampling_params. Deletes
    those fields and adds the constructed logits processors to the
    logits_processors field. Modifies sampling params in-place and returns
    the modified sampling params."""
    if sampling_params.guided_decoding is None:
        return sampling_params

    # Defensively copy sampling params since guided decoding logits
    # processors can have different state for each request
    sampling_params = copy.copy(sampling_params)
    guided_decoding = sampling_params.guided_decoding

    logger.debug(
        "Building guided decoding logits processor. "
        "guided_decoding: %s%s", guided_decoding,
        f", reasoning_backend: {reasoning_backend}"
        if reasoning_backend is not None else "")

    guided_decoding.backend = guided_decoding.backend or default_guided_backend

    processor = await get_guided_decoding_logits_processor(
        guided_params=guided_decoding,
        tokenizer=tokenizer,
        reasoning_backend=reasoning_backend,
        model_config=model_config)

    if processor:
        if sampling_params.logits_processors is None:
            sampling_params.logits_processors = []
        sampling_params.logits_processors.append(processor)

    # Unset guided decoding params after constructing the lp from them
    sampling_params.guided_decoding = None

    return sampling_params


class AsyncLLMEngine(EngineClient):
    """An asynchronous wrapper for :class:`LLMEngine`.

    This class is used to wrap the :class:`LLMEngine` class to make it
    asynchronous. It uses asyncio to create a background loop that keeps
    processing incoming requests. The :class:`LLMEngine` is kicked by the
    generate method when there are requests in the waiting queue. The generate
    method yields the outputs from the :class:`LLMEngine` to the caller.

    Args:
        log_requests: Whether to log the requests.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for :class:`LLMEngine`.
        **kwargs: Arguments for :class:`LLMEngine`.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 *args,
                 log_requests: bool = True,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        if envs.VLLM_USE_V1:
            raise ValueError(
                "Using V0 AsyncLLMEngine, but envs.VLLM_USE_V1=True. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        self.log_requests = log_requests
        self.engine = self._engine_class(*args, **kwargs)

        # This ensures quick processing of request outputs
        # so the append to asyncio queues is not delayed,
        # especially for multi-step.
        self.use_process_request_outputs_callback = (
            self.engine.model_config.use_async_output_proc)

        if self.use_process_request_outputs_callback:
            self.engine.process_request_outputs_callback = \
                weak_bind(self.process_request_outputs)

        self.background_loop: Optional[asyncio.Future] = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded: Optional[asyncio.Task] = None
        self.start_engine_loop = start_engine_loop
        self._errored_with: Optional[BaseException] = None

        # Lazy initialized fields
        self._request_tracker: RequestTracker

    def __del__(self):
        if rt := getattr(self, "request_tracker", None):
            # Wake up engine loop so that it will exit cleanly
            rt.new_requests_event.set()

    @classmethod
    def _get_executor_cls(cls,
                          engine_config: VllmConfig) -> Type[ExecutorBase]:
        return LLMEngine._get_executor_cls(engine_config)

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[dict[str, StatLoggerBase]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "AsyncLLMEngine":
        """Create an AsyncLLMEngine from the EngineArgs."""

        return cls(
            vllm_config=vllm_config,
            executor_class=cls._get_executor_cls(vllm_config),
            start_engine_loop=start_engine_loop,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""

        vllm_config = engine_args.create_engine_config(usage_context)

        async_engine_cls = cls
        if envs.VLLM_USE_V1:
            from vllm.v1.engine.async_llm import AsyncLLM as V1AsyncLLMEngine
            async_engine_cls = V1AsyncLLMEngine

        return async_engine_cls.from_vllm_config(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            disable_log_stats=engine_args.disable_log_stats,
            disable_log_requests=engine_args.disable_log_requests,
        )

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

    @property
    def dead_error(self) -> BaseException:
        return AsyncEngineDeadError(
            "Background loop is not running. If it was running, "
            "inspect the output to find the stacktrace of the "
            "error that caused the background loop to stop "
            "(AsyncEngineDeadError).")

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.engine.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        return await self.engine.get_tokenizer_async(lora_request)

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
        ).create_task(self.run_engine_loop(weakref.ref(self)))
        self._background_loop_unshielded.add_done_callback(
            partial(_log_task_completion, error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def shutdown_background_loop(self) -> None:
        """
        Shut down the background loop.

        This method needs to be called during cleanup to remove
        references to `self` and properly GC the resources held
        by the async LLM engine (e.g., the executors as well as
        their resources).
        """
        if self._background_loop_unshielded is not None:
            self._background_loop_unshielded.cancel()
            self._background_loop_unshielded = None
        self.background_loop = None

    async def engine_step(self, virtual_engine: int) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""

        new_requests, aborted_requests = (
            self._request_tracker.get_new_and_aborted_requests())

        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            try:
                await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )

        if aborted_requests:
            await self._engine_abort(aborted_requests)

        request_outputs = await self.engine.step_async(virtual_engine)

        # Put the outputs into the corresponding streams.
        # If used as a callback, then already invoked inside
        # LLMEngine's _process_model_outputs
        if not self.use_process_request_outputs_callback:
            all_finished = self.process_request_outputs(request_outputs)
        else:
            # For callback case, we only need to detect when all
            # requests are finished
            all_finished = all(request_output.finished
                               for request_output in request_outputs)

        return not all_finished

    def process_request_outputs(self, request_outputs) -> bool:
        # Put the outputs into the corresponding streams.
        all_finished = True
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                request_output, verbose=self.log_requests)
            all_finished = all_finished and request_output.finished

        return all_finished

    async def _engine_abort(self, request_ids: Iterable[str]):
        self.engine.abort_request(request_ids)

    @staticmethod
    async def run_engine_loop(engine_ref: ReferenceType):
        """We use a weakref to the engine so that the running loop
        doesn't prevent the engine being garbage collected."""
        engine: Optional[AsyncLLMEngine] = engine_ref()
        if not engine:
            return

        pipeline_parallel_size = \
                engine.engine.parallel_config.pipeline_parallel_size
        has_requests_in_progress = [False] * pipeline_parallel_size
        while True:
            if not any(has_requests_in_progress):
                logger.debug("Waiting for new requests...")
                # Stop the execute model loop in parallel workers until there
                # are more requests to process. This avoids waiting
                # indefinitely in torch.distributed ops which may otherwise
                # timeout, and unblocks the RPC thread in the workers so that
                # they can process any other queued control plane messages,
                # such as add/remove lora adapters.
                await engine.engine.stop_remote_worker_execution_loop_async()
                request_tracker = engine._request_tracker
                # Allow engine to be garbage collected while
                # waiting for new requests
                del engine
                await asyncio.sleep(0)
                if engine_ref() is None:
                    return
                await request_tracker.wait_for_new_requests()
                engine = engine_ref()
                if not engine:
                    return
                logger.debug("Got new requests!")
                requests_in_progress = [
                    asyncio.create_task(engine.engine_step(ve))
                    for ve in range(pipeline_parallel_size)
                ]
                has_requests_in_progress = [True] * pipeline_parallel_size

            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                async with asyncio_timeout(ENGINE_ITERATION_TIMEOUT_S):
                    done, _ = await asyncio.wait(
                        requests_in_progress,
                        return_when=asyncio.FIRST_COMPLETED)
                    for _ in range(pipeline_parallel_size):
                        await asyncio.sleep(0)
                for task in done:
                    result = task.result()
                    virtual_engine = requests_in_progress.index(task)
                    has_unfinished_requests = (
                        engine.engine.
                        has_unfinished_requests_for_virtual_engine(
                            virtual_engine))
                    if result or has_unfinished_requests:
                        requests_in_progress[virtual_engine] = (
                            asyncio.create_task(
                                engine.engine_step(virtual_engine)))
                        has_requests_in_progress[virtual_engine] = True
                    else:
                        has_requests_in_progress[virtual_engine] = False
            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                engine.set_errored(exc)
                raise
            await asyncio.sleep(0)

    # This method does not need to be async, but kept that way
    # for backwards compatibility.
    @overload
    @deprecated("'inputs' will be renamed to 'prompt")
    def add_request(
        self,
        request_id: str,
        *,
        inputs: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> Coroutine[None, None, AsyncGenerator[Union[
            RequestOutput, PoolingRequestOutput], None]]:
        ...

    @overload
    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> Coroutine[None, None, AsyncGenerator[Union[
            RequestOutput, PoolingRequestOutput], None]]:
        ...

    @deprecate_kwargs(
        "inputs",
        additional_message="Please use the 'prompt' parameter instead.",
    )
    async def add_request(
        self,
        request_id: str,
        prompt: Optional[PromptType] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        *,
        inputs: Optional[PromptType] = None,  # DEPRECATED
    ) -> AsyncGenerator[Union[RequestOutput, PoolingRequestOutput], None]:
        if inputs is not None:
            prompt = inputs
        assert prompt is not None and params is not None

        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")

        if (priority != 0
                and not self.engine.scheduler_config.policy == "priority"):
            raise ValueError(f"Got priority {priority} but "
                             "Priority scheduling is not enabled.")

        stream = self._request_tracker.add_request(
            request_id,
            verbose=self.log_requests,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time or time.time(),
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

        return stream.generator()

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            prompt_adapter_request: Prompt Adapter request to use
                                            for generation, if any.
            priority: The priority of the request.
                Only applicable with priority scheduling.

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
            >>> # note that engine_args here is AsyncEngineArgs instance
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
        try:
            async for output in await self.add_request(
                    request_id,
                    prompt,
                    sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    prompt_adapter_request=prompt_adapter_request,
                    priority=priority,
            ):
                yield LLMEngine.validate_output(output, RequestOutput)
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt to the LLM. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each input.
            pooling_params: The pooling parameters of the request.
            request_id: The unique id of the request.
            lora_request: LoRA request to use for generation, if any.
            trace_headers: OpenTelemetry trace headers.
            priority: The priority of the request.
                Only applicable with priority scheduling.

        Yields:
            The output `PoolingRequestOutput` objects from the LLMEngine
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
            >>> # note that engine_args here is AsyncEngineArgs instance
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
        try:
            async for output in await self.add_request(
                    request_id,
                    prompt,
                    pooling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
            ):
                yield LLMEngine.validate_output(output, PoolingRequestOutput)
        except asyncio.CancelledError:
            await self.abort(request_id)
            raise

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
                                            exception=asyncio.CancelledError,
                                            verbose=self.log_requests)

    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        return self.engine.get_vllm_config()

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        return self.engine.get_model_config()

    async def get_parallel_config(self) -> ParallelConfig:
        """Get the parallel configuration of the vLLM engine."""
        return self.engine.get_parallel_config()

    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        return self.engine.get_decoding_config()

    async def get_scheduler_config(self) -> SchedulerConfig:
        """Get the scheduling configuration of the vLLM engine."""
        return self.engine.get_scheduler_config()

    async def get_lora_config(self) -> LoRAConfig:
        """Get the lora configuration of the vLLM engine."""
        return self.engine.get_lora_config()

    async def do_log_stats(
            self,
            scheduler_outputs: Optional[SchedulerOutputs] = None,
            model_output: Optional[List[SamplerOutput]] = None) -> None:
        self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        await self.engine.check_health_async()
        logger.debug("Health check took %fs", time.perf_counter() - t)

    async def is_tracing_enabled(self) -> bool:
        return self.engine.is_tracing_enabled()

    def add_logger(self, logger_name: str, logger: StatLoggerBase) -> None:
        self.engine.add_logger(logger_name=logger_name, logger=logger)

    def remove_logger(self, logger_name: str) -> None:
        self.engine.remove_logger(logger_name=logger_name)

    async def start_profile(self) -> None:
        self.engine.start_profile()

    async def stop_profile(self) -> None:
        self.engine.stop_profile()

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        self.engine.reset_prefix_cache(device)

    async def sleep(self, level: int = 1) -> None:
        self.engine.sleep(level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        self.engine.wake_up(tags)

    async def is_sleeping(self) -> bool:
        return self.engine.is_sleeping()

    async def add_lora(self, lora_request: LoRARequest) -> None:
        self.engine.add_lora(lora_request)

    async def collective_rpc(self,
                             method: str,
                             timeout: Optional[float] = None,
                             args: tuple = (),
                             kwargs: Optional[dict] = None):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine.collective_rpc_async(method, timeout, args,
                                                      kwargs)


# TODO(v1): Remove this class proxy when V1 goes default.
if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
    from vllm.v1.engine.async_llm import AsyncLLM

    AsyncLLMEngine = AsyncLLM  # type: ignore
