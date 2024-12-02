import multiprocessing
import pickle
import queue
import threading
import time
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Iterator, List, Tuple, Type, Union

import zmq
import zmq.asyncio
from msgspec import msgpack

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.sequence import Logprob
from vllm.usage.usage_lib import UsageContext
from vllm.v1.core.scheduler import Scheduler, SchedulerOutput
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreProfile, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.engine.mm_input_mapper import MMInputMapper
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import PickleEncoder
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000
LOGGING_TIME_S = 5000


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
    ):
        assert vllm_config.model_config.task != "embedding"

        logger.info("Initializing an LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config.cache_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # Set up multimodal input mapper (e.g., convert PIL images to tensors).
        self.mm_input_mapper = MMInputMapper(vllm_config.model_config)

        # Setup scheduler.
        self.scheduler = Scheduler(vllm_config.scheduler_config,
                                   vllm_config.cache_config,
                                   vllm_config.lora_config)

        self._last_logging_time = time.time()

    def _initialize_kv_caches(self,
                              cache_config: CacheConfig) -> Tuple[int, int]:
        num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks(
        )

        if cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        num_cpu_blocks = 0
        self.model_executor.initialize_cache(num_gpu_blocks)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        req = Request.from_engine_core_request(request)
        # FIXME(woosuk): The input mapping (e.g., PIL images to tensors) may
        # take 10-50 ms, which can cause a spike in the latency. We should
        # consider moving this to a separate thread.
        if req.mm_data:
            req.mm_inputs = self.mm_input_mapper.process_inputs(
                req.mm_data, req.mm_processor_kwargs)
        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def _pythonize_logprobs(
        self,
        do_logprobs: bool,
        do_prompt_logprobs: bool,
        model_runner_output: "ModelRunnerOutput",
    ) -> Tuple[List, List, List, List]:
        """Convert logprobs tensors to Python data structures.
        
        Args:
          do_logprobs: sample logprobs are required
          do_prompt_logprobs: prompt logprobs are required
          model_runner_output: model runner output contains CPU logprobs tensors

        Returns:
          logprob_token_ids_list
          logprob_values_list
          prompt_logprob_token_ids_list
          prompt_logprob_values_list
        """
        if do_logprobs:
            # Pythonize sample logprobs if needed
            assert model_runner_output.logprob_token_ids_cpu is not None
            logprob_token_ids_list = (
                model_runner_output.logprob_token_ids_cpu.tolist())
            logprob_values_list = (model_runner_output.logprobs_cpu.tolist())
        else:
            (
                logprob_token_ids_list,
                logprob_values_list,
            ) = (None, None)
        if do_prompt_logprobs:
            # Pythonize prompt logprobs if needed
            assert model_runner_output.prompt_logprob_token_ids_cpu is not None
            prompt_logprob_token_ids_list = (
                model_runner_output.prompt_logprob_token_ids_cpu.tolist())
            prompt_logprob_values_list = (
                model_runner_output.prompt_logprobs_cpu.tolist())
        else:
            (
                prompt_logprob_token_ids_list,
                prompt_logprob_values_list,
            ) = (None, None)

        return (logprob_token_ids_list, logprob_values_list,
                prompt_logprob_token_ids_list, prompt_logprob_values_list)

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> List[EngineCoreOutput]:
        """Build engine core output from model runner output.
        
        Args:
          scheduler_output: scheduler output prior to engine step.
          model_runner_output: model runner output from engine step.

        Returns:
          Engine core output which tracks the progress of generation.
        """
        scheduler = self.scheduler
        # NOTE(woosuk): This method doesn't consider speculative decoding.
        sampled_token_ids = model_runner_output.sampled_token_ids_cpu.tolist()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        do_logprobs = model_runner_output.logprobs_cpu is not None
        do_prompt_logprobs = (
            model_runner_output.prompt_logprobs_cpu is not None
            and len(model_runner_output.prompt_logprobs_cpu) > 0)

        # Get logprobs as Python data structures
        (
            logprob_token_ids_list,
            logprob_values_list,
            prompt_logprob_token_ids_list,
            prompt_logprob_values_list,
        ) = self._pythonize_logprobs(do_logprobs, do_prompt_logprobs,
                                     model_runner_output)

        if do_prompt_logprobs:
            # Index into prompt tokens, for building
            # prompt logprobs output data structure
            curr_prompt_base_idx = 0
        new_running: List[Request] = []
        engine_core_outputs: List[EngineCoreOutput] = []
        for request in scheduler.running:
            req_id = request.request_id
            request.num_computed_tokens += num_scheduled_tokens[req_id]
            req_index = model_runner_output.req_id_to_index[req_id]
            num_new_tokens = 1
            max_logprobs = request.max_logprobs
            request_do_logprobs = (do_logprobs and max_logprobs is not None
                                   and max_logprobs > 0)

            if do_prompt_logprobs:
                max_prompt_logprobs = request.max_prompt_logprobs
                # Number of new prompt tokens is the number of scheduled
                # tokens *if* the request is partial (because the sampled
                # token is discarded and all sequence offsets are prompt
                # offsets), otherwise it is the number of scheduled
                # tokens minus one (for the sampled token)
                num_new_prompt_tokens = (
                    num_scheduled_tokens[request.request_id] -
                    int(scheduler_output.partial_req_index != req_index))

                request_do_prompt_logprobs = (max_prompt_logprobs is not None
                                              and max_prompt_logprobs > 0
                                              and num_new_prompt_tokens > 0)

                if request_do_prompt_logprobs:

                    # Construct prompt logprobs, under the condition that
                    # prompt logprobs were requested & a nonzero number of
                    # prompt tokens were computed in this step for this request.
                    #
                    # Note that this scenario returns an EngineCoreOutput which
                    # is empty except for the prompt logprobs which were
                    # computed for these prompt tokens.

                    slice_upper_index = (curr_prompt_base_idx +
                                         num_new_prompt_tokens)
                    prompt_logprob_token_ids = prompt_logprob_token_ids_list[
                        curr_prompt_base_idx:slice_upper_index]
                    prompt_logprob_values = prompt_logprob_values_list[
                        curr_prompt_base_idx:slice_upper_index]
                    curr_prompt_base_idx = slice_upper_index

                    logprob_cnt = max_prompt_logprobs
                    prompt_logprobs = [{
                        lpt: Logprob(lpv, (idx + 1), None)
                        for idx, (lpv, lpt) in enumerate(
                            zip(plp_tok_values[0:logprob_cnt],
                                plp_tok_token_ids[0:logprob_cnt]))
                    } for plp_tok_values, plp_tok_token_ids in zip(
                        prompt_logprob_values, prompt_logprob_token_ids)]

                    if not request.prompt_logprobs:
                        # Ensure that None is the first prompt logprob
                        prompt_logprobs = [None] + prompt_logprobs

                    curr_prompt_base_idx = slice_upper_index

                    prompt_slice_range_upper = request.num_computed_tokens
                    prompt_slice_range_lower = (prompt_slice_range_upper -
                                                num_new_prompt_tokens)
                    request.prompt_logprobs.extend(prompt_logprobs)
                else:
                    curr_prompt_base_idx += num_new_prompt_tokens
            else:
                request_do_prompt_logprobs = False

            # When the request's num_computed_tokens catches up its num_tokens,
            # the request generates output tokens. Otherwise, we ignore the
            # sampler output for the request.
            assert request.num_computed_tokens <= request.num_tokens

            cached_encoder_input_ids = (
                scheduler.encoder_cache_manager.get_cached_input_ids(request))
            for input_id in list(cached_encoder_input_ids):
                start_pos = request.mm_positions[input_id]["offset"]
                num_tokens = request.mm_positions[input_id]["length"]
                if start_pos + num_tokens <= request.num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    scheduler.encoder_cache_manager.free(request, input_id)

            if request.num_computed_tokens == request.num_tokens:
                # NOTE(woosuk): Currently, we assume that each request
                # generates at most one token at each step.
                token_id = sampled_token_ids[req_index]
                if request_do_logprobs:
                    # Construct logprobs, if requested (TODO: assumes one
                    # generated token).
                    logprob_token_ids = logprob_token_ids_list[req_index]
                    logprob_values = logprob_values_list[req_index]
                    logprob_cnt = max_logprobs
                    if token_id not in logprob_token_ids[0:max_logprobs]:
                        # Sampled token is not in the in the top logprobs;
                        # inject it & resort, ensuring that excess logprobs
                        # not requested by the user have -inf probability
                        logprob_values[max_logprobs:-1] = (
                            [float('-inf')] *
                            (len(logprob_values) - 1 - max_logprobs))

                        indices = sorted(range(len(logprob_values)),
                                         key=lambda k: logprob_values[k],
                                         reverse=True)
                        logprob_values = [logprob_values[i] for i in indices]
                        logprob_token_ids = [
                            logprob_token_ids[i] for i in indices
                        ]

                        # There will be one more logprob than the user requested
                        logprob_cnt = max_logprobs + 1

                    # Only keep the number of logprobs specified by the request
                    # (plus possibly the sampled token id & its logprob)
                    logprob_values = logprob_values[0:logprob_cnt]
                    logprob_token_ids = logprob_token_ids[0:logprob_cnt]

                    request.logprobs.append({
                        lpt: Logprob(lpv, (idx + 1), None)
                        for idx, (lpv, lpt) in enumerate(
                            zip(logprob_values, logprob_token_ids))
                    })
                request.append_output_token_ids(token_id)
                # TODO: Update the KV cache manager for prefix caching.

                # Check for stop and update request state.
                # This must be called before me make the EngineCoreOutput.
                stopped = scheduler._check_stop(request)

                # Add EngineCoreOutput for this Request.
                # Return the logprob for the most recently computed tokens.
                # Return no prompt logprobs in decode-phase.
                output = EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=request.output_token_ids[-num_new_tokens:],
                    finished=request.is_finished(),
                    finish_reason=request.get_finished_reason(),
                    stop_reason=request.stop_reason,
                    logprobs=(request.logprobs[-num_new_tokens:]
                              if request_do_logprobs else None),
                    prompt_logprobs=(prompt_logprobs
                                     if request_do_prompt_logprobs else None),
                    prompt_logprobs_token_ids=(request.prompt_token_ids
                                               if request_do_prompt_logprobs
                                               else None))
                engine_core_outputs.append(output)

                # Breakout of the loop.
                if stopped:
                    continue

            elif request_do_prompt_logprobs:
                # This request is still partial but prompt logprobs were
                # requested
                engine_core_outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=[],
                        finished=request.is_finished(),
                        finish_reason=request.get_finished_reason(),
                        stop_reason=request.stop_reason,
                        logprobs=[] if request_do_logprobs else None,
                        prompt_logprobs=(
                            prompt_logprobs if request_do_prompt_logprobs else
                            ([] if request_do_prompt_logprobs else None)),
                        prompt_logprobs_token_ids=(
                            request.prompt_token_ids[prompt_slice_range_lower:
                                                     prompt_slice_range_upper]
                            if request_do_prompt_logprobs else
                            ([] if request_do_prompt_logprobs else None))))

            new_running.append(request)
        scheduler.running = new_running
        return engine_core_outputs

    def step(self) -> List[EngineCoreOutput]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return []

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.update_from_output(scheduler_output, output)
        return engine_core_outputs

    def profile(self, is_start=True):
        self.model_executor.worker.profile(is_start)


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
        should_shutdown: Synchronized,
    ):
        super().__init__(vllm_config, executor_class, usage_context)

        # Signal from main process to shutdown (multiprocessing.Value).
        self.should_shutdown = should_shutdown

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, ),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, ),
                         daemon=True).start()

        # Send Readiness signal to EngineClient.
        with self.make_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            ready_socket.send_string(EngineCoreProc.READY_STR)

    @contextmanager
    def make_socket(self, path: str, type: Any) -> Iterator[zmq.Socket]:
        """Context manager for use """

        ctx = zmq.Context()
        try:
            socket = ctx.socket(type)

            if type == zmq.constants.PULL:
                socket.connect(path)
            elif type == zmq.constants.PUSH:
                socket.bind(path)
            else:
                raise ValueError(f"Unknown Socket Type: {type}")

            yield socket

        except KeyboardInterrupt:
            logger.debug("EngineCore had Keyboard Interrupt.")

        finally:
            ctx.destroy(linger=0)

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> None:
        """Wait until the EngineCore is ready."""

        try:
            sync_ctx = zmq.Context()  # type: ignore[attr-defined]
            socket = sync_ctx.socket(zmq.constants.PULL)
            socket.connect(ready_path)

            # Wait for EngineCore to send EngineCoreProc.READY_STR.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for EngineCoreProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("EngineCoreProc failed to start.")

            message = socket.recv_string()
            assert message == EngineCoreProc.READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

    @staticmethod
    def make_engine_core_process(
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
        should_shutdown: Synchronized,
    ) -> BaseProcess:
        # The current process might have CUDA context,
        # so we need to spawn a new process.
        # NOTE(rob): this is a problem for using EngineCoreProc w/
        # LLM, since we need a if __name__ == "__main__" guard.
        context = multiprocessing.get_context("spawn")

        process_kwargs = {
            "input_path": input_path,
            "output_path": output_path,
            "ready_path": ready_path,
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            "usage_context": usage_context,
            "should_shutdown": should_shutdown
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=EngineCoreProc.run_engine_core,
                               kwargs=process_kwargs)
        proc.start()

        # Wait for startup
        EngineCoreProc.wait_for_startup(proc, ready_path)
        return proc

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except KeyboardInterrupt:
            logger.debug("EngineCore interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise e

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until we get a shutdown signal.
        while not self.should_shutdown:
            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(req)
                        break
                    except queue.Empty:
                        self._log_stats()
                        logger.debug("EngineCore busy loop waiting.")
                        if self.should_shutdown:
                            return

            # 2) Handle any new client requests (Abort or Add).
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(req)

            # 3) Step the engine core.
            outputs = self.step()

            # 4) Put EngineCoreOutputs into the output queue.
            self.output_queue.put_nowait(outputs)

            self._log_stats()

    def _log_stats(self):
        """Log basic stats every LOGGING_TIME_S"""

        now = time.time()

        if now - self._last_logging_time > LOGGING_TIME_S:
            logger.info(
                "RUNNING: %s | WAITING: %s",
                len(self.scheduler.running),
                len(self.scheduler.waiting),
            )

            self._last_logging_time = now

    def _handle_client_request(
        self, request: Union[EngineCoreRequest, EngineCoreProfile,
                             List[str]]) -> None:
        """Handle EngineCoreRequest or EngineCoreABORT from Client."""

        if isinstance(request, EngineCoreRequest):
            self.add_request(request)
        elif isinstance(request, EngineCoreProfile):
            self.model_executor.worker.profile(request.is_start)
        else:
            # TODO: make an EngineCoreAbort wrapper
            assert isinstance(request, list)
            self.abort_requests(request)

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        decoder_add_req = PickleEncoder()
        decoder_abort_req = PickleEncoder()

        with self.make_socket(input_path, zmq.constants.PULL) as socket:
            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = type_frame.buffer
                request_data = data_frame.buffer

                # Deserialize the request data.
                if request_type == EngineCoreRequestType.ADD.value:
                    request = decoder_add_req.decode(request_data)
                elif request_type == EngineCoreRequestType.ABORT.value:
                    request = decoder_abort_req.decode(request_data)
                elif request_type == EngineCoreRequestType.PROFILE.value:
                    request = pickle.loads(request_data)
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait(request)

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with self.make_socket(output_path, zmq.constants.PUSH) as socket:
            while True:
                engine_core_outputs = self.output_queue.get()
                outputs = EngineCoreOutputs(outputs=engine_core_outputs)
                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)
