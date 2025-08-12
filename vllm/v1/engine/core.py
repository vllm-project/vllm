# SPDX-License-Identifier: Apache-2.0
import os
import queue
import signal
import sys
import threading
import time
from concurrent.futures import Future
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import psutil
import zmq
import zmq.asyncio

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.executor.multiproc_worker_utils import _add_prefix
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import (get_exception_traceback, resolve_obj_by_qualname,
                        zmq_socket_ctx)
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MMInputCacheServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5

_R = TypeVar('_R')  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        self.scheduler: SchedulerInterface = Scheduler(
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size
            > 1,
            log_stats=self.log_stats,
        )

        # Setup MM Input Mapper.
        self.mm_input_cache_server = MMInputCacheServer(
            vllm_config.model_config)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

    def _initialize_kv_caches(
            self, vllm_config: VllmConfig) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = self.model_executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)
        # Get the kv cache tensor size
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for a multimodal input, then it will be
            # fetched from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client cache, so
            # anything that has a hash must have a HIT cache entry here
            # as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_cache_server.get_and_update(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self.structured_output_manager.grammar_init(req)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return EngineCoreOutputs(
                outputs=[],
                scheduler_stats=self.scheduler.make_stats(),
            )
        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)  # type: ignore

        return engine_core_outputs

    def step_with_batch_queue(self) -> Optional[EngineCoreOutputs]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if there are unscheduled requests
        and the job queue is not full. If a new batch is scheduled, directly
        return an empty engine core output. In other words, we won't check
        and return model outputs before the batch queue is full.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        assert self.batch_queue is not None

        engine_core_outputs = None
        scheduler_output = None
        # If there are unscheduled requests and the job queue
        # is not full, schedule a new batch. Note that this is not blocking.
        if (self.scheduler.get_num_unscheduled_requests() > 0
                and not self.batch_queue.full()):
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens > 0:
                future = self.model_executor.execute_model(scheduler_output)
                self.batch_queue.put_nowait(
                    (future, scheduler_output))  # type: ignore

        scheduled_batch = (scheduler_output is not None
                           and scheduler_output.total_num_scheduled_tokens > 0)

        # If no more requests can be scheduled and the job queue is not empty,
        # block until the first batch in the job queue is finished.
        if not scheduled_batch and not self.batch_queue.empty():
            future, scheduler_output = self.batch_queue.get_nowait()
            # Blocking until the first result is available.
            model_output = future.result()
            self.batch_queue.task_done()
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output)

        return engine_core_outputs

    def shutdown(self):
        self.model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.model_executor.save_sharded_state(path=path,
                                               pattern=pattern,
                                               max_size=max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args,
                                                  kwargs)


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        super().__init__(vllm_config, executor_class, log_stats)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

        self.global_unfinished_reqs = False

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue: queue.Queue[tuple[EngineCoreRequestType,
                                            Any]] = queue.Queue()
        self.output_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()
        threading.Thread(target=self.process_input_socket,
                         args=(input_path, engine_index),
                         daemon=True).start()
        threading.Thread(target=self.process_output_socket,
                         args=(output_path, engine_index),
                         daemon=True).start()

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()
        engine_core: Optional[EngineCoreProc] = None
        try:
            parallel_config: ParallelConfig = kwargs[
                "vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("EngineCore hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGUSR1)

        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.global_unfinished_reqs and not (
                self.scheduler.has_requests()):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug(
                "EngineCore loop active - local unfinished: %s, finished: %s.",
                self.scheduler.has_unfinished_requests(),
                self.scheduler.has_finished_requests())

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self):
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        if outputs is not None:
            self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.START_DP:
            if not self.global_unfinished_reqs:
                logger.debug("EngineCore starting idle loop.")
                self.global_unfinished_reqs = True
        elif request_type == EngineCoreRequestType.UTILITY:
            call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                EngineCoreOutputs(utility_output=output))

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def process_input_socket(self, input_path: str, engine_index: int):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()
        identity = engine_index.to_bytes(length=2, byteorder="little")

        with zmq_socket_ctx(input_path,
                            zmq.DEALER,
                            identity=identity,
                            bind=False) as socket:

            # Send ready message to front-end once input socket is connected.
            socket.send(b'READY')

            while True:
                # (RequestType, RequestData)
                type_frame, data_frame = socket.recv_multipart(copy=False)
                request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                # Deserialize the request data.
                decoder = add_request_decoder if (
                    request_type
                    == EngineCoreRequestType.ADD) else generic_decoder
                request = decoder.decode(data_frame.buffer)

                # Push to input queue for core busy loop.
                self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_path: str, engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                outputs.engine_index = engine_index
                encoder.encode_into(outputs, buffer)
                socket.send(buffer, copy=False)


ENGINE_PAUSED_OUTPUTS = EngineCoreOutputs(engine_paused=True)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ):
        # Add process-specific prefix to stdout and stderr before
        # we initialize the engine.
        from multiprocessing import current_process
        process_name = current_process().name
        pid = os.getpid()
        _add_prefix(sys.stdout, process_name, pid)
        _add_prefix(sys.stderr, process_name, pid)

        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        from vllm.platforms import current_platform
        if current_platform.is_cuda_alike():
            from vllm.platforms.cuda import device_id_to_physical_device_id
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(device_id_to_physical_device_id(i))
                for i in range(local_dp_rank * tp_size, (local_dp_rank + 1) *
                               tp_size))

        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

        # Initialize the engine after setting up environment.
        super().__init__(input_path, output_path, vllm_config, executor_class,
                         log_stats, dp_rank)

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.counter = 0

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            if local_unfinished_reqs:
                # 2) Step the engine core.
                self._process_engine_step()

                # Check if we have now finished all requests.
                local_unfinished_reqs = (
                    self.scheduler.has_unfinished_requests())
            else:
                if self.scheduler.has_finished_requests():
                    # There are no unfinished requests, but there are some
                    # finished requests remaining to be removed from the
                    # batch state. This engine step won't perform a forward
                    # pass but will flush the finished requests to ensure
                    # up-to-date state is returned in the engine outputs.
                    self._process_engine_step()

                if not self.global_unfinished_reqs:
                    # All engines are idle.
                    continue

                # There must be unfinished requests in DP peers, run a
                # dummy forward pass.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.global_unfinished_reqs = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.global_unfinished_reqs:
                # Notify client that we are pausing the loop.
                self.output_queue.put_nowait(ENGINE_PAUSED_OUTPUTS)

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:

        # Optimization - only perform finish-sync all-reduce every 16 steps.
        self.counter += 1
        if self.counter != 16:
            return True
        self.counter = 0

        return ParallelConfig.has_unfinished_dp(self.dp_group,
                                                local_unfinished)
