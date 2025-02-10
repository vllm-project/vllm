# SPDX-License-Identifier: Apache-2.0

import queue
import signal
import threading
import time
from multiprocessing.connection import Connection
from typing import Any, List, Tuple, Type, Union

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.utils import zmq_socket_ctx
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.engine.mm_input_mapper import MMInputMapperServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
    ):
        assert vllm_config.model_config.runner_type != "pooling"

        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # Setup scheduler.
        self.scheduler = Scheduler(
            scheduler_config=vllm_config.scheduler_config,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            lora_config=vllm_config.lora_config,
        )

        self.mm_input_mapper_server = MMInputMapperServer(
            vllm_config.model_config)

    def _initialize_kv_caches(self,
                              vllm_config: VllmConfig) -> Tuple[int, int]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_spec = self.model_executor.get_kv_cache_spec()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        availble_gpu_memory = self.model_executor.determine_available_memory()

        # Get the kv cache tensor size
        kv_cache_config = get_kv_cache_config(vllm_config, kv_cache_spec,
                                              availble_gpu_memory)
        num_gpu_blocks = kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize(kv_cache_config)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        if request.mm_hashes is not None:
            # Here, if hash exists for an image, then it will be fetched
            # from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client side of the
            # MM mapper, so anything that has a hash must have a HIT cache
            # entry here as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self.mm_input_mapper_server.process_inputs(
                request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)

        self.scheduler.add_request(req)

    def abort_requests(self, request_ids: List[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids,
                                       RequestStatus.FINISHED_ABORTED)

    def step(self) -> EngineCoreOutputs:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return EngineCoreOutputs(
                outputs=[], scheduler_stats=self.scheduler.make_stats())

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def shutdown(self):
        if self.model_executor:
            model_executor.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        input_path: str,
        output_path: str,
        ready_pipe: Connection,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool = False,
    ):
        try:
            super().__init__(vllm_config, executor_class)

            self.log_stats = log_stats

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            self.input_queue: queue.Queue[Tuple[EngineCoreRequestType,
                                                Any]] = queue.Queue()
            self.output_queue: queue.Queue[Union[EngineCoreOutputs,
                                                 bytes]] = queue.Queue()
            self.errored_sent_event = threading.Event()
            threading.Thread(target=self.process_input_socket,
                             args=(input_path, ),
                             daemon=True).start()
            threading.Thread(target=self.process_output_socket,
                             args=(output_path, ),
                             daemon=True).start()

            # Send Readiness signal to EngineClient.
            ready_pipe.send({"status": "READY"})

        except Exception as e:
            logger.exception("EngineCore got error at startup:", exc_info=e)
            ready_pipe.send({"status": "FAILED"})
            raise e

        finally:
            ready_pipe.close()

    @staticmethod
    def run_engine_core(*args, **kwargs):
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

        engine_core = EngineCoreProc(*args, **kwargs)
        try:
            engine_core.run_busy_loop()
        except Exception as e:
            logger.exception("EngineCore got an Exception:", exc_info=e)
            engine_core._send_engine_dead()
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while True:
                    try:
                        req = self.input_queue.get(timeout=POLLING_TIMEOUT_S)
                        self._handle_client_request(*req)
                        break
                    except queue.Empty:
                        logger.debug("EngineCore busy loop waiting.")
                        # Break out the loop so we can log_stats in step().
                        if self.log_stats:
                            break
                    except Exception:
                        raise

            # 2) Handle any new client requests.
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)

            # 3) Step the engine core.
            outputs = self.step()

            # 4) Put EngineCoreOutputs into the output queue.
            self.output_queue.put_nowait(outputs)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.RESET_PREFIX_CACHE:
            self.reset_prefix_cache()
        elif request_type == EngineCoreRequestType.PROFILE:
            self.model_executor.profile(request)

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD to the front of the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        if not self.errored_sent_event.wait(timeout=10.):
            logger.fatal("vLLM shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_socket(self, input_path: str):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with zmq_socket_ctx(input_path, zmq.constants.PULL) as socket:
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

    def process_output_socket(self, output_path: str):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Reuse send buffer.
        buffer = bytearray()

        with zmq_socket_ctx(output_path, zmq.constants.PUSH) as socket:
            while True:
                outputs = self.output_queue.get()
                if outputs == EngineCoreProc.ENGINE_CORE_DEAD:
                    socket.send_multipart((outputs, ), copy=False)
                    break

                encoder.encode_into(outputs, buffer)
                socket.send_multipart((buffer, ), copy=False)

        # Signal to main thread that ENGINE_CORE_DEAD was sent.
        self.errored_sent_event.set()
