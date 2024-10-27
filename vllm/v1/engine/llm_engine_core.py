import multiprocessing
import pickle
from typing import List, Optional, Type

import msgspec
import zmq

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.request import Request

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 10000


class LLMEngineCore(multiprocessing.Process):

    def __init__(
        self,
        input_path: str,
        output_path: str,
        executor_class: Type[GPUExecutor],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.executor_class = executor_class
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.lora_config = lora_config
        self.observability_config = observability_config
        self.prompt_adapter_config = prompt_adapter_config

    def run(self):
        # Initialize these objects after the process is forked.
        self.msgpack_encoder = msgspec.msgpack.Encoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Get EngineCoreRequests from the LLMEngine.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.connect(self.input_path)

        # Send EngineCoreOutput to the LLMEngine.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.connect(self.output_path)

        # Setup Model.
        self.model_executor = self.executor_class(
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            lora_config=self.lora_config,
            speculative_config=self.speculative_config,
            load_config=self.load_config,
            prompt_adapter_config=self.prompt_adapter_config,
            observability_config=self.observability_config,
        )

        # Setup KV Caches.
        # NOTE: the cache_config isn updated with the numbers of GPU and CPU
        # blocks, which are profiled in the distributed executor.
        self._initialize_kv_caches()

        # Setup Scheduler.
        self.scheduler = Scheduler(self.scheduler_config, self.cache_config,
                                   self.lora_config)

        # Kickoff the busy loop.
        self._run_busy_loop()

        # TODO: add heartbeat thread.

    def _initialize_kv_caches(self) -> None:
        num_gpu_blocks, _ = self.model_executor.determine_num_available_blocks(
        )

        if self.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = 0
        self.model_executor.initialize_cache(num_gpu_blocks)

    def _run_busy_loop(self):
        """Core busy loop of the LLMEngineCore."""

        # Poll the input socket until there is work to do.
        if not self.scheduler.has_unfinished_requests():
            while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for new requests in engine loop.")

        # Handle new input from the socket.
        self._handle_new_input()

        # Forward pass.
        outputs = self._step()

        # Send outputs back to the LLMEngine.
        self._send_outputs(outputs)

    def _step(self) -> Optional[List[EngineCoreOutputs]]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return None

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def _handle_new_input(self):
        """Handle new input from the LLMEngine."""
        try:
            while self.input_socket.poll(timeout=0) != 0:
                frames = self.input_socket.recv_multipart(copy=False)
                request = pickle.loads(frames[0].buffer)

                assert isinstance(request, Request)
                self.scheduler.add_request(request)

                # TODO: handle abort
                # TODO: handle logits processors - (cloudpickle)
                # TODO: handle profiling

        except Exception as e:
            # TODO: handle gracefully
            raise e

    def _send_outputs(
            self,
            engine_core_outputs: Optional[List[EngineCoreOutput]]) -> None:
        """Serialize and send output to the LLMEngine."""

        if engine_core_outputs is None:
            return

        outputs = EngineCoreOutputs(data=engine_core_outputs)
        outputs_serialized = self.msgpack_encoder.encode(outputs)
        self.output_socket.send_multipart((outputs_serialized, ),
                                          copy=False,
                                          flags=zmq.NOBLOCK)
