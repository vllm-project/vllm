import multiprocessing
from typing import List, Optional, Tuple, Type

import pickle
import zmq

from vllm.config import (CacheConfig, DeviceConfig,
                         LoadConfig, LoRAConfig, ModelConfig,
                         ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput, NewTokens
from vllm.v1.request import Request
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.executor.gpu_executor import GPUExecutor

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
        
        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Recieve input from the LLMEngine.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(self.input_path)

        # Send input to the LLMEngine detokenizer.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(self.output_path)

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

        self._initialize_kv_caches()

        # NOTE: the cache_config here have been updated with the numbers of
        # GPU and CPU blocks, which are profiled in the distributed executor.
        self.scheduler = Scheduler(self.scheduler_config, 
                                   self.cache_config, 
                                   self.lora_config)

        # TODO: add heartbeat thread.

        # Run core loop.
        self._run_busy_loop()

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

        # If there is no work to do, poll until there is.
        if not self.scheduler.has_unfinished_requests():
            while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for new requests in engine loop.")

        # Handle any new input.
        self._handle_new_input()

        # Run a single step.
        self._step()

    def _handle_new_input(self):
        """Handle new input from the Engine."""
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
            # TODO: handle gracefull
            raise e
    
    def _step(self) -> None:
        """Schedule, execute, and send output to Detokenizer."""
        if self.scheduler.has_unfinished_requests():
            scheduler_output = self.scheduler.schedule()
            output = self.model_executor.execute_model(scheduler_output)
            sampled = self.scheduler.update_from_output(
                scheduler_output, output)
            self._send_sampled(sampled)

    def _send_sampled(self, sampled: List[Tuple[Request, int]]):
        # TODO(robertgshaw2): We could avoid this conversion loop by either/or:
        #   - scheduler.update_from_output() creates DetokenizerInputData
        #   - serializing and sending the Requests directly to the Detokenizer
        # The negative of this is that the Detokenizer is then more coupled.
        
        input_data = [
            DetokenizerInputData(
                request_id=req.request_id,
                new_token_ids=req.output_token_ids[-num_tokens:],
                finished=req.is_finished(),
                finish_reason=req.get_finished_reason(),
                stop_reason=req.stop_reason) for req, num_tokens in sampled
        ]

        self.detokenizer.send(DetokenizerInputs(data=input_data))
    
    
