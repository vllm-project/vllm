from typing import List, Optional, Tuple, Type

import msgspec
import zmq

from vllm.config import (CacheConfig, DeviceConfig, DecodingConfig, LoadConfig,
                         LoRAConfig, ModelConfig, ObservabilityConfig,
                         ParallelConfig, PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.engine import (LLM_ENGINE_CORE_READY_STR, POLLING_TIMEOUT_MS,
                            EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreRequest)
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.request import Request
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


class LLMEngineCore:

    def __init__(
        self,
        executor_class: Type[GPUExecutor],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        decoding_config: Optional[DecodingConfig],
        observability_config: Optional[ObservabilityConfig],
        prompt_adapter_config: Optional[PromptAdapterConfig],
        async_mode: bool = False,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        ready_path: Optional[str] = None,
    ):
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, speculative_config=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "override_neuron_config=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, tensor_parallel_size=%d, "
            "pipeline_parallel_size=%d, "
            "disable_custom_all_reduce=%s, quantization=%s, "
            "enforce_eager=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "decoding_config=%r, observability_config=%r, "
            "seed=%d, served_model_name=%s, "
            "num_scheduler_steps=%d, enable_prefix_caching=%s, "
            "use_async_output_proc=%s, mm_processor_kwargs=%s)", VLLM_VERSION,
            model_config.model, speculative_config, model_config.tokenizer,
            model_config.skip_tokenizer_init, model_config.tokenizer_mode,
            model_config.revision, model_config.override_neuron_config,
            model_config.rope_scaling, model_config.rope_theta,
            model_config.tokenizer_revision, model_config.trust_remote_code,
            model_config.dtype, model_config.max_model_len,
            load_config.download_dir, load_config.load_format,
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size,
            parallel_config.disable_custom_all_reduce,
            model_config.quantization, model_config.enforce_eager,
            cache_config.cache_dtype, model_config.quantization_param_path,
            device_config.device, decoding_config, observability_config,
            model_config.seed, model_config.served_model_name,
            scheduler_config.num_scheduler_steps,
            cache_config.enable_prefix_caching,
            model_config.use_async_output_proc,
            model_config.mm_processor_kwargs)

        # Setup Model.
        self.model_executor = executor_class(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            prompt_adapter_config=prompt_adapter_config,
            observability_config=observability_config,
        )

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            cache_config)
        cache_config.num_gpu_blocks = num_gpu_blocks
        cache_config.num_cpu_blocks = num_cpu_blocks

        # Setup scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config, lora_config)

        # Setup IPC if running in async mode.
        if async_mode:
            assert (input_path is not None and output_path is not None
                    and ready_path is not None)

            self.msgpack_encoder = msgspec.msgpack.Encoder()
            self.msgpack_decoder = msgspec.msgpack.Decoder(EngineCoreRequest)

            self.ctx = zmq.Context()  # type: ignore[attr-defined]

            # Get EngineCoreRequests from the LLMEngine.
            self.input_socket = self.ctx.socket(zmq.constants.PULL)
            self.input_socket.connect(input_path)

            # Send EngineCoreOutput to the LLMEngine.
            self.output_socket = self.ctx.socket(zmq.constants.PUSH)
            self.output_socket.bind(output_path)

            # Send Readiness signal to LLMEngine.
            try:
                ready_socket = self.ctx.socket(zmq.constants.PUSH)
                ready_socket.bind(ready_path)
                ready_socket.send_string(LLM_ENGINE_CORE_READY_STR)
            finally:
                ready_socket.close(linger=0)

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

    def check_health(self):
        self.model_executor.check_health()

    def add_request(self, engine_core_request: EngineCoreRequest):
        """Add request to the scheduler."""

        request = Request.from_engine_core_request(engine_core_request)
        self.scheduler.add_request(request)

    def step(self) -> List[EngineCoreOutputs]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return []

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def run_busy_loop(self):
        """Core busy loop of the LLMEngineCore for async mode."""

        while True:
            # Poll the input socket until there is work to do.
            if not self.scheduler.has_unfinished_requests():
                while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                    logger.debug("Waiting for new requests from LLMEngine.")

            # Handle new input from the socket.
            self._handle_new_input()

            # Forward pass.
            outputs = self.step()

            # Send outputs to the AsyncLLMEngine.
            self._send_outputs(outputs)

    def _handle_new_input(self):
        """Handle new input from the AsyncLLMEngine for async mode."""

        try:
            if self.input_socket.poll(timeout=0) != 0:
                frames = self.input_socket.recv_multipart(copy=False)
                engine_core_request = self.msgpack_decoder.decode(
                    frames[0].buffer)
                self.add_request(engine_core_request)

                # TODO: handle abort via another socket
                # TODO: handle logits processors via cloudpickle
                # TODO: handle profiling

        except Exception as e:
            # TODO: handle gracefully
            raise e

    def _send_outputs(self,
                      engine_core_outputs: List[EngineCoreOutput]) -> None:
        """Serialize and send output to the AsyncLLMEngine for async mode."""

        if len(engine_core_outputs) == 0:
            return

        outputs = EngineCoreOutputs(outputs=engine_core_outputs)
        outputs_serialized = self.msgpack_encoder.encode(outputs)
        self.output_socket.send_multipart((outputs_serialized, ),
                                          copy=False,
                                          flags=zmq.NOBLOCK)
