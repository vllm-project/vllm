import multiprocessing
from multiprocessing.process import BaseProcess
from typing import List, Tuple, Type

import msgspec
import zmq
import zmq.asyncio

from vllm.config import CacheConfig, EngineConfig
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.engine import (POLLING_TIMEOUT_MS, EngineCoreOutput,
                            EngineCoreOutputs, EngineCoreRequest)
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.request import Request
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

LLM_ENGINE_CORE_READY_STR = "READY"


class EngineCore:

    def __init__(
        self,
        vllm_config: EngineConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
    ):
        # Override the configs for V1.
        # FIXME
        if usage_context == UsageContext.LLM_CLASS:
            vllm_config.scheduler_config.max_num_seqs = 1024
            vllm_config.scheduler_config.max_num_batched_tokens = 8192
        elif usage_context == UsageContext.OPENAI_API_SERVER:
            vllm_config.scheduler_config.max_num_seqs = 1024
            vllm_config.scheduler_config.max_num_batched_tokens = 2048

        assert vllm_config.model_config.task != "embedding"

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
            vllm_config.model_config.model, vllm_config.speculative_config,
            vllm_config.model_config.tokenizer,
            vllm_config.model_config.skip_tokenizer_init,
            vllm_config.model_config.tokenizer_mode,
            vllm_config.model_config.revision,
            vllm_config.model_config.override_neuron_config,
            vllm_config.model_config.rope_scaling,
            vllm_config.model_config.rope_theta,
            vllm_config.model_config.tokenizer_revision,
            vllm_config.model_config.trust_remote_code,
            vllm_config.model_config.dtype,
            vllm_config.model_config.max_model_len,
            vllm_config.load_config.download_dir,
            vllm_config.load_config.load_format,
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.pipeline_parallel_size,
            vllm_config.parallel_config.disable_custom_all_reduce,
            vllm_config.model_config.quantization,
            vllm_config.model_config.enforce_eager,
            vllm_config.cache_config.cache_dtype,
            vllm_config.model_config.quantization_param_path,
            vllm_config.device_config.device, vllm_config.decoding_config,
            vllm_config.observability_config, vllm_config.model_config.seed,
            vllm_config.model_config.served_model_name,
            vllm_config.scheduler_config.num_scheduler_steps,
            vllm_config.cache_config.enable_prefix_caching,
            vllm_config.model_config.use_async_output_proc,
            vllm_config.model_config.mm_processor_kwargs)

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
            vllm_config.cache_config)
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

        # Setup scheduler.
        self.scheduler = Scheduler(vllm_config.scheduler_config,
                                   vllm_config.cache_config,
                                   vllm_config.lora_config)

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

    def add_request(self, engine_core_request: EngineCoreRequest):
        """Add request to the scheduler."""

        request = Request.from_engine_core_request(engine_core_request)
        self.scheduler.add_request(request)

    def step(self) -> List[EngineCoreOutput]:
        """Schedule, execute, and make output."""

        if not self.scheduler.has_unfinished_requests():
            return []

        scheduler_output = self.scheduler.schedule()
        output = self.model_executor.execute_model(scheduler_output)
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, output)
        return engine_core_outputs

    def check_health(self):
        self.model_executor.check_health()


class EngineCoreProc(EngineCore):
    """ZMQ-based busy loop wrapper for EngineCore"""

    def __init__(
        self,
        vllm_config: EngineConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
    ):
        super().__init__(vllm_config, executor_class, usage_context)

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
        ready_socket = None
        try:
            ready_socket = self.ctx.socket(zmq.constants.PUSH)
            ready_socket.bind(ready_path)
            ready_socket.send_string(LLM_ENGINE_CORE_READY_STR)
        finally:
            if ready_socket:
                ready_socket.close(linger=0)

    @staticmethod
    def make_engine_core_process(
        vllm_config: EngineConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
        input_path: str,
        output_path: str,
        ready_path: str,
    ) -> BaseProcess:
        # The current process might have CUDA context,
        # so we need to spawn a new process
        # context = multiprocessing.get_context("spawn")
        context = multiprocessing.get_context("fork")

        process_kwargs = {
            "input_path": input_path,
            "output_path": output_path,
            "ready_path": ready_path,
            "vllm_config": vllm_config,
            "executor_class": executor_class,
            "usage_context": usage_context,
        }
        # Run EngineCore busy loop in background process.
        return context.Process(target=EngineCoreProc.run_engine_core,
                               kwargs=process_kwargs)

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch EngineCore busy loop in background process."""

        logger.debug("Initializing EngineCore in background process.")
        engine_core = EngineCoreProc(*args, **kwargs)

        logger.debug("Starting EngineCore busy loop in background process.")
        engine_core.run_busy_loop()

    def run_busy_loop(self):
        """Core busy loop of the EngineCoreProc"""

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

        if not engine_core_outputs:
            return

        outputs = EngineCoreOutputs(outputs=engine_core_outputs)
        outputs_serialized = self.msgpack_encoder.encode(outputs)
        self.output_socket.send_multipart((outputs_serialized, ),
                                          copy=False,
                                          flags=zmq.NOBLOCK)


class EngineCoreClient:
    """Client for the EngineCore."""

    def __init__(
        self,
        vllm_config: EngineConfig,
        executor_class: Type[GPUExecutor],
        usage_context: UsageContext,
        asyncio_mode: bool = True,
    ):
        # Serialization setup.
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # IPC Setup
        self.asyncio_mode = asyncio_mode
        self.ctx = (
            zmq.asyncio.Context() if self.asyncio_mode else zmq.Context()
        )  # type: ignore[attr-defined]

        # Path for IPC.
        ready_path = get_open_zmq_ipc_path()
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()

        # Get output (EngineCoreOutput) from EngineCore.
        self.output_socket = self.ctx.socket(zmq.constants.PULL)
        self.output_socket.connect(output_path)

        # Send input (EngineCoreRequest) to EngineCore.
        self.input_socket = self.ctx.socket(zmq.constants.PUSH)
        self.input_socket.bind(input_path)

        # Start EngineCore in background process.
        self.proc = EngineCoreProc.make_engine_core_process(
            vllm_config,
            executor_class,
            usage_context,
            input_path,
            output_path,
            ready_path,
        )
        self.proc.start()
        self.wait_for_startup(self.proc, ready_path)

    def __del__(self):
        # Hack.
        self.proc.kill()

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

            # Poll ready socket socket until
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for EngineCore to startup.")

                if not proc.is_alive():
                    raise RuntimeError("EngineCore process failed to start.")

            message = socket.recv_string()
            assert message == LLM_ENGINE_CORE_READY_STR

        except BaseException as e:
            logger.exception(e)
            raise e

        finally:
            sync_ctx.destroy(linger=0)

    def get_output(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from the EngineCore (non-blocking)."""

        assert not self.asyncio_mode

        while self.output_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for output from EngineCore.")

        frames = self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    async def get_output_async(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from EngineCore (non-blocking) in asyncio."""

        assert self.asyncio_mode

        while await self.output_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for output from EngineCore.")

        frames = await self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    def add_request(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore (non-blocking)."""

        assert not self.asyncio_mode

        self.input_socket.send_multipart((self.encoder.encode(request), ),
                                         copy=False,
                                         flags=zmq.NOBLOCK)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore (non-blocking) in asyncio."""

        assert self.asyncio_mode

        await self.input_socket.send_multipart(
            (self.encoder.encode(request), ), copy=False, flags=zmq.NOBLOCK)
