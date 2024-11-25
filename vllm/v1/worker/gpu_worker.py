"""A GPU worker class."""
import gc
import multiprocessing
import os
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Optional, Tuple

import msgspec
import torch
import torch.distributed
import zmq

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.device_communicators.shm_broadcast import (Handle,
                                                                 MessageQueue)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        get_open_zmq_ipc_path)
from vllm.v1.core.scheduler_output import ExecutorMsg, ExecutorMsgType
from vllm.v1.outputs import (ModelRunnerOutput, NumBlocksMsg, NumGPUBlocks,
                             ShmHandleMsg, WorkerInitOutputType,
                             WorkerInitRequestType)
from vllm.v1.utils import make_zmq_socket
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

POLLING_TIMEOUT_MS = 5000
POLLING_TIMEOUT_S = POLLING_TIMEOUT_MS // 1000
LOGGING_TIME_S = 5000

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput


class Worker:

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
    ):

        # TODO: use WorkerBase.__init__(self, vllm_config=vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def initialize(self):
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner = GPUModelRunner(self.vllm_config, self.device)

    def load_model(self) -> None:
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _, total_gpu_memory = torch.cuda.mem_get_info()
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.cuda.synchronize()

        free_gpu_memory, _ = torch.cuda.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_gpu_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # gpu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.cuda.empty_cache()
        torch_allocated_bytes = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = _get_cache_block_size(self.cache_config,
                                                 self.model_config,
                                                 self.parallel_config)
        num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        return num_gpu_blocks, 0

    def initialize_cache(self, num_gpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks."""
        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_gpu_blocks
        max_model_len = self.model_config.max_model_len
        if max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`gpu_memory_utilization` or decreasing `max_model_len` when "
                "initializing the engine.")

        self.model_runner.initialize_kv_cache(num_gpu_blocks)

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        output = self.model_runner.execute_model(scheduler_output)
        return output

    def profile(self, is_start=True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()


@dataclass
class WorkerProcHandle:
    proc: BaseProcess
    initialization_input_path: str
    initialization_output_path: str
    model_output_mq_handle: Optional[Handle]

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        with make_zmq_socket(self.initialization_output_path,
                             zmq.constants.PUSH) as send_socket, \
             make_zmq_socket(self.initialization_input_path,
                             zmq.constants.PULL) as recv_socket:

            send_socket.send_multipart(
                (WorkerInitRequestType.DETERMINE_NUM_BLOCKS.value, ))
            type_frame, data_frame = recv_socket.recv_multipart(copy=False)

            request_type = type_frame.buffer
            request_data = data_frame.buffer

            if request_type == WorkerInitOutputType.NUM_BLOCKS.value:
                decoder = msgspec.msgpack.Decoder(NumBlocksMsg)
                num_blocks = decoder.decode(request_data).num_blocks
                return num_blocks
            else:
                raise ValueError(f"Unknown RequestType: {request_type}")

    def initialize_cache(self, num_gpu_blocks: int) -> int:
        with make_zmq_socket(self.initialization_output_path,
                             zmq.constants.PUSH) as socket:
            encoder = msgspec.msgpack.Encoder()
            msg = encoder.encode(NumGPUBlocks(num_gpu_blocks))
            socket.send_multipart(
                (WorkerInitRequestType.INIT_CACHE.value, msg))

    def start_busy_loop(self) -> None:
        with make_zmq_socket(self.initialization_output_path,
                             zmq.constants.PUSH) as socket:
            socket.send_multipart(
                (WorkerInitRequestType.BEGIN_MODEL_EXECUTION.value, ))


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        initialization_input_path: str,
        initialization_output_path: str,
        ready_path: str,
    ):
        self.rank = rank
        self.worker = Worker(vllm_config, local_rank, rank,
                             distributed_init_method)

        # Initialize MessageQueue for receiving SchedulerOutput
        self.scheduler_output_receiver = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)

        # Send Readiness signal to EngineCore process.
        with make_zmq_socket(ready_path, zmq.constants.PUSH) as ready_socket:
            ready_socket.send_string(WorkerProc.READY_STR)

        # Worker 0 initializes a message queue for sending the model output
        if self.rank == 0:
            self.model_output_mq = MessageQueue(1, 1)
            output_mq_handle = self.model_output_mq.export_handle()
            with make_zmq_socket(initialization_output_path,
                                 zmq.constants.PUSH) as socket:
                encoder = msgspec.msgpack.Encoder()
                msg = encoder.encode(ShmHandleMsg(output_mq_handle))
                socket.send_multipart(
                    (WorkerInitOutputType.MODEL_OUTPUT_MSG_QUEUE.value, msg))
        else:
            self.model_output_mq = None

        self.worker.initialize()
        self.worker.load_model()

    @staticmethod
    def make_worker_process(
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            input_shm_handle,  # Receive SchedulerOutput
    ) -> WorkerProcHandle:
        # The current process might have CUDA context,
        # so we need to spawn a new process.
        # NOTE(rob): this is a problem for using EngineCoreProc w/
        # LLM, since we need a if __name__ == "__main__" guard.

        # TODO(tms): fix before landing
        context = multiprocessing.get_context("fork")

        # ZMQ paths to send back and forth to worker process
        # Used for initialization.
        initialization_input_path = get_open_zmq_ipc_path()
        initialization_output_path = get_open_zmq_ipc_path()
        ready_path = get_open_zmq_ipc_path()

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_path": ready_path,
            "initialization_input_path": initialization_output_path,
            "initialization_output_path": initialization_input_path,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=WorkerProc.run_worker,
                               kwargs=process_kwargs,
                               daemon=True)
        proc.start()

        # Wait for startup
        WorkerProc.wait_for_startup(proc, ready_path)

        # Read Shm MessageQueue from rank 0
        if rank == 0:
            model_output_mq_handle = WorkerProc.read_model_output_mq_handle(
                initialization_input_path)
        else:
            model_output_mq_handle = None

        return WorkerProcHandle(proc, initialization_input_path,
                                initialization_output_path,
                                model_output_mq_handle)

    @staticmethod
    def run_worker(*args, **kwargs):
        """Launch Worker busy loop in background process."""

        try:
            worker = WorkerProc(*args, **kwargs)
            worker.model_initialization_loop(
                kwargs["initialization_input_path"],
                kwargs["initialization_output_path"])

            worker.execute_model_busy_loop()

        except KeyboardInterrupt:
            logger.debug("Worker interrupted.")

        except BaseException as e:
            logger.exception(e)
            raise e

    @staticmethod
    def wait_for_startup(
        proc: BaseProcess,
        ready_path: str,
    ) -> None:
        """Wait until the Worker is ready."""
        with make_zmq_socket(ready_path, zmq.constants.PULL) as socket:

            # Wait for Worker to send Worker.READY_STR.
            while socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                logger.debug("Waiting for WorkerProc to startup.")

                if not proc.is_alive():
                    raise RuntimeError("WorkerProc failed to start.")

            message = socket.recv_string()
            assert message == WorkerProc.READY_STR

    @staticmethod
    def read_model_output_mq_handle(init_input_path: str, ) -> Handle:
        with make_zmq_socket(init_input_path,
                             zmq.constants.PULL) as recv_socket:
            type_frame, data_frame = recv_socket.recv_multipart(copy=False)
            request_type = type_frame.buffer
            request_data = data_frame.buffer

            if (request_type ==
                    WorkerInitOutputType.MODEL_OUTPUT_MSG_QUEUE.value):
                decoder = msgspec.msgpack.Decoder(ShmHandleMsg)
                handle = decoder.decode(request_data).handle
                return handle
            else:
                raise ValueError(f"Unknown RequestType: {request_type}")

    # Busy loop used for initializing Multiprocessing Workers
    def model_initialization_loop(self, init_input_path, init_output_path):
        # Msgpack serialization encoding.
        encoder = msgspec.msgpack.Encoder()
        # Reuse send buffer.
        buffer = bytearray()

        with make_zmq_socket(init_output_path,
                             zmq.constants.PUSH) as send_socket, \
             make_zmq_socket(init_input_path,
                             zmq.constants.PULL) as recv_socket:
            while True:
                # (RequestType, RequestData)
                thing = recv_socket.recv_multipart(copy=False)
                request_type = thing[0].buffer

                # Deserialize the request data.
                if (request_type ==
                        WorkerInitRequestType.DETERMINE_NUM_BLOCKS.value):
                    num_blocks = self.worker.determine_num_available_blocks()
                    output = NumBlocksMsg(num_blocks)
                    encoder.encode_into(output, buffer)
                    send_socket.send_multipart(
                        (WorkerInitOutputType.NUM_BLOCKS.value, buffer),
                        copy=False)
                elif request_type == WorkerInitRequestType.INIT_CACHE.value:
                    request_data = thing[1].buffer
                    decoder = msgspec.msgpack.Decoder(NumGPUBlocks)
                    num_gpu_blocks = decoder.decode(
                        request_data).num_gpu_blocks
                    self.worker.initialize_cache(num_gpu_blocks)
                    self.worker.compile_or_warm_up_model()
                elif (request_type ==
                      WorkerInitRequestType.BEGIN_MODEL_EXECUTION.value):
                    # Make sure message queues are ready.
                    self.scheduler_output_receiver.wait_until_ready()

                    if self.model_output_mq is not None:
                        self.model_output_mq.wait_until_ready()

                    # Exit initialization loop to begin model execution loop
                    return
                else:
                    raise ValueError(f"Unknown RequestType: {request_type}")

    # Main busy loop for Multiprocessing Workers
    def execute_model_busy_loop(self):
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1000,  # Wait 1000 steps so we profile middle iters
                    warmup=10,  # Warm up the scheduler
                    active=3,  # Run a small number of steps so it's legible
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./traces/",
                    worker_name=f"worker_{self.worker.rank}",
                ),
                with_stack=True,
        ) as p:

            while True:
                msg = self.scheduler_output_receiver.dequeue(ExecutorMsg)

                if msg.message_type == ExecutorMsgType.TERMINATE:
                    return
                elif msg.message_type == ExecutorMsgType.TOIL:
                    output = self.worker.execute_model(msg.payload)
                    if self.worker.rank == 0:
                        self.model_output_mq.enqueue(output)
                else:
                    raise ValueError(
                        f"Unknown RequestType: {msg.message_type}")

                p.step()


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")


def _get_cache_block_size(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = model_config.get_num_attention_layers(
        parallel_config)

    key_cache_block = cache_config.block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    if cache_config.cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
    dtype_size = get_dtype_size(dtype)
    return dtype_size * total
