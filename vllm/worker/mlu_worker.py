"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed

import vllm.envs as envs
from vllm.config import (VllmConfig, ParallelConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import (ExecuteModelRequest, SequenceGroupMetadata)
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.mlu_model_runner import MLUModelRunnerBase, MLUModelRunner
from vllm.worker.worker_base import (LoraNotSupportedWorkerBase, WorkerInput, WorkerBase)
from vllm.worker.worker import Worker
from vllm.logger import init_logger

logger = init_logger(__name__)


class MLUWorker(Worker):
    """A worker class that executes (a partition of) the model on a MLU.

    Each worker is associated with a single MLU. The worker is responsible for
    maintaining the KV cache and executing the model on the MLU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[MLUModelRunnerBase]] = None,
    ) -> None:
        WorkerBase.__init__(self, vllm_config)
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if is_driver_worker:
            assert rank % self.parallel_config.tensor_parallel_size == 0, \
                   "Driver worker should be rank 0 of tensor parallel group."
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Return hidden states from target model if the draft model is an
        # mlp_speculator
        speculative_config = self.speculative_config
        model_config = self.model_config
        speculative_args = {} if speculative_config is None \
            or (speculative_config.draft_model_config.model ==
                model_config.model) \
            or (speculative_config.draft_model_config.hf_config.model_type
                not in ["medusa", "mlp_speculator", "eagle"]) \
                    else {"return_hidden_states": True}

        ModelRunnerClass: Type[MLUModelRunnerBase] = MLUModelRunner
        if model_runner_cls is not None:
            ModelRunnerClass = model_runner_cls
        elif model_config.task == "embedding":
            ModelRunnerClass = EmbeddingModelRunner
        elif self._is_encoder_decoder_model():
            ModelRunnerClass = EncoderDecoderModelRunner
        self.model_runner: MLUModelRunnerBase = ModelRunnerClass(
            vllm_config=self.vllm_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker,
            **speculative_args,
        )
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.gpu_cache: Optional[List[List[torch.Tensor]]] = None
        self._seq_group_metadata_cache: Dict[str, SequenceGroupMetadata] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def init_device(self) -> None:
        if self.device_config.device.type == "mlu":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_CNCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("CNCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"mlu:{self.local_rank}")
            torch.mlu.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.mlu.empty_cache()
            self.init_gpu_memory = torch.mlu.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of MLU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of MLU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.mlu.empty_cache()
        torch.mlu.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.mlu.mem_get_info()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.mlu.synchronize()

        self._assert_memory_footprint_increased_during_profiling()

        # Get the peak memory allocation recorded by torch
        peak_memory = torch.mlu.memory_stats()["allocated_bytes.all.peak"]

        # Check for any memory left around that may have been allocated on the
        # mlu outside of `torch`. NCCL operations, for example, can use a few
        # GB during a forward pass
        torch.mlu.empty_cache()
        torch_allocated_bytes = torch.mlu.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch.mlu.mem_get_info(
        )[1] - torch.mlu.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations

        available_kv_cache_memory = (
            total_gpu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        cache_block_size = self.get_cache_block_size_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
            num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        logger.info(
            "Memory profiling results: total_mlu_memory=%.2fGiB"
            " initial_memory_usage=%.2fGiB peak_torch_memory=%.2fGiB"
            " memory_usage_post_profile=%.2fGib"
            " non_torch_memory=%.2fGiB kv_cache_size=%.2fGiB"
            " mlu_memory_utilization=%.2f", total_gpu_memory / (1024**3),
            (total_gpu_memory - free_memory_pre_profile) / (1024**3),
            (peak_memory - non_torch_allocations) / (1024**3),
            total_allocated_bytes / (1024**3),
            non_torch_allocations / (1024**3),
            available_kv_cache_memory / (1024**3),
            self.cache_config.gpu_memory_utilization)

        # Final cleanup
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()

        return num_gpu_blocks, num_cpu_blocks

    def _assert_memory_footprint_increased_during_profiling(self):
        # NOTE(woosuk): Here we assume that the other processes using the same
        # MLU did not change their memory usage during the profiling.
        free_gpu_memory, _ = torch.mlu.mem_get_info()
        assert self.init_gpu_memory - free_gpu_memory > 0, (
            "Error in memory profiling. "
            f"Initial free memory {self.init_gpu_memory}, current free memory"
            f" {free_gpu_memory}. This happens when the MLU memory was "
            "not properly cleaned up before initializing the vLLM instance.")


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank,
                                 backend='cncl')

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the MLU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(50):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on MLUs with compute capability "
                f"of at least 8.0. Your {gpu_name} MLU {compute_str}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")