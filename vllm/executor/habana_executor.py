###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from typing import Dict, List, Optional

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.executor.utils import check_block_size_valid
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async, HabanaMemoryProfiler, format_bytes)
import os
import contextlib
logger = init_logger(__name__)


class HabanaExecutor(ExecutorBase):

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        # Instantiate the worker and load the model to GPU.
        self._init_worker()

        # Profile the memory usage and initialize the cache.
        self._init_cache()

    def _init_worker(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.habana_worker import HabanaWorker

        assert self.parallel_config.world_size == 1, (
            "HabanaExecutor only supports single GPU.")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        self.driver_worker = HabanaWorker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=True,
        )
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache.

        The engine first profiles the existing memory usage.
        Then, it allocates the remaining memory for KV blocks.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_gpu_blocks, num_cpu_blocks = (
            self.driver_worker.profile_num_available_blocks(
                block_size=self.cache_config.block_size,
                hpu_memory_utilization=self.cache_config.
                gpu_memory_utilization,
                cpu_swap_space=self.cache_config.swap_space_bytes,
                cache_dtype=self.cache_config.cache_dtype,
            ))

        logger.info(f"# HPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        check_block_size_valid(num_gpu_blocks, self.cache_config.block_size,
                               self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        with HabanaMemoryProfiler() as cache_init_m:
            self.driver_worker.init_cache_engine(cache_config=self.cache_config)
        logger.info(f"init_cache_engine took "
                    f"{format_bytes(cache_init_m.consumed_memory)} ({cache_init_m.consumed_memory/HabanaMemoryProfiler.total_memory():.2%} of total memory, gpu_memory_utilization: {self.cache_config.gpu_memory_utilization}, {format_bytes(HabanaMemoryProfiler.current_memory_usage())}/{format_bytes(HabanaMemoryProfiler.total_memory())} used)")

        # Warm up the model. This includes capturing the model into CUDA graph
        # if enforce_eager is False.
        with HabanaMemoryProfiler() as warmup_m:
            self.driver_worker.warm_up_model()
        logger.info(f"Model warmup took "
                    f"{format_bytes(warmup_m.consumed_memory)} ({format_bytes(HabanaMemoryProfiler.current_memory_usage())}/{format_bytes(HabanaMemoryProfiler.total_memory())} used)")

    def execute_model(self,
                      seq_group_metadata_list: List[SequenceGroupMetadata],
                      blocks_to_swap_in: Dict[int, int],
                      blocks_to_swap_out: Dict[int, int],
                      blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:

        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION     - will log graph compilations per engine step, only when there was any - highly recommended to use alongside PT_HPU_METRICS_GC_DETAILS!
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL - will log graph compilations per engine step, always, even if there were none
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS         - will log cpu fallbacks per engine step, only when there was any
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL     - will log cpu fallbacks per engine step, always, even if there were none
        log_graph_compilation_all = os.environ.get('VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL', '0') != '0'
        log_graph_compilation = os.environ.get('VLLM_HPU_LOG_STEP_GRAPH_COMPILATION', '0') != '0' or log_graph_compilation_all
        log_cpu_fallbacks_all = os.environ.get('VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL', '0') != '0'
        log_cpu_fallbacks = os.environ.get('VLLM_HPU_LOG_STEP_CPU_FALLBACKS', '0') != '0' or log_cpu_fallbacks_all
        if log_graph_compilation or log_cpu_fallbacks:
            from habana_frameworks.torch.hpu.metrics import metric_localcontext
            is_prompt = any([seq_group_metadata.is_prompt for seq_group_metadata in seq_group_metadata_list])
            max_context_len = max([max([len(v.prompt_token_ids) + len(v.output_token_ids) for v in seq_group_metadata.seq_data.values()]) for seq_group_metadata in seq_group_metadata_list]) # whoa, that's some spicy stuff right here
            max_num_blocks = ((max_context_len - 1) // self.cache_config.block_size) + 1
            input_stats = f'is_prompt: {is_prompt}, num_seqs: {len(seq_group_metadata_list)} max_context_len: {max_context_len}, max_num_blocks {max_num_blocks}'
            gc_ctx = metric_localcontext("graph_compilation") if log_graph_compilation else contextlib.nullcontext()
            cpu_fallback_ctx = metric_localcontext("cpu_fallback") if log_cpu_fallbacks else contextlib.nullcontext()
            with gc_ctx as gc_local_metric, cpu_fallback_ctx as cpu_fallback_local_metric:
                output = self.driver_worker.execute_model(
                    seq_group_metadata_list=seq_group_metadata_list,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                )
            if (log_graph_compilation and gc_local_metric.stats()[0][1] > 0) or log_graph_compilation_all:
                logger.warning(f"VLLM_HPU_STEP_GRAPH_COMPILATION: {gc_local_metric.stats()}, {input_stats}")
            if (log_cpu_fallbacks and cpu_fallback_local_metric.stats()[0][1] > 0) or log_cpu_fallbacks_all:
                logger.warning(f"VLLM_HPU_STEP_CPU_FALLBACK: {cpu_fallback_local_metric.stats()}, {input_stats}")
            
            return output

        output = self.driver_worker.execute_model(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def list_loras(self) -> List[int]:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class HabanaExecutorAsync(HabanaExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        output = await make_async(self.driver_worker.execute_model)(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy)
        return output

    async def check_health_async(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
