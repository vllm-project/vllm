###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from typing import Any, Dict, List, Optional, Set, Tuple
from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async, HabanaMemoryProfiler)
import os
import contextlib
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class HabanaExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self._init_worker()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=rank == 0,
        )

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        wrapper = WorkerWrapperBase(
            worker_module_name="vllm.worker.habana_worker",
            worker_class_name="HabanaWorker",
        )
        wrapper.init_worker(**self._get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

    def _init_worker(self):
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.driver_worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# HPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        with HabanaMemoryProfiler() as cache_init_m:
            self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        logger.info(
            f"init_cache_engine took {cache_init_m.get_summary_string()}")

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION     - will log graph compilations per engine step, only when there was any - highly recommended to use alongside PT_HPU_METRICS_GC_DETAILS!
        # VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL - will log graph compilations per engine step, always, even if there were none
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS         - will log cpu fallbacks per engine step, only when there was any
        # VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL     - will log cpu fallbacks per engine step, always, even if there were none
        log_graph_compilation_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL', '0') != '0'
        log_graph_compilation = os.environ.get(
            'VLLM_HPU_LOG_STEP_GRAPH_COMPILATION',
            '0') != '0' or log_graph_compilation_all
        log_cpu_fallbacks_all = os.environ.get(
            'VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL', '0') != '0'
        log_cpu_fallbacks = os.environ.get('VLLM_HPU_LOG_STEP_CPU_FALLBACKS',
                                           '0') != '0' or log_cpu_fallbacks_all
        if log_graph_compilation or log_cpu_fallbacks:
            from habana_frameworks.torch.hpu.metrics import metric_localcontext
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list
            is_prompt = any([
                seq_group_metadata.is_prompt
                for seq_group_metadata in seq_group_metadata_list
            ])
            max_context_len = max([
                max([
                    len(v.prompt_token_ids) + len(v.output_token_ids)
                    for v in seq_group_metadata.seq_data.values()
                ]) for seq_group_metadata in seq_group_metadata_list
            ])  # whoa, that's some spicy stuff right here
            max_num_blocks = (
                (max_context_len - 1) // self.cache_config.block_size) + 1
            input_stats = f'is_prompt: {is_prompt}, num_seqs: {len(seq_group_metadata_list)} max_context_len: {max_context_len}, max_num_blocks {max_num_blocks}'
            gc_ctx = metric_localcontext(
                "graph_compilation"
            ) if log_graph_compilation else contextlib.nullcontext()
            cpu_fallback_ctx = metric_localcontext(
                "cpu_fallback"
            ) if log_cpu_fallbacks else contextlib.nullcontext()
            with gc_ctx as gc_local_metric, cpu_fallback_ctx as cpu_fallback_local_metric:
                output = self.driver_worker.execute_model(execute_model_req)
            if (log_graph_compilation and gc_local_metric.stats()[0][1] > 0
                ) or log_graph_compilation_all:
                logger.warning(
                    f"VLLM_HPU_STEP_GRAPH_COMPILATION: {gc_local_metric.stats()}, {input_stats}"
                )
            if (log_cpu_fallbacks and cpu_fallback_local_metric.stats()[0][1] >
                    0) or log_cpu_fallbacks_all:
                logger.warning(
                    f"VLLM_HPU_STEP_CPU_FALLBACK: {cpu_fallback_local_metric.stats()}, {input_stats}"
                )

            return output

        output = self.driver_worker.execute_model(execute_model_req)
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return


class HabanaExecutorAsync(HabanaExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req, )
        return output
