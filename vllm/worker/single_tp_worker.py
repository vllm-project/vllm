from typing import List, Optional
import logging

import torch

from vllm.sequence import (SamplerOutput, ExecuteModelData)
from vllm.model_executor.parallel_utils.parallel_state import patch_tensor_parallel_group
from vllm.config import CacheConfig, ParallelConfig
from vllm.worker.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class SingleTpWorker(BaseWorker):
    """Class which allows a speculative draft model to run with tensor parallel
    degree of 1, while target model runs with larger tensor parallel degree.
    This reduces the overhead of small draft models.

    This is implemented by changing vLLM's tensor parallel group to a group of
    size 1 during forward passes.
    """

    @classmethod
    def maybe_wrap_worker(cls, worker, draft_parallel_config: ParallelConfig,
                          target_parallel_config: ParallelConfig):
        """Wrap the worker in a SingleTpWorker if necessary.
        """
        draft_tp = draft_parallel_config.tensor_parallel_size
        if draft_tp == target_parallel_config.tensor_parallel_size:
            return worker

        if draft_tp != 1:
            raise ValueError("{cls} only supports tp=1, found "
                             f"{draft_tp=}")

        logger.info(f"Wrapping {type(worker)} in {cls}")
        return cls(worker)

    def __init__(
        self,
        worker: BaseWorker,
    ):
        self._worker = worker
        self._single_tp_group = None

    def init_model(self):
        """Initialize the model on all ranks.

        This also creates a single-rank process group containing only the
        self process.
        """
        world_rank = torch.distributed.get_rank()
        self._single_tp_group = torch.distributed.new_group([world_rank])

        with patch_tensor_parallel_group(self._single_tp_group):
            self._worker.init_model(should_init_distributed_env=False)

    def profile_num_available_blocks(self, block_size: int,
                                     gpu_memory_utilization: float,
                                     cpu_swap_space: int):
        """Profile the model on all ranks.
        """
        with patch_tensor_parallel_group(self._single_tp_group):
            return self._worker.profile_num_available_blocks(
                block_size, gpu_memory_utilization, cpu_swap_space)

    def init_cache_engine(self, cache_config: CacheConfig):
        """Initialize the cache engine on all ranks.
        """
        with patch_tensor_parallel_group(self._single_tp_group):
            self._worker.init_cache_engine(cache_config)

    @property
    def model_config(self):
        return self._worker.model_config

    @property
    def parallel_config(self):
        return self._worker.parallel_config

    @property
    def model(self):
        return self._worker.model

    @property
    def rank(self):
        return self._worker.rank

    def get_metadata_cache_len(self) -> int:
        """Metadata cache not currently supported.
        """
        return 0

    def get_runtime_context(self) -> Optional[dict]:
        return self._worker.get_runtime_context()

    @property
    def _vocab_size(self) -> int:
        return self.model.config.vocab_size

    @torch.inference_mode()
    def execute_model(
            self,
            execute_model_data: ExecuteModelData,
            *,
            return_python_output: bool = True) -> List[SamplerOutput]:
        """Execute the model separately on each rank.
        """
        with patch_tensor_parallel_group(self._single_tp_group):
            return self._worker.execute_model(
                execute_model_data, return_python_output=return_python_output)
