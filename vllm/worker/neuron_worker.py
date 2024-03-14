"""A Neuron worker class."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.neuron_model_runner import NeuronModelRunner


class Worker:
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config

        self.model_runner = NeuronModelRunner(model_config, parallel_config,
                                              scheduler_config, device_config)

    def init_model(self) -> None:
        # Initialize the model.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        if cache_events is not None:
            raise NotImplementedError(
                "cache operations are not implemented for neuron backend.")

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        num_seq_groups = len(seq_group_metadata_list)
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list)
        return output
