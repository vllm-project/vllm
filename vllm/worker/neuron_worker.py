# SPDX-License-Identifier: Apache-2.0
"""A Neuron worker class."""
import enum
import os
from functools import cache
from typing import List, Optional, Tuple

import torch
import torch.distributed

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerBase,
                                     WorkerInput)
from vllm.platforms import current_platform

logger = init_logger(__name__)

DEFAULT_WORLD_SIZE = "1"
DEFAULT_NEURON_RANK_ID = "0"
DEFAULT_ENABLE_NEURON_MULTI_NODE = "False"


class NeuronFramework(enum.Enum):
    TRANSFORMERS_NEURONX = "transformers-neuronx"
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"


@cache
def get_neuron_framework_to_use():
    """Return the specified framework if corresponding installations are
    available.

    If no framework is specified, use neuronx-distributed-inference by default.
    If that's unavailable, check and switch to transformers-neuronx.
    """
    if not current_platform.is_neuron():
        raise AssertionError(
            f"Neuron Framework unavailable for platform: {current_platform}")

    tnx_installed = current_platform.is_transformers_neuronx()
    nxd_installed = current_platform.is_neuronx_distributed_inference()

    specified_framework = os.environ.get("VLLM_NEURON_FRAMEWORK")
    tnx_framework = NeuronFramework.TRANSFORMERS_NEURONX.value
    nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE.value
    if (specified_framework == tnx_framework and
            tnx_installed):
        return NeuronFramework.TRANSFORMERS_NEURONX

    if ((specified_framework == nxd_framework and
         nxd_installed) or
            (specified_framework is None and nxd_installed)):
        return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

    if specified_framework is None and tnx_installed:
        return NeuronFramework.TRANSFORMERS_NEURONX

    return None


@cache
def use_neuronx_distributed():
    """
    Return True if the framework determined in get_neuron_framework_to_use() is
    NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, False otherwise. This is used
    to select the Neuron model framework and framework-specific configuration to
    apply during model compilation.
    """
    nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
    return get_neuron_framework_to_use() == nxd_framework


@cache
def use_transformers_neuronx():
    """
    Return True if the framework determined in get_neuron_framework_to_use() is
    NeuronFramework.TRANSFORMERS_NEURONX, False otherwise. This is used to
    select the Neuron model framework and framework-specific configuration to
    apply during model compilation.
    """
    return get_neuron_framework_to_use() == NeuronFramework.TRANSFORMERS_NEURONX


class NeuronWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        self.enable_neuron_multi_node = (
                os.getenv("ENABLE_NEURON_MULTI_NODE",
                          DEFAULT_ENABLE_NEURON_MULTI_NODE).lower() == "true")

        self.world_size = int(os.getenv("WORLD_SIZE", DEFAULT_WORLD_SIZE))

        if self.enable_neuron_multi_node:
            self.rank = int(os.getenv("NEURON_RANK_ID", DEFAULT_NEURON_RANK_ID))
            self.distributed_init_method = "env://"
            self.is_driver_worker = self.rank == 0

            logger.info("Neuron multi-node parameters",
                        extra={
                            "Rank": self.rank,
                            "distributed_init_method":
                                self.distributed_init_method,
                            "is_driver_worker": self.is_driver_worker
                        })

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        neuron_framework = get_neuron_framework_to_use()
        if neuron_framework == NeuronFramework.TRANSFORMERS_NEURONX:
            self.model_runner = self.get_tnx_model_runner(vllm_config)
        elif neuron_framework == NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE:
            self.model_runner = self.get_neuronx_distributed_model_runner(
                vllm_config)
        else:
            raise NotImplementedError(
                "Specified framework" +
                f" {os.environ.get('VLLM_NEURON_FRAMEWORK')}" +
                " is either not installed or not supported." +
                " Supported frameworks: " +
                "[transformers-neuronx, neuronx-distributed-inference]")

    def get_tnx_model_runner(self, vllm_config):
        from vllm.worker.neuron_model_runner import NeuronModelRunner
        from vllm.worker.multi_step_neuron_model_runner import \
            MultiStepNeuronModelRunner
        if self.speculative_config is not None:
            return MultiStepNeuronModelRunner(
                vllm_config=vllm_config)
        else:
            return NeuronModelRunner(
                vllm_config=vllm_config)

    def get_neuronx_distributed_model_runner(self, vllm_config):
        from vllm.worker.neuronx_distributed_model_runner import \
            NeuronxDistributedModelRunner
        from vllm.worker.multi_step_neuronx_distributed_model_runner import \
            MultiStepNeuronxDistributedModelRunner
        if self.speculative_config is not None:
            return MultiStepNeuronxDistributedModelRunner(
                vllm_config=vllm_config)
        else:
            return NeuronxDistributedModelRunner(
                vllm_config=vllm_config)

    def init_device(self) -> None:
        self.init_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs

        # Swap not yet supported with Neuron backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """

        # Different values are not tested.
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == self.scheduler_config.max_num_seqs

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.enable_neuron_multi_node and self.world_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(num_seq_groups=len(
            execute_model_req.seq_group_metadata_list), )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    def init_distributed_environment(self):
        """Neuron uses transformers-neuronx for tensor parallelism.

        vLLM still needs the environment inited when TP/PP > 1
        """
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        # The equation must hold: world_size === TP * PP
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=self.world_size,
            # pipeline parallelism is not yet supported
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
