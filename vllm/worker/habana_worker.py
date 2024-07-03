###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import gc
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import habana_frameworks.torch as htorch
import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         SpeculativeConfig, VisionLanguageConfig)
from vllm.distributed import (broadcast_tensor_dict,
                              ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.habana_model_runner import HabanaModelRunner
from vllm.worker.worker_base import WorkerBase


class HabanaWorker(WorkerBase):
    """A worker class that executes (a partition of) the model on a HPU.

    Each worker is associated with a single HPU. The worker is responsible for
    maintaining the KV cache and executing the model on the HPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")
            raise AssertionError("To be tested: vision language model on HPU")

        self.model_runner = HabanaModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            load_config=load_config,
            cache_config=cache_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CacheEngine
        self.hpu_cache: List[torch.Tensor]

    def init_device(self) -> None:
        if self.device_config.device.type == "hpu":
            self.device = torch.device("hpu")
            torch.hpu.set_device(self.device)
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
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

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()
        torch.hpu.synchronize()

        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        cache_block_size = self.get_cache_block_size_bytes()
        graph_headroom = 1 - (float(
            os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.4'))
                              if not self.model_config.enforce_eager else 0)
        num_hpu_blocks = int(free_hpu_memory * graph_headroom *
                             self.cache_config.gpu_memory_utilization //
                             cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_hpu_blocks = max(num_hpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # NOTE(kzawora): Restore this once LoRA support is added
        # if self.model_runner.lora_manager:
        #     self.model_runner.remove_all_loras()

        gc.collect()
        return num_hpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()
        self._warm_up_model()

    def _init_cache_engine(self) -> None:
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config,
                                        self.device_config)
        self.hpu_cache = self.cache_engine.gpu_cache
        # we want to materialize cache tensors before we proceed with
        # graph capture/execution
        htorch.hpu.synchronize()

    def _warm_up_model(self) -> None:
        self.model_runner.warmup_model(self.hpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: torch.Tensor,
    ) -> None:
        # Issue cache operations.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy.numel() > 0:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        if execute_model_req is None:
            seq_group_metadata_list = None
        else:
            seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            assert execute_model_req is not None
            num_seq_groups = len(seq_group_metadata_list)
            blocks_to_swap_in = execute_model_req.blocks_to_swap_in
            blocks_to_swap_out = execute_model_req.blocks_to_swap_out
            blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                          device=self.device,
                                          dtype=torch.int64).view(-1, 2)
            data: Dict[str, Any] = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.hpu_cache)
        return [output]

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop in parallel worker.

        You can stop the loop by executing a driver worker with an empty output.
        See `stop_remote_worker_execution_loop` for more details.
        """
        while self._execute_model_non_driver():
            pass

    def _execute_model_non_driver(self) -> bool:
        """Execute model in parallel worker.

        Returns True iff there are remaining sequences to process.
        """
        assert not self.is_driver_worker
        data = broadcast_tensor_dict(src=0)
        if not data:
            return False

        num_seq_groups = data.get("num_seq_groups", 0)
        blocks_to_swap_in = data.get("blocks_to_swap_in")
        blocks_to_swap_out = data.get("blocks_to_swap_out")
        blocks_to_copy = data.get("blocks_to_copy")
        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return False

        self.model_runner.execute_model(None, self.hpu_cache)
        return True

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not implemented for HPU backend.")

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size,
                                 rank,
                                 distributed_init_method,
                                 local_rank,
                                 backend='hccl')

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="hccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup & checking conformance.
    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
