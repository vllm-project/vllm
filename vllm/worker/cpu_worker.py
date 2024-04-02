"""A CPU worker class."""
from typing import Dict, List, Optional

import torch
import torch.distributed

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.model_runner import ModelRunner

logger = init_logger(__name__)


class CPUModelRunner(ModelRunner):

    def load_model(self) -> None:
        self.model = get_model(self.model_config,
                               self.device_config,
                               lora_config=self.lora_config,
                               parallel_config=self.parallel_config,
                               scheduler_config=self.scheduler_config)


class CPUCacheEngine:
    """Manages the KV cache for CPU backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on CPU."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape, dtype=self.dtype, device="cpu"))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.cpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        return dtype_size * total


class CPUWorker:
    """A worker class that executes (a partition of) the model on a CPU socket.

    Each worker is associated with a single CPU socket. The worker is 
    responsible for maintaining the KV cache and executing the model on the 
    CPU. In case of distributed inference, each worker is assigned a partition
    of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = CPUModelRunner(model_config,
                                           parallel_config,
                                           scheduler_config,
                                           device_config,
                                           lora_config=self.lora_config,
                                           kv_cache_dtype=kv_cache_dtype,
                                           is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cpu_cache = None

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def get_cpu_cache_block_num(
        self,
        block_size: int,
        cache_space: int,
        cache_dtype: str,
    ) -> int:
        """
        Args:
            block_size: The size of the cache block.
            cache_space: The size of the CPU KV cache space in bytes.
        """
        # For CPU device, the block number will be calculated based on the
        # cpu_kvcache_space.
        cache_block_size = CPUCacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)
        num_cpu_blocks = int(cache_space // cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        return num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CPUCacheEngine(self.cache_config,
                                           self.model_config,
                                           self.parallel_config,
                                           self.device_config)
        self.cpu_cache = self.cache_engine.cpu_cache
        self.model_runner.block_size = self.cache_engine.block_size

        assert self.cpu_cache is not None

        # Populate the cache to warmup the memory
        for layer_cache in self.cpu_cache:
            layer_cache.fill_(0)

    def cache_copy(
        self,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
    ) -> Optional[SamplerOutput]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            assert len(blocks_to_swap_in) == 0
            assert len(blocks_to_swap_out) == 0
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_copy(blocks_to_copy)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.cpu_cache)
        return output

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method

        if torch.distributed.is_initialized():
            torch_world_size = torch.distributed.get_world_size()
            if torch_world_size != parallel_config.world_size:
                raise RuntimeError(
                    "torch.distributed is already initialized but the torch "
                    "world size does not match parallel_config.world_size "
                    f"({torch_world_size} vs. {parallel_config.world_size}).")
        elif not distributed_init_method:
            raise ValueError(
                "distributed_init_method must be set if torch.distributed "
                "is not already initialized")
        else:
            backend = "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                world_size=parallel_config.world_size,
                rank=rank,
                init_method=distributed_init_method,
            )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)
