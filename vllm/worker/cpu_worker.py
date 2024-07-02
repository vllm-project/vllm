"""A CPU worker class."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed

from vllm.attention import get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.cpu_model_runner import CPUModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

logger = init_logger(__name__)


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
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

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


class CPUWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
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
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.model_runner: CPUModelRunner = CPUModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config=self.load_config,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: CPUCacheEngine
        self.cpu_cache: List[torch.Tensor]

    def init_device(self) -> None:
        self.init_distributed_environment()
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of blocks available for the KV cache.

        This determines how many KV blocks can fit into the configured CPU
        KV cache space.

        Note that since vLLM assumes a block resides on GPU if it can be
        modified, we return num_gpu_blocks=num_cpu_blocks and num_cpu_blocks=0.
        This allows us to reuse the scheduler of vLLM without generalizing it
        to different devices.
        """
        # For CPU device, the block number will be calculated based on the
        # cpu_kvcache_space.
        cache_block_size = self.get_cache_block_size_bytes()
        num_cpu_blocks = int(self.cache_config.cpu_kvcache_space_bytes //
                             cache_block_size)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_gpu_blocks = num_cpu_blocks
        num_cpu_blocks = 0
        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache. Currently, swappable CPU memory is not
        supported.

        Since this worker does not support GPUs, we use the num_gpu_blocks to
        determine how many non-swappable CPU blocks to allocate.
        """
        assert (num_cpu_blocks == 0
                ), f"{type(self)} does not support swappable cache"

        # Note: To reuse the cache management procedure,
        # use cpu cache as 'gpu cache'.
        num_cpu_blocks = num_gpu_blocks

        self._validate_num_cpu_blocks(num_cpu_blocks)
        self.cache_config.num_gpu_blocks = num_cpu_blocks
        self.cache_config.num_cpu_blocks = 0

        # Initialize the cache.
        self._init_cache_engine()

    def _validate_num_cpu_blocks(self, num_cpu_blocks: int) -> None:
        """Raise errors if the num_cpu_blocks is invalid.
        """
        if num_cpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `VLLM_CPU_KVCACHE_SPACE` when "
                             "initializing the engine.")

        max_seq_len = self.cache_config.block_size * num_cpu_blocks
        if self.model_config.max_model_len > max_seq_len:
            raise ValueError(
                f"The model's max seq len ({self.model_config.max_model_len}) "
                "is larger than the maximum number of tokens that can be "
                f"stored in KV cache ({max_seq_len}). Try increasing "
                "`VLLM_CPU_KVCACHE_SPACE` or decreasing `max_model_len` when "
                "initializing the engine.")

    def _init_cache_engine(self) -> None:
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

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[torch.Tensor]]:
        return self.cpu_cache

    def execute_worker(
        self,
        worker_input: WorkerInput,
    ) -> None:
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine.copy(worker_input.blocks_to_copy)

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        assert execute_model_req is not None
        num_seq_groups: int = len(execute_model_req.seq_group_metadata_list)
        blocks_to_copy = execute_model_req.blocks_to_copy
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)
        assert len(execute_model_req.blocks_to_swap_in) == 0
        assert len(execute_model_req.blocks_to_swap_out) == 0
        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_copy=blocks_to_copy,
        )

    def init_distributed_environment(self) -> None:
        """Initialize the distributed environment."""

        parallel_config = self.parallel_config
        rank = self.rank
        distributed_init_method = self.distributed_init_method
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            distributed_init_method=distributed_init_method,
            backend="gloo",
        )

        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cpu())

        ensure_model_parallel_initialized(
            parallel_config.tensor_parallel_size,
            parallel_config.pipeline_parallel_size)

    def get_cache_block_size_bytes(self) -> int:
        """Return the size in bytes of a single KV cache block.
        """
        return CPUCacheEngine.get_cache_block_size(
            self.cache_config.block_size, self.cache_config.cache_dtype,
            self.model_config, self.parallel_config)
